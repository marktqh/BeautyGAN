"""
Training script for face beautification GAN.
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import config
from models import Generator, Discriminator
from utils.dataset import create_dataloaders
from utils.losses import LSGANLoss, PerceptualLoss, total_variation_loss
from utils.visualization import create_comparison_grid

# Setup and cleanup for distributed training
def setup(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed training."""
    dist.destroy_process_group()

# Checkpoint utilities
def save_checkpoint(model_dict, optimizer_dict, step, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'generator': model_dict['generator'].state_dict(),
        'discriminator': model_dict['discriminator'].state_dict(),
        'g_optimizer': optimizer_dict['generator'].state_dict(),
        'd_optimizer': optimizer_dict['discriminator'].state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(model_dict, optimizer_dict, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model_dict['generator'].load_state_dict(checkpoint['generator'])
    model_dict['discriminator'].load_state_dict(checkpoint['discriminator'])
    optimizer_dict['generator'].load_state_dict(checkpoint['g_optimizer'])
    optimizer_dict['discriminator'].load_state_dict(checkpoint['d_optimizer'])
    return checkpoint['step']

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not os.path.isdir(checkpoint_dir):
        return None

    latest_step = -1
    latest_checkpoint = None

    for fname in os.listdir(checkpoint_dir):
        import re
        match = re.match(r"beauty_gan_step_(\d+)\.pth", fname)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = os.path.join(checkpoint_dir, fname)

    return latest_checkpoint

# Training step
def train_one_step(model_dict, optimizer_dict, criterion_dict, data, device, tv_weight):
    """
    Train for one step.
    
    Args:
        model_dict (dict): Dictionary of models
        optimizer_dict (dict): Dictionary of optimizers
        criterion_dict (dict): Dictionary of loss functions
        data (tuple): Batch of data
        device (torch.device): Device to run on
        tv_weight (float): Weight for total variation loss
        
    Returns:
        dict: Dictionary of loss values
    """
    generator = model_dict['generator']
    discriminator = model_dict['discriminator']
    
    g_optimizer = optimizer_dict['generator']
    d_optimizer = optimizer_dict['discriminator']
    
    gan_loss = criterion_dict['gan_loss']
    rec_loss = criterion_dict['rec_loss']
    perceptual_loss = criterion_dict['perceptual_loss']
    
    raw_img, target_img, _, _ = data
    raw_img = raw_img.to(device)
    target_img = target_img.to(device)
    
    # Train discriminator
    d_optimizer.zero_grad()
    with torch.no_grad():
        fake_img = generator(raw_img)
    real_pred = discriminator(raw_img, target_img)
    d_real_loss = gan_loss(real_pred, True)
    fake_pred = discriminator(raw_img, fake_img.detach())
    d_fake_loss = gan_loss(fake_pred, False)
    d_loss = (d_real_loss + d_fake_loss) * 0.5
    d_loss.backward()
    d_optimizer.step()
    
    # Train generator
    g_optimizer.zero_grad()
    fake_img = generator(raw_img)
    fake_pred = discriminator(raw_img, fake_img)
    g_gan_loss = gan_loss(fake_pred, True)
    g_rec_loss = rec_loss(fake_img, target_img)
    g_percep_loss = perceptual_loss(fake_img, target_img)
    tv = total_variation_loss(fake_img)
    
    # Total generator loss with TV loss added
    g_loss = (g_gan_loss * config.ADVERSARIAL_WEIGHT + 
              g_rec_loss * config.RECONSTRUCTION_WEIGHT + 
              g_percep_loss * config.PERCEPTUAL_WEIGHT + 
              tv * tv_weight)
    g_loss.backward()
    g_optimizer.step()
    
    return {
        'g_total': g_loss.item(),
        'g_gan': g_gan_loss.item(),
        'g_rec': g_rec_loss.item(),
        'g_percep': g_percep_loss.item(),
        'tv': tv.item(),
        'd_total': d_loss.item()
    }

# Main training function
def train_model(rank, world_size, args):
    """
    Train the face beautification model.
    
    Args:
        rank (int): Process rank
        world_size (int): Number of processes
        args (argparse.Namespace): Command line arguments
    """
    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Create directories
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(config.LOG_DIR) if rank == 0 else None
    
    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        config.RAW_DIR,
        config.AFT_DIR,
        transform,
        config.BATCH_SIZE,
        config.BATCH_SIZE_TEST,
        num_workers=16 if not args.debug else 0
    )
    
    # Create generator and discriminator
    generator = Generator(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        down_blocks=4,
        transformer_blocks=config.TRANSFORMER_BLOCKS,
        transformer_heads=config.TRANSFORMER_HEADS,
        ffn_ratio=config.FFN_RATIO,
        dropout=config.DROPOUT,
        init_alpha=config.INIT_ALPHA
    ).to(device)
    
    discriminator = Discriminator(
        in_channels=6,
        base_channels=64,
        n_layers=3
    ).to(device)
    
    # Wrap models with DDP
    generator = DDP(generator, device_ids=[rank], find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[rank], find_unused_parameters=False)
    
    # Create optimizers
    g_optimizer = optim.Adam(
        generator.parameters(), 
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(), 
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create loss functions
    gan_loss = LSGANLoss().to(device)
    rec_loss = nn.L1Loss().to(device)
    perceptual_loss = PerceptualLoss(device=device).to(device)
    
    # Create dictionaries for models, optimizers, and loss functions
    model_dict = {'generator': generator, 'discriminator': discriminator}
    optimizer_dict = {'generator': g_optimizer, 'discriminator': d_optimizer}
    criterion_dict = {'gan_loss': gan_loss, 'rec_loss': rec_loss, 'perceptual_loss': perceptual_loss}
    
    # Create learning rate schedulers
    t_max = config.NUM_EPOCHS * (len(train_loader.dataset) / config.BATCH_SIZE)
    lr_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=t_max, eta_min=1e-5)
    lr_scheduler_d = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=t_max, eta_min=1e-5)
    print(f'Max steps: {t_max}')
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        checkpoint_path = find_latest_checkpoint(config.CHECKPOINT_DIR)
        if checkpoint_path and rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
            start_step = load_checkpoint(model_dict, optimizer_dict, checkpoint_path, device)
            print(f"Resuming from step {start_step}")
    
    # Broadcast start_step to all processes
    start_step_tensor = torch.tensor([start_step], dtype=torch.long, device=device)
    dist.broadcast(start_step_tensor, src=0)
    start_step = start_step_tensor.item()
    
    # Training loop
    global_step = start_step
    generator.train()
    discriminator.train()
    
    running_losses = {'g_total': 0, 'g_gan': 0, 'g_rec': 0, 'g_percep': 0, 'tv': 0, 'd_total': 0}
    start_time = time.time()
    steps_since_log = 0
    
    print('Starting training...')
    
    for epoch in range(args.start_epoch, config.NUM_EPOCHS):
        for batch_idx, data in enumerate(train_loader):
            # Train one step
            losses = train_one_step(
                model_dict, 
                optimizer_dict, 
                criterion_dict, 
                data, 
                device, 
                config.TV_WEIGHT
            )
            
            # Step learning rate schedulers
            lr_scheduler_g.step()
            lr_scheduler_d.step()
            
            # Update running losses
            for k, v in losses.items():
                running_losses[k] += v
            steps_since_log += 1
            global_step += 1
            
            # Log losses
            if rank == 0 and (global_step == 1 or global_step % config.LOG_INTERVAL == 0):
                time_elapsed = time.time() - start_time
                steps_per_sec = steps_since_log / time_elapsed
                
                # Calculate average losses
                for k in running_losses:
                    running_losses[k] /= steps_since_log
                
                # Print losses
                print(f"[Step {global_step}] G_total: {running_losses['g_total']:.4f}, "
                      f"G_gan: {running_losses['g_gan']:.4f}, "
                      f"G_rec: {running_losses['g_rec']:.4f}, "
                      f"G_percep: {running_losses['g_percep']:.4f}, "
                      f"TV: {running_losses['tv']:.4f}, "
                      f"D_total: {running_losses['d_total']:.4f}, "
                      f"Steps/sec: {steps_per_sec:.4f}", flush=True)
                
                # Write to tensorboard
                if writer:
                    for k, v in running_losses.items():
                        writer.add_scalar(f'Loss/{k}', v, global_step)
                    writer.add_scalar('Stats/steps_per_sec', steps_per_sec, global_step)
                    writer.add_scalar('Stats/learning_rate', g_optimizer.param_groups[0]['lr'], global_step)
                
                # Reset running losses and timer
                running_losses = {k: 0 for k in running_losses}
                steps_since_log = 0
                start_time = time.time()
            
            # Generate and save sample images
            if rank == 0 and (global_step == 1 or global_step % config.SAMPLE_INTERVAL == 0):
                generator.eval()
                with torch.no_grad():
                    # Create a fixed transform for testing
                    test_transform = transforms.Compose([
                        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                        transforms.ToTensor(),
                    ])
                    
                    # Test sample 1 (3401.jpg)
                    try:
                        from PIL import Image
                        # Load test image 1
                        raw_img = Image.open(os.path.join(config.RAW_DIR, "3401.jpg")).convert("RGB")
                        aft_img = Image.open(os.path.join(config.AFT_DIR, "3401.jpg")).convert("RGB")
                        
                        raw_img = test_transform(raw_img)
                        aft_img = test_transform(aft_img)
                        
                        raw_img = raw_img.unsqueeze(0).to(device)
                        aft_img = aft_img.unsqueeze(0).to(device)
                        fake_img = generator(raw_img)
                        
                        save_path = os.path.join(config.RESULTS_DIR, f"test_sample_step_{global_step}.png")
                        create_comparison_grid(raw_img[0], fake_img[0], aft_img[0], save_path)
                    except Exception as e:
                        print(f"Error generating test sample 1: {e}")
                    
                    # Test sample 2 (1201.JPG)
                    try:
                        # Load test image 2
                        raw_img = Image.open(os.path.join(config.RAW_DIR, "1201.JPG")).convert("RGB")
                        aft_img = Image.open(os.path.join(config.AFT_DIR, "1201.JPG")).convert("RGB")
                        
                        raw_img = test_transform(raw_img)
                        aft_img = test_transform(aft_img)
                        
                        raw_img = raw_img.unsqueeze(0).to(device)
                        aft_img = aft_img.unsqueeze(0).to(device)
                        fake_img = generator(raw_img)
                        
                        save_path = os.path.join(config.RESULTS_DIR, f"train_sample_step_{global_step}.png")
                        create_comparison_grid(raw_img[0], fake_img[0], aft_img[0], save_path)
                    except Exception as e:
                        print(f"Error generating test sample 2: {e}")
                        
                generator.train()
            
            # Save checkpoint
            if rank == 0 and (global_step == 1 or global_step % config.SAVE_INTERVAL == 0):
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"beauty_gan_step_{global_step}.pth")
                save_checkpoint(model_dict, optimizer_dict, global_step, checkpoint_path)
            
            # Manual learning rate decay if scheduler fails
            if global_step % config.LR_DECAY_STEPS == 0:
                current_lr_g = g_optimizer.param_groups[0]['lr']
                current_lr_d = d_optimizer.param_groups[0]['lr']
                
                if rank == 0:
                    print(f"[Step {global_step}] Learning rate: G={current_lr_g:.6f}, D={current_lr_d:.6f}")
    
    # Save final model
    if rank == 0:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"beauty_gan_final.pth")
        save_checkpoint(model_dict, optimizer_dict, global_step, checkpoint_path)
        print(f"Training complete. Final model saved at {checkpoint_path}")
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    # Cleanup distributed training
    cleanup()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train face beautification GAN")
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    
    # Create directories
    for dir_path in [config.LOG_DIR, config.CHECKPOINT_DIR, config.RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get world size (number of GPUs)
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    if world_size == 0:
        raise ValueError("No GPUs found. This model requires at least one GPU for training.")
    
    # Launch processes
    mp.spawn(
        train_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()