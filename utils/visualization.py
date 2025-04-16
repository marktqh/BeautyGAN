"""
Visualization utilities for face beautification GAN.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch


def create_comparison_grid(raw_img, output_img, target_img, save_path):
    """
    Create and save a comparison grid showing raw, output, and target images.
    
    Args:
        raw_img (torch.Tensor): Raw input image
        output_img (torch.Tensor): Output image from generator
        target_img (torch.Tensor): Target beautified image
        save_path (str): Path to save the comparison grid
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to numpy arrays and transpose to HWC format
    raw_np = raw_img.cpu().detach().numpy().transpose(1, 2, 0)
    output_np = output_img.cpu().detach().numpy().transpose(1, 2, 0)
    target_np = target_img.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1]
    raw_np = np.clip(raw_np, 0, 1)
    output_np = np.clip(output_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)

    # Assuming all images have same shape (H, W, C) – use raw_np for base
    img_height, img_width = raw_np.shape[:2]
    aspect_ratio = img_width / img_height

    # Set a base height (in inches), scale width by aspect ratio
    base_height_inch = 5  # controls overall vertical size
    fig_width = base_height_inch * aspect_ratio * 3  # 3 images side-by-side
    fig_height = base_height_inch * 1.4
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    
    # Display images
    axes[0].imshow(raw_np)
    axes[0].set_title('Input (Raw)')
    axes[0].axis('off')
    
    axes[1].imshow(output_np)
    axes[1].set_title('Model Output')
    axes[1].axis('off')
    
    axes[2].imshow(target_np)
    axes[2].set_title('Target (Beautified)')
    axes[2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison image saved to {save_path}")


def save_batch_comparison(input_batch, output_batch, target_batch, save_path, nrow=4):
    """
    Save a grid of comparisons from batches of images.
    
    Args:
        input_batch (torch.Tensor): Batch of input images (B, C, H, W)
        output_batch (torch.Tensor): Batch of output images (B, C, H, W)
        target_batch (torch.Tensor): Batch of target images (B, C, H, W)
        save_path (str): Path to save the comparison grid
        nrow (int): Number of images per row in the grid
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ensure all tensors are on CPU and detached from graph
    input_batch = input_batch.cpu().detach()
    output_batch = output_batch.cpu().detach()
    target_batch = target_batch.cpu().detach()
    
    # Concatenate along batch dimension in the order: input, output, target
    batch_size = input_batch.size(0)
    
    # Create composite images with borders to separate them
    composite_batch = []
    for i in range(batch_size):
        # Create a white border around output and target for visual separation
        border_width = 2
        border_color = 1.0  # white
        
        # Create the composite image: input | output | target
        composite = torch.cat([
            input_batch[i],
            output_batch[i],
            target_batch[i]
        ], dim=2)  # Concatenate horizontally (along width)
        
        composite_batch.append(composite)
    
    # Stack the composites back into a batch
    composite_batch = torch.stack(composite_batch)
    
    # Create and save the grid
    grid = make_grid(composite_batch, nrow=nrow)
    save_image(grid, save_path)
    
    print(f"Batch comparison saved to {save_path}")


def visualize_attention_maps(attention_maps, input_image, save_path):
    """
    Visualize attention maps from transformer layers.
    
    Args:
        attention_maps (list): List of attention maps
        input_image (torch.Tensor): Input image
        save_path (str): Path to save the visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ensure input_image is on CPU and detached from graph
    input_image = input_image.cpu().detach().numpy().transpose(1, 2, 0)
    input_image = np.clip(input_image, 0, 1)
    
    # Number of attention heads to visualize
    num_heads = len(attention_maps[0])
    num_layers = len(attention_maps)
    
    # Create figure
    fig, axes = plt.subplots(num_layers, num_heads + 1, figsize=(3 * (num_heads + 1), 3 * num_layers))
    
    # Show input image in first column of each row
    for i in range(num_layers):
        if num_layers > 1:
            axes[i, 0].imshow(input_image)
            axes[i, 0].set_title(f'Input')
            axes[i, 0].axis('off')
        else:
            axes[0].imshow(input_image)
            axes[0].set_title(f'Input')
            axes[0].axis('off')
    
    # Show attention maps
    for i, layer_maps in enumerate(attention_maps):
        for j, attn_map in enumerate(layer_maps):
            # Reshape attention map to image shape if necessary
            attn_map = attn_map.cpu().detach().numpy()
            
            if num_layers > 1:
                axes[i, j + 1].imshow(attn_map, cmap='viridis')
                axes[i, j + 1].set_title(f'Layer {i+1}, Head {j+1}')
                axes[i, j + 1].axis('off')
            else:
                axes[j + 1].imshow(attn_map, cmap='viridis')
                axes[j + 1].set_title(f'Head {j+1}')
                axes[j + 1].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Attention maps saved to {save_path}")
