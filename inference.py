"""
Inference script for face beautification GAN.
"""
import os
import argparse
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import config
from models import Generator
from utils.visualization import create_comparison_grid


def preprocess_image(image_path, target_size=None):
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image
        target_size (tuple, optional): Target size to resize the image
        
    Returns:
        tuple: (preprocessed image tensor, original size, original PIL image)
    """
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    if target_size:
        img_resized = img.resize(target_size, Image.LANCZOS)
    else:
        img_resized = img
        
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    return img_tensor, original_size, img


def run_inference(model_path, input_path, output_path, device='cuda', compare_with=None):
    """
    Run inference on an image.
    
    Args:
        model_path (str): Path to the model checkpoint
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
        device (str): Device to run on
        compare_with (str, optional): Path to the comparison image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
        
    generator.eval()
    
    # Preprocess image
    img_tensor, original_size, img = preprocess_image(input_path, target_size=(config.IMG_SIZE, config.IMG_SIZE))
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = generator(img_tensor)
    
    # Convert output to PIL image and resize to original size
    output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    output_img = output_img.resize(original_size, Image.LANCZOS)
    
    # If compare_with is provided, create comparison grid
    if compare_with:
        try:
            comparison_img = Image.open(compare_with).convert('RGB')
            comparison_tensor = transforms.ToTensor()(comparison_img)
            
            # Create and save comparison grid
            create_comparison_grid(
                transforms.ToTensor()(img), 
                transforms.ToTensor()(output_img), 
                comparison_tensor, 
                output_path
            )
        except Exception as e:
            print(f"Error creating comparison grid: {e}")
            # Fall back to saving just the output
            output_img.save(output_path)
    else:
        # Create a side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title('Input')
        axes[0].axis('off')
        axes[1].imshow(output_img)
        axes[1].set_title('Beautified')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    print(f"Inference complete. Output saved to {output_path}")
    
    return output_img


def process_directory(model_path, input_dir, output_dir, device='cuda'):
    """
    Process all images in a directory.
    
    Args:
        model_path (str): Path to the model checkpoint
        input_dir (str): Path to the input directory
        output_dir (str): Path to the output directory
        device (str): Device to run on
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.webp', '*.WEBP']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Load model (only once for all images)
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
        
    generator.eval()
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        try:
            # Get filename without extension
            filename = os.path.basename(image_path)
            filename_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{filename_without_ext}_beautified.png")
            
            # Preprocess image
            img_tensor, original_size, img = preprocess_image(
                image_path, 
                target_size=(config.IMG_SIZE, config.IMG_SIZE)
            )
            img_tensor = img_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                output = generator(img_tensor)
            
            # Convert output to PIL image and resize to original size
            output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
            output_img = output_img.resize(original_size, Image.LANCZOS)
            
            # Create and save comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img)
            axes[0].set_title('Input')
            axes[0].axis('off')
            axes[1].imshow(output_img)
            axes[1].set_title('Beautified')
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            print(f"[{i+1}/{len(image_paths)}] Processed {filename} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print(f"All images processed. Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run inference with face beautification GAN")
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to output image or directory')
    parser.add_argument('--compare-with', type=str, default=None, help='Path to comparison image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Find latest checkpoint if model is not specified
    if args.model is None:
        args.model = os.path.join(config.CHECKPOINT_DIR, "beauty_gan_final.pth")
        if not os.path.exists(args.model):
            # Find latest checkpoint
            checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "beauty_gan_step_*.pth"))
            if checkpoint_files:
                args.model = max(checkpoint_files, key=os.path.getctime)
                print(f"Using latest checkpoint: {args.model}")
            else:
                raise ValueError("No model checkpoint found. Please specify --model.")
    
    # Set default output path if not specified
    if args.output is None:
        if os.path.isdir(args.input):
            args.output = os.path.join(config.RESULTS_DIR, "inference")
        else:
            filename = os.path.splitext(os.path.basename(args.input))[0]
            args.output = os.path.join(config.RESULTS_DIR, f"{filename}_beautified.png")
    
    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        process_directory(args.model, args.input, args.output, args.device)
    else:
        run_inference(args.model, args.input, args.output, args.device, args.compare_with)


if __name__ == "__main__":
    main()
