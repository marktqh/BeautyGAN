"""
Dataset and dataloader utilities for face beautification GAN.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import rawpy


class BeautificationDataset(Dataset):
    """
    Dataset for facial beautification.
    
    Args:
        raw_dir (str): Directory containing raw images
        aft_dir (str): Directory containing beautified images
        transform (callable, optional): Optional transform to be applied on a sample
        indices (range or list, optional): Specific indices to use
    """
    def __init__(self, raw_dir, aft_dir, transform=None, indices=range(1, 4001)):
        self.raw_dir = raw_dir
        self.aft_dir = aft_dir
        self.transform = transform
        self.extensions = [".jpg", ".JPG", ".jpeg", ".png", ".webp", ".arw", ".cr2"]
        self.pairs = []
        self.indices = indices

        for i in self.indices:
            if i < 1000:
                index_str = f"{i:03d}"
            else:
                index_str = str(i)

            raw_exist = False
            aft_exist = False

            # Find raw image file
            for ext in self.extensions:
                raw_path = os.path.join(raw_dir, index_str + ext)
                if os.path.isfile(raw_path):
                    raw_exist = True
                    break
                    
            # Find corresponding aft image file
            for ext in self.extensions:
                aft_path = os.path.join(aft_dir, index_str + ext)
                if os.path.isfile(raw_path) and os.path.isfile(aft_path):
                    aft_exist = True
                    break
                    
            # Add to pairs if both exist
            if raw_exist and aft_exist:
                self.pairs.append((raw_path, aft_path, index_str))
                    
    def __len__(self):
        return len(self.pairs)
    
    def load_raw_image(self, filepath):
        """Load raw camera file (e.g., .arw, .cr2) using rawpy"""
        with rawpy.imread(filepath) as raw:
            # Process raw data (demosaic, etc.) to get an RGB image
            rgb = raw.postprocess()
        return rgb
    
    def load_raw_as_pil(self, filepath):
        """Load raw camera file as PIL Image"""
        rgb = self.load_raw_image(filepath)
        # Convert from uint16 if necessary
        if rgb.dtype != np.uint8:
            rgb = (rgb / 65535.0 * 255.0).astype(np.uint8)
        return Image.fromarray(rgb)
        
    def __getitem__(self, idx):
        raw_path, aft_path, index_str = self.pairs[idx]
        
        # Load images based on file extension
        if raw_path.lower().endswith(('.arw', '.cr2')):
            raw_img = self.load_raw_as_pil(raw_path)
        else:
            raw_img = Image.open(raw_path).convert("RGB")
            
        if aft_path.lower().endswith(('.arw', '.cr2')):
            aft_img = self.load_raw_as_pil(aft_path)
        else:
            aft_img = Image.open(aft_path).convert("RGB")
        
        # Store original size
        original_size = raw_img.size  # (width, height)
        
        # Apply transforms
        if self.transform:
            raw_img = self.transform(raw_img)
            aft_img = self.transform(aft_img)
            
        return raw_img, aft_img, original_size, index_str


def create_dataloaders(raw_dir, aft_dir, transform, batch_size_train=4, batch_size_test=1, num_workers=4, seed=42):
    """
    Create train and test dataloaders for face beautification.
    
    Args:
        raw_dir (str): Directory containing raw images
        aft_dir (str): Directory containing beautified images
        transform (callable): Transform to be applied on images
        batch_size_train (int): Batch size for training
        batch_size_test (int): Batch size for testing
        num_workers (int): Number of workers for dataloaders
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create indices for train/test split (90/10)
    indices = list(range(1, 4001))
    np.random.shuffle(indices)
    split_index = int(0.9 * len(indices))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # Ensure specific samples are in the right split (for monitoring/comparison)
    if 1201 not in train_indices:
        train_indices.append(1201)
        if 1201 in test_indices:
            test_indices.remove(1201)
            
    if 3401 not in test_indices:
        test_indices.append(3401)
        if 3401 in train_indices:
            train_indices.remove(3401)
    
    # Create datasets
    train_dataset = BeautificationDataset(raw_dir, aft_dir, transform, train_indices)
    test_dataset = BeautificationDataset(raw_dir, aft_dir, transform, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, test_loader
