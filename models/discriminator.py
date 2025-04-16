"""
Discriminator model for the face beautification GAN.
Implements a PatchGAN discriminator with spectral normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalization.
    
    Args:
        in_channels (int): Input channels (raw + beautified)
        base_channels (int): Base number of channels
        n_layers (int): Number of convolutional layers
    """
    def __init__(self, in_channels=6, base_channels=64, n_layers=3):
        super(Discriminator, self).__init__()
        
        # Initial convolution without normalization
        sequence = [
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Downsampling convolutions with spectral normalization
        in_ch = base_channels
        for i in range(n_layers):
            out_ch = min(in_ch * 2, 512)
            sequence += [
                spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_ch = out_ch
            
        # Output convolution
        sequence += [
            spectral_norm(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, raw, beautified):
        """
        Forward pass of the discriminator.
        
        Args:
            raw (Tensor): Raw input image
            beautified (Tensor): Beautified image (either generated or ground truth)
            
        Returns:
            Tensor: Patch-based prediction map (not averaged)
        """
        # Concatenate raw and beautified images along the channel dimension
        x = torch.cat([raw, beautified], dim=1)
        return self.model(x)