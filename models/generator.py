"""
Generator model for the face beautification GAN.
Consists of an encoder, transformer bottleneck, and decoder with skip connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, TransposeConvBlock, ResidualBlock
from .transformer import TransformerBottleneck, DyT


class Encoder(nn.Module):
    """
    CNN-based encoder for the generator.
    
    Args:
        in_channels (int): Input channels
        base_channels (int): Base number of channels
        blocks (int): Number of downsampling blocks
    """
    def __init__(self, in_channels=3, base_channels=64, blocks=5):
        super(Encoder, self).__init__()
        
        # Initial convolution
        layers = [
            ConvBlock(in_channels, base_channels, kernel_size=7, stride=1, padding=3)
        ]
        
        # Downsampling layers
        in_ch = base_channels
        for i in range(blocks):
            out_ch = min(in_ch * 2, 512)
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            in_ch = out_ch
            
        self.layers = nn.ModuleList(layers)
        self.out_channels = in_ch
        
    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features


class Decoder(nn.Module):
    """
    CNN-based decoder with skip connections.
    
    Args:
        in_channels (int): Input channels from bottleneck
        out_channels (int): Output channels (usually 3 for RGB)
        blocks (int): Number of upsampling blocks
        dropout (float): Dropout probability
    """
    def __init__(self, in_channels, out_channels=3, blocks=5, dropout=0.0):
        super(Decoder, self).__init__()
        
        layers = []
        out_channels_per_block = []
        for i in range(blocks):
            out_ch = max(in_channels // 2, 64)
            layers.append(TransposeConvBlock(in_channels, out_ch, kernel_size=4, stride=2, padding=1))
            if i < 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            out_channels_per_block.append(out_ch)
            in_channels = out_ch

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        # Skip connections (assumed from encoder features)
        self.skip_channels = [512, 256, 128, 64]
        self.out_channels_per_block = [256, 128, 64, 64]  # Expected channels for skip addition

        self.skip_convs = nn.ModuleList()
        for sc, xc in zip(self.skip_channels, self.out_channels_per_block):
            if sc != xc:
                self.skip_convs.append(nn.Conv2d(sc, xc, kernel_size=1))
            else:
                self.skip_convs.append(nn.Identity())

    def forward(self, x, skip_features=None):
        if skip_features is not None:
            x_out = x
            # Exclude the final encoder feature; we expect len(skip_features)==4 for 4 blocks.
            skip_features = skip_features[:-1]
            skip_counter = 0
            for layer in self.layers[:-2]:
                x_out = layer(x_out)
                # Only apply skip addition when the layer is a TransposeConvBlock.
                if isinstance(layer, TransposeConvBlock) and skip_counter < len(skip_features):
                    skip_idx = len(skip_features) - 1 - skip_counter
                    skip = skip_features[skip_idx]
                    if x_out.shape[2:] != skip.shape[2:]:
                        x_out = F.interpolate(x_out, size=skip.shape[2:], mode='bilinear', align_corners=False)
                    skip = self.skip_convs[skip_counter](skip)
                    x_out = x_out + skip
                    skip_counter += 1
            x_out = self.layers[-2:](x_out)
            return x_out
        else:
            return self.layers(x)


class Generator(nn.Module):
    """
    Full generator with encoder, transformer bottleneck, and decoder.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        base_channels (int): Base number of channels
        down_blocks (int): Number of downsampling blocks
        transformer_blocks (int): Number of transformer blocks
        transformer_heads (int): Number of attention heads
        ffn_ratio (float): Ratio for hidden dimension in FFN
        dropout (float): Dropout probability
        init_alpha (float): Initial alpha value for DyT
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 down_blocks=4, transformer_blocks=3, transformer_heads=8, 
                 ffn_ratio=4.0, dropout=0.2, init_alpha=0.5):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = Encoder(in_channels, base_channels, down_blocks)
        
        # Transformer bottleneck
        self.transformer = TransformerBottleneck(
            self.encoder.out_channels, 
            transformer_blocks, 
            transformer_heads,
            ffn_ratio,
            dropout,
            init_alpha
        )
        
        # Decoder
        self.decoder = Decoder(self.encoder.out_channels, out_channels, down_blocks, dropout)
        
    def forward(self, x):
        # Encoder
        x, features = self.encoder(x)
        
        # Transformer bottleneck
        x = self.transformer(x)
        
        # Decoder with skip connections
        x = self.decoder(x, features)
        
        return x
