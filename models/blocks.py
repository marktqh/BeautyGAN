"""
Basic building blocks for the face beautification model.
Includes convolutional blocks, transposed convolutional blocks, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional block with instance normalization.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
        use_norm (bool): Whether to use normalization
        use_act (bool): Whether to use activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_norm=True, use_act=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class TransposeConvBlock(nn.Module):
    """
    Transposed convolutional block with instance normalization.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
        output_padding (int): Output padding
        use_norm (bool): Whether to use normalization
        use_act (bool): Whether to use activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0, use_norm=True, use_act=True):
        super(TransposeConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=not use_norm)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for Channel-Wise Attention
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for bottleneck
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  # Scale feature maps


class ResidualBlock(nn.Module):
    """
    Residual Block with Skip Connection and SE module
    
    Args:
        channels (int): Number of input/output channels
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.se(x)
        return x + residual  # Skip connection
