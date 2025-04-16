"""
Transformer-based modules for face beautification model.
Includes DyT (Dynamic Tanh) normalization layers as an alternative to LayerNorm.
Based on "Transformers without Normalization" paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) Layer - Replacement for LayerNorm in Transformer blocks.
    DyT(x) = γ * tanh(αx) + β
    
    Args:
        dim (int): Feature dimension
        init_alpha (float): Initial value for the α parameter
    """
    def __init__(self, dim, init_alpha=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if x.dim() == 4:  # For 2D convolution (B, C, H, W)
            return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
        elif x.dim() == 3:  # For sequences/transformers (B, L, C)
            return self.gamma.view(1, 1, -1) * x + self.beta.view(1, 1, -1)
        return self.gamma * x + self.beta


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for Transformer blocks.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Project to query, key, value
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed Forward Network for Transformer blocks.
    
    Args:
        dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout probability
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block with DyT normalization.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        ffn_ratio (float): Ratio for hidden dimension in FFN
        dropout (float): Dropout probability
        init_alpha (float): Initial alpha value for DyT
    """
    def __init__(self, dim, num_heads=8, ffn_ratio=4.0, dropout=0.0, init_alpha=0.5):
        super(TransformerBlock, self).__init__()
        self.attn_norm = DyT(dim, init_alpha)
        self.ffn_norm = DyT(dim, init_alpha)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, int(dim * ffn_ratio), dropout)
        
    def forward(self, x):
        # Self-attention
        norm_x = self.attn_norm(x)
        x = x + self.attn(norm_x)
        
        # Feed-forward
        norm_x = self.ffn_norm(x)
        x = x + self.ffn(norm_x)
        
        return x


class TransformerBottleneck(nn.Module):
    """
    Transformer bottleneck for the generator.
    
    Args:
        in_channels (int): Input channels
        num_blocks (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        ffn_ratio (float): Ratio for hidden dimension in FFN
        dropout (float): Dropout probability
        init_alpha (float): Initial alpha value for DyT
    """
    def __init__(self, in_channels, num_blocks=3, num_heads=8, ffn_ratio=4.0, dropout=0.0, init_alpha=0.5):
        super(TransformerBottleneck, self).__init__()
        self.in_channels = in_channels
        
        # Pre-norm
        self.pre_norm = DyT(in_channels, init_alpha)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channels, num_heads, ffn_ratio, dropout, init_alpha)
            for _ in range(num_blocks)
        ])
        
        # Post-norm
        self.post_norm = DyT(in_channels, init_alpha)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Reshape to sequence
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Pre-norm
        x = self.pre_norm(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Post-norm
        x = self.post_norm(x)
        
        # Reshape back to spatial
        x = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        
        return x
