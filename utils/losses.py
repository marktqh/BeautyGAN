"""
Loss functions for the face beautification GAN.
Includes GAN loss, perceptual loss, and TV loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LSGANLoss(nn.Module):
    """
    Least Squares GAN loss.
    
    Args:
        target_real (float): Target value for real samples
        target_fake (float): Target value for fake samples
    """
    def __init__(self, target_real=1.0, target_fake=0.0):
        super(LSGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real))
        self.register_buffer('fake_label', torch.tensor(target_fake))
        self.loss = nn.MSELoss()
        
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss without global normalization.
    
    Args:
        layers (list): List of layer indices to extract features from.
        weights (list): Weights for each layer's contribution.
        device (str): Device to run the model on.
    """
    def __init__(self, layers=[2, 7, 12, 21, 30], 
                 weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0], 
                 device='cuda'):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19 and freeze parameters.
        vgg = models.vgg19(pretrained=True).features.to(device)
        vgg.eval()
        self.vgg = vgg
        self.layers = layers
        self.weights = weights
        
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Dictionary to store activations.
        self.activations = {}
        for i, layer in enumerate(self.vgg):
            if i in self.layers:
                def hook_fn(module, input, output, layer_idx=i):
                    self.activations[layer_idx] = output
                self.vgg[i].register_forward_hook(hook_fn)
                
    def forward(self, x, y):
        # VGG preprocessing: normalize with ImageNet mean and std.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Clear activations and compute features for x.
        self.activations.clear()
        self.vgg(x)
        x_activations = {k: v.detach() for k, v in self.activations.items()}
        
        # Clear activations and compute features for y.
        self.activations.clear()
        self.vgg(y)
        y_activations = {k: v for k, v in self.activations.items()}
        
        # Calculate the perceptual loss without global normalization.
        loss = 0
        for i, layer_idx in enumerate(self.layers):
            x_feat = x_activations[layer_idx]
            y_feat = y_activations[layer_idx]
            layer_loss = self.weights[i] * F.mse_loss(x_feat, y_feat)
            loss += layer_loss
        return loss


def total_variation_loss(x):
    """
    Compute Total Variation (TV) loss for noise reduction.
    TV loss encourages spatial smoothness in the generated image.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
    Returns:
        torch.Tensor: Total variation loss
    """
    # Compute difference in height dimension
    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    # Compute difference in width dimension
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    # Sum both differences and compute mean
    return diff_h.mean() + diff_w.mean()
