"""
Time-series data augmentation techniques.
"""
import torch
import numpy as np

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class GaussianNoise:
    """Add Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class FeatureMasking:
    """Randomly mask features in a tensor."""
    def __init__(self, mask_ratio=0.1):
        self.mask_ratio = mask_ratio

    def __call__(self, tensor):
        mask = torch.rand(tensor.size()) > self.mask_ratio
        return tensor * mask.float()

def get_train_augmentations(noise_std=0.02, mask_ratio=0.05):
    """Get a composition of training augmentations."""
    return Compose([
        GaussianNoise(std=noise_std),
        FeatureMasking(mask_ratio=mask_ratio)
    ])



