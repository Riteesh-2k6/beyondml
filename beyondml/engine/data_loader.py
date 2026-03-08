"""
Data Loader for Image Datasets using torchvision.
Provides beginner-friendly access to MNIST, CIFAR-10, and custom image folders.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any

def get_transforms(img_size: int = 32, normalize: bool = True):
    """Get standard transforms for image classification."""
    t_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        t_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(t_list)

def load_mnist(batch_size: int = 64, data_dir: str = "data/mnist") -> Tuple[DataLoader, DataLoader]:
    """Load and return MNIST train and test loaders."""
    os.makedirs(data_dir, exist_ok=True)
    transform = get_transforms(img_size=28)
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_cifar10(batch_size: int = 64, data_dir: str = "data/cifar10") -> Tuple[DataLoader, DataLoader]:
    """Load and return CIFAR-10 train and test loaders."""
    os.makedirs(data_dir, exist_ok=True)
    transform = get_transforms(img_size=32)
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_custom_images(data_dir: str, batch_size: int = 32, split_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """Load custom images from a directory structured as ImageFolder."""
    transform = get_transforms()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_map = {v: k for k, v in full_dataset.class_to_idx.items()}
    return train_loader, test_loader, class_map
