"""
Data loading and noise injection utilities.

This module provides custom PyTorch Dataset classes for:
1. Adding label noise (symmetric noise for CIFAR-100)
2. Adding input noise (Gaussian noise for MNIST and CLOUD)
3. Index tracking for RHO-LOSS and sample history tracking
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100, MNIST, ImageFolder
from PIL import Image


class AddGaussianNoise(object):
    """
    Transform that adds Gaussian noise to input tensors.
    
    Args:
        mean: Mean of the Gaussian noise (default: 0.0)
        std: Standard deviation of the Gaussian noise (default: 1.0)
        p: Probability of applying noise (default: 0.5)
    """
    def __init__(self, mean=0., std=1., p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor


class NoisyIndexedDataset(Dataset):
    """
    Wraps a torchvision dataset to add:
    1. Symmetric label noise
    2. An index to each returned sample: (data, label, idx)
    
    This index is essential for RHO-LOSS (to look up Irreducible Loss)
    and for tracking historical loss of each sample.
    
    Args:
        dataset_name: 'CIFAR100', 'MNIST', or 'CLOUD'
        root: Root directory for the dataset
        train: Whether to use training set
        transform: Transform to apply to images
        download: Whether to download the dataset
        noise_type: Type of noise ('symmetric' or 'none')
        noise_rate: Fraction of labels to corrupt (e.g., 0.4 for 40%)
        random_seed: Random seed for reproducibility
    """
    def __init__(self, dataset_name, root, train=True, transform=None, download=True, 
                 noise_type='none', noise_rate=0.0, random_seed=42):
        
        self.transform = transform
        self.dataset_name = dataset_name
        
        # Load base dataset
        if dataset_name == 'CIFAR100':
            self.base_dataset = CIFAR100(root=root, train=train, transform=transform, download=download)
            self.num_classes = 100
        elif dataset_name == 'MNIST':
            self.base_dataset = MNIST(root=root, train=train, transform=transform, download=download)
            self.num_classes = 10
        elif dataset_name == 'CLOUD':
            self.base_dataset = ImageFolder(root=root, transform=transform)
            self.num_classes = len(self.base_dataset.classes)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # Extract targets
        if hasattr(self.base_dataset, 'targets'):
            self.targets = np.array(self.base_dataset.targets)
        else:
            # Fallback: extract targets from samples if .targets is missing
            self.targets = np.array([s[1] for s in self.base_dataset.samples])

        if hasattr(self.base_dataset, 'data'):
            self.data = self.base_dataset.data
        else:
            # For ImageFolder (CLOUD), we store file paths
            self.data = [s[0] for s in self.base_dataset.samples]

        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(random_seed)
        
        self.original_targets = self.targets.copy()
        self.noisy_targets = self.targets.copy()
        self.noise_mask = np.zeros(len(self.targets), dtype=bool)

        if train and self.noise_type != 'none' and self.noise_rate > 0:
            self._apply_label_noise()

    def _apply_label_noise(self):
        """Modifies self.noisy_targets with the specified noise."""
        num_samples = len(self.targets)
        num_noisy = int(num_samples * self.noise_rate)
        
        # Select indices to corrupt
        noisy_indices = self.rng.choice(num_samples, num_noisy, replace=False)
        self.noise_mask[noisy_indices] = True
        
        if self.noise_type == 'symmetric':
            print(f"Applying {self.noise_rate*100}% symmetric label noise...")
            for i in noisy_indices:
                original_label = self.targets[i]
                
                # Generate a random new label, different from the original
                new_label_candidates = list(range(self.num_classes))
                new_label_candidates.remove(original_label)
                
                new_label = self.rng.choice(new_label_candidates)
                self.noisy_targets[i] = new_label
            
            # Verify noise
            actual_noise = (self.noisy_targets != self.original_targets).mean()
            print(f"Noise applied. Original targets modified. Actual noise rate: {actual_noise:.4f}")
        else:
            print("No noise type specified or 'none', labels remain clean.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns (data, label, index).
        Note: The transform is applied here (internal to base_dataset for ImageFolder/CIFAR).
        """
        data, _ = self.base_dataset[idx] 
        label = self.noisy_targets[idx]
        
        return data, label, idx


class CloudMergedDataset(Dataset):
    """
    Reads from both 'clouds_train' and 'clouds_test', merges them,
    and allows applying different transforms based on the split.
    
    Args:
        root_dir: Root directory containing 'clouds_train' and 'clouds_test' folders
        transform: Transform to apply to images (can be None if applying later)
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Define classes (assume consistent across folders)
        train_dir = os.path.join(root_dir, 'clouds_train')
        self.classes = sorted([d.name for d in os.scandir(train_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all images from both Train and Test folders
        folders_to_scan = [
            os.path.join(root_dir, 'clouds_train'), 
            os.path.join(root_dir, 'clouds_test')
        ]
        
        for folder in folders_to_scan:
            for cls_name in self.classes:
                cls_dir = os.path.join(folder, cls_name)
                if not os.path.exists(cls_dir):
                    continue
                
                # Get all images
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.samples.append((
                            os.path.join(cls_dir, img_name), 
                            self.class_to_idx[cls_name]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns (image, target, idx) for compatibility with RHO-LOSS.
        """
        path, target = self.samples[idx]
        # Load image
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Return (data, label, idx) for HASA/RHO-Loss compatibility
        return img, target, idx


class ApplyTransformSubset(Dataset):
    """
    Wrapper that applies a transform to a subset of a dataset.
    Used for splitting Cloud dataset into train/holdout/test sets.
    
    Args:
        underlying_dataset: The base dataset (e.g., CloudMergedDataset)
        indices: List of indices to include in this subset
        transform: Transform to apply to images
    """
    def __init__(self, underlying_dataset, indices, transform):
        self.dataset = underlying_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # Map local subset index to global dataset index
        global_idx = self.indices[idx]
        
        # Get raw data (path, target)
        path, target = self.dataset.samples[global_idx]
        
        # Load and Transform
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        # Return global_idx to track sample history uniquely
        return img, target, global_idx

