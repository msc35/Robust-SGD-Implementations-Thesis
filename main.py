#!/usr/bin/env python3
"""
Main script for running robust SGD benchmarking experiments.

This script implements the benchmark setup from the notebook, allowing you to run
experiments on CIFAR-100, MNIST, and CLOUD datasets with different algorithms:
- Standard SGD (baseline)
- Min-k Loss SGD (MKL-SGD)
- RHO-LOSS

Usage:
    python main.py --task cifar100 --algorithm uniform_sgd --epochs 100
    python main.py --task mnist --algorithm mkl_sgd --k_ratio 2.0
    python main.py --task cloud --algorithm rho_loss --selection_ratio 0.1
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import numpy as np

from src.data_loader import (
    NoisyIndexedDataset, AddGaussianNoise, CloudMergedDataset, ApplyTransformSubset
)
from src.models import MNIST_CNN, VGG_Small, Cloud_ResNet18
from src.trainers import (
    train_standard_sgd, train_min_k_loss, train_rho_loss,
    compute_irreducible_loss, train_il_model
)
from src.utils import validate, plot_results_custom, create_summary_table


# Hyperparameters (matching notebook exactly)
LEARNING_RATE = 0.001
MAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 256
DATA_ROOT = './data'
CLOUD_ROOT = './data/task_2_clouds'


def setup_device():
    """Setup and return the appropriate device."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device


def setup_cifar100_data(device):
    """Setup CIFAR-100 dataset with 40% label noise."""
    # Transforms
    transform_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Datasets
    cifar_train_dataset = NoisyIndexedDataset(
        'CIFAR100', DATA_ROOT, train=True, transform=transform_cifar_train,
        noise_type='symmetric', noise_rate=0.4  # 40% label noise
    )
    cifar_test_dataset = NoisyIndexedDataset(
        'CIFAR100', DATA_ROOT, train=False, transform=transform_cifar_test, noise_type='none'
    )
    
    # Holdout for RHO-LOSS
    c_clean = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform_cifar_train)
    cifar_holdout_dataset = Subset(c_clean, list(range(len(c_clean)-10000, len(c_clean))))
    
    # DataLoaders
    train_loader = DataLoader(cifar_train_dataset, batch_size=MAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(cifar_test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    holdout_loader = DataLoader(cifar_holdout_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_nored = nn.CrossEntropyLoss(reduction='none').to(device)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'holdout_loader': holdout_loader,
        'train_dataset': cifar_train_dataset,
        'criterion': criterion,
        'criterion_nored': criterion_nored,
        'num_classes': 100
    }


def setup_mnist_data(device):
    """Setup MNIST dataset with Gaussian input noise (std=1.5)."""
    # Transforms
    transform_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(mean=0., std=1.5, p=1.0)  # Hard mode: std=1.5
    ])
    transform_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Datasets
    mnist_train_dataset = NoisyIndexedDataset(
        'MNIST', DATA_ROOT, train=True, transform=transform_mnist_train, noise_type='none'
    )
    mnist_test_dataset = NoisyIndexedDataset(
        'MNIST', DATA_ROOT, train=False, transform=transform_mnist_test, noise_type='none'
    )
    
    # Holdout for RHO-LOSS
    m_clean = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform_mnist_test)
    mnist_holdout_dataset = Subset(m_clean, list(range(len(m_clean)-5000, len(m_clean))))
    
    # DataLoaders
    train_loader = DataLoader(mnist_train_dataset, batch_size=MAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(mnist_test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    holdout_loader = DataLoader(mnist_holdout_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_nored = nn.CrossEntropyLoss(reduction='none').to(device)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'holdout_loader': holdout_loader,
        'train_dataset': mnist_train_dataset,
        'criterion': criterion,
        'criterion_nored': criterion_nored,
        'num_classes': 10
    }


def setup_cloud_data(device):
    """Setup CLOUD dataset with Gaussian input noise (std=1.0)."""
    # Transforms
    transform_cloud_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AddGaussianNoise(mean=0., std=1.0, p=1.0)  # Hard mode: std=1.0
    ])
    transform_cloud_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Merge and split
    full_cloud_data = CloudMergedDataset(CLOUD_ROOT, transform=None)
    total_cloud = len(full_cloud_data)
    
    # Split: 50% Train, 10% Holdout, 40% Test
    train_size = int(0.50 * total_cloud)
    holdout_size = int(0.10 * total_cloud)
    test_size = total_cloud - train_size - holdout_size
    
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_cloud, generator=gen).tolist()
    
    train_indices = indices[:train_size]
    holdout_indices = indices[train_size : train_size + holdout_size]
    test_indices = indices[train_size + holdout_size:]
    
    # Create datasets with transforms
    cloud_train_ds = ApplyTransformSubset(full_cloud_data, train_indices, transform=transform_cloud_train)
    cloud_holdout_ds = ApplyTransformSubset(full_cloud_data, holdout_indices, transform=transform_cloud_test)
    cloud_test_ds = ApplyTransformSubset(full_cloud_data, test_indices, transform=transform_cloud_test)
    
    # DataLoaders
    train_loader = DataLoader(cloud_train_ds, batch_size=MAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    holdout_loader = DataLoader(cloud_holdout_ds, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(cloud_test_ds, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_nored = nn.CrossEntropyLoss(reduction='none').to(device)
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'holdout_loader': holdout_loader,
        'train_dataset': cloud_train_ds,
        'criterion': criterion,
        'criterion_nored': criterion_nored,
        'num_classes': len(full_cloud_data.classes)
    }


def get_il_map(task_name, data_config, device, checkpoint_dir):
    """Get or compute the Irreducible Loss map for RHO-LOSS."""
    il_map_path = os.path.join(checkpoint_dir, f"il_map_{task_name}.npy")
    il_model_path = os.path.join(checkpoint_dir, f"il_model_{task_name}.pth")
    
    # Load if exists
    if os.path.exists(il_map_path) and os.path.exists(il_model_path):
        print(f"Loading existing IL map from {il_map_path}")
        return np.load(il_map_path)
    
    # Otherwise, compute it
    print(f"Computing IL map for {task_name}...")
    
    # Create IL model
    if task_name == 'cifar100':
        il_model = VGG_Small(num_classes=100).to(device)
        num_epochs = 100
    elif task_name == 'mnist':
        il_model = MNIST_CNN().to(device)
        num_epochs = 30
    elif task_name == 'cloud':
        il_model = Cloud_ResNet18(num_classes=data_config['num_classes']).to(device)
        num_epochs = 20
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Train IL model on holdout
    il_model = train_il_model(
        il_model, data_config['holdout_loader'], data_config['test_loader'],
        device, num_epochs=num_epochs
    )
    torch.save(il_model.state_dict(), il_model_path)
    
    # Compute IL map
    il_map = compute_irreducible_loss(
        il_model, data_config['train_dataset'],
        data_config['criterion_nored'], device
    )
    np.save(il_map_path, il_map)
    
    return il_map


def run_training_experiment(
    algorithm, model, data_config, device, num_epochs, checkpoint_path=None,
    mkl_k_ratio=2.0, rho_il_map=None, rho_selection_ratio=0.1
):
    """
    Main training loop with checkpointing.
    
    Returns:
        Dictionary with training history: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    """
    train_loader = data_config['train_loader']
    test_loader = data_config['test_loader']
    criterion = data_config['criterion']
    criterion_nored = data_config['criterion_nored']
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_epoch = 0
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            train_losses = checkpoint.get('train_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            print(f"Resuming training from Epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint, starting from scratch. Error: {e}")
            start_epoch = 0

    print(f"--- Starting Training: {algorithm} ---")

    for epoch in range(start_epoch, num_epochs):
        # Select training function based on algorithm
        if algorithm == 'uniform_sgd':
            train_loss, train_acc = train_standard_sgd(
                model, train_loader, criterion, optimizer, device
            )
        elif algorithm == 'mkl_sgd':
            train_loss, train_acc = train_min_k_loss(
                model, train_loader, criterion_nored, optimizer, device, k_ratio=mkl_k_ratio
            )
        elif algorithm == 'rho_loss':
            if rho_il_map is None:
                raise ValueError("rho_il_map must be provided for RHO-LOSS")
            train_loss, train_acc = train_rho_loss(
                model, rho_il_map, train_loader, criterion_nored, optimizer,
                device, selection_ratio=rho_selection_ratio
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Validation
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        # History
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Tr Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        if checkpoint_path:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            torch.save(state, checkpoint_path)
            if is_best:
                best_path = checkpoint_path.replace('.pth', '_best.pth')
                torch.save(state, best_path)
                print(f"  [New Best] Saved to {best_path}")

    print(f"--- Finished: {algorithm} ---")
    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }


def main():
    parser = argparse.ArgumentParser(description='Robust SGD Benchmarking')
    parser.add_argument('--task', type=str, required=True,
                        choices=['cifar100', 'mnist', 'cloud'],
                        help='Dataset to use')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['uniform_sgd', 'mkl_sgd', 'rho_loss'],
                        help='Training algorithm')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--k_ratio', type=float, default=2.0,
                        help='k_ratio for MKL-SGD (default: 2.0)')
    parser.add_argument('--selection_ratio', type=float, default=0.1,
                        help='Selection ratio for RHO-LOSS (default: 0.1)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    print(f"\n=== Setting up {args.task.upper()} dataset ===")
    if args.task == 'cifar100':
        data_config = setup_cifar100_data(device)
        model = VGG_Small(num_classes=100).to(device)
    elif args.task == 'mnist':
        data_config = setup_mnist_data(device)
        model = MNIST_CNN().to(device)
    elif args.task == 'cloud':
        data_config = setup_cloud_data(device)
        model = Cloud_ResNet18(num_classes=data_config['num_classes']).to(device)
    
    # Get IL map for RHO-LOSS if needed
    rho_il_map = None
    if args.algorithm == 'rho_loss':
        rho_il_map = get_il_map(args.task, data_config, device, args.checkpoint_dir)
    
    # Setup checkpoint path
    checkpoint_name = f"{args.task}_{args.algorithm}"
    if args.algorithm == 'mkl_sgd':
        checkpoint_name += f"_k{args.k_ratio}"
    elif args.algorithm == 'rho_loss':
        checkpoint_name += f"_sel{args.selection_ratio}"
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{checkpoint_name}.pth")
    
    if args.resume:
        checkpoint_path = args.resume
    
    # Run training
    results = run_training_experiment(
        algorithm=args.algorithm,
        model=model,
        data_config=data_config,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        mkl_k_ratio=args.k_ratio,
        rho_il_map=rho_il_map,
        rho_selection_ratio=args.selection_ratio
    )
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Best Val Acc: {max(results['val_acc'])*100:.2f}%")
    print(f"Final Val Acc: {results['val_acc'][-1]*100:.2f}%")
    
    # Plot results
    plot_results_custom(
        [results],
        [args.algorithm],
        title_prefix=f"{args.task.upper()} - {args.algorithm}"
    )
    
    # Create summary table
    table = create_summary_table([results], [args.algorithm])
    if table is not None:
        print("\n=== Summary Table ===")
        print(table)


if __name__ == '__main__':
    main()

