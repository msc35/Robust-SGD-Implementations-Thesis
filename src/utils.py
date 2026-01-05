"""
Utility functions for validation and visualization.

This module provides:
1. Validation function for evaluating models on test sets
2. Plotting functions for visualizing training curves and results
3. Summary table generation for comparing algorithms
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def validate(model, test_loader, criterion, device):
    """
    Standard validation loop to compute loss and accuracy on test/validation set.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test/validation data
        criterion: Loss function with 'mean' reduction
        device: The device to run on (cpu, cuda, or mps)
        
    Returns:
        epoch_loss: Average test loss
        epoch_acc: Test accuracy
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_samples = 0
    total_samples = 0

    with torch.no_grad():
        # Handle both (inputs, labels) and (inputs, labels, idx) formats
        for batch in test_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Use mean reduction loss

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_samples / total_samples

    return epoch_loss, epoch_acc


def plot_results_custom(all_results, algorithm_names, title_prefix=""):
    """
    Plots the training/validation loss and accuracy for a list of experiments.
    
    Args:
        all_results: List of result dictionaries, each containing:
                    {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        algorithm_names: List of algorithm names (strings)
        title_prefix: Prefix for plot titles
    """
    if not all_results:
        print(f"Skipping plot '{title_prefix}': No results provided.")
        return

    # Find the shortest number of epochs in case one run was cut short
    num_epochs = min(len(res['val_loss']) for res in all_results if res.get('val_loss'))
    if num_epochs == 0:
        print(f"Skipping plot '{title_prefix}': No epoch data found.")
        return

    epochs = range(1, num_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # --- Plot 1: Loss ---
    ax1.set_title(f"{title_prefix} - Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    
    for results, name in zip(all_results, algorithm_names):
        if results.get('val_loss'):
            # Plot Validation Loss (Solid Line)
            ax1.plot(epochs, results['val_loss'][:num_epochs], label=f'{name}', linewidth=2)
        
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Accuracy ---
    ax2.set_title(f"{title_prefix} - Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    
    for results, name in zip(all_results, algorithm_names):
        if results.get('val_acc'):
            # Plot Validation Accuracy (Solid Line)
            val_acc_percent = [acc * 100 for acc in results['val_acc'][:num_epochs]]
            ax2.plot(epochs, val_acc_percent, label=f'{name}', linewidth=2)
        
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def create_summary_table(all_results, algorithm_names):
    """
    Creates a pandas DataFrame to summarize the key metrics.
    
    Args:
        all_results: List of result dictionaries
        algorithm_names: List of algorithm names
        
    Returns:
        Styled pandas DataFrame with summary metrics
    """
    summary_data = {
        'Algorithm': [],
        'Best Val Acc (%)': [],
        'Epoch': [],
        'Final Val Acc (%)': [],
        'Final Train Acc (%)': []
    }
    
    for results, name in zip(all_results, algorithm_names):
        if not results.get('val_acc'):  # Skip if no data
            continue
            
        # Find best validation accuracy and its epoch
        best_val_acc = max(results['val_acc']) * 100
        best_epoch = np.argmax(results['val_acc']) + 1  # +1 for 1-based epoch
        
        # Get final metrics
        final_val_acc = results['val_acc'][-1] * 100
        final_train_acc = results['train_acc'][-1] * 100
        
        summary_data['Algorithm'].append(name)
        summary_data['Best Val Acc (%)'].append(best_val_acc)
        summary_data['Epoch'].append(best_epoch)
        summary_data['Final Val Acc (%)'].append(final_val_acc)
        summary_data['Final Train Acc (%)'].append(final_train_acc)
        
    df = pd.DataFrame(summary_data)
    
    if df.empty:
        print("Summary table is empty.")
        return None

    # Format for better readability
    return df.set_index('Algorithm').style.format({
        'Best Val Acc (%)': '{:.2f}',
        'Final Val Acc (%)': '{:.2f}',
        'Final Train Acc (%)': '{:.2f}'
    }).highlight_max(subset=['Best Val Acc (%)'], color='lightgreen')


def plot_task_history(results_dict, task_title):
    """
    Plots Train/Val Loss and Accuracy for a single task.
    
    Args:
        results_dict: dict { 'AlgoName': {'train_loss':[], ...}, ... }
        task_title: str
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    # Color palette to ensure distinct lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (name, metrics), color in zip(results_dict.items(), colors):
        if not metrics.get('val_loss'):
            continue
        epochs = range(1, len(metrics['val_loss']) + 1)
        
        # --- Plot 1: Validation Loss ---
        ax1.plot(epochs, metrics['val_loss'], label=name, linewidth=2.5, color=color)

        # --- Plot 2: Validation Accuracy ---
        val_acc_pct = [x * 100 for x in metrics['val_acc']]
        ax2.plot(epochs, val_acc_pct, label=name, linewidth=2.5, color=color)

    # Styling Plot 1 (Loss)
    ax1.set_title(f"{task_title} - Validation Loss (Lower is Better)", fontsize=14)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Styling Plot 2 (Accuracy)
    ax2.set_title(f"{task_title} - Validation Accuracy (Higher is Better)", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

