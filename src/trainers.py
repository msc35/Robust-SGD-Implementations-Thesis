"""
Training functions for robust SGD algorithms.

This module implements three training algorithms:
1. Standard SGD: Uniform random sampling (baseline)
2. Min-k Loss SGD (MKL-SGD): Selects samples with lowest loss
3. RHO-LOSS: Selects samples with highest reducible loss (current loss - irreducible loss)

The key difference between MKL-SGD and RHO-LOSS:
- MKL-SGD: Selects samples with LOWEST current loss (easy samples)
- RHO-LOSS: Selects samples with HIGHEST reducible loss (current - irreducible)
  The reducible loss represents how much the model can still learn from a sample.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_standard_sgd(model, train_loader, criterion, optimizer, device):
    """
    Standard training loop for one epoch using Uniform SGD.
    
    This is the non-robust baseline that samples uniformly at random from
    the entire training dataset in each epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data (returns (inputs, labels) or (inputs, labels, idx))
        criterion: Loss function with 'mean' reduction
        optimizer: The optimizer (e.g., SGD, Adam)
        device: The device to run on (cpu, cuda, or mps)
        
    Returns:
        epoch_loss: Average training loss
        epoch_acc: Training accuracy
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_samples = 0
    total_samples = 0

    # The default train_loader already implements uniform random sampling
    # (with shuffling enabled)
    for batch in train_loader:
        # Handle both (inputs, labels) and (inputs, labels, idx) formats
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # 1. Zero the parameter gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)

        # 3. Calculate the loss (with 'mean' reduction)
        loss = criterion(outputs, labels)

        # 4. Backward pass
        loss.backward()

        # 5. Optimize
        optimizer.step()

        # --- Statistics ---
        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_samples += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_samples / total_samples

    return epoch_loss, epoch_acc


def train_min_k_loss(model, train_loader, criterion_nored, optimizer, device, k_ratio=2.0):
    """
    Training loop for one epoch using Min-k Loss SGD (MKL-SGD).
    
    Based on the "practical batched variant" from the paper:
    "Choosing the Sample with Lowest Loss makes SGD Robust"
    
    Algorithm:
    1. Load a mini-batch of size b
    2. Calculate the per-sample loss for all b samples
    3. Select the m = b/k samples with the LOWEST loss
    4. Perform gradient update using the mean loss of these m selected samples
    
    Key insight: Noisy samples or outliers will often have a high loss, so
    selecting low-loss samples filters out noise.
    
    Args:
        model: The neural network
        train_loader: DataLoader for training data (returns (inputs, labels) or (inputs, labels, idx))
        criterion_nored: Loss function with reduction='none' (for per-sample loss)
        optimizer: The optimizer
        device: The device to run on
        k_ratio: The denominator for sample selection (e.g., 2.0 means b/2 samples are selected)
        
    Returns:
        epoch_loss: Average loss over selected samples
        epoch_acc: Training accuracy (calculated on full batch for fairness)
    """
    model.train()
    running_selected_loss = 0.0
    correct_samples = 0
    total_samples = 0
    total_selected_samples = 0

    for batch in train_loader:
        # Handle both (inputs, labels) and (inputs, labels, idx) formats
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        batch_size = inputs.size(0)

        # Determine number of samples to select (m = b/k)
        # Ensure we select at least one sample
        num_to_select = int(batch_size / k_ratio)
        if num_to_select == 0:
            num_to_select = 1

        # 1. Zero the parameter gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(inputs)

        # 3. Calculate per-sample loss (reduction='none' is required)
        per_sample_loss = criterion_nored(outputs, labels)

        # 4. Select the m = b/k samples with the LOWEST loss
        sorted_loss, sorted_indices = torch.sort(per_sample_loss)
        selected_loss = sorted_loss[:num_to_select]

        # 5. Calculate the mean loss *only* for the selected samples
        mean_selected_loss = selected_loss.mean()

        # 6. Backward pass and optimize on the selected mean loss
        mean_selected_loss.backward()
        optimizer.step()

        # --- Statistics ---
        # We track the loss of the selected samples
        running_selected_loss += selected_loss.sum().item()
        total_selected_samples += num_to_select

        # Accuracy is calculated on the entire batch for a fair comparison
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_samples += (predicted == labels).sum().item()

    # Average loss over all *selected* samples
    epoch_loss = running_selected_loss / total_selected_samples

    # Average accuracy over *all* processed samples
    epoch_acc = correct_samples / total_samples

    return epoch_loss, epoch_acc


def compute_irreducible_loss(il_model, train_dataset, criterion_nored, device, batch_size=128):
    """
    Computes the Irreducible Loss (IL) for every sample in the train_dataset
    using the pre-trained il_model.
    
    This is Phase 1 of RHO-LOSS:
    1. A holdout set D_ho is set aside
    2. The IL Model is trained only on this holdout set D_ho
    3. We perform a single forward pass of the entire training dataset through
       the (frozen) IL Model to calculate the loss for every training sample
    4. This loss, L[y_i|x_i; D_ho], is called the Irreducible Loss (IL)
       It represents the "unlearnable" part of the sample (e.g., noise)
    
    Args:
        il_model: The pre-trained Irreducible Loss model (frozen)
        train_dataset: The entire training dataset object
        criterion_nored: Loss function with reduction='none'
        device: CPU or CUDA
        batch_size: Batch size for this one-time forward pass
        
    Returns:
        A NumPy array containing the IL for each training sample, in order.
    """
    il_model.eval()  # Set IL model to evaluation mode
    all_il_losses = []

    # Use a DataLoader to process the dataset efficiently
    # IMPORTANT: shuffle=False to maintain dataset order
    il_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Computing Irreducible Losses (IL) for all training samples...")
    with torch.no_grad():  # No gradients needed
        for batch in tqdm(il_loader, desc="IL Computation"):
            # Handle both (inputs, labels) and (inputs, labels, idx) formats
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass through the IL model
            outputs = il_model(inputs)

            # Calculate per-sample loss
            loss = criterion_nored(outputs, labels)

            all_il_losses.append(loss.cpu())

    # Concatenate all batch losses into a single tensor
    il_loss_map = torch.cat(all_il_losses).numpy()

    # Add a check to make sure the map size matches the dataset
    if len(il_loss_map) != len(train_dataset):
        print(f"Warning: IL map size ({len(il_loss_map)}) does not match"
              f" dataset size ({len(train_dataset)}). Check for errors.")

    print(f"Computed IL map with shape: {il_loss_map.shape}")
    return il_loss_map


def train_il_model(il_model, holdout_loader, test_loader, device, num_epochs=50):
    """
    Trains the Irreducible Loss (IL) model on the holdout set.
    
    This is used in Phase 1 of RHO-LOSS to pre-train a model that will
    be used to compute irreducible losses for all training samples.
    
    Args:
        il_model: The model to train (will be used as IL model)
        holdout_loader: DataLoader for the holdout set (clean data)
        test_loader: DataLoader for validation during IL training
        device: CPU or CUDA
        num_epochs: Number of epochs to train the IL model
        
    Returns:
        The trained il_model
    """
    il_model.to(device)
    il_criterion = nn.CrossEntropyLoss().to(device)
    il_optimizer = optim.Adam(il_model.parameters(), lr=0.001)

    print("--- Training IL Model on Holdout Set ---")

    best_val_loss = float('inf')

    # Internal validation function
    def validate_il(model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / total, correct / total

    for epoch in range(num_epochs):
        il_model.train()
        
        # Training loop
        for batch in holdout_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            il_optimizer.zero_grad()
            outputs = il_model(inputs)
            loss = il_criterion(outputs, labels)
            loss.backward()
            il_optimizer.step()

        # Validate on the test set
        val_loss, val_acc = validate_il(il_model, test_loader, il_criterion, device)

        # Print every 10 epochs or first/last to reduce clutter
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"IL Model Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    print(f"--- IL Model Training Complete. Best Val Loss: {best_val_loss:.4f} ---")
    return il_model


def train_rho_loss(model, il_loss_map, train_loader, criterion_nored, optimizer, device, selection_ratio=0.1):
    """
    Training loop for one epoch using RHO-LOSS selection.
    
    This is Phase 2 of RHO-LOSS (the main training loop).
    
    Algorithm:
    1. At each step, a large "candidate" batch B_t (size n_B) is loaded
    2. The Target Model calculates the current training loss for each sample:
       L[y_i|x_i; D_t]
    3. The RHO-LOSS score is computed for each sample:
       RHO-LOSS[i] = Current_Loss[i] - IL[i]
    4. The algorithm selects the n_b samples with the HIGHEST RHO-LOSS scores
    5. A gradient step is performed using the mean of the Current_Loss
       (not the RHO-LOSS) of these n_b selected samples
    
    Key insight: RHO-LOSS selects samples that are:
    - Learnable (not noisy): Low IL means the sample is learnable
    - Worth learning (not outliers): High current loss means it's important
    - Not yet learnt (not redundant): High reducible loss means there's room to learn
    
    Args:
        model: The main target model to train
        il_loss_map: NumPy array of pre-computed Irreducible Losses (one per training sample)
        train_loader: DataLoader that MUST yield (inputs, labels, indices)
                     The indices are used to look up IL values
        criterion_nored: Loss function with reduction='none'
        optimizer: The optimizer
        device: CPU or CUDA
        selection_ratio: Ratio of samples to select (n_b / n_B), e.g., 0.1 for 10%
        
    Returns:
        epoch_loss: Average loss over selected samples
        epoch_acc: Training accuracy (calculated on full batch for fairness)
    """
    model.train()
    running_selected_loss = 0.0
    correct_samples = 0
    total_samples = 0
    total_selected_samples = 0

    # Convert IL map to a tensor on the correct device for fast lookup
    il_loss_map_tensor = torch.tensor(il_loss_map, dtype=torch.float32).to(device)

    for batch in train_loader:
        # RHO-LOSS requires indices, so we expect (inputs, labels, indices)
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        indices = batch[2].to(device)

        batch_size_nB = inputs.size(0)  # This is n_B (large batch size)

        # Determine number of samples to select (n_b)
        num_to_select_nb = int(batch_size_nB * selection_ratio)
        if num_to_select_nb == 0:
            num_to_select_nb = 1

        # 1. Zero the parameter gradients
        optimizer.zero_grad()

        # 2. Forward pass (Target Model)
        outputs = model(inputs)

        # 3. Calculate *current* per-sample loss: L[y|x; D_t]
        current_loss_per_sample = criterion_nored(outputs, labels)

        # 4. Look up pre-computed Irreducible Loss: L[y|x; D_ho]
        # We use the indices to get the correct IL for each sample in the batch
        batch_il_loss = il_loss_map_tensor[indices]

        # 5. Compute RHO-LOSS score: L[D_t] - L[D_ho]
        rho_loss_per_sample = current_loss_per_sample - batch_il_loss

        # 6. Select the top-nb samples with the *highest* RHO-LOSS score
        # We get the indices *within the batch* of the top samples
        _, top_batch_indices = torch.topk(rho_loss_per_sample, num_to_select_nb)

        # 7. Get the *current loss* (not RHO-LOSS) for the selected samples
        # The gradient is computed on the actual loss of the selected samples
        selected_current_loss = current_loss_per_sample[top_batch_indices]

        # 8. Calculate the mean loss for the backward pass
        mean_selected_loss = selected_current_loss.mean()

        # 9. Backward pass and optimize
        mean_selected_loss.backward()
        optimizer.step()

        # --- Statistics ---
        running_selected_loss += selected_current_loss.sum().item()
        total_selected_samples += num_to_select_nb

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_samples += (predicted == labels).sum().item()

    epoch_loss = running_selected_loss / total_selected_samples
    epoch_acc = correct_samples / total_samples

    return epoch_loss, epoch_acc

