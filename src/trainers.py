"""
Training functions for robust SGD algorithms.

This module implements four training algorithms:
1. Standard SGD: Uniform random sampling (baseline)
2. Min-k Loss SGD (MKL-SGD): Selects samples with lowest loss
3. RHO-LOSS: Selects samples with highest reducible loss (current loss - irreducible loss)
4. HASA (History-Aware Sampling Algorithm): Uses variance of loss history as instability cost

The key difference between MKL-SGD and RHO-LOSS:
- MKL-SGD: Selects samples with LOWEST current loss (easy samples)
- RHO-LOSS: Selects samples with HIGHEST reducible loss (current - irreducible)
  The reducible loss represents how much the model can still learn from a sample.
- HASA: Selects samples with LOWEST variance in loss history (stable samples)
  Uses Bayesian-inspired data selection treating SGD trajectory as posterior sampler.
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


def train_min_k_loss(model, train_loader, criterion_nored, optimizer, device, k_ratio=2.0, return_selected_indices=False):
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
    selected_indices_list = [] if return_selected_indices else None

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
        if return_selected_indices and len(batch) >= 3:
            idx_tensor = batch[2]
            if torch.is_tensor(idx_tensor):
                global_sel = idx_tensor[sorted_indices[:num_to_select].to(idx_tensor.device)].cpu().numpy()
            else:
                global_sel = np.array(idx_tensor)[sorted_indices[:num_to_select].cpu().numpy()]
            selected_indices_list.append(global_sel)

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

    if return_selected_indices and selected_indices_list:
        all_selected = np.unique(np.concatenate(selected_indices_list))
        return epoch_loss, epoch_acc, all_selected
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


def train_rho_loss(model, il_loss_map, train_loader, criterion_nored, optimizer, device, selection_ratio=0.1, global_to_local_map=None, return_selected_indices=False):
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
        global_to_local_map: Optional dict mapping global indices to local indices
                            (needed for datasets like CLOUD that use global indices)
        
    Returns:
        epoch_loss: Average loss over selected samples
        epoch_acc: Training accuracy (calculated on full batch for fairness)
    """
    model.train()
    running_selected_loss = 0.0
    correct_samples = 0
    total_samples = 0
    total_selected_samples = 0
    selected_indices_list = [] if return_selected_indices else None

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
        # Convert global indices to local indices if mapping is provided
        if global_to_local_map is not None:
            # Convert global indices to local indices
            local_indices = torch.tensor(
                [global_to_local_map[idx.item()] for idx in indices],
                device=device, dtype=torch.long
            )
            batch_il_loss = il_loss_map_tensor[local_indices]
        else:
            # Use indices directly (for datasets like CIFAR-100, MNIST)
            batch_il_loss = il_loss_map_tensor[indices]

        # 5. Compute RHO-LOSS score: L[D_t] - L[D_ho]
        rho_loss_per_sample = current_loss_per_sample - batch_il_loss

        # 6. Select the top-nb samples with the *highest* RHO-LOSS score
        # We get the indices *within the batch* of the top samples
        _, top_batch_indices = torch.topk(rho_loss_per_sample, num_to_select_nb)
        if return_selected_indices:
            idx_tensor = batch[2]
            if torch.is_tensor(idx_tensor):
                global_sel = idx_tensor[top_batch_indices.to(idx_tensor.device)].cpu().numpy()
            else:
                global_sel = np.array(idx_tensor)[top_batch_indices.cpu().numpy()]
            selected_indices_list.append(global_sel)

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

    if return_selected_indices and selected_indices_list:
        all_selected = np.unique(np.concatenate(selected_indices_list))
        return epoch_loss, epoch_acc, all_selected
    return epoch_loss, epoch_acc


class HASATrainer:
    """
    History-Aware Sampling Algorithm (HASA) trainer.
    
    Maintains a FIFO history buffer to track loss history for each sample.
    Uses variance of loss history as a proxy for "Instability Cost" to filter noisy data.
    Implements SGLD (Stochastic Gradient Langevin Dynamics) noise injection for Bayesian sampling.
    
    Based on Mandt et al. (2017) for Bayesian justification of noise and history.
    """
    
    def __init__(self, num_samples, window_size_T, device):
        """
        Initialize HASA trainer with history buffer.
        
        Args:
            num_samples: Total number of samples in the dataset
            window_size_T: Window size (number of epochs to track)
            device: Device for model training (history buffer uses CPU for stability)
        """
        self.num_samples = num_samples
        self.T = window_size_T
        self.device = device
        self.current_epoch = 0
        
        # History buffer: (num_samples, T) - FIFO queue
        # Store on CPU for stability (MPS can crash with large indexing operations)
        # Each row stores the loss history for one sample
        self.history_buffer = torch.zeros(num_samples, window_size_T, dtype=torch.float32, device='cpu')
        # Track how many epochs of history we have for each sample
        self.history_count = torch.zeros(num_samples, dtype=torch.long, device='cpu')
    
    def update_history(self, indices, losses):
        """
        Update the history buffer with new losses (FIFO queue).
        
        Args:
            indices: Tensor of sample indices (shape: [batch_size])
            losses: Tensor of per-sample losses (shape: [batch_size])
        """
        # Convert to CPU for indexing operations (like notebook does)
        indices_cpu = indices.cpu()
        losses_cpu = losses.detach().cpu()
        
        # Filter out invalid indices
        valid_mask = (indices_cpu >= 0) & (indices_cpu < self.num_samples)
        if not valid_mask.any():
            return
        
        valid_indices = indices_cpu[valid_mask]
        valid_losses = losses_cpu[valid_mask]
        
        # Process each unique index (handle duplicates by taking last occurrence)
        unique_indices, inverse_indices = torch.unique(valid_indices, return_inverse=True)
        # For duplicates, we want the last loss value
        for i in range(len(unique_indices)):
            mask = (inverse_indices == i)
            if mask.any():
                idx = unique_indices[i].item()
                loss_val = valid_losses[mask][-1].item()  # Take last if duplicate, convert to Python float
                
                # Shift history left (FIFO)
                if self.history_count[idx] < self.T:
                    # Buffer not full yet, just append
                    pos = self.history_count[idx].item()
                    self.history_buffer[idx, pos] = loss_val
                    self.history_count[idx] += 1
                else:
                    # Buffer full, shift left and append
                    self.history_buffer[idx, :-1] = self.history_buffer[idx, 1:].clone()
                    self.history_buffer[idx, -1] = loss_val
    
    def get_variance(self, indices):
        """
        Calculate variance of loss history for given samples.
        
        Args:
            indices: Tensor of sample indices (shape: [batch_size])
            
        Returns:
            Tensor of variances (shape: [batch_size]) on the same device as indices
        """
        # Convert to CPU for indexing (like notebook pattern)
        indices_cpu = indices.cpu()
        batch_size = len(indices_cpu)
        variances_cpu = torch.full((batch_size,), float('inf'), dtype=torch.float32, device='cpu')
        
        # Filter valid indices
        valid_mask = (indices_cpu >= 0) & (indices_cpu < self.num_samples)
        if not valid_mask.any():
            return torch.tensor(variances_cpu, device=self.device)
        
        valid_indices = indices_cpu[valid_mask]
        valid_positions = torch.where(valid_mask)[0]
        
        # Get counts for valid indices
        valid_counts = self.history_count[valid_indices]
        
        # Find indices with enough history (count >= 2)
        enough_history_mask = valid_counts >= 2
        if not enough_history_mask.any():
            return torch.tensor(variances_cpu, device=self.device)
        
        # Process indices with enough history
        process_indices = valid_indices[enough_history_mask]
        process_positions = valid_positions[enough_history_mask]
        process_counts = valid_counts[enough_history_mask]
        
        # Calculate variances
        for i, (idx, pos, count) in enumerate(zip(process_indices, process_positions, process_counts)):
            idx_val = idx.item()
            count_val = count.item()
            # Get valid history (only the filled portion)
            history = self.history_buffer[idx_val, :count_val]
            variances_cpu[pos] = torch.var(history, unbiased=False)
        
        # Return on the original device (like notebook: torch.tensor(..., device=device))
        return torch.tensor(variances_cpu, device=self.device)
    
    def increment_epoch(self):
        """Increment the current epoch counter."""
        self.current_epoch += 1
    
    def is_warmup_phase(self):
        """Check if we're still in warm-up phase (epoch < T)."""
        return self.current_epoch < self.T


def _inject_sgld_noise(model, learning_rate, noise_scale=None):
    """
    Inject SGLD (Stochastic Gradient Langevin Dynamics) noise to model parameters.
    
    Following Welling & Teh (2011) rule: noise_std = sqrt(2 * learning_rate)
    unless noise_scale is explicitly provided.
    
    This ensures we are sampling from the posterior distribution, as justified
    by Mandt et al. (2017) for Bayesian interpretation of SGD.
    
    Args:
        model: The neural network model
        learning_rate: Current learning rate
        noise_scale: Optional custom noise scale. If None, uses sqrt(2 * lr)
    """
    # Get device from first model parameter
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # Model has no parameters, skip noise injection
        return
    
    if noise_scale is None:
        noise_std = torch.sqrt(torch.tensor(2.0 * learning_rate, device=device, dtype=torch.float32))
    else:
        # Convert noise_scale to tensor on correct device if it's a scalar
        if isinstance(noise_scale, (int, float)):
            noise_std = torch.tensor(noise_scale, device=device, dtype=torch.float32)
        else:
            noise_std = noise_scale.to(device) if hasattr(noise_scale, 'to') else noise_scale
    
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Generate noise and add
                noise = torch.randn_like(param, dtype=param.dtype) * noise_std
                param.data.add_(noise)


def train_hasa(model, train_loader, criterion_nored, optimizer, device, 
                hasa_trainer=None, window_size_T=5, k_ratio=0.6, noise_scale=None, 
                train_dataset=None, current_epoch=0, return_selected_indices=False):
    """
    Training loop for one epoch using HASA (History-Aware Sampling Algorithm).
    
    HASA is a Bayesian-inspired data selection method that uses the variance of loss
    history as a proxy for "Instability Cost" to filter noisy data. It treats the
    SGD trajectory as a posterior sampler (Mandt et al., 2017).
    
    Algorithm:
    Phase A (Warm-Up, Epoch < T):
    1. Perform standard SGD updates on all data
    2. Inject SGLD noise: params = params - lr * grad + Normal(0, sqrt(2 * lr))
    3. Record loss into history buffer
    
    Phase B (Selection Phase, Epoch >= T):
    1. Calculate Instability: Variance of loss history over window T
    2. Hard Selection: Sort by variance, keep top k% (lowest variance = most stable)
    3. Robust Update: Compute loss only on selected samples, update parameters
    4. Re-inject SGLD noise: params = params + Normal(0, sqrt(2 * lr))
    5. Update history buffer with current losses
    
    Args:
        model: The neural network model
        train_loader: DataLoader that MUST yield (inputs, labels, indices)
        criterion_nored: Loss function with reduction='none' (for per-sample loss)
        optimizer: The optimizer (must have learning rate accessible)
        device: The device to run on
        hasa_trainer: HASATrainer instance (created if None)
        window_size_T: Window size for history buffer (default: 5)
        k_ratio: Selection ratio - percentage of data to keep (default: 0.6)
        noise_scale: Optional custom noise scale. If None, uses sqrt(2 * lr)
        train_dataset: Training dataset (required if hasa_trainer is None)
        current_epoch: Current epoch number (for warm-up phase detection)
        
    Returns:
        epoch_loss: Average loss over selected samples (or all samples in warm-up)
        epoch_acc: Training accuracy (calculated on full batch for fairness)
        hasa_trainer: The HASATrainer instance (for state persistence)
    """
    model.train()
    
    # Initialize or get HASATrainer
    if hasa_trainer is None:
        if train_dataset is None:
            raise ValueError("train_dataset must be provided if hasa_trainer is None")
        num_samples = len(train_dataset)
        hasa_trainer = HASATrainer(num_samples, window_size_T, device)
        hasa_trainer.current_epoch = current_epoch
    
    # Get learning rate from optimizer
    try:
        learning_rate = optimizer.param_groups[0]['lr']
    except (IndexError, KeyError) as e:
        raise ValueError(f"Could not get learning rate from optimizer: {e}")
    
    # Statistics
    running_selected_loss = 0.0
    correct_samples = 0
    total_samples = 0
    total_selected_samples = 0
    selected_indices_list = [] if return_selected_indices else None
    
    # Check if we're in warm-up phase
    is_warmup = hasa_trainer.is_warmup_phase()
    
    for batch_idx, batch in enumerate(train_loader):
        # HASA requires indices, so we expect (inputs, labels, indices)
        if len(batch) < 3:
            raise ValueError(f"HASA requires DataLoader to return (inputs, labels, indices), got {len(batch)} items")
        
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        indices = batch[2].to(device)
        
        # Skip empty batches
        if inputs.size(0) == 0:
            continue
        
        batch_size = inputs.size(0)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate per-sample loss
        per_sample_loss = criterion_nored(outputs, labels)
        
        # MPS memory management (periodic cache clearing)
        if device.type == 'mps' and batch_idx % 50 == 0:
            torch.mps.empty_cache()
        
        if is_warmup:
            # Phase A: Warm-Up - use all samples
            # Calculate mean loss for backward pass
            mean_loss = per_sample_loss.mean()
            mean_loss.backward()
            optimizer.step()
            
            # Inject SGLD noise after gradient update
            _inject_sgld_noise(model, learning_rate, noise_scale)
            
            # Update history buffer
            hasa_trainer.update_history(indices, per_sample_loss)
            
            # Statistics
            running_selected_loss += per_sample_loss.sum().item()
            total_selected_samples += batch_size
            if return_selected_indices:
                selected_indices_list.append(indices.cpu().numpy())
            
        else:
            # Phase B: Selection Phase
            # 1. Calculate Instability (variance of loss history)
            variances = hasa_trainer.get_variance(indices)
            
            # 2. Hard Selection: Sort by variance (low variance = good, high variance = bad)
            # We want to keep samples with LOWEST variance (most stable)
            num_to_select = int(batch_size * k_ratio)
            if num_to_select == 0:
                num_to_select = 1
            if num_to_select > batch_size:
                num_to_select = batch_size
            
            # Get indices of samples with lowest variance
            _, selected_batch_indices = torch.topk(-variances, num_to_select)  # Negative for ascending order
            
            # 3. Robust Update: Calculate loss only on selected samples
            selected_loss = per_sample_loss[selected_batch_indices]
            mean_selected_loss = selected_loss.mean()
            
            # Backward pass and optimize
            mean_selected_loss.backward()
            optimizer.step()
            
            # Re-inject SGLD noise
            _inject_sgld_noise(model, learning_rate, noise_scale)
            
            # 4. Update history buffer with current losses
            hasa_trainer.update_history(indices, per_sample_loss)
            
            # Statistics
            running_selected_loss += selected_loss.sum().item()
            total_selected_samples += num_to_select
            if return_selected_indices:
                global_sel = indices[selected_batch_indices].cpu().numpy()
                selected_indices_list.append(global_sel)
        
        # Accuracy is calculated on the entire batch for fairness
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_samples += (predicted == labels).sum().item()
    
    # Increment epoch counter
    hasa_trainer.increment_epoch()
    
    # Calculate epoch statistics
    if total_selected_samples > 0:
        epoch_loss = running_selected_loss / total_selected_samples
    else:
        epoch_loss = 0.0
    
    epoch_acc = correct_samples / total_samples
    
    if return_selected_indices and selected_indices_list:
        all_selected = np.unique(np.concatenate(selected_indices_list))
        return epoch_loss, epoch_acc, hasa_trainer, all_selected
    return epoch_loss, epoch_acc, hasa_trainer

