# Robust SGD Implementation - Benchmarking Suite

This repository contains a clean, modular implementation of robust stochastic gradient descent (SGD) algorithms for benchmarking on noisy datasets. The code has been refactored from a Jupyter notebook into a professional Python package structure.

## Overview

This benchmarking suite compares three training algorithms on datasets with known noise:

1. **Standard SGD (Baseline)**: Uniform random sampling from the training set
2. **Min-k Loss SGD (MKL-SGD)**: Selects samples with the lowest loss
3. **RHO-LOSS**: Selects samples with the highest reducible loss (current loss - irreducible loss)

## Datasets and Noise Types

The benchmark includes three tasks with different noise characteristics:

### Task 1: CIFAR-100 (Label Noise)
- **Dataset**: CIFAR-100
- **Noise Type**: 40% symmetric label noise
- **Model**: VGG-Small CNN
- **Architecture**: 3 conv layers + 2 FC layers

### Task 2: MNIST (Input Noise)
- **Dataset**: MNIST
- **Noise Type**: Gaussian input noise with std=1.5
- **Model**: Simple CNN (2 conv layers + 2 FC layers)
- **Architecture**: 32→64 channels, 128 hidden units

### Task 3: CLOUD (Input Noise)
- **Dataset**: Cloud classification dataset
- **Noise Type**: Gaussian input noise with std=1.0
- **Model**: ResNet-18 (ImageNet pretrained)
- **Architecture**: Standard ResNet-18 with custom classifier

## Algorithms

### 1. Standard SGD (`uniform_sgd`)

The baseline algorithm that samples uniformly at random from the training dataset. This is the standard, non-robust training procedure.

**Key Features:**
- Uniform random sampling (handled by DataLoader shuffling)
- Standard gradient descent update
- No sample selection or filtering

### 2. Min-k Loss SGD (`mkl_sgd`)

Based on the paper: **"Choosing the Sample with Lowest Loss makes SGD Robust"**

**Algorithm:**
1. Load a mini-batch of size `b`
2. Calculate per-sample loss for all `b` samples
3. Select the `m = b/k` samples with the **lowest** loss
4. Perform gradient update using the mean loss of these `m` selected samples

**Key Insight**: Noisy samples or outliers will often have a high loss, so selecting low-loss samples filters out noise.

**Hyperparameters:**
- `k_ratio`: Denominator for sample selection (default: 2.0, meaning b/2 samples are selected)
- Common values: 1.25, 1.5, 2.0

### 3. RHO-LOSS (`rho_loss`)

Based on the paper: **"Prioritized Training on Points that are learnable, Worth Learning, and Not Yet Learnt"**

**Algorithm (Two-Phase):**

**Phase 1: Pre-computation (Done Once)**
1. A holdout set `D_ho` is set aside (10% of data)
2. An **Irreducible Loss (IL) Model** is trained only on this holdout set
3. Perform a single forward pass of the entire training dataset through the frozen IL Model
4. Store the loss for each training sample as the **Irreducible Loss (IL)**
   - This represents the "unlearnable" part of the sample (e.g., noise)

**Phase 2: Main Training Loop**
1. At each step, load a large candidate batch `B_t` (size `n_B`)
2. Calculate current training loss: `L[y_i|x_i; D_t]`
3. Compute **RHO-LOSS score**: `RHO-LOSS[i] = Current_Loss[i] - IL[i]`
4. Select the `n_b` samples with the **highest** RHO-LOSS scores
5. Perform gradient update using the mean of the **current loss** (not RHO-LOSS) of selected samples

**Key Insight**: RHO-LOSS selects samples that are:
- **Learnable** (not noisy): Low IL means the sample is learnable
- **Worth learning** (not outliers): High current loss means it's important
- **Not yet learnt** (not redundant): High reducible loss means there's room to learn

**Hyperparameters:**
- `selection_ratio`: Ratio of samples to select (n_b / n_B), e.g., 0.1 for 10%
- Common values: 0.1, 0.2, 0.3, 0.4

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Robust-SGD-Implementation-Thesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure datasets are available:
- **CIFAR-100**: Will be automatically downloaded on first run
- **MNIST**: Will be automatically downloaded on first run
- **CLOUD**: Should be placed in `./data/task_2_clouds/` with `clouds_train/` and `clouds_test/` subdirectories

## Usage

### Basic Usage

Run a single experiment:

```bash
# Standard SGD on CIFAR-100
python main.py --task cifar100 --algorithm uniform_sgd --epochs 100

# MKL-SGD on MNIST with k_ratio=2.0
python main.py --task mnist --algorithm mkl_sgd --k_ratio 2.0 --epochs 100

# RHO-LOSS on CLOUD with selection_ratio=0.1
python main.py --task cloud --algorithm rho_loss --selection_ratio 0.1 --epochs 100
```

### Command-Line Arguments

```
--task: Dataset to use (cifar100, mnist, cloud)
--algorithm: Training algorithm (uniform_sgd, mkl_sgd, rho_loss)
--epochs: Number of training epochs (default: 100)
--k_ratio: k_ratio for MKL-SGD (default: 2.0)
--selection_ratio: Selection ratio for RHO-LOSS (default: 0.1)
--checkpoint_dir: Directory for checkpoints (default: ./checkpoints)
--resume: Path to checkpoint to resume from (optional)
```

### Example: Running Full Benchmark Suite

To reproduce the full benchmark suite from the notebook:

```bash
# CIFAR-100 experiments
python main.py --task cifar100 --algorithm uniform_sgd --epochs 100
python main.py --task cifar100 --algorithm mkl_sgd --k_ratio 2.0 --epochs 100
python main.py --task cifar100 --algorithm mkl_sgd --k_ratio 1.5 --epochs 100
python main.py --task cifar100 --algorithm mkl_sgd --k_ratio 1.25 --epochs 100
python main.py --task cifar100 --algorithm rho_loss --selection_ratio 0.2 --epochs 100
python main.py --task cifar100 --algorithm rho_loss --selection_ratio 0.3 --epochs 100
python main.py --task cifar100 --algorithm rho_loss --selection_ratio 0.4 --epochs 100

# MNIST experiments (similar pattern)
# CLOUD experiments (similar pattern)
```

## Project Structure

```
Robust-SGD-Implementation-Thesis/
├── src/
│   ├── data_loader.py      # Dataset classes and noise injection
│   ├── models.py            # Neural network architectures
│   ├── trainers.py          # Training algorithms (SGD, MKL-SGD, RHO-LOSS)
│   └── utils.py             # Validation and plotting utilities
├── main.py                  # Main script with argparse
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── notebooks/
│   └── Benchmark_Algorithms.ipynb  # Original notebook (unchanged)
├── data/                    # Dataset storage
```

## Key Differences: MKL-SGD vs RHO-LOSS

### MKL-SGD
- **Selection Criterion**: Samples with **lowest** current loss
- **Rationale**: Easy samples are more reliable, noisy samples have high loss
- **Computation**: Single forward pass, simple sorting
- **Hyperparameter**: `k_ratio` (how many samples to keep: b/k)

### RHO-LOSS
- **Selection Criterion**: Samples with **highest** reducible loss (current - irreducible)
- **Rationale**: Balances learnability (low IL) with importance (high current loss)
- **Computation**: Requires pre-computation of IL map (one-time cost)
- **Hyperparameter**: `selection_ratio` (percentage of batch to keep)

## Hyperparameters

All hyperparameters match the original notebook implementation:

- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 32 (training), 256 (validation)
- **Epochs**: 100 (default)
- **CIFAR-100 Label Noise**: 40% symmetric
- **MNIST Input Noise**: Gaussian std=1.5
- **CLOUD Input Noise**: Gaussian std=1.0

## Checkpointing

The code supports automatic checkpointing:
- Checkpoints are saved after each epoch
- Best model (highest validation accuracy) is saved separately
- Training can be resumed using `--resume <checkpoint_path>`
- IL maps for RHO-LOSS are cached in the checkpoint directory

## Results Visualization

The script automatically generates:
1. **Training curves**: Loss and accuracy plots
2. **Summary table**: Best validation accuracy, final metrics

For more detailed analysis, use the plotting functions in `src/utils.py`:
- `plot_results_custom()`: Plot training curves
- `create_summary_table()`: Generate comparison tables

## Notes

- The original notebook (`notebooks/Benchmark_Algorithms.ipynb`) is preserved and unchanged
- All hyperparameters and model architectures match the notebook exactly
- The code cannot be re-run without the source data, but the logic is preserved exactly as implemented
- RHO-LOSS requires pre-computation of the Irreducible Loss map (done automatically on first run)

## Citation

If you use this code, please cite the original papers:

MKL-SGD: Shah, V., Wu, X., & Sanghavi, S. (2020). Choosing the Sample with Lowest Loss makes SGD Robust. Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS). 

RHO-LOSS: Mindermann, S., et al. (2022). Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt. Proceedings of the 39th International Conference on Machine Learning (ICML). "

