# Robust SGD Implementations — Thesis Benchmarking Suite

A modular implementation of robust stochastic gradient descent (SGD) algorithms for benchmarking on noisy datasets. Supports **Standard SGD**, **Min-k Loss SGD (MKL-SGD)**, **RHO-LOSS**, and **HASA (History-Aware Sampling Algorithm)** on CIFAR-100, MNIST, and a cloud classification dataset.

Pypi Library Available at (https://github.com/msc35/hasa-py). Use: "pip install hasa".

## Overview

This suite compares four training algorithms on datasets with known noise:

1. **Standard SGD (Baseline)** — Uniform random sampling; no sample selection.
2. **Min-k Loss SGD (MKL-SGD)** — Selects samples with the lowest loss per batch.
3. **RHO-LOSS** — Selects samples by highest reducible loss (current loss − irreducible loss).
4. **HASA (History-Aware Sampling Algorithm)** — Bayesian-inspired selection using loss-history variance and SGLD noise.

## Datasets and Noise

| Task   | Dataset   | Noise                         | Model        |
|--------|-----------|-------------------------------|--------------|
| CIFAR-100 | CIFAR-100 | 40% symmetric label noise     | VGG-Small    |
| MNIST  | MNIST     | Gaussian input noise (std=1.5)| Simple CNN   |
| CLOUD  | Cloud     | Gaussian input noise (std=1.0)| ResNet-18    |

## Algorithms

### 1. Standard SGD (`standard` / `uniform_sgd`)

Baseline: uniform sampling, standard gradient updates. No selection or filtering.

### 2. Min-k Loss SGD (`mkl` / `mkl_sgd`)

- **Idea**: Keep the `b/k` samples with **lowest** loss in each batch; update on their mean loss.
- **Hyperparameter**: `--k_ratio` (e.g. 1.25, 1.5, 2.0).

### 3. RHO-LOSS (`rho` / `rho_loss`)

- **Idea**: Precompute an *irreducible loss* (IL) map; select samples with **highest** reducible loss (current − IL).
- **Hyperparameter**: `--selection_ratio` (e.g. 0.1–0.4).
- IL map is cached in the checkpoint directory; RHO runs need that one-time precomputation.

### 4. HASA (`hasa`)

- **Idea**: History-aware selection: track per-sample loss over a window T; select low-variance (stable) samples; inject SGLD noise.
- **Hyperparameters**: `--window_size` (T), `--selection_ratio` (k), `--noise_scale` (SGLD).

---

## Installation

**Requirements:** Python 3.9+ (tested with 3.9–3.11), pip.

```bash
git clone <repository-url>
cd Robust-SGD-Implementations-Thesis
pip install -r requirements.txt
```

**Datasets:**

- **CIFAR-100** and **MNIST**: Downloaded automatically on first run.
- **CLOUD**: Place data under `./data/` with the expected structure (e.g. `task_2_clouds/` with train/test splits). See `src/data_loader.py` for expected paths.

**Device:** The code uses CUDA if available, else MPS (Apple Silicon), else CPU. No code changes required.

---

## How to Run

### Single experiment (`main.py`)

```bash
# Standard SGD on CIFAR-100 (100 epochs)
python main.py --task cifar100 --algorithm standard --epochs 100

# MKL-SGD on MNIST
python main.py --task mnist --algorithm mkl --k_ratio 2.0 --epochs 100

# RHO-LOSS on CLOUD
python main.py --task cloud --algorithm rho --selection_ratio 0.1 --epochs 100

# HASA on CIFAR-100 (T=10, k=0.7)
python main.py --task cifar100 --algorithm hasa --window_size 10 --selection_ratio 0.7 --epochs 100
```

**Main CLI arguments:**

| Argument            | Description                                      | Default   |
|---------------------|--------------------------------------------------|-----------|
| `--task`            | Dataset: `cifar100`, `mnist`, `cloud`            | required  |
| `--algorithm`       | `standard`, `mkl`, `rho`, `hasa` (or long names)| required  |
| `--epochs`          | Number of epochs                                 | 100       |
| `--k_ratio`         | MKL-SGD: keep batch/k samples                   | 2.0       |
| `--selection_ratio` | RHO / HASA: selection ratio                      | 0.1       |
| `--window_size`     | HASA: history window T                          | 5         |
| `--noise_scale`     | HASA: SGLD noise scale                          | 0.0001    |
| `--checkpoint_dir`  | Where to save/load checkpoints                  | `./checkpoints` |
| `--plot_dir`        | Where to save plots                             | `./plots` |
| `--resume`          | Path to checkpoint to resume                    | —         |
| `--no_plot`         | Skip saving plots                               | off       |
| `--force_cpu`      | Use CPU only                                    | off       |
| `--prefer_cpu`      | Prefer CPU over MPS                             | off       |

Training resumes automatically if a checkpoint exists at the path implied by task + algorithm + hyperparameters. Best model (by validation accuracy) is saved as `*_best.pth`.

---

## Scripts (no functional code changes)

### Debug suite — quick sanity check

Runs **one config per algorithm per dataset** (12 runs) with **5 epochs** to verify the pipeline:

```bash
python run_debug_suite.py
```

Uses `main.py` with `--epochs 5`; plots can be disabled inside the script if needed. Output is printed in real time.

### Long-run experiments — full benchmark

Runs **all benchmark configs** (36 base + 24 HASA fine-tuning = 60 experiments) with **100 epochs** each, saving plots to `./plots`:

```bash
python run_long_experiments.py
```

Configs live in the script (CIFAR-100, MNIST, CLOUD × Standard, MKL, RHO, HASA with various hyperparameters). Checkpoints go to `./checkpoints`.

### Continue training — extend to 150 epochs

Extends **selected CIFAR and CLOUD configs** to **150 epochs**, resuming from existing checkpoints when present:

```bash
python continue_training.py
# Or: python continue_training.py --epochs 150 --checkpoint_dir ./checkpoints --plot_dir ./plots
```

Included configs: CIFAR-100 Normal SGD, HASA (T=10,k=0.9), HASA (T=10,k=0.8), HASA (T=15,k=0.8), RHO (sel=0.4), MKL (k=1.5); CLOUD HASA (T=15,k=0.7). If only a `*_best.pth` exists, the script copies it to the base checkpoint name so `main.py` can resume.

### Plot and compare results — no training

Loads **only checkpoints** from `./checkpoints` and builds comparison plots and tables (no training):

```bash
python plot_all_results_and_compare.py
# Or: python plot_all_results_and_compare.py --checkpoint_dir ./checkpoints --out_dir ./plots
```

Produces per-task “best algorithms” plots, per-task summary CSVs, and an overview table + bar chart. Requires existing checkpoints.

---

## Project structure

```
Robust-SGD-Implementations-Thesis/
├── main.py                      # Single-experiment entry point (argparse)
├── run_debug_suite.py           # 12 short runs (5 epochs) for debugging
├── run_long_experiments.py      # 60 full runs (100 epochs)
├── continue_training.py        # Extend selected runs to 150 epochs
├── plot_all_results_and_compare.py  # Load checkpoints, plot and compare
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py          # Datasets, noise, loaders
│   ├── models.py                # VGG_Small, MNIST_CNN, Cloud_ResNet18
│   ├── trainers.py              # Standard, MKL, RHO, HASA training loops
│   └── utils.py                 # Validation, plotting, summary tables
├── notebooks/
│   └── Benchmark_Algorithms.ipynb
├── checkpoints/                 # Saved .pth and IL data (git-ignored in practice)
├── plots/                       # Output plots and CSVs
└── data/                        # Datasets (CIFAR, MNIST, CLOUD)
```

---

## Reproducibility

- **Environment:** Use a virtual environment and install with `pip install -r requirements.txt`. Tested with Python 3.9–3.11 and PyTorch 2.x (MPS works with recent PyTorch).
- **Seeds:** Not explicitly set in the scripts; for exact reproducibility you would set `torch.manual_seed`, `numpy.random.seed`, and (if used) `random.seed` before training.
- **Order of runs:** Results depend on the order of experiments only if you reuse the same checkpoint path; each run is deterministic for a given seed and data.
- **Data:** CIFAR-100 and MNIST are standard splits; CLOUD layout must match what `data_loader.py` expects.

---

## Checkpointing and resuming

- Checkpoints are saved every epoch at `{checkpoint_dir}/{task}_{algorithm_suffix}.pth`.
- Best model (by validation accuracy) is also saved as `*_best.pth`.
- To resume, run the **same** `--task`, `--algorithm`, and hyperparameters; the script will load the checkpoint and continue from the next epoch. You can also pass `--resume /path/to/checkpoint.pth` to force a path.
- RHO-LOSS uses a precomputed IL map (stored in the checkpoint dir); it is reused across runs for the same task.

---

## Citation

If you use this code, please cite the relevant work:

- **MKL-SGD:** Shah, V., Wu, X., & Sanghavi, S. (2020). *Choosing the Sample with Lowest Loss makes SGD Robust.* AISTATS.
- **RHO-LOSS:** Mindermann, S., et al. (2022). *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt.* ICML.
- **HASA / SGLD:** Welling & Teh (2011); Mandt et al. (2017) for Bayesian interpretation of SGD noise and posterior sampling.
