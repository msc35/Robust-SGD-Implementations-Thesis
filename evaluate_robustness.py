#!/usr/bin/env python3
"""
Evaluate robustness of a trained model from a saved checkpoint.

1. Calibration (ECE): Expected Calibration Error on the test set.
2. Input Stability (Flip Rate): Fraction of predictions that change under
   small Gaussian input noise (std=0.05).
3. Noise Detection (CIFAR-100 only): Precision/Recall of "high loss" samples
   vs ground-truth noisy label indices (requires same noise seed or noisy_indices.npy).
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.models import MNIST_CNN, VGG_Small, Cloud_ResNet18
from src.data_loader import NoisyIndexedDataset, AddGaussianNoise, CloudMergedDataset, ApplyTransformSubset
from torchvision import transforms, datasets
from torch.utils.data import Subset

# Reuse main's data paths and setup (avoids duplication)
DATA_ROOT = './data'
CLOUD_ROOT = './data/task_2_clouds'
VALID_BATCH_SIZE = 256
MAIN_BATCH_SIZE = 32
NOISE_SEED = 42
CIFAR_NOISE_RATE = 0.4
ECE_N_BINS = 15
FLIP_NOISE_STD = 0.05

# Predefined list for --run_all: (dataset, checkpoint_basename, display_name)
EVAL_RUN_ALL = [
    # CIFAR-100 (40% Label Noise)
    ("cifar100", "cifar100_mkl_sgd_k1.5", "CIFAR-100 MKL (k=1.5)"),
    ("cifar100", "cifar100_rho_loss_sel0.4", "CIFAR-100 RHO (sel=0.4)"),
    ("cifar100", "cifar100_hasa_T15_k0.8_ns0.0001", "CIFAR-100 HASA (T=15, k=0.8)"),
    ("cifar100", "cifar100_mkl_sgd_k1.25", "CIFAR-100 MKL (k=1.25)"),
    ("cifar100", "cifar100_uniform_sgd", "CIFAR-100 Standard SGD"),
    ("cifar100", "cifar100_hasa_T10_k0.8_ns0.0001", "CIFAR-100 HASA (T=10, k=0.8)"),
    ("cifar100", "cifar100_hasa_T10_k0.9_ns0.0001", "CIFAR-100 HASA (T=10, k=0.9)"),
    ("cifar100", "cifar100_hasa_T5_k0.9_ns0.0001", "CIFAR-100 HASA (T=5, k=0.9)"),
    # CLOUD (Input Noise std=1.0)
    ("cloud", "cloud_mkl_sgd_k1.25", "CLOUD MKL (k=1.25)"),
    ("cloud", "cloud_hasa_T15_k0.7_ns0.0001", "CLOUD HASA (T=15, k=0.7)"),
    ("cloud", "cloud_hasa_T10_k0.7_ns0.0001", "CLOUD HASA (T=10, k=0.7)"),
    ("cloud", "cloud_hasa_T10_k0.9_ns0.0001", "CLOUD HASA (T=10, k=0.9)"),
    ("cloud", "cloud_mkl_sgd_k2.0", "CLOUD MKL (k=2.0)"),
    ("cloud", "cloud_uniform_sgd", "CLOUD Standard SGD"),
    ("cloud", "cloud_hasa_T10_k0.8_ns0.0001", "CLOUD HASA (T=10, k=0.8)"),
    ("cloud", "cloud_hasa_T5_k0.8_ns0.0001", "CLOUD HASA (T=5, k=0.8)"),
    # MNIST (Input Noise std=1.5)
    ("mnist", "mnist_uniform_sgd", "MNIST Standard SGD"),
    ("mnist", "mnist_hasa_T10_k0.5_ns0.0001", "MNIST HASA (T=10, k=0.5)"),
    ("mnist", "mnist_hasa_T10_k0.6_ns0.0001", "MNIST HASA (T=10, k=0.6)"),
    ("mnist", "mnist_hasa_T10_k0.8_ns0.0001", "MNIST HASA (T=10, k=0.8)"),
    ("mnist", "mnist_hasa_T10_k0.7_ns0.0001", "MNIST HASA (T=10, k=0.7)"),
    ("mnist", "mnist_hasa_T10_k0.9_ns0.0001", "MNIST HASA (T=10, k=0.9)"),
    ("mnist", "mnist_hasa_T5_k0.9_ns0.0001", "MNIST HASA (T=5, k=0.9)"),
]


def setup_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            _ = torch.randn(1, device=torch.device("mps"))
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def get_model_and_data(dataset_name, device):
    """Return (model, data_config) for the given dataset. Model is on device and in eval mode."""
    if dataset_name == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_ds = NoisyIndexedDataset(
            'CIFAR100', DATA_ROOT, train=True, transform=transform_test,
            noise_type='symmetric', noise_rate=CIFAR_NOISE_RATE, random_seed=NOISE_SEED
        )
        test_ds = NoisyIndexedDataset(
            'CIFAR100', DATA_ROOT, train=False, transform=transform_test, noise_type='none'
        )
        model = VGG_Small(num_classes=100).to(device)
        num_classes = 100
    elif dataset_name == 'mnist':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = NoisyIndexedDataset(
            'MNIST', DATA_ROOT, train=True, transform=transform_test, noise_type='none'
        )
        test_ds = NoisyIndexedDataset(
            'MNIST', DATA_ROOT, train=False, transform=transform_test, noise_type='none'
        )
        model = MNIST_CNN().to(device)
        num_classes = 10
    elif dataset_name == 'cloud':
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        full_ds = CloudMergedDataset(CLOUD_ROOT, transform=None)
        total = len(full_ds)
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(total, generator=gen).tolist()
        train_size = int(0.50 * total)
        holdout_size = int(0.10 * total)
        train_indices = idx[:train_size]
        test_indices = idx[train_size + holdout_size:]
        train_ds = ApplyTransformSubset(full_ds, train_indices, transform=transform_test)
        test_ds = ApplyTransformSubset(full_ds, test_indices, transform=transform_test)
        num_classes = len(full_ds.classes)
        model = Cloud_ResNet18(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(test_ds, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    train_loader_no_shuffle = DataLoader(
        train_ds, batch_size=MAIN_BATCH_SIZE, shuffle=False, num_workers=0
    )
    criterion_nored = nn.CrossEntropyLoss(reduction='none').to(device)

    data_config = {
        'test_loader': test_loader,
        'train_loader_no_shuffle': train_loader_no_shuffle,
        'train_dataset': train_ds,
        'criterion_nored': criterion_nored,
        'num_classes': num_classes,
    }
    return model, data_config


def load_checkpoint(model, checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return ck


def compute_ece(model, test_loader, device, n_bins=ECE_N_BINS):
    """Expected Calibration Error: bin by confidence, compare accuracy vs confidence per bin."""
    all_confs = []
    all_accs = []  # 1 if correct, 0 otherwise (per sample)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            correct = (pred == y).float()
            all_confs.append(conf.cpu().numpy())
            all_accs.append(correct.cpu().numpy())
    confs = np.concatenate(all_confs)
    accs = np.concatenate(all_accs)
    n = len(confs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= low) & (confs < high) if i < n_bins - 1 else (confs >= low) & (confs <= high)
        if mask.sum() == 0:
            continue
        acc_bin = accs[mask].mean()
        conf_bin = confs[mask].mean()
        ece += np.abs(acc_bin - conf_bin) * mask.sum() / n
    return float(ece)


def compute_flip_rate(model, test_loader, device, noise_std=FLIP_NOISE_STD):
    """Fraction of test predictions that change when adding Gaussian noise to inputs."""
    preds_clean = []
    preds_noisy = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            _ = batch[1]
            pred_clean = model(x).argmax(1)
            preds_clean.append(pred_clean.cpu())
            x_noisy = x + torch.randn_like(x, device=device) * noise_std
            pred_noisy = model(x_noisy).argmax(1)
            preds_noisy.append(pred_noisy.cpu())
    p_clean = torch.cat(preds_clean)
    p_noisy = torch.cat(preds_noisy)
    flips = (p_clean != p_noisy).float().sum().item()
    total = p_clean.numel()
    return flips / total if total else 0.0


def compute_noise_detection_pr(model, train_loader_no_shuffle, criterion_nored, device, noisy_gt_set, noise_rate=CIFAR_NOISE_RATE):
    """
    Assume high-loss samples are "predicted noisy". Compare to ground-truth noisy indices.
    Returns precision, recall. noisy_gt_set: set of indices that are truly noisy (e.g. CIFAR-100).
    """
    index_to_loss = {}
    model.eval()
    with torch.no_grad():
        for batch in train_loader_no_shuffle:
            x = batch[0].to(device)
            y = batch[1].to(device)
            idx = batch[2] if len(batch) >= 3 else torch.arange(x.size(0), device=x.device)
            if torch.is_tensor(idx):
                idx = idx.cpu().numpy()
            logits = model(x)
            loss = criterion_nored(logits, y)
            for j, i in enumerate(idx):
                index_to_loss[int(i)] = float(loss[j].cpu().item())
    indices = np.array(list(index_to_loss.keys()))
    losses = np.array([index_to_loss[i] for i in indices])
    n = len(indices)
    n_pred_noisy = max(1, int(n * noise_rate))
    order = np.argsort(-losses)[:n_pred_noisy]
    top_indices = set(indices[order].tolist())
    tp = len(noisy_gt_set & top_indices)
    fp = len(top_indices - noisy_gt_set)
    fn = len(noisy_gt_set - top_indices)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def run_one_evaluation(checkpoint_path, dataset, args, device):
    """Run ECE, flip rate, and (for CIFAR-100) noise detection. Returns (ece, flip_rate, precision, recall)."""
    model, data_config = get_model_and_data(dataset, device)
    load_checkpoint(model, checkpoint_path, device)
    test_loader = data_config['test_loader']
    train_loader_ns = data_config['train_loader_no_shuffle']
    train_dataset = data_config['train_dataset']
    criterion_nored = data_config['criterion_nored']

    ece = compute_ece(model, test_loader, device, n_bins=args.n_bins)
    flip_rate = compute_flip_rate(model, test_loader, device, noise_std=args.noise_std)
    precision, recall = None, None
    if not args.no_noise_detection and dataset == 'cifar100':
        if args.noisy_indices_path and os.path.isfile(args.noisy_indices_path):
            noisy_gt_set = set(np.load(args.noisy_indices_path).tolist())
        elif hasattr(train_dataset, 'noise_mask'):
            noisy_gt_set = set(np.where(train_dataset.noise_mask)[0])
        else:
            noisy_gt_set = None
        if noisy_gt_set is not None:
            precision, recall = compute_noise_detection_pr(
                model, train_loader_ns, criterion_nored, device, noisy_gt_set, noise_rate=CIFAR_NOISE_RATE
            )
    return ece, flip_rate, precision, recall


def _short_name(display_name):
    """Strip dataset prefix for x-axis labels (e.g. 'CIFAR-100 MKL (k=1.5)' -> 'MKL (k=1.5)')."""
    for prefix in ("CIFAR-100 ", "CLOUD ", "MNIST "):
        if display_name.startswith(prefix):
            return display_name[len(prefix):]
    return display_name


def plot_and_save_results(results, out_dir):
    """Save results to CSV and generate KPI bar charts (per-dataset)."""
    os.makedirs(out_dir, exist_ok=True)
    # Build DataFrame: Name, Dataset, ECE, FlipRate, Precision, Recall
    rows = []
    for name, dataset, ece, fr, prec, rec in results:
        rows.append({
            "Name": name,
            "Dataset": dataset,
            "ECE": ece if ece is not None else float("nan"),
            "FlipRate": fr if fr is not None else float("nan"),
            "Precision": prec if prec is not None else float("nan"),
            "Recall": rec if rec is not None else float("nan"),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "robustness_evaluation.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"CSV saved: {csv_path}")

    # Filter to rows with at least one numeric KPI for plotting
    plot_df = df.dropna(subset=["ECE"], how="all")
    if plot_df.empty:
        return

    datasets = ["cifar100", "cloud", "mnist"]
    for ds in datasets:
        sub = plot_df[plot_df["Dataset"] == ds]
        if sub.empty:
            continue
        labels = [_short_name(n) for n in sub["Name"]]
        x = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Robustness KPIs — {ds.upper()}", fontsize=14)

        # 1. ECE (lower is better)
        ax1 = axes[0]
        vals = sub["ECE"].fillna(0).values
        colors = np.where(np.isnan(sub["ECE"].values), "lightgray", "steelblue")
        bars = ax1.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("ECE")
        ax1.set_title("Expected Calibration Error (lower is better)")
        ax1.grid(axis="y", alpha=0.3)

        # 2. Flip Rate (lower is better)
        ax2 = axes[1]
        vals = sub["FlipRate"].fillna(0).values
        colors = np.where(np.isnan(sub["FlipRate"].values), "lightgray", "coral")
        ax2.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Flip Rate")
        ax2.set_title("Input Stability — Flip Rate (lower is better)")
        ax2.grid(axis="y", alpha=0.3)

        # 3. Precision & Recall (higher is better; CIFAR-100 only has values)
        ax3 = axes[2]
        prec = sub["Precision"].fillna(0).values
        rec = sub["Recall"].fillna(0).values
        has_pr = ~(np.isnan(sub["Precision"].values) & np.isnan(sub["Recall"].values))
        w = width / 2
        ax3.bar(x - w, prec, width=w, label="Precision", color="seagreen", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax3.bar(x + w / 2, rec, width=w, label="Recall", color="mediumseagreen", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Score")
        ax3.set_title("Noise Detection — Precision & Recall (CIFAR-100 only)")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"robustness_kpis_{ds}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {fig_path}")

    # One overview figure: all models, ECE and Flip Rate (color by dataset)
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    colors = {"cifar100": "steelblue", "cloud": "coral", "mnist": "seagreen"}
    all_labels = [_short_name(n) for n in plot_df["Name"]]
    x_all = np.arange(len(all_labels))
    for i, (kpi, ylabel, title) in enumerate([
        ("ECE", "ECE", "Expected Calibration Error (lower is better)"),
        ("FlipRate", "Flip Rate", "Input Stability — Flip Rate (lower is better)"),
    ]):
        ax = axes2[i]
        vals = np.nan_to_num(plot_df[kpi].values, nan=0.0)
        bar_colors = [colors.get(d, "gray") for d in plot_df["Dataset"]]
        ax.bar(x_all, vals, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.set_xticks(x_all)
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(handles=[Patch(facecolor=colors[d], label=d.upper()) for d in datasets if d in colors], loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    overview_path = os.path.join(out_dir, "robustness_kpis_overview.png")
    fig2.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Overview plot saved: {overview_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate robustness: ECE, Flip Rate, Noise Detection (CIFAR-100).',
        epilog='Single run: python evaluate_robustness.py --checkpoint_path ./checkpoints/cifar100_hasa_T10_k0.9_ns0.0001.pth --dataset cifar100\n'
               'Run all predefined: python evaluate_robustness.py --run_all'
    )
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to .pth checkpoint (required if not --run_all)')
    parser.add_argument('--dataset', type=str, default=None, choices=['cifar100', 'mnist', 'cloud'],
                        help='Dataset (required if not --run_all)')
    parser.add_argument('--run_all', action='store_true',
                        help='Evaluate all predefined checkpoint/dataset pairs (see EVAL_RUN_ALL)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory for --run_all (default: ./checkpoints)')
    parser.add_argument('--out_dir', type=str, default='./evaluation_results',
                        help='Output directory for CSV and plots when using --run_all (default: ./evaluation_results)')
    parser.add_argument('--noisy_indices_path', type=str, default=None,
                        help='Optional: path to noisy_indices.npy (CIFAR-100). If not set, re-generate with seed 42.')
    parser.add_argument('--n_bins', type=int, default=ECE_N_BINS, help='Number of bins for ECE')
    parser.add_argument('--noise_std', type=float, default=FLIP_NOISE_STD, help='Gaussian std for flip-rate test')
    parser.add_argument('--no_noise_detection', action='store_true',
                        help='Skip noise detection (e.g. for MNIST/CLOUD)')
    args = parser.parse_args()

    if args.run_all:
        device = setup_device()
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        print(f"Device: {device}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Running {len(EVAL_RUN_ALL)} evaluations.\n")
        results = []
        for i, (dataset, basename, display_name) in enumerate(EVAL_RUN_ALL, 1):
            path = os.path.join(checkpoint_dir, basename + ".pth")
            if not os.path.isfile(path):
                path = os.path.join(checkpoint_dir, basename + "_best.pth")
            if not os.path.isfile(path):
                print(f"[{i}/{len(EVAL_RUN_ALL)}] SKIP (not found): {display_name}\n")
                results.append((display_name, dataset, None, None, None, None))
                continue
            print(f"[{i}/{len(EVAL_RUN_ALL)}] {display_name}")
            try:
                ece, flip_rate, precision, recall = run_one_evaluation(path, dataset, args, device)
                results.append((display_name, dataset, ece, flip_rate, precision, recall))
                print(f"   ECE: {ece:.4f}  FlipRate: {flip_rate:.4f}" + (f"  Precision: {precision:.4f}  Recall: {recall:.4f}" if precision is not None else ""))
                print()
            except Exception as e:
                print(f"   ERROR: {e}\n")
                results.append((display_name, dataset, None, None, None, None))
        # Summary table
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Name':<45} {'ECE':>8} {'FlipRate':>10} {'Prec':>8} {'Recall':>8}")
        print("-" * 80)
        for name, ds, ece, fr, prec, rec in results:
            ece_s = f"{ece:.4f}" if ece is not None else "N/A"
            fr_s = f"{fr:.4f}" if fr is not None else "N/A"
            prec_s = f"{prec:.4f}" if prec is not None else "N/A"
            rec_s = f"{rec:.4f}" if rec is not None else "N/A"
            print(f"{name:<45} {ece_s:>8} {fr_s:>10} {prec_s:>8} {rec_s:>8}")
        # Save CSV and KPI plots
        out_dir = os.path.abspath(args.out_dir)
        plot_and_save_results(results, out_dir)
        print(f"\nResults and plots saved to: {out_dir}")
        return

    if not args.checkpoint_path or not args.dataset:
        parser.error("--checkpoint_path and --dataset are required unless --run_all is set.")

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = setup_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}\n")

    ece, flip_rate, precision, recall = run_one_evaluation(args.checkpoint_path, args.dataset, args, device)

    print("=" * 60)
    print("1. Calibration (ECE)")
    print("=" * 60)
    print(f"   Expected Calibration Error: {ece:.4f}")
    print("   (Lower is better; well-calibrated models have ECE ≈ 0.)\n")

    print("2. Input Stability (Flip Rate)")
    print("=" * 60)
    print(f"   Fraction of predictions that changed under Gaussian noise (std={args.noise_std}): {flip_rate:.4f}")
    print("   (Lower is better; more stable models flip less.)\n")

    if precision is not None:
        print("3. Noise Detection (Reconstruction)")
        print("=" * 60)
        print(f"   Ground truth: CIFAR-100 label noise ({CIFAR_NOISE_RATE*100:.0f}%). High-loss samples = predicted noisy.")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print("   (Higher precision/recall = better at identifying noisy labels.)\n")
    else:
        if args.no_noise_detection:
            print("3. Noise Detection: skipped (--no_noise_detection).\n")
        else:
            print("3. Noise Detection: skipped (only supported for CIFAR-100 with label noise).\n")


if __name__ == '__main__':
    main()
