#!/usr/bin/env python3
"""
Compute ECE and Flip Rate for the 6 Phase 1 diagnostic experiments + Standard SGD (CIFAR-100).
Saves results to diagnostic_phase1/diagnostic_ece_flip.csv.

Uses same CIFAR-100 test set and same ECE/Flip definitions as evaluate_robustness.py.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models import VGG_Small
from src.data_loader import NoisyIndexedDataset

DATA_ROOT = "./data"
DIAGNOSTIC_DIR = "./diagnostic_phase1"
CHECKPOINTS_DIR = "./checkpoints"
VALID_BATCH_SIZE = 256
ECE_N_BINS = 15
FLIP_NOISE_STD = 0.05


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


def get_cifar100_test_loader():
    """CIFAR-100 test loader with same transform as diagnostic/main."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_ds = NoisyIndexedDataset(
        "CIFAR100", DATA_ROOT, train=False, transform=transform, noise_type="none"
    )
    return DataLoader(test_ds, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=0)


def compute_ece(model, test_loader, device, n_bins=ECE_N_BINS):
    """Expected Calibration Error: bin by confidence, compare accuracy vs confidence per bin."""
    all_confs = []
    all_accs = []
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


def load_checkpoint(model, path, device):
    """Load model_state_dict from diagnostic or main.py checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    import pandas as pd

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    device = setup_device()
    test_loader = get_cifar100_test_loader()

    # 6 diagnostic runs (checkpoints in diagnostic_phase1/) + Standard SGD (in checkpoints/)
    experiments = [
        ("mkl_k1.5", "MKL (k=1.5)", os.path.join(DIAGNOSTIC_DIR, "checkpoint_mkl_k1.5.pth")),
        ("rho_sel0.4", "RHO-LOSS (sel=0.4)", os.path.join(DIAGNOSTIC_DIR, "checkpoint_rho_sel0.4.pth")),
        ("hasa_T15_k0.8", "HASA (T=15, k=0.8)", os.path.join(DIAGNOSTIC_DIR, "checkpoint_hasa_T15_k0.8.pth")),
        ("mkl_k1.25", "MKL (k=1.25)", os.path.join(DIAGNOSTIC_DIR, "checkpoint_mkl_k1.25.pth")),
        ("hasa_T10_k0.8", "HASA (T=10, k=0.8)", os.path.join(DIAGNOSTIC_DIR, "checkpoint_hasa_T10_k0.8.pth")),
        ("hasa_T10_k0.9", "HASA (T=10, k=0.9)", os.path.join(DIAGNOSTIC_DIR, "checkpoint_hasa_T10_k0.9.pth")),
        ("standard", "Standard SGD", None),  # path resolved below
    ]

    # Resolve Standard SGD checkpoint (try base then _best)
    std_base = os.path.join(CHECKPOINTS_DIR, "cifar100_uniform_sgd.pth")
    std_best = os.path.join(CHECKPOINTS_DIR, "cifar100_uniform_sgd_best.pth")
    std_path = std_base if os.path.isfile(std_base) else (std_best if os.path.isfile(std_best) else None)
    experiments = [(k, n, std_path if k == "standard" else p) for k, n, p in experiments]

    rows = []
    for alg_key, display_name, ckpt_path in experiments:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            print(f"  Skip {display_name}: checkpoint not found at {ckpt_path}")
            rows.append({"Algorithm": display_name, "ECE": float("nan"), "FlipRate": float("nan")})
            continue
        print(f"  Evaluating {display_name} ...")
        model = VGG_Small(num_classes=100).to(device)
        try:
            load_checkpoint(model, ckpt_path, device)
        except Exception as e:
            print(f"    Load failed: {e}")
            rows.append({"Algorithm": display_name, "ECE": float("nan"), "FlipRate": float("nan")})
            continue
        ece = compute_ece(model, test_loader, device, n_bins=ECE_N_BINS)
        flip = compute_flip_rate(model, test_loader, device, noise_std=FLIP_NOISE_STD)
        rows.append({"Algorithm": display_name, "ECE": ece, "FlipRate": flip})
        print(f"    ECE: {ece:.4f}  FlipRate: {flip:.4f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(DIAGNOSTIC_DIR, "diagnostic_ece_flip.csv")
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
