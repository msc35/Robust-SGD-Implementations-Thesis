#!/usr/bin/env python3
"""
Phase 1 Diagnostic Run: Fixed seed (42) for reproducible 40% label noise on CIFAR-100.
Trains 6 selection algorithms for 150 epochs, saves ground-truth noisy indices,
final selection masks, checkpoints, then computes selection composition analysis.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============ CRITICAL: Global seed BEFORE any dataset creation ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============ Imports after seed ============
from src.data_loader import NoisyIndexedDataset
from src.models import VGG_Small
from src.trainers import train_min_k_loss, train_rho_loss, train_hasa, train_il_model, compute_irreducible_loss
from src.utils import validate

DATA_ROOT = "./data"
BATCH_SIZE = 32
EPOCHS = 150
OUTPUT_DIR = "./diagnostic_phase1"
CIFAR_NOISE_RATE = 0.4


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


def setup_diagnostic_cifar100(device):
    """CIFAR-100 with 40% label noise, random_seed=SEED, batch_size=BATCH_SIZE."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = NoisyIndexedDataset(
        "CIFAR100", DATA_ROOT, train=True, transform=transform_train,
        noise_type="symmetric", noise_rate=CIFAR_NOISE_RATE, random_seed=SEED
    )
    test_dataset = NoisyIndexedDataset(
        "CIFAR100", DATA_ROOT, train=False, transform=transform_test, noise_type="none"
    )
    c_clean = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=transform_train)
    holdout_dataset = Subset(c_clean, list(range(len(c_clean) - 10000, len(c_clean))))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    holdout_loader = DataLoader(holdout_dataset, batch_size=256, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_nored = nn.CrossEntropyLoss(reduction="none").to(device)
    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "holdout_loader": holdout_loader,
        "train_dataset": train_dataset,
        "criterion": criterion,
        "criterion_nored": criterion_nored,
        "num_classes": 100,
    }


def get_il_map_diagnostic(data_config, device, out_dir):
    """Compute or load IL map for RHO; use diagnostic output dir."""
    il_map_path = os.path.join(out_dir, "il_map_cifar100.npy")
    il_model_path = os.path.join(out_dir, "il_model_cifar100.pth")
    if os.path.exists(il_map_path) and os.path.exists(il_model_path):
        print("  Loading existing IL map from", il_map_path)
        return np.load(il_map_path)
    print("  Computing IL map for RHO-LOSS...")
    il_model = VGG_Small(num_classes=100).to(device)
    il_model = train_il_model(
        il_model, data_config["holdout_loader"], data_config["test_loader"],
        device, num_epochs=100
    )
    torch.save(il_model.state_dict(), il_model_path)
    il_map = compute_irreducible_loss(
        il_model, data_config["train_dataset"],
        data_config["criterion_nored"], device
    )
    np.save(il_map_path, il_map)
    return il_map


# Experiment configs: (alg_key, display_name, train_kwargs, is_rho, is_hasa)
EXPERIMENTS = [
    ("mkl_k1.5", "MKL (k=1.5)", {"k_ratio": 1.5}, False, False),
    ("rho_sel0.4", "RHO-LOSS (sel=0.4)", {"selection_ratio": 0.4}, True, False),
    ("hasa_T15_k0.8", "HASA (T=15, k=0.8)", {"window_size_T": 15, "k_ratio": 0.8, "noise_scale": 1e-4}, False, True),
    ("mkl_k1.25", "MKL (k=1.25)", {"k_ratio": 1.25}, False, False),
    ("hasa_T10_k0.8", "HASA (T=10, k=0.8)", {"window_size_T": 10, "k_ratio": 0.8, "noise_scale": 1e-4}, False, True),
    ("hasa_T10_k0.9", "HASA (T=10, k=0.9)", {"window_size_T": 10, "k_ratio": 0.9, "noise_scale": 1e-4}, False, True),
]


def _precision_recall(selected_indices, clean_indices, noisy_indices, n_clean):
    """Compute precision and recall (percent) from selected indices vs ground truth."""
    if selected_indices is None:
        return 0.0, 0.0
    sel = selected_indices.ravel() if isinstance(selected_indices, np.ndarray) else selected_indices
    sel_set = set(np.asarray(sel).tolist())
    clean_set = set(clean_indices.tolist())
    clean_ret = len(sel_set & clean_set)
    tot_sel = len(sel_set)
    prec = (clean_ret / tot_sel * 100) if tot_sel > 0 else 0.0
    rec = (clean_ret / n_clean * 100) if n_clean > 0 else 0.0
    return prec, rec


def run_one_experiment(exp_idx, alg_key, display_name, train_kwargs, is_rho, is_hasa,
                       data_config, rho_il_map, device, out_dir,
                       clean_indices, noisy_indices, n_clean):
    """Run 150 epochs; save selection mask from best val-acc epoch. Save checkpoint and mask."""
    model = VGG_Small(num_classes=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = data_config["train_loader"]
    test_loader = data_config["test_loader"]
    criterion = data_config["criterion"]
    criterion_nored = data_config["criterion_nored"]
    hasa_trainer = None
    final_train_loss = None
    final_test_acc = None
    best_val_acc = 0.0
    best_epoch_selected_indices = None  # selection at epoch with best val acc (for precision/recall and saved mask)

    for epoch in range(EPOCHS):
        # Request selected indices every epoch for precision/recall logging
        if is_rho:
            out = train_rho_loss(
                model, rho_il_map, train_loader, criterion_nored, optimizer, device,
                return_selected_indices=True, **{k: v for k, v in train_kwargs.items() if k != "window_size_T" and k != "noise_scale"}
            )
            train_loss, train_acc, sel_epoch = out
        elif is_hasa:
            out = train_hasa(
                model, train_loader, criterion_nored, optimizer, device,
                hasa_trainer=hasa_trainer,
                train_dataset=data_config["train_dataset"],
                current_epoch=epoch,
                return_selected_indices=True,
                **train_kwargs
            )
            train_loss, train_acc, hasa_trainer, sel_epoch = out
        else:
            out = train_min_k_loss(
                model, train_loader, criterion_nored, optimizer, device,
                return_selected_indices=True, **train_kwargs
            )
            train_loss, train_acc, sel_epoch = out

        val_loss, val_acc = validate(model, test_loader, criterion, device)
        final_train_loss = train_loss
        final_test_acc = val_acc

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # Keep selection from this best-epoch for final precision/recall and saved mask
            best_epoch_selected_indices = np.copy(sel_epoch) if isinstance(sel_epoch, np.ndarray) else np.array(sel_epoch)

        prec, rec = _precision_recall(sel_epoch, clean_indices, noisy_indices, n_clean)
        # Match main.py style: print every epoch with Tr Loss/Acc, Val Loss/Acc, Best Val Acc, Precision, Recall
        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Tr Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"Best Val Acc: {best_val_acc*100:.2f}% | "
              f"Precision: {prec:.2f}% Recall: {rec:.2f}%")
        if is_best:
            print(f"  [New Best] Val Acc: {val_acc*100:.2f}%")

    # Save checkpoint
    ckpt_path = os.path.join(out_dir, f"checkpoint_{alg_key}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS - 1,
    }, ckpt_path)
    print(f"  Saved {ckpt_path}")

    # Save selection mask from best val-acc epoch (for precision/recall and analysis)
    selected_indices = best_epoch_selected_indices  # use best-epoch selection for downstream
    if selected_indices is not None:
        mask_path = os.path.join(out_dir, f"final_selection_mask_{alg_key}.npy")
        np.save(mask_path, selected_indices)
        print(f"  Saved {mask_path} (best-epoch selection)")

    return final_train_loss, final_test_acc, selected_indices


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = setup_device()
    print(f"Device: {device}")
    print(f"SEED: {SEED} | EPOCHS: {EPOCHS} | BATCH_SIZE: {BATCH_SIZE}")
    print(f"Output dir: {OUTPUT_DIR}\n")

    # Load dataset once and save ground-truth noisy indices
    print("Loading CIFAR-100 (40% label noise, fixed seed)...")
    data_config = setup_diagnostic_cifar100(device)
    train_dataset = data_config["train_dataset"]
    noisy_indices = np.where(train_dataset.noise_mask)[0]
    clean_indices = np.where(~train_dataset.noise_mask)[0]
    n_clean = len(clean_indices)
    n_noisy = len(noisy_indices)
    gt_path = os.path.join(OUTPUT_DIR, "noisy_indices_ground_truth.npy")
    np.save(gt_path, noisy_indices)
    print(f"Saved {gt_path} | Noisy: {n_noisy} | Clean: {n_clean}\n")

    # IL map for RHO (only needed once)
    rho_il_map = None
    for cfg in EXPERIMENTS:
        if cfg[3]:  # is_rho
            rho_il_map = get_il_map_diagnostic(data_config, device, OUTPUT_DIR)
            break

    # Run 6 experiments
    diagnostic_rows = []
    selection_masks = {}  # alg_key -> selected indices array
    for i, (alg_key, display_name, train_kwargs, is_rho, is_hasa) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*60}")
        print(f"Running Experiment [{i}/6]: {display_name}...")
        print(f"{'='*60}")
        try:
            final_train_loss, final_test_acc, selected = run_one_experiment(
                i, alg_key, display_name, train_kwargs, is_rho, is_hasa,
                data_config, rho_il_map, device, OUTPUT_DIR,
                clean_indices, noisy_indices, n_clean
            )
            diagnostic_rows.append({
                "Algorithm": display_name,
                "Final_Test_Acc": final_test_acc if final_test_acc is not None else float("nan"),
                "Final_Train_Loss": final_train_loss if final_train_loss is not None else float("nan"),
            })
            if selected is not None:
                selection_masks[alg_key] = selected
                # Print Precision and Recall after each run (Phase 1 requirement)
                sel_set = set(selected.ravel().tolist()) if isinstance(selected, np.ndarray) else set(selected)
                clean_ret = len(sel_set & set(clean_indices.tolist()))
                noisy_ret = len(sel_set & set(noisy_indices.tolist()))
                tot_sel = len(sel_set)
                prec = (clean_ret / tot_sel * 100) if tot_sel > 0 else 0.0
                rec = (clean_ret / n_clean * 100) if n_clean > 0 else 0.0
                print(f"  -> Precision: {prec:.2f}% | Recall: {rec:.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            diagnostic_rows.append({"Algorithm": display_name, "Final_Test_Acc": float("nan"), "Final_Train_Loss": float("nan")})

    # Save diagnostic_results.csv
    df_diag = pd.DataFrame(diagnostic_rows)
    diag_csv = os.path.join(OUTPUT_DIR, "diagnostic_results.csv")
    df_diag.to_csv(diag_csv, index=False)
    print(f"\nSaved {diag_csv}")

    # Selection analysis (only for algorithms that have a mask)
    selection_rows = []
    for alg_key, display_name, _, _, _ in EXPERIMENTS:
        if alg_key not in selection_masks:
            continue
        sel = selection_masks[alg_key]
        if isinstance(sel, np.ndarray):
            selected_set = set(sel.ravel().tolist())
        else:
            selected_set = set(sel)
        clean_retained = len(selected_set & set(clean_indices.tolist()))
        noisy_retained = len(selected_set & set(noisy_indices.tolist()))
        total_selected = len(selected_set)
        precision_pct = (clean_retained / total_selected * 100) if total_selected > 0 else 0.0
        recall_pct = (clean_retained / n_clean * 100) if n_clean > 0 else 0.0
        selection_rows.append({
            "Algorithm": display_name,
            "Total_Selected": total_selected,
            "Clean_Count": clean_retained,
            "Noisy_Count": noisy_retained,
            "Precision_Percent": round(precision_pct, 2),
            "Recall_Percent": round(recall_pct, 2),
        })
        print(f"  {display_name} | Precision: {precision_pct:.2f}% | Recall: {recall_pct:.2f}%")

    df_sel = pd.DataFrame(selection_rows)
    sel_csv = os.path.join(OUTPUT_DIR, "selection_analysis.csv")
    df_sel.to_csv(sel_csv, index=False)
    print(f"Saved {sel_csv}\n")

    # Stacked bar chart: selection composition
    if selection_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r["Algorithm"] for r in selection_rows]
        clean_kept = [r["Clean_Count"] for r in selection_rows]
        noisy_kept = [r["Noisy_Count"] for r in selection_rows]
        x = np.arange(len(names))
        width = 0.5
        ax.bar(x, clean_kept, width, label="Clean Data Kept", color="green", alpha=0.8)
        ax.bar(x, noisy_kept, width, bottom=clean_kept, label="Noisy Data Kept", color="red", alpha=0.8)
        ax.axhline(y=n_clean, color="gray", linestyle="--", linewidth=1.5, label=f"Total Clean in Dataset ({n_clean})")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Selection Composition (Phase 1 Diagnostic)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "selection_composition.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plot_path}")

    print("\nPhase 1 diagnostic complete.")


if __name__ == "__main__":
    main()
