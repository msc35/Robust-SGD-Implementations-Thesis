#!/usr/bin/env python3
"""
Standalone thesis diagnostics:
- Single HASA run on CIFAR-100 (40% label noise)
- Per-sample tracking and variance logging
- Generates thesis figures
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_loader import NoisyIndexedDataset
from src.models import VGG_Small
from src.trainers import HASATrainer, _inject_sgld_noise
from src.utils import validate

# =========================
# Configuration (hardcoded)
# =========================
SEED = 42
DATA_ROOT = "./data"
OUT_DIR = "./new_plots/new_run"
EPOCHS = 150
BATCH_SIZE = 32
NOISE_RATE = 0.4
WINDOW_T = 15
K_RATIO = 0.8
NOISE_SCALE = 1e-4
LEARNING_RATE = 0.001


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


def setup_data():
    """Load CIFAR-100 with 40% symmetric noise. Return loaders, noise mask, tracked indices."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

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

    train_ds = NoisyIndexedDataset(
        "CIFAR100", DATA_ROOT, train=True, transform=transform_train,
        noise_type="symmetric", noise_rate=NOISE_RATE, random_seed=SEED
    )
    val_ds = NoisyIndexedDataset(
        "CIFAR100", DATA_ROOT, train=False, transform=transform_test, noise_type="none"
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    noise_mask = train_ds.noise_mask
    clean_indices = np.where(~noise_mask)[0]
    noisy_indices = np.where(noise_mask)[0]

    rng = np.random.RandomState(SEED)
    tracked_clean = rng.choice(clean_indices, size=5, replace=False).tolist()
    tracked_noisy = rng.choice(noisy_indices, size=5, replace=False).tolist()

    tracked = {
        "clean_indices": tracked_clean,
        "noisy_indices": tracked_noisy
    }

    return train_loader, val_loader, train_ds, noise_mask, tracked


def _compute_all_variances(hasa_trainer):
    """Compute variance for all samples in HASATrainer history buffer."""
    num_samples = hasa_trainer.num_samples
    variances = np.full(num_samples, np.inf, dtype=np.float32)
    for i in range(num_samples):
        count = int(hasa_trainer.history_count[i].item())
        if count >= 2:
            history = hasa_trainer.history_buffer[i, :count].cpu().numpy()
            variances[i] = float(np.var(history))
    return variances


def train_with_logging(device, train_loader, val_loader, train_ds, noise_mask, tracked):
    """Train HASA with per-sample tracking and selection summaries."""
    model = VGG_Small(num_classes=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_nored = nn.CrossEntropyLoss(reduction="none").to(device)

    num_samples = len(train_ds)
    hasa_trainer = HASATrainer(num_samples, WINDOW_T, device)

    clean_set = set(np.where(~noise_mask)[0].tolist())
    noisy_set = set(np.where(noise_mask)[0].tolist())
    total_clean = len(clean_set)

    tracked_set = set(tracked["clean_indices"] + tracked["noisy_indices"])
    per_sample_losses = []
    selection_summary = []

    variance_epoch16 = None
    variance_epoch150 = None
    tau_epoch16 = None
    tau_epoch150 = None

    best_val_acc = 0.0
    final_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        epoch_selected = set()
        running_selected_loss = 0.0
        total_selected = 0
        correct = 0
        total = 0

        is_warmup = epoch < WINDOW_T

        for batch in train_loader:
            inputs, labels, indices = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            per_sample_loss = criterion_nored(outputs, labels)

            # Track per-sample losses for the 10 tracked indices
            idx_cpu = indices.detach().cpu().numpy()
            loss_cpu = per_sample_loss.detach().cpu().numpy()
            for i, idx in enumerate(idx_cpu):
                if int(idx) in tracked_set:
                    per_sample_losses.append({
                        "epoch": epoch + 1,
                        "sample_idx": int(idx),
                        "loss": float(loss_cpu[i]),
                        "is_noisy": 1 if int(idx) in noisy_set else 0
                    })

            if is_warmup:
                loss_mean = per_sample_loss.mean()
                loss_mean.backward()
                optimizer.step()
                _inject_sgld_noise(model, LEARNING_RATE, NOISE_SCALE)
                hasa_trainer.update_history(indices, per_sample_loss)
                running_selected_loss += per_sample_loss.sum().item()
                total_selected += inputs.size(0)
                epoch_selected.update(idx_cpu.tolist())
            else:
                variances = hasa_trainer.get_variance(indices)
                num_to_select = int(inputs.size(0) * K_RATIO)
                if num_to_select <= 0:
                    num_to_select = 1
                if num_to_select > inputs.size(0):
                    num_to_select = inputs.size(0)

                _, selected_batch_idx = torch.topk(-variances, num_to_select)
                selected_losses = per_sample_loss[selected_batch_idx]
                selected_indices = indices[selected_batch_idx]

                selected_losses.mean().backward()
                optimizer.step()
                _inject_sgld_noise(model, LEARNING_RATE, NOISE_SCALE)
                hasa_trainer.update_history(indices, per_sample_loss)

                running_selected_loss += selected_losses.sum().item()
                total_selected += num_to_select
                epoch_selected.update(selected_indices.detach().cpu().numpy().tolist())

            # Accuracy on full batch
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        final_val_acc = val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{EPOCHS} — val_acc: {val_acc*100:.2f}%")

        # Selection summary (from epoch 16 onward)
        if not is_warmup:
            total_sel = len(epoch_selected)
            clean_sel = len(epoch_selected & clean_set)
            noisy_sel = len(epoch_selected & noisy_set)
            precision = clean_sel / total_sel if total_sel > 0 else 0.0
            recall = clean_sel / total_clean if total_clean > 0 else 0.0
            selection_summary.append({
                "epoch": epoch + 1,
                "total_selected": total_sel,
                "clean_selected": clean_sel,
                "noisy_selected": noisy_sel,
                "precision": precision,
                "recall": recall
            })

        # Variance logging at epoch 16 (T+1) and epoch 150
        if epoch == WINDOW_T:
            variances = _compute_all_variances(hasa_trainer)
            variance_epoch16 = variances
            tau_epoch16 = float(np.percentile(variances[np.isfinite(variances)], 80))
        if epoch == EPOCHS - 1:
            variances = _compute_all_variances(hasa_trainer)
            variance_epoch150 = variances
            tau_epoch150 = float(np.percentile(variances[np.isfinite(variances)], 80))

    return {
        "model": model,
        "per_sample_losses": per_sample_losses,
        "selection_summary": selection_summary,
        "variance_epoch16": variance_epoch16,
        "variance_epoch150": variance_epoch150,
        "tau_epoch16": tau_epoch16,
        "tau_epoch150": tau_epoch150,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc
    }


def plot_sample_trajectories(df_losses, out_dir):
    """Plot A: per-sample loss trajectories for tracked samples."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    clean = df_losses[df_losses["is_noisy"] == 0]
    noisy = df_losses[df_losses["is_noisy"] == 1]

    for sid in clean["sample_idx"].unique():
        sub = clean[clean["sample_idx"] == sid]
        ax.plot(sub["epoch"], sub["loss"], color="blue", alpha=0.5, linewidth=1)
    for sid in noisy["sample_idx"].unique():
        sub = noisy[noisy["sample_idx"] == sid]
        ax.plot(sub["epoch"], sub["loss"], color="red", alpha=0.5, linewidth=1)

    # Mean lines
    mean_clean = clean.groupby("epoch")["loss"].mean()
    mean_noisy = noisy.groupby("epoch")["loss"].mean()
    ax.plot(mean_clean.index, mean_clean.values, color="blue", linewidth=2.5, label="Clean mean")
    ax.plot(mean_noisy.index, mean_noisy.values, color="red", linewidth=2.5, label="Noisy mean")

    ax.axvline(x=WINDOW_T, color="gray", linestyle="--", linewidth=1.5)
    ax.text(WINDOW_T + 1, ax.get_ylim()[1] * 0.95, "Selection begins", color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Trajectories: Clean vs. Noisy Samples Under HASA (CIFAR-100)")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="blue", linewidth=1, alpha=0.5, label="Clean samples"),
        plt.Line2D([0], [0], color="red", linewidth=1, alpha=0.5, label="Noisy samples"),
        plt.Line2D([0], [0], color="blue", linewidth=2.5, label="Clean mean"),
        plt.Line2D([0], [0], color="red", linewidth=2.5, label="Noisy mean"),
    ], loc="upper right", fontsize=10)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_loss_trajectories.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_histogram(df_var16, tau, out_dir):
    """Plot B: variance distribution histogram at epoch 16."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    clean = df_var16[df_var16["is_noisy"] == 0]["variance"].values
    noisy = df_var16[df_var16["is_noisy"] == 1]["variance"].values

    use_log = (clean > 0).all() and (noisy > 0).all()
    if use_log:
        ax.set_xscale("log")

    ax.hist(clean, bins=50, alpha=0.6, color="blue", label="Clean")
    ax.hist(noisy, bins=50, alpha=0.6, color="red", label="Noisy")
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.5)

    # Selected vs Dropped labels
    ax.text(0.02, 0.95, "Selected (Var ≤ τ)", transform=ax.transAxes, color="black")
    ax.text(0.70, 0.95, "Dropped (Var > τ)", transform=ax.transAxes, color="black")

    # Selected clean/noisy percentages
    selected_mask = df_var16["variance"] <= tau
    selected = df_var16[selected_mask]
    if len(selected) > 0:
        clean_pct = 100.0 * (selected["is_noisy"] == 0).mean()
        noisy_pct = 100.0 * (selected["is_noisy"] == 1).mean()
    else:
        clean_pct, noisy_pct = 0.0, 0.0
    text = f"Clean in Selected: {clean_pct:.1f}%\\nNoisy in Selected: {noisy_pct:.1f}%"
    ax.text(0.62, 0.70, text, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Variance of loss history")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Loss Variance at Selection Onset (Epoch 16)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_histogram.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selection_over_time(df_sel, out_dir):
    """Plot C: selection quality over time (precision/recall)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    df_sel = df_sel[df_sel["epoch"] >= WINDOW_T + 1]
    ax.plot(df_sel["epoch"], df_sel["precision"] * 100, color="blue", linewidth=2, label="HASA Precision")
    ax.plot(df_sel["epoch"], df_sel["recall"] * 100, color="red", linewidth=2, label="HASA Recall")
    ax.axhline(84.42, color="orange", linestyle="--", linewidth=1.5, label="MKL Precision (84.4%)")
    ax.axhline(92.34, color="green", linestyle="--", linewidth=1.5, label="MKL Recall (92.3%)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("HASA Selection Quality Over Training (CIFAR-100)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "selection_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_evolution(df_var16, tau16, df_var150, tau150, out_dir):
    """Plot D: variance evolution early vs late."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    clean16 = df_var16[df_var16["is_noisy"] == 0]["variance"].values
    noisy16 = df_var16[df_var16["is_noisy"] == 1]["variance"].values
    clean150 = df_var150[df_var150["is_noisy"] == 0]["variance"].values
    noisy150 = df_var150[df_var150["is_noisy"] == 1]["variance"].values

    all_vals = np.concatenate([clean16, noisy16, clean150, noisy150])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    bins = np.linspace(vmin, vmax, 50)

    ax1.hist(clean16, bins=bins, alpha=0.6, color="blue", label="Clean")
    ax1.hist(noisy16, bins=bins, alpha=0.6, color="red", label="Noisy")
    ax1.axvline(tau16, color="black", linestyle="--", linewidth=1.5)
    ax1.set_title("Epoch 16 (Selection Onset)")
    ax1.set_xlabel("Variance")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    ax2.hist(clean150, bins=bins, alpha=0.6, color="blue", label="Clean")
    ax2.hist(noisy150, bins=bins, alpha=0.6, color="red", label="Noisy")
    ax2.axvline(tau150, color="black", linestyle="--", linewidth=1.5)
    ax2.set_title("Epoch 150 (Final)")
    ax2.set_xlabel("Variance")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Evolution of Variance Separation: Early vs. Late Training", fontsize=14)
    ax1.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "variance_evolution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = setup_device()

    # Phase 1: Setup
    train_loader, val_loader, train_ds, noise_mask, tracked = setup_data()
    tracked_path = os.path.join(OUT_DIR, "tracked_samples.json")
    with open(tracked_path, "w") as f:
        json.dump(tracked, f, indent=2)
    print("Tracked samples saved:", tracked_path)

    # Phase 2: Train with logging
    logs = train_with_logging(device, train_loader, val_loader, train_ds, noise_mask, tracked)

    # Save per-sample losses
    df_losses = pd.DataFrame(logs["per_sample_losses"])
    losses_path = os.path.join(OUT_DIR, "per_sample_losses.csv")
    df_losses.to_csv(losses_path, index=False)

    # Save variance at epoch 16 and 150
    idxs = np.arange(len(train_ds))
    df_var16 = pd.DataFrame({
        "sample_idx": idxs,
        "variance": logs["variance_epoch16"],
        "is_noisy": noise_mask.astype(int)
    })
    var16_path = os.path.join(OUT_DIR, "variance_at_epoch16.csv")
    df_var16.to_csv(var16_path, index=False)

    df_var150 = pd.DataFrame({
        "sample_idx": idxs,
        "variance": logs["variance_epoch150"],
        "is_noisy": noise_mask.astype(int)
    })
    var150_path = os.path.join(OUT_DIR, "variance_at_epoch150.csv")
    df_var150.to_csv(var150_path, index=False)

    # Save selection threshold at epoch 16
    tau_path = os.path.join(OUT_DIR, "variance_threshold_epoch16.txt")
    with open(tau_path, "w") as f:
        f.write(f"{logs['tau_epoch16']:.6f}\n")

    # Save selection summary
    df_sel = pd.DataFrame(logs["selection_summary"])
    sel_path = os.path.join(OUT_DIR, "selection_summary.csv")
    df_sel.to_csv(sel_path, index=False)

    # Phase 3: Plots
    plot_sample_trajectories(df_losses, OUT_DIR)
    plot_variance_histogram(df_var16, logs["tau_epoch16"], OUT_DIR)
    plot_selection_over_time(df_sel, OUT_DIR)
    plot_variance_evolution(df_var16, logs["tau_epoch16"], df_var150, logs["tau_epoch150"], OUT_DIR)

    print("\n=== Summary ===")
    print(f"Final val_acc: {logs['final_val_acc']*100:.2f}%")
    print(f"Best val_acc: {logs['best_val_acc']*100:.2f}%")
    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
