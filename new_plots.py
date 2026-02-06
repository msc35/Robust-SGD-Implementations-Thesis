#!/usr/bin/env python3
"""
Generate publication-ready plots from existing CSV and checkpoint data.
Saves all figures to new_plots/ as high-DPI PNGs (dpi=300).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Paths: try project root first, then fallbacks
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "new_plots")
DIAG_DIR = os.path.join(PROJECT_ROOT, "diagnostic_phase1")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# Style
plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
DPI = 300


def _csv_path(name, *fallback_dirs):
    """Resolve CSV path: project root first, then fallback dirs."""
    for d in [PROJECT_ROOT] + list(fallback_dirs):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return os.path.join(PROJECT_ROOT, name)


def _color_family(alg):
    """Return (color, marker) by method family for scatter/bar."""
    a = str(alg).upper()
    if "MKL" in a:
        return "orange", "o"
    if "RHO" in a:
        return "seagreen", "s"
    if "HASA" in a:
        return "indianred", "^"
    return "steelblue", "o"


def plot1_precision_recall_scatter():
    """Plot 1: Precision-Recall scatter from selection_analysis.csv"""
    path = _csv_path("selection_analysis.csv", DIAG_DIR)
    if not os.path.isfile(path):
        print(f"  Skip Plot 1: {path} not found")
        return
    df = pd.read_csv(path)
    df["Algorithm"] = df["Algorithm"].astype(str).str.strip('"')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for _, row in df.iterrows():
        alg = row["Algorithm"]
        color, marker = _color_family(alg)
        ax.scatter(row["Recall_Percent"], row["Precision_Percent"], c=color, marker=marker, s=80, edgecolors="black", linewidths=0.5)
        ax.annotate(alg, (row["Recall_Percent"], row["Precision_Percent"]), xytext=(5, 5), textcoords="offset points", fontsize=10)
    ax.plot([50, 100], [50, 100], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision–Recall Trade-off in Sample Selection (CIFAR-100)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "precision_recall_scatter.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved precision_recall_scatter.png")


def plot2_ece_fliprate_bars():
    """Plot 2: ECE & Flip Rate side-by-side bars from diagnostic_ece_flip.csv"""
    path = _csv_path("diagnostic_ece_flip.csv", DIAG_DIR)
    if not os.path.isfile(path):
        print(f"  Skip Plot 2: {path} not found")
        return
    df = pd.read_csv(path)
    df["Algorithm"] = df["Algorithm"].astype(str).str.strip('"')
    std_ece = df.loc[df["Algorithm"] == "Standard SGD", "ECE"].values
    std_ece = float(std_ece[0]) if len(std_ece) else 0.1959
    algos = df["Algorithm"].tolist()
    colors = [_color_family(a)[0] for a in algos]
    x = np.arange(len(algos))
    width = 0.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax1.bar(x, df["ECE"], width=width, color=colors, edgecolor="black", linewidth=0.3)
    ax1.axhline(y=std_ece, color="red", linestyle="--", linewidth=1.5, label=f"Standard SGD ECE ({std_ece:.4f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=30, ha="right")
    ax1.set_ylabel("ECE")
    ax1.set_title("ECE (lower is better)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x, df["FlipRate"], width=width, color=colors, edgecolor="black", linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos, rotation=30, ha="right")
    ax2.set_ylabel("Flip Rate")
    ax2.set_title("Flip Rate (lower is better)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Calibration and Prediction Stability (CIFAR-100)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ece_fliprate_bars.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved ece_fliprate_bars.png")


def _parse_hasa(alg_str):
    """Parse 'HASA (T=15, k=0.8)' -> (15, 0.8). Returns None if not HASA."""
    m = re.search(r"HASA\s*\(T\s*=\s*(\d+)\s*,\s*k\s*=\s*([\d.]+)\)", str(alg_str), re.I)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def plot3_hasa_heatmap(task_filter, task_title_short, out_name):
    """Plot 3: HASA heatmap for one task (CIFAR-100 or CLOUD)."""
    path = _csv_path("overview_table.csv", PLOTS_DIR)
    if not os.path.isfile(path):
        print(f"  Skip Plot 3 {out_name}: {path} not found")
        return
    df = pd.read_csv(path)
    df = df[df["Task"].str.contains(task_filter, case=False, na=False)]
    df = df[df["Algorithm"].str.contains("HASA", case=False, na=False)]
    if df.empty:
        print(f"  Skip Plot 3 {out_name}: no HASA rows for {task_filter}")
        return
    rows = []
    for _, r in df.iterrows():
        p = _parse_hasa(r["Algorithm"])
        if p is None:
            continue
        t, k = p
        rows.append({"T": t, "k": k, "Best Val Acc (%)": float(r["Best Val Acc (%)"])})
    if not rows:
        return
    heat_df = pd.DataFrame(rows)
    T_vals = sorted(heat_df["T"].unique())
    k_vals = sorted(heat_df["k"].unique())
    grid = np.full((len(T_vals), len(k_vals)), np.nan)
    for _, r in heat_df.iterrows():
        i = T_vals.index(r["T"])
        j = k_vals.index(r["k"])
        grid[i, j] = r["Best Val Acc (%)"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    cmap = plt.cm.YlOrRd
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=np.nanmin(grid) if np.any(~np.isnan(grid)) else 0, vmax=np.nanmax(grid) if np.any(~np.isnan(grid)) else 100)
    ax.set_xticks(np.arange(len(k_vals)))
    ax.set_yticks(np.arange(len(T_vals)))
    ax.set_xticklabels(k_vals)
    ax.set_yticklabels(T_vals)
    ax.set_xlabel("k (selection ratio)")
    ax.set_ylabel("T (window size)")
    for i in range(len(T_vals)):
        for j in range(len(k_vals)):
            v = grid[i, j]
            if np.isnan(v):
                text = "N/A"
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, facecolor="lightgray", alpha=0.7))
            else:
                text = f"{v:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="Best Val Acc (%)")
    ax.set_title(f"HASA Hyperparameter Sensitivity — {task_title_short} (Best Val Acc %)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot4_hasa_sensitivity_lines():
    """Plot 4: HASA sensitivity — fix T=10 vs k; fix k=0.8 vs T."""
    path = _csv_path("overview_table.csv", PLOTS_DIR)
    if not os.path.isfile(path):
        print(f"  Skip Plot 4: {path} not found")
        return
    df = pd.read_csv(path)
    df = df[df["Task"].str.contains("CIFAR-100", case=False, na=False)]
    df = df[df["Algorithm"].str.contains("HASA", case=False, na=False)]
    if df.empty:
        return
    rows = []
    for _, r in df.iterrows():
        p = _parse_hasa(r["Algorithm"])
        if p is None:
            continue
        rows.append({"T": p[0], "k": p[1], "Best Val Acc (%)": float(r["Best Val Acc (%)"])})
    heat_df = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    # (a) T=10, vary k
    sub = heat_df[heat_df["T"] == 10].sort_values("k")
    if not sub.empty:
        ax1.plot(sub["k"], sub["Best Val Acc (%)"], "o-", color="indianred", linewidth=2, markersize=8)
        ax1.set_xlabel("k (selection ratio)")
        ax1.set_ylabel("Best Val Acc (%)")
        ax1.set_title("T = 10 (vary k)")
    ax1.axhline(47.64, color="steelblue", linestyle="--", linewidth=1.5, label="Standard SGD (47.64%)")
    ax1.axhline(51.09, color="orange", linestyle="--", linewidth=1.5, label="MKL (k=1.5) (51.09%)")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    # (b) k=0.8, vary T
    sub2 = heat_df[heat_df["k"] == 0.8].sort_values("T")
    if not sub2.empty:
        ax2.plot(sub2["T"], sub2["Best Val Acc (%)"], "o-", color="indianred", linewidth=2, markersize=8)
        ax2.set_xlabel("T (window size)")
        ax2.set_ylabel("Best Val Acc (%)")
        ax2.set_title("k = 0.8 (vary T)")
    ax2.axhline(47.64, color="steelblue", linestyle="--", linewidth=1.5, label="Standard SGD (47.64%)")
    ax2.axhline(51.09, color="orange", linestyle="--", linewidth=1.5, label="MKL (k=1.5) (51.09%)")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Effect of History Window (T) and Selection Ratio (k) on CIFAR-100", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hasa_sensitivity_lines.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved hasa_sensitivity_lines.png")


def _match_algorithm(sub_df, alg_want):
    """Return row from sub_df where Algorithm matches alg_want (strip quotes, exact or key parts)."""
    want = str(alg_want).strip('"')
    for _, r in sub_df.iterrows():
        a = str(r["Algorithm"]).strip('"')
        if a == want:
            return r
        if want.startswith("HASA") and "HASA" in a:
            # e.g. want "HASA (T=15, k=0.8)" and a is "HASA (T=15, k=0.8)"
            if want.split("(")[-1].rstrip(")") in a:
                return r
        if want.startswith("MKL") and "MKL" in a and want.split("(")[-1].rstrip(")") in a:
            return r
        if want.startswith("RHO") and "RHO" in a and "sel=0.4" in a:
            return r
        if want == "Standard SGD" and a == "Standard SGD":
            return r
    return None


def plot5_best_final_gap():
    """Plot 5: Best vs Final accuracy gap by dataset and algorithm."""
    path = _csv_path("overview_table.csv", PLOTS_DIR)
    if not os.path.isfile(path):
        print(f"  Skip Plot 5: {path} not found")
        return
    df = pd.read_csv(path)
    picks = [
        ("CIFAR-100 (40% Label Noise)", ["Standard SGD", "MKL (k=1.5)", "RHO (sel=0.4)", "HASA (T=15, k=0.8)"]),
        ("CLOUD (Input Noise std=1.0)", ["Standard SGD", "MKL (k=1.25)", "RHO (sel=0.4)", "HASA (T=15, k=0.7)"]),
        ("MNIST (Input Noise std=1.5)", ["Standard SGD", "MKL (k=1.25)", "RHO (sel=0.4)", "HASA (T=10, k=0.5)"]),
    ]
    datasets = []
    gaps = []
    for task, algs in picks:
        sub = df[df["Task"] == task]
        if sub.empty:
            continue
        datasets.append(task.replace(" (40% Label Noise)", "").replace(" (Input Noise std=1.0)", "").replace(" (Input Noise std=1.5)", ""))
        row_gaps = []
        for alg in algs:
            r = _match_algorithm(sub, alg)
            if r is not None:
                best = float(r["Best Val Acc (%)"])
                final = float(r["Final Val Acc (%)"])
                row_gaps.append(best - final)
            else:
                row_gaps.append(0)
        gaps.append(row_gaps)
    if not datasets:
        return
    # Grouped bar
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    x = np.arange(len(datasets))
    width = 0.2
    alg_names = ["Standard SGD", "MKL", "RHO", "HASA"]
    colors = ["steelblue", "orange", "seagreen", "indianred"]
    for i in range(4):
        vals = [g[i] for g in gaps]
        ax.bar(x + (i - 1.5) * width, vals, width, label=alg_names[i], color=colors[i], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Accuracy Drop (Best − Final, pp)")
    ax.set_title("Training Stability: Gap Between Peak and Final Accuracy")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "best_final_gap.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved best_final_gap.png")


def plot6_reliability_diagram():
    """Plot 6: Reliability diagram from 3 checkpoints (Standard SGD, MKL k=1.25, HASA T=15 k=0.8)."""
    from src.models import VGG_Small
    from src.data_loader import NoisyIndexedDataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_ds = NoisyIndexedDataset("CIFAR100", DATA_ROOT, train=False, transform=transform, noise_type="none")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
        except Exception:
            pass
    # Checkpoint paths: Standard from checkpoints, MKL and HASA from diagnostic_phase1
    models_config = [
        ("Standard SGD", os.path.join(CHECKPOINTS_DIR, "cifar100_uniform_sgd.pth"), "steelblue"),
        ("MKL (k=1.25)", os.path.join(DIAG_DIR, "checkpoint_mkl_k1.25.pth"), "orange"),
        ("HASA (T=15, k=0.8)", os.path.join(DIAG_DIR, "checkpoint_hasa_T15_k0.8.pth"), "indianred"),
    ]
    # Resolve Standard SGD path (try _best if base missing)
    for i, (name, path, _) in enumerate(models_config):
        if "Standard" in name and not os.path.isfile(path):
            alt = os.path.join(CHECKPOINTS_DIR, "cifar100_uniform_sgd_best.pth")
            if os.path.isfile(alt):
                models_config[i] = (name, alt, models_config[i][2])
    n_bins = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], sharex=True, figsize=(8, 8), gridspec_kw={"hspace": 0.25})
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    bin_edges_global = np.linspace(0, 1, n_bins + 1)
    bin_counts_for_bar = None
    calibration_curves = []  # (name, color, bin_confs, bin_accs) for separate save
    for name, ckpt_path, color in models_config:
        if not os.path.isfile(ckpt_path):
            print(f"  Skip {name}: {ckpt_path} not found")
            continue
        model = VGG_Small(num_classes=100).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        model.load_state_dict(state, strict=True)
        model.eval()
        all_conf = []
        all_correct = []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                conf, _ = probs.max(1)
                correct = (logits.argmax(1) == y).float()
                all_conf.append(conf.cpu().numpy())
                all_correct.append(correct.cpu().numpy())
        conf = np.concatenate(all_conf)
        correct = np.concatenate(all_correct)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (conf >= low) & (conf < high) if i < n_bins - 1 else (conf >= low) & (conf <= high)
            if mask.sum() > 0:
                bin_accs.append(correct[mask].mean())
                bin_confs.append(conf[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)
        bin_accs = np.array(bin_accs)
        bin_confs = np.array(bin_confs)
        if bin_counts_for_bar is None:
            bin_counts_for_bar = np.array(bin_counts)
        calibration_curves.append((name, color, np.array(bin_confs), np.array(bin_accs)))
        valid = ~np.isnan(bin_accs)
        if np.any(valid):
            ax1.plot(bin_confs[valid], bin_accs[valid], "o-", color=color, linewidth=2, markersize=6, label=name)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram — CIFAR-100 (40% Label Noise)")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", labelbottom=False)
    # Bottom subplot: bar chart of bin counts
    if bin_counts_for_bar is not None:
        bin_centers = (bin_edges_global[:-1] + bin_edges_global[1:]) / 2
        ax2.bar(bin_centers, bin_counts_for_bar, width=0.08, color="gray", alpha=0.7, edgecolor="black", linewidth=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Mean predicted confidence")
        ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(pad=1.2)
    fig.savefig(os.path.join(OUT_DIR, "reliability_diagram.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved reliability_diagram.png")
    # Save the two parts separately (no extra inference)
    if bin_counts_for_bar is not None:
        bin_centers = (bin_edges_global[:-1] + bin_edges_global[1:]) / 2
        # Calibration-only figure
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
        ax_cal.set_facecolor("white")
        fig_cal.patch.set_facecolor("white")
        ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
        for name, color, bin_confs, bin_accs in calibration_curves:
            valid = ~np.isnan(bin_accs)
            if np.any(valid):
                ax_cal.plot(bin_confs[valid], bin_accs[valid], "o-", color=color, linewidth=2, markersize=6, label=name)
        ax_cal.set_xlabel("Mean predicted confidence")
        ax_cal.set_ylabel("Accuracy")
        ax_cal.set_title("Reliability Diagram — CIFAR-100 (40% Label Noise)")
        ax_cal.legend(loc="lower right", fontsize=10)
        ax_cal.grid(True, alpha=0.3)
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        plt.tight_layout()
        fig_cal.savefig(os.path.join(OUT_DIR, "reliability_diagram_calibration_only.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig_cal)
        # Bin-counts-only figure
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        ax_bar.set_facecolor("white")
        fig_bar.patch.set_facecolor("white")
        ax_bar.bar(bin_centers, bin_counts_for_bar, width=0.08, color="gray", alpha=0.7, edgecolor="black", linewidth=0.3)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylabel("Count")
        ax_bar.set_xlabel("Mean predicted confidence")
        ax_bar.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig_bar.savefig(os.path.join(OUT_DIR, "reliability_diagram_bin_counts.png"), dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig_bar)
        print("  Saved reliability_diagram_calibration_only.png, reliability_diagram_bin_counts.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating plots in", OUT_DIR)
    plot1_precision_recall_scatter()
    plot2_ece_fliprate_bars()
    plot3_hasa_heatmap("CIFAR-100", "CIFAR-100", "hasa_heatmap_cifar100.png")
    plot3_hasa_heatmap("CLOUD", "CLOUD", "hasa_heatmap_cloud.png")
    plot4_hasa_sensitivity_lines()
    plot5_best_final_gap()
    plot6_reliability_diagram()
    print("Done.")


if __name__ == "__main__":
    main()
