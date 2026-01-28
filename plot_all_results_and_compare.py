#!/usr/bin/env python3
"""
Load results from ./checkpoints (no training), compare algorithms per task,
and produce summary tables and overview plots.

- Per-task: "best algorithms" plot (Standard, MKL, RHO, HASA; always include HASA)
- Per-task: summary table of all 4 method types + HASA
- One overview table (all tasks × methods) and one overview bar chart
"""

import os
import re
import argparse
import torch
import pandas as pd
import numpy as np

# Use Agg for headless saving
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import plot_results_custom, create_summary_table, plot_task_history

TASKS = ('cifar100', 'mnist', 'cloud')
TASK_TITLES = {
    'cifar100': 'CIFAR-100 (40% Label Noise)',
    'mnist': 'MNIST (Input Noise std=1.5)',
    'cloud': 'CLOUD (Input Noise std=1.0)',
}
METHOD_STANDARD = 'Standard'
METHOD_MKL = 'MKL'
METHOD_RHO = 'RHO'
METHOD_HASA = 'HASA'


def _parse_checkpoint_filename(basename):
    """
    Parse checkpoint filename (no path, no .pth) into (task, method_type, display_name).
    Returns None if not a training run we care about (e.g. il_model_*).
    """
    if basename.startswith('il_model_') or basename.startswith('il_map'):
        return None
    m = re.match(r'^(cifar100|mnist|cloud)_(.+)$', basename)
    if not m:
        return None
    task, rest = m.group(1), m.group(2)
    if task not in TASKS:
        return None
    # Strip _best if present
    if rest.endswith('_best'):
        rest = rest[:-5]
    # uniform_sgd -> Standard SGD
    if rest == 'uniform_sgd':
        return task, METHOD_STANDARD, 'Standard SGD'
    # mkl_sgd_kX
    km = re.match(r'^mkl_sgd_k([\d.]+)$', rest)
    if km:
        return task, METHOD_MKL, f'MKL (k={km.group(1)})'
    # rho_loss_selX
    rm = re.match(r'^rho_loss_sel([\d.]+)$', rest)
    if rm:
        return task, METHOD_RHO, f'RHO (sel={rm.group(1)})'
    # hasa_Tx_kY_nsZ
    hm = re.match(r'^hasa_T(\d+)_k([\d.]+)_ns([\d.e+-]+)$', rest)
    if hm:
        t, k, ns = hm.group(1), hm.group(2), hm.group(3)
        return task, METHOD_HASA, f'HASA (T={t}, k={k})'
    return None


def _load_result_from_checkpoint(path, map_location='cpu'):
    """Load checkpoint and return result dict with keys train_loss, train_acc, val_loss, val_acc."""
    try:
        ck = torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        return None
    tl = ck.get('train_losses') or []
    ta = ck.get('train_accs') or []
    vl = ck.get('val_losses') or []
    va = ck.get('val_accs') or []
    if not va and not vl:
        return None
    return {
        'train_loss': tl,
        'train_acc': ta,
        'val_loss': vl,
        'val_acc': va,
    }


def discover_and_load(checkpoint_dir):
    """
    Scan checkpoint_dir for .pth files, parse names, load histories.
    Prefer non-_best when both exist (longer curve). Skip il_model* etc.

    Returns: list of (task, method_type, display_name, result_dict, path)
    """
    if not os.path.isdir(checkpoint_dir):
        return []
    candidates = {}  # (task, display_name) -> (method_type, [(path, is_best), ...])
    for f in os.listdir(checkpoint_dir):
        if not f.endswith('.pth') or f.startswith('il_model') or 'il_map' in f:
            continue
        stem = f[:-4]  # drop .pth
        parsed = _parse_checkpoint_filename(stem)
        if parsed is None:
            continue
        task, method_type, display_name = parsed
        path = os.path.join(checkpoint_dir, f)
        is_best = stem.endswith('_best')
        key = (task, display_name)
        if key not in candidates:
            candidates[key] = (method_type, [])
        candidates[key][1].append((path, is_best))

    out = []
    for (task, display_name), (method_type, paths_best) in candidates.items():
        paths_best.sort(key=lambda x: (x[1], -1))  # non-best first for full-length curves
        chosen = paths_best[0][0]
        res = _load_result_from_checkpoint(chosen)
        if res is None:
            continue
        out.append((task, method_type, display_name, res, chosen))
    return out


def best_per_method(runs):
    """
    runs: list of (task, method_type, display_name, result_dict, path)
    Returns: list of (display_name, result_dict) for the best run in each method type (by max val_acc).
    Always include at least one HASA if any exist.
    """
    by_method = {}
    for t, m, name, res, _ in runs:
        if m not in by_method:
            by_method[m] = []
        by_method[m].append((name, res))
    best = []
    for m in (METHOD_STANDARD, METHOD_MKL, METHOD_RHO, METHOD_HASA):
        if m not in by_method:
            continue
        arr = by_method[m]
        # Pick run with highest max(val_acc)
        def score(item):
            name, res = item
            va = res.get('val_acc') or []
            return max(va) if va else -1.0
        chosen = max(arr, key=score)
        best.append(chosen)
    return best


def run(checkpoint_dir='./checkpoints', out_dir='./plots'):
    os.makedirs(out_dir, exist_ok=True)
    all_runs = discover_and_load(checkpoint_dir)
    if not all_runs:
        print("No checkpoint runs found. Ensure ./checkpoints contains .pth training checkpoints.")
        return

    by_task = {}
    for t, m, name, res, _ in all_runs:
        if t not in by_task:
            by_task[t] = []
        by_task[t].append((m, name, res))

    for task in TASKS:
        if task not in by_task:
            continue
        runs = by_task[task]
        task_title = TASK_TITLES[task]

        # (1) Best algorithms plot (always include HASA)
        best_list = best_per_method([(task, m, name, res, None) for m, name, res in runs])
        if best_list:
            dict_best = {n: r for n, r in best_list}
            plot_path = os.path.join(out_dir, f"compare_{task}_best.png")
            plot_task_history(dict_best, task_title, save_path=plot_path)

        # (2) Per-task summary table (all 4 method types + HASA)
        algo_names = [name for _, name, _ in runs]
        algo_results = [res for _, _, res in runs]
        table = create_summary_table(algo_results, algo_names)
        if table is not None:
            csv_path = os.path.join(out_dir, f"summary_{task}.csv")
            summary_data = []
            for res, name in zip(algo_results, algo_names):
                va = res.get('val_acc') or []
                if not va:
                    continue
                summary_data.append({
                    'Algorithm': name,
                    'Best Val Acc (%)': round(max(va) * 100, 2),
                    'Epoch': int(np.argmax(va) + 1),
                    'Final Val Acc (%)': round(va[-1] * 100, 2),
                    'Final Train Acc (%)': round((res.get('train_acc') or [0])[-1] * 100, 2),
                })
            if summary_data:
                df_sum = pd.DataFrame(summary_data)
                df_sum.to_csv(csv_path, index=False)
                print(f"\n>>> Summary Table: {task_title}\n")
                print(df_sum.to_string(index=False))

    # (3) Overview table (all tasks × methods)
    overview_rows = []
    for task in TASKS:
        if task not in by_task:
            continue
        for _, name, res in by_task[task]:
            va = res.get('val_acc') or []
            if not va:
                continue
            overview_rows.append({
                'Task': TASK_TITLES[task],
                'Algorithm': name,
                'Best Val Acc (%)': round(max(va) * 100, 2),
                'Epoch': np.argmax(va) + 1,
                'Final Val Acc (%)': round(va[-1] * 100, 2),
            })
    if overview_rows:
        df_overview = pd.DataFrame(overview_rows)
        overview_csv = os.path.join(out_dir, "overview_table.csv")
        df_overview.to_csv(overview_csv, index=False)
        print("\n>>> Overview Table (all tasks × methods)\n")
        print(df_overview.to_string())

    # (4) Overview bar chart: best val acc per (task, method_type)
    if by_task:
        # best per method type per task
        task_order = [t for t in TASKS if t in by_task]
        method_types = [METHOD_STANDARD, METHOD_MKL, METHOD_RHO, METHOD_HASA]
        data = []  # (task_title, method_type, best_val_acc)
        for task in task_order:
            runs = by_task[task]
            best_list = best_per_method([(task, m, name, res, None) for m, name, res in runs])
            for name, res in best_list:
                va = res.get('val_acc') or []
                if not va:
                    continue
                # Map display name to method type for grouping
                m = METHOD_HASA if 'HASA' in name else METHOD_MKL if 'MKL' in name else METHOD_RHO if 'RHO' in name else METHOD_STANDARD
                data.append((TASK_TITLES[task], m, name, max(va) * 100))
        if data:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_bar = pd.DataFrame(data, columns=['Task', 'Method', 'Algorithm', 'Best Val Acc (%)'])
            tasks_here = [TASK_TITLES[t] for t in task_order]
            x = np.arange(len(tasks_here))
            n_methods = len(method_types)
            width = 0.8 / max(n_methods, 1)
            for i, m in enumerate(method_types):
                vals = []
                for tt in tasks_here:
                    row = df_bar[(df_bar['Task'] == tt) & (df_bar['Method'] == m)]
                    vals.append(row['Best Val Acc (%)'].max() if not row.empty else 0)
                off = (i - n_methods / 2 + 0.5) * width
                ax.bar(x + off, vals, width, label=m)
            ax.set_xticks(x)
            ax.set_xticklabels(tasks_here, rotation=15, ha='right')
            ax.set_ylabel('Best Val Acc (%)')
            ax.set_title('Best Validation Accuracy by Task and Method')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            bar_path = os.path.join(out_dir, "overview_bars.png")
            plt.savefig(bar_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Overview bar chart saved to {bar_path}")

    print(f"\nAll outputs written under {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot and compare results from checkpoints (no training).')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory containing .pth checkpoints')
    parser.add_argument('--out_dir', '--plot_dir', type=str, default='./plots', dest='out_dir', help='Output directory for plots and tables')
    args = parser.parse_args()
    run(checkpoint_dir=args.checkpoint_dir, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
