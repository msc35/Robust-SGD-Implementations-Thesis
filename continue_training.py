#!/usr/bin/env python3
"""
Extend training to 150 epochs for selected configs. Resumes from existing
checkpoints when present; otherwise starts from scratch. Uses main.py.

CIFAR-100 (150 epochs):
  - Normal SGD
  - HASA (T=10, k=0.9), HASA (T=10, k=0.8), HASA (T=15, k=0.8)
  - RHO (sel=0.4)
  - MKL (k=1.5)

CLOUD (150 epochs):
  - HASA (T=15, k=0.7)
"""

import argparse
import subprocess
import sys
import os
import shutil
from typing import List, Tuple

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

DEFAULT_EPOCHS = 150
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_PLOT_DIR = "./plots"


def ensure_resumable_checkpoint(checkpoint_dir: str, base_name: str) -> None:
    """
    If the canonical checkpoint base_name.pth does not exist but base_name_best.pth
    does, copy _best to base so main.py can resume and write to the same path.
    """
    path = os.path.join(checkpoint_dir, f"{base_name}.pth")
    best_path = os.path.join(checkpoint_dir, f"{base_name}_best.pth")
    if not os.path.exists(path) and os.path.exists(best_path):
        shutil.copy2(best_path, path)
        print(f"  (copied {base_name}_best.pth -> {base_name}.pth for resuming)")


def run_one(cmd: List[str], exp_name: str) -> Tuple[bool, str]:
    """Run one main.py process; return (success, error_message)."""
    print(f"\n{'='*80}\n{BOLD}Running: {exp_name}{RESET}\nCommand: {' '.join(cmd)}\n{'='*80}\n")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        out_lines = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line)
                out_lines.append(line)
        proc.wait(timeout=36000)
        out = "\n".join(out_lines)
        if proc.returncode == 0:
            print(f"\n{GREEN}✓ SUCCESS: {exp_name}{RESET}")
            return True, ""
        print(f"\n{RED}✗ FAILED: {exp_name} (exit {proc.returncode}){RESET}\n{out[-500:]}")
        return False, out[-500:]
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"\n{RED}✗ TIMEOUT: {exp_name}{RESET}")
        return False, "Timeout"
    except Exception as e:
        print(f"\n{RED}✗ EXCEPTION: {exp_name} — {e}{RESET}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Extend selected configs to 150 epochs (CIFAR + CLOUD), resuming when checkpoints exist."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Target total epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Checkpoint directory (must match where runs were saved)")
    parser.add_argument("--plot_dir", type=str, default=DEFAULT_PLOT_DIR,
                        help="Directory to save plots")
    args = parser.parse_args()
    target_epochs = args.epochs
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    plot_dir = os.path.abspath(args.plot_dir)

    base_cmd = [
        sys.executable, "main.py",
        "--epochs", str(target_epochs),
        "--checkpoint_dir", checkpoint_dir,
        "--plot_dir", plot_dir,
    ]

    # (display_name, task, extra_args, checkpoint_base_name)
    experiments = [
        # CIFAR-100
        (
            f"CIFAR-100 Normal SGD → {target_epochs} epochs",
            "cifar100",
            ["--algorithm", "standard"],
            "cifar100_uniform_sgd",
        ),
        (
            f"CIFAR-100 HASA (T=10, k=0.9) → {target_epochs} epochs",
            "cifar100",
            ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.9", "--noise_scale", "0.0001"],
            "cifar100_hasa_T10_k0.9_ns0.0001",
        ),
        (
            f"CIFAR-100 HASA (T=10, k=0.8) → {target_epochs} epochs",
            "cifar100",
            ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.8", "--noise_scale", "0.0001"],
            "cifar100_hasa_T10_k0.8_ns0.0001",
        ),
        (
            f"CIFAR-100 HASA (T=15, k=0.8) → {target_epochs} epochs",
            "cifar100",
            ["--algorithm", "hasa", "--window_size", "15", "--selection_ratio", "0.8", "--noise_scale", "0.0001"],
            "cifar100_hasa_T15_k0.8_ns0.0001",
        ),
        (
            f"CIFAR-100 RHO (sel=0.4) → {target_epochs} epochs",
            "cifar100",
            ["--algorithm", "rho", "--selection_ratio", "0.4"],
            "cifar100_rho_loss_sel0.4",
        ),
        (
            f"CIFAR-100 MKL (k=1.5) → {target_epochs} epochs",
            "cifar100",
            ["--algorithm", "mkl", "--k_ratio", "1.5"],
            "cifar100_mkl_sgd_k1.5",
        ),
        # CLOUD
        (
            f"CLOUD HASA (T=15, k=0.7) → {target_epochs} epochs",
            "cloud",
            ["--algorithm", "hasa", "--window_size", "15", "--selection_ratio", "0.7", "--noise_scale", "0.0001"],
            "cloud_hasa_T15_k0.7_ns0.0001",
        ),
    ]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{BOLD}Continue training → {target_epochs} epochs (CIFAR + CLOUD){RESET}")
    print(f"Checkpoint dir: {checkpoint_dir}. Resumes when checkpoint exists.\n")

    results = []
    for i, (exp_name, task, extra_args, ckpt_base) in enumerate(experiments, 1):
        ensure_resumable_checkpoint(checkpoint_dir, ckpt_base)
        cmd = base_cmd + ["--task", task] + extra_args
        ok, err = run_one(cmd, f"[{i}/{len(experiments)}] {exp_name}")
        results.append((exp_name, ok))

    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{RED}Failed: {len(failed)} run(s){RESET}: {failed}")
        return 1
    print(f"\n{GREEN}All {len(experiments)} runs completed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
