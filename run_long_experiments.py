#!/usr/bin/env python3
"""
Long-run script to execute all experiment configurations with full epochs (100).

This script runs all 36 experiment configurations (3 datasets × 12 algorithm settings)
with 100 epochs each for the complete benchmark suite.
"""

import subprocess
import sys
import os
from typing import List, Dict, Tuple

# ANSI color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_colored(text: str, color: str = RESET):
    """Print colored text."""
    print(f"{color}{text}{RESET}")


def run_experiment(cmd: List[str], exp_name: str) -> Tuple[bool, str]:
    """
    Run a single experiment and return success status and error message.
    
    Args:
        cmd: Command to run as list of strings
        exp_name: Name of the experiment for logging
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    print_colored(f"\n{'='*80}", BLUE)
    print_colored(f"Running: {exp_name}", BOLD)
    print_colored(f"Command: {' '.join(cmd)}", YELLOW)
    print_colored(f"{'='*80}\n", BLUE)
    
    try:
        # Run with real-time output to see progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output in real-time
        output_lines = []
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(line)  # Print in real-time
                    output_lines.append(line)
            process.wait(timeout=36000)  # 10 hour timeout per experiment (for long runs)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise subprocess.TimeoutExpired(cmd, 36000)
        
        output = '\n'.join(output_lines)
        
        if process.returncode == 0:
            print_colored(f"\n✓ SUCCESS: {exp_name}", GREEN)
            return True, ""
        else:
            error_msg = f"Return code: {process.returncode}\n"
            error_msg += f"Output:\n{output[-500:]}\n"  # Last 500 chars
            print_colored(f"\n✗ FAILED: {exp_name}", RED)
            print_colored(f"Error details:\n{error_msg}", RED)
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = f"Experiment timed out after 10 hours"
        print_colored(f"\n✗ TIMEOUT: {exp_name}", RED)
        print_colored(f"Error: {error_msg}", RED)
        return False, error_msg
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        print_colored(f"\n✗ EXCEPTION: {exp_name}", RED)
        print_colored(f"Error: {error_msg}", RED)
        return False, error_msg


def main():
    """Main function to run all long-run experiments."""
    
    # Long-run settings
    LONG_RUN_EPOCHS = 100
    
    # Base command (plots will be saved to ./plots directory automatically)
    # Use MPS for speed (optimizations applied to prevent crashes)
    base_cmd = ["python", "main.py", "--plot_dir", "./plots"]
    
    # Experiment configurations - ALL 36 experiments
    experiments = []
    
    # ===== Group 1: CIFAR-100 (40% Label Noise) =====
    base_cifar = ["--task", "cifar100", "--epochs", str(LONG_RUN_EPOCHS)]
    
    experiments.extend([
        (base_cifar + ["--algorithm", "standard"], "CIFAR-100 - Standard SGD"),
        (base_cifar + ["--algorithm", "mkl", "--k_ratio", "2.0"], "CIFAR-100 - MKL-SGD (k=2.0)"),
        (base_cifar + ["--algorithm", "mkl", "--k_ratio", "1.5"], "CIFAR-100 - MKL-SGD (k=1.5)"),
        (base_cifar + ["--algorithm", "mkl", "--k_ratio", "1.25"], "CIFAR-100 - MKL-SGD (k=1.25)"),
        (base_cifar + ["--algorithm", "rho", "--selection_ratio", "0.1"], "CIFAR-100 - RHO-LOSS (sel=0.1)"),
        (base_cifar + ["--algorithm", "rho", "--selection_ratio", "0.2"], "CIFAR-100 - RHO-LOSS (sel=0.2)"),
        (base_cifar + ["--algorithm", "rho", "--selection_ratio", "0.3"], "CIFAR-100 - RHO-LOSS (sel=0.3)"),
        (base_cifar + ["--algorithm", "rho", "--selection_ratio", "0.4"], "CIFAR-100 - RHO-LOSS (sel=0.4)"),
        (base_cifar + ["--algorithm", "hasa", "--window_size", "5", "--selection_ratio", "0.6"], 
         "CIFAR-100 - HASA (T=5, k=0.6)"),
        (base_cifar + ["--algorithm", "hasa", "--window_size", "5", "--selection_ratio", "0.7"], 
         "CIFAR-100 - HASA (T=5, k=0.7)"),
        (base_cifar + ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.6"], 
         "CIFAR-100 - HASA (T=10, k=0.6)"),
        (base_cifar + ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.7"], 
         "CIFAR-100 - HASA (T=10, k=0.7)"),
    ])
    
    # ===== Group 2: MNIST (Input Noise std=1.5) =====
    base_mnist = ["--task", "mnist", "--epochs", str(LONG_RUN_EPOCHS)]
    
    experiments.extend([
        (base_mnist + ["--algorithm", "standard"], "MNIST - Standard SGD"),
        (base_mnist + ["--algorithm", "mkl", "--k_ratio", "2.0"], "MNIST - MKL-SGD (k=2.0)"),
        (base_mnist + ["--algorithm", "mkl", "--k_ratio", "1.5"], "MNIST - MKL-SGD (k=1.5)"),
        (base_mnist + ["--algorithm", "mkl", "--k_ratio", "1.25"], "MNIST - MKL-SGD (k=1.25)"),
        (base_mnist + ["--algorithm", "rho", "--selection_ratio", "0.1"], "MNIST - RHO-LOSS (sel=0.1)"),
        (base_mnist + ["--algorithm", "rho", "--selection_ratio", "0.2"], "MNIST - RHO-LOSS (sel=0.2)"),
        (base_mnist + ["--algorithm", "rho", "--selection_ratio", "0.3"], "MNIST - RHO-LOSS (sel=0.3)"),
        (base_mnist + ["--algorithm", "rho", "--selection_ratio", "0.4"], "MNIST - RHO-LOSS (sel=0.4)"),
        (base_mnist + ["--algorithm", "hasa", "--window_size", "5", "--selection_ratio", "0.6"], 
         "MNIST - HASA (T=5, k=0.6)"),
        (base_mnist + ["--algorithm", "hasa", "--window_size", "5", "--selection_ratio", "0.7"], 
         "MNIST - HASA (T=5, k=0.7)"),
        (base_mnist + ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.6"], 
         "MNIST - HASA (T=10, k=0.6)"),
        (base_mnist + ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.7"], 
         "MNIST - HASA (T=10, k=0.7)"),
    ])
    
    # ===== Group 3: CLOUD (Input Noise std=1.0) =====
    base_cloud = ["--task", "cloud", "--epochs", str(LONG_RUN_EPOCHS)]
    
    experiments.extend([
        (base_cloud + ["--algorithm", "standard"], "CLOUD - Standard SGD"),
        (base_cloud + ["--algorithm", "mkl", "--k_ratio", "2.0"], "CLOUD - MKL-SGD (k=2.0)"),
        (base_cloud + ["--algorithm", "mkl", "--k_ratio", "1.5"], "CLOUD - MKL-SGD (k=1.5)"),
        (base_cloud + ["--algorithm", "mkl", "--k_ratio", "1.25"], "CLOUD - MKL-SGD (k=1.25)"),
        (base_cloud + ["--algorithm", "rho", "--selection_ratio", "0.1"], "CLOUD - RHO-LOSS (sel=0.1)"),
        (base_cloud + ["--algorithm", "rho", "--selection_ratio", "0.2"], "CLOUD - RHO-LOSS (sel=0.2)"),
        (base_cloud + ["--algorithm", "rho", "--selection_ratio", "0.3"], "CLOUD - RHO-LOSS (sel=0.3)"),
        (base_cloud + ["--algorithm", "rho", "--selection_ratio", "0.4"], "CLOUD - RHO-LOSS (sel=0.4)"),
        (base_cloud + ["--algorithm", "hasa", "--window_size", "5", "--selection_ratio", "0.6"], 
         "CLOUD - HASA (T=5, k=0.6)"),
        (base_cloud + ["--algorithm", "hasa", "--window_size", "5", "--selection_ratio", "0.7"], 
         "CLOUD - HASA (T=5, k=0.7)"),
        (base_cloud + ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.6"], 
         "CLOUD - HASA (T=10, k=0.6)"),
        (base_cloud + ["--algorithm", "hasa", "--window_size", "10", "--selection_ratio", "0.7"], 
         "CLOUD - HASA (T=10, k=0.7)"),
    ])
    
    # Create plots directory
    os.makedirs("./plots", exist_ok=True)
    
    # Run all experiments
    print_colored(f"\n{'='*80}", BOLD)
    print_colored(f"LONG-RUN SUITE: Running {len(experiments)} experiments with {LONG_RUN_EPOCHS} epochs each", BOLD)
    print_colored(f"Plots will be saved to: ./plots/", BOLD)
    print_colored(f"Checkpoints will be saved to: ./checkpoints/", BOLD)
    print_colored(f"{'='*80}\n", BOLD)
    
    results = []
    for i, (cmd_args, exp_name) in enumerate(experiments, 1):
        print_colored(f"\n{'#'*80}", BOLD)
        print_colored(f"EXPERIMENT [{i}/{len(experiments)}]: {exp_name}", BOLD)
        print_colored(f"{'#'*80}", BOLD)
        success, error = run_experiment(base_cmd + cmd_args, exp_name)
        results.append((exp_name, success, error))
        
        # Show progress summary
        successful_so_far = sum(1 for r in results if r[1])
        print_colored(f"\nProgress: {successful_so_far}/{len(results)} successful so far", YELLOW)
        print_colored(f"Remaining: {len(experiments) - len(results)} experiments", YELLOW)
    
    # Summary
    print_colored(f"\n{'='*80}", BOLD)
    print_colored("LONG-RUN SUITE SUMMARY", BOLD)
    print_colored(f"{'='*80}\n", BOLD)
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print_colored(f"Total Experiments: {len(results)}", BOLD)
    print_colored(f"Successful: {len(successful)}", GREEN)
    print_colored(f"Failed: {len(failed)}", RED if failed else GREEN)
    
    if failed:
        print_colored(f"\n{'='*80}", RED)
        print_colored("FAILED EXPERIMENTS:", RED)
        print_colored(f"{'='*80}\n", RED)
        for exp_name, success, error in failed:
            print_colored(f"✗ {exp_name}", RED)
            print_colored(f"  {error[:200]}...", RED)  # First 200 chars of error
        return 1
    else:
        print_colored("\n✓ All experiments completed successfully!", GREEN)
        print_colored(f"All plots saved to: ./plots/", GREEN)
        return 0


if __name__ == '__main__':
    sys.exit(main())
