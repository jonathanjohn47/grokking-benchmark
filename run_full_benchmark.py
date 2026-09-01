#!/usr/bin/env python3
"""
Full benchmark orchestration: runs all predictor experiments in sequence.
- Single-head baseline (one run, fixed seed for reproducibility)
- Four-head baseline (3 independent runs with different seeds)
Cleans previous results before starting.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def cleanup_results():
    """Remove existing results to start fresh."""
    print("\n" + "="*70)
    print("CLEANUP: Removing previous results")
    print("="*70)

    results_dir = Path("results/single_head")
    runs_dir = Path("runs/four_head")

    removed = []

    if results_dir.exists():
        shutil.rmtree(results_dir)
        removed.append(str(results_dir))
        print(f"  ✓ Removed {results_dir}")

    if runs_dir.exists():
        shutil.rmtree(runs_dir)
        removed.append(str(runs_dir))
        print(f"  ✓ Removed {runs_dir}")

    if removed:
        print(f"\nCleaned {len(removed)} directories. Starting fresh.\n")
    else:
        print("\nNo previous results found. Starting fresh.\n")


def run_single_head():
    """Run single-head baseline training."""
    print("\n" + "="*70)
    print("STAGE 1: Single-Head Baseline Training")
    print("="*70)
    print("Command: python src/train.py")
    print("Expected output: results/single_head/{training,l2_norm,dropout,reports}/")
    print("="*70 + "\n")

    result = subprocess.run([sys.executable, "src/train.py"], cwd=".")

    if result.returncode != 0:
        print("\n❌ Single-head training FAILED")
        return False

    print("\n✓ Single-head training complete")
    return True


def run_four_head(run_num):
    """Run one four-head training session."""
    print("\n" + "="*70)
    print(f"STAGE 2.{run_num}: Four-Head Baseline Training (Run {run_num})")
    print("="*70)
    print(f"Command: python src/train_four_head.py")
    print(f"Expected output: runs/four_head/run_{run_num}/{{training,l2_norm,dropout,reports}}/")
    print("="*70 + "\n")

    result = subprocess.run([sys.executable, "src/train_four_head.py"], cwd=".")

    if result.returncode != 0:
        print(f"\n❌ Four-head run {run_num} FAILED")
        return False

    print(f"\n✓ Four-head run {run_num} complete")
    return True


def main():
    print("\n" + "="*70)
    print("GROKKING BENCHMARK: Full Experimental Pipeline")
    print("="*70)
    print("\nThis script will:")
    print("  1. Clean all previous results (fresh start)")
    print("  2. Run single-head baseline training (1 run)")
    print("  3. Run four-head baseline training (3 runs with independent seeds)")
    print("  4. Generate complete measurements for both L2 Norm & Dropout")
    print("\nTotal runs: 4 (1 single-head + 3 four-head)")
    print("Expected time: ~30-60 minutes depending on hardware")
    print("="*70)

    # Stage 0: Cleanup
    cleanup_results()

    # Stage 1: Single-head
    if not run_single_head():
        print("\n" + "="*70)
        print("PIPELINE ABORTED: Single-head training failed")
        print("="*70)
        sys.exit(1)

    # Stage 2: Four-head (3 runs)
    for run_num in [1, 2, 3]:
        if not run_four_head(run_num):
            print("\n" + "="*70)
            print(f"PIPELINE ABORTED: Four-head run {run_num} failed")
            print("="*70)
            sys.exit(1)

    # Success
    print("\n" + "="*70)
    print("✓ FULL BENCHMARK PIPELINE COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - results/single_head/ : single-head baseline")
    print("  - runs/four_head/run_1/ : four-head run 1")
    print("  - runs/four_head/run_2/ : four-head run 2")
    print("  - runs/four_head/run_3/ : four-head run 3")
    print("\nEach run contains:")
    print("  - training/ : raw training metrics")
    print("  - l2_norm/ : L2 Norm measurements + visualizations")
    print("  - dropout/ : Dropout measurements + visualizations")
    print("  - reports/ : combined PDF report (4 pages)")
    print("\nNext: Analyze results with:")
    print("  python analyze_benchmark_results.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
