#!/usr/bin/env python3
"""
Full benchmark pipeline: runs all predictor experiments in sequence, then
analyzes and visualizes the results. Single command, single script.

Stage 1: Single-head baseline training (1 run)
Stage 2: Four-head baseline training (3 independent runs, different seeds)
Stage 3: Analysis — comparison charts + PDF report across all runs
"""

import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# ======================================================================
# STAGE 1 & 2: TRAINING
# ======================================================================

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


# ======================================================================
# STAGE 3: ANALYSIS & VISUALIZATION
# ======================================================================

class BenchmarkAnalyzer:
    def __init__(self):
        self.single_head_dir = Path("results/single_head")
        self.four_head_base = Path("runs/four_head")
        self.output_dir = Path("benchmark_analysis")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def load_single_head(self):
        """Load single-head baseline results."""
        print("Loading single-head baseline...")

        training_dir = self.single_head_dir / "training"
        l2_norm_dir = self.single_head_dir / "l2_norm"
        dropout_dir = self.single_head_dir / "dropout"

        if not training_dir.exists():
            print("  ✗ Single-head results not found")
            return False

        self.results["single_head"] = {
            "train_acc": np.load(training_dir / "train_acc_history.npy"),
            "test_acc": np.load(training_dir / "test_acc_history.npy"),
            "loss": np.load(training_dir / "loss_history.npy"),
            "l2_norm": np.load(l2_norm_dir / "l2_norm_history.npy"),
            "dropout_gap_epochs": np.load(dropout_dir / "dropout_gap_epochs.npy"),
            "dropout_gap": np.load(dropout_dir / "dropout_gap_history.npy"),
            "dropout_gap_by_rate": np.load(dropout_dir / "dropout_gap_by_rate.npy"),
        }

        print(f"  ✓ Loaded single-head (epochs: {len(self.results['single_head']['train_acc'])})")
        return True

    def load_four_head_runs(self):
        """Load all four-head runs."""
        print("Loading four-head runs...")

        for run_num in [1, 2, 3]:
            run_dir = self.four_head_base / f"run_{run_num}"
            training_dir = run_dir / "training"
            l2_norm_dir = run_dir / "l2_norm"
            dropout_dir = run_dir / "dropout"

            if not training_dir.exists():
                print(f"  ⚠ Run {run_num} not found")
                continue

            self.results[f"four_head_run_{run_num}"] = {
                "train_acc": np.load(training_dir / "train_acc_history.npy"),
                "test_acc": np.load(training_dir / "test_acc_history.npy"),
                "loss": np.load(training_dir / "loss_history.npy"),
                "l2_norm": np.load(l2_norm_dir / "l2_norm_history.npy"),
                "dropout_gap_epochs": np.load(dropout_dir / "dropout_gap_epochs.npy"),
                "dropout_gap": np.load(dropout_dir / "dropout_gap_history.npy"),
                "dropout_gap_by_rate": np.load(dropout_dir / "dropout_gap_by_rate.npy"),
            }

            print(f"  ✓ Loaded run {run_num} (epochs: {len(self.results[f'four_head_run_{run_num}']['train_acc'])})")

        return len(self.results) > 1

    def grok_epoch(self, test_acc):
        """Find grokking epoch (test acc > 90%)."""
        return np.argmax(np.array(test_acc) > 0.9)

    def generate_grokking_curves(self):
        """Compare grokking curves across all runs."""
        print("\nGenerating grokking curves comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Single-head
        sh_data = self.results["single_head"]
        epochs_sh = range(1, len(sh_data["train_acc"]) + 1)
        grok_sh = self.grok_epoch(sh_data["test_acc"])

        ax = axes[0]
        ax.plot(epochs_sh, sh_data["train_acc"], label="Train Acc", color="blue", linewidth=2)
        ax.plot(epochs_sh, sh_data["test_acc"], label="Test Acc", color="green", linewidth=2)
        ax.axvline(x=grok_sh, color="red", linestyle="--", linewidth=2, label=f"Grok ({grok_sh})")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Single-Head Grokking Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Four-head runs overlay
        ax = axes[1]
        colors = ["red", "green", "blue"]
        for i, run_num in enumerate([1, 2, 3]):
            key = f"four_head_run_{run_num}"
            if key in self.results:
                data = self.results[key]
                epochs = range(1, len(data["test_acc"]) + 1)
                grok = self.grok_epoch(data["test_acc"])
                ax.plot(epochs, data["test_acc"], label=f"Run {run_num} (grok={grok})",
                       color=colors[i], linewidth=2, alpha=0.8)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Four-Head Runs: Grokking Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "01_grokking_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("  ✓ Saved: 01_grokking_curves.png")

    def generate_l2_norm_comparison(self):
        """Compare L2 Norm curves."""
        print("Generating L2 Norm comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Single-head
        sh_data = self.results["single_head"]
        epochs_sh = range(1, len(sh_data["l2_norm"]) + 1)

        ax = axes[0]
        ax.plot(epochs_sh, sh_data["l2_norm"], color="purple", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Single-Head L2 Norm")
        ax.grid(True, alpha=0.3)

        # Four-head runs
        ax = axes[1]
        colors = ["red", "green", "blue"]
        for i, run_num in enumerate([1, 2, 3]):
            key = f"four_head_run_{run_num}"
            if key in self.results:
                data = self.results[key]
                epochs = range(1, len(data["l2_norm"]) + 1)
                ax.plot(epochs, data["l2_norm"], label=f"Run {run_num}",
                       color=colors[i], linewidth=2, alpha=0.8)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Four-Head L2 Norm Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "02_l2_norm_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("  ✓ Saved: 02_l2_norm_comparison.png")

    def generate_dropout_comparison(self):
        """Compare Dropout gap curves."""
        print("Generating Dropout gap comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Single-head
        sh_data = self.results["single_head"]
        ax = axes[0]
        ax.plot(sh_data["dropout_gap_epochs"], sh_data["dropout_gap"],
               color="crimson", linewidth=2, label="p=0.9")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Dropout Gap")
        ax.set_title("Single-Head Dropout Gap")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Four-head runs
        ax = axes[1]
        colors = ["red", "green", "blue"]
        for i, run_num in enumerate([1, 2, 3]):
            key = f"four_head_run_{run_num}"
            if key in self.results:
                data = self.results[key]
                ax.plot(data["dropout_gap_epochs"], data["dropout_gap"],
                       label=f"Run {run_num}", color=colors[i], linewidth=2, alpha=0.8)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Dropout Gap")
        ax.set_title("Four-Head Dropout Gap Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "03_dropout_gap_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("  ✓ Saved: 03_dropout_gap_comparison.png")

    def generate_run_consistency_report(self):
        """Generate PDF report with run consistency metrics."""
        print("Generating run consistency report...")

        with PdfPages(self.output_dir / "benchmark_report.pdf") as pdf:

            # Page 1: Grokking epoch comparison
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            grok_epochs = []
            run_labels = []

            # Single-head
            grok_epochs.append(self.grok_epoch(self.results["single_head"]["test_acc"]))
            run_labels.append("Single-Head")

            # Four-head runs
            for run_num in [1, 2, 3]:
                key = f"four_head_run_{run_num}"
                if key in self.results:
                    grok = self.grok_epoch(self.results[key]["test_acc"])
                    grok_epochs.append(grok)
                    run_labels.append(f"4-Head Run {run_num}")

            colors = ["blue"] + ["orange", "green", "red"]
            bars = ax.bar(range(len(grok_epochs)), grok_epochs, color=colors, alpha=0.7, edgecolor="black", linewidth=2)

            # Add value labels on bars
            for bar, epoch in zip(bars, grok_epochs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(epoch)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_ylabel("Grokking Epoch", fontsize=12)
            ax.set_title("Grokking Epoch Comparison Across Runs", fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(grok_epochs)))
            ax.set_xticklabels(run_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: Loss curves
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # Single-head
            epochs_sh = range(1, len(self.results["single_head"]["loss"]) + 1)
            ax.plot(epochs_sh, self.results["single_head"]["loss"], label="Single-Head", linewidth=2)

            # Four-head runs
            for run_num in [1, 2, 3]:
                key = f"four_head_run_{run_num}"
                if key in self.results:
                    epochs = range(1, len(self.results[key]["loss"]) + 1)
                    ax.plot(epochs, self.results[key]["loss"], label=f"4-Head Run {run_num}",
                           linewidth=2, alpha=0.8)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch (log scale)", fontsize=12)
            ax.set_ylabel("Loss (log scale)", fontsize=12)
            ax.set_title("Loss Curves Comparison", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Run consistency statistics
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.axis('off')

            # Statistics text
            stats_text = "RUN CONSISTENCY ANALYSIS\n\n"

            four_head_groks = []
            for run_num in [1, 2, 3]:
                key = f"four_head_run_{run_num}"
                if key in self.results:
                    grok = self.grok_epoch(self.results[key]["test_acc"])
                    four_head_groks.append(grok)

            if four_head_groks:
                mean_grok = np.mean(four_head_groks)
                std_grok = np.std(four_head_groks)
                stats_text += f"Four-Head Grokking Epochs (3 runs):\n"
                for i, grok in enumerate(four_head_groks, 1):
                    stats_text += f"  Run {i}: {grok} epochs\n"
                stats_text += f"  Mean: {mean_grok:.1f} ± {std_grok:.1f} epochs\n"
                stats_text += f"  Consistency: {'High' if std_grok < 500 else 'Moderate' if std_grok < 1000 else 'Low'}\n\n"

            stats_text += "Interpretation:\n"
            stats_text += "  • Low std dev: predictor consistent across random seeds\n"
            stats_text += "  • High std dev: predictor sensitive to initialization\n\n"

            stats_text += "Next Steps:\n"
            stats_text += "  1. Validate L2 Norm & Dropout against 3-criteria protocol\n"
            stats_text += "  2. If both pass: move to Spectral predictor\n"
            stats_text += "  3. Build baseline benchmark with validated predictors\n"

            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            pdf.savefig(fig)
            plt.close(fig)

        print(f"  ✓ Saved: benchmark_report.pdf (3 pages)")

    def run(self):
        """Execute full analysis."""
        print("\n" + "="*70)
        print("BENCHMARK ANALYSIS: Results Visualization")
        print("="*70)

        if not self.load_single_head():
            print("\n❌ Cannot proceed: single-head baseline not found")
            return False

        if not self.load_four_head_runs():
            print("\n⚠ Warning: no four-head runs found")

        print("\nGenerating visualizations...")
        self.generate_grokking_curves()
        self.generate_l2_norm_comparison()
        self.generate_dropout_comparison()
        self.generate_run_consistency_report()

        print("\n" + "="*70)
        print(f"✓ ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nVisualizations saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  - 01_grokking_curves.png")
        print("  - 02_l2_norm_comparison.png")
        print("  - 03_dropout_gap_comparison.png")
        print("  - benchmark_report.pdf")
        print("\n" + "="*70 + "\n")

        return True


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("\n" + "="*70)
    print("GROKKING BENCHMARK: Full Experimental Pipeline")
    print("="*70)
    print("\nThis script will:")
    print("  1. Run single-head baseline training (1 run)")
    print("  2. Run four-head baseline training (3 runs with independent seeds)")
    print("  3. Analyze results and generate comparison charts + PDF report")
    print("\nTotal training runs: 4 (1 single-head + 3 four-head)")
    print("Expected time: ~1-2 hours depending on hardware")
    print("="*70)

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

    # Stage 3: Analysis
    print("\n" + "="*70)
    print("STAGE 3: Analysis & Visualization")
    print("="*70)

    analyzer = BenchmarkAnalyzer()
    if not analyzer.run():
        print("\n" + "="*70)
        print("PIPELINE WARNING: Analysis failed (training results still saved)")
        print("="*70)
        sys.exit(1)

    # Success
    print("\n" + "="*70)
    print("✓ FULL BENCHMARK PIPELINE COMPLETE")
    print("="*70)
    print("\nTraining results saved to:")
    print("  - results/single_head/ : single-head baseline")
    print("  - runs/four_head/run_1/ : four-head run 1")
    print("  - runs/four_head/run_2/ : four-head run 2")
    print("  - runs/four_head/run_3/ : four-head run 3")
    print("\nAnalysis saved to:")
    print("  - benchmark_analysis/ : comparison charts + PDF report")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
