#!/usr/bin/env python3
"""
Four-head benchmark pipeline: trains the four-head model several times with
independent seeds, then analyzes and visualizes the results. Single command,
single script.

Stage 1: Four-head baseline training (3 independent runs, different seeds)
Stage 2: Analysis — comparison charts + PDF report across the runs

Single-head is no longer part of this pipeline. The four-head model is the
faithful Nanda et al. architecture (d_model=128, 4 attention heads), so the
benchmark stands on that alone. The single-head code has been moved to
archive/single_head/ and can be brought back if a later robustness check
needs it. See context.md for the reasoning.
"""

import re
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# ======================================================================
# STAGE 1: TRAINING (four-head, N seeded runs)
# ======================================================================

TARGET_FOUR_HEAD_RUNS = 3


def completed_four_head_runs():
    """Count run_N/ dirs under runs/four_head/ that have finished training."""
    base = Path("runs/four_head")
    if not base.is_dir():
        return 0
    count = 0
    for d in base.iterdir():
        if (d.is_dir() and re.fullmatch(r"run_\d+", d.name)
                and (d / "training" / "train_acc_history.npy").exists()):
            count += 1
    return count


def run_four_head(run_num):
    """Run one four-head training session."""
    print("\n" + "="*70)
    print(f"STAGE 1.{run_num}: Four-Head Baseline Training (Run {run_num})")
    print("="*70)
    print("Command: python src/train_four_head.py")
    print(f"Expected output: runs/four_head/run_{run_num}/{{training,l2_norm,dropout,reports}}/")
    print("="*70 + "\n")

    result = subprocess.run([sys.executable, "src/train_four_head.py"], cwd=".")

    if result.returncode != 0:
        print(f"\n❌ Four-head run {run_num} FAILED")
        return False

    print(f"\n✓ Four-head run {run_num} complete")
    return True


# ======================================================================
# STAGE 2: ANALYSIS & VISUALIZATION
# ======================================================================

class BenchmarkAnalyzer:
    def __init__(self):
        self.four_head_base = Path("runs/four_head")
        self.output_dir = Path("benchmark_analysis")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.run_colors = ["red", "green", "blue"]

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

        return len(self.results) >= 1

    def run_keys(self):
        """Ordered keys of the four-head runs that actually loaded."""
        return [f"four_head_run_{n}" for n in [1, 2, 3]
                if f"four_head_run_{n}" in self.results]

    def grok_epoch(self, test_acc):
        """Find grokking epoch (test acc > 90%)."""
        return np.argmax(np.array(test_acc) > 0.9)

    def generate_grokking_curves(self):
        """Overlay grokking curves across the four-head runs."""
        print("\nGenerating grokking curves comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, run_num in enumerate([1, 2, 3]):
            key = f"four_head_run_{run_num}"
            if key not in self.results:
                continue
            data = self.results[key]
            epochs = range(1, len(data["test_acc"]) + 1)
            grok = self.grok_epoch(data["test_acc"])
            color = self.run_colors[i]
            ax.plot(epochs, data["test_acc"], label=f"Run {run_num} test acc (grok={grok})",
                    color=color, linewidth=2, alpha=0.85)
            ax.plot(epochs, data["train_acc"], color=color, linewidth=1,
                    alpha=0.35, linestyle="--")
            if grok > 0:
                ax.axvline(x=grok, color=color, linestyle=":", linewidth=1.5, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Four-Head Runs: Grokking Comparison\n(solid = test, dashed = train, dotted = grok epoch)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "01_grokking_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("  ✓ Saved: 01_grokking_curves.png")

    def generate_l2_norm_comparison(self):
        """Overlay L2 Norm curves across the four-head runs."""
        print("Generating L2 Norm comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, run_num in enumerate([1, 2, 3]):
            key = f"four_head_run_{run_num}"
            if key not in self.results:
                continue
            data = self.results[key]
            epochs = range(1, len(data["l2_norm"]) + 1)
            ax.plot(epochs, data["l2_norm"], label=f"Run {run_num}",
                    color=self.run_colors[i], linewidth=2, alpha=0.85)

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
        """Overlay Dropout gap curves (p=0.9) across the four-head runs."""
        print("Generating Dropout gap comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, run_num in enumerate([1, 2, 3]):
            key = f"four_head_run_{run_num}"
            if key not in self.results:
                continue
            data = self.results[key]
            ax.plot(data["dropout_gap_epochs"], data["dropout_gap"],
                    label=f"Run {run_num}", color=self.run_colors[i], linewidth=2, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Dropout Gap (p=0.9)")
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

            # Page 1: Grokking epoch per run + mean line
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            grok_epochs = []
            run_labels = []
            for run_num in [1, 2, 3]:
                key = f"four_head_run_{run_num}"
                if key in self.results:
                    grok_epochs.append(self.grok_epoch(self.results[key]["test_acc"]))
                    run_labels.append(f"4-Head Run {run_num}")

            bars = ax.bar(range(len(grok_epochs)), grok_epochs,
                          color=self.run_colors[:len(grok_epochs)],
                          alpha=0.7, edgecolor="black", linewidth=2)

            for bar, epoch in zip(bars, grok_epochs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(epoch)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

            if len(grok_epochs) >= 2:
                mean_grok = float(np.mean(grok_epochs))
                std_grok = float(np.std(grok_epochs))
                ax.axhline(y=mean_grok, color="black", linestyle="--", linewidth=1.5,
                           label=f"Mean {mean_grok:.0f} ± {std_grok:.0f}")
                ax.legend()

            ax.set_ylabel("Grokking Epoch", fontsize=12)
            ax.set_title("Four-Head Grokking Epoch Across Seeded Runs", fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(grok_epochs)))
            ax.set_xticklabels(run_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: Loss curves
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            for i, run_num in enumerate([1, 2, 3]):
                key = f"four_head_run_{run_num}"
                if key in self.results:
                    epochs = range(1, len(self.results[key]["loss"]) + 1)
                    ax.plot(epochs, self.results[key]["loss"], label=f"4-Head Run {run_num}",
                            color=self.run_colors[i], linewidth=2, alpha=0.85)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch (log scale)", fontsize=12)
            ax.set_ylabel("Loss (log scale)", fontsize=12)
            ax.set_title("Four-Head Loss Curves", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Run consistency statistics
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.axis('off')

            stats_text = "FOUR-HEAD RUN CONSISTENCY ANALYSIS\n\n"

            four_head_groks = []
            for run_num in [1, 2, 3]:
                key = f"four_head_run_{run_num}"
                if key in self.results:
                    four_head_groks.append(self.grok_epoch(self.results[key]["test_acc"]))

            if four_head_groks:
                mean_grok = np.mean(four_head_groks)
                std_grok = np.std(four_head_groks)
                stats_text += f"Grokking Epochs ({len(four_head_groks)} seeded runs):\n"
                for i, grok in enumerate(four_head_groks, 1):
                    stats_text += f"  Run {i}: {grok} epochs\n"
                stats_text += f"  Mean: {mean_grok:.1f} ± {std_grok:.1f} epochs\n"
                stats_text += f"  Consistency: {'High' if std_grok < 500 else 'Moderate' if std_grok < 1000 else 'Low'}\n\n"

            stats_text += "Interpretation:\n"
            stats_text += "  • Low std dev: predictor signal consistent across random seeds\n"
            stats_text += "  • High std dev: predictor signal sensitive to initialization\n\n"

            stats_text += "Next Steps:\n"
            stats_text += "  1. Validate L2 Norm & Dropout against the 3-criteria protocol\n"
            stats_text += "  2. If both pass: move to the Spectral predictor\n"
            stats_text += "  3. Build the baseline benchmark with the validated predictors\n"

            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            pdf.savefig(fig)
            plt.close(fig)

        print("  ✓ Saved: benchmark_report.pdf (3 pages)")

    def run(self):
        """Execute full analysis."""
        print("\n" + "="*70)
        print("BENCHMARK ANALYSIS: Four-Head Results Visualization")
        print("="*70)

        if not self.load_four_head_runs():
            print("\n❌ Cannot proceed: no four-head runs found under runs/four_head/")
            return False

        print("\nGenerating visualizations...")
        self.generate_grokking_curves()
        self.generate_l2_norm_comparison()
        self.generate_dropout_comparison()
        self.generate_run_consistency_report()

        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
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
    print("GROKKING BENCHMARK: Four-Head Experimental Pipeline")
    print("="*70)
    print("\nThis script will:")
    print(f"  1. Run four-head baseline training ({TARGET_FOUR_HEAD_RUNS} runs with independent seeds)")
    print("  2. Analyze results and generate comparison charts + PDF report")
    print(f"\nTotal training runs: {TARGET_FOUR_HEAD_RUNS} (four-head only)")
    print("Expected time: depends on hardware")
    print("="*70)

    print("\nResuming: four-head runs that already finished are kept, not re-run.")

    # Stage 1: Four-head — top up to TARGET_FOUR_HEAD_RUNS, keeping finished runs.
    # train_four_head.py picks its own next run_N, so we just invoke it the
    # remaining number of times.
    already_done = completed_four_head_runs()
    if already_done:
        print("\n" + "="*70)
        print(f"STAGE 1: {already_done} four-head run(s) already complete — keeping them")
        print("="*70)

    for run_num in range(already_done + 1, TARGET_FOUR_HEAD_RUNS + 1):
        if not run_four_head(run_num):
            print("\n" + "="*70)
            print(f"PIPELINE ABORTED: Four-head run {run_num} failed")
            print("="*70)
            sys.exit(1)

    # Stage 2: Analysis
    print("\n" + "="*70)
    print("STAGE 2: Analysis & Visualization")
    print("="*70)

    analyzer = BenchmarkAnalyzer()
    if not analyzer.run():
        print("\n" + "="*70)
        print("PIPELINE WARNING: Analysis failed (training results still saved)")
        print("="*70)
        sys.exit(1)

    # Success
    print("\n" + "="*70)
    print("✓ FOUR-HEAD BENCHMARK PIPELINE COMPLETE")
    print("="*70)
    print("\nTraining results saved to:")
    for run_num in range(1, TARGET_FOUR_HEAD_RUNS + 1):
        print(f"  - runs/four_head/run_{run_num}/")
    print("\nAnalysis saved to:")
    print("  - benchmark_analysis/ : comparison charts + PDF report")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
