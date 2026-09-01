#!/usr/bin/env python3
"""
Four-head benchmark pipeline: trains the four-head model several times with
independent seeds, then analyzes and visualizes the results. Single command,
single script.

Stage 1: Four-head baseline training (3 independent runs, different seeds)
Stage 2: Analysis — comparison charts + PDF report across the runs

RESUMABLE: this script can be stopped at any point (Ctrl-C, crash, machine
sleep) and re-run. Finished runs are detected and kept, a half-built run
folder from the interrupted attempt is cleared, and the analysis step is
regenerated only when a new run was actually produced. Nothing already
computed is computed again.

Single-head is no longer part of this pipeline. The four-head model is the
faithful Nanda et al. architecture (d_model=128, 4 attention heads), so the
benchmark stands on that alone. The single-head code has been moved to
archive/single_head/ and can be brought back if a later robustness check
needs it. See context.md for the reasoning.
"""

import re
import shutil
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

FOUR_HEAD_BASE = Path("runs/four_head")

# A run_N/ folder counts as "finished" only when every one of these is on
# disk. train_four_head.py writes them all right at the end (after all
# epochs), and reports/training_report.pdf is the very last artifact, so
# its presence means the run truly completed. Anything less = the run was
# interrupted (Ctrl-C, crash, machine sleep, ...) and must be redone.
REQUIRED_RUN_FILES = [
    "training/train_acc_history.npy",
    "training/test_acc_history.npy",
    "training/loss_history.npy",
    "l2_norm/l2_norm_history.npy",
    "dropout/dropout_gap_epochs.npy",
    "dropout/dropout_gap_by_rate.npy",
    "dropout/dropout_rates.npy",
    "reports/training_report.pdf",
]

ANALYSIS_OUTPUTS = [
    "01_grokking_curves.png",
    "02_l2_norm_comparison.png",
    "03_dropout_gap_comparison.png",
    "benchmark_report.pdf",
]


def _run_dirs():
    """All run_<N> dirs under runs/four_head/, sorted by number."""
    if not FOUR_HEAD_BASE.is_dir():
        return []
    dirs = [d for d in FOUR_HEAD_BASE.iterdir()
            if d.is_dir() and re.fullmatch(r"run_\d+", d.name)]
    return sorted(dirs, key=lambda d: int(d.name[4:]))


def _run_is_complete(run_dir):
    return all((run_dir / rel).exists() for rel in REQUIRED_RUN_FILES)


def prepare_four_head_dir():
    """Make runs/four_head/ safe to resume into, and return how many runs
    are already fully finished.

    A stopped run leaves a half-built run_N/ folder (usually just seed.npy
    plus empty sub-folders — training does not checkpoint mid-way, so there
    is nothing worth keeping there). Such folders are deleted here so that:
      * they are never miscounted as a finished run, and
      * train_four_head.py's own run-numbering picks the right next number
        instead of skipping over the dead folder.

    Finished runs are produced strictly in order and the pipeline aborts on
    any failure, so after this cleanup the surviving run_N/ folders are a
    contiguous 1..k and the next run is k+1.
    """
    complete = 0
    for d in _run_dirs():
        if _run_is_complete(d):
            complete += 1
        else:
            print(f"  ↻ clearing incomplete run folder (interrupted / did not finish): {d}")
            shutil.rmtree(d)
    return complete


def _analysis_outputs_present():
    return (Path("benchmark_analysis").is_dir()
            and all((Path("benchmark_analysis") / f).exists() for f in ANALYSIS_OUTPUTS))


def run_four_head(run_num):
    """Run one four-head training session, then verify it really finished."""
    print("\n" + "="*70)
    print(f"STAGE 1.{run_num}: Four-Head Baseline Training (Run {run_num})")
    print("="*70)
    print("Command: python src/train_four_head.py")
    print(f"Expected output: runs/four_head/run_{run_num}/{{training,l2_norm,dropout,reports}}/")
    print("="*70 + "\n")

    result = subprocess.run([sys.executable, "src/train_four_head.py"], cwd=".")

    if result.returncode != 0:
        print(f"\n❌ Four-head run {run_num} FAILED (exit code {result.returncode})")
        return False

    produced = _run_dirs()[-1] if _run_dirs() else None
    if produced is None or not _run_is_complete(produced):
        print(f"\n❌ Four-head run {run_num}: process exited cleanly but its output "
              f"folder is incomplete ({produced}). Treating as a failed run.")
        return False

    print(f"\n✓ Four-head run {run_num} complete ({produced.name})")
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
                "dropout_gap_by_rate": np.load(dropout_dir / "dropout_gap_by_rate.npy"),
                "dropout_rates": np.load(dropout_dir / "dropout_rates.npy"),
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
        """Dropout gap: full multi-rate sweep, one panel per four-head run.

        dropout_gap_by_rate is indexed [rate_index, epoch_index], with
        rate_index following dropout_rates.
        """
        print("Generating Dropout gap sweep comparison...")

        keys = self.run_keys()
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

        for col, key in enumerate(keys):
            data = self.results[key]
            epochs = data["dropout_gap_epochs"]
            rates = data["dropout_rates"]
            gap_by_rate = data["dropout_gap_by_rate"]
            ax = axes[0][col]
            for r_idx, rate in enumerate(rates):
                ax.plot(epochs, gap_by_rate[r_idx], linewidth=1.8, alpha=0.9,
                        label=f"p={rate:g}")
            ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("Epoch (log scale)")
            ax.set_ylabel("Dropout Gap")
            ax.set_title(f"{key.replace('four_head_', '').replace('_', ' ').title()} — rate sweep")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Four-Head Dropout Gap: Multi-Rate Sweep", fontsize=13)
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

    print("\nResume mode: finished runs are kept, interrupted run folders are")
    print("cleared, and analysis is only redone when something new was produced.")

    # Stage 1: Four-head — resume up to TARGET_FOUR_HEAD_RUNS.
    already_done = prepare_four_head_dir()
    if already_done >= TARGET_FOUR_HEAD_RUNS:
        print("\n" + "="*70)
        print(f"STAGE 1: all {TARGET_FOUR_HEAD_RUNS} four-head runs already complete — nothing to train")
        print("="*70)
    elif already_done:
        print("\n" + "="*70)
        print(f"STAGE 1: {already_done} run(s) complete — resuming from run {already_done + 1}")
        print("="*70)

    newly_run = 0
    for run_num in range(already_done + 1, TARGET_FOUR_HEAD_RUNS + 1):
        if not run_four_head(run_num):
            print("\n" + "="*70)
            print(f"PIPELINE STOPPED at run {run_num}. Re-run this script to resume "
                  f"from here — completed runs will not be repeated.")
            print("="*70)
            sys.exit(1)
        newly_run += 1

    # Stage 2: Analysis — skip if nothing new and the charts already exist.
    print("\n" + "="*70)
    print("STAGE 2: Analysis & Visualization")
    print("="*70)

    if newly_run == 0 and _analysis_outputs_present():
        print("Nothing new since last run and benchmark_analysis/ is already "
              "populated — skipping analysis regeneration.")
    else:
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
