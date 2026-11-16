#!/usr/bin/env python3
"""
nanda_l2_p113 benchmark pipeline: trains the four-head model on (a + b) mod 113
several times with independent seeds, then analyzes and visualizes the L2-Norm
predictor results. Single command, single script.

Stage 1: training (3 independent seeded runs)
Stage 2: analysis — cross-run comparison charts + PDF report

This is a trimmed copy of run_full_benchmark.py. Two differences:
  * the task is (a + b) mod 113 with AdamW betas=(0.9, 0.98) — see train.py;
  * there is no Dropout predictor, so every Dropout load / plot / file is gone,
    and it does NOT shell out to src/generate_master_report.py (that tool is
    Dropout-aware and belongs to the main experiment).

RESUMABLE: stop this at any point (Ctrl-C, crash, sleep) and re-run. Finished
runs are detected and kept, a half-built run folder is cleared, and the
analysis step is regenerated only when a new run was produced.
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


PKG = Path(__file__).resolve().parent
RUNS_BASE = PKG / "runs"
ANALYSIS_DIR = PKG / "benchmark_analysis"
TRAIN_SCRIPT = PKG / "train.py"

TARGET_RUNS = 3

# A run_N/ folder counts as "finished" only when every one of these is on
# disk. train.py writes them all right at the end (after all epochs), and
# reports/training_report.pdf is the very last artifact, so its presence means
# the run truly completed. Anything less = interrupted, must be redone.
REQUIRED_RUN_FILES = [
    "training/train_acc_history.npy",
    "training/test_acc_history.npy",
    "training/loss_history.npy",
    "l2_norm/l2_norm_history.npy",
    "l2_norm/epoch_grid.npy",
    "reports/training_report.pdf",
]

ANALYSIS_OUTPUTS = [
    "01_grokking_curves.png",
    "02_l2_norm_comparison.png",
    "benchmark_report.pdf",
]


# ======================================================================
# STAGE 1: TRAINING (four-head, N seeded runs)
# ======================================================================

def _run_dirs():
    """All run_<N> dirs under runs/, sorted by number."""
    if not RUNS_BASE.is_dir():
        return []
    dirs = [d for d in RUNS_BASE.iterdir()
            if d.is_dir() and re.fullmatch(r"run_\d+", d.name)]
    return sorted(dirs, key=lambda d: int(d.name[4:]))


def _run_is_complete(run_dir):
    return all((run_dir / rel).exists() for rel in REQUIRED_RUN_FILES)


def prepare_runs_dir():
    """Make runs/ safe to resume into, and return how many runs are already
    fully finished.

    A stopped run leaves a half-built run_N/ folder (usually just seed.npy plus
    empty sub-folders — training does not checkpoint mid-way). Such folders are
    deleted here so they are never miscounted as finished and so train.py's own
    run-numbering picks the right next number.

    Finished runs are produced strictly in order and the pipeline aborts on any
    failure, so after this cleanup the surviving run_N/ folders are a contiguous
    1..k and the next run is k+1.
    """
    complete = 0
    for d in _run_dirs():
        if _run_is_complete(d):
            complete += 1
        else:
            print(f"  clearing incomplete run folder (interrupted / did not finish): {d}")
            shutil.rmtree(d)
    return complete


def _analysis_outputs_present():
    return (ANALYSIS_DIR.is_dir()
            and all((ANALYSIS_DIR / f).exists() for f in ANALYSIS_OUTPUTS))


def run_training(run_num):
    """Run one training session, then verify it really finished."""
    print("\n" + "="*70)
    print(f"STAGE 1.{run_num}: Four-Head Training on (a + b) mod 113 (Run {run_num})")
    print("="*70)
    print(f"Command: python {TRAIN_SCRIPT}")
    print(f"Expected output: nanda_l2_p113/runs/run_{run_num}/{{training,l2_norm,reports}}/")
    print("="*70 + "\n")

    result = subprocess.run([sys.executable, str(TRAIN_SCRIPT)])

    if result.returncode != 0:
        print(f"\nRun {run_num} FAILED (exit code {result.returncode})")
        return False

    produced = _run_dirs()[-1] if _run_dirs() else None
    if produced is None or not _run_is_complete(produced):
        print(f"\nRun {run_num}: process exited cleanly but its output folder is "
              f"incomplete ({produced}). Treating as a failed run.")
        return False

    print(f"\nRun {run_num} complete ({produced.name})")
    return True


# ======================================================================
# STAGE 2: ANALYSIS & VISUALIZATION
# ======================================================================

class BenchmarkAnalyzer:
    def __init__(self):
        self.output_dir = ANALYSIS_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.run_colors = ["red", "green", "blue"]

    def load_runs(self):
        print("Loading runs...")

        for run_num in range(1, TARGET_RUNS + 1):
            run_dir = RUNS_BASE / f"run_{run_num}"
            training_dir = run_dir / "training"
            l2_norm_dir = run_dir / "l2_norm"

            if not training_dir.exists():
                print(f"  Run {run_num} not found")
                continue

            self.results[f"run_{run_num}"] = {
                "train_acc": np.load(training_dir / "train_acc_history.npy"),
                "test_acc": np.load(training_dir / "test_acc_history.npy"),
                "loss": np.load(training_dir / "loss_history.npy"),
                "l2_norm": np.load(l2_norm_dir / "l2_norm_history.npy"),
            }

            print(f"  Loaded run {run_num} (epochs: {len(self.results[f'run_{run_num}']['train_acc'])})")

        return len(self.results) >= 1

    def run_keys(self):
        return [f"run_{n}" for n in range(1, TARGET_RUNS + 1) if f"run_{n}" in self.results]

    def grok_epoch(self, test_acc):
        return np.argmax(np.array(test_acc) > 0.9)

    def generate_grokking_curves(self):
        print("\nGenerating grokking curves comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, run_num in enumerate(range(1, TARGET_RUNS + 1)):
            key = f"run_{run_num}"
            if key not in self.results:
                continue
            data = self.results[key]
            epochs = range(1, len(data["test_acc"]) + 1)
            grok = self.grok_epoch(data["test_acc"])
            color = self.run_colors[i % len(self.run_colors)]
            ax.plot(epochs, data["test_acc"], label=f"Run {run_num} test acc (grok={grok})",
                    color=color, linewidth=2, alpha=0.85)
            ax.plot(epochs, data["train_acc"], color=color, linewidth=1,
                    alpha=0.35, linestyle="--")
            if grok > 0:
                ax.axvline(x=grok, color=color, linestyle=":", linewidth=1.5, alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_title("p=113 Runs: Grokking Comparison\n(solid = test, dashed = train, dotted = grok epoch)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "01_grokking_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("  Saved: 01_grokking_curves.png")

    def generate_l2_norm_comparison(self):
        print("Generating L2 Norm comparison...")

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, run_num in enumerate(range(1, TARGET_RUNS + 1)):
            key = f"run_{run_num}"
            if key not in self.results:
                continue
            data = self.results[key]
            epochs = range(1, len(data["l2_norm"]) + 1)
            ax.plot(epochs, data["l2_norm"], label=f"Run {run_num}",
                    color=self.run_colors[i % len(self.run_colors)], linewidth=2, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title("p=113 L2 Norm Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "02_l2_norm_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        print("  Saved: 02_l2_norm_comparison.png")

    def generate_run_consistency_report(self):
        print("Generating run consistency report...")

        with PdfPages(self.output_dir / "benchmark_report.pdf") as pdf:

            # Page 1: Grokking epoch per run + mean line
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            grok_epochs = []
            run_labels = []
            for run_num in range(1, TARGET_RUNS + 1):
                key = f"run_{run_num}"
                if key in self.results:
                    grok_epochs.append(self.grok_epoch(self.results[key]["test_acc"]))
                    run_labels.append(f"Run {run_num}")

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
                           label=f"Mean {mean_grok:.0f} +/- {std_grok:.0f}")
                ax.legend()

            ax.set_ylabel("Grokking Epoch", fontsize=12)
            ax.set_title("p=113 Grokking Epoch Across Seeded Runs", fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(grok_epochs)))
            ax.set_xticklabels(run_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: Loss curves
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            for i, run_num in enumerate(range(1, TARGET_RUNS + 1)):
                key = f"run_{run_num}"
                if key in self.results:
                    epochs = range(1, len(self.results[key]["loss"]) + 1)
                    ax.plot(epochs, self.results[key]["loss"], label=f"Run {run_num}",
                            color=self.run_colors[i % len(self.run_colors)], linewidth=2, alpha=0.85)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch (log scale)", fontsize=12)
            ax.set_ylabel("Loss (log scale)", fontsize=12)
            ax.set_title("p=113 Loss Curves", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: Run consistency statistics
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.axis('off')

            stats_text = "p=113 RUN CONSISTENCY ANALYSIS (L2-Norm predictor)\n\n"

            groks = []
            for run_num in range(1, TARGET_RUNS + 1):
                key = f"run_{run_num}"
                if key in self.results:
                    groks.append(self.grok_epoch(self.results[key]["test_acc"]))

            if groks:
                mean_grok = np.mean(groks)
                std_grok = np.std(groks)
                stats_text += f"Grokking Epochs ({len(groks)} seeded runs):\n"
                for i, grok in enumerate(groks, 1):
                    stats_text += f"  Run {i}: {grok} epochs\n"
                stats_text += f"  Mean: {mean_grok:.1f} +/- {std_grok:.1f} epochs\n"
                stats_text += f"  Consistency: {'High' if std_grok < 500 else 'Moderate' if std_grok < 1000 else 'Low'}\n\n"

            stats_text += "Setup:\n"
            stats_text += "  Task            : (a + b) mod 113\n"
            stats_text += "  Optimizer       : AdamW lr=1e-3 wd=1.0 betas=(0.9, 0.98)\n"
            stats_text += "  Model           : TransformerFourHead, 4 heads, d_model=128\n"
            stats_text += "  Predictor       : L2 Norm only\n\n"

            stats_text += "Interpretation:\n"
            stats_text += "  - Low std dev: predictor signal consistent across random seeds\n"
            stats_text += "  - High std dev: predictor signal sensitive to initialization\n\n"

            stats_text += "Next Steps:\n"
            stats_text += "  1. Overlay the L2 norm curves on Nanda Figure 7 (Literature/nanda_figures/)\n"
            stats_text += "  2. Check whether betas=(0.9, 0.98) shrank the pre-grok weight-norm spike\n"
            stats_text += "  3. Validate the L2 Norm predictor against the 3-criteria protocol\n"

            ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            pdf.savefig(fig)
            plt.close(fig)

        print("  Saved: benchmark_report.pdf (3 pages)")

    def run(self):
        print("\n" + "="*70)
        print("BENCHMARK ANALYSIS: p=113 L2-Norm Results Visualization")
        print("="*70)

        if not self.load_runs():
            print("\nCannot proceed: no runs found under nanda_l2_p113/runs/")
            return False

        print("\nGenerating visualizations...")
        self.generate_grokking_curves()
        self.generate_l2_norm_comparison()
        self.generate_run_consistency_report()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nVisualizations saved to: {self.output_dir}")
        print("\nGenerated files:")
        for f in ANALYSIS_OUTPUTS:
            print(f"  - {f}")
        print("\n" + "="*70 + "\n")

        return True


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("\n" + "="*70)
    print("GROKKING BENCHMARK: nanda_l2_p113 (L2-Norm only, p = 113)")
    print("="*70)
    print("\nThis script will:")
    print(f"  1. Run training ({TARGET_RUNS} runs with independent seeds)")
    print("  2. Analyze results and generate comparison charts + PDF report")
    print("="*70)

    print("\nResume mode: finished runs are kept, interrupted run folders are")
    print("cleared, and analysis is only redone when something new was produced.")

    # Stage 1: resume up to TARGET_RUNS.
    already_done = prepare_runs_dir()
    if already_done >= TARGET_RUNS:
        print("\n" + "="*70)
        print(f"STAGE 1: all {TARGET_RUNS} runs already complete — nothing to train")
        print("="*70)
    elif already_done:
        print("\n" + "="*70)
        print(f"STAGE 1: {already_done} run(s) complete — resuming from run {already_done + 1}")
        print("="*70)

    newly_run = 0
    for run_num in range(already_done + 1, TARGET_RUNS + 1):
        if not run_training(run_num):
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

    print("\n" + "="*70)
    print("nanda_l2_p113 BENCHMARK PIPELINE COMPLETE")
    print("="*70)
    print("\nTraining results saved to:")
    for run_num in range(1, TARGET_RUNS + 1):
        print(f"  - nanda_l2_p113/runs/run_{run_num}/")
    print("\nAnalysis saved to nanda_l2_p113/benchmark_analysis/ :")
    for f in ANALYSIS_OUTPUTS:
        print(f"  - {f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
