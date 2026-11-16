#!/usr/bin/env python3
"""Plot Salah & Yevick Dropout signal-extraction results.

Reads only the isolated MLP p=53 outputs written by
train_dropout_salah_yevick.py. Does not touch the Nanda-Unified
Transformer figures.

Usage:
    python3 06_Code/scripts/plot_dropout_salah_yevick.py
    python3 06_Code/scripts/plot_dropout_salah_yevick.py \\
        --results-dir 08_Experiments/results/dropout_salah_yevick
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "08_Experiments" / "results" / "dropout_salah_yevick"


def load_seed(seed_dir: Path):
    drop = seed_dir / "dropout"
    train = seed_dir / "training"
    signal_path = drop / "dropout_signal.json"
    summary_path = seed_dir / "summary.json"
    if not signal_path.is_file():
        raise FileNotFoundError(f"missing {signal_path}")

    with open(signal_path) as handle:
        signal = json.load(handle)
    summary = {}
    if summary_path.is_file():
        with open(summary_path) as handle:
            summary = json.load(handle)

    return {
        "seed": int(summary.get("seed", seed_dir.name.split("_")[-1])),
        "epochs": np.load(drop / "epochs.npy"),
        "variance": np.load(drop / "dropout_variance_history.npy"),
        "var_mean_acc": np.load(drop / "dropout_variance_mean_acc_history.npy"),
        "drc_rates": np.load(drop / "drc_rates.npy"),
        "drc": np.load(drop / "drc_test_acc_by_rate.npy"),  # [rate, epoch]
        "train_acc": np.load(train / "train_acc_history.npy"),
        "test_acc": np.load(train / "test_acc_history.npy"),
        "loss": np.load(train / "loss_history.npy"),
        "signal": signal,
        "summary": summary,
    }


def discover_seeds(results_dir: Path):
    seeds = []
    for path in sorted(results_dir.glob("seed_*")):
        if path.is_dir() and (path / "dropout" / "dropout_signal.json").is_file():
            seeds.append(path)
    return seeds


def mark_events(ax, grok_epoch, peak_epoch):
    if grok_epoch is not None:
        ax.axvline(
            grok_epoch, color="#b00020", linestyle="-", linewidth=1.4,
            label=f"Grok epoch G={grok_epoch}",
        )
    if peak_epoch is not None:
        ax.axvline(
            peak_epoch, color="#6a3d9a", linestyle="--", linewidth=1.4,
            label=f"Variance peak P={peak_epoch}",
        )


def plot_accuracy(run, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = run["epochs"]
    ax.plot(epochs, run["train_acc"], color="steelblue", lw=1.6, label="Train accuracy")
    ax.plot(epochs, run["test_acc"], color="seagreen", lw=1.6, label="Test accuracy")
    mark_events(
        ax,
        run["signal"].get("grok_epoch"),
        run["signal"].get("variance_peak_epoch"),
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Seed {run['seed']}: train vs test accuracy (MLP p=53)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_variance(run, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = run["epochs"]
    rate = run["signal"].get("variance_rate", 0.3)
    ax.plot(
        epochs, run["variance"], color="#6a3d9a", lw=1.5,
        label=f"Test-accuracy variance ({int(run['signal'].get('n_samples', 100))} passes, rate={rate})",
    )
    mark_events(
        ax,
        run["signal"].get("grok_epoch"),
        run["signal"].get("variance_peak_epoch"),
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Variance of test accuracy")
    ax.set_title(f"Seed {run['seed']}: Dropout Variance signal (Salah & Yevick)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drc(run, out_path, n_curves=6):
    """Dropout Robustness Curve: test accuracy vs dropout rate at selected epochs."""
    epochs = run["epochs"]
    rates = run["drc_rates"]
    drc = run["drc"]  # [n_rates, n_epochs]
    grok = run["signal"].get("grok_epoch")
    peak = run["signal"].get("variance_peak_epoch")

    # Pick a few epoch indices spanning training, always include grok/peak if present.
    n_epochs = len(epochs)
    wanted = set(np.linspace(0, n_epochs - 1, num=min(n_curves, n_epochs), dtype=int).tolist())
    for event in (grok, peak):
        if event is not None and 0 <= int(event) < n_epochs:
            wanted.add(int(event))
    chosen = sorted(wanted)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(chosen)))
    for color, idx in zip(cmap, chosen):
        ep = int(epochs[idx])
        tag = f"epoch {ep}"
        if grok is not None and ep == int(grok):
            tag += " (grok)"
        if peak is not None and ep == int(peak):
            tag += " (var peak)"
        ax.plot(rates, drc[:, idx], marker="o", ms=3.5, lw=1.5, color=color, label=tag)

    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Seed {run['seed']}: Dropout Robustness Curve (DRC)")
    ax.set_xlim(-0.02, 0.92)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_pdf(run, out_path):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_path) as pdf:
        # Page 1: accuracy
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(run["epochs"], run["train_acc"], color="steelblue", lw=1.6, label="Train")
        ax.plot(run["epochs"], run["test_acc"], color="seagreen", lw=1.6, label="Test")
        mark_events(ax, run["signal"].get("grok_epoch"), run["signal"].get("variance_peak_epoch"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Seed {run['seed']}: Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: variance
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(run["epochs"], run["variance"], color="#6a3d9a", lw=1.5)
        mark_events(ax, run["signal"].get("grok_epoch"), run["signal"].get("variance_peak_epoch"))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Variance of test accuracy")
        ax.set_title(f"Seed {run['seed']}: Dropout Variance (rate 0.3, 100 passes)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: DRC
        fig, ax = plt.subplots(figsize=(11, 6))
        epochs = run["epochs"]
        n_epochs = len(epochs)
        chosen = sorted(set(np.linspace(0, n_epochs - 1, num=min(6, n_epochs), dtype=int).tolist()))
        for idx in chosen:
            ax.plot(run["drc_rates"], run["drc"][:, idx], marker="o", ms=3, lw=1.4,
                    label=f"epoch {int(epochs[idx])}")
        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Seed {run['seed']}: Dropout Robustness Curve")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Salah & Yevick Dropout variance + DRC results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS),
        help="Directory containing seed_*/dropout/ outputs",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    seed_dirs = discover_seeds(results_dir)
    if not seed_dirs:
        raise SystemExit(f"No completed seeds found under {results_dir}")

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    print(f"Results dir: {results_dir}")
    print(f"Seeds: {[p.name for p in seed_dirs]}")

    for seed_dir in seed_dirs:
        run = load_seed(seed_dir)
        seed = run["seed"]
        paths = {
            "acc": fig_dir / f"seed_{seed}_accuracy.png",
            "var": fig_dir / f"seed_{seed}_dropout_variance.png",
            "drc": fig_dir / f"seed_{seed}_drc.png",
            "pdf": fig_dir / f"seed_{seed}_dropout_report.pdf",
        }
        plot_accuracy(run, paths["acc"])
        plot_variance(run, paths["var"])
        plot_drc(run, paths["drc"])
        plot_combined_pdf(run, paths["pdf"])
        for key, path in paths.items():
            written.append(str(path))
            print(f"[OK] seed {seed} {key}: {path}")

        sig = run["signal"]
        grok = sig.get("grok_epoch")
        peak = sig.get("variance_peak_epoch")
        ratio = None
        if grok and peak is not None and grok:
            ratio = peak / grok
        print(
            f"  metrics: variance_peak_epoch={peak}  grok_epoch={grok}  "
            f"peak/grok={ratio if ratio is not None else 'n/a'}  "
            f"peak_value={sig.get('variance_peak_value')}"
        )

    print(f"\nWrote {len(written)} figure files under {fig_dir}")


if __name__ == "__main__":
    main()
