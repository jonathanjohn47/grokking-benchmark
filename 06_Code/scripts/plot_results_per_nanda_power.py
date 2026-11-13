#!/usr/bin/env python3
"""
Figures for the finished 40,000-epoch retrain in
08_Experiments/results/results_per_nanda_power.

Reads only the saved json and npy files. Does not load checkpoints and does
not invent series. Training histories are one sample per loop index
(0 .. epochs-1), which is the same index stored as grok_epoch. The L2 moving
averages were computed on l2_norm/epoch_grid.npy (uniform in log10 epoch);
detection epochs from the json are positions on that grid and are drawn at
those epoch values.

Usage (from the repo root):
    python 06_Code/scripts/plot_results_per_nanda_power.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS = os.path.join(
    REPO_ROOT, "08_Experiments", "results", "results_per_nanda_power"
)

GROK_STYLE = {
    "0.99": {"color": "#b00020", "ls": "-", "lw": 1.6, "label": "Grok cut 0.99 (primary)"},
    "0.95": {"color": "#e07a00", "ls": "--", "lw": 1.2, "label": "Grok cut 0.95"},
    "0.9": {"color": "#1f4e79", "ls": ":", "lw": 1.4, "label": "Grok cut 0.9"},
}
CUT_ORDER = ("0.99", "0.95", "0.9")


def results_dir():
    return os.path.abspath(DEFAULT_RESULTS)


def figures_dir(root):
    path = os.path.join(root, "figures")
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def epoch_index(n):
    """Loop index stored with each training history (0 .. n-1)."""
    return np.arange(n, dtype=float)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "-",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })


def mark_grok(ax, grok_by_threshold, already=None):
    labelled = already if already is not None else set()
    for cut in CUT_ORDER:
        epoch = grok_by_threshold.get(cut)
        if epoch is None:
            continue
        style = GROK_STYLE[cut]
        label = style["label"] if cut not in labelled else None
        ax.axvline(float(epoch), color=style["color"], ls=style["ls"], lw=style["lw"], label=label, zorder=3)
        labelled.add(cut)
    return labelled


def mark_named(ax, epoch, color, label, labelled):
    if epoch is None:
        return
    show = label if label not in labelled else None
    ax.axvline(float(epoch), color=color, ls="-.", lw=1.15, label=show, zorder=3)
    labelled.add(label)


def log_x(ax, n_epochs):
    ax.set_xscale("log")
    ax.set_xlim(1, max(n_epochs - 1, 2))
    ax.set_xlabel("Epoch index (0-based, as stored; log scale, epoch 0 off the left edge)")


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(path)


def load_seed(root, seed):
    seed_dir = os.path.join(root, f"seed_{seed}")
    summary = load_json(os.path.join(seed_dir, "summary.json"))
    train = os.path.join(seed_dir, "training")
    l2 = os.path.join(seed_dir, "l2_norm")
    drop = os.path.join(seed_dir, "dropout")
    return {
        "seed": seed,
        "summary": summary,
        "epochs": epoch_index(len(np.load(os.path.join(train, "train_acc_history.npy")))),
        "train_acc": np.load(os.path.join(train, "train_acc_history.npy")),
        "test_acc": np.load(os.path.join(train, "test_acc_history.npy")),
        "train_loss": np.load(os.path.join(train, "loss_history.npy")),
        "test_loss": np.load(os.path.join(train, "test_loss_history.npy")),
        "l2": np.load(os.path.join(l2, "l2_norm_history.npy")),
        "sum_w2": np.load(os.path.join(l2, "sum_w2_history.npy")),
        "epoch_grid": np.load(os.path.join(l2, "epoch_grid.npy")),
        "var_epochs": np.load(os.path.join(drop, "dropout_variance_checkpoints.npy")),
        "var": np.load(os.path.join(drop, "dropout_variance_history.npy")),
        "var_mean_acc": np.load(os.path.join(drop, "dropout_variance_mean_acc_history.npy")),
    }


def plot_accuracy(run, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = run["epochs"]
    ax.plot(x, run["train_acc"], color="#1b7f4e", lw=1.3, label="Train accuracy")
    ax.plot(x, run["test_acc"], color="#1f4e79", lw=1.3, label="Test accuracy")
    labelled = mark_grok(ax, run["summary"]["grok_epochs_by_threshold"])
    l2 = run["summary"]["l2_predictor"] or {}
    dv = run["summary"]["dropout_variance_predictor"] or {}
    mark_named(ax, l2.get("ma_crossover_epoch"), "#6a3d9a", "L2 MA crossover", labelled)
    mark_named(ax, l2.get("ma_of_ma_zero_crossing_epoch"), "#8c6bb1", "L2 MA-of-MA zero crossing", labelled)
    mark_named(ax, dv.get("variance_peak_epoch"), "#333333", "Dropout variance peak", labelled)
    log_x(ax, len(x))
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Seed {run['seed']}: train and test accuracy, modular addition mod 113")
    ax.legend(loc="upper left", framealpha=0.95)
    save(fig, path)


def plot_loss(run, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = run["epochs"]
    ax.plot(x, run["train_loss"], color="#1b7f4e", lw=1.2, label="Train loss")
    ax.plot(x, run["test_loss"], color="#1f4e79", lw=1.2, label="Test loss")
    mark_grok(ax, run["summary"]["grok_epochs_by_threshold"])
    ax.set_yscale("log")
    log_x(ax, len(x))
    ax.set_ylabel("Cross-entropy loss (log scale)")
    ax.set_title(f"Seed {run['seed']}: train and test loss")
    ax.legend(loc="center left", framealpha=0.95)
    save(fig, path)


def plot_l2(run, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = run["epochs"]
    ax.plot(x, run["l2"], color="#1f4e79", lw=1.3, label=r"Parameter L2 norm $\|w\|_2$")
    ax2 = ax.twinx()
    ax2.plot(x, run["sum_w2"], color="#1b7f4e", lw=1.1, alpha=0.9, label=r"Sum of squared weights $\sum w^2$")
    ax2.grid(False)
    labelled = mark_grok(ax, run["summary"]["grok_epochs_by_threshold"])
    l2 = run["summary"]["l2_predictor"] or {}
    mark_named(ax, l2.get("ma_crossover_epoch"), "#6a3d9a", "L2 MA crossover", labelled)
    mark_named(ax, l2.get("ma_of_ma_zero_crossing_epoch"), "#8c6bb1", "L2 MA-of-MA zero crossing", labelled)
    log_x(ax, len(x))
    ax.set_ylabel(r"$\|w\|_2$")
    ax2.set_ylabel(r"$\sum w^2$")
    ax.set_title(f"Seed {run['seed']}: weight norm (per-epoch history, not the log-uniform MA grid)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", framealpha=0.95)
    save(fig, path)


def plot_dropout(run, path):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    epochs = run["var_epochs"].astype(float)
    # Checkpoint 0 is in the file. A log axis cannot show it, so the line starts at the next saved checkpoint.
    keep = epochs >= 1
    ax.plot(epochs[keep], run["var"][keep], color="#6a3d9a", lw=1.4, marker="o", ms=2.2, label="Accuracy variance across 100 dropout samples (rate 0.5)")
    labelled = mark_grok(ax, run["summary"]["grok_epochs_by_threshold"])
    dv = run["summary"]["dropout_variance_predictor"] or {}
    mark_named(ax, dv.get("variance_peak_epoch"), "#333333", "Dropout variance peak", labelled)
    ax.set_xscale("log")
    positive = epochs[epochs >= 1]
    if len(positive):
        ax.set_xlim(float(positive.min()), float(max(run["epochs"][-1], positive.max())))
    ax.set_xlabel("Checkpoint epoch (saved dropout-variance grid)")
    ax.set_ylabel("Variance of test accuracy")
    ax.set_title(f"Seed {run['seed']}: dropout variance signal")
    ax.legend(loc="best", framealpha=0.95)
    save(fig, path)


def plot_overlay(runs, path):
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    colours = ["#1f4e79", "#b00020", "#1b7f4e", "#e07a00", "#6a3d9a"]
    for run, colour in zip(runs, colours):
        ax.plot(run["epochs"], run["test_acc"], color=colour, lw=1.35, label=f"Seed {run['seed']}")
        g = run["summary"]["grok_epochs_by_threshold"].get("0.99")
        if g is not None:
            ax.scatter([g], [run["test_acc"][int(g)]], color=colour, s=28, zorder=4)
    ax.axhline(0.99, color="#b00020", ls="--", lw=0.9, label="Accuracy 0.99")
    log_x(ax, len(runs[0]["epochs"]))
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("All seeds: test accuracy (dots mark the stored 0.99 grok index)")
    ax.legend(loc="upper left", framealpha=0.95)
    save(fig, path)


def plot_grok_bars(aggregate, path):
    by = aggregate["grok_epochs_by_threshold"]
    seeds = [row["seed"] for row in aggregate["seeds"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(seeds), dtype=float)
    width = 0.24
    for i, cut in enumerate(CUT_ORDER):
        values = by[cut]["per_seed"]
        style = GROK_STYLE[cut]
        ax.bar(x + (i - 1) * width, values, width=width, color=style["color"], label=style["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Grok epoch (stored index)")
    ax.set_title("Grok epoch by seed and accuracy cut")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_axisbelow(True)
    save(fig, path)


def main():
    apply_style()
    root = results_dir()
    out = figures_dir(root)
    aggregate = load_json(os.path.join(root, "aggregate.json"))
    runs = [load_seed(root, row["seed"]) for row in aggregate["seeds"]]
    for run in runs:
        seed = run["seed"]
        if len(run["l2"]) != len(run["epochs"]):
            raise SystemExit(f"seed {seed}: l2 history length does not match training history")
        if len(run["epoch_grid"]) != len(run["l2"]):
            raise SystemExit(f"seed {seed}: epoch_grid length does not match l2 history")
        if len(run["var"]) != len(run["var_epochs"]):
            raise SystemExit(f"seed {seed}: dropout variance length does not match its checkpoints")
        plot_accuracy(run, os.path.join(out, f"seed_{seed}_train_test_accuracy.png"))
        plot_loss(run, os.path.join(out, f"seed_{seed}_train_test_loss.png"))
        plot_l2(run, os.path.join(out, f"seed_{seed}_l2_norm.png"))
        plot_dropout(run, os.path.join(out, f"seed_{seed}_dropout_variance.png"))
    plot_overlay(runs, os.path.join(out, "overview_test_accuracy.png"))
    plot_grok_bars(aggregate, os.path.join(out, "overview_grok_epochs.png"))


if __name__ == "__main__":
    main()
