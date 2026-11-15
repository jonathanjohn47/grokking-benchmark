#!/usr/bin/env python3
"""
Figures for the finished 40,000-epoch retrain in
08_Experiments/results/results_per_nanda_power.

Reads only the saved json and npy files. Does not load checkpoints and does
not invent series. Training histories are one sample per loop index
(0 .. epochs-1), which is the same index stored as grok_epoch. The L2
figure uses l2_norm/sum_w2_history.npy (total sum of squared weights) and
analysis/l2_predictor_results.json (raw firing epochs, grokking epochs,
Validity Gate metrics, and Superiority Index).

Usage (from the repo root):
    python 06_Code/scripts/plot_results_per_nanda_power.py
"""
import json
import os
import sys

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


def load_l2_predictor_results(root):
    """Load Validity Gate / Superiority Index payload for the L2 row.

    Expected keys:
      raw_firing_epochs, grokking_epochs,
      validity_gate_metrics (lead_fraction, correlation, noise_margin),
      validity_gate_verdict, superiority_index
    """
    path = os.path.join(root, "analysis", "l2_predictor_results.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"Missing {path}. Run analyze_nanda_unified.py --l2-only first."
        )
    data = load_json(path)
    required = (
        "raw_firing_epochs",
        "grokking_epochs",
        "validity_gate_metrics",
        "validity_gate_verdict",
        "superiority_index",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"{path} is missing keys: {missing}")
    metrics = data["validity_gate_metrics"]
    for key in ("lead_fraction", "correlation", "noise_margin"):
        if key not in metrics:
            raise SystemExit(f"{path}: validity_gate_metrics missing '{key}'")
    return data


def format_superiority_index(gate_pass, superiority_index):
    """Safe display string for SI; failed gate (null or 0.0) becomes N/A."""
    if not gate_pass:
        return "N/A - Gate Failed"
    if superiority_index is None:
        return "N/A - Gate Failed"
    try:
        return f"{float(superiority_index):.4f}"
    except (TypeError, ValueError):
        return "N/A - Gate Failed"


def format_metric(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def l2_eval_annotation(l2_results):
    """Plain-text box for Validity Gate + Superiority Index."""
    metrics = l2_results["validity_gate_metrics"]
    gate_pass = bool(l2_results["validity_gate_verdict"])
    verdict = "PASS" if gate_pass else "FAIL"
    si_text = format_superiority_index(
        gate_pass, l2_results["superiority_index"]
    )
    return (
        f"Validity Gate Verdict: {verdict}\n"
        f"Lead Fraction: {format_metric(metrics['lead_fraction'])}\n"
        f"Correlation: {format_metric(metrics['correlation'])}\n"
        f"Noise Margin: {format_metric(metrics['noise_margin'])}\n"
        f"Superiority Index: {si_text}"
    )


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
        "sum_w2": np.load(os.path.join(l2, "sum_w2_history.npy")),
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
    dv = run["summary"]["dropout_variance_predictor"] or {}
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


def plot_l2(run, path, l2_results):
    """L2 curve with raw firing / grokking epochs and Validity Gate box.

    Replaces the old Quality Index S annotations. Reads epochs and gate
    metrics from analysis/l2_predictor_results.json only.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = run["epochs"]
    series = run["sum_w2"]
    seed = int(run["seed"])
    ax.plot(
        x,
        series,
        color="#1b7f4e",
        lw=1.3,
        label=r"Sum of squared weights $\sum w^2$",
    )

    firing = l2_results["raw_firing_epochs"]
    grokking = l2_results["grokking_epochs"]
    if seed < 0 or seed >= len(firing) or seed >= len(grokking):
        raise SystemExit(
            f"seed {seed}: out of range for l2_predictor_results "
            f"(n_firing={len(firing)}, n_grok={len(grokking)})"
        )
    p_i = firing[seed]
    g_i = grokking[seed]

    labelled = set()
    mark_named(ax, p_i, "#6a3d9a", r"Raw firing epoch $P_i$", labelled)
    mark_named(ax, g_i, "#b00020", r"Grokking epoch $G_i$ (cut 0.99)", labelled)

    gate_pass = bool(l2_results["validity_gate_verdict"])
    verdict_colour = "#1b7f4e" if gate_pass else "#b00020"
    box = l2_eval_annotation(l2_results)
    ax.text(
        0.98,
        0.98,
        box,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": verdict_colour,
            "linewidth": 1.4,
            "alpha": 0.95,
        },
        zorder=5,
    )
    # Colour the PASS/FAIL word by drawing a small coloured tag in the title.
    verdict = "PASS" if gate_pass else "FAIL"
    ax.set_title(
        f"Seed {seed}: L2 Norm  |  Validity Gate: {verdict}",
        color=verdict_colour,
        fontweight="bold",
    )

    log_x(ax, len(x))
    ax.set_ylabel(r"$\sum w^2$")
    ax.legend(loc="upper left", framealpha=0.95)
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
    l2_results = load_l2_predictor_results(root)
    runs = [load_seed(root, row["seed"]) for row in aggregate["seeds"]]
    for run in runs:
        seed = run["seed"]
        if len(run["sum_w2"]) != len(run["epochs"]):
            raise SystemExit(f"seed {seed}: sum of squared weights length does not match training history")
        if len(run["var"]) != len(run["var_epochs"]):
            raise SystemExit(f"seed {seed}: dropout variance length does not match its checkpoints")
        plot_accuracy(run, os.path.join(out, f"seed_{seed}_train_test_accuracy.png"))
        plot_loss(run, os.path.join(out, f"seed_{seed}_train_test_loss.png"))
        plot_l2(run, os.path.join(out, f"seed_{seed}_l2_norm.png"), l2_results)
        plot_dropout(run, os.path.join(out, f"seed_{seed}_dropout_variance.png"))
    plot_overlay(runs, os.path.join(out, "overview_test_accuracy.png"))
    plot_grok_bars(aggregate, os.path.join(out, "overview_grok_epochs.png"))


if __name__ == "__main__":
    main()
