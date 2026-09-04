#!/usr/bin/env python3
"""
plot_nanda_results.py
======================

Full plotting tool for a completed `run_nanda_benchmark.py` run (the
Nanda-Unified benchmark: (a+b) mod 113, L2-Norm + Dropout predictors,
N seeded runs under `results/nanda_unified/seed_{0..N-1}/`).

This is a NEW file — NOT a modification of
`archive/single_head/plot_results.py`, which is the frozen old
single-head snapshot and stays untouched (per every prior session's
decision on record in context.md). This script plots the current
four-head Nanda-Unified data instead.

For every seed it reads `summary.json` plus the raw per-epoch `.npy`
histories saved by `unified_measurements.PredictorMeasurements`
(`training/`, `l2_norm/`) — nothing is recomputed, everything already
exists on disk from the benchmark run.

Output (two forms of the same figures, as requested):
  1. Separate PNGs — one file per plot, per seed under
     <output_dir>/seed_{n}/, and cross-seed comparison plots under
     <output_dir>/comparison/.
  2. One combined PDF, `<output_dir>/nanda_results_report.pdf` — every
     one of those same figures as a page, plus text summary pages (the
     full numbers, human-readable, same content as make_pdf.py's
     RESULTS section) so the PDF alone is a complete human-readable
     report of the run.

Usage:
    python plot_nanda_results.py
    python plot_nanda_results.py --results_dir results/nanda_unified \
        --output_dir results/nanda_unified/plots
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Colours kept consistent with the rest of the project's plotting
# (unified_measurements.py / nanda_l2_p113/measurements.py conventions):
# train=steelblue, test=seagreen, L2/Σw²=purple, fast MA=blue,
# slow MA=orange, MA-of-MA diff=darkred, grok marker=green, trigger=red.
TRAIN_COLOR = "steelblue"
TEST_COLOR = "seagreen"
L2_COLOR = "purple"
FAST_MA_COLOR = "blue"
SLOW_MA_COLOR = "orange"
DIFF_COLOR = "darkred"
GROK_COLOR = "green"
TRIGGER_COLOR = "red"
LOSS_COLOR = "darkorange"


# ============================================================
# Data loading
# ============================================================

def discover_seeds(results_dir):
    """Return sorted seed indices for every seed_{n}/summary.json found."""
    seeds = []
    if not os.path.isdir(results_dir):
        return seeds
    for name in os.listdir(results_dir):
        if not name.startswith("seed_"):
            continue
        summary_path = os.path.join(results_dir, name, "summary.json")
        if os.path.isfile(summary_path):
            seeds.append(int(name.split("_", 1)[1]))
    return sorted(seeds)


def load_seed(results_dir, seed):
    """Load one seed's summary.json + raw per-epoch npy histories."""
    seed_dir = os.path.join(results_dir, f"seed_{seed}")
    training_dir = os.path.join(seed_dir, "training")
    l2_dir = os.path.join(seed_dir, "l2_norm")

    with open(os.path.join(seed_dir, "summary.json")) as handle:
        summary = json.load(handle)

    return {
        "seed": seed,
        "summary": summary,
        "train_acc": np.load(os.path.join(training_dir, "train_acc_history.npy")),
        "test_acc": np.load(os.path.join(training_dir, "test_acc_history.npy")),
        "loss": np.load(os.path.join(training_dir, "loss_history.npy")),
        "l2_norm": np.load(os.path.join(l2_dir, "l2_norm_history.npy")),
        "sum_w2": np.load(os.path.join(l2_dir, "sum_w2_history.npy")),
        "per_module_sum_w2": np.load(os.path.join(l2_dir, "per_module_sum_w2.npy")),
        "per_module_names": np.load(os.path.join(l2_dir, "per_module_sum_w2_names.npy")),
        "epoch_grid": np.load(os.path.join(l2_dir, "epoch_grid.npy")),
        "fast_ma": np.load(os.path.join(l2_dir, "fast_ma.npy")),
        "slow_ma": np.load(os.path.join(l2_dir, "slow_ma.npy")),
        "ma_of_ma_diff": np.load(os.path.join(l2_dir, "ma_of_ma_diff.npy")),
    }


def load_aggregate(results_dir, seeds_data):
    """Prefer aggregate.json; fall back to computing the same summary
    from the loaded per-seed summaries if it isn't there."""
    agg_path = os.path.join(results_dir, "aggregate.json")
    if os.path.isfile(agg_path):
        with open(agg_path) as handle:
            return json.load(handle)

    groks = [d["summary"]["grok_epoch"] for d in seeds_data
             if d["summary"]["grok_epoch"] is not None]
    n_limit_cycle = sum(
        1 for d in seeds_data
        if d["summary"]["limit_cycle_check"].get("limit_cycle")
    )
    return {
        "n_seeds": len(seeds_data),
        "epochs": seeds_data[0]["summary"]["epochs"] if seeds_data else None,
        "grok_epoch_mean": float(np.mean(groks)) if groks else None,
        "grok_epoch_std": float(np.std(groks)) if groks else None,
        "grok_epochs": groks,
        "n_limit_cycle": n_limit_cycle,
        "seeds": [d["summary"] for d in seeds_data],
    }


# ============================================================
# Small helpers
# ============================================================

def new_fig():
    return plt.subplots(figsize=(12, 7))


def mark_grok(ax, grok_epoch, color=GROK_COLOR):
    if grok_epoch is not None:
        ax.axvline(x=grok_epoch, color=color, linestyle=":", linewidth=2,
                    label=f"Grok epoch ({grok_epoch})")


def fmt_epoch(value):
    return "None" if value is None else f"{value:.2f}"


class Collector:
    """Saves every figure both as a standalone PNG (under a per-section
    subdirectory) and as a page in the shared combined PDF, then closes
    it. One object per output run."""

    def __init__(self, output_dir, pdf):
        self.output_dir = output_dir
        self.pdf = pdf
        self.count = 0

    def add(self, subdir, filename, fig):
        section_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(section_dir, exist_ok=True)
        png_path = os.path.join(section_dir, filename)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        self.pdf.savefig(fig)
        plt.close(fig)
        self.count += 1
        print(f"  [{self.count}] {os.path.relpath(png_path, self.output_dir)}")

    def add_text_page(self, subdir, filename, lines, title=None, fontsize=8):
        fig, ax = new_fig()
        fig.set_size_inches(8.5, 11)
        ax.axis("off")
        y = 0.97
        if title:
            ax.text(0.03, y, title, fontsize=14, fontfamily="monospace",
                     fontweight="bold", va="top", transform=ax.transAxes)
            y -= 0.035
        ax.text(0.03, y, "\n".join(str(line) for line in lines), fontsize=fontsize,
                 fontfamily="monospace", va="top", transform=ax.transAxes)
        self.add(subdir, filename, fig)


# ============================================================
# Per-seed plots
# ============================================================

def plot_grokking_curve(d):
    s = d["summary"]
    fig, ax = new_fig()
    epochs_axis = range(1, len(d["train_acc"]) + 1)
    ax.plot(epochs_axis, d["train_acc"], label="Train Accuracy", color=TRAIN_COLOR, linewidth=2)
    ax.plot(epochs_axis, d["test_acc"], label="Test Accuracy", color=TEST_COLOR, linewidth=2)
    mark_grok(ax, s["grok_epoch"])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"seed {d['seed']} — Grokking Curve (grok epoch {s['grok_epoch']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_loss_curve(d):
    s = d["summary"]
    fig, ax = new_fig()
    epochs_axis = range(1, len(d["loss"]) + 1)
    ax.plot(epochs_axis, d["loss"], color=LOSS_COLOR, linewidth=2)
    mark_grok(ax, s["grok_epoch"])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title(f"seed {d['seed']} — Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_l2_norm_curve(d):
    s = d["summary"]
    lp = s["l2_predictor"]
    fig, ax = new_fig()
    epochs_axis = range(1, len(d["l2_norm"]) + 1)
    ax.plot(epochs_axis, d["l2_norm"], color=L2_COLOR, linewidth=2, label="L2 Norm")
    if lp["ma_crossover_epoch"] is not None:
        ax.axvline(x=lp["ma_crossover_epoch"], color=TRIGGER_COLOR, linestyle="--", linewidth=2,
                    label=f"MA Crossover (epoch {lp['ma_crossover_epoch']:.0f})")
    mark_grok(ax, s["grok_epoch"])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("L2 Norm")
    ax.set_title(f"seed {d['seed']} — L2 Norm of Model Weights")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_ma_crossover_detail(d):
    s = d["summary"]
    lp = s["l2_predictor"]
    fig, ax = new_fig()
    ax.plot(d["epoch_grid"], d["fast_ma"], label="Fast MA (w=50)", linewidth=2, color=FAST_MA_COLOR)
    ax.plot(d["epoch_grid"], d["slow_ma"], label="Slow MA (w=200)", linewidth=2, color=SLOW_MA_COLOR)
    if lp["ma_crossover_epoch"] is not None:
        ax.axvline(x=lp["ma_crossover_epoch"], color=TRIGGER_COLOR, linestyle="--", linewidth=2,
                    label=f"Crossover (epoch {lp['ma_crossover_epoch']:.0f})")
    mark_grok(ax, s["grok_epoch"])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("L2 Norm")
    ax.set_title(f"seed {d['seed']} — Moving Average Crossover Detection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_sum_w2_curve(d):
    """The Nanda Figure 7 quantity (Σw², not its square root)."""
    s = d["summary"]
    lp = s["l2_predictor"]
    fig, ax = new_fig()
    epochs_axis = range(1, len(d["sum_w2"]) + 1)
    ax.plot(epochs_axis, d["sum_w2"], color=L2_COLOR, linewidth=2, label="Σw² (sum of squared weights)")
    if lp["ma_crossover_epoch"] is not None:
        ax.axvline(x=lp["ma_crossover_epoch"], color=TRIGGER_COLOR, linestyle="--", linewidth=2,
                    label=f"MA Crossover (epoch {lp['ma_crossover_epoch']:.0f})")
    mark_grok(ax, s["grok_epoch"])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Σw²")
    ax.set_title(f"seed {d['seed']} — Sum of Squared Weights (Nanda Fig. 7 quantity)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_per_module_sum_w2(d):
    s = d["summary"]
    fig, ax = new_fig()
    epochs_axis = range(1, d["per_module_sum_w2"].shape[1] + 1)
    for row, name in zip(d["per_module_sum_w2"], d["per_module_names"]):
        ax.plot(epochs_axis, row, linewidth=2, label=str(name))
    mark_grok(ax, s["grok_epoch"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Σw² per module (log scale)")
    ax.set_title(f"seed {d['seed']} — Per-Module Σw² Breakdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_ma_of_ma_diff(d, log_x):
    s = d["summary"]
    lp = s["l2_predictor"]
    fig, ax = new_fig()
    ax.plot(d["epoch_grid"], d["ma_of_ma_diff"], color=DIFF_COLOR, linewidth=2,
             label="Fast MA of Slow MA − Slow MA")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(d["epoch_grid"], 0, d["ma_of_ma_diff"],
                     where=(d["ma_of_ma_diff"] > 0), alpha=0.2, color="green")
    ax.fill_between(d["epoch_grid"], 0, d["ma_of_ma_diff"],
                     where=(d["ma_of_ma_diff"] <= 0), alpha=0.2, color="red")
    if lp["ma_of_ma_zero_crossing_epoch"] is not None:
        ax.axvline(x=lp["ma_of_ma_zero_crossing_epoch"], color=TRIGGER_COLOR, linestyle="--",
                    linewidth=2,
                    label=f"Zero-crossing (epoch {lp['ma_of_ma_zero_crossing_epoch']:.0f})")
    mark_grok(ax, s["grok_epoch"])
    scale = "log" if log_x else "linear"
    ax.set_xscale(scale)
    ax.set_xlabel(f"Epoch ({scale} scale)")
    ax.set_ylabel("Fast MA of Slow MA − Slow MA")
    ax.set_title(f"seed {d['seed']} — MA-of-MA Differential ({scale} scale) — "
                 f"Zero-Crossing Trigger")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_dropout_gap_bar(d):
    s = d["summary"]
    gaps = s["dropout_final_gap_by_rate"]
    rates = list(gaps.keys())
    values = [gaps[r] for r in rates]
    fig, ax = new_fig()
    bars = ax.bar(rates, values, color="slateblue")
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Dropout gap (final model)")
    ax.set_title(f"seed {d['seed']} — Dropout Gap by Rate (final epoch {s['epochs']})")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def seed_summary_lines(d):
    """Full human-readable numbers for one seed — same content/format as
    make_pdf.py's RESULTS section."""
    s = d["summary"]
    lp = s["l2_predictor"]
    gaps = s["dropout_final_gap_by_rate"]
    lc = s["limit_cycle_check"]
    w = lp["windows"]
    lines = [
        f"seed {s['seed']}  (modulus={s['modulus']}  epochs={s['epochs']}  "
        f"wall_time={s['wall_time_sec']}s)",
        "",
        f"  grok_epoch                    : {s['grok_epoch']}",
        f"  final_train_acc               : {s['final_train_acc']:.4f}",
        f"  final_test_acc                : {s['final_test_acc']:.4f}",
        f"  l2_norm   init -> final       : {s['l2_norm_init']:.4f} -> {s['l2_norm_final']:.4f}",
        f"  sum_w2    init -> final       : {s['sum_w2_init']:.3f} -> {s['sum_w2_final']:.3f}",
        f"  token_embedding_share (init)  : {s['token_embedding_share_init']:.4f}",
        "",
        "  L2 predictor:",
        f"    MA-crossover epoch           : {fmt_epoch(lp['ma_crossover_epoch'])}",
        f"    MA-of-MA zero-crossing epoch : {fmt_epoch(lp['ma_of_ma_zero_crossing_epoch'])}",
        f"    noise_floor                  : {lp['noise_floor']:.6f}",
        f"    windows: fast={w['fast']} slow={w['slow']} ma_of_ma_fast={w['ma_of_ma_fast']} "
        f"skip_epochs={w['skip_epochs']} quiet_epoch_cutoff={w['quiet_epoch_cutoff']}",
        "",
        "  Dropout gap by rate:",
        "    " + "  ".join(f"p{r}={gaps[r]:+.4f}" for r in gaps),
        "",
    ]
    if lc.get("applicable"):
        label = "LIMIT CYCLE" if lc["limit_cycle"] else "stable"
        lines.append(f"  Limit-cycle check             : {label}")
        lines.append(f"    window_start={lc['window_start_epoch']}  "
                      f"post_grok_min={lc['post_grok_min']:.4f}  "
                      f"post_grok_std={lc['post_grok_std']:.4f}")
        lines.append(f"    post_grok_final={lc['post_grok_final']:.4f}  "
                      f"dips<0.9={lc['epochs_below_0.9_post_grok']}")
    else:
        lines.append(f"  Limit-cycle check             : n/a ({lc.get('reason', 'n/a')})")
    return lines


# ============================================================
# Cross-seed comparison plots
# ============================================================

def plot_grokking_overlay(seeds_data):
    fig, ax = new_fig()
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds_data)))
    for d, color in zip(seeds_data, colors):
        epochs_axis = range(1, len(d["test_acc"]) + 1)
        ax.plot(epochs_axis, d["test_acc"], linewidth=2, color=color,
                 label=f"seed {d['seed']} (grok={d['summary']['grok_epoch']})")
    ax.axhline(y=0.9, color="black", linestyle="--", linewidth=1, alpha=0.6,
                label="Grok threshold (0.9)")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("All seeds — Test Accuracy Overlay (Grokking Curves)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_sum_w2_overlay(seeds_data):
    fig, ax = new_fig()
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds_data)))
    for d, color in zip(seeds_data, colors):
        epochs_axis = range(1, len(d["sum_w2"]) + 1)
        ax.plot(epochs_axis, d["sum_w2"], linewidth=2, color=color,
                 label=f"seed {d['seed']}")
        gk = d["summary"]["grok_epoch"]
        if gk is not None:
            ax.axvline(x=gk, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Σw²")
    ax.set_title("All seeds — Σw² Overlay (dotted lines = each seed's own grok epoch)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_grok_epoch_bar(seeds_data, aggregate):
    fig, ax = new_fig()
    seed_ids = [d["seed"] for d in seeds_data]
    groks = [d["summary"]["grok_epoch"] for d in seeds_data]
    bars = ax.bar([str(s) for s in seed_ids], groks, color="teal")
    ax.bar_label(bars, fmt="%d", fontsize=9)
    mean = aggregate.get("grok_epoch_mean")
    std = aggregate.get("grok_epoch_std")
    if mean is not None:
        ax.axhline(y=mean, color="black", linestyle="--", linewidth=1.5,
                    label=f"mean={mean:.1f}  std={std:.1f}")
    ax.set_xlabel("seed")
    ax.set_ylabel("Grok epoch (test acc > 0.9)")
    ax.set_title("All seeds — Grok Epoch")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_predictor_vs_grok_scatter(seeds_data):
    """The key negative-result plot: does either L2-Norm predictor signal
    track the actual grok epoch across seeds? (It should, on the diagonal,
    if the predictor worked.)"""
    fig, ax = new_fig()
    groks = [d["summary"]["grok_epoch"] for d in seeds_data]
    ma_cross = [d["summary"]["l2_predictor"]["ma_crossover_epoch"] for d in seeds_data]
    ma_of_ma = [d["summary"]["l2_predictor"]["ma_of_ma_zero_crossing_epoch"] for d in seeds_data]
    seed_ids = [d["seed"] for d in seeds_data]

    ax.scatter(groks, ma_cross, color=SLOW_MA_COLOR, s=100, marker="o", label="MA-crossover epoch")
    ax.scatter(groks, ma_of_ma, color=DIFF_COLOR, s=100, marker="^", label="MA-of-MA zero-crossing epoch")
    for g, mc, mm, sid in zip(groks, ma_cross, ma_of_ma, seed_ids):
        ax.annotate(f"s{sid}", (g, mc), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.annotate(f"s{sid}", (g, mm), fontsize=8, xytext=(4, 4), textcoords="offset points")

    lims = [min(groks + ma_cross + ma_of_ma) * 0.5, max(groks + ma_cross + ma_of_ma) * 1.1]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, alpha=0.5,
             label="y = x (perfect predictor)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual grok epoch (log scale)")
    ax.set_ylabel("Predictor signal epoch (log scale)")
    ax.set_title("L2-Norm predictor signal vs. actual grok epoch, across seeds\n"
                 "(flat / off-diagonal = predictor does not track grok timing)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_dropout_gap_grouped_bar(seeds_data):
    fig, ax = new_fig()
    rates = list(seeds_data[0]["summary"]["dropout_final_gap_by_rate"].keys())
    n_seeds = len(seeds_data)
    x = np.arange(len(rates))
    width = 0.8 / n_seeds
    colors = plt.cm.tab10(np.linspace(0, 1, n_seeds))
    for i, (d, color) in enumerate(zip(seeds_data, colors)):
        gaps = d["summary"]["dropout_final_gap_by_rate"]
        values = [gaps[r] for r in rates]
        ax.bar(x + i * width, values, width=width, color=color, label=f"seed {d['seed']}")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xticks(x + width * (n_seeds - 1) / 2)
    ax.set_xticklabels(rates)
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Dropout gap (final model)")
    ax.set_title("All seeds — Dropout Gap by Rate")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_limit_cycle_bar(seeds_data):
    fig, ax = new_fig()
    seed_ids, mins, stds, labels = [], [], [], []
    for d in seeds_data:
        lc = d["summary"]["limit_cycle_check"]
        if not lc.get("applicable"):
            continue
        seed_ids.append(d["seed"])
        mins.append(lc["post_grok_min"])
        stds.append(lc["post_grok_std"])
        labels.append("LIMIT CYCLE" if lc["limit_cycle"] else "stable")
    x = np.arange(len(seed_ids))
    colors = ["crimson" if lbl == "LIMIT CYCLE" else "teal" for lbl in labels]
    bars = ax.bar(x, mins, color=colors)
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in mins], fontsize=9)
    ax.axhline(y=0.9, color="black", linestyle="--", linewidth=1, label="Limit-cycle threshold (0.9)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seed_ids])
    ax.set_xlabel("seed")
    ax.set_ylabel("Post-grok minimum test accuracy")
    ax.set_title("All seeds — Post-Grok Limit-Cycle Check (min test acc after settling)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_token_embedding_share_bar(seeds_data):
    fig, ax = new_fig()
    seed_ids = [d["seed"] for d in seeds_data]
    shares = [d["summary"]["token_embedding_share_init"] for d in seeds_data]
    bars = ax.bar([str(s) for s in seed_ids], shares, color="darkcyan")
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_xlabel("seed")
    ax.set_ylabel("token_embedding share of Σw² at init")
    ax.set_title("All seeds — Init Token-Embedding Share (small-init sanity check)")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def aggregate_summary_lines(aggregate):
    lines = [
        f"n_seeds          : {aggregate['n_seeds']}",
        f"epochs           : {aggregate['epochs']}",
        f"grok_epoch_mean  : {aggregate['grok_epoch_mean']}",
        f"grok_epoch_std   : {aggregate['grok_epoch_std']}",
        f"grok_epochs      : {aggregate['grok_epochs']}",
        f"n_limit_cycle    : {aggregate['n_limit_cycle']} / {aggregate['n_seeds']}",
    ]
    return lines


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot every result from a completed run_nanda_benchmark.py run "
                     "— separate PNGs plus one combined human-readable PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str, default="results/nanda_unified")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="default: <results_dir>/plots")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(REPO_ROOT, results_dir)
    output_dir = args.output_dir or os.path.join(results_dir, "plots")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    seeds = discover_seeds(results_dir)
    if not seeds:
        print(f"No seed_*/summary.json found under {results_dir} — "
              f"run run_nanda_benchmark.py first.")
        raise SystemExit(1)

    print("=" * 72)
    print("NANDA-UNIFIED RESULTS PLOTTER")
    print("=" * 72)
    print(f"results_dir : {results_dir}")
    print(f"output_dir  : {output_dir}")
    print(f"seeds found : {seeds}")
    print("=" * 72)

    seeds_data = [load_seed(results_dir, seed) for seed in seeds]
    aggregate = load_aggregate(results_dir, seeds_data)

    pdf_path = os.path.join(output_dir, "nanda_results_report.pdf")
    with PdfPages(pdf_path) as pdf:
        collector = Collector(output_dir, pdf)

        # ---- Title / overview page ----
        overview_lines = [
            f"results_dir : {results_dir}",
            f"seeds       : {seeds}",
            "",
            "=== AGGREGATE SUMMARY ===",
            *aggregate_summary_lines(aggregate),
        ]
        collector.add_text_page("_overview", "00_overview.png", overview_lines,
                                 title="NANDA-UNIFIED BENCHMARK — RESULTS REPORT")

        # ---- Cross-seed comparison section ----
        print("\nComparison plots (all seeds):")
        comparison_plots = [
            ("01_grokking_overlay.png", plot_grokking_overlay(seeds_data)),
            ("02_sum_w2_overlay.png", plot_sum_w2_overlay(seeds_data)),
            ("03_grok_epoch_bar.png", plot_grok_epoch_bar(seeds_data, aggregate)),
            ("04_predictor_vs_grok_scatter.png", plot_predictor_vs_grok_scatter(seeds_data)),
            ("05_dropout_gap_grouped_bar.png", plot_dropout_gap_grouped_bar(seeds_data)),
            ("06_limit_cycle_bar.png", plot_limit_cycle_bar(seeds_data)),
            ("07_token_embedding_share_bar.png", plot_token_embedding_share_bar(seeds_data)),
        ]
        for filename, fig in comparison_plots:
            collector.add("comparison", filename, fig)

        # ---- Per-seed sections ----
        for d in seeds_data:
            seed = d["seed"]
            subdir = f"seed_{seed}"
            print(f"\nSeed {seed} plots:")

            collector.add_text_page(subdir, "00_summary.png", seed_summary_lines(d),
                                     title=f"seed {seed} — full numbers")
            collector.add(subdir, "01_grokking_curve.png", plot_grokking_curve(d))
            collector.add(subdir, "02_loss_curve.png", plot_loss_curve(d))
            collector.add(subdir, "03_l2_norm_curve.png", plot_l2_norm_curve(d))
            collector.add(subdir, "04_ma_crossover_detail.png", plot_ma_crossover_detail(d))
            collector.add(subdir, "05_sum_w2_curve.png", plot_sum_w2_curve(d))
            collector.add(subdir, "06_per_module_sum_w2.png", plot_per_module_sum_w2(d))
            collector.add(subdir, "07_ma_of_ma_diff_log.png", plot_ma_of_ma_diff(d, log_x=True))
            collector.add(subdir, "08_ma_of_ma_diff_linear.png", plot_ma_of_ma_diff(d, log_x=False))
            collector.add(subdir, "09_dropout_gap_bar.png", plot_dropout_gap_bar(d))

    print("\n" + "=" * 72)
    print(f"Wrote {collector.count} figures under {output_dir}/")
    print(f"Combined PDF: {pdf_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
