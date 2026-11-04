#!/usr/bin/env python3
"""
plot_commutator_defect_stage6.py
================================

STAGE 6 — raw Commutator Defect curve plots from saved Stage 5 outputs.

Reads only:
  08_Experiments/results/nanda_unified/seed_*/commutator_defect/
  08_Experiments/results/nanda_unified/seed_*/commutator_defect_eta1e-4/
  08_Experiments/results/nanda_unified/seed_*/training/
  08_Experiments/results/nanda_unified/seed_*/summary.json

Does NOT recompute D, does NOT change thresholds / eta / rules, and does NOT
run Stage 7 official scoring.

Outputs under:
  08_Experiments/results/nanda_unified/figures/commutator_defect/

Usage (from repo root):
  .venv/bin/python 06_Code/scripts/plot_commutator_defect_stage6.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO_ROOT, "08_Experiments", "results", "nanda_unified")
FIG_DIR = os.path.join(RESULTS, "figures", "commutator_defect")

# Directory name under each seed_*/ for the two eta runs.
PRIMARY_SUBDIR = "commutator_defect"
SECONDARY_SUBDIR = "commutator_defect_eta1e-4"
PRIMARY_ETA_LABEL = "1e-3"
SECONDARY_ETA_LABEL = "1e-4"

SEEDS = (0, 1, 2, 3, 4)
X_MAX = 39999
ZOOM_MAX = 2000


def xu_baseline_and_threshold(median_series: np.ndarray) -> tuple[float, float]:
    """Frozen Xu onset helpers for plotting only (same arithmetic as Stage 1/5).

    b = median of the first 3 finite D̃ values.
    threshold line = max(10 * max(b, 0.1), 20)
    """
    first_three = np.asarray(median_series[:3], dtype=float)
    if first_three.size < 3 or not np.all(np.isfinite(first_three)):
        raise ValueError("Need three finite early medians to plot Xu baseline.")
    b = float(np.median(first_three))
    threshold = float(max(10.0 * max(b, 0.1), 20.0))
    return b, threshold


def load_run(seed: int, subdir: str) -> dict:
    seed_dir = os.path.join(RESULTS, f"seed_{seed}")
    d = os.path.join(seed_dir, subdir)
    with open(os.path.join(d, "commutator_defect_signal.json")) as handle:
        signal = json.load(handle)
    with open(os.path.join(seed_dir, "summary.json")) as handle:
        summary = json.load(handle)

    checkpoints = np.load(os.path.join(d, "commutator_defect_checkpoints.npy"))
    median = np.load(os.path.join(d, "commutator_defect_median.npy")).astype(float)
    q25 = np.load(os.path.join(d, "commutator_defect_q25.npy")).astype(float)
    q75 = np.load(os.path.join(d, "commutator_defect_q75.npy")).astype(float)
    grad_a = np.load(os.path.join(d, "commutator_defect_grad_norm_A.npy")).astype(float)
    grad_b = np.load(os.path.join(d, "commutator_defect_grad_norm_B.npy")).astype(float)

    train_path = os.path.join(seed_dir, "training", "train_acc_history.npy")
    test_path = os.path.join(seed_dir, "training", "test_acc_history.npy")
    train_acc = np.load(train_path).astype(float) if os.path.exists(train_path) else None
    test_acc = np.load(test_path).astype(float) if os.path.exists(test_path) else None

    grok_epoch = signal.get("grok_epoch", summary.get("grok_epoch"))
    if grok_epoch is None and test_acc is not None:
        hits = np.where(test_acc > 0.9)[0]
        grok_epoch = int(hits[0]) if hits.size else None
    else:
        grok_epoch = int(grok_epoch) if grok_epoch is not None else None

    event_epoch = signal.get("event_epoch")
    event_epoch = int(event_epoch) if event_epoch is not None else None

    b, threshold = xu_baseline_and_threshold(median)
    eta = float(signal.get("eta", np.load(os.path.join(d, "commutator_defect_eta.npy")).item()))

    return {
        "seed": seed,
        "subdir": subdir,
        "eta": eta,
        "checkpoints": checkpoints.astype(int),
        "median": median,
        "q25": q25,
        "q75": q75,
        "grad_norm_A_median": np.nanmedian(grad_a, axis=1),
        "grad_norm_B_median": np.nanmedian(grad_b, axis=1),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "grok_epoch": grok_epoch,
        "event_epoch": event_epoch,
        "b": b,
        "threshold": threshold,
    }


def _positive_for_log(y: np.ndarray) -> np.ndarray:
    """Replace non-positive / non-finite with NaN so log axes skip them."""
    out = np.asarray(y, dtype=float).copy()
    out[~np.isfinite(out) | (out <= 0.0)] = np.nan
    return out


def _draw_d_panel(ax, run: dict, x_min: float, x_max: float, title: str | None = None) -> None:
    ck = run["checkpoints"]
    med = _positive_for_log(run["median"])
    q25 = _positive_for_log(run["q25"])
    q75 = _positive_for_log(run["q75"])
    mask = (ck >= x_min) & (ck <= x_max)
    ax.fill_between(ck[mask], q25[mask], q75[mask], color="tab:blue", alpha=0.25, label="q25–q75")
    ax.plot(ck[mask], med[mask], color="tab:blue", lw=1.4, label=r"median $\tilde{D}$")
    ax.axhline(run["threshold"], color="black", ls="--", lw=1.1,
               label=f"threshold={run['threshold']:.2f}")
    ax.axhline(run["b"], color="grey", ls=":", lw=1.1, label=f"b={run['b']:.2f}")
    if run["event_epoch"] is not None and x_min <= run["event_epoch"] <= x_max:
        ax.axvline(run["event_epoch"], color="red", lw=1.3,
                   label=f"alarm={run['event_epoch']}")
    if run["grok_epoch"] is not None and x_min <= run["grok_epoch"] <= x_max:
        ax.axvline(run["grok_epoch"], color="green", lw=1.3,
                   label=f"grok={run['grok_epoch']}")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel(r"$\tilde{D}$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, which="both", alpha=0.25)


def _draw_grad_acc_panel(ax, run: dict, x_min: float, x_max: float) -> None:
    ck = run["checkpoints"]
    mask = (ck >= x_min) & (ck <= x_max)
    g_a = _positive_for_log(run["grad_norm_A_median"])
    g_b = _positive_for_log(run["grad_norm_B_median"])
    ax.plot(ck[mask], g_a[mask], color="tab:purple", lw=1.2, label=r"median $\|g_A\|$")
    ax.plot(ck[mask], g_b[mask], color="tab:orange", lw=1.2, label=r"median $\|g_B\|$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\|g\|$ (median)")
    ax.set_xlim(x_min, x_max)

    if run["event_epoch"] is not None and x_min <= run["event_epoch"] <= x_max:
        ax.axvline(run["event_epoch"], color="red", lw=1.0, alpha=0.8)
    if run["grok_epoch"] is not None and x_min <= run["grok_epoch"] <= x_max:
        ax.axvline(run["grok_epoch"], color="green", lw=1.0, alpha=0.8)

    train_acc = run["train_acc"]
    test_acc = run["test_acc"]
    if train_acc is not None or test_acc is not None:
        ax2 = ax.twinx()
        ref_len = len(train_acc) if train_acc is not None else len(test_acc)
        end = int(min(x_max, ref_len - 1))
        start = int(max(0, x_min))
        epochs = np.arange(start, end + 1, dtype=int)
        if train_acc is not None:
            ax2.plot(epochs, train_acc[epochs], color="tab:gray", lw=1.0, alpha=0.85,
                     label="train acc")
        if test_acc is not None:
            ax2.plot(epochs, test_acc[epochs], color="tab:cyan", lw=1.0, alpha=0.85,
                     label="test acc")
        ax2.set_ylabel("accuracy")
        ax2.set_ylim(-0.05, 1.05)
        lines_a, labels_a = ax.get_legend_handles_labels()
        lines_b, labels_b = ax2.get_legend_handles_labels()
        ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc="best")
    else:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.25)


def plot_seed_figure(run: dict, eta_label: str, path: str) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.6, 1.4]},
    )
    seed = run["seed"]
    fig.suptitle(
        f"Commutator Defect — seed {seed}, eta={eta_label} "
        f"(alarm={run['event_epoch']}, grok={run['grok_epoch']})",
        y=0.995,
    )
    _draw_d_panel(axes[0], run, 0, X_MAX, title=r"median $\tilde{D}$ with q25–q75 band")
    _draw_grad_acc_panel(axes[1], run, 0, X_MAX)
    _draw_d_panel(axes[2], run, 0, ZOOM_MAX, title="zoom: epochs 0–2000")
    axes[1].set_xlabel("epoch")
    axes[2].set_xlabel("epoch")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_overview(primary_runs: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colours = plt.cm.tab10(np.arange(len(primary_runs)))
    for colour, run in zip(colours, primary_runs):
        grok = run["grok_epoch"]
        if grok is None or grok <= 0:
            continue
        x = run["checkpoints"].astype(float) / float(grok)
        y = _positive_for_log(run["median"])
        ax.plot(x, y, color=colour, lw=1.4, label=f"seed {run['seed']}")
        if run["event_epoch"] is not None:
            # Mark alarm on the same normalised axis.
            alarm_x = float(run["event_epoch"]) / float(grok)
            # Interpolate D̃ at the nearest checkpoint for the marker height.
            idx = int(np.argmin(np.abs(run["checkpoints"] - run["event_epoch"])))
            alarm_y = y[idx]
            if np.isfinite(alarm_y):
                ax.scatter([alarm_x], [alarm_y], color=colour, marker="x", s=55, zorder=5)
            ax.axvline(alarm_x, color=colour, ls=":", lw=0.9, alpha=0.7)
    ax.axvline(1.0, color="green", lw=1.4, label="grok (x=1)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch / grok_epoch")
    ax.set_ylabel(r"median $\tilde{D}$")
    ax.set_title(
        r"Commutator Defect PRIMARY (eta=1e-3): all seeds, "
        r"x = epoch/grok_epoch (× = alarm)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def acc_at_epoch(acc: np.ndarray | None, epoch: int | None) -> float | None:
    if acc is None or epoch is None:
        return None
    if epoch < 0 or epoch >= len(acc):
        return None
    return float(acc[epoch])


def first_memorisation_epoch(train_acc: np.ndarray | None, threshold: float = 0.99) -> int | None:
    if train_acc is None:
        return None
    hits = np.where(train_acc > threshold)[0]
    return int(hits[0]) if hits.size else None


def print_per_seed_table(primary_runs: list[dict]) -> None:
    print()
    print("Per-seed numbers (PRIMARY eta=1e-3) — alarm vs memorisation")
    print(
        f"{'seed':>4}  {'alarm':>6}  {'train@alarm':>11}  {'test@alarm':>10}  "
        f"{'mem_epoch(train>0.99)':>22}  {'grok':>6}"
    )
    for run in primary_runs:
        alarm = run["event_epoch"]
        train_a = acc_at_epoch(run["train_acc"], alarm)
        test_a = acc_at_epoch(run["test_acc"], alarm)
        mem = first_memorisation_epoch(run["train_acc"])
        train_s = f"{train_a:.4f}" if train_a is not None else "n/a"
        test_s = f"{test_a:.4f}" if test_a is not None else "n/a"
        mem_s = str(mem) if mem is not None else "n/a"
        alarm_s = str(alarm) if alarm is not None else "miss"
        print(
            f"{run['seed']:>4}  {alarm_s:>6}  {train_s:>11}  {test_s:>10}  "
            f"{mem_s:>22}  {run['grok_epoch']:>6}"
        )
    print()


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    saved: list[str] = []

    primary_runs = [load_run(seed, PRIMARY_SUBDIR) for seed in SEEDS]
    secondary_runs = [load_run(seed, SECONDARY_SUBDIR) for seed in SEEDS]

    for run in primary_runs:
        path = os.path.join(FIG_DIR, f"commutator_defect_seed{run['seed']}.png")
        plot_seed_figure(run, PRIMARY_ETA_LABEL, path)
        saved.append(path)
        print(f"Wrote {path}")

    for run in secondary_runs:
        path = os.path.join(
            FIG_DIR, f"commutator_defect_seed{run['seed']}_eta1e-4.png"
        )
        plot_seed_figure(run, SECONDARY_ETA_LABEL, path)
        saved.append(path)
        print(f"Wrote {path}")

    overview = os.path.join(FIG_DIR, "commutator_defect_overview_primary.png")
    plot_overview(primary_runs, overview)
    saved.append(overview)
    print(f"Wrote {overview}")

    print_per_seed_table(primary_runs)

    print("Saved figure paths:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
