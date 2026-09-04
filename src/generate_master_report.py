"""
Master report generator for the four-head grokking benchmark.

Reads every COMPLETE runs/four_head/run_<N>/ folder (the unified_measurements
subdir layout) and writes, into benchmark_analysis/:

  1. master_plots.pdf  - every plot for every predictor, per run + cross-run
  2. master_data.pdf   - a full, literal per-epoch dump of every number
  3. data/*.csv        - the same numbers as CSV (machine-friendly; use these
                         if the PDF is too large to feed somewhere)

Standalone:  python src/generate_master_report.py
It is also called at the end of run_full_benchmark.py when a new run was
produced.

Nothing about the results is hardcoded here - every figure and table is
computed from the loaded runs.
"""

import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# predictors/ sits next to this file; add src/ to the path so this works
# whether run as "python src/generate_master_report.py" or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictors.l2_norm import (  # noqa: E402
    compute_noise_floor, detect_ma_of_ma_zero_crossing, detect_ma_crossover,
)

RUNS_DIR = Path("runs/four_head")
OUT_DIR = Path("benchmark_analysis")
PLOTS_PDF = OUT_DIR / "master_plots.pdf"
DATA_PDF = OUT_DIR / "master_data.pdf"
CSV_DIR = OUT_DIR / "data"

SKIP_EPOCHS = 100
QUIET_EPOCH_CUTOFF = 90
ROWS_PER_PAGE = 110
DUMP_FONTSIZE = 4.5
RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

REQUIRED = [
    "training/train_acc_history.npy", "training/test_acc_history.npy",
    "training/loss_history.npy", "l2_norm/l2_norm_history.npy",
    "dropout/dropout_gap_epochs.npy", "dropout/dropout_gap_by_rate.npy",
    "dropout/dropout_rates.npy", "reports/training_report.pdf",
]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _run_numbers():
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for d in RUNS_DIR.iterdir():
        if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit():
            if all((d / r).exists() for r in REQUIRED):
                out.append(int(d.name[4:]))
    return sorted(out)


def _opt(path):
    return np.load(path) if path.exists() else None


def load_run(n):
    d = RUNS_DIR / f"run_{n}"
    tr = d / "training"
    l2 = d / "l2_norm"
    dp = d / "dropout"
    data = {
        "run": n,
        "seed": int(_opt(d / "seed.npy")) if (d / "seed.npy").exists() else None,
        "train_acc": np.load(tr / "train_acc_history.npy"),
        "test_acc": np.load(tr / "test_acc_history.npy"),
        "loss": np.load(tr / "loss_history.npy"),
        "l2_norm": np.load(l2 / "l2_norm_history.npy"),
        "l2_norm_smoothed": _opt(l2 / "l2_norm_smoothed.npy"),
        "epoch_grid": _opt(l2 / "epoch_grid.npy"),
        "fast_ma": _opt(l2 / "fast_ma.npy"),
        "slow_ma": _opt(l2 / "slow_ma.npy"),
        "fast_ma_of_slow_ma": _opt(l2 / "fast_ma_of_slow_ma.npy"),
        "ma_of_ma_diff": _opt(l2 / "ma_of_ma_diff.npy"),
        "accel_raw": _opt(l2 / "acceleration_raw.npy"),
        "accel_smoothed": _opt(l2 / "acceleration_smoothed.npy"),
        "accel_double": _opt(l2 / "acceleration_double_smoothed.npy"),
        "detection_epoch": (float(_opt(l2 / "detection_epoch.npy")[0])
                            if (l2 / "detection_epoch.npy").exists() else None),
        "dropout_gap_epochs": np.load(dp / "dropout_gap_epochs.npy"),
        "dropout_rates": np.load(dp / "dropout_rates.npy"),
        "dropout_gap_by_rate": np.load(dp / "dropout_gap_by_rate.npy"),
        "dropout_train_acc_by_rate": _opt(dp / "dropout_train_acc_by_rate.npy"),
        "dropout_eval_acc_by_rate": _opt(dp / "dropout_eval_acc_by_rate.npy"),
    }
    data["num_epochs"] = len(data["train_acc"])
    data["grok_epoch"] = grok_epoch(data["test_acc"])

    # derived L2 detection signals (computed, for the plots + summary)
    if data["epoch_grid"] is not None and data["ma_of_ma_diff"] is not None:
        eg, diff = data["epoch_grid"], data["ma_of_ma_diff"]
        data["trigger_epoch"] = detect_ma_of_ma_zero_crossing(eg, diff, skip_epochs=SKIP_EPOCHS)
        data["noise_floor"] = compute_noise_floor(diff, eg, quiet_epoch_cutoff=QUIET_EPOCH_CUTOFF)
        data["ma_crossover_epoch"] = detect_ma_crossover(eg, data["fast_ma"], data["slow_ma"],
                                                         skip_epochs=SKIP_EPOCHS)
    else:
        data["trigger_epoch"] = data["noise_floor"] = data["ma_crossover_epoch"] = None
    return data


def grok_epoch(test_acc):
    ta = np.asarray(test_acc)
    return int(np.argmax(ta > 0.9)) if (ta > 0.9).any() else None


# ----------------------------------------------------------------------
# PDF 1 - plots
# ----------------------------------------------------------------------

def _grok_vline(ax, g, color="green"):
    if g:
        ax.axvline(x=g, color=color, linestyle=":", linewidth=1.6, alpha=0.75,
                   label=f"grok ({g})")


def build_plots_pdf(runs):
    with PdfPages(PLOTS_PDF) as pdf:
        # ---- per-run pages ----
        for i, d in enumerate(runs):
            n = d["run"]
            ep = np.arange(1, d["num_epochs"] + 1)
            c = RUN_COLORS[i % len(RUN_COLORS)]

            # 1. grokking curve
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(ep, d["train_acc"], label="train acc", color="steelblue", lw=2)
            ax.plot(ep, d["test_acc"], label="test acc", color="seagreen", lw=2)
            _grok_vline(ax, d["grok_epoch"])
            ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("accuracy")
            ax.set_title(f"Run {n} (seed {d['seed']}): Grokking Curve")
            ax.legend(); ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

            # 2. loss
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(ep, d["loss"], color="darkorange", lw=2)
            _grok_vline(ax, d["grok_epoch"])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("epoch (log)"); ax.set_ylabel("loss (log)")
            ax.set_title(f"Run {n}: Training Loss")
            ax.legend(); ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

            # 3. L2 norm (raw + smoothed)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(ep, d["l2_norm"], color="purple", lw=2, label="L2 norm")
            if d["l2_norm_smoothed"] is not None:
                ax.plot(ep, d["l2_norm_smoothed"][:len(ep)], color="magenta", lw=1,
                        alpha=0.7, label="L2 smoothed (w=50)")
            if d["detection_epoch"]:
                ax.axvline(x=d["detection_epoch"], color="red", ls="--", lw=1.5,
                           label=f"MA crossover ({d['detection_epoch']:.0f})")
            _grok_vline(ax, d["grok_epoch"])
            ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("L2 norm")
            ax.set_title(f"Run {n}: L2 Norm of Weights")
            ax.legend(); ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

            # 4. fast/slow MA
            if d["fast_ma"] is not None:
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.plot(d["epoch_grid"], d["fast_ma"], color="blue", lw=2, label="fast MA (w=50)")
                ax.plot(d["epoch_grid"], d["slow_ma"], color="orange", lw=2, label="slow MA (w=200)")
                if d["ma_crossover_epoch"]:
                    ax.axvline(x=d["ma_crossover_epoch"], color="red", ls="--", lw=1.5,
                               label=f"crossover ({d['ma_crossover_epoch']:.0f})")
                _grok_vline(ax, d["grok_epoch"])
                ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("L2 norm")
                ax.set_title(f"Run {n}: L2 Norm Moving Averages")
                ax.legend(); ax.grid(True, alpha=0.3)
                pdf.savefig(fig); plt.close(fig)

            # 5. MA-of-MA differential (log + linear)
            if d["ma_of_ma_diff"] is not None:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for ax, xs in zip(axes, ("log", "linear")):
                    ax.plot(d["epoch_grid"], d["ma_of_ma_diff"], color="darkred", lw=2)
                    ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
                    if d["trigger_epoch"]:
                        ax.axvline(x=d["trigger_epoch"], color="red", ls="--", lw=1.5,
                                   label=f"zero-crossing ({d['trigger_epoch']:.0f})")
                    _grok_vline(ax, d["grok_epoch"])
                    ax.set_xscale(xs); ax.set_xlabel(f"epoch ({xs})")
                    ax.set_ylabel("MA-of-MA differential"); ax.legend(); ax.grid(True, alpha=0.3)
                fig.suptitle(f"Run {n}: MA-of-MA Differential "
                             f"(noise floor {d['noise_floor']:.2e})" if d["noise_floor"] else
                             f"Run {n}: MA-of-MA Differential")
                pdf.savefig(fig); plt.close(fig)

            # 6. L2 acceleration
            if d["accel_raw"] is not None:
                fig, ax = plt.subplots(figsize=(11, 6))
                xa = np.arange(1, len(d["accel_raw"]) + 1)
                ax.plot(xa, d["accel_raw"], color="grey", lw=0.6, alpha=0.5, label="accel raw")
                ax.plot(xa, d["accel_smoothed"][:len(xa)], color="teal", lw=1.5, label="accel smoothed")
                ax.plot(xa, d["accel_double"][:len(xa)], color="crimson", lw=2, label="accel double-smoothed")
                ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
                _grok_vline(ax, d["grok_epoch"])
                ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("d(L2)/d(epoch)")
                ax.set_title(f"Run {n}: L2 Norm Acceleration")
                ax.legend(); ax.grid(True, alpha=0.3)
                pdf.savefig(fig); plt.close(fig)

            # 7. dropout gap sweep
            fig, ax = plt.subplots(figsize=(11, 6))
            for j, r in enumerate(d["dropout_rates"]):
                ax.plot(d["dropout_gap_epochs"], d["dropout_gap_by_rate"][j], lw=1.8,
                        label=f"p={r:g}")
            ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
            _grok_vline(ax, d["grok_epoch"])
            ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("dropout gap")
            ax.set_title(f"Run {n}: Dropout Gap - Multi-Rate Sweep")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

            # 8. dropout train-acc vs clean-acc
            if d["dropout_train_acc_by_rate"] is not None:
                fig, ax = plt.subplots(figsize=(11, 6))
                for j, r in enumerate(d["dropout_rates"]):
                    ax.plot(d["dropout_gap_epochs"], d["dropout_train_acc_by_rate"][j], lw=1.5,
                            label=f"train acc p={r:g}")
                ax.plot(d["dropout_gap_epochs"], d["dropout_eval_acc_by_rate"][0], color="black",
                        lw=2, ls="--", label="clean acc (p=0)")
                _grok_vline(ax, d["grok_epoch"])
                ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("accuracy")
                ax.set_title(f"Run {n}: Dropout Train Acc (per rate) vs Clean Acc")
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                pdf.savefig(fig); plt.close(fig)

            # 9. final dropout gap vs rate
            fig, ax = plt.subplots(figsize=(9, 5))
            finals = [d["dropout_gap_by_rate"][j][-1] for j in range(len(d["dropout_rates"]))]
            ax.plot(d["dropout_rates"], finals, marker="o", color=c)
            ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
            ax.set_xlabel("dropout rate p"); ax.set_ylabel("final dropout gap")
            ax.set_title(f"Run {n}: Final Dropout Gap vs Rate")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig); plt.close(fig)

        # ---- cross-run comparison pages ----
        if len(runs) >= 1:
            _comparison_pages(pdf, runs)

    return PLOTS_PDF


def _comparison_pages(pdf, runs):
    # test acc overlay
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, d in enumerate(runs):
        ep = np.arange(1, d["num_epochs"] + 1)
        c = RUN_COLORS[i % len(RUN_COLORS)]
        ax.plot(ep, d["test_acc"], color=c, lw=2, label=f"Run {d['run']} (grok {d['grok_epoch']})")
        _grok_vline(ax, d["grok_epoch"], color=c)
    ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("test accuracy")
    ax.set_title("Comparison: Test Accuracy Across Runs")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    pdf.savefig(fig); plt.close(fig)

    # loss overlay
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, d in enumerate(runs):
        ep = np.arange(1, d["num_epochs"] + 1)
        ax.plot(ep, d["loss"], color=RUN_COLORS[i % len(RUN_COLORS)], lw=2, label=f"Run {d['run']}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("epoch (log)"); ax.set_ylabel("loss (log)")
    ax.set_title("Comparison: Training Loss Across Runs")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    pdf.savefig(fig); plt.close(fig)

    # L2 norm overlay
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, d in enumerate(runs):
        ep = np.arange(1, d["num_epochs"] + 1)
        ax.plot(ep, d["l2_norm"], color=RUN_COLORS[i % len(RUN_COLORS)], lw=2, label=f"Run {d['run']}")
    ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("L2 norm")
    ax.set_title("Comparison: L2 Norm Across Runs")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    pdf.savefig(fig); plt.close(fig)

    # grok epoch bar
    groks = [d["grok_epoch"] for d in runs if d["grok_epoch"] is not None]
    if groks:
        fig, ax = plt.subplots(figsize=(9, 6))
        labels = [f"Run {d['run']}" for d in runs if d["grok_epoch"] is not None]
        bars = ax.bar(range(len(groks)), groks,
                      color=RUN_COLORS[:len(groks)], alpha=0.75, edgecolor="black")
        for b, g in zip(bars, groks):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(g),
                    ha="center", va="bottom", fontweight="bold")
        if len(groks) >= 2:
            ax.axhline(y=float(np.mean(groks)), color="black", ls="--",
                       label=f"mean {np.mean(groks):.0f} +/- {np.std(groks):.0f}")
            ax.legend()
        ax.set_xticks(range(len(groks))); ax.set_xticklabels(labels)
        ax.set_ylabel("grok epoch"); ax.set_title("Comparison: Grokking Epoch")
        ax.grid(True, alpha=0.3, axis="y")
        pdf.savefig(fig); plt.close(fig)

    # dropout sweep - one panel per run
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, d in enumerate(runs):
        ax = axes[0][i]
        for j, r in enumerate(d["dropout_rates"]):
            ax.plot(d["dropout_gap_epochs"], d["dropout_gap_by_rate"][j], lw=1.6, label=f"p={r:g}")
        ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
        ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("dropout gap")
        ax.set_title(f"Run {d['run']}"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("Comparison: Dropout Gap Multi-Rate Sweep")
    pdf.savefig(fig); plt.close(fig)

    # final dropout gap vs rate - line per run
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, d in enumerate(runs):
        finals = [d["dropout_gap_by_rate"][j][-1] for j in range(len(d["dropout_rates"]))]
        ax.plot(d["dropout_rates"], finals, marker="o",
                color=RUN_COLORS[i % len(RUN_COLORS)], label=f"Run {d['run']}")
    ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("dropout rate p"); ax.set_ylabel("final dropout gap")
    ax.set_title("Comparison: Final Dropout Gap vs Rate")
    ax.legend(); ax.grid(True, alpha=0.3)
    pdf.savefig(fig); plt.close(fig)

    # MA-of-MA diff overlay
    if all(d["ma_of_ma_diff"] is not None for d in runs):
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, d in enumerate(runs):
            ax.plot(d["epoch_grid"], d["ma_of_ma_diff"], color=RUN_COLORS[i % len(RUN_COLORS)],
                    lw=2, label=f"Run {d['run']}")
        ax.axhline(y=0, color="black", lw=0.6, alpha=0.5)
        ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.set_ylabel("MA-of-MA differential")
        ax.set_title("Comparison: MA-of-MA Differential Across Runs")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        pdf.savefig(fig); plt.close(fig)


# ----------------------------------------------------------------------
# PDF 2 - full numeric dump
# ----------------------------------------------------------------------

def _series_stats(name, a):
    a = np.asarray(a, dtype=float)
    return (f"  {name:<26} n={a.size:<7} first={a[0]:+.6e} last={a[-1]:+.6e} "
            f"min={a.min():+.6e} max={a.max():+.6e} mean={a.mean():+.6e} std={a.std():+.6e}")


def _text_page(pdf, text, fontsize=7):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.04, 0.98, text, va="top", ha="left", family="monospace", fontsize=fontsize)
    pdf.savefig(fig); plt.close(fig)


def _dump_table(pdf, title, header, rows, fontsize=DUMP_FONTSIZE, rows_per_page=ROWS_PER_PAGE):
    total_pages = (len(rows) + rows_per_page - 1) // rows_per_page
    for p in range(total_pages):
        chunk = rows[p * rows_per_page:(p + 1) * rows_per_page]
        body = f"{title}  (page {p + 1}/{total_pages})\n\n{header}\n" + "\n".join(chunk)
        fig = plt.figure(figsize=(14, 8.5)); fig.patch.set_facecolor("white")
        fig.text(0.02, 0.98, body, va="top", ha="left", family="monospace", fontsize=fontsize)
        pdf.savefig(fig); plt.close(fig)


def build_data_pdf(runs):
    with PdfPages(DATA_PDF) as pdf:
        # --- summary section ---
        head = ["FOUR-HEAD GROKKING BENCHMARK - EXHAUSTIVE NUMERIC DUMP", "",
                f"Runs included: {[d['run'] for d in runs]}",
                f"Task: (a+b) mod 113, 30/70 split, AdamW lr=1e-3 weight_decay=1.0 betas=(0.9,0.98)  [Nanda-Unified]",
                f"Model: 4 heads, d_model=128, head_dim=32", ""]
        groks = [d["grok_epoch"] for d in runs if d["grok_epoch"] is not None]
        if len(groks) >= 2:
            head.append(f"Grok epochs: {groks}  mean={np.mean(groks):.1f} "
                        f"std={np.std(groks):.1f} min={min(groks)} max={max(groks)}")
        _text_page(pdf, "\n".join(head), fontsize=8)

        for d in runs:
            lines = [f"RUN {d['run']}  (seed {d['seed']})", "",
                     f"  num_epochs        = {d['num_epochs']}",
                     f"  grok_epoch        = {d['grok_epoch']}",
                     f"  MA crossover      = {d['ma_crossover_epoch']}",
                     f"  MA-of-MA trigger  = {d['trigger_epoch']}",
                     f"  noise_floor       = {d['noise_floor']}",
                     f"  detection_epoch   = {d['detection_epoch']}", "",
                     "  --- scalar series stats ---",
                     _series_stats("train_acc", d["train_acc"]),
                     _series_stats("test_acc", d["test_acc"]),
                     _series_stats("loss", d["loss"]),
                     _series_stats("l2_norm", d["l2_norm"])]
            if d["l2_norm_smoothed"] is not None:
                lines.append(_series_stats("l2_norm_smoothed", d["l2_norm_smoothed"]))
            if d["accel_raw"] is not None:
                lines += [_series_stats("accel_raw", d["accel_raw"]),
                          _series_stats("accel_smoothed", d["accel_smoothed"]),
                          _series_stats("accel_double", d["accel_double"])]
            lines.append("")
            lines.append("  --- dropout gap by rate ---")
            for j, r in enumerate(d["dropout_rates"]):
                lines.append(_series_stats(f"gap p={r:g}", d["dropout_gap_by_rate"][j]))
            if d["fast_ma"] is not None:
                lines += ["", "  --- L2 MA grid stats ---",
                          _series_stats("epoch_grid", d["epoch_grid"]),
                          _series_stats("fast_ma", d["fast_ma"]),
                          _series_stats("slow_ma", d["slow_ma"]),
                          _series_stats("fast_ma_of_slow_ma", d["fast_ma_of_slow_ma"]),
                          _series_stats("ma_of_ma_diff", d["ma_of_ma_diff"])]
            _text_page(pdf, "\n".join(lines), fontsize=7)

        # --- literal per-epoch dump ---
        for d in runs:
            n = d["run"]
            ne = d["num_epochs"]
            rates = d["dropout_rates"]

            # Table A: core per-epoch series
            hdr = (f"{'epoch':>6} {'train_acc':>11} {'test_acc':>11} {'loss':>14} "
                   f"{'l2_norm':>12} {'l2_smoothed':>13}")
            rows = []
            lsm = d["l2_norm_smoothed"]
            for e in range(ne):
                sm = f"{lsm[e]:>13.6f}" if lsm is not None and e < len(lsm) else f"{'':>13}"
                rows.append(f"{e + 1:>6} {d['train_acc'][e]:>11.6f} {d['test_acc'][e]:>11.6f} "
                            f"{d['loss'][e]:>14.6e} {d['l2_norm'][e]:>12.6f} {sm}")
            _dump_table(pdf, f"RUN {n} - Table A: core per-epoch series", hdr, rows)

            # Table B: dropout sweep per epoch (gap + train acc per rate + clean acc)
            gcols = " ".join(f"{'gap p=' + format(r, 'g'):>12}" for r in rates)
            tcols = " ".join(f"{'ta p=' + format(r, 'g'):>12}" for r in rates)
            hdrB = f"{'epoch':>6} {gcols} {tcols} {'clean_acc':>12}"
            ta = d["dropout_train_acc_by_rate"]
            ca = d["dropout_eval_acc_by_rate"]
            rowsB = []
            for k, e in enumerate(d["dropout_gap_epochs"]):
                g = " ".join(f"{d['dropout_gap_by_rate'][j][k]:>12.6f}" for j in range(len(rates)))
                if ta is not None:
                    t = " ".join(f"{ta[j][k]:>12.6f}" for j in range(len(rates)))
                    cl = f"{ca[0][k]:>12.6f}"
                else:
                    t = " ".join(f"{'':>12}" for _ in rates)
                    cl = f"{'':>12}"
                rowsB.append(f"{int(e):>6} {g} {t} {cl}")
            _dump_table(pdf, f"RUN {n} - Table B: dropout sweep per epoch", hdrB, rowsB)

            # Table C: L2 acceleration per epoch
            if d["accel_raw"] is not None:
                hdrC = f"{'epoch':>6} {'accel_raw':>15} {'accel_smoothed':>16} {'accel_double':>16}"
                ar, asm, ad = d["accel_raw"], d["accel_smoothed"], d["accel_double"]
                rowsC = [f"{e + 1:>6} {ar[e]:>15.6e} {asm[e]:>16.6e} {ad[e]:>16.6e}"
                         for e in range(len(ar))]
                _dump_table(pdf, f"RUN {n} - Table C: L2 acceleration per epoch", hdrC, rowsC)

            # Table D: L2 MA grid (own length)
            if d["fast_ma"] is not None:
                hdrD = (f"{'epoch_grid':>14} {'fast_ma':>14} {'slow_ma':>14} "
                        f"{'fast_ma_of_slow':>16} {'ma_of_ma_diff':>16}")
                eg = d["epoch_grid"]
                rowsD = [f"{eg[k]:>14.4f} {d['fast_ma'][k]:>14.6f} {d['slow_ma'][k]:>14.6f} "
                         f"{d['fast_ma_of_slow_ma'][k]:>16.6f} {d['ma_of_ma_diff'][k]:>16.6e}"
                         for k in range(len(eg))]
                _dump_table(pdf, f"RUN {n} - Table D: L2 moving-average grid", hdrD, rowsD)

    return DATA_PDF


# ----------------------------------------------------------------------
# CSV export
# ----------------------------------------------------------------------

def write_csvs(runs):
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for d in runs:
        n = d["run"]
        rates = d["dropout_rates"]
        ne = d["num_epochs"]

        with open(CSV_DIR / f"run_{n}_timeseries.csv", "w", newline="") as f:
            w = csv.writer(f)
            cols = ["epoch", "train_acc", "test_acc", "loss", "l2_norm", "l2_norm_smoothed",
                    "accel_raw", "accel_smoothed", "accel_double"]
            cols += [f"gap_p{r:g}" for r in rates]
            cols += [f"train_acc_p{r:g}" for r in rates] + ["clean_acc"]
            w.writerow(cols)
            lsm = d["l2_norm_smoothed"]; ar = d["accel_raw"]; asm = d["accel_smoothed"]; ad = d["accel_double"]
            gbr = d["dropout_gap_by_rate"]; ta = d["dropout_train_acc_by_rate"]; ca = d["dropout_eval_acc_by_rate"]
            for e in range(ne):
                row = [e + 1, d["train_acc"][e], d["test_acc"][e], d["loss"][e], d["l2_norm"][e],
                       lsm[e] if lsm is not None and e < len(lsm) else "",
                       ar[e] if ar is not None and e < len(ar) else "",
                       asm[e] if asm is not None and e < len(asm) else "",
                       ad[e] if ad is not None and e < len(ad) else ""]
                row += [gbr[j][e] for j in range(len(rates))]
                if ta is not None:
                    row += [ta[j][e] for j in range(len(rates))] + [ca[0][e]]
                else:
                    row += ["" for _ in rates] + [""]
                w.writerow(row)

        if d["fast_ma"] is not None:
            with open(CSV_DIR / f"run_{n}_l2_ma_grid.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch_grid", "fast_ma", "slow_ma", "fast_ma_of_slow_ma", "ma_of_ma_diff"])
                for k in range(len(d["epoch_grid"])):
                    w.writerow([d["epoch_grid"][k], d["fast_ma"][k], d["slow_ma"][k],
                                d["fast_ma_of_slow_ma"][k], d["ma_of_ma_diff"][k]])

    with open(CSV_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "num_epochs", "grok_epoch", "ma_crossover_epoch",
                    "ma_of_ma_trigger_epoch", "noise_floor",
                    "final_train_acc", "final_test_acc", "final_loss",
                    "l2_initial", "l2_final",
                    *[f"final_gap_p{r:g}" for r in runs[0]["dropout_rates"]]])
        for d in runs:
            finals = [d["dropout_gap_by_rate"][j][-1] for j in range(len(d["dropout_rates"]))]
            w.writerow([d["run"], d["seed"], d["num_epochs"], d["grok_epoch"],
                        d["ma_crossover_epoch"], d["trigger_epoch"], d["noise_floor"],
                        d["train_acc"][-1], d["test_acc"][-1], d["loss"][-1],
                        d["l2_norm"][0], d["l2_norm"][-1], *finals])
    return CSV_DIR


# ----------------------------------------------------------------------

def main():
    nums = _run_numbers()
    if not nums:
        print("No complete four-head runs under runs/four_head/. Nothing to report.")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading complete runs: {nums}")
    runs = [load_run(n) for n in nums]

    print("Building master_plots.pdf ...")
    build_plots_pdf(runs)
    print(f"  -> {PLOTS_PDF}")

    print("Building master_data.pdf (full per-epoch dump - this can be large) ...")
    build_data_pdf(runs)
    print(f"  -> {DATA_PDF}")

    print("Writing CSVs ...")
    write_csvs(runs)
    print(f"  -> {CSV_DIR}/  (timeseries + l2_ma_grid + summary)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
