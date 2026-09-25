#!/usr/bin/env python3
"""
analyze_nanda_unified.py
========================

Post-hoc analysis + plots for the 5-seed Nanda-Unified run (L2 Norm, Dropout,
Spectral, AGE, HTSR Alpha). Reads only saved results, trains nothing.

For every predictor signal it asks one question: does the signal move BEFORE
the model groks (test acc first > 0.9), and does the moment it moves follow
the grok epoch from seed to seed?

Event epochs (all computed from the saved per-checkpoint arrays):
  * Implemented predictor outputs, taken straight from summary.json
    (L2 MA-crossover, L2 MA-of-MA zero crossing, dropout-variance peak,
    spectral k90-min / alignment-max, AGE NC1-min, HTSR mean-alpha minimum).
  * "Fraction-of-change" epochs: first epoch at which a signal has covered
    10% / 50% / 90% of the way from its starting value to its final extreme.
    NC1 is measured in log10 because it falls by three orders of magnitude.

lead = grok_epoch - event_epoch   (positive = signal fired before grok)

Usage (from the repo root):
    python 06_Code/scripts/analyze_nanda_unified.py
Outputs go to 08_Experiments/results/nanda_unified/analysis/.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO_ROOT, "08_Experiments", "results", "nanda_unified")
OUT = os.path.join(RESULTS, "analysis")
FRACTIONS = (0.1, 0.5, 0.9)


def load_seed(seed):
    d = os.path.join(RESULTS, f"seed_{seed}")
    with open(os.path.join(d, "summary.json")) as handle:
        summary = json.load(handle)
    ld = lambda sub, name: np.load(os.path.join(d, sub, name))
    return {
        "summary": summary,
        "grok": summary["grok_epoch"],
        "test_acc": ld("training", "test_acc_history.npy"),
        "train_acc": ld("training", "train_acc_history.npy"),
        "sum_w2": ld("l2_norm", "sum_w2_history.npy"),
        "var_ep": ld("dropout", "dropout_variance_checkpoints.npy"),
        "var": ld("dropout", "dropout_variance_history.npy"),
        "sp_ep": ld("spectral", "spectral_checkpoints.npy"),
        "align": ld("spectral", "spectral_alignment.npy"),
        "k90": ld("spectral", "spectral_k90.npy"),
        "age_ep": ld("age", "age_checkpoints.npy"),
        "nc1": ld("age", "age_nc1.npy"),
        "htsr_ep": ld("htsr", "htsr_checkpoints.npy"),
        "alpha": ld("htsr", "htsr_alpha.npy"),
    }


def smooth(y, window):
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fraction_epochs(epochs, y, fractions=FRACTIONS):
    """First epoch at which y has covered `f` of the way from its start to
    its farthest value. Returns {f: epoch}."""
    y0 = np.median(y[:5])
    extreme = y[np.argmax(np.abs(y - y0))]
    sign = np.sign(extreme - y0)
    out = {}
    for f in fractions:
        target = y0 + f * (extreme - y0)
        idx = np.nonzero(sign * (y - target) >= 0)[0]
        out[f] = int(epochs[idx[0]])
    return out


def l2_decline_epoch(sum_w2, fraction=0.5):
    """First epoch after the (smoothed) sum-of-squares peak at which it has
    fallen `fraction` of the way from that peak to its final value."""
    y = smooth(sum_w2, 200)
    peak = int(np.argmax(y))
    target = y[peak] - fraction * (y[peak] - y[-1])
    idx = np.nonzero(y[peak:] <= target)[0]
    return peak, int(peak + idx[0])


def build_events(runs):
    """events[predictor_signal][seed] = epoch."""
    ev = {}

    def put(name, seed, epoch):
        ev.setdefault(name, {})[seed] = float(epoch)

    for s, r in enumerate(runs):
        sm = r["summary"]
        put("L2: MA crossover (implemented)", s, sm["l2_predictor"]["ma_crossover_epoch"])
        put("L2: MA-of-MA zero crossing (implemented)", s, sm["l2_predictor"]["ma_of_ma_zero_crossing_epoch"])
        _, half = l2_decline_epoch(r["sum_w2"], 0.5)
        put("L2: sum(w^2) half-way down from peak", s, half)

        put("Dropout: variance peak (implemented)", s, sm["dropout_variance_predictor"]["variance_peak_epoch"])
        fr = fraction_epochs(r["var_ep"], r["var"])
        put("Dropout: variance 10% of rise", s, fr[0.1])
        put("Dropout: variance 50% of rise", s, fr[0.5])

        put("Spectral: k90 minimum (implemented)", s, sm["spectral_predictor"]["k_90_min_epoch"])
        put("Spectral: alignment maximum (implemented)", s, sm["spectral_predictor"]["alignment_max_epoch"])
        fr = fraction_epochs(r["sp_ep"], r["align"])
        put("Spectral: alignment 10% of rise", s, fr[0.1])
        put("Spectral: alignment 50% of rise", s, fr[0.5])
        fr = fraction_epochs(r["sp_ep"], r["k90"])
        put("Spectral: k90 50% of fall", s, fr[0.5])

        put("AGE: NC1 minimum (implemented)", s, sm["age_predictor"]["nc1_min_epoch"])
        fr = fraction_epochs(r["age_ep"], np.log10(r["nc1"]))
        put("AGE: log10 NC1 10% of fall", s, fr[0.1])
        put("AGE: log10 NC1 50% of fall", s, fr[0.5])

        # HTSR Alpha: published-style rule only (epoch of lowest mean alpha).
        # No fraction-of-change level is added (scope decision: no new rules).
        put("HTSR: mean alpha minimum (implemented)", s, sm["htsr_predictor"]["alpha_min_epoch"])
    return ev


def rank_corr(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def summarise(ev, grok):
    rows = []
    g = np.array(grok, dtype=float)
    for name, per_seed in ev.items():
        e = np.array([per_seed[s] for s in range(len(grok))])
        lead = g - e
        pear = float(np.corrcoef(e, g)[0, 1]) if e.std() > 0 else float("nan")
        rows.append({
            "signal": name,
            "epochs": e.tolist(),
            "lead": lead.tolist(),
            "seeds_before_grok": int((lead > 0).sum()),
            "spearman": rank_corr(e, g),
            "pearson": pear,
            "epoch_cv": float(e.std() / e.mean()),
        })
    return rows


def print_table(rows, grok):
    print(f"\ngrok epochs: {grok}")
    print(f"{'signal':<46}{'event epoch per seed':<38}{'before':>7}{'rho':>7}{'r':>7}")
    for r in rows:
        ep = " ".join(f"{int(x):>6}" for x in r["epochs"])
        print(f"{r['signal']:<46}{ep:<38}{r['seeds_before_grok']:>5}/5{r['spearman']:>7.2f}{r['pearson']:>7.2f}")


# ─────────────────────────── plots ───────────────────────────

def norm01(y):
    lo, hi = float(np.min(y)), float(np.max(y))
    return (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)


def plot_curves(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    cols = plt.cm.tab10(np.arange(len(runs)))
    for s, r in enumerate(runs):
        ax[0].plot(r["test_acc"], color=cols[s], lw=1.2, label=f"seed {s}")
        ax[0].axvline(r["grok"], color=cols[s], ls=":", lw=0.9)
        ax[1].plot(r["train_acc"], color=cols[s], lw=1.2)
    ax[0].set(title="Test accuracy (dotted = grok epoch, test acc > 0.9)", xlabel="epoch", ylabel="accuracy")
    ax[1].set(title="Train accuracy", xlabel="epoch", ylabel="accuracy")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_signals(runs, ev, path):
    cols = [
        ("Weight norm  sum(w^2)", lambda r: (np.arange(len(r["sum_w2"])), r["sum_w2"]),
         [("L2: MA crossover (implemented)", "tab:blue"),
          ("L2: MA-of-MA zero crossing (implemented)", "tab:cyan")]),
        ("Dropout variance", lambda r: (r["var_ep"], r["var"]),
         [("Dropout: variance peak (implemented)", "tab:orange")]),
        ("Spectral alignment", lambda r: (r["sp_ep"], r["align"]),
         [("Spectral: alignment maximum (implemented)", "tab:green"),
          ("Spectral: alignment 50% of rise", "tab:olive")]),
        ("AGE  NC1 (log scale)", lambda r: (r["age_ep"], r["nc1"]),
         [("AGE: NC1 minimum (implemented)", "tab:red"),
          ("AGE: log10 NC1 50% of fall", "tab:pink")]),
        ("HTSR  mean alpha", lambda r: (r["htsr_ep"], r["alpha"]),
         [("HTSR: mean alpha minimum (implemented)", "tab:purple")]),
    ]
    n = len(runs)
    fig, axes = plt.subplots(n, len(cols), figsize=(4.4 * len(cols), 2.4 * n), sharex=True)
    for s, r in enumerate(runs):
        for c, (title, getter, marks) in enumerate(cols):
            ax = axes[s, c]
            x, y = getter(r)
            ax.plot(x, y, color="black", lw=0.9)
            if c == 3:
                ax.set_yscale("log")
            ax.axvline(r["grok"], color="crimson", ls="--", lw=1.2, label="grok")
            for name, colour in marks:
                ax.axvline(ev[name][s], color=colour, lw=1.1, label=name.split(":")[1].strip()[:26])
            if s == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"seed {s}")
            if s == n - 1:
                ax.set_xlabel("epoch")
            if s == 0:
                ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("Predictor signals vs grok epoch (red dashed = grok, coloured lines = predictor event)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(ev, grok, path):
    show = [
        "L2: MA crossover (implemented)",
        "L2: MA-of-MA zero crossing (implemented)",
        "Dropout: variance peak (implemented)",
        "Spectral: alignment maximum (implemented)",
        "Spectral: alignment 10% of rise",
        "AGE: NC1 minimum (implemented)",
        "AGE: log10 NC1 10% of fall",
        "HTSR: mean alpha minimum (implemented)",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    lim = 42000
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1)
    ax.text(28000, 24500, "fires at grok", color="grey", rotation=33, fontsize=8)
    ax.fill_between([0, lim], [0, lim], 0, color="tab:green", alpha=0.06)
    ax.text(1500, 900, "below the line = fires BEFORE grok", color="tab:green", fontsize=8)
    for name, mk in zip(show, "os^Dv<>P"):
        e = [ev[name][s] for s in range(len(grok))]
        ax.scatter(grok, e, marker=mk, s=55, label=name, alpha=0.85)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="grok epoch of the seed", ylabel="predictor event epoch")
    ax.set_title("Does the predictor fire before grok, and does it follow grok?")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_leads(ev, grok, path):
    names = [
        "L2: MA crossover (implemented)",
        "L2: MA-of-MA zero crossing (implemented)",
        "Dropout: variance peak (implemented)",
        "Spectral: alignment maximum (implemented)",
        "Spectral: alignment 10% of rise",
        "AGE: NC1 minimum (implemented)",
        "AGE: log10 NC1 10% of fall",
        "HTSR: mean alpha minimum (implemented)",
    ]
    n = len(grok)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / n
    for s in range(n):
        leads = [grok[s] - ev[nm][s] for nm in names]
        ax.bar(np.arange(len(names)) + s * width, leads, width, label=f"seed {s}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
    ax.set_xticklabels([nm.replace(" (implemented)", "") for nm in names], rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="lead time = grok epoch - event epoch   (positive = early warning)",
           title="Lead time per predictor and seed")
    ax.legend(fontsize=7, ncol=n)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = sorted(int(n.split("_")[1]) for n in os.listdir(RESULTS) if n.startswith("seed_"))
    runs = [load_seed(s) for s in seeds]
    grok = [r["grok"] for r in runs]
    ev = build_events(runs)
    rows = summarise(ev, grok)
    print_table(rows, grok)

    with open(os.path.join(OUT, "predictor_events.json"), "w") as handle:
        json.dump({"grok_epochs": grok, "signals": rows}, handle, indent=2)

    plot_curves(runs, os.path.join(OUT, "01_grokking_curves.png"))
    plot_signals(runs, ev, os.path.join(OUT, "02_signals_vs_grok.png"))
    plot_scatter(ev, grok, os.path.join(OUT, "03_event_vs_grok_scatter.png"))
    plot_leads(ev, grok, os.path.join(OUT, "04_lead_times.png"))
    print(f"\nWrote plots and predictor_events.json to {OUT}")


if __name__ == "__main__":
    main()
