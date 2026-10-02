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
        "wpca_ep": ld("weight_pca", "weight_pca_checkpoints.npy"),
        "norm_eff_rank": ld("weight_pca", "weight_pca_norm_eff_rank.npy"),
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

        # Weight-PCA: PRIMARY rule only (global minimum of mean NormEffRank),
        # frozen 2026-09-17, same precedent as HTSR above. The exploratory
        # steepest-drop / threshold-90 epochs in weight_pca_predictor[
        # "exploratory"] are deliberately NOT added here — they are thesis-
        # appendix / sensitivity-analysis material, not part of the scored table.
        put("Weight-PCA: effective-rank minimum (implemented)", s,
            sm["weight_pca_predictor"]["norm_eff_rank_min_epoch"])

        # Higher-MI (Pomarico et al. 2025; see src/predictors/higher_mi.py).
        # PRIMARY rule, frozen 2026-10-03: global MAXIMUM of the population
        # O-information signal Omega_pop(t) (peak redundancy) -- same
        # "one PRIMARY rule declared in the module" convention as HTSR /
        # Weight-PCA above. Always defined (find_global_maximum_epoch has
        # no None case), so it's safe inside this unconditional per-seed loop.
        put("Higher-MI: population omega maximum (implemented)", s,
            sm["higher_mi_predictor"]["population_omega_max_epoch"])

    # EXPLORATORY Higher-MI signals, added OUTSIDE the per-seed loop above.
    # Both can legitimately come back None for a seed (population
    # zero-crossing: the signal may never cross from negative to positive
    # -- this is exactly why it is EXPLORATORY, not PRIMARY, per
    # higher_mi.py's "EVENT RULE" section; temporal-control max: only None
    # if that seed's run had fewer checkpoints than the W=100 window,
    # which does not happen on the real 401-checkpoint grid but is
    # guarded in run_nanda_benchmark.py regardless). summarise() below
    # needs every seed to have a value for every signal name in `ev`, so
    # each of these is added only if ALL seeds produced a value --
    # otherwise it is left out of the table with a console note, rather
    # than crashing on a per_seed[s] KeyError for whichever seed is None.
    if runs and all("higher_mi_predictor" in r["summary"] for r in runs):
        hmi = [r["summary"]["higher_mi_predictor"] for r in runs]

        zc_vals = [b["exploratory"]["population_omega_zero_crossing_epoch"] for b in hmi]
        if all(v is not None for v in zc_vals):
            for s, v in enumerate(zc_vals):
                put("Higher-MI: population omega zero-crossing (exploratory)", s, v)
        else:
            missing = [s for s, v in enumerate(zc_vals) if v is None]
            print(f"[build_events] Higher-MI population zero-crossing (exploratory) "
                  f"left out of the table -- undefined (never crossed) for seed(s) {missing}")

        tctrl_vals = [b["exploratory"]["temporal_control"]["omega_max_epoch"] for b in hmi]
        if all(v is not None for v in tctrl_vals):
            for s, v in enumerate(tctrl_vals):
                put("Higher-MI: temporal-control omega maximum (exploratory)", s, v)
        else:
            missing = [s for s, v in enumerate(tctrl_vals) if v is None]
            print(f"[build_events] Higher-MI temporal-control maximum (exploratory) "
                  f"left out of the table -- undefined for seed(s) {missing}")
    return ev


def rank_corr(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# Frozen, synthetic-control-validated evaluation rule (context.md,
# 2026-09-29 entry; run_control_experiment.py::evaluate / RHO_THRESH).
# Copied here deliberately rather than imported — see comment above.
RHO_THRESH = 0.8999999999999998  # 99th percentile, Run 2 calibration, frozen

def evaluate_frozen_protocol(g_in, p_in, margin=100.0):
    """Same logic as run_control_experiment.py::evaluate(). g_in, p_in:
    arrays, length 5 (grok epoch, predictor event epoch per seed)."""
    from scipy.stats import spearmanr
    g_arr = np.asarray(g_in, dtype=float)
    p_arr = np.asarray(p_in, dtype=float)
    lead = p_arr < (g_arr - margin)
    lead_frac = float(np.mean(lead))
    rel_lead = (g_arr - p_arr) / g_arr
    mean_rel_lead = float(np.mean(rel_lead))
    if len(g_arr) >= 3 and np.std(p_arr) > 1e-9:
        rho, _ = spearmanr(g_arr, p_arr)
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0
    return {"lead_frac": lead_frac, "rho": rho, "mean_rel_lead": mean_rel_lead}


# Signals with an explicitly frozen PRIMARY event rule (declared in the
# predictor's own module docstring -- see e.g. weight_pca.py's "PRIMARY"
# label / htsr_alpha.py's single-rule-only convention, both frozen
# 2026-09-17, context.md). Only these are scored against the frozen
# protocol and get a verdict persisted into predictor_events.json.
#
# L2 Norm, Dropout-Variance, Spectral and AGE predate this convention
# (their own modules never declared one rule as PRIMARY) and were each
# closed by separate reasoning recorded in context.md instead -- NOT
# backfilled here, since guessing which of their several candidate
# rules was "the" scored one risks contradicting an already-closed
# verdict. When a predictor's own module declares a PRIMARY rule
# (Higher-MI, Commutator Defect should follow the same convention),
# add its "(implemented)" signal name here -- nothing else changes.
PRIMARY_SIGNALS = [
    "HTSR: mean alpha minimum (implemented)",
    "Weight-PCA: effective-rank minimum (implemented)",
    "Higher-MI: population omega maximum (implemented)",
]


def summarise(ev, grok):
    rows = []
    g = np.array(grok, dtype=float)
    for name, per_seed in ev.items():
        e = np.array([per_seed[s] for s in range(len(grok))])
        lead = g - e
        pear = float(np.corrcoef(e, g)[0, 1]) if e.std() > 0 else float("nan")
        row = {
            "signal": name,
            "epochs": e.tolist(),
            "lead": lead.tolist(),
            "seeds_before_grok": int((lead > 0).sum()),
            "spearman": rank_corr(e, g),
            "pearson": pear,
            "epoch_cv": float(e.std() / e.mean()),
        }
        if name in PRIMARY_SIGNALS:
            # Official frozen-protocol verdict (evaluate_frozen_protocol
            # above): PASS iff lead_frac >= 0.8 AND rho >= RHO_THRESH AND
            # 0.05 <= mean_rel_lead <= 0.9. Persisted here (not just
            # printed) so it survives in predictor_events.json.
            verdict = evaluate_frozen_protocol(grok, e.tolist())
            verdict["passed"] = (verdict["lead_frac"] >= 0.8
                                  and verdict["rho"] >= RHO_THRESH
                                  and 0.05 <= verdict["mean_rel_lead"] <= 0.9)
            row["frozen_protocol_verdict"] = verdict
        rows.append(row)
    return rows


def print_table(rows, grok):
    print(f"\ngrok epochs: {grok}")
    print(f"{'signal':<46}{'event epoch per seed':<38}{'before':>7}{'rho':>7}{'r':>7}")
    for r in rows:
        ep = " ".join(f"{int(x):>6}" for x in r["epochs"])
        print(f"{r['signal']:<46}{ep:<38}{r['seeds_before_grok']:>5}/5{r['spearman']:>7.2f}{r['pearson']:>7.2f}")


def print_verdicts(rows):
    """Prints (and, via `rows` already being json.dump'd in main(), persists)
    the official frozen-protocol verdict for every PRIMARY_SIGNALS entry
    present in this run. Generalises the old hardcoded Weight-PCA-only
    block so a future predictor needs only a PRIMARY_SIGNALS entry, no
    changes here."""
    scored = [r for r in rows if "frozen_protocol_verdict" in r]
    if not scored:
        return
    print("\nOfficial frozen-protocol verdicts (PRIMARY rule only, see PRIMARY_SIGNALS):")
    for r in scored:
        v = r["frozen_protocol_verdict"]
        print(f"\n{r['signal']}")
        print(f"  lead_frac     = {v['lead_frac']:.4f}  (need >= 0.8)")
        print(f"  rho           = {v['rho']:.4f}  (need >= {RHO_THRESH:.4f})")
        print(f"  mean_rel_lead = {v['mean_rel_lead']:.4f}  (need in [0.05, 0.9])")
        print(f"  VERDICT = {'PASS' if v['passed'] else 'FAIL'}")


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
        ("Weight-PCA  mean NormEffRank", lambda r: (r["wpca_ep"], r["norm_eff_rank"]),
         [("Weight-PCA: effective-rank minimum (implemented)", "tab:brown")]),
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
        "Weight-PCA: effective-rank minimum (implemented)",
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
        "Weight-PCA: effective-rank minimum (implemented)",
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
    print_verdicts(rows)

    with open(os.path.join(OUT, "predictor_events.json"), "w") as handle:
        json.dump({"grok_epochs": grok, "signals": rows}, handle, indent=2)

    # Higher-MI-only aggregate, requested alongside the full
    # predictor_events.json above (which already carries Higher-MI's row
    # too, via the generic PRIMARY_SIGNALS / build_events wiring).
    higher_mi_rows = [r for r in rows if r["signal"].startswith("Higher-MI")]
    if higher_mi_rows:
        agg_path = os.path.join(RESULTS, "aggregate_higher_mi.json")
        with open(agg_path, "w") as handle:
            json.dump({"grok_epochs": grok, "signals": higher_mi_rows}, handle, indent=2)
        print(f"Wrote {agg_path}")

    plot_curves(runs, os.path.join(OUT, "01_grokking_curves.png"))
    plot_signals(runs, ev, os.path.join(OUT, "02_signals_vs_grok.png"))
    plot_scatter(ev, grok, os.path.join(OUT, "03_event_vs_grok_scatter.png"))
    plot_leads(ev, grok, os.path.join(OUT, "04_lead_times.png"))
    print(f"\nWrote plots and predictor_events.json (now including each"
          f" PRIMARY_SIGNALS verdict) to {OUT}")


if __name__ == "__main__":
    main()
