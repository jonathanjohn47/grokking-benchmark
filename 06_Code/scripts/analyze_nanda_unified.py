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
        "cd_ep": ld("commutator_defect", "commutator_defect_checkpoints.npy")
            if os.path.exists(os.path.join(d, "commutator_defect",
                                           "commutator_defect_checkpoints.npy"))
            else None,
        "cd_median": ld("commutator_defect", "commutator_defect_median.npy")
            if os.path.exists(os.path.join(d, "commutator_defect",
                                           "commutator_defect_median.npy"))
            else None,
    }


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


def build_events(runs):
    """events[predictor_signal][seed] = epoch (or None for a miss)."""
    ev = {}

    def put(name, seed, epoch):
        # None is a deliberate miss (Commutator Defect Xu onset never fired).
        ev.setdefault(name, {})[seed] = (
            None if epoch is None else float(epoch)
        )

    for s, r in enumerate(runs):
        sm = r["summary"]
        # ma_crossover_epoch is a real epoch on the log-uniform grid
        # (10 ** linspace), not an integer training step and not a grid
        # index. The decimals (about 113.41, 113.20, ...) are that grid
        # point. Checked against seed_*/l2_norm/epoch_grid.npy on 10 Nov 2026.
        put("L2: MA crossover (implemented)", s, sm["l2_predictor"]["ma_crossover_epoch"])
        put("L2: MA-of-MA zero crossing (implemented)", s, sm["l2_predictor"]["ma_of_ma_zero_crossing_epoch"])

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

        # Commutator Defect (Xu 2026): PRIMARY Xu onset on median D̃.
        # event_epoch may be None (miss) — kept as None, not coerced.
        if "commutator_defect_predictor" in sm and sm["commutator_defect_predictor"] is not None:
            put("Commutator Defect: Xu onset on median D (implemented)", s,
                sm["commutator_defect_predictor"].get("event_epoch"))

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

# Lead time scales with G (Xu arXiv:2602.16967; Golwala arXiv:2604.20923).
# No paper fixes the fraction c, so c is a config parameter (default 0.8,
# configs/nanda_unified.yaml evaluation.lead_fraction_c). The 10 Nov 2026
# sweep scores c in {0.7, 0.8, 0.9} and keeps the old fixed margin as a
# comparison. Seed i leads under the default rule when P_i < c * G_i.
DEFAULT_LEAD_C = 0.8
LEAD_MARGIN_EPOCHS = 100.0


def lead_flags(g_arr, p_arr, rule="fraction", c=DEFAULT_LEAD_C, margin=LEAD_MARGIN_EPOCHS):
    """Boolean lead mask, one entry per firing seed.

    rule="fraction": P_i < c * G_i (default). Lead time scales with G.
    rule="margin":   P_i < G_i - margin (the old fixed 100-epoch comparison).
    """
    g_arr = np.asarray(g_arr, dtype=float)
    p_arr = np.asarray(p_arr, dtype=float)
    if rule == "fraction":
        return p_arr < (float(c) * g_arr)
    if rule == "margin":
        return p_arr < (g_arr - float(margin))
    raise ValueError(f"unknown lead rule {rule!r}")


def evaluate_frozen_protocol(g_in, p_in, margin=LEAD_MARGIN_EPOCHS, rule="fraction", c=DEFAULT_LEAD_C):
    """Same logic as run_control_experiment.py::evaluate(), with the fractional lead test.

    g_in, p_in: length-5 sequences (grok epoch, predictor event epoch).
    p_in entries may be None (no detection / miss).

    Lead rule (10 Nov 2026):
      - default rule="fraction", c=0.8: seed i leads if P_i < c * G_i
      - comparison rule="margin": seed i leads if P_i < G_i - margin
    Lead time scales with G (Xu arXiv:2602.16967; Golwala arXiv:2604.20923).
    No paper fixes c.

    Miss handling (matches run_control_experiment.py::evaluate), 11 Nov 2026:
      - a miss (P_i is None) counts as no lead in lead_frac
      - lead_frac = (number of leading seeds) / (number of seeds), so a
        miss stays in the denominator. 4 leads and 1 miss is 4/5 = 0.8,
        not 4/4 = 1.0
      - Spearman rho and mean relative lead still use only firing seeds,
        because a miss has no P_i
      - if fewer than 4 of 5 seeds fire -> FAIL immediately
        (lead_frac=0.0, rho=nan, mean_rel_lead=0.0)
      - negative relative leads stay signed (not dropped, not abs-valued)

    Other PASS conditions are unchanged: at least 4 firing seeds,
    lead_frac >= 0.8, rho >= RHO_THRESH, 0.05 <= mean relative lead <= 0.9.
    """
    from scipy.stats import spearmanr
    pairs = list(zip(g_in, p_in))
    n_seeds = len(pairs)
    valid = [(g, p) for g, p in pairs if p is not None]
    rule_meta = {
        "lead_rule": rule,
        "lead_c": float(c) if rule == "fraction" else None,
        "lead_margin": float(margin) if rule == "margin" else None,
    }
    if len(valid) < 4:
        return {
            "lead_frac": 0.0,
            "rho": float("nan"),
            "mean_rel_lead": 0.0,
            "n_firing": len(valid),
            **rule_meta,
        }
    g_arr = np.asarray([g for g, _ in valid], dtype=float)
    p_arr = np.asarray([p for _, p in valid], dtype=float)
    lead = lead_flags(g_arr, p_arr, rule=rule, c=c, margin=margin)
    # A miss is not a lead and stays in the denominator.
    lead_frac = float(np.sum(lead) / n_seeds) if n_seeds else 0.0
    rel_lead = (g_arr - p_arr) / g_arr
    mean_rel_lead = float(np.mean(rel_lead))
    if len(g_arr) >= 3 and np.std(p_arr) > 1e-9:
        rho, _ = spearmanr(g_arr, p_arr)
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0
    return {
        "lead_frac": lead_frac,
        "rho": rho,
        "mean_rel_lead": mean_rel_lead,
        "n_firing": len(valid),
        **rule_meta,
    }


# Every stored event row is scored. Rows whose name contains "exploratory"
# are reported and do not count towards the predictor-level verdict.
# A predictor PASSES only if at least one of its other (pre-declared) rows
# passes. L2, Dropout, Spectral and AGE are included here; they are no
# longer skipped. 10 Nov 2026.
PREDICTORS = (
    ("L2:", "L2 Norm"),
    ("Dropout:", "Dropout"),
    ("Spectral:", "Spectral"),
    ("AGE:", "AGE"),
    ("HTSR:", "HTSR Alpha"),
    ("Weight-PCA:", "Weight-PCA"),
    ("Higher-MI:", "Higher-MI"),
    ("Commutator Defect:", "Commutator Defect"),
)


def predictor_name(signal):
    for prefix, name in PREDICTORS:
        if signal.startswith(prefix):
            return name
    return None


def is_exploratory(signal):
    return "exploratory" in signal.lower()


def _rho_ok(rho):
    return not (isinstance(rho, float) and np.isnan(rho)) and rho >= RHO_THRESH


def attach_pass_checks(verdict):
    """Four separate PASS checks. The row passes only when every one is true."""
    rho = verdict["rho"]
    checks = {
        "n_firing_at_least_4": verdict["n_firing"] >= 4,
        "lead_frac_at_least_0.8": verdict["lead_frac"] >= 0.8,
        "rho_at_least_threshold": bool(_rho_ok(rho)),
        "mean_rel_lead_in_window": 0.05 <= verdict["mean_rel_lead"] <= 0.9,
    }
    verdict["checks"] = checks
    verdict["passed"] = all(checks.values())
    verdict["verdict"] = "PASS" if verdict["passed"] else "FAIL"
    if isinstance(rho, float) and np.isnan(rho):
        verdict["rho"] = None
    return verdict


def summarise(ev, grok, lead_c=DEFAULT_LEAD_C):
    rows = []
    g = np.array(grok, dtype=float)
    for name, per_seed in ev.items():
        # Preserve None misses (do not coerce to NaN for the verdict path).
        epochs_raw = [per_seed.get(s) for s in range(len(grok))]
        firing = [(gi, ei) for gi, ei in zip(g, epochs_raw) if ei is not None]
        if firing:
            e_fire = np.array([ei for _, ei in firing], dtype=float)
            g_fire = np.array([gi for gi, _ in firing], dtype=float)
            lead_vals = [
                None if ei is None else float(gi - ei)
                for gi, ei in zip(g, epochs_raw)
            ]
            seeds_before = int(sum(
                1 for gi, ei in zip(g, epochs_raw)
                if ei is not None and (gi - ei) > 0
            ))
            pear = (
                float(np.corrcoef(e_fire, g_fire)[0, 1])
                if e_fire.std() > 0 and len(e_fire) >= 2 else float("nan")
            )
            spear = rank_corr(e_fire, g_fire) if len(e_fire) >= 2 else float("nan")
            epoch_cv = (
                float(e_fire.std() / e_fire.mean())
                if e_fire.mean() != 0 else float("nan")
            )
        else:
            lead_vals = [None] * len(grok)
            seeds_before = 0
            pear = float("nan")
            spear = float("nan")
            epoch_cv = float("nan")
        row = {
            "signal": name,
            "epochs": epochs_raw,
            "lead": lead_vals,
            "seeds_before_grok": seeds_before,
            "spearman": spear,
            "pearson": pear,
            "epoch_cv": epoch_cv,
        }
        # Every predictor row is scored. Exploratory rows are reported
        # and do not count towards the predictor-level verdict.
        verdict = evaluate_frozen_protocol(
            grok, epochs_raw, rule="fraction", c=lead_c
        )
        attach_pass_checks(verdict)
        row["row_role"] = "exploratory" if is_exploratory(name) else "predeclared"
        row["counts_towards_predictor"] = row["row_role"] == "predeclared"
        row["frozen_protocol_verdict"] = verdict
        rows.append(row)
    return rows


def predictor_level_verdicts(rows):
    """A predictor PASSES only if at least one pre-declared row PASSES."""
    grouped = {name: [] for _, name in PREDICTORS}
    for row in rows:
        name = predictor_name(row["signal"])
        if name is None or "frozen_protocol_verdict" not in row:
            continue
        grouped[name].append(row)
    out = []
    for _, name in PREDICTORS:
        members = grouped[name]
        counting = [r for r in members if r["counts_towards_predictor"]]
        n_pass = sum(1 for r in counting if r["frozen_protocol_verdict"]["passed"])
        out.append({
            "predictor": name,
            "verdict": "PASS" if n_pass >= 1 else "FAIL",
            "n_predeclared_rows": len(counting),
            "n_predeclared_passing": n_pass,
            "rows": [
                {
                    "signal": r["signal"],
                    "role": r["row_role"],
                    "counts_towards_predictor": r["counts_towards_predictor"],
                    "P_i": r["epochs"],
                    "n_firing": r["frozen_protocol_verdict"]["n_firing"],
                    "lead_frac": r["frozen_protocol_verdict"]["lead_frac"],
                    "rho": r["frozen_protocol_verdict"]["rho"],
                    "mean_rel_lead": r["frozen_protocol_verdict"]["mean_rel_lead"],
                    "checks": r["frozen_protocol_verdict"]["checks"],
                    "verdict": r["frozen_protocol_verdict"]["verdict"],
                }
                for r in members
            ],
        })
    return out


def print_table(rows, grok):
    print(f"\ngrok epochs: {grok}")
    print(f"{'signal':<52}{'event epoch per seed':<38}{'before':>7}{'rho':>7}{'r':>7}")
    for r in rows:
        ep = " ".join(
            f"{'None':>6}" if x is None else f"{int(x):>6}"
            for x in r["epochs"]
        )
        print(
            f"{r['signal']:<52}{ep:<38}"
            f"{r['seeds_before_grok']:>5}/{len(grok)}"
            f"{r['spearman']:>7.2f}{r['pearson']:>7.2f}"
        )


def print_verdicts(rows, predictors):
    """Print every row verdict and the predictor-level verdict."""
    print("\nOfficial frozen-protocol verdicts (every stored row; exploratory does not count):")
    for r in rows:
        if "frozen_protocol_verdict" not in r:
            continue
        v = r["frozen_protocol_verdict"]
        rho = v["rho"]
        rho_s = "nan" if rho is None else f"{rho:.4f}"
        print(f"\n{r['signal']}  [{r['row_role']}]")
        print(f"  n_firing      = {v['n_firing']}  check={v['checks']['n_firing_at_least_4']}")
        print(f"  lead_frac     = {v['lead_frac']:.4f}  check={v['checks']['lead_frac_at_least_0.8']}")
        print(f"  rho           = {rho_s}  check={v['checks']['rho_at_least_threshold']}")
        print(f"  mean_rel_lead = {v['mean_rel_lead']:.4f}  check={v['checks']['mean_rel_lead_in_window']}")
        print(f"  VERDICT = {v['verdict']}")
    print("\nPredictor-level verdicts (PASS only if a pre-declared row PASSES):")
    for p in predictors:
        print(f"  {p['predictor']:<22} {p['verdict']}  ({p['n_predeclared_passing']}/{p['n_predeclared_rows']} pre-declared rows pass)")


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
        ("Commutator Defect  median D",
         lambda r: (r["cd_ep"], r["cd_median"])
         if r.get("cd_ep") is not None else (None, None),
         [("Commutator Defect: Xu onset on median D (implemented)", "tab:gray")]),
    ]
    # Drop panels whose data is missing on every seed (e.g. Commutator not
    # yet recomputed for this results tree).
    cols = [
        col for col in cols
        if not all(col[1](r)[0] is None for r in runs)
    ]
    n = len(runs)
    fig, axes = plt.subplots(n, len(cols), figsize=(4.4 * len(cols), 2.4 * n), sharex=True)
    if len(cols) == 1:
        axes = np.asarray(axes).reshape(n, 1)
    for s, r in enumerate(runs):
        for c, (title, getter, marks) in enumerate(cols):
            ax = axes[s, c]
            x, y = getter(r)
            if x is None or y is None:
                ax.set_visible(False)
                continue
            ax.plot(x, y, color="black", lw=0.9)
            if "NC1" in title:
                ax.set_yscale("log")
            ax.axvline(r["grok"], color="crimson", ls="--", lw=1.2, label="grok")
            for name, colour in marks:
                if name not in ev or s not in ev[name]:
                    continue
                epoch = ev[name][s]
                if epoch is None:
                    continue  # miss — no vertical line
                ax.axvline(epoch, color=colour, lw=1.1,
                           label=name.split(":")[1].strip()[:26])
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
        "Commutator Defect: Xu onset on median D (implemented)",
    ]
    show = [name for name in show if name in ev]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    lim = 42000
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1)
    ax.text(28000, 24500, "fires at grok", color="grey", rotation=33, fontsize=8)
    ax.fill_between([0, lim], [0, lim], 0, color="tab:green", alpha=0.06)
    ax.text(1500, 900, "below the line = fires BEFORE grok", color="tab:green", fontsize=8)
    markers = "os^Dv<>PXh"
    for name, mk in zip(show, markers):
        # Skip None misses — a miss has no (grok, event) point.
        pairs = [
            (grok[s], ev[name][s])
            for s in range(len(grok))
            if ev[name].get(s) is not None
        ]
        if not pairs:
            continue
        gx, ey = zip(*pairs)
        ax.scatter(gx, ey, marker=mk, s=55, label=name, alpha=0.85)
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
        "Commutator Defect: Xu onset on median D (implemented)",
    ]
    names = [nm for nm in names if nm in ev]
    n = len(grok)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / max(n, 1)
    for s in range(n):
        leads = []
        for nm in names:
            epoch = ev[nm].get(s)
            # Miss -> NaN bar (matplotlib skips); do not pretend lead=0.
            leads.append(float("nan") if epoch is None else grok[s] - epoch)
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


def load_lead_c_from_config():
    """Read evaluation.lead_fraction_c. Falls back to DEFAULT_LEAD_C."""
    path = os.path.join(REPO_ROOT, "06_Code", "configs", "nanda_unified.yaml")
    try:
        with open(path) as handle:
            for line in handle:
                stripped = line.split("#", 1)[0].strip()
                if stripped.startswith("lead_fraction_c:"):
                    return float(stripped.split(":", 1)[1].strip())
    except OSError:
        pass
    return DEFAULT_LEAD_C


def score_saved_run(lead_c=None, out_dir=None, write=True):
    """Official scorer. Reads saved seed results. Trains nothing.

    lead_c: fraction in P_i < c * G_i. None loads the config (default 0.8).
    write: when True, write predictor_events.json, frozen_protocol_verdicts.json,
    and the four plots under out_dir. The Higher-MI aggregate at the results
    root is rewritten only for the default analysis folder, so a c sweep
    does not replace the c = 0.8 summary.
    """
    if lead_c is None:
        lead_c = load_lead_c_from_config()
    lead_c = float(lead_c)
    if out_dir is None:
        out_dir = OUT if abs(lead_c - DEFAULT_LEAD_C) < 1e-12 else os.path.join(
            OUT, f"c_{lead_c:.1f}"
        )
    seeds = sorted(
        int(n.split("_")[1]) for n in os.listdir(RESULTS) if n.startswith("seed_")
    )
    runs = [load_seed(s) for s in seeds]
    grok = [r["grok"] for r in runs]
    ev = build_events(runs)
    rows = summarise(ev, grok, lead_c=lead_c)
    predictors = predictor_level_verdicts(rows)
    print_table(rows, grok)
    print_verdicts(rows, predictors)

    if write:
        os.makedirs(out_dir, exist_ok=True)
        events_path = os.path.join(out_dir, "predictor_events.json")
        with open(events_path, "w") as handle:
            json.dump({"grok_epochs": grok, "lead_c": lead_c, "signals": rows}, handle, indent=2)
        verdict_path = os.path.join(out_dir, "frozen_protocol_verdicts.json")
        with open(verdict_path, "w") as handle:
            json.dump({
                "lead_c": lead_c,
                "lead_rule": "P_i < c * G_i",
                "rho_thresh": RHO_THRESH,
                "grok_epochs": grok,
                "predictors": predictors,
            }, handle, indent=2)
        print(f"Wrote {verdict_path}")

        if os.path.abspath(out_dir) == os.path.abspath(OUT):
            higher_mi_rows = [r for r in rows if r["signal"].startswith("Higher-MI")]
            if higher_mi_rows:
                agg_path = os.path.join(RESULTS, "aggregate_higher_mi.json")
                with open(agg_path, "w") as handle:
                    json.dump({"grok_epochs": grok, "lead_c": lead_c, "signals": higher_mi_rows}, handle, indent=2)
                print(f"Wrote {agg_path}")

        plot_curves(runs, os.path.join(out_dir, "01_grokking_curves.png"))
        plot_signals(runs, ev, os.path.join(out_dir, "02_signals_vs_grok.png"))
        plot_scatter(ev, grok, os.path.join(out_dir, "03_event_vs_grok_scatter.png"))
        plot_leads(ev, grok, os.path.join(out_dir, "04_lead_times.png"))
        print(f"Wrote plots and verdicts to {out_dir}")

    return {"grok": grok, "rows": rows, "predictors": predictors, "lead_c": lead_c, "out_dir": out_dir}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Score saved Nanda-unified predictor events.")
    parser.add_argument(
        "--lead-c", type=float, default=None,
        help="Lead fraction c in P_i < c * G_i. Default: evaluation.lead_fraction_c (0.8).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output folder. Default: analysis/ at c=0.8, analysis/c_<value>/ otherwise.",
    )
    args = parser.parse_args(argv)
    score_saved_run(lead_c=args.lead_c, out_dir=args.out_dir, write=True)


if __name__ == "__main__":
    main()
