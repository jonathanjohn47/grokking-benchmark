"""
Synthetic control experiment for the pre-grok predictor evaluation criterion.

This does NOT touch any real predictor code or checkpoints. It only tests
whether the evaluation rule itself (lead + correlation + relative-lead
window, with a calibrated Spearman threshold) is a sensible pass/fail rule
by throwing known-good and known-bad synthetic signals at it.

Run:
    python run_control_experiment.py

Output folder: ./controls_results/
    summary_table.csv        one row per control
    plot_sensitivity.png     pass_rate vs noise_frac for Control D
    plot_null_distribution.png   histogram of random rho with RHO_THRESH line
    detailed_results.json    all raw numbers
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

np.random.seed(42)

# ---------------------------------------------------------------------------
# Fixed problem setup
# ---------------------------------------------------------------------------

Gs = np.array([12987, 6921, 9510, 11019, 23747], dtype=float)  # actual grok epochs, 5 seeds
T_MAX = 40000.0
MARGIN = 100.0
N_TRIALS_NOISY = 10000    # per noise level, Control D
N_RAND = 10000            # Control E, each variant
NOISE_FRACS = [0.0, 0.10, 0.25, 0.50, 1.0]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "controls_results")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Evaluation function (as specified — lead_frac, rho, mean_rel_lead)
# ---------------------------------------------------------------------------

def evaluate(g_in, p_in, margin=MARGIN):
    """
    g_in, p_in: arrays, length 5. p_in entries may be None (no detection).

    lead_frac: fraction of seeds where P_s < G_s - margin (predictor fired
        with a real safety margin before the actual grok epoch).
    rho: Spearman rank correlation between G and P across seeds. If P is
        constant (std == 0), correlation is undefined, so rho is forced to
        0.0 rather than NaN — a constant signal carries no seed-to-seed
        timing information, so it must not be rewarded with rho = NaN
        silently passing a downstream `rho >= threshold` check.
    mean_rel_lead: mean of (G - P) / G, i.e. how much of the pre-grok
        window, as a fraction of G, the predictor's lead covers.
    """
    valid = [(g, p) for g, p in zip(g_in, p_in) if p is not None]
    if len(valid) < 4:
        return {"pass": False, "lead_frac": 0.0, "rho": np.nan, "mean_rel_lead": 0.0}

    g_arr = np.array([x[0] for x in valid], dtype=float)
    p_arr = np.array([x[1] for x in valid], dtype=float)

    lead = p_arr < (g_arr - margin)
    lead_frac = float(np.mean(lead))
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
        "g": g_arr,
        "p": p_arr,
    }


def passes(result, rho_thresh):
    return (
        result["lead_frac"] >= 0.8
        and result["rho"] >= rho_thresh
        and 0.05 <= result["mean_rel_lead"] <= 0.9
    )


# ---------------------------------------------------------------------------
# Step 1 — calibrate RHO_THRESH from Control E Variant 1 (unconditional random)
# ---------------------------------------------------------------------------
# Rationale: instead of hand-picking a Spearman cutoff, generate 10,000
# fully random predictors (Ps ~ Uniform[0, T_MAX], independent of Gs) and
# take the 99th percentile of their rho distribution. Any real predictor
# must clear that bar to be called correlated with grokking, which caps
# the false-positive rate of the rho check at <= 1% by construction
# (for the unconditional variant; the conditional variant, Control E
# Variant 2, still runs higher because it always leads by construction —
# see its "verdict" in summary_table.csv).

random_rhos_unconditional = np.empty(N_RAND)
for i in range(N_RAND):
    p_rand = np.random.uniform(0, T_MAX, size=5)
    res = evaluate(Gs, p_rand)
    random_rhos_unconditional[i] = res["rho"]

RHO_THRESH = float(np.percentile(random_rhos_unconditional, 99))
print(f"Calibrated RHO_THRESH (99th percentile of {N_RAND} random rho values) = {RHO_THRESH:.4f}")


# ---------------------------------------------------------------------------
# Control A — Ideal perfect predictor: Ps = 0.5 * Gs
# ---------------------------------------------------------------------------

Ps_A = 0.5 * Gs
res_A = evaluate(Gs, Ps_A)
pass_A = passes(res_A, RHO_THRESH)


# ---------------------------------------------------------------------------
# Control B — Constant early signal: Ps = 5000 for all seeds
# ---------------------------------------------------------------------------

Ps_B = np.full(5, 5000.0)
res_B = evaluate(Gs, Ps_B)
pass_B = passes(res_B, RHO_THRESH)


# ---------------------------------------------------------------------------
# Control C — Post-grok signal: Ps = 1.5 * Gs
# ---------------------------------------------------------------------------

Ps_C = 1.5 * Gs
res_C = evaluate(Gs, Ps_C)
pass_C = passes(res_C, RHO_THRESH)


# ---------------------------------------------------------------------------
# Control D — Noisy-good predictor: Ps = 0.5*Gs + noise_frac*(0.5*Gs)*U[-1,1]
# ---------------------------------------------------------------------------
# Normalization: Gs ranges 6921..23747, a >3x spread. A fixed absolute noise
# (e.g. +-1000) would be a large fraction of the signal for the smallest
# seed and a small fraction for the largest — biased across seeds. Scaling
# noise by 0.5*Gs (the target value itself) keeps the noise fraction equal,
# in relative terms, for every seed.

control_D_summary = {}
control_D_raw = {}

for frac in NOISE_FRACS:
    lead_fracs = np.empty(N_TRIALS_NOISY)
    rhos = np.empty(N_TRIALS_NOISY)
    rel_leads = np.empty(N_TRIALS_NOISY)
    pass_flags = np.empty(N_TRIALS_NOISY, dtype=bool)

    for i in range(N_TRIALS_NOISY):
        U = np.random.uniform(-1.0, 1.0, size=5)
        base = 0.5 * Gs
        eps = frac * base * U
        p_trial = base + eps
        r = evaluate(Gs, p_trial)
        lead_fracs[i] = r["lead_frac"]
        rhos[i] = r["rho"]
        rel_leads[i] = r["mean_rel_lead"]
        pass_flags[i] = passes(r, RHO_THRESH)

    control_D_summary[frac] = {
        "lead_frac_mean": float(np.mean(lead_fracs)),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "mean_rel_lead": float(np.mean(rel_leads)),
        "pass_rate": float(np.mean(pass_flags)),
    }
    control_D_raw[frac] = {
        "lead_fracs": lead_fracs.tolist(),
        "rhos": rhos.tolist(),
        "rel_leads": rel_leads.tolist(),
    }


# ---------------------------------------------------------------------------
# Control E — Random predictor (false-positive rate estimate)
# ---------------------------------------------------------------------------
# Variant 1 (unconditional): Ps ~ Uniform[0, T_MAX] — tests lead AND rho.
#   (random_rhos_unconditional was already generated above for calibration;
#    reuse it here instead of drawing a fresh batch.)
# Variant 2 (conditional): Ps ~ Uniform[0, Gs] per seed — always leads by
#   construction, isolates the correlation check's false-positive rate.

lead_fracs_v1 = np.empty(N_RAND)
rel_leads_v1 = np.empty(N_RAND)
pass_flags_v1 = np.empty(N_RAND, dtype=bool)
for i in range(N_RAND):
    p_rand = np.random.uniform(0, T_MAX, size=5)
    r = evaluate(Gs, p_rand)
    lead_fracs_v1[i] = r["lead_frac"]
    rel_leads_v1[i] = r["mean_rel_lead"]
    pass_flags_v1[i] = passes(r, RHO_THRESH)
rhos_v1 = random_rhos_unconditional

lead_fracs_v2 = np.empty(N_RAND)
rhos_v2 = np.empty(N_RAND)
rel_leads_v2 = np.empty(N_RAND)
pass_flags_v2 = np.empty(N_RAND, dtype=bool)
for i in range(N_RAND):
    p_rand = np.random.uniform(0, Gs)  # per-seed upper bound, elementwise
    r = evaluate(Gs, p_rand)
    lead_fracs_v2[i] = r["lead_frac"]
    rhos_v2[i] = r["rho"]
    rel_leads_v2[i] = r["mean_rel_lead"]
    pass_flags_v2[i] = passes(r, RHO_THRESH)


# ---------------------------------------------------------------------------
# Assemble summary table
# ---------------------------------------------------------------------------

rows = []

rows.append({
    "Control": "A: Ideal (0.5*Gs)",
    "lead_frac": res_A["lead_frac"],
    "rho_mean": res_A["rho"],
    "rho_std": 0.0,
    "mean_rel_lead": res_A["mean_rel_lead"],
    "pass_rate": float(pass_A),
    "expected_pass_rate": "100%",
    "verdict": "PASS" if pass_A else "FAIL (BROKEN CRITERION)",
})

rows.append({
    "Control": "B: Constant (5000)",
    "lead_frac": res_B["lead_frac"],
    "rho_mean": res_B["rho"],
    "rho_std": 0.0,
    "mean_rel_lead": res_B["mean_rel_lead"],
    "pass_rate": float(pass_B),
    "expected_pass_rate": "0%",
    "verdict": "FAIL (as required)" if not pass_B else "PASS (BROKEN CRITERION)",
})

rows.append({
    "Control": "C: Post-grok (1.5*Gs)",
    "lead_frac": res_C["lead_frac"],
    "rho_mean": res_C["rho"],
    "rho_std": 0.0,
    "mean_rel_lead": res_C["mean_rel_lead"],
    "pass_rate": float(pass_C),
    "expected_pass_rate": "0%",
    "verdict": "FAIL (as required)" if not pass_C else "PASS (BROKEN CRITERION)",
})

for frac in NOISE_FRACS:
    s = control_D_summary[frac]
    rows.append({
        "Control": f"D: Noisy-good (frac={frac})",
        "lead_frac": s["lead_frac_mean"],
        "rho_mean": s["rho_mean"],
        "rho_std": s["rho_std"],
        "mean_rel_lead": s["mean_rel_lead"],
        "pass_rate": s["pass_rate"],
        "expected_pass_rate": "decreasing with noise_frac",
        "verdict": "see plot_sensitivity.png",
    })

rows.append({
    "Control": "E: Random Uniform[0,T_MAX] (Variant 1, unconditional)",
    "lead_frac": float(np.mean(lead_fracs_v1)),
    "rho_mean": float(np.mean(rhos_v1)),
    "rho_std": float(np.std(rhos_v1)),
    "mean_rel_lead": float(np.mean(rel_leads_v1)),
    "pass_rate": float(np.mean(pass_flags_v1)),
    "expected_pass_rate": "<=1% (FPR, by calibration construction)",
    "verdict": "OK" if np.mean(pass_flags_v1) <= 0.01 else "FPR ABOVE TARGET",
})

rows.append({
    "Control": "E: Random Uniform[0,Gs] (Variant 2, conditional, always leads)",
    "lead_frac": float(np.mean(lead_fracs_v2)),
    "rho_mean": float(np.mean(rhos_v2)),
    "rho_std": float(np.std(rhos_v2)),
    "mean_rel_lead": float(np.mean(rel_leads_v2)),
    "pass_rate": float(np.mean(pass_flags_v2)),
    "expected_pass_rate": "<=5% (FPR of rho check alone)",
    "verdict": "OK" if np.mean(pass_flags_v2) <= 0.05 else "FPR ABOVE TARGET",
})

# CSV (no pandas dependency — numpy/scipy/matplotlib only)
csv_path = os.path.join(OUT_DIR, "summary_table.csv")
fieldnames = ["Control", "lead_frac", "rho_mean", "rho_std", "mean_rel_lead", "pass_rate", "expected_pass_rate", "verdict"]
with open(csv_path, "w") as f:
    f.write(",".join(fieldnames) + "\n")
    for row in rows:
        vals = []
        for k in fieldnames:
            v = row[k]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v).replace(",", ";"))
        f.write(",".join(vals) + "\n")
print(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Plot 1 — sensitivity curve (Control D: pass_rate vs noise_frac)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
xs = NOISE_FRACS
ys = [control_D_summary[f]["pass_rate"] * 100 for f in NOISE_FRACS]
ax.plot(xs, ys, marker="o", linewidth=2)
ax.set_xlabel("noise_frac")
ax.set_ylabel("pass_rate (%)")
ax.set_title("Control D — Noisy-good predictor: pass rate vs noise level")
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)
fig.tight_layout()
sens_path = os.path.join(OUT_DIR, "plot_sensitivity.png")
fig.savefig(sens_path, dpi=150)
plt.close(fig)
print(f"Wrote {sens_path}")


# ---------------------------------------------------------------------------
# Plot 2 — null distribution (Control E Variant 1 rho histogram + threshold)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(rhos_v1, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvline(RHO_THRESH, color="red", linewidth=2, linestyle="--",
           label=f"RHO_THRESH (99th pct) = {RHO_THRESH:.3f}")
ax.set_xlabel("Spearman rho (Gs vs random Ps)")
ax.set_ylabel("count")
ax.set_title(f"Null distribution of rho, {N_RAND} random predictors (Control E, Variant 1)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
null_path = os.path.join(OUT_DIR, "plot_null_distribution.png")
fig.savefig(null_path, dpi=150)
plt.close(fig)
print(f"Wrote {null_path}")


# ---------------------------------------------------------------------------
# detailed_results.json
# ---------------------------------------------------------------------------

detailed = {
    "Gs": Gs.tolist(),
    "T_MAX": T_MAX,
    "margin": MARGIN,
    "RHO_THRESH": RHO_THRESH,
    "n_rand_for_calibration": N_RAND,
    "control_A": {"Ps": Ps_A.tolist(), **{k: v for k, v in res_A.items() if k not in ("g", "p")}, "pass": pass_A},
    "control_B": {"Ps": Ps_B.tolist(), **{k: v for k, v in res_B.items() if k not in ("g", "p")}, "pass": pass_B},
    "control_C": {"Ps": Ps_C.tolist(), **{k: v for k, v in res_C.items() if k not in ("g", "p")}, "pass": pass_C},
    "control_D": {
        "noise_fracs": NOISE_FRACS,
        "n_trials_per_frac": N_TRIALS_NOISY,
        "summary": control_D_summary,
    },
    "control_E_variant1_unconditional": {
        "n_trials": N_RAND,
        "lead_frac_mean": float(np.mean(lead_fracs_v1)),
        "rho_mean": float(np.mean(rhos_v1)),
        "rho_std": float(np.std(rhos_v1)),
        "mean_rel_lead": float(np.mean(rel_leads_v1)),
        "pass_rate": float(np.mean(pass_flags_v1)),
    },
    "control_E_variant2_conditional": {
        "n_trials": N_RAND,
        "lead_frac_mean": float(np.mean(lead_fracs_v2)),
        "rho_mean": float(np.mean(rhos_v2)),
        "rho_std": float(np.std(rhos_v2)),
        "mean_rel_lead": float(np.mean(rel_leads_v2)),
        "pass_rate": float(np.mean(pass_flags_v2)),
    },
}

json_path = os.path.join(OUT_DIR, "detailed_results.json")
with open(json_path, "w") as f:
    json.dump(detailed, f, indent=2)
print(f"Wrote {json_path}")

print("\nDone. All outputs in:", OUT_DIR)
