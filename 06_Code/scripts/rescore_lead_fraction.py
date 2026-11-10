"""Re-score saved P_i and G_i under the fractional lead rule. No training.

Reads 08_Experiments/results/nanda_unified/analysis/predictor_events.json
and writes a NEW file next to it. Does not overwrite predictor_events.json
or any frozen verdict file.

Scored predictors (PRIMARY rows only): HTSR Alpha, Weight-PCA, Higher-MI,
Commutator Defect at the saved eta=1e-3 onset epochs.

Rules: margin-100, and P_i < c * G_i for c in {0.7, 0.8, 0.9}.
Default PASS rule is c=0.8. Other PASS conditions are unchanged.

Also re-runs the noise stress test P_i ~ Uniform[0, G_i] for each rule.
The random-predictor rho calibration does not use the lead comparison
(it is the 99th percentile of Spearman rho), so RHO_THRESH is unchanged.
This script still recomputes that percentile once and checks it matches
the frozen constant.
"""

import json
import os

import numpy as np

from analyze_nanda_unified import (
    DEFAULT_LEAD_C,
    LEAD_MARGIN_EPOCHS,
    RHO_THRESH,
    evaluate_frozen_protocol,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EVENTS = os.path.join(
    REPO, "08_Experiments", "results", "nanda_unified", "analysis", "predictor_events.json"
)
OUT = os.path.join(
    REPO,
    "08_Experiments",
    "results",
    "nanda_unified",
    "analysis",
    "lead_fraction_sweep_2026-11-10.json",
)

SCORED = [
    "HTSR: mean alpha minimum (implemented)",
    "Weight-PCA: effective-rank minimum (implemented)",
    "Higher-MI: population omega maximum (implemented)",
    "Commutator Defect: Xu onset on median D (implemented)",
]
EARLY = ("L2:", "Dropout:", "Spectral:", "AGE:")
RULES = [
    {"name": "margin100", "rule": "margin", "c": None, "margin": LEAD_MARGIN_EPOCHS},
    {"name": "c=0.7", "rule": "fraction", "c": 0.7, "margin": LEAD_MARGIN_EPOCHS},
    {"name": "c=0.8", "rule": "fraction", "c": 0.8, "margin": LEAD_MARGIN_EPOCHS},
    {"name": "c=0.9", "rule": "fraction", "c": 0.9, "margin": LEAD_MARGIN_EPOCHS},
]
T_MAX = 40000.0
N_NOISE = 10000
N_CAL = 10000


def passes(verdict):
    rho = verdict["rho"]
    rho_ok = not (isinstance(rho, float) and np.isnan(rho)) and rho >= RHO_THRESH
    return bool(
        verdict["n_firing"] >= 4
        and verdict["lead_frac"] >= 0.8
        and rho_ok
        and 0.05 <= verdict["mean_rel_lead"] <= 0.9
    )


def score_one(grok, epochs, spec):
    kwargs = {"rule": spec["rule"], "margin": spec["margin"]}
    if spec["c"] is not None:
        kwargs["c"] = spec["c"]
    verdict = evaluate_frozen_protocol(grok, epochs, **kwargs)
    verdict["passed"] = passes(verdict)
    verdict["rule_name"] = spec["name"]
    return verdict


def main():
    with open(EVENTS) as handle:
        events = json.load(handle)
    grok = events["grok_epochs"]
    by_name = {row["signal"]: row for row in events["signals"]}

    early = []
    for row in events["signals"]:
        if row["signal"].startswith(EARLY):
            epochs = row.get("epochs")
            early.append({
                "signal": row["signal"],
                "saved_P_i_present": epochs is not None and any(v is not None for v in epochs),
                "n_epochs": 0 if epochs is None else len(epochs),
            })

    rows = []
    frozen_verdicts = {}
    for name in SCORED:
        row = by_name[name]
        per_rule = [score_one(grok, row["epochs"], spec) for spec in RULES]
        frozen = row.get("frozen_protocol_verdict")
        frozen_verdicts[name] = None if frozen is None else bool(frozen.get("passed"))
        rows.append({
            "signal": name,
            "eta": 1e-3 if name.startswith("Commutator") else None,
            "G_i": grok,
            "P_i": row["epochs"],
            "frozen_file_passed": frozen_verdicts[name],
            "rules": per_rule,
            "verdict_changed_vs_frozen_file": any(
                item["passed"] != frozen_verdicts[name] for item in per_rule
            ) if frozen_verdicts[name] is not None else None,
        })

    # Noise stress test: P_i ~ Uniform[0, G_i]. Same draws for every rule.
    rng = np.random.default_rng(42)
    g_arr = np.asarray(grok, dtype=float)
    noise_P = rng.uniform(0.0, 1.0, size=(N_NOISE, len(grok))) * g_arr
    noise = []
    for spec in RULES:
        flags = []
        lead_fracs = []
        for i in range(N_NOISE):
            verdict = score_one(grok, noise_P[i].tolist(), spec)
            flags.append(verdict["passed"])
            lead_fracs.append(verdict["lead_frac"])
        noise.append({
            "rule_name": spec["name"],
            "n_trials": N_NOISE,
            "distribution": "P_i ~ Uniform[0, G_i]",
            "pass_rate": float(np.mean(flags)),
            "lead_frac_mean": float(np.mean(lead_fracs)),
            "seed": 42,
        })

    # Rho calibration does not read the lead mask. Same generator and seed
    # as run_control_experiment.py (np.random.seed(42), Uniform[0, T_MAX]).
    # Score each draw under both lead rules and check that rho is identical.
    np.random.seed(42)
    rhos_fraction = np.empty(N_CAL)
    rhos_margin = np.empty(N_CAL)
    for i in range(N_CAL):
        p_rand = np.random.uniform(0.0, T_MAX, size=len(grok)).tolist()
        rhos_fraction[i] = evaluate_frozen_protocol(
            grok, p_rand, rule="fraction", c=DEFAULT_LEAD_C
        )["rho"]
        rhos_margin[i] = evaluate_frozen_protocol(
            grok, p_rand, rule="margin", margin=LEAD_MARGIN_EPOCHS
        )["rho"]
    rho_hat = float(np.percentile(rhos_fraction, 99))
    rho_unchanged = bool(np.array_equal(rhos_fraction, rhos_margin))

    payload = {
        "date": "2026-11-10",
        "source": os.path.relpath(EVENTS, REPO),
        "note": (
            "Re-score of saved P_i and G_i only. No retraining. "
            "Frozen predictor_events.json was not overwritten. "
            "Commutator Defect epochs are the saved eta=1e-3 PRIMARY onset. "
            "Lead time scales with G (Xu arXiv:2602.16967; Golwala arXiv:2604.20923). "
            "No paper fixes c."
        ),
        "default_c": DEFAULT_LEAD_C,
        "rho_thresh_frozen": RHO_THRESH,
        "rho_calibration": {
            "uses_lead_rule": False,
            "reason": (
                "RHO_THRESH is the 99th percentile of Spearman rho for "
                "P ~ Uniform[0, T_MAX]. Rho does not depend on the lead test, "
                "so the threshold is the same at every c."
            ),
            "recomputed_99th_percentile": rho_hat,
            "matches_frozen_constant": rho_hat == RHO_THRESH,
            "rho_identical_under_margin_and_fraction": rho_unchanged,
            "n_trials": N_CAL,
            "T_MAX": T_MAX,
            "generator": "numpy.random.seed(42), same as run_control_experiment.py",
        },
        "early_predictors_not_rescored": early,
        "scored": rows,
        "noise_stress_test": noise,
    }
    with open(OUT, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {OUT}")
    print(f"recomputed rho 99th percentile = {rho_hat:.6f} (frozen {RHO_THRESH})")
    for row in rows:
        print(row["signal"])
        for item in row["rules"]:
            print(
                f"  {item['rule_name']:<12} lead_frac={item['lead_frac']:.4f} "
                f"rho={item['rho']:.4f} mean_r={item['mean_rel_lead']:.4f} "
                f"{'PASS' if item['passed'] else 'FAIL'}"
            )
    for item in noise:
        print(f"noise {item['rule_name']:<12} pass_rate={item['pass_rate']:.4f}")


if __name__ == "__main__":
    main()
