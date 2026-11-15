"""
Synthetic control experiment — 17 engineered Superiority Index signals.

This does NOT touch any real predictor code or checkpoints. It reverse-engineers
the three raw gate metrics (Lead fraction, Correlation coefficient, Noise margin)
so that the evaluation pipeline returns exact expected Superiority Index values.

Part 1 — 7 validity-gate failures (all permutations of Lead / Correlation / Noise)
         Expected Superiority Index = 0.0 for every signal.

Part 2 — 10 passing signals on the superiority spectrum
         Raw values sit exactly fraction t in {0.10, ..., 1.00} above the gate
         floors toward the theoretical perfect values, so SI = t exactly.

Run:
    python 06_Code/scripts/run_control_experiment.py

Output folder: ./controls_results/
    summary_table.csv
    detailed_results.json

The script exits with status 1 if any assertion fails.
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO_ROOT, "06_Code") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "06_Code"))

from src.quality_score import (
    CORR_PERFECT,
    CORR_THRESH,
    LEAD_PERFECT,
    LEAD_THRESH,
    NOISE_HI,
    NOISE_LO,
    NOISE_PERFECT,
    passes,
    raw_values_for_superiority_target,
    superiority_index,
    superiority_index_from_result,
)

TOL = 1e-4

OUT_DIR = os.path.join(REPO_ROOT, "controls_results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Evaluation pipeline for a synthetic metric signal
# ---------------------------------------------------------------------------

def evaluate_metrics(lead_frac, rho, noise_margin, corr_thresh=CORR_THRESH):
    """Run the three raw metrics through the validity gate and Superiority Index.

    Returns a result dict compatible with passes() / superiority_index_from_result().
    """
    result = {
        "lead_frac": float(lead_frac),
        "rho": float(rho),
        "mean_rel_lead": float(noise_margin),
        "noise_margin": float(noise_margin),
        "n_firing": 5,  # metric-level signals assume a full firing set
    }
    gate_pass = passes(result, corr_thresh)
    si = superiority_index_from_result(result, corr_thresh=corr_thresh)
    result["passed"] = gate_pass
    result["superiority_index"] = si
    return result


# ---------------------------------------------------------------------------
# Part 1 — 7 failing signals (every non-empty subset of the three gate checks)
# ---------------------------------------------------------------------------
# Passing reference values used when a dimension is NOT the failure mode:
#   Lead = 1.0, Correlation = 0.95, Noise = 0.5
# Failing values:
#   Lead = 0.6 (< 0.8), Correlation = 0.7 (< 0.9), Noise = 0.01 (out of window)

_PASS_LEAD = LEAD_PERFECT
_PASS_CORR = 0.95
_PASS_NOISE = NOISE_PERFECT
_FAIL_LEAD = 0.6
_FAIL_CORR = 0.7
_FAIL_NOISE = 0.01  # below NOISE_LO; also could use 0.95 (> NOISE_HI)

FAILING_SIGNALS = [
    {
        "id": 1,
        "name": "fail_lead_only",
        "lead_frac": _FAIL_LEAD,
        "rho": _PASS_CORR,
        "noise_margin": _PASS_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("lead",),
    },
    {
        "id": 2,
        "name": "fail_corr_only",
        "lead_frac": _PASS_LEAD,
        "rho": _FAIL_CORR,
        "noise_margin": _PASS_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("corr",),
    },
    {
        "id": 3,
        "name": "fail_noise_only",
        "lead_frac": _PASS_LEAD,
        "rho": _PASS_CORR,
        "noise_margin": _FAIL_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("noise",),
    },
    {
        "id": 4,
        "name": "fail_lead_corr",
        "lead_frac": _FAIL_LEAD,
        "rho": _FAIL_CORR,
        "noise_margin": _PASS_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("lead", "corr"),
    },
    {
        "id": 5,
        "name": "fail_lead_noise",
        "lead_frac": _FAIL_LEAD,
        "rho": _PASS_CORR,
        "noise_margin": _FAIL_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("lead", "noise"),
    },
    {
        "id": 6,
        "name": "fail_corr_noise",
        "lead_frac": _PASS_LEAD,
        "rho": _FAIL_CORR,
        "noise_margin": _FAIL_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("corr", "noise"),
    },
    {
        "id": 7,
        "name": "fail_all_three",
        "lead_frac": _FAIL_LEAD,
        "rho": _FAIL_CORR,
        "noise_margin": _FAIL_NOISE,
        "expected_si": 0.0,
        "failure_modes": ("lead", "corr", "noise"),
    },
]


# ---------------------------------------------------------------------------
# Part 2 — 10 passing signals (Superiority spectrum 0.10 … 1.00)
# ---------------------------------------------------------------------------

PASSING_TARGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

PASSING_SIGNALS = []
for i, t in enumerate(PASSING_TARGETS, start=8):
    raw = raw_values_for_superiority_target(t)
    PASSING_SIGNALS.append({
        "id": i,
        "name": f"pass_si_{t:.2f}",
        "lead_frac": raw["lead_frac"],
        "rho": raw["rho"],
        "noise_margin": raw["noise_margin"],
        "expected_si": float(t),
        "target_fraction": float(t),
    })


ALL_SIGNALS = FAILING_SIGNALS + PASSING_SIGNALS
assert len(ALL_SIGNALS) == 17, f"expected 17 signals, got {len(ALL_SIGNALS)}"


# ---------------------------------------------------------------------------
# Run pipeline + strict assertions
# ---------------------------------------------------------------------------

def _fmt_discrepancy(sig, got, expected):
    return (
        f"  [{sig['id']:02d}] {sig['name']}: "
        f"got SI={got:.8f}, expected={expected:.8f}, "
        f"|diff|={abs(got - expected):.8e} "
        f"(lead={sig['lead_frac']:.6f}, "
        f"rho={sig['rho']:.6f}, "
        f"noise={sig['noise_margin']:.6f})"
    )


def main():
    print("=" * 72)
    print("Control experiment — 17 Superiority Index signals")
    print("=" * 72)
    print(
        f"Gate: Lead>={LEAD_THRESH}, Corr>={CORR_THRESH}, "
        f"Noise in [{NOISE_LO}, {NOISE_HI}]"
    )
    print(
        f"Perfect: Lead={LEAD_PERFECT}, Corr={CORR_PERFECT}, "
        f"Noise={NOISE_PERFECT}"
    )
    print(f"Assertion tolerance: {TOL}")
    print()

    rows = []
    failures = []

    for sig in ALL_SIGNALS:
        result = evaluate_metrics(
            sig["lead_frac"], sig["rho"], sig["noise_margin"]
        )
        got = result["superiority_index"]
        expected = float(sig["expected_si"])
        ok = abs(got - expected) <= TOL

        # Direct formula check (same pipeline path, explicit call).
        got_direct = superiority_index(
            sig["lead_frac"], sig["rho"], sig["noise_margin"]
        )
        if abs(got_direct - got) > TOL:
            failures.append(
                f"  [{sig['id']:02d}] {sig['name']}: "
                f"pipeline SI ({got:.8f}) != direct SI ({got_direct:.8f})"
            )

        if not ok:
            failures.append(_fmt_discrepancy(sig, got, expected))

        # Failing signals must also fail the validity gate.
        if sig["id"] <= 7 and result["passed"]:
            failures.append(
                f"  [{sig['id']:02d}] {sig['name']}: "
                f"expected gate FAIL, but passes() returned True"
            )

        # Passing signals must clear the validity gate.
        if sig["id"] >= 8 and not result["passed"]:
            failures.append(
                f"  [{sig['id']:02d}] {sig['name']}: "
                f"expected gate PASS, but passes() returned False"
            )

        status = "OK" if ok else "MISMATCH"
        print(
            f"[{sig['id']:02d}] {sig['name']:<22} "
            f"lead={sig['lead_frac']:.4f}  rho={sig['rho']:.4f}  "
            f"noise={sig['noise_margin']:.4f}  "
            f"SI={got:.6f}  expected={expected:.2f}  [{status}]"
        )

        rows.append({
            "id": sig["id"],
            "name": sig["name"],
            "lead_frac": sig["lead_frac"],
            "rho": sig["rho"],
            "noise_margin": sig["noise_margin"],
            "gate_pass": bool(result["passed"]),
            "superiority_index": got,
            "expected_si": expected,
            "abs_error": abs(got - expected),
            "assertion_ok": ok,
            "failure_modes": list(sig.get("failure_modes", [])),
            "target_fraction": sig.get("target_fraction"),
        })

    print()

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    csv_path = os.path.join(OUT_DIR, "summary_table.csv")
    fieldnames = [
        "id", "name", "lead_frac", "rho", "noise_margin",
        "gate_pass", "superiority_index", "expected_si",
        "abs_error", "assertion_ok",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            vals = []
            for k in fieldnames:
                v = row[k]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"Wrote {csv_path}")

    detailed = {
        "description": (
            "17 synthetic Superiority Index signals: "
            "7 validity-gate failures + 10 passing spectrum points."
        ),
        "tolerance": TOL,
        "gate": {
            "lead_thresh": LEAD_THRESH,
            "corr_thresh": CORR_THRESH,
            "noise_lo": NOISE_LO,
            "noise_hi": NOISE_HI,
        },
        "perfect": {
            "lead": LEAD_PERFECT,
            "corr": CORR_PERFECT,
            "noise": NOISE_PERFECT,
        },
        "formula": (
            "SI = 0 if gate fails; else "
            "mean( (L-0.8)/0.2, (rho-0.9)/0.1, (noise-0.05)/0.45 )"
        ),
        "n_signals": len(rows),
        "signals": rows,
        "all_assertions_passed": len(failures) == 0,
    }
    json_path = os.path.join(OUT_DIR, "detailed_results.json")
    with open(json_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Wrote {json_path}")

    # -----------------------------------------------------------------------
    # Hard fail on any discrepancy
    # -----------------------------------------------------------------------
    if failures:
        print()
        print("ASSERTION FAILURES — criteria do not fit:")
        for line in failures:
            print(line)
        print()
        print(f"{len(failures)} discrepancy(ies). Exiting with status 1.")
        sys.exit(1)

    print()
    print(
        "All 17 assertions passed: "
        "7 failing signals -> SI = 0.0; "
        "10 passing signals -> SI in {0.10, ..., 1.00}."
    )
    print("Done. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
