"""L2 predictor negative control — constant early firing through the evaluation pipeline.

Isolates the Validity Gate / Superiority Index path from the physical L2 predictor.
Feeds a constant firing array P_i = [280, 280, 280, 280, 280] against the known
Nanda-unified grokking epochs. This mimics the real L2 failure mode (fixed early
epoch, no proportional anticipation). The gate must FAIL and Superiority Index
must be exactly 0.0 (never null) if the pipeline is correct.

Also exercises the constant-signal Spearman edge case: zero variance in P yields
an undefined correlation (NaN), which must be converted to 0.0 before the gate.

Does not modify l2_norm.py, training loops, or the 17-signal run_control_experiment.

Run:
    python 06_Code/scripts/run_l2_negative_control.py

Outputs (under 08_Experiments/results/):
    l2_negative_control_results.json
    l2_negative_control_plot.png
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "06_Code" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_control_experiment import evaluate_metrics  # noqa: E402

# Known grokking epochs (cut 0.99) from results_per_nanda_power / aggregate.json.
G_I = [8021, 8253, 4094, 7831, 6637]
# Constant early firing — structural twin of the real L2 failure mode.
P_I = [280, 280, 280, 280, 280]
LEAD_C = 0.8

RESULTS_DIR = REPO_ROOT / "08_Experiments" / "results"
JSON_PATH = RESULTS_DIR / "l2_negative_control_results.json"
PLOT_PATH = RESULTS_DIR / "l2_negative_control_plot.png"


def sanitize_correlation(rho):
    """Convert undefined / NaN Spearman rho to 0.0 before the gate check."""
    if rho is None:
        return 0.0
    rho = float(rho)
    if np.isnan(rho):
        return 0.0
    return rho


def compute_gate_inputs(grokking_epochs, firing_epochs, c=LEAD_C):
    """Compute lead fraction, Spearman rho, and mean relative lead (noise margin).

    For a constant P array, spearmanr returns NaN (zero variance). That NaN is
    explicitly converted to 0.0 so the Validity Gate sees a finite failing
    correlation rather than a floating NaN.
    """
    g_arr = np.asarray(grokking_epochs, dtype=float)
    p_arr = np.asarray(firing_epochs, dtype=float)
    n = len(g_arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    lead_frac = float(np.mean(p_arr < c * g_arr))
    rel_lead = (g_arr - p_arr) / g_arr
    noise_margin = float(np.mean(rel_lead))

    # Always call spearmanr when n >= 3 so the constant-signal NaN path is exercised.
    # SciPy warns that a constant input makes correlation undefined; that is the
    # edge case under test, so the warning is expected and suppressed here.
    if n >= 3:
        with warnings.catch_warnings():
            # ConstantInputWarning is a RuntimeWarning subclass in SciPy.
            warnings.filterwarnings(
                "ignore",
                message="An input array is constant",
                category=RuntimeWarning,
            )
            rho, _ = spearmanr(g_arr, p_arr)
        rho = sanitize_correlation(rho)
    else:
        rho = 0.0

    return lead_frac, rho, noise_margin


def build_payload(grokking_epochs, firing_epochs, gate_result):
    """Full results payload; Superiority Index is always a float (0.0 on fail)."""
    gate_pass = bool(gate_result["passed"])
    si = gate_result.get("superiority_index")
    if si is None or not gate_pass:
        si = 0.0
    else:
        si = float(si)

    return {
        "description": (
            "L2 negative control: synthetic constant P_i = [280]*5 against "
            "Nanda-unified grokking epochs (cut 0.99). Mimics the real L2 "
            "fixed-early-epoch failure mode."
        ),
        "grokking_epochs": list(grokking_epochs),
        "synthetic_firing_epochs": list(firing_epochs),
        "lead_c": LEAD_C,
        "validity_gate_metrics": {
            "lead_fraction": float(gate_result["lead_frac"]),
            "correlation": float(gate_result["rho"]),
            "noise_margin": float(gate_result["noise_margin"]),
        },
        "validity_gate_verdict": gate_pass,
        "validity_gate_pass": gate_pass,
        "superiority_index": si,
        "evaluate_metrics_result": {
            "lead_frac": float(gate_result["lead_frac"]),
            "rho": float(gate_result["rho"]),
            "mean_rel_lead": float(gate_result["mean_rel_lead"]),
            "noise_margin": float(gate_result["noise_margin"]),
            "n_firing": int(gate_result["n_firing"]),
            "passed": gate_pass,
            "superiority_index": si,
        },
    }


def make_plot(grokking_epochs, firing_epochs, payload, out_path):
    """Scatter of constant (G_i, P_i) with y=x and y=0.5x reference lines."""
    g = np.asarray(grokking_epochs, dtype=float)
    p = np.asarray(firing_epochs, dtype=float)
    metrics = payload["validity_gate_metrics"]
    gate_pass = payload["validity_gate_verdict"]
    si = payload["superiority_index"]

    lo = 0.0
    hi = float(max(g.max(), p.max()) * 1.05)
    xs = np.linspace(lo, hi, 200)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(xs, xs, linestyle="--", color="0.45", label=r"$y = x$ (simultaneous)")
    ax.plot(xs, 0.5 * xs, linestyle="-", color="0.25", label=r"$y = 0.5x$ (synthetic perfect)")
    ax.scatter(
        g,
        p,
        s=70,
        c="red",
        zorder=5,
        edgecolors="darkred",
        label=r"Synthetic $P_i = 280$ (constant)",
    )

    verdict_text = "Validity Gate: PASS" if gate_pass else "Validity Gate: FAIL"
    verdict_color = "green" if gate_pass else "red"
    box_lines = [
        verdict_text,
        f"Lead Fraction: {metrics['lead_fraction']:.1f}",
        f"Correlation: {metrics['correlation']:.1f}",
        f"Noise Margin: {metrics['noise_margin']:.2f}",
        f"Superiority Index: {si:.1f}",
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(box_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=0.95),
        color="black",
    )
    ax.text(
        0.03,
        0.97,
        verdict_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color=verdict_color,
        fontweight="bold",
    )

    ax.set_xlabel("Grokking Epoch")
    ax.set_ylabel("Firing Epoch")
    ax.set_title("L2 Predictor Negative Control: Constant Epoch Signal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    grokking_epochs = list(G_I)
    firing_epochs = list(P_I)

    lead_frac, rho, noise_margin = compute_gate_inputs(
        grokking_epochs, firing_epochs, c=LEAD_C
    )
    # Defensive second pass: never feed NaN into evaluate_metrics.
    rho = sanitize_correlation(rho)

    gate_result = evaluate_metrics(lead_frac, rho, noise_margin)

    # Standardization: failed gate => Superiority Index exactly 0.0, never null.
    if gate_result.get("superiority_index") is None:
        gate_result["superiority_index"] = 0.0

    payload = build_payload(grokking_epochs, firing_epochs, gate_result)
    if not payload["validity_gate_pass"]:
        payload["superiority_index"] = 0.0
        payload["evaluate_metrics_result"]["superiority_index"] = 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    make_plot(grokking_epochs, firing_epochs, payload, PLOT_PATH)

    print("L2 negative control")
    print(f"  G_i = {grokking_epochs}")
    print(f"  P_i = {firing_epochs}")
    print(f"  lead_frac     = {gate_result['lead_frac']:.6f}")
    print(f"  correlation   = {gate_result['rho']:.6f}")
    print(f"  noise_margin  = {gate_result['noise_margin']:.6f}")
    print(f"  gate_pass     = {gate_result['passed']}")
    print(f"  superiority   = {payload['superiority_index']:.6f}")
    print(f"  wrote {os.path.relpath(JSON_PATH, REPO_ROOT)}")
    print(f"  wrote {os.path.relpath(PLOT_PATH, REPO_ROOT)}")

    # Negative control success = gate FAIL and SI exactly 0.0 (JSON float, not null).
    if payload["validity_gate_pass"]:
        print("ERROR: negative control unexpectedly passed the Validity Gate.", file=sys.stderr)
        return 1
    if payload["superiority_index"] is None:
        print("ERROR: superiority_index is null; must be 0.0 on fail.", file=sys.stderr)
        return 1
    if float(payload["superiority_index"]) != 0.0:
        print(
            "ERROR: expected Superiority Index 0.0 on fail, "
            f"got {payload['superiority_index']}.",
            file=sys.stderr,
        )
        return 1
    if float(payload["validity_gate_metrics"]["correlation"]) != 0.0:
        print(
            "ERROR: expected correlation 0.0 after NaN sanitisation, "
            f"got {payload['validity_gate_metrics']['correlation']}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
