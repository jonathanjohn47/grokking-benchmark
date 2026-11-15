"""L2 predictor positive control — synthetic perfect signal through the evaluation pipeline.

Isolates the Validity Gate / Superiority Index path from the physical L2 predictor.
Feeds a designed firing array P_i = int(0.5 * G_i) against the known Nanda-unified
grokking epochs, so the gate must PASS with Superiority Index 1.0 if the pipeline
is correct.

Does not modify l2_norm.py, training loops, or the 17-signal run_control_experiment.

Run:
    python 06_Code/scripts/run_l2_positive_control.py

Outputs (under 08_Experiments/results/):
    l2_positive_control_results.json
    l2_positive_control_plot.png
"""

from __future__ import annotations

import json
import os
import sys
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
LEAD_C = 0.8

RESULTS_DIR = REPO_ROOT / "08_Experiments" / "results"
JSON_PATH = RESULTS_DIR / "l2_positive_control_results.json"
PLOT_PATH = RESULTS_DIR / "l2_positive_control_plot.png"


def synthetic_firing_epochs(grokking_epochs):
    """P_i = int(0.5 * G_i) — mid-point relative lead for every seed."""
    return [int(0.5 * g) for g in grokking_epochs]


def compute_gate_inputs(grokking_epochs, firing_epochs, c=LEAD_C):
    """Compute lead fraction, Spearman rho, and mean relative lead (noise margin)."""
    g_arr = np.asarray(grokking_epochs, dtype=float)
    p_arr = np.asarray(firing_epochs, dtype=float)
    n = len(g_arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    lead_frac = float(np.mean(p_arr < c * g_arr))
    rel_lead = (g_arr - p_arr) / g_arr
    noise_margin = float(np.mean(rel_lead))

    if n >= 3 and not np.all(p_arr == p_arr[0]):
        rho, _ = spearmanr(g_arr, p_arr)
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0

    return lead_frac, rho, noise_margin


def build_payload(grokking_epochs, firing_epochs, gate_result):
    """Full results payload; Superiority Index is always a float (0.0 on fail)."""
    si = gate_result.get("superiority_index")
    if si is None:
        si = 0.0
    else:
        si = float(si)

    return {
        "description": (
            "L2 positive control: synthetic P_i = int(0.5 * G_i) against "
            "Nanda-unified grokking epochs (cut 0.99)."
        ),
        "grokking_epochs": list(grokking_epochs),
        "synthetic_firing_epochs": list(firing_epochs),
        "lead_c": LEAD_C,
        "validity_gate_metrics": {
            "lead_fraction": float(gate_result["lead_frac"]),
            "correlation": float(gate_result["rho"]),
            "noise_margin": float(gate_result["noise_margin"]),
        },
        "validity_gate_verdict": bool(gate_result["passed"]),
        "superiority_index": si,
        "evaluate_metrics_result": {
            "lead_frac": float(gate_result["lead_frac"]),
            "rho": float(gate_result["rho"]),
            "mean_rel_lead": float(gate_result["mean_rel_lead"]),
            "noise_margin": float(gate_result["noise_margin"]),
            "n_firing": int(gate_result["n_firing"]),
            "passed": bool(gate_result["passed"]),
            "superiority_index": si,
        },
    }


def make_plot(grokking_epochs, firing_epochs, payload, out_path):
    """Scatter of synthetic (G_i, P_i) with y=x and y=0.5x reference lines."""
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
        c="green",
        zorder=5,
        edgecolors="darkgreen",
        label="Synthetic $P_i = \\mathrm{int}(0.5\\,G_i)$",
    )

    verdict_text = "Validity Gate: PASS" if gate_pass else "Validity Gate: FAIL"
    verdict_color = "green" if gate_pass else "red"
    box_lines = [
        verdict_text,
        f"Lead Fraction: {metrics['lead_fraction']:.1f}",
        f"Correlation: {metrics['correlation']:.1f}",
        f"Noise Margin: {metrics['noise_margin']:.1f}",
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
    # Colour only the gate line inside the box by over-drawing the first line.
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
    ax.set_title("L2 Predictor Positive Control: Synthetic Perfect Signal")
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
    firing_epochs = synthetic_firing_epochs(grokking_epochs)

    lead_frac, rho, noise_margin = compute_gate_inputs(
        grokking_epochs, firing_epochs, c=LEAD_C
    )
    gate_result = evaluate_metrics(lead_frac, rho, noise_margin)
    # Project convention for this control: failed SI is 0.0, never null.
    if gate_result.get("superiority_index") is None:
        gate_result["superiority_index"] = 0.0

    payload = build_payload(grokking_epochs, firing_epochs, gate_result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    make_plot(grokking_epochs, firing_epochs, payload, PLOT_PATH)

    print("L2 positive control")
    print(f"  G_i = {grokking_epochs}")
    print(f"  P_i = {firing_epochs}")
    print(f"  lead_frac     = {gate_result['lead_frac']:.6f}")
    print(f"  correlation   = {gate_result['rho']:.6f}")
    print(f"  noise_margin  = {gate_result['noise_margin']:.6f}")
    print(f"  gate_pass     = {gate_result['passed']}")
    print(f"  superiority   = {gate_result['superiority_index']:.6f}")
    print(f"  wrote {os.path.relpath(JSON_PATH, REPO_ROOT)}")
    print(f"  wrote {os.path.relpath(PLOT_PATH, REPO_ROOT)}")

    if not gate_result["passed"]:
        print("ERROR: positive control failed the Validity Gate.", file=sys.stderr)
        return 1
    if abs(float(gate_result["superiority_index"]) - 1.0) > 1e-6:
        print(
            "ERROR: expected Superiority Index 1.0, "
            f"got {gate_result['superiority_index']}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
