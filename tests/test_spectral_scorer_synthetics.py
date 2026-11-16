import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(REPO_ROOT, "06_Code")
SCRIPTS_DIR = os.path.join(CODE_DIR, "scripts")
SRC_DIR = os.path.join(CODE_DIR, "src")

for p in (CODE_DIR, SCRIPTS_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_control_experiment import evaluate_metrics
from analyze_nanda_unified import evaluate_frozen_protocol, attach_pass_checks
from src.quality_score import passes, superiority_index_from_result


G = np.array([4000, 6500, 8000, 11000, 13000], dtype=float)


def _score_signal(grok_epochs, firing_epochs, lead_c=0.8):
    """Feeds epochs to the official frozen protocol and returns metrics."""
    res = evaluate_frozen_protocol(grok_epochs, firing_epochs, rule="fraction", c=lead_c)
    rho = 0.0 if (res["rho"] is None or np.isnan(res["rho"])) else float(res["rho"])
    metrics = evaluate_metrics(
        lead_frac=res["lead_frac"],
        rho=rho,
        noise_margin=res["mean_rel_lead"],
    )
    # Also evaluate through attach_pass_checks for frozen protocol compatibility
    frozen_checked = attach_pass_checks(res)
    return {
        "lead_frac": res["lead_frac"],
        "rho": rho,
        "noise_margin": res["mean_rel_lead"],
        "n_firing": res["n_firing"],
        "validity_gate_passed": bool(metrics["passed"]),
        "superiority_index": float(metrics["superiority_index"]),
        "frozen_verdict": frozen_checked["verdict"],
    }


def test_perfect_predictive_signal_passes():
    """S1: P = 0.5 * G elementwise [2000, 3250, 4000, 5500, 6500].

    Assert Validity Gate PASS, lead fraction == 1.0, Spearman rho == 1.0,
    Superiority Index approximately 1.0 (absolute tolerance 1e-6).
    """
    P = [2000, 3250, 4000, 5500, 6500]
    out = _score_signal(G, P)
    assert out["validity_gate_passed"] is True
    assert out["lead_frac"] == 1.0
    assert abs(out["rho"] - 1.0) < 1e-6
    assert abs(out["superiority_index"] - 1.0) < 1e-6


def test_zero_variance_signal_fails():
    """S2: P = [280, 280, 280, 280, 280].

    Assert rho == 0.0, Superiority Index == 0.0, Validity Gate FAIL.
    """
    P = [280, 280, 280, 280, 280]
    out = _score_signal(G, P)
    assert out["rho"] == 0.0
    assert out["superiority_index"] == 0.0
    assert out["validity_gate_passed"] is False


def test_uncorrelated_noise_fails():
    """S3: P drawn as integers in [1000, 40000] with default_rng(7).

    Assert Validity Gate FAIL and Superiority Index == 0.0.
    """
    rng = np.random.default_rng(7)
    P = rng.integers(1000, 40001, size=5).tolist()
    out = _score_signal(G, P)
    assert out["validity_gate_passed"] is False
    assert out["superiority_index"] == 0.0


def test_all_none_fails():
    """S4: P = [None, None, None, None, None].

    Assert Validity Gate FAIL, Superiority Index == 0.0, and no exception raised.
    """
    P = [None, None, None, None, None]
    out = _score_signal(G, P)
    assert out["validity_gate_passed"] is False
    assert out["superiority_index"] == 0.0


def test_postdictive_signal_fails_lead_gate():
    """S5: P = G + 500 elementwise.

    Assert lead fraction == 0.0 and Validity Gate FAIL even though rho == 1.0.
    """
    P = (G + 500).astype(int).tolist()
    out = _score_signal(G, P)
    assert out["lead_frac"] == 0.0
    assert abs(out["rho"] - 1.0) < 1e-6
    assert out["validity_gate_passed"] is False
    assert out["superiority_index"] == 0.0


def test_monotone_ladder_scaling():
    """S6: construct five tuples interpolating from perfect-lead/perfect-correlation to zero-lead/zero-correlation.

    Assert Superiority Index is non-increasing along the ladder.
    """
    ladder = [
        [2000, 3250, 4000, 5500, 6500],  # 0.5 * G
        [2400, 3900, 4800, 6600, 7800],  # 0.6 * G
        [2800, 4550, 5600, 7700, 9100],  # 0.7 * G
        [3000, 4875, 6000, 8250, 9750],  # 0.75 * G
        [4500, 7000, 8500, 11500, 13500],  # G + 500 (fails lead gate)
    ]
    si_values = [_score_signal(G, p)["superiority_index"] for p in ladder]
    for i in range(len(si_values) - 1):
        assert si_values[i] >= si_values[i + 1], f"Ladder non-monotone at step {i}: {si_values[i]} < {si_values[i+1]}"
    assert si_values[0] > 0.99
    assert si_values[-1] == 0.0
