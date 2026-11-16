import os
import sys
import numpy as np
import pytest
from scipy.stats import spearmanr

# Ensure 06_Code and 06_Code/scripts are on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(REPO_ROOT, "06_Code")
SCRIPTS_DIR = os.path.join(CODE_DIR, "scripts")

for p in (CODE_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.quality_score import (
    passes,
    superiority_index,
    superiority_index_from_result,
    LEAD_THRESH,
    CORR_THRESH,
    NOISE_LO,
    NOISE_HI,
)
from run_control_experiment import evaluate_metrics
from analyze_nanda_unified import evaluate_frozen_protocol, attach_pass_checks


def detect_variance_drop_epoch(variance_series, epochs=None):
    """Detects the epoch of a sharp, sustained drop in variance.

    Identifies the epoch of maximum negative drop (-dVar/dt), or returns the index
    where the signal falls below half its maximum if a clear step transition occurs.
    """
    variance_series = np.asarray(variance_series, dtype=float)
    n = len(variance_series)
    if epochs is None:
        epochs = np.arange(n)
    else:
        epochs = np.asarray(epochs)

    if n < 2:
        return None

    # Compute difference (negative drop)
    diffs = np.diff(variance_series)
    # Most negative derivative represents the steepest fall
    drop_idx = int(np.argmin(diffs)) + 1
    return int(epochs[drop_idx])


def score_dropout_metric_signal(
    grokking_epochs,
    firing_epochs,
    lead_c=0.8,
    corr_thresh=CORR_THRESH,
):
    """Feeds extracted firing epochs and ground-truth grokking epochs to the benchmark scorer.

    Computes:
      - Lead fraction: fraction of seeds satisfying P_i < lead_c * G_i
      - Rank correlation: Spearman rho between G_i and P_i
      - Noise margin: mean relative lead (G_i - P_i) / G_i
      - Validity Gate verdict and Superiority Index via evaluate_metrics()
    """
    g_arr = np.asarray(grokking_epochs, dtype=float)
    p_arr = np.asarray(firing_epochs, dtype=float)
    n = len(g_arr)

    # Lead fraction
    lead_frac = float(np.mean(p_arr < lead_c * g_arr))

    # Relative lead and noise margin
    rel_leads = (g_arr - p_arr) / g_arr
    noise_margin = float(np.mean(rel_leads))

    # Rank correlation
    if n >= 3 and not np.all(p_arr == p_arr[0]):
        rho, _ = spearmanr(g_arr, p_arr)
        rho = 0.0 if np.isnan(rho) else float(rho)
    else:
        rho = 0.0

    # Evaluate through canonical validity gate and Superiority Index pipeline
    metrics_result = evaluate_metrics(
        lead_frac=lead_frac,
        rho=rho,
        noise_margin=noise_margin,
        corr_thresh=corr_thresh,
    )

    # Also evaluate through legacy frozen protocol for comprehensive cross-validation
    frozen_result = evaluate_frozen_protocol(grokking_epochs, firing_epochs, rule="fraction", c=lead_c)
    frozen_scored = attach_pass_checks(frozen_result)

    return {
        "lead_frac": lead_frac,
        "rho": rho,
        "noise_margin": noise_margin,
        "validity_gate_passed": bool(metrics_result["passed"]),
        "superiority_index": float(metrics_result["superiority_index"]),
        "frozen_verdict": frozen_scored["verdict"],
        "metrics_result": metrics_result,
        "frozen_scored": frozen_scored,
    }


# ==============================================================================
# TEST CATEGORY 1: Perfect Predictive Signal
# ==============================================================================

def test_perfect_predictive_signal():
    """Test Category 1: Perfect Predictive Signal.

    Feed the scorer a synthetic Dropout metric array that shows a clear, consistent
    drop in variance exactly 500 epochs before the ground truth grokking epochs (G_i)
    across all 5 seeds.
    Assert that the scorer awards a high Superiority Index (e.g., > 0.8) and passes
    the Validity Gate.
    """
    # 5 deterministic seeds with ground truth grokking epochs
    grokking_epochs = [1200, 1400, 1600, 1800, 2000]
    lead_offset = 500
    n_epochs = 3000
    epochs = np.arange(n_epochs)

    firing_epochs = []
    # Generate synthetic variance time series for each of the 5 seeds
    for G_i in grokking_epochs:
        drop_epoch = G_i - lead_offset  # Exactly 500 epochs before G_i
        # Pre-drop: high variance during memorization (~0.05)
        # Post-drop: low variance after circuit consolidation (~0.001)
        var_series = np.full(n_epochs, 0.05, dtype=float)
        var_series[drop_epoch:] = 0.001

        # Detect the drop epoch from the metric array
        detected_P_i = detect_variance_drop_epoch(var_series, epochs=epochs)
        assert detected_P_i == drop_epoch, (
            f"Expected drop epoch at {drop_epoch}, got {detected_P_i}"
        )
        firing_epochs.append(detected_P_i)

    # Feed to the scorer
    score = score_dropout_metric_signal(
        grokking_epochs=grokking_epochs,
        firing_epochs=firing_epochs,
        lead_c=0.8,
    )

    # Assertions
    assert score["validity_gate_passed"] is True, (
        f"Expected Validity Gate to PASS, but got {score['validity_gate_passed']}"
    )
    assert score["superiority_index"] > 0.8, (
        f"Expected Superiority Index > 0.8, but got {score['superiority_index']:.4f}"
    )
    assert score["lead_frac"] == 1.0, f"Expected 100% lead fraction, got {score['lead_frac']}"
    assert score["rho"] >= 0.99, f"Expected near-perfect rank correlation, got {score['rho']}"
    assert NOISE_LO <= score["noise_margin"] <= NOISE_HI, (
        f"Expected noise margin in [{NOISE_LO}, {NOISE_HI}], got {score['noise_margin']}"
    )
    assert score["frozen_verdict"] == "PASS", (
        f"Expected frozen protocol verdict PASS, got {score['frozen_verdict']}"
    )


# ==============================================================================
# TEST CATEGORY 2: Uncorrelated/Random Signal
# ==============================================================================

def test_uncorrelated_random_signal():
    """Test Category 2: Uncorrelated/Random Signal.

    Feed the scorer a synthetic Dropout metric array generated from pure random noise,
    completely decoupled from the ground truth G_i.
    Assert that the scorer correctly identifies the lack of correlation, fails
    the Validity Gate, and awards a Superiority Index near 0.0 (exact 0.0 under gating).
    """
    grokking_epochs = [8021, 8253, 4094, 7831, 6637]
    n_epochs = 10000
    epochs = np.arange(n_epochs)
    rng = np.random.default_rng(42)

    firing_epochs = []
    # Generate pure random noise arrays across all 5 seeds, decoupled from G_i
    for _ in grokking_epochs:
        # Meaningless Gaussian noise
        noise_series = rng.normal(loc=0.03, scale=0.01, size=n_epochs)
        detected_P_i = detect_variance_drop_epoch(noise_series, epochs=epochs)
        firing_epochs.append(detected_P_i)

    # Feed to the scorer
    score = score_dropout_metric_signal(
        grokking_epochs=grokking_epochs,
        firing_epochs=firing_epochs,
        lead_c=0.8,
    )

    # Assertions
    assert score["validity_gate_passed"] is False, (
        f"Expected Validity Gate to FAIL for random noise, but got {score['validity_gate_passed']}"
    )
    assert score["superiority_index"] == 0.0, (
        f"Expected Superiority Index of exactly 0.0 for failed gate, got {score['superiority_index']}"
    )
    assert score["rho"] < 0.90, (
        f"Expected Spearman rho < 0.90 for uncorrelated noise, got {score['rho']}"
    )
    assert score["frozen_verdict"] == "FAIL", (
        f"Expected frozen protocol verdict FAIL for random noise, got {score['frozen_verdict']}"
    )
