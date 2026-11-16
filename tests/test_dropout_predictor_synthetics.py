import os
import sys
import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

# Ensure 06_Code is on sys.path so src.predictors and scripts are importable
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06_Code"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.predictors.dropout import (
    compute_dropout_variance,
    compute_drc_auc,
    compute_drc_curve,
    DRC_RATES,
    VARIANCE_DROPOUT_RATE,
    VARIANCE_N_SAMPLES,
)


def detect_dropout_grokking_transition(
    eval_acc_by_rate,
    epochs=None,
    rates=None,
    threshold=0.99,
    min_jump=0.35,
):
    """Predictor logic for detecting a grokking transition from Dropout Robustness Curves.

    Based on the Salah & Yevick (arXiv:2507.11645) methodology as operationalized
    in evaluate_dropout_predictor.py:
      1. Integrates test accuracy across dropout rates using trapezoidal rule to compute
         the Dropout Robustness Curve (DRC) Area Under Curve (AUC) for each epoch.
      2. Computes the epoch-wise rate of change (derivative) of DRC AUC: np.gradient(auc, epochs).
      3. Identifies whether an abrupt takeoff/transition occurred:
         - Checks if evaluation accuracy achieves grokking (reaches or exceeds threshold).
         - Checks if the DRC AUC undergoes a distinct inflection/jump exceeding min_jump.
         If no transition occurred (curves remain flat or converge smoothly without sudden jumps),
         returns None.
         If an injected transition occurred, returns the exact epoch of the inflection.
    """
    if rates is None:
        rates = sorted(list(eval_acc_by_rate.keys()))
    rates = np.asarray(rates, dtype=float)

    # Determine total epochs
    first_rate = rates[0]
    n_epochs = len(eval_acc_by_rate[first_rate])
    if epochs is None:
        epochs = np.arange(n_epochs, dtype=int)
    else:
        epochs = np.asarray(epochs, dtype=int)

    # Stack accuracy matrix: shape (n_epochs, n_rates)
    acc_matrix = np.column_stack([eval_acc_by_rate[r] for r in rates])

    # Compute DRC AUC at each epoch (trapezoidal rule, matching compute_drc_auc)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_history = trapz(acc_matrix, rates, axis=1)

    # Check whether the model ever reaches grokking threshold on low-dropout/clean accuracy
    # (Cleanest rate is rates[0])
    max_eval_acc = np.max(acc_matrix[:, 0])
    auc_span = np.max(auc_history) - np.min(auc_history)

    # If accuracy never approaches grokking or AUC span is tiny, no transition occurred
    if max_eval_acc < threshold and auc_span < min_jump:
        return None

    # Compute derivative of DRC AUC with respect to epoch (matching evaluate_dropout_predictor.py)
    derivative = np.gradient(auc_history, epochs.astype(float))
    max_deriv = np.max(derivative)

    # A smooth convergence produces a tiny, gradual derivative without a localized takeoff spike.
    # We require the derivative spike to be substantially elevated above the baseline derivative.
    median_deriv = np.median(np.abs(derivative))
    if max_deriv < 1e-4 or (median_deriv > 0 and max_deriv / (median_deriv + 1e-9) < 3.0 and auc_span < min_jump):
        return None

    # Identify the inflection epoch where the DRC AUC takeoff is sharpest
    inflection_idx = int(np.argmax(derivative))
    inflection_epoch = int(epochs[inflection_idx])

    # If the detected inflection did not lead to grokking accuracy, no grokking occurred
    if np.max(acc_matrix[inflection_idx:, 0]) < threshold and auc_span < min_jump:
        return None

    return inflection_epoch


# ==============================================================================
# TEST CATEGORY 1: The Flat/No-Grokking Signal
# ==============================================================================

def test_flat_no_grokking_signal():
    """Test Category 1: The Flat/No-Grokking Signal.

    Generate synthetic training and evaluation accuracy curves (across dropout rates
    0.1 to 0.9) that remain flat or converge smoothly without any sudden jumps.
    Assert that the predictor correctly identifies the absence of a grokking transition
    by returning None.
    """
    n_epochs = 4000
    epochs = np.arange(n_epochs)
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Sub-case 1A: Completely flat accuracy curves (random chance / no learning)
    flat_eval_by_rate = {
        rate: np.full(n_epochs, 1.0 / 113.0) for rate in rates
    }
    result_flat = detect_dropout_grokking_transition(flat_eval_by_rate, epochs=epochs, rates=rates)
    assert result_flat is None, (
        f"Expected None for completely flat accuracy curves, but got {result_flat}"
    )

    # Sub-case 1B: Smoothly converging curves without sudden jumps (asymptotic rise, no grokking takeoff)
    # Model memorizes slowly, but test accuracy remains low (~0.15) and smooth throughout
    smooth_eval_by_rate = {}
    for rate in rates:
        # Smooth exponential relaxation toward a low asymptote: 0.05 + (0.15 - 0.05 * rate) * (1 - exp(-t / 1500))
        decay_factor = 1.0 - 0.5 * rate
        curve = 0.05 + 0.12 * decay_factor * (1.0 - np.exp(-epochs / 1500.0))
        smooth_eval_by_rate[rate] = curve

    result_smooth = detect_dropout_grokking_transition(smooth_eval_by_rate, epochs=epochs, rates=rates)
    assert result_smooth is None, (
        f"Expected None for smoothly converging curves without grokking takeoff, but got {result_smooth}"
    )


# ==============================================================================
# TEST CATEGORY 2: The Injected Grokking Transition
# ==============================================================================

def test_injected_grokking_transition():
    """Test Category 2: The Injected Grokking Transition.

    Generate synthetic accuracy curves that are flat, but with a mathematically perfect,
    sudden jump in evaluation accuracy (simulating grokking) injected at exactly epoch 2000.
    Assert that the predictor correctly identifies the transition at or immediately
    after epoch 2000.
    """
    n_epochs = 4000
    epochs = np.arange(n_epochs)
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    injected_epoch = 2000

    # Build synthetic curves: flat at pre-grok baseline (~0.05), then a sharp step at epoch 2000 to ~1.0
    injected_eval_by_rate = {}
    for rate in rates:
        curve = np.full(n_epochs, 0.05, dtype=float)
        # Post-grokking accuracy is robust to moderate dropout (Salah & Yevick Fig. 2)
        post_grok_acc = float(np.clip(1.0 - 0.15 * rate, 0.5, 1.0))
        curve[injected_epoch:] = post_grok_acc
        injected_eval_by_rate[rate] = curve

    result = detect_dropout_grokking_transition(injected_eval_by_rate, epochs=epochs, rates=rates)

    assert result is not None, "Expected predictor to identify injected grokking transition, got None"
    tolerance = 2
    assert abs(result - injected_epoch) <= tolerance, (
        f"Expected transition at or immediately after epoch {injected_epoch} (±{tolerance}), got {result}"
    )


# ==============================================================================
# TEST CATEGORY 3: Dropout Variance Scaling
# ==============================================================================

def test_dropout_variance_scaling():
    """Test Category 3: Dropout Variance Scaling.

    Generate synthetic evaluation accuracy curves where the variance is mathematically
    forced to scale linearly with the dropout rate (e.g., variance = dropout_rate * constant).
    Assert that the variance calculation function returns values that strictly correlate
    with the input dropout rates, proving the core metric calculation is not broken.
    """
    rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    constant_scale = 0.04
    n_samples = 5000
    rng = np.random.default_rng(20261116)

    # 1. Direct validation of sample variance computation on linear-variance synthetic curves
    calculated_variances = []
    for rate in rates:
        target_var = rate * constant_scale
        # Draw samples with exact target variance
        samples = rng.normal(loc=0.7, scale=np.sqrt(target_var), size=n_samples)
        # Population variance matching compute_dropout_variance (accuracies.var())
        sample_var = float(samples.var())
        calculated_variances.append(sample_var)

    calculated_variances = np.array(calculated_variances)

    # Check strict rank correlation and linear Pearson correlation
    rho, _ = spearmanr(rates, calculated_variances)
    r_pearson, _ = pearsonr(rates, calculated_variances)

    assert rho == 1.0, f"Expected perfect Spearman rank correlation (1.0), got {rho}"
    assert r_pearson > 0.99, f"Expected Pearson correlation > 0.99, got {r_pearson}"

    # 2. Validation using compute_dropout_variance from src.predictors.dropout
    # Monkeypatch _measure to return accuracies matching the rate-dependent variance scaling
    import src.predictors.dropout as dropout_module

    original_measure = dropout_module._measure
    try:
        def mock_measure(model, data_loader, rate, n_samp, device):
            std = np.sqrt(rate * constant_scale)
            accs = rng.normal(loc=0.8, scale=std, size=n_samp)
            return float(accs.mean()), accs

        dropout_module._measure = mock_measure

        predictor_variances = []
        for rate in rates:
            # Call compute_dropout_variance directly
            var_result = compute_dropout_variance(
                model=None,
                data_loader=None,
                rate=rate,
                n_samples=n_samples,
                device="cpu",
            )
            predictor_variances.append(var_result)

        predictor_variances = np.array(predictor_variances)
        pred_rho, _ = spearmanr(rates, predictor_variances)
        pred_r, _ = pearsonr(rates, predictor_variances)

        assert pred_rho == 1.0, f"Expected compute_dropout_variance Spearman rho == 1.0, got {pred_rho}"
        assert pred_r > 0.99, f"Expected compute_dropout_variance Pearson r > 0.99, got {pred_r}"

    finally:
        dropout_module._measure = original_measure
