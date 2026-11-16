import os
import sys
import numpy as np
import pytest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.predictors.htsr_alpha import fit_powerlaw_alpha

def test_injected_powerlaw_alpha():
    """Test Category 1: Injected Feature.
    Generate a synthetic power-law distribution with a known alpha=2.0.
    The fitter must detect it within a tolerance.
    """
    # Generate power-law distributed data (alpha = 2.0, xmin = 1.0)
    rng = np.random.default_rng(42)
    alpha_true = 2.0
    n_samples = 10000
    # Inverse transform sampling for power law: x = xmin * (1 - u)^(-1/(alpha-1))
    u = rng.uniform(0, 1, n_samples)
    evals = 1.0 * (1.0 - u) ** (-1.0 / (alpha_true - 1.0))
    
    alpha_est, xmin, D, n_tail = fit_powerlaw_alpha(evals)
    
    assert not np.isnan(alpha_est), "Alpha fit returned NaN"
    assert abs(alpha_est - alpha_true) < 0.2, f"Expected alpha ~2.0, got {alpha_est}"

def test_featureless_null():
    """Test Category 2: Null Signal.
    Feed the fitter a random Gaussian distribution (exponential tail).
    The fitter should either fail to find a power law or return a very high alpha.
    """
    rng = np.random.default_rng(42)
    # Gaussian data does not have a power-law tail
    evals = rng.normal(loc=10.0, scale=1.0, size=5000)
    # Evals must be positive for ESD fitting
    evals = np.abs(evals)
    
    alpha_est, xmin, D, n_tail = fit_powerlaw_alpha(evals)
    
    # A Gaussian tail typically fits an alpha > 4 or fails entirely
    if not np.isnan(alpha_est):
        assert alpha_est >= 3.0 - 1e-5, f"Expected high alpha for Gaussian noise, got {alpha_est}"
