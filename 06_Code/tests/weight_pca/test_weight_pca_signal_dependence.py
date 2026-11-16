import os
import sys
import numpy as np
import pytest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.predictors.weight_pca import effective_rank

def test_effective_rank_full_rank():
    """Test Category 1: Full Rank (Maximum Entropy)."""
    # A matrix with perfectly equal singular values should have max effective rank
    sv = np.ones(100)
    eff_rank, norm_eff_rank = effective_rank(sv)
    
    # If all singular values are equal, p_i = 1/N.
    # Entropy H = -sum(1/N * log(1/N)) = log(N). ER = exp(log(N)) = N.
    assert np.isclose(eff_rank, 100.0), f"Expected eff_rank ~100, got {eff_rank}"
    assert np.isclose(norm_eff_rank, 1.0), f"Expected norm_eff_rank ~1.0, got {norm_eff_rank}"

def test_effective_rank_low_rank():
    """Test Category 2: Low Rank (Minimum Entropy)."""
    # A matrix dominated by one singular value should have eff rank ~1
    sv = np.zeros(100)
    sv[0] = 100.0
    eff_rank, norm_eff_rank = effective_rank(sv)
    
    assert np.isclose(eff_rank, 1.0), f"Expected eff_rank ~1.0, got {eff_rank}"
    assert np.isclose(norm_eff_rank, 0.01), f"Expected norm_eff_rank ~0.01, got {norm_eff_rank}"
