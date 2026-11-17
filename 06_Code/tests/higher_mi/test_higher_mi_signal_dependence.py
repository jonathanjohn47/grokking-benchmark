import os
import sys
import numpy as np
import pytest

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.predictors.higher_mi import o_information_three_way

def test_independent_null():
    """Test Category 1: Null Signal (Independent Variables).
    If three variables are completely independent, their O-information should be ~0.
    """
    rng = np.random.default_rng(42)
    n = 5000
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)

    omega_mean, omega_std = o_information_three_way(f0, f1, f2)
    
    # O-information for independent variables is 0
    assert abs(omega_mean) < 0.1, f"Expected O-info ~0 for independent vars, got {omega_mean}"

def test_synergistic_feature():
    """Test Category 2: Injected Feature (Synergy).
    If F2 = F0 + F1, the variables are highly synergistic, yielding negative O-information.
    """
    rng = np.random.default_rng(42)
    n = 5000
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    f2 = f0 + f1  # Perfect synergy

    omega_mean, omega_std = o_information_three_way(f0, f1, f2)
    
    # Synergistic systems have negative O-information
    assert omega_mean < -0.5, f"Expected strongly negative O-info for synergy, got {omega_mean}"
