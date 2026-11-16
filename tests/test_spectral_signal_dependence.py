import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_SRC = os.path.join(REPO_ROOT, "06_Code", "src")
if CODE_SRC not in sys.path:
    sys.path.insert(0, CODE_SRC)

from predictors.spectral import detect_spectral_trigger


GRID = np.arange(40001, dtype=int)


def test_flat_constant_returns_none():
    """T1: k90 = 1000 everywhere, alignment = 0.30 everywhere. Expected: None."""
    k90 = np.full(40001, 1000.0)
    alignment = np.full(40001, 0.30)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is None


def test_low_amplitude_gaussian_noise_returns_none():
    """T2: k90 = 1000 + N(0, 10), alignment = 0.30 + N(0, 0.01) with default_rng(0). Expected: None."""
    rng = np.random.default_rng(0)
    k90 = 1000.0 + rng.normal(0.0, 10.0, size=40001)
    alignment = 0.30 + rng.normal(0.0, 0.01, size=40001)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is None


def test_smooth_convergence_without_feature_returns_none():
    """T3: k90 decays smoothly from 1000 to 950; alignment rises smoothly from 0.30 to 0.45. Expected: None."""
    k90 = np.linspace(1000.0, 950.0, 40001)
    alignment = np.linspace(0.30, 0.45, 40001)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is None


def test_injected_joint_step_fires_at_2500():
    """T4: k90 = 1000 (<2500) and 600 (>=2500); alignment = 0.30 (<2500) and 0.80 (>=2500).

    Expected: not None, abs(P_i - 2500) <= 2, exact value 2501.
    """
    k90 = np.where(GRID < 2500, 1000.0, 600.0)
    alignment = np.where(GRID < 2500, 0.30, 0.80)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is not None
    assert abs(result - 2500) <= 2
    assert result == 2501


def test_injected_k90_step_only_fires_exactly_2500():
    """T5: k90 = 1000 (<2500) and 600 (>=2500); alignment = 0.80 everywhere. Expected: exactly 2500."""
    k90 = np.where(GRID < 2500, 1000.0, 600.0)
    alignment = np.full(40001, 0.80)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result == 2500


def test_alignment_only_step_returns_none():
    """T6: k90 = 1000 everywhere; alignment = 0.30 (<2500) and 0.80 (>=2500). Expected: None."""
    k90 = np.full(40001, 1000.0)
    alignment = np.where(GRID < 2500, 0.30, 0.80)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is None


def test_burnin_suppression():
    """T7: k90 = 1000 everywhere except drop to 600 in [0, 150]; alignment = 0.80 everywhere. Expected: None."""
    k90 = np.full(40001, 1000.0)
    k90[0:151] = 600.0
    alignment = np.full(40001, 0.80)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is None


def test_persistence_requirement():
    """T8: single-checkpoint spike, k90 = 600 only at epoch 2500 and 1000 elsewhere; alignment = 0.80 everywhere.

    Expected: None (3-consecutive persistence fails).
    """
    k90 = np.full(40001, 1000.0)
    k90[2500] = 600.0
    alignment = np.full(40001, 0.80)
    result = detect_spectral_trigger(GRID, k90, alignment)
    assert result is None


def test_determinism():
    """T9: run T4 twice, assert identical output."""
    k90 = np.where(GRID < 2500, 1000.0, 600.0)
    alignment = np.where(GRID < 2500, 0.30, 0.80)
    res1 = detect_spectral_trigger(GRID, k90, alignment)
    res2 = detect_spectral_trigger(GRID, k90, alignment)
    assert res1 == res2
    assert res1 == 2501
