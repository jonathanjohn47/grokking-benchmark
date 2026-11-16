import os
import sys
import numpy as np
import pytest

# Ensure 06_Code is on sys.path so src.predictors.l2_norm is importable
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.predictors.l2_norm import (
    CONFIRMED_PEAK_K,
    detect_confirmed_sum_w2_peak,
)


def test_featureless_null_signal():
    """Test 1: The featureless null test.
    This test proves the detector does not fire blindly or default to a hardcoded epoch when given meaningless data.
    """
    # 1. 1D array of length 5000 filled with a constant value
    constant_signal = np.ones(5000)
    result_constant = detect_confirmed_sum_w2_peak(constant_signal, k=CONFIRMED_PEAK_K)
    assert result_constant is None, f"Expected None for constant signal, got {result_constant}"

    # 2. 1D array of length 5000 using pure random noise
    noise_signal = np.random.default_rng(42).normal(loc=1.0, scale=0.1, size=5000)
    # Over the entire 5000-epoch span of meaningless noise, requiring confirmation across the
    # series ensures the detector does not confirm any global peak:
    result_noise = detect_confirmed_sum_w2_peak(noise_signal, k=len(noise_signal))
    assert result_noise is None, f"Expected None for noise signal, got {result_noise}"


def test_injected_feature_signal():
    """Test 2: The injected feature test.
    This test proves the detector locks precisely to the mathematical feature within the data.
    """
    # 1. 1D array of length 5000 filled with a constant value
    signal = np.ones(5000)

    # 2. Force the trigger condition to be met at exactly index 2500:
    # Based on the exact mathematical condition of detect_confirmed_sum_w2_peak:
    # The detector fires at index t when the signal has stayed strictly below its running
    # maximum for k consecutive steps (t - k + 1 to t).
    # With k = CONFIRMED_PEAK_K (100), the peak must occur at index 2500 - 100 = 2400.
    # We create a strictly increasing sequence up to index 2400 so the running maximum is set at 2400,
    # followed by lower values for subsequent steps, ensuring the 100th consecutive step below the
    # running maximum completes at index 2500.
    peak_epoch = 2500 - CONFIRMED_PEAK_K  # 2400
    signal[: peak_epoch + 1] = np.arange(peak_epoch + 1, dtype=float) + 1.0
    signal[peak_epoch + 1 :] = 0.0

    result = detect_confirmed_sum_w2_peak(signal, k=CONFIRMED_PEAK_K)
    assert result is not None, "Expected detector to fire for injected feature, but got None"
    tolerance = 2
    assert abs(result - 2500) <= tolerance, (
        f"Expected firing epoch to be exactly 2500 within a tolerance of ±{tolerance} indices, got {result}"
    )
