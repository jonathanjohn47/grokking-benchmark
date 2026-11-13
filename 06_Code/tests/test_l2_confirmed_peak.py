"""Confirmed-peak firing rule for the L2 sum of squared weights.

k = 100 is an own-choice confirmation window. No paper gives one.
The retrain histories are read only. Nothing is trained and nothing is written.
"""

import os
import sys

import numpy as np
import pytest

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))

from src.predictors.l2_norm import (  # noqa: E402
    CONFIRMED_PEAK_K,
    detect_confirmed_sum_w2_peak,
    sum_w2_peak,
)
from analyze_nanda_unified import score_l2_saved_histories  # noqa: E402

REPO = os.path.abspath(os.path.join(CODE, ".."))
RETRAIN = os.path.join(REPO, "08_Experiments", "results", "results_per_nanda_power")


def test_k_is_the_predeclared_own_choice():
    assert CONFIRMED_PEAK_K == 100


def test_fires_k_epochs_after_a_peak_that_is_never_beaten():
    series = np.concatenate([np.linspace(1.0, 5.0, 6), np.linspace(4.0, 1.0, 20)])
    # index 5 is the unique maximum. The next 100-length window is shorter
    # than k, so use k=4: epochs 6,7,8,9 are strictly below the running max.
    assert sum_w2_peak(series)[0] == 5
    assert detect_confirmed_sum_w2_peak(series, k=4) == 9


def test_later_higher_peak_does_not_move_the_first_completed_window():
    series = np.zeros(30)
    series[0] = 1.0
    series[1] = 3.0
    series[2:8] = 2.0
    series[8] = 10.0
    series[9:] = 4.0
    # Running max is set at epoch 1. Epochs 2..6 are five steps below it.
    assert detect_confirmed_sum_w2_peak(series, k=5) == 6
    assert sum_w2_peak(series) == (8, 10.0)


def test_never_below_the_running_maximum_returns_none():
    series = np.arange(50, dtype=float)
    assert detect_confirmed_sum_w2_peak(series, k=100) is None


def test_retrain_official_row_fails_the_frozen_protocol():
    scored = score_l2_saved_histories(results_dir=RETRAIN, lead_c=0.8)
    assert scored["grok_epochs"] == [8021, 8253, 4094, 7831, 6637]
    official = next(r for r in scored["rows"] if r["counts_towards_predictor"])
    assert official["signal"] == "L2: confirmed peak of sum w^2 (k=100)"
    assert official["epochs"] == [279.0, 280.0, 276.0, 281.0, 281.0]
    verdict = official["frozen_protocol_verdict"]
    assert verdict["n_firing"] == 5
    assert verdict["verdict"] == "FAIL"
    assert verdict["checks"]["mean_rel_lead_in_window"] is False
    assert verdict["lead_frac"] == 1.0
    assert round(verdict["rho"], 4) == 0.1539
    assert round(verdict["mean_rel_lead"], 4) == 0.9571
    assert [r for r in scored["rows"] if not r["counts_towards_predictor"]] == []


@pytest.mark.parametrize("lead_c", [0.7, 0.9])
def test_retrain_official_row_fails_at_the_check_fractions(lead_c):
    scored = score_l2_saved_histories(results_dir=RETRAIN, lead_c=lead_c)
    official = next(r for r in scored["rows"] if r["counts_towards_predictor"])
    assert official["frozen_protocol_verdict"]["verdict"] == "FAIL"
