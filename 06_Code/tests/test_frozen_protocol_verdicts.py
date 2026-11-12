"""The official scorer's predictor-level verdicts, checked in code.

Loads the saved Nanda-unified results and calls score_saved_run.
Does not write files. Every predictor is FAIL at c = 0.8: no pre-declared
row clears all four PASS checks.
"""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

from analyze_nanda_unified import score_saved_run  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Saved run from the previous setup. The canonical folder is
# results_per_nanda_power; these tests name the legacy folder explicitly.
LEGACY_RESULTS = os.path.join(REPO, "08_Experiments", "results", "nanda_unified")


@pytest.fixture(scope="module")
def scored():
    return score_saved_run(lead_c=0.8, write=False, results_dir=LEGACY_RESULTS)


def test_l2_official_rows_are_the_two_moving_average_rules(scored):
    """The hindsight half-way-down-from-peak reading is not an official row."""
    l2 = next(p for p in scored["predictors"] if p["predictor"] == "L2 Norm")
    names = [r["signal"] for r in l2["rows"]]
    assert "L2: sum(w^2) half-way down from peak" not in names
    assert names == [
        "L2: MA crossover (implemented)",
        "L2: MA-of-MA zero crossing (implemented)",
    ]
    assert l2["n_predeclared_rows"] == 2
    assert l2["n_predeclared_passing"] == 0
    assert l2["verdict"] == "FAIL"
    assert all(r["verdict"] == "FAIL" and r["counts_towards_predictor"] for r in l2["rows"])


def test_l2_still_fails_at_neighbouring_lead_fractions():
    for lead_c in (0.7, 0.9):
        scored = score_saved_run(lead_c=lead_c, write=False, results_dir=LEGACY_RESULTS)
        l2 = next(p for p in scored["predictors"] if p["predictor"] == "L2 Norm")
        names = [r["signal"] for r in l2["rows"]]
        assert names == [
            "L2: MA crossover (implemented)",
            "L2: MA-of-MA zero crossing (implemented)",
        ]
        assert l2["n_predeclared_rows"] == 2
        assert l2["n_predeclared_passing"] == 0
        assert l2["verdict"] == "FAIL"


def test_every_predictor_fails(scored):
    expected = {
        "L2 Norm": "FAIL",
        "Dropout": "FAIL",
        "Spectral": "FAIL",
        "AGE": "FAIL",
        "HTSR Alpha": "FAIL",
        "Weight-PCA": "FAIL",
        "Higher-MI": "FAIL",
        "Commutator Defect": "FAIL",
    }
    got = {p["predictor"]: p["verdict"] for p in scored["predictors"]}
    assert got == expected


def test_exploratory_rows_do_not_count(scored):
    higher = next(p for p in scored["predictors"] if p["predictor"] == "Higher-MI")
    exploratory = [r for r in higher["rows"] if r["role"] == "exploratory"]
    assert exploratory
    assert all(r["counts_towards_predictor"] is False for r in exploratory)
    predeclared = [r for r in higher["rows"] if r["counts_towards_predictor"]]
    assert predeclared
    assert all(r["verdict"] == "FAIL" for r in predeclared)


def test_each_row_has_separate_checks(scored):
    keys = {
        "n_firing_at_least_4",
        "lead_frac_at_least_0.8",
        "rho_at_least_threshold",
        "mean_rel_lead_in_window",
    }
    for predictor in scored["predictors"]:
        assert predictor["rows"], predictor["predictor"]
        for row in predictor["rows"]:
            assert set(row["checks"]) == keys
            assert row["verdict"] in ("PASS", "FAIL")
            assert row["verdict"] == ("PASS" if all(row["checks"].values()) else "FAIL")
            assert "n_firing" in row and "lead_frac" in row
            assert "rho" in row and "mean_rel_lead" in row
