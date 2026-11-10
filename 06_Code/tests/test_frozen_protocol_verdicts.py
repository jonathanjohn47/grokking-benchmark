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


@pytest.fixture(scope="module")
def scored():
    return score_saved_run(lead_c=0.8, write=False)


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
