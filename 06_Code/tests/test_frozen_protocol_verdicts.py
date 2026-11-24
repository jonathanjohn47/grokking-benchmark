"""The official scorer's predictor-level verdicts, checked in code.

Loads the saved Nanda-unified results and calls score_saved_run.
Does not write files. On that legacy folder the confirmed-peak L2 row
passes the frozen checks. The other seven predictors stay FAIL.
The retrain folder is scored in test_l2_confirmed_peak.py.
"""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

from analyze_nanda_unified import (  # noqa: E402
    predictor_level_verdicts,
    score_saved_run,
    summarise,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Saved run from the previous setup. The canonical folder is
# results_per_nanda_power; these tests name the legacy folder explicitly.
LEGACY_RESULTS = os.path.join(REPO, "08_Experiments", "results", "nanda_unified")


@pytest.fixture(scope="module")
def scored():
    return score_saved_run(lead_c=0.8, write=False, results_dir=LEGACY_RESULTS)


def test_l2_official_row_is_the_confirmed_peak(scored):
    """One official row: the confirmed peak of the total sum of squared weights."""
    l2 = next(p for p in scored["predictors"] if p["predictor"] == "L2 Norm")
    names = [r["signal"] for r in l2["rows"]]
    assert names == ["L2: confirmed peak of sum w^2 (k=100)"]
    official = [r for r in l2["rows"] if r["counts_towards_predictor"]]
    assert len(official) == 1
    assert l2["n_predeclared_rows"] == 1
    assert l2["n_predeclared_passing"] == 1
    assert l2["verdict"] == "PASS"
    assert official[0]["verdict"] == "PASS"


def test_l2_confirmed_peak_passes_legacy_at_neighbouring_lead_fractions():
    for lead_c in (0.7, 0.9):
        scored = score_saved_run(lead_c=lead_c, write=False, results_dir=LEGACY_RESULTS)
        l2 = next(p for p in scored["predictors"] if p["predictor"] == "L2 Norm")
        official = [r for r in l2["rows"] if r["counts_towards_predictor"]]
        assert [r["signal"] for r in official] == [
            "L2: confirmed peak of sum w^2 (k=100)",
        ]
        assert l2["n_predeclared_rows"] == 1
        assert l2["n_predeclared_passing"] == 1
        assert l2["verdict"] == "PASS"


def test_every_predictor_fails(scored):
    expected = {
        "L2 Norm": "PASS",
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


# ---------------------------------------------------------------------------
# Canonical run: HTSR Alpha, Weight-PCA, Higher-MI and Commutator Defect are
# read from seed_N/<name>/<name>_signal.json (summary.json blocks are null).
# Expected values were computed from the saved signal files with the frozen
# rule at c = 0.8. G_i = [8021, 8253, 4094, 7831, 6637].
# ---------------------------------------------------------------------------

CANONICAL_RESULTS = os.path.join(
    REPO, "08_Experiments", "results", "results_per_nanda_power"
)

# predictor -> (row name, P_i, lead_frac, rho, mean_rel_lead, checks)
EXPECTED_CANONICAL = {
    "HTSR Alpha": (
        "HTSR: mean alpha minimum (implemented)",
        [31100.0, 17300.0, 17600.0, 16000.0, 28700.0],
        0.0, -0.09999999999999999, -2.327981650307303,
        {"n_firing_at_least_4": True, "lead_frac_at_least_0.8": False,
         "rho_at_least_threshold": False, "mean_rel_lead_in_window": False},
    ),
    "Weight-PCA": (
        "Weight-PCA: effective-rank minimum (implemented)",
        [33200.0, 40000.0, 38400.0, 30500.0, 23000.0],
        0.0, 0.39999999999999997, -4.145127067970694,
        {"n_firing_at_least_4": True, "lead_frac_at_least_0.8": False,
         "rho_at_least_threshold": False, "mean_rel_lead_in_window": False},
    ),
    "Higher-MI": (
        "Higher-MI: population omega maximum (implemented)",
        [20100.0, 27800.0, 12000.0, 12300.0, 7300.0],
        0.0, 0.8999999999999998, -1.2952175789744158,
        {"n_firing_at_least_4": True, "lead_frac_at_least_0.8": False,
         "rho_at_least_threshold": True, "mean_rel_lead_in_window": False},
    ),
    "Commutator Defect": (
        "Commutator Defect: Xu onset on median D (implemented)",
        [400.0, 300.0, 500.0, 300.0, 300.0],
        1.0, -0.44721359549995787, 0.9416280226732147,
        {"n_firing_at_least_4": True, "lead_frac_at_least_0.8": True,
         "rho_at_least_threshold": False, "mean_rel_lead_in_window": False},
    ),
}


@pytest.fixture(scope="module")
def canonical():
    return score_saved_run(lead_c=0.8, write=False, results_dir=CANONICAL_RESULTS)


def test_canonical_four_predictors_are_scored_and_fail(canonical):
    by_name = {p["predictor"]: p for p in canonical["predictors"]}
    for predictor, expected in EXPECTED_CANONICAL.items():
        signal, p_i, lead_frac, rho, mean_rel, checks = expected
        block = by_name[predictor]
        assert block["verdict"] == "FAIL", predictor
        assert block["n_predeclared_rows"] == 1, predictor
        assert block["n_predeclared_passing"] == 0, predictor
        assert [r["signal"] for r in block["rows"]] == [signal]
        row = block["rows"][0]
        assert row["role"] == "predeclared"
        assert row["counts_towards_predictor"] is True
        assert row["P_i"] == p_i
        assert row["n_firing"] == 5
        assert row["lead_frac"] == pytest.approx(lead_frac, abs=1e-12)
        assert row["rho"] == pytest.approx(rho, abs=1e-12)
        assert row["mean_rel_lead"] == pytest.approx(mean_rel, abs=1e-12)
        assert row["checks"] == checks
        assert row["verdict"] == "FAIL"
        # Validity Gate + Superiority Index are attached like L2/Spectral/Dropout/AGE.
        assert row["gate_lead_frac"] == pytest.approx(lead_frac, abs=1e-12)
        assert row["gate_correlation"] == pytest.approx(rho, abs=1e-12)
        assert row["gate_noise_margin"] == pytest.approx(mean_rel, abs=1e-12)
        assert row["gate_pass"] is False
        assert row["superiority_index"] == 0.0


def test_canonical_every_predictor_is_scored_and_fails(canonical):
    got = {p["predictor"]: p["verdict"] for p in canonical["predictors"]}
    assert got == {
        "L2 Norm": "FAIL",
        "Dropout": "FAIL",
        "Spectral": "FAIL",
        "AGE": "FAIL",
        "HTSR Alpha": "FAIL",
        "Weight-PCA": "FAIL",
        "Higher-MI": "FAIL",
        "Commutator Defect": "FAIL",
    }
    rows = {p["predictor"]: p["n_predeclared_rows"] for p in canonical["predictors"]}
    assert rows == {
        "L2 Norm": 1,
        "Dropout": 2,
        "Spectral": 6,
        "AGE": 3,
        "HTSR Alpha": 1,
        "Weight-PCA": 1,
        "Higher-MI": 1,
        "Commutator Defect": 1,
    }


def test_predictor_with_no_rows_is_not_scored():
    grok = [100.0, 200.0, 300.0, 400.0, 500.0]
    events = {
        "L2: confirmed peak of sum w^2 (k=100)": {
            0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0,
        },
    }
    predictors = predictor_level_verdicts(summarise(events, grok, lead_c=0.8))
    got = {p["predictor"]: p["verdict"] for p in predictors}
    assert got["L2 Norm"] in ("PASS", "FAIL")
    for name in ("Dropout", "Spectral", "AGE", "HTSR Alpha", "Weight-PCA",
                 "Higher-MI", "Commutator Defect"):
        assert got[name] == "NOT_SCORED", name
        block = next(p for p in predictors if p["predictor"] == name)
        assert block["n_predeclared_rows"] == 0
        assert block["rows"] == []
