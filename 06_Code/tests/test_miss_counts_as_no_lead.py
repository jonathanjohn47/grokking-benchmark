"""Value 6: a miss counts as no lead in lead_frac and stays in the denominator.

Rho and mean relative lead still use only the firing seeds.
Fewer than 4 firing seeds still fails immediately.
"""

import math
import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

from analyze_nanda_unified import evaluate_frozen_protocol  # noqa: E402


G = [100.0, 200.0, 300.0, 400.0, 500.0]
# Every firing epoch is well before 0.8 * G, and the four values keep rank order.
P_FOUR_LEAD = [10.0, 20.0, 30.0, 40.0, None]
P_THREE = [10.0, 20.0, 30.0, None, None]


def test_four_leads_and_one_miss_is_four_fifths():
    out = evaluate_frozen_protocol(G, P_FOUR_LEAD, rule="fraction", c=0.8)
    assert out["n_firing"] == 4
    assert out["lead_frac"] == pytest.approx(0.8)
    assert out["lead_frac"] != pytest.approx(1.0)
    assert out["rho"] == pytest.approx(1.0)
    # mean of (G - P) / G on the four firing seeds only: 0.9 four times.
    assert out["mean_rel_lead"] == pytest.approx(0.9)


def test_three_firing_seeds_fail_immediately():
    out = evaluate_frozen_protocol(G, P_THREE, rule="fraction", c=0.8)
    assert out["n_firing"] == 3
    assert out["lead_frac"] == 0.0
    assert math.isnan(out["rho"])
    assert out["mean_rel_lead"] == 0.0
