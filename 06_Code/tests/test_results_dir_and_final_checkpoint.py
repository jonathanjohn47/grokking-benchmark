"""Canonical results folder and the final-epoch checkpoint name.

The saved nanda_unified run is legacy. Tests that score it pass that
path themselves. These checks do not train.
"""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))

from analyze_nanda_unified import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    load_seed,
    resolve_results_dir,
)
from run_nanda_benchmark import (  # noqa: E402
    save_checkpoint_schedule,
    select_eval_epochs,
)


def test_default_results_dir_is_the_new_folder():
    assert resolve_results_dir(None).endswith(
        os.path.join("08_Experiments", "results", "results_per_nanda_power")
    )
    assert DEFAULT_RESULTS_DIR == resolve_results_dir(None)
    legacy = resolve_results_dir("08_Experiments/results/nanda_unified")
    assert legacy.endswith(os.path.join("results", "nanda_unified"))
    assert legacy != resolve_results_dir(None)


def test_results_dir_argument_changes_the_read_path(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_seed(0, results_dir=str(tmp_path))
    assert os.path.join(str(tmp_path), "seed_0", "summary.json") in str(exc.value)


def test_final_checkpoint_is_saved_once_on_the_eval_grid():
    saved = save_checkpoint_schedule(40000, save_every=100)
    assert saved[0] == 0
    assert saved[-1] == 40000
    assert 39900 in saved
    assert saved.count(40000) == 1
    chosen = select_eval_epochs(saved, eval_every=100)
    assert chosen.count(40000) == 1
    assert chosen == sorted(set(chosen))
    assert chosen[-1] == 40000
