# 06_Code

All source code for the thesis benchmark.

## Structure

- `src/` — core implementation
  - `models/` — network architectures (transformer, four-head, shadow-layernorm variants)
  - `data/` — dataset generation and loading (synthetic modular-arithmetic tasks)
  - `predictors/` — grokking predictors (L2 norm, dropout, spectral, age)
  - `train*.py`, `*measurements.py`, `plot_*.py` — training and analysis scripts for the variants in `src/`
- `scripts/` — standalone utility and driver scripts
  - `run_nanda_benchmark.py` — main entry point for the unified benchmark (see root README for usage)
  - `compile_context_bundle.py`, `compile_python_files.py` — documentation/report generation tools
  - `analyze_threshold.py`, `debug_detection.py` — analysis and debugging utilities
  - `md_to_image.py`, `collage_images.py`, `generate_file.py`, `scaffold.py` — misc helpers
  - `markPhase1Task1Complete.gs` — a Google Apps Script helper, unrelated to the Python codebase
- `configs/` — YAML configuration files for benchmark runs (e.g. `nanda_unified.yaml`)
- `notebooks/` — reserved for exploratory notebooks (currently empty)
- `tests/` — reserved for unit tests (currently empty)
- `outputs/` — reserved for ad-hoc script outputs (currently empty)
- `requirements.txt` — Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

See the root `README.md` for the main benchmark command. Individual training scripts under `src/` (e.g. `train_four_head.py`, `train_shadow_layernorm.py`) can also be run directly with `python 06_Code/src/<script>.py`; check each script's own `--help` for arguments.

Note: the self-contained `nanda_l2_p113` experiment package has its own code and is kept under `08_Experiments/nanda_l2_p113/` rather than here, since it is a specific experiment rather than shared library code.
