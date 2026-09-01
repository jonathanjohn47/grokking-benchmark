# Grokking Predictors Benchmark

**Thesis:** A Unified Benchmark of Grokking Predictors in Neural Networks

**Stack:** Python, PyTorch, Apple Silicon (MPS)

---

## Directory Structure

### Core Project Files

- **`CLAUDE.md`** — Project rules, instructions, and guidelines
- **`context.md`** — Project memory, session history, current status, and technical decisions
- **`requirements.txt`** — Python dependencies

### Source Code

- **`src/`** — Core implementation
  - `models/` — Neural network architectures (transformer, four-head, shadow models)
  - `data/` — Dataset and data loading
  - `predictors/` — Grokking predictors (L2 Norm, Dropout, etc.)
  - `train*.py` — Training scripts

### Experiments & Results

- **`results/`** — Experiment outputs (organized by model type and predictor)
  - `single_head/` — Single-head transformer results
  - `four_head/` — Four-head transformer results
  - `experiments/` — Exploratory experiments

### Tools & Utilities

- **`tools/`** — Miscellaneous Python scripts
  - PDF compilation tools
  - Image processing utilities
  - Analysis and debugging scripts
  - One-off utility scripts

### Documentation

- **`docs/`** — All documentation and analysis
  - `*.md` — Analysis notes and project documentation
  - `reports/` — PDF reports and analysis outputs
  - `Thesis Gantt*.csv` — Project timeline
  - Supporting files

### Other

- **`graphify-out/`** — Knowledge graph outputs (auto-generated)
- **`images/`** — Generated images and visualizations
- **`_to_delete/`** — Files marked for deletion

---

## Quick Reference

**To understand the project:** Read `context.md`

**To run experiments:** See training scripts in `src/train*.py`

**To view results:** Check `results/` for per-run metrics and reports

**To use utilities:** Scripts are in `tools/`
