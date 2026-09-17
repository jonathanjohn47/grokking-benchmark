# A Unified Benchmark of Grokking Predictors in Neural Networks

## Student

Jonathan John — M.Sc. Artificial Intelligence, IU Internationale Hochschule (Germany)

## Supervisor

Sheikh Faisal Rashid

## Research Objectives

This thesis builds a unified benchmark to compare early predictors of grokking (delayed generalization) in small transformer models trained on algorithmic tasks. The main objectives are:

1. Reproduce known grokking behaviour on modular-arithmetic tasks (following Nanda et al.) as a reference baseline.
2. Implement and compare several candidate predictors of the grokking transition — including the L2 weight-norm predictor, a dropout-gap predictor, an age-based signal, and a spectral (task-model alignment) predictor.
3. Run these predictors across multiple seeds and configurations to measure how early and how reliably each one detects the onset of grokking.
4. Draw conclusions on which predictor, or combination of predictors, gives the most reliable early-warning signal, and document the evidence in the thesis document.

## Repository Structure

```
grokking-benchmark/
├── README.md                  This file
├── 01_Admin/                  Proposal, timeline, ethics approval, misc admin material
├── 02_Progress_Reports/       Weekly/monthly progress reports
├── 03_Meeting_Minutes/        Supervision meeting minutes
├── 04_Literature/             Papers, reading notes, summaries
├── 05_Thesis_Document/        Thesis drafts, figures, tables, bibliography
├── 06_Code/                   All source code (see 06_Code/README.md)
├── 07_Data/                   Datasets (see 07_Data/README.md)
├── 08_Experiments/            Per-experiment configs, logs, results
├── 09_Presentations/          Slides for proposal, midterm, final defense, posters
├── 10_Publications/           Any manuscripts, reviews, correspondence
└── Archive/                   Deprecated code and superseded drafts, kept for reference
```

Files kept outside this numbered structure, and left where they are:

- `CLAUDE.md` — working instructions for the AI coding assistant used during development; not thesis content.
- `.claude/`, `graphify-out/` — assistant tooling caches, regenerated automatically.
- `.venv/`, `__pycache__/`, `.idea/`, `.vs/` — local Python environment and IDE state, excluded from version control.
- `runs/` — scratch output folder used by older scripts; empty and gitignored.

## Software Requirements

- Python 3.10+ (see `06_Code/requirements.txt` for exact packages, mainly PyTorch)
- A machine with a GPU or Apple Silicon (MPS) is recommended for training speed, but not required for small-scale runs.

Install with:

```bash
pip install -r 06_Code/requirements.txt
```

## Instructions to Reproduce Experiments

1. Install dependencies as above.
2. The main unified benchmark is run with:

   ```bash
   python 06_Code/scripts/run_nanda_benchmark.py --seeds 5 --epochs 40000 \
       --output_dir 08_Experiments/results/nanda_unified --config 06_Code/configs/nanda_unified.yaml
   ```

   For a quick smoke test:

   ```bash
   python 06_Code/scripts/run_nanda_benchmark.py --seeds 1 --epochs 100 \
       --output_dir 08_Experiments/results/test_smoke
   ```

3. A separate, self-contained reproduction of the Nanda L2-norm predictor at p=113 lives in `08_Experiments/nanda_l2_p113/` and can be run with its own `run_benchmark.py` — see `08_Experiments/nanda_l2_p113/README.md`.
4. Existing results for the unified benchmark are already checked into `08_Experiments/results/nanda_unified/` (per-seed metrics, plots, and an `aggregate.json` summary), so plots and reports can be regenerated from these without re-running training.
5. Data used for training is generated synthetically by the code itself (modular arithmetic tasks); see `07_Data/README.md` for details.
