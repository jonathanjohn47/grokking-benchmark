# Results Directory Structure

This directory contains all experiment results organized by architecture variant and predictor.

## Overview

```
results/
├── single_head/              Single-head transformer results (main model)
│   ├── training/             General training data (accuracy, loss)
│   ├── l2_norm/              L2 Norm predictor results
│   ├── dropout/              Dropout predictor results
│   └── reports/              Summary PDF reports
│
├── four_head/                Four-head transformer results (Nanda et al. architecture)
│   ├── run_1/                First independent training run
│   ├── run_2/                Second independent training run
│   ├── run_3/                Third independent training run
│   ├── comparisons/          Cross-run comparison plots
│   └── reports/              Four-head specific PDF reports
│
└── experiments/              Alternative configurations and side investigations
    └── shadow_layernorm/     LayerNorm experiment (negative result, not adopted)
```

## Single-Head Results

### `training/`
Contains general training metrics for a single 10,000-epoch run:
- `train_acc_history.npy` — Training accuracy per epoch
- `test_acc_history.npy` — Test accuracy per epoch  
- `loss_history.npy` — Training loss per epoch
- `training_report.pdf` — 4-page PDF with graphs (grokking curve, loss, L2 norm, Dropout Gap)

### `l2_norm/`
Contains all L2 Norm predictor data and analysis:
- **Raw metrics:** `l2_norm_history.npy` — L2 norm of model weights per epoch
- **Moving averages:** `slow_ma.npy`, `fast_ma.npy`, `fast_ma_of_slow_ma.npy` — Smoothed L2 norm curves
- **Trigger data:** `ma_of_ma_diff.npy`, `epoch_grid.npy` — Used for zero-crossing detection
- **Plots:**
  - `ma_of_slow_ma_crossover.png` — Slow MA vs. fast-MA-of-slow-MA with trigger marked
  - `ma_of_slow_ma_diff.png` — Difference curve (log-x scale)
  - `ma_of_slow_ma_diff_linear.png` — Difference curve (linear x-scale)
  - `ma_of_ma_diff_vs_grokking_linear.png` — Difference overlaid on grokking curve

### `dropout/`
Contains all Dropout predictor data:
- **Single-rate (p=0.9):**
  - `dropout_gap_history.npy` — Gap between clean and dropout-stressed accuracy
  - `dropout_train_acc_history.npy` — Training accuracy under dropout (p=0.9)
  - `dropout_eval_acc_history.npy` — Evaluation accuracy without dropout
  - `dropout_gap_epochs.npy` — Epoch indices
- **Multi-rate sweep (p ∈ {0.1, 0.3, 0.5, 0.7, 0.9}):**
  - `dropout_gap_by_rate.npy` — Gap history for all 5 rates (shape: 5 × num_epochs)
  - `dropout_rates.npy` — The 5 dropout rates used
- **Infographic:**
  - `dropout_gap_infographic.png` — Visual explanation of the Dropout predictor concept

### `reports/`
Summary PDF reports on L2 Norm predictor results:
- `L2_Norm_Comprehensive_Report_CORRECTED.pdf` — Full analysis with corrected epoch axis
- `L2_Norm_Comprehensive_Report.pdf` — Earlier version (has epoch-axis bug)
- `L2_Norm_Predictor_Report.pdf` — Initial single-head analysis

## Four-Head Results

The four-head transformer (matching Nanda et al. 2023: `d_model=128`, `num_heads=4`) is being evaluated to compare against the single-head baseline.

### Per-Run Structure (`run_1/`, `run_2/`, `run_3/`)
Each run contains the same predictor-organized subdirectories:

- **`training/`:**
  - `train_acc_history.npy`, `test_acc_history.npy`, `loss_history.npy`
  - `grokking_curve.png`, `loss_curve.png`
  - `training_report.pdf` — 4-page PDF report for this run

- **`l2_norm/`:**
  - All L2 norm metrics and plots (same as single-head)
  - Additional data: `seed.npy` — Random seed used for this run

- **`dropout/`:**
  - Dropout predictor results (if collected for this run)
  - Note: Not all runs may have dropout data yet

- **`reports/`:**
  - Any run-specific summary reports

### `comparisons/`
Cross-run overlay plots showing stochastic variation:
- `comparison_grokking_curve.png` — All runs' grokking curves overlaid
- `comparison_loss_curve.png` — All runs' loss curves overlaid
- `comparison_l2_norm_curve.png` — All runs' L2 norm curves overlaid
- `comparison_dropout_gap_curve.png` — All runs' Dropout Gap curves overlaid

## Experiments

### `shadow_layernorm/`
Investigation of LayerNorm addition (negative result, not adopted):
- `shadow_ln_train_acc_history.npy`, `shadow_ln_test_acc_history.npy`, `shadow_ln_loss_history.npy`
- `shadow_ln_grokking_curve.png` — Shows that naive LayerNorm addition caused training instability

**Finding:** LayerNorm was not adopted because adding it destabilized training under `weight_decay=1.0`, requiring full hyperparameter re-tuning that would break fair comparison across predictors.

## Predictor Evaluation Order

Results are being collected for all 9 grokking predictors in this order:

1. ✅ **L2 Norm** — Formally closed (negative result)
2. 🔄 **Dropout** — Under investigation (multi-rate sweep in progress)
3. ⏳ Spectral
4. ⏳ AGE (Adaptive Grokking Epoch)
5. ⏳ HTSR Alpha
6. ⏳ Correlation Traps
7. ⏳ Weight-PCA
8. ⏳ Higher-MI
9. ⏳ Commutator Defect

## How to Use This Structure

1. **Quick summary of single-head results:** Open `single_head/reports/L2_Norm_Comprehensive_Report_CORRECTED.pdf`

2. **Reproduce plots:** Run:
   ```bash
   python src/plot_results.py  # Single-head L2 Norm plots
   python src/plot_results_four_head.py  # Four-head comparison plots
   ```

3. **Run new training:**
   ```bash
   python src/train.py  # Single-head (10,000 epochs, saves to single_head/)
   python src/train_four_head.py  # Four-head (40,000 epochs, saves to four_head/run_N/)
   ```

4. **Access raw data programmatically:**
   ```python
   import numpy as np
   
   # Single-head training data
   train_acc = np.load("results/single_head/training/train_acc_history.npy")
   test_acc = np.load("results/single_head/training/test_acc_history.npy")
   
   # Single-head L2 Norm data
   l2_norm = np.load("results/single_head/l2_norm/l2_norm_history.npy")
   
   # Single-head Dropout data
   dropout_gap = np.load("results/single_head/dropout/dropout_gap_history.npy")
   dropout_gap_by_rate = np.load("results/single_head/dropout/dropout_gap_by_rate.npy")
   
   # Four-head run 1 training
   train_acc_4h = np.load("results/four_head/run_1/training/train_acc_history.npy")
   ```

## Key Findings

- **L2 Norm predictor:** Formally evaluated on both single-head and four-head models. Five detection strategies tested, none met the validation criteria (must always precede grokking, show tight consistent gap, be clearly above noise floor).

- **Dropout predictor:** Single-rate (p=0.9) shows promise but insufficient to prove robustness of the signal. Multi-rate sweep (0.1–0.9) is underway to determine if gap-narrowing is real or rate-dependent artifact.

- **Four-head baseline:** Both single-head and four-head models grok to perfect accuracy, validating that single-head simplification does not fundamentally change grokking behavior for this task.

## Last Updated

August 24, 2026 — Results reorganized from project root into predictor-based structure with separate single-head and four-head variants.
