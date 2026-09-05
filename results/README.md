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

1. ✅ **L2 Norm** — Formally closed (negative result: not a valid causal predictor)
2. ✅ **Dropout** — Formally closed, negative (September 5, 2026): variance-under-stochastic-dropout hypothesis from Salah & Yevick (arXiv:2507.11645), implemented on the Nanda-Unified four-head model (k=30 stochastic passes, rate=0.5); variance peak lands at or after grok in 5/5 seeds
3. ✅ **Spectral** — Formally closed, negative (September 6, 2026): Canatar et al. 2021 task-model alignment (representation-kernel `eigh` on the 3830×3830 centred Gram matrix, `k_90` / `alignment_score` per checkpoint); `k_90_min` and `alignment_max` land at or after grok in 5/5 seeds. Canatar's theory is an NTK / lazy regime result; modular-addition grokking is feature-learning regime, so no pre-grok task-power concentration exists to detect
4. ✅ **AGE (Adaptive Grokking Epoch via Neural Collapse)** — Formally closed, negative (September 6, 2026): NC1 within-class variability collapse from Papyan et al. 2020 PNAS (`NC1 = Tr(Sigma_W)/Tr(Sigma_B)` on the penultimate training representations, per checkpoint; mechanism cites Beaglehole et al. 2024 + Mallinar et al. 2024). NC1 collapses hard (~45 → ~0.05) in 5/5 seeds, but `nc1_min_epoch` lags grok in 5/5 seeds (ratio 1.67–3.92; the argmin is pinned to the final checkpoint in 3/5 because NC1 is near-monotone-decreasing). Checkpoint-only, loss-agnostic recompute; same failure shape as Spectral — collapse forms after the test-accuracy jump, not before
5. ⏳ HTSR Alpha — next
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

- **Dropout predictor:** Reopened (September 4, 2026), reversing an earlier same-day decision to close it. A literature search turned up Ahmed Salah & David Yevick, "Tracing the Path to Grokking: Embeddings, Dropout, and Network Activation" (arXiv:2507.11645; also *Neural Processing Letters*, DOI 10.1007/s11063-026-11843-4), which studies grokking on a 2-layer MLP (modular addition, p=53) and reports that the **variance of test accuracy across 100 stochastic dropout forward passes**, measured at checkpoints through training, is near-zero during memorization, spikes at the onset of generalization, and decays back to near-zero once the model fully groks — a "Dropout Robustness Curve" (test accuracy vs. dropout rate 0.0–0.9, at several checkpoints) shows the same story: pre-grok, any dropout collapses accuracy; post-grok, accuracy survives dropout up to roughly rate 0.5. This is a precise, testable hypothesis that maps closely onto infrastructure already built here (`compute_dropout_gap_multi_rate`) — unlike the single final-epoch snapshot the pipeline currently computes, it requires per-checkpoint tracking with repeated stochastic sampling at each checkpoint. Decision: implement this on the current Nanda-Unified four-head model (not the archived single-head architecture), then evaluate the resulting variance-vs-epoch curves against the same proportional-consistency rigor (Criterion 2 from the L2 Norm closure) that Salah & Yevick's own paper does not report. See `opencode_prompt_dropout_variance_predictor.md` (repo root) for the implementation spec. **Outcome (September 5, 2026):** implemented and formally closed, negative — the variance peak lands at or after grok in 5/5 seeds, so it does not lead generalisation.

- **Spectral predictor:** Formally closed, negative (September 6, 2026). The first attempt used a per-weight-matrix SVD proxy (`stable_rank`, `effective_rank`); that was deleted, along with its 5-seed results, because it was not the published method. The predictor was rebuilt as a faithful implementation of Canatar, Bordelon & Pehlevan, "Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks" (*Nature Communications* 12, 2914, 2021): per checkpoint, take the model's hidden state feeding `output_head` over the full training split (`Phi`, N=3830, d=128), centre it, form the Gram matrix `K = Phi_c @ Phi_c.T`, eigendecompose (`eigh` on CPU float64 — MPS has neither), then measure `k_90` / `k_95` (leading eigenmodes holding 90 / 95 % of the one-hot task power), `alignment_score = Σ eta_k_norm · p_k`, and the power entropy. Prediction under test: `k_90` should drop and `alignment_score` should rise at or before grok. **Result:** across 5 seeds, `k_90` falls only 2–3 % over all of training and its minimum, together with `alignment_score`'s maximum, lands at or after grok in every seed. Canatar's theory is an NTK / lazy-training result; modular-addition grokking is a feature-learning phenomenon (a Fourier-multiplication circuit), so there is no pre-grok concentration of task power in the top of the representation-kernel spectrum for it to detect. Valid falsification, correct implementation.

- **AGE predictor:** Formally closed, negative (September 6, 2026). AGE = Adaptive Grokking Epoch via Neural Collapse: the epoch at which Papyan et al. 2020's NC1 "within-class variability collapse" occurs, `NC1 = Tr(Sigma_W)/Tr(Sigma_B) → 0`, measured on the penultimate training representations (`Phi`, N=3830, d=128) at each saved checkpoint — within-class scatter about the per-class means over between-class scatter of the class means about the global mean. Mechanism citations: Beaglehole et al. 2024 (AGOP drives deep neural collapse, arXiv:2402.13728) and Mallinar et al. 2024 (the same AGOP mechanism produces grokking on `(a+b) mod p`, arXiv:2407.20199). Checkpoint-only, loss-agnostic, no weight-matrix SVD. Prediction under test: NC1 bottoms out at or before grok, so `nc1_min_epoch` leads/coincides with the test-accuracy jump. **Result:** across 5 seeds NC1 collapses strongly (~45 → ~0.05) but `nc1_min_epoch` lags grok in 5/5 seeds (`nc1_min / grok` ratio 1.67–3.92), and because NC1 is near-monotone-decreasing the argmin sits at the final checkpoint in 3/5 seeds — it carries essentially no timing information relative to grok. Same failure shape as Spectral: the collapse forms after generalisation, not before it. A not-yet-tried variant (read the collapse *onset* — first checkpoint with NC1 below a fixed threshold — instead of the argmin) is noted in `context.md` for any future revisit. Valid falsification, correct implementation.

- **Four-head baseline:** Both single-head and four-head models grok to perfect accuracy, validating that single-head simplification does not fundamentally change grokking behavior for this task.

## Last Updated

September 6, 2026 — AGE predictor (Adaptive Grokking Epoch via Neural Collapse; Papyan et al. 2020 NC1, mechanism Beaglehole 2024 + Mallinar 2024) formally closed, negative: 5-seed checkpoint-only recompute on the Nanda-Unified four-head model shows NC1 collapsing ~45 → ~0.05 but `nc1_min_epoch` lagging grok in 5/5 seeds (ratio 1.67–3.92). Four of nine predictors now closed (L2 Norm, Dropout, Spectral, AGE — all negative); HTSR Alpha is next. (Same-day earlier update: Spectral predictor (Canatar et al. 2021 task-model alignment) formally closed, negative — `k_90_min` and `alignment_max` land at or after grok in 5/5 seeds; the old per-weight-matrix SVD (`stable_rank`) proxy and its results were deleted first. Previous update: September 4–5, 2026 — Dropout predictor reopened after a literature match with Salah & Yevick, arXiv:2507.11645, then implemented and closed negative.)
