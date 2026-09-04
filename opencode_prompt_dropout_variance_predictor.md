# Objective

Add a per-checkpoint, stochastic-dropout **variance** signal to the Nanda-Unified benchmark's Dropout predictor, so it can be evaluated as a live grokking indicator — not just the single final-epoch dropout-gap sweep the pipeline currently computes. Implement the hypothesis reported in Salah & Yevick, "Tracing the Path to Grokking: Embeddings, Dropout, and Network Activation" (arXiv:2507.11645; also *Neural Processing Letters*, DOI 10.1007/s11063-026-11843-4): the variance of test accuracy across repeated stochastic dropout forward passes, tracked over training, is near-zero during memorization, spikes around the onset of generalization, and decays back to near-zero after the model fully groks.

# Context

This is a Ph.D./M.Sc. thesis benchmark ("A Unified Benchmark of Grokking Predictors in Neural Networks") that trains a Nanda-et-al.-style four-head Transformer on modular addition and evaluates candidate scalar signals ("predictors") for whether they can flag the grokking transition before or as it happens. L2 Norm was already evaluated and formally closed as a negative result (its trigger epoch does not track the actual grok epoch across seeds). Dropout was evaluated only as a single post-training, multi-rate accuracy-gap snapshot on the final (already-grokked) model — this told us nothing about *when*, during training, anything interesting happens, because no per-epoch trajectory was ever collected for Dropout under the current (Nanda-Unified, four-head, modulus 113) pipeline.

A literature search today found the Salah & Yevick paper, which is the closest existing match for a dropout-based grokking predictor and reports a concrete, per-checkpoint trajectory measurement — one this project's own `compute_dropout_gap_multi_rate` infrastructure is already close to. Their setup is a much smaller 2-layer MLP (not a Transformer) on modular addition p=53; this project must reproduce the *measurement*, not their architecture — stay on the existing Nanda-Unified four-head Transformer, modulus 113, do NOT revive the archived single-head architecture (`archive/single_head/`, `archive/src_p97_mainstream_old/`).

# Relevant Findings

- Salah & Yevick's method: at a training checkpoint, freeze the model's weights, then perform 100 stochastic forward passes over the test set with dropout active (`model.train()`, dropout layers at some rate), recording test accuracy on each pass. Compute the variance of those 100 test-accuracy values. Repeat at multiple checkpoints spanning training. Their result: this variance is near-zero during memorization, rises sharply "at the onset of generalization," then decays back toward zero once both train and test accuracy have saturated at their maxima. They frame it as a "concurrent or slightly leading" correlate of the grok transition — their paper does **not** report hard trigger-epoch-vs-grok-epoch numbers per run the way this project's L2 Norm work does, so its status as a genuinely leading (as opposed to merely coincident) signal is unproven and must be tested here, not assumed.
- They separately define a **Dropout Robustness Curve (DRC)**: test accuracy vs. dropout rate (0.0–0.9), plotted at a handful of representative epoch checkpoints. Pre-grok, accuracy collapses under even light dropout; post-grok, accuracy survives dropout up to roughly rate 0.5. This is a qualitative/visual companion to the variance metric, not the primary quantitative signal.
- This project's `src/predictors/dropout.py` already has `compute_dropout_gap_multi_rate(model, data_loader, dropout_rates)`, which does one clean-accuracy pass plus one dropout-mode pass per rate — reusable as the building block for repeated stochastic sampling, but it currently returns a single accuracy value per rate, not a distribution over repeated stochastic passes.
- `run_nanda_benchmark.py` already has a working pattern for a "signals from a trajectory" predictor in its L2-Norm block (`compute_fast_slow_moving_averages`, log-uniform resampling from `resample_to_log_uniform_grid`, noise-floor and trigger detection) and writes both raw per-epoch `.npy` arrays and a compact `*_signals.json` per seed, then rolls per-seed signals into `aggregate.json`. Follow the same shape of pattern for this new signal instead of inventing a new one.
- **Compute-cost constraint, worked out this session:** doing K=100 stochastic passes at many checkpoints, at all 5 existing dropout rates, for all 5 seeds, is too expensive (rough estimate: tens of minutes of extra compute on top of the existing ~80-minute full training run per 5-seed batch, potentially comparable to the training cost itself). Default to a cheaper configuration (below) and let Jonathan/opencode dial it up if the signal looks promising and budget allows.

# Files To Inspect

- `run_nanda_benchmark.py` — the master runner; find `train_one_seed()`, its per-epoch loop, and its post-training "Dropout predictor" block (`compute_dropout_gap_multi_rate` call) to see exactly where to hook in.
- `src/predictors/dropout.py` — `compute_accuracy`, `compute_dropout_gap_multi_rate`. Add new functions here rather than modifying these two.
- `src/predictors/l2_norm.py` — read `resample_to_log_uniform_grid`, `compute_noise_floor`, and the trigger-detection functions purely as a **style reference** for how this project structures a trajectory-based signal, its log-epoch checkpoint spacing, and its noise-floor-relative thresholding. Do not import from this file for dropout logic.
- `configs/nanda_unified.yaml` — check for any existing `logging.*` keys that already define a checkpoint cadence or rate list, to avoid inventing a second, inconsistent convention.
- `results/README.md` and the two most recent entries in `context.md` (search for "Dropout predictor REOPENED" and the immediately preceding "formally closed" entry) for the full decision history and rationale behind this task.
- `results/nanda_unified/seed_0/dropout/` (and sibling seed dirs) — current output shape (`dropout_gap_final.json`, length-1 `.npy` files) that any new output should sit alongside without breaking.

# Requirements

1. **New checkpoint schedule.** Define a log-uniform-spaced set of epoch checkpoints from early training to the final epoch (e.g. ~20–25 points), reusing the log-spacing *idea* already used for L2 Norm (`resample_to_log_uniform_grid`) — do not literally resample already-collected data; instead pick actual epoch indices at training time (round each log-spaced point to the nearest integer epoch, dedupe, always include epoch 1 and the final epoch).
2. **New dropout-variance measurement function**, added to `src/predictors/dropout.py` (new function, not a modification of `compute_dropout_gap_multi_rate`): given a model, a test dataloader, a dropout rate, and `k` (default 100, matching the paper), run `k` stochastic forward passes (`model.train()`, dropout at that rate) over the full test set, compute test accuracy for each pass, and return the mean and variance of those `k` accuracy values, restoring `model.eval()` / dropout `p=0.0` before returning (same cleanup discipline as the existing function).
3. **Default rate for the variance signal: 0.5.** Justify in a code comment: it's the rate at which Salah & Yevick's own DRC shows the clearest pre/post-grok separation (their Figure 2: post-grok accuracy survives up to ≈0.5), and it keeps the primary signal to one rate for cost reasons. Still also run the existing 5-rate DRC-style sweep (mean accuracy only, `k=1` — i.e. today's existing `compute_dropout_gap_multi_rate`, no change needed there) at a **small number of the same checkpoints** (e.g. 5, evenly picked across the checkpoint list, mirroring the handful of example epochs Salah & Yevick plot) purely for a qualitative DRC-style plot — do not run the expensive `k=100` sweep at all 5 rates.
4. **Wire into `train_one_seed()` in `run_nanda_benchmark.py`:** when the current epoch is in the checkpoint set, call the new variance function (rate 0.5, k=100) and, at the smaller DRC-checkpoint subset, also the existing multi-rate function. Store results in-memory per seed (parallel arrays: checkpoint epoch, variance, mean-accuracy-at-that-rate; plus the DRC snapshots) and write them out at the end of the seed's training loop, the same way L2 Norm's per-epoch arrays are collected and saved.
5. **Save outputs** under `seed_{n}/dropout/`, alongside the existing files, without renaming or removing anything already there:
   - `dropout_variance_checkpoints.npy` — the checkpoint epoch indices used.
   - `dropout_variance_history.npy` — variance of test accuracy at rate 0.5, one value per checkpoint.
   - `dropout_variance_mean_acc_history.npy` — mean test accuracy at rate 0.5 across the k passes, one value per checkpoint (needed to later confirm the variance spike lines up with the actual accuracy transition, not some other artifact).
   - `dropout_drc_snapshots.json` — `{epoch: {rate: mean_accuracy}}` for the small DRC-checkpoint subset.
6. **Per-seed signal extraction**, mirroring the L2-Norm `l2_predictor_signals.json` pattern: find the checkpoint epoch at which `dropout_variance_history` is maximal (`variance_peak_epoch`), and its ratio to that seed's own grok epoch (`variance_peak_epoch / grok_epoch`), written to `seed_{n}/dropout/dropout_variance_signal.json`.
7. **Aggregate rollup:** extend `aggregate()` in `run_nanda_benchmark.py` to include, per seed, the new `variance_peak_epoch` and its ratio to `grok_epoch` in `aggregate.json`, in the same place the existing `l2_predictor` block sits (add a sibling `dropout_variance_predictor` block, do not touch `l2_predictor`).
8. **Plotting** (in `plot_nanda_results.py`, following its existing per-seed and comparison-plot structure): (a) a per-seed plot of `dropout_variance_history` vs. checkpoint epoch, with that seed's grok epoch marked as a vertical line (same convention as the existing grokking-curve plots); (b) a cross-seed comparison plot, `dropout_variance_vs_grok_scatter.png`, plotting `variance_peak_epoch` (y) against `grok_epoch` (x) for all 5 seeds on a log-log scale with the `y = x` reference line — directly modeled on the existing `04_predictor_vs_grok_scatter.png` for L2 Norm, so the two are visually comparable.

# Constraints

- Do not modify `src/predictors/l2_norm.py`, the existing `compute_dropout_gap_multi_rate` function's signature or behavior, or any already-saved single-head/archived results.
- Do not revive or import from `archive/single_head/` or `archive/src_p97_mainstream_old/`. Stay entirely on the current Nanda-Unified four-head substrate (`src/models/transformer_four_head.py`, `configs/nanda_unified.yaml`, modulus 113).
- Keep the default configuration cheap enough to run a full `--seeds 5 --epochs 40000` batch without materially changing the ~80-minute-per-batch wall-clock budget the L2-Norm+Dropout runner currently has — if a chosen `k`/checkpoint-count would clearly blow past that, reduce `k` or the checkpoint count and say so in the PR/commit notes rather than silently accepting a much slower run.
- Do not delete or rename any existing file under `results/nanda_unified/` or `results/test_smoke/`.
- Run the existing `--seeds 1 --epochs 100 --output_dir results/test_smoke` smoke test after the change (same as the original `run_nanda_benchmark.py` verification) before touching the real 40,000-epoch run, and delete the smoke directory afterward, matching existing project practice.

# Implementation Steps

1. Add the log-uniform checkpoint-schedule helper (can live in `run_nanda_benchmark.py` itself, next to where seeds/epochs are configured, since it is a scheduling concern, not a predictor-math concern) and the small DRC-checkpoint subset selection.
2. Add the new stochastic-variance function to `src/predictors/dropout.py`.
3. Wire both into `train_one_seed()`'s per-epoch loop, collecting the parallel history arrays.
4. Add the `.npy`/`.json` saves at the end of `train_one_seed()`, next to the existing dropout saves.
5. Add `dropout_variance_signal.json` computation (peak epoch + ratio).
6. Extend `aggregate()` to roll the new per-seed signal into `aggregate.json`.
7. Add the two new plots to `plot_nanda_results.py`.
8. Run `py_compile` on every changed file.
9. Run the `--seeds 1 --epochs 100 --output_dir results/test_smoke` smoke test; confirm the new files appear with sane shapes and that the checkpoint schedule actually produced checkpoints within a 100-epoch run (if the log-spacing formula would place zero checkpoints below epoch 100, fix the formula before moving on — don't discover this only at full scale). Delete `results/test_smoke/` afterward.

# Validation Steps

- `python -m py_compile run_nanda_benchmark.py src/predictors/dropout.py plot_nanda_results.py` — must pass with no errors.
- Smoke test (`--seeds 1 --epochs 100`) produces `dropout_variance_checkpoints.npy`, `dropout_variance_history.npy`, `dropout_variance_mean_acc_history.npy`, `dropout_drc_snapshots.json`, and `dropout_variance_signal.json` under `results/test_smoke/seed_0/dropout/`, all with consistent lengths (checkpoints array length == variance history length == mean-acc history length).
- Confirm the new variance-vs-epoch values are not NaN/degenerate over the 100-epoch smoke run (the model won't grok in 100 epochs, so variance should stay low/flat throughout — this at least confirms the plumbing works, not the hypothesis).
- Manually inspect one full-scale seed's `dropout_variance_history.npy` after the real run (not part of this implementation pass, but note it in the completion summary) to eyeball whether a peak appears anywhere near that seed's `grok_epoch` before doing the full 5-seed run and the aggregate/plot step.

# Acceptance Criteria

- All 5 new/changed files (`run_nanda_benchmark.py`, `src/predictors/dropout.py`, `plot_nanda_results.py`, plus the two new per-seed output files) compile and run cleanly through the existing smoke-test path with no changes to any other predictor's behavior or output files.
- A full `--seeds 5 --epochs 40000` run produces, per seed, a `dropout_variance_signal.json` with a `variance_peak_epoch` and a `variance_peak_epoch / grok_epoch` ratio, and `aggregate.json` contains a `dropout_variance_predictor` block with all 5 seeds' signals — in a form directly comparable to how `l2_predictor` signals were used to judge L2 Norm's Criterion 2 (tight, consistent, proportional gap across seeds whose grok epochs differ by 3x+).
- `dropout_variance_vs_grok_scatter.png` exists and is legible: if the hypothesis is real, points should sit near the `y = x` line rather than in a flat horizontal band (contrast with `04_predictor_vs_grok_scatter.png`, where L2 Norm's points sit in a flat band and were judged to fail).
- No existing predictor's saved data, L2 Norm's closed verdict, or any single-head/archived result is altered.
