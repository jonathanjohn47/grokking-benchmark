# Graph Report - grokking-benchmark  (2026-09-25)

## Corpus Check
- 149 files · ~982,884 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 548 nodes · 658 edges · 43 communities (38 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f0659392`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]

## God Nodes (most connected - your core abstractions)
1. `PredictorMeasurements` - 19 edges
2. `Indian English Communication Skill` - 16 edges
3. `train_one_seed()` - 13 edges
4. `train_one_seed()` - 12 edges
5. `PredictorMeasurements` - 11 edges
6. `L2 Norm Predictor — Easy Notes for Revision and Viva` - 11 edges
7. `main()` - 10 edges
8. `Literature` - 10 edges
9. `main()` - 10 edges
10. `BenchmarkAnalyzer` - 10 edges

## Surprising Connections (you probably didn't know these)
- `recompute_from_checkpoints()` --calls--> `PredictorMeasurements`  [INFERRED]
  run_nanda_benchmark.py → 06_Code/src/unified_measurements.py
- `train_one_seed()` --calls--> `PredictorMeasurements`  [INFERRED]
  run_nanda_benchmark.py → 06_Code/src/unified_measurements.py
- `Apply moving average smoothing to reduce noise.     window_size: number of epoch` --rationale_for--> `apply_moving_average()`  [EXTRACTED]
  src/predictors/l2_norm.py → archive/src_p97_mainstream_old/predictors/l2_norm.py
- `Compute two moving averages on a LOG-EPOCH-UNIFORM grid, so the window     cover` --rationale_for--> `compute_fast_slow_moving_averages()`  [EXTRACTED]
  src/predictors/l2_norm.py → archive/src_p97_mainstream_old/predictors/l2_norm.py
- `train_one_seed()` --calls--> `PredictorMeasurements`  [INFERRED]
  06_Code/scripts/run_nanda_benchmark.py → 06_Code/src/unified_measurements.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (43 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (34): Transformer, compute_accuracy(), compute_dropout_gap_multi_rate(), compute_dropout_variance(), # NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate, # NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate, Dropout-variance predictor signal (Salah & Yevick, arXiv:2507.11645):     at a f, aggregate() (+26 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (10): generate_pairs(), get_dataloaders(), ModularArithmeticDataset, Dataset, Transformer, TransformerFourHead, migrate_legacy_flat_run(), Earlier versions of this script saved directly into results/four_head/     inste (+2 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (11): 10. Likely Viva Questions and Model Answers, 1. What This Report Is About, in One Line, 2. Objective — In Simple Words, 3. Experimental Setup — Table, 4. Five Detection Strategies Tried, in Order, 5. Three Formal Criteria for Judging Any Trigger, 6. Cross-Run Evidence — What the Numbers Actually Mean, 7. Side Finding — Test Accuracy Plateau in Run 3 (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (5): Archived — single-head experiment, How to bring it back, Nothing was deleted, Open item (still postponed), What is here

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (20): apply_moving_average(), compute_acceleration(), compute_fast_slow_moving_averages(), compute_ma_of_slow_ma(), compute_noise_floor(), detect_inflection(), detect_ma_crossover(), detect_ma_of_ma_trigger() (+12 more)

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (16): build_events(), fraction_epochs(), l2_decline_epoch(), load_seed(), main(), plot_curves(), plot_leads(), plot_scatter() (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (9): Core Project Files, Directory Structure, Documentation, Experiments & Results, Grokking Predictors Benchmark, Other, Quick Reference, Source Code (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (7): data\modular_arithmetic.py, models\transformer.py, plot_results.py, predictors\dropout.py, predictors\l2_norm.py, Python Project Compilation, train.py

### Community 9 - "Community 9"
Cohesion: 0.07
Nodes (27): 1.1 What is grokking, 1.2 What a grokking predictor is supposed to do, 1. Prerequisites Recap, 2.1 What a kernel is, 2.2 Why a trained network's hidden layer defines a kernel, 2.3 The representation matrix and the Gram matrix, 2. Kernel Regression Basics, 3.1 Eigenvalues and eigenvectors of $K$ (+19 more)

### Community 10 - "Community 10"
Cohesion: 0.40
Nodes (4): data/modular_arithmetic.py, models/transformer.py, Python Project Compilation, train.py

### Community 11 - "Community 11"
Cohesion: 0.21
Nodes (11): compute_htsr_for_model(), compute_htsr_metrics_for_checkpoint(), fit_powerlaw_alpha(), _layer_eigenvalues(), HTSR Alpha predictor (Predictor 5 of 9) — Heavy-Tailed Self-Regularization.  DEF, HTSR Alpha metrics for ONE frozen model checkpoint.      Returns a dict:, Thin wrapper kept parallel to compute_spectral_for_model and     compute_age_for, Eigenvalues of X = W^T W / N for one weight matrix (float64, CPU),     sorted as (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.09
Nodes (40): aggregate(), apply_config_constants(), _checkpoint_predictor_age(), _checkpoint_predictor_dropout_variance(), _checkpoint_predictor_spectral(), checkpoints_dir_for(), dips_key(), dropout_variance_checkpoint_schedule() (+32 more)

### Community 19 - "Community 19"
Cohesion: 0.06
Nodes (25): PredictorMeasurements, Unified measurement collection for L2 Norm and Dropout predictors. Ensures consi, Save all Dropout measurements — full multi-rate sweep only.          There is no, Save all Dropout measurements — full multi-rate sweep only.          There is no, Save all Spectral (Predictor 3) measurements — Canatar et al.         2021 task-, Save all Spectral (Predictor 3) measurements — Canatar et al.         2021 task-, Collects and saves all measurements for both predictors., Save all AGE (Predictor 4) measurements — Papyan et al. 2020 NC1         variabi (+17 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (8): discover_run_dirs(), migrate_legacy_flat_run(), plot_comparison(), plot_single_run(), Recreates this run's own 8 plots, saved inside results/four_head/run_<N>/ itself, Builds 4 plots overlaying every discovered run together, saved     directly in r, Same migration as train_four_head.py — kept here too so this     script can be r, Finds every run_<N> folder inside base_dir, sorted by run number     (not alphab

### Community 21 - "Community 21"
Cohesion: 0.20
Nodes (10): Path, get_images(), horizontal_collage(), main(), vertical_collage(), main(), should_skip(), main() (+2 more)

### Community 22 - "Community 22"
Cohesion: 0.05
Nodes (42): apply_moving_average(), compute_acceleration(), compute_fast_slow_moving_averages(), compute_ma_of_slow_ma(), compute_noise_floor(), compute_per_module_sum_of_squared_weights(), compute_sum_of_squared_weights(), detect_inflection() (+34 more)

### Community 23 - "Community 23"
Cohesion: 0.06
Nodes (33): A. Master Table, B. Combined Glossary, Complete Inventory --- 21 distinct papers/documents, Exhaustive Raw Literature Extraction Report, Generalization and optimization, Grokking and mechanistic interpretability, Kernel and spectral concepts, Merged PDF: `ilovepdf_merged.pdf` (+25 more)

### Community 24 - "Community 24"
Cohesion: 0.18
Nodes (10): 1. Grokking — direct, 2. Heavy-tailed self-regularisation / weight-matrix spectra, 3. Neural collapse (terminal phase of training), 4. Generalisation dynamics / double descent / interpolation, 5. Kernel / random-features / mean-field theory, 6. Scaling laws / representations, 7. Robustness / margins / shortcut learning, 8. Other (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.20
Nodes (10): _analysis_outputs_present(), BenchmarkAnalyzer, main(), prepare_runs_dir(), Run one training session, then verify it really finished., All run_<N> dirs under runs/, sorted by number., Make runs/ safe to resume into, and return how many runs are already     fully f, _run_dirs() (+2 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (16): 10. Do not overuse enthusiasm, 11. Professional tone, 12. When rewriting user text, 13. Final quality check, 1. Non-negotiable rule, 2. What Indian English means here, 3. Do not confuse Indian English with simple English, 4. Preferred Indian teaching voice (+8 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (9): PredictorMeasurements, Measurement collection for the L2 Norm predictor only.  This is a trimmed copy o, Generate combined PDF report: training + L2 Norm pages only., Apply simple moving average smoothing., Collects and saves all measurements for the L2 Norm predictor., Create subdirectories for the predictor outputs., Save core training metrics., Save all L2 Norm measurements. (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.12
Nodes (16): `comparisons/`, `dropout/`, Experiments, Four-Head Results, How to Use This Structure, Key Findings, `l2_norm/`, Last Updated (+8 more)

### Community 29 - "Community 29"
Cohesion: 0.36
Nodes (3): generate_pairs(), get_dataloaders(), ModularArithmeticDataset

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (6): How to run, nanda_l2_p113 — L2-Norm predictor on (a + b) mod 113, Outputs, The four deliberate differences from `src/`, What this is, Why `p = 113`, `betas = (0.9, 0.98)`, and the small init

### Community 31 - "Community 31"
Cohesion: 0.10
Nodes (11): PredictorMeasurements, Unified measurement collection for L2 Norm and Dropout predictors. Ensures consi, Collects and saves all measurements for both predictors., Generate standalone Dropout visualization graphs., Generate combined PDF report with all measurements., Create subdirectories for each predictor., Apply simple moving average smoothing., Save core training metrics. (+3 more)

### Community 43 - "Community 43"
Cohesion: 0.40
Nodes (4): archive/src_p97_mainstream_old/ — the pre-Nanda-Unified `src/` snapshot, Not a runnable tree, What the old code did (differs from the current `src/`), Why it is kept

### Community 46 - "Community 46"
Cohesion: 0.27
Nodes (9): _collect_representations_and_labels(), compute_spectral_for_model(), compute_spectral_metrics_for_checkpoint(), Spectral predictor (Predictor 3 of 9) — FAITHFUL Canatar et al. 2021.  Paper: Ab, Canatar et al. 2021 task-model-alignment metrics for ONE frozen     model checkp, Thin wrapper kept for run_nanda_benchmark.py's import. One frozen     checkpoint, Run the model over the full training split and capture, per example,     the hid, _checkpoint_predictor_spectral() (+1 more)

### Community 47 - "Community 47"
Cohesion: 0.24
Nodes (9): _collect_representations_and_labels(), compute_age_for_model(), compute_age_metrics_for_checkpoint(), AGE predictor (Predictor 4 of 9) — Adaptive Grokking Epoch via Neural Collapse., Papyan et al. 2020 NC1 variability-collapse metrics for ONE frozen     model che, Thin wrapper kept parallel to     src/predictors/spectral.py::compute_spectral_f, Run the model over the full training split and capture, per example,     the hid, _checkpoint_predictor_age() (+1 more)

## Knowledge Gaps
- **127 isolated node(s):** `1.1 What is grokking`, `1.2 What a grokking predictor is supposed to do`, `2.1 What a kernel is`, `2.2 Why a trained network's hidden layer defines a kernel`, `2.3 The representation matrix and the Gram matrix` (+122 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PredictorMeasurements` connect `Community 19` to `Community 0`, `Community 14`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `train_one_seed()` connect `Community 0` to `Community 19`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `PredictorMeasurements` (e.g. with `recompute_from_checkpoints()` and `train_one_seed()`) actually correct?**
  _`PredictorMeasurements` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Epoch indices (0-based, matching the training loop's `epoch`) at which     a mod`, `Subsample the evaluation grid from the SAVED checkpoint epochs.     Keeps saved`, `LEGACY — for Archive comparison only, not used in unified benchmark     (24 log-` to the rest of the system?**
  _260 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08717948717948718 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07661290322580645 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.09420289855072464 - nodes in this community are weakly interconnected._