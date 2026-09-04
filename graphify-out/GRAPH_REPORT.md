# Graph Report - grokking-benchmark  (2026-09-05)

## Corpus Check
- 72 files · ~483,572 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 571 nodes · 743 edges · 47 communities (44 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8e62e63c`
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
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 26 edges
2. `new_fig()` - 19 edges
3. `Indian English Communication Skill` - 16 edges
4. `PredictorMeasurements` - 11 edges
5. `PredictorMeasurements` - 11 edges
6. `BenchmarkAnalyzer` - 11 edges
7. `L2 Norm Predictor — Easy Notes for Revision and Viva` - 11 edges
8. `train_one_seed()` - 10 edges
9. `BenchmarkAnalyzer` - 10 edges
10. `Literature` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Apply moving average smoothing to reduce noise.     window_size: number of epoch` --rationale_for--> `apply_moving_average()`  [EXTRACTED]
  src/predictors/l2_norm.py → archive/src_p97_mainstream_old/predictors/l2_norm.py
- `Compute two moving averages on a LOG-EPOCH-UNIFORM grid, so the window     cover` --rationale_for--> `compute_fast_slow_moving_averages()`  [EXTRACTED]
  src/predictors/l2_norm.py → archive/src_p97_mainstream_old/predictors/l2_norm.py
- `Detect where fast MA crosses above slow MA, searching along the     log-uniform` --rationale_for--> `detect_ma_crossover()`  [EXTRACTED]
  src/predictors/l2_norm.py → archive/src_p97_mainstream_old/predictors/l2_norm.py
- `load_run()` --calls--> `detect_ma_crossover()`  [EXTRACTED]
  src/generate_master_report.py → archive/src_p97_mainstream_old/predictors/l2_norm.py
- `Estimate the normal noise level of the MA-of-MA difference from the quiet     ea` --rationale_for--> `compute_noise_floor()`  [EXTRACTED]
  src/predictors/l2_norm.py → archive/src_p97_mainstream_old/predictors/l2_norm.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (47 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (23): Transformer, compute_accuracy(), compute_dropout_gap_multi_rate(), compute_dropout_variance(), # NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate, # NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate, Dropout-variance predictor signal (Salah & Yevick, arXiv:2507.11645):     at a f, aggregate() (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.23
Nodes (12): create_report(), extract_metrics(), find_grok_epoch(), find_trough_epoch(), load_run_data(), main(), Professional L2 Norm Analysis Report for Four-Head Transformer -- CORRECTED VERS, Create professional PDF report. (+4 more)

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
Cohesion: 0.12
Nodes (36): aggregate_summary_lines(), Collector, discover_seeds(), fmt_epoch(), load_aggregate(), load_seed(), main(), mark_grok() (+28 more)

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (9): Core Project Files, Directory Structure, Documentation, Experiments & Results, Grokking Predictors Benchmark, Other, Quick Reference, Source Code (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.47
Nodes (3): main(), render_page(), wrap_markdown()

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (7): data\modular_arithmetic.py, models\transformer.py, plot_results.py, predictors\dropout.py, predictors\l2_norm.py, Python Project Compilation, train.py

### Community 9 - "Community 9"
Cohesion: 0.31
Nodes (10): create_report(), discover_run_numbers(), extract_metrics(), find_grok_epoch(), load_run_data(), main(), L2 Norm + Dropout analysis report for the four-head transformer.  Standalone too, Load one run from the per-predictor subdir layout. (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.40
Nodes (4): data/modular_arithmetic.py, models/transformer.py, Python Project Compilation, train.py

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (10): generate_pairs(), get_dataloaders(), ModularArithmeticDataset, Dataset, Transformer, TransformerFourHead, migrate_legacy_flat_run(), Earlier versions of this script saved directly into results/four_head/     inste (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (11): PredictorMeasurements, Unified measurement collection for L2 Norm and Dropout predictors. Ensures consi, Generate standalone L2 Norm visualization graphs., Collects and saves all measurements for both predictors., Generate standalone Dropout visualization graphs., Generate combined PDF report with all measurements., Create subdirectories for each predictor., Apply simple moving average smoothing. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (8): discover_run_dirs(), migrate_legacy_flat_run(), plot_comparison(), plot_single_run(), Recreates this run's own 8 plots, saved inside results/four_head/run_<N>/ itself, Builds 4 plots overlaying every discovered run together, saved     directly in r, Same migration as train_four_head.py — kept here too so this     script can be r, Finds every run_<N> folder inside base_dir, sorted by run number     (not alphab

### Community 21 - "Community 21"
Cohesion: 0.09
Nodes (24): Path, _analysis_outputs_present(), BenchmarkAnalyzer, main(), prepare_four_head_dir(), Run one four-head training session, then verify it really finished., Load all four-head runs., Ordered keys of the four-head runs that actually loaded. (+16 more)

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
Cohesion: 0.23
Nodes (11): compute_l2_statistics(), create_pdf_report(), find_grok_epoch(), load_run_data(), main(), L2 Norm Behavior Analysis for Four-Head Transformer Detailed technical report on, Create a comprehensive PDF report., Load all relevant data from a specific run. (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (6): How to run, nanda_l2_p113 — L2-Norm predictor on (a + b) mod 113, Outputs, The four deliberate differences from `src/`, What this is, Why `p = 113`, `betas = (0.9, 0.98)`, and the small init

### Community 31 - "Community 31"
Cohesion: 0.10
Nodes (11): PredictorMeasurements, Unified measurement collection for L2 Norm and Dropout predictors. Ensures consi, Collects and saves all measurements for both predictors., Generate standalone Dropout visualization graphs., Generate combined PDF report with all measurements., Create subdirectories for each predictor., Apply simple moving average smoothing., Save core training metrics. (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.26
Nodes (14): build_data_pdf(), build_plots_pdf(), _comparison_pages(), _dump_table(), grok_epoch(), _grok_vline(), load_run(), main() (+6 more)

### Community 33 - "Community 33"
Cohesion: 0.26
Nodes (14): build_data_pdf(), build_plots_pdf(), _comparison_pages(), _dump_table(), grok_epoch(), _grok_vline(), load_run(), main() (+6 more)

### Community 34 - "Community 34"
Cohesion: 0.23
Nodes (11): compute_l2_statistics(), create_pdf_report(), find_grok_epoch(), load_run_data(), main(), L2 Norm Behavior Analysis for Four-Head Transformer Detailed technical report on, Create a comprehensive PDF report., Load all relevant data from a specific run. (+3 more)

### Community 41 - "Community 41"
Cohesion: 0.31
Nodes (10): create_report(), discover_run_numbers(), extract_metrics(), find_grok_epoch(), load_run_data(), main(), L2 Norm + Dropout analysis report for the four-head transformer.  Standalone too, Load one run from the per-predictor subdir layout. (+2 more)

### Community 42 - "Community 42"
Cohesion: 0.36
Nodes (3): generate_pairs(), get_dataloaders(), ModularArithmeticDataset

### Community 43 - "Community 43"
Cohesion: 0.40
Nodes (4): archive/src_p97_mainstream_old/ — the pre-Nanda-Unified `src/` snapshot, Not a runnable tree, What the old code did (differs from the current `src/`), Why it is kept

### Community 45 - "Community 45"
Cohesion: 0.26
Nodes (11): add_results_section(), add_text_lines(), compile_py_to_pdf(), _escape_line(), _fmt_epoch(), Find all Python (.py) files in the current project directory,     excluding virt, Escape a plain-text line for ReportLab's Paragraph XML/HTML parser,     convert, Append plain-text lines to the story as Paragraphs (or blank     Spacers), using (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.70
Nodes (4): get_images(), horizontal_collage(), main(), vertical_collage()

## Knowledge Gaps
- **106 isolated node(s):** `Why it is kept`, `What the old code did (differs from the current `src/`)`, `Not a runnable tree`, `What this is`, `The four deliberate differences from `src/`` (+101 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModularArithmeticDataset` connect `Community 42` to `Community 11`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **What connects `Return sorted seed indices for every seed_{n}/summary.json found.`, `Load one seed's summary.json + raw per-epoch npy histories.`, `Prefer aggregate.json; fall back to computing the same summary     from the load` to the rest of the system?**
  _238 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.10837438423645321 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.09420289855072464 - nodes in this community are weakly interconnected._
- **Should `Community 5` be split into smaller, more focused modules?**
  _Cohesion score 0.11794871794871795 - nodes in this community are weakly interconnected._
- **Should `Community 11` be split into smaller, more focused modules?**
  _Cohesion score 0.07661290322580645 - nodes in this community are weakly interconnected._
- **Should `Community 19` be split into smaller, more focused modules?**
  _Cohesion score 0.10476190476190476 - nodes in this community are weakly interconnected._