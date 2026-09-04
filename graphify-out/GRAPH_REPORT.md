# Graph Report - grokking-benchmark  (2026-09-04)

## Corpus Check
- 55 files · ~465,445 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 409 nodes · 475 edges · 38 communities (34 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `029b7390`
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
- [[_COMMUNITY_Community 33|Community 33]]

## God Nodes (most connected - your core abstractions)
1. `Indian English Communication Skill` - 16 edges
2. `PredictorMeasurements` - 11 edges
3. `BenchmarkAnalyzer` - 11 edges
4. `L2 Norm Predictor — Easy Notes for Revision and Viva` - 11 edges
5. `BenchmarkAnalyzer` - 10 edges
6. `Literature` - 10 edges
7. `PredictorMeasurements` - 9 edges
8. `Results Directory Structure` - 9 edges
9. `B. Combined Glossary` - 8 edges
10. `ModularArithmeticDataset` - 7 edges

## Surprising Connections (you probably didn't know these)
- `_analysis_outputs_present()` --calls--> `Path`  [INFERRED]
  run_full_benchmark.py → tools/compile_python_files.py
- `main()` --calls--> `Path`  [INFERRED]
  run_full_benchmark.py → tools/compile_python_files.py
- `main()` --calls--> `Path`  [INFERRED]
  tools/collage_images.py → tools/compile_python_files.py
- `main()` --calls--> `Path`  [INFERRED]
  tools/md_to_image.py → tools/compile_python_files.py
- `compile_python_files_to_pdf()` --calls--> `Path`  [INFERRED]
  tools/compile_python_files_to_pdf.py → tools/compile_python_files.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (38 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.16
Nodes (7): generate_pairs(), get_dataloaders(), ModularArithmeticDataset, Dataset, generate_pairs(), get_dataloaders(), ModularArithmeticDataset

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
Cohesion: 0.70
Nodes (4): get_images(), horizontal_collage(), main(), vertical_collage()

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
Cohesion: 0.11
Nodes (7): TransformerFourHead, Transformer, compute_accuracy(), compute_dropout_gap_multi_rate(), # NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate, migrate_legacy_flat_run(), Earlier versions of this script saved directly into results/four_head/     inste

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
Cohesion: 0.07
Nodes (38): apply_moving_average(), compute_acceleration(), compute_fast_slow_moving_averages(), compute_ma_of_slow_ma(), compute_noise_floor(), compute_per_module_sum_of_squared_weights(), compute_sum_of_squared_weights(), detect_inflection() (+30 more)

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

## Knowledge Gaps
- **103 isolated node(s):** `What this is`, `The four deliberate differences from `src/``, `Why `p = 113`, `betas = (0.9, 0.98)`, and the small init`, `How to run`, `Outputs` (+98 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Path` connect `Community 21` to `Community 5`, `Community 7`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **What connects `L2 Norm + Dropout analysis report for the four-head transformer.  Standalone too`, `Load one run from the per-predictor subdir layout.`, `First 1-indexed epoch where test accuracy exceeds threshold.` to the rest of the system?**
  _180 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.09420289855072464 - nodes in this community are weakly interconnected._
- **Should `Community 11` be split into smaller, more focused modules?**
  _Cohesion score 0.10822510822510822 - nodes in this community are weakly interconnected._
- **Should `Community 19` be split into smaller, more focused modules?**
  _Cohesion score 0.10476190476190476 - nodes in this community are weakly interconnected._
- **Should `Community 21` be split into smaller, more focused modules?**
  _Cohesion score 0.0915915915915916 - nodes in this community are weakly interconnected._
- **Should `Community 22` be split into smaller, more focused modules?**
  _Cohesion score 0.06755260243632337 - nodes in this community are weakly interconnected._