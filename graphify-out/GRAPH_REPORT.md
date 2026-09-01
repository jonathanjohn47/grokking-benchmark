# Graph Report - grokking-benchmark  (2026-09-02)

## Corpus Check
- 42 files · ~377,111 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 212 nodes · 259 edges · 24 communities (21 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `35fbe0b7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
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

## God Nodes (most connected - your core abstractions)
1. `BenchmarkAnalyzer` - 11 edges
2. `PredictorMeasurements` - 11 edges
3. `L2 Norm Predictor — Easy Notes for Revision and Viva` - 11 edges
4. `Directory Structure` - 7 edges
5. `Python Project Compilation` - 7 edges
6. `ModularArithmeticDataset` - 7 edges
7. `run_four_head()` - 6 edges
8. `main()` - 6 edges
9. `main()` - 6 edges
10. `prepare_four_head_dir()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `Path`  [INFERRED]
  tools/collage_images.py → tools/compile_python_files.py
- `compile_python_files_to_pdf()` --calls--> `Path`  [INFERRED]
  tools/compile_python_files_to_pdf.py → tools/compile_python_files.py
- `gather_python_files()` --calls--> `Path`  [INFERRED]
  tools/compile_python_files_to_pdf.py → tools/compile_python_files.py
- `main()` --calls--> `Path`  [INFERRED]
  tools/md_to_image.py → tools/compile_python_files.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (24 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (8): get_dataloaders(), ModularArithmeticDataset, generate_pairs(), get_dataloaders(), ModularArithmeticDataset, Dataset, Transformer, L2 Norm Predictor  This module implements the L2 norm predictor.

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
Cohesion: 0.40
Nodes (4): data/modular_arithmetic.py, models/transformer.py, Python Project Compilation, train.py

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (9): Core Project Files, Directory Structure, Documentation, Experiments & Results, Grokking Predictors Benchmark, Other, Quick Reference, Source Code (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.20
Nodes (9): Path, main(), should_skip(), compile_python_files_to_pdf(), gather_python_files(), wrap_text(), main(), render_page() (+1 more)

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
Cohesion: 0.29
Nodes (5): compute_accuracy(), compute_dropout_gap_multi_rate(), # NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate, migrate_legacy_flat_run(), Earlier versions of this script saved directly into results/four_head/     inste

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (11): PredictorMeasurements, Unified measurement collection for L2 Norm and Dropout predictors. Ensures consi, Collects and saves all measurements for both predictors., Generate standalone Dropout visualization graphs., Generate combined PDF report with all measurements., Create subdirectories for each predictor., Apply simple moving average smoothing., Save core training metrics. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (8): discover_run_dirs(), migrate_legacy_flat_run(), plot_comparison(), plot_single_run(), Recreates this run's own 8 plots, saved inside results/four_head/run_<N>/ itself, Builds 4 plots overlaying every discovered run together, saved     directly in r, Same migration as train_four_head.py — kept here too so this     script can be r, Finds every run_<N> folder inside base_dir, sorted by run number     (not alphab

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (18): _analysis_outputs_present(), BenchmarkAnalyzer, main(), prepare_four_head_dir(), Run one four-head training session, then verify it really finished., Load all four-head runs., Ordered keys of the four-head runs that actually loaded., Find grokking epoch (test acc > 90%). (+10 more)

### Community 22 - "Community 22"
Cohesion: 0.26
Nodes (14): build_data_pdf(), build_plots_pdf(), _comparison_pages(), _dump_table(), grok_epoch(), _grok_vline(), load_run(), main() (+6 more)

### Community 23 - "Community 23"
Cohesion: 0.70
Nodes (4): get_images(), horizontal_collage(), main(), vertical_collage()

## Knowledge Gaps
- **34 isolated node(s):** `What is here`, `Nothing was deleted`, `How to bring it back`, `Open item (still postponed)`, `Core Project Files` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `All run_<N> dirs under runs/four_head/, sorted by number.`, `Make runs/four_head/ safe to resume into, and return how many runs     are alrea`, `Run one four-head training session, then verify it really finished.` to the rest of the system?**
  _73 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.10666666666666667 - nodes in this community are weakly interconnected._
- **Should `Community 19` be split into smaller, more focused modules?**
  _Cohesion score 0.10476190476190476 - nodes in this community are weakly interconnected._
- **Should `Community 21` be split into smaller, more focused modules?**
  _Cohesion score 0.11904761904761904 - nodes in this community are weakly interconnected._