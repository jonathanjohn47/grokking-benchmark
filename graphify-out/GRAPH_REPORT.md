# Graph Report - grokking-benchmark  (2026-09-01)

## Corpus Check
- 41 files · ~398,296 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 163 nodes · 181 edges · 22 communities (18 shown, 4 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b1ecdc70`
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

## God Nodes (most connected - your core abstractions)
1. `PredictorMeasurements` - 11 edges
2. `L2 Norm Predictor — Easy Notes for Revision and Viva` - 11 edges
3. `BenchmarkAnalyzer` - 10 edges
4. `Directory Structure` - 7 edges
5. `Python Project Compilation` - 7 edges
6. `ModularArithmeticDataset` - 7 edges
7. `main()` - 5 edges
8. `extract_metrics()` - 5 edges
9. `ModularArithmeticDataset` - 5 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `Path`  [INFERRED]
  tools/collage_images.py → tools/compile_python_files.py
- `main()` --calls--> `Path`  [INFERRED]
  tools/md_to_image.py → tools/compile_python_files.py
- `compile_python_files_to_pdf()` --calls--> `Path`  [INFERRED]
  tools/compile_python_files_to_pdf.py → tools/compile_python_files.py
- `gather_python_files()` --calls--> `Path`  [INFERRED]
  tools/compile_python_files_to_pdf.py → tools/compile_python_files.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (22 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (9): get_dataloaders(), ModularArithmeticDataset, generate_pairs(), get_dataloaders(), ModularArithmeticDataset, Dataset, L2 Norm Predictor  This module implements the L2 norm predictor., migrate_legacy_flat_run() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.23
Nodes (12): create_report(), extract_metrics(), find_grok_epoch(), find_trough_epoch(), load_run_data(), main(), Professional L2 Norm Analysis Report for Four-Head Transformer -- CORRECTED VERS, Create professional PDF report. (+4 more)

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (11): 10. Likely Viva Questions and Model Answers, 1. What This Report Is About, in One Line, 2. Objective — In Simple Words, 3. Experimental Setup — Table, 4. Five Detection Strategies Tried, in Order, 5. Three Formal Criteria for Judging Any Trigger, 6. Cross-Run Evidence — What the Numbers Actually Mean, 7. Side Finding — Test Accuracy Plateau in Run 3 (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.40
Nodes (4): data/modular_arithmetic.py, models/transformer.py, Python Project Compilation, train.py

### Community 6 - "Community 6"
Cohesion: 0.20
Nodes (9): Core Project Files, Directory Structure, Documentation, Experiments & Results, Grokking Predictors Benchmark, Other, Quick Reference, Source Code (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.36
Nodes (6): Path, main(), should_skip(), compile_python_files_to_pdf(), gather_python_files(), wrap_text()

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (7): data\modular_arithmetic.py, models\transformer.py, plot_results.py, predictors\dropout.py, predictors\l2_norm.py, Python Project Compilation, train.py

### Community 9 - "Community 9"
Cohesion: 0.47
Nodes (3): main(), render_page(), wrap_markdown()

### Community 10 - "Community 10"
Cohesion: 0.40
Nodes (4): data/modular_arithmetic.py, models/transformer.py, Python Project Compilation, train.py

### Community 11 - "Community 11"
Cohesion: 0.70
Nodes (4): get_images(), horizontal_collage(), main(), vertical_collage()

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (11): PredictorMeasurements, Unified measurement collection for L2 Norm and Dropout predictors. Ensures consi, Collects and saves all measurements for both predictors., Generate standalone Dropout visualization graphs., Generate combined PDF report with all measurements., Create subdirectories for each predictor., Apply simple moving average smoothing., Save core training metrics. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.15
Nodes (9): BenchmarkAnalyzer, Compare L2 Norm curves., Compare Dropout gap curves., Generate PDF report with run consistency metrics., Load single-head baseline results., Execute full analysis., Load all four-head runs., Find grokking epoch (test acc > 90%). (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.36
Nodes (7): cleanup_results(), main(), Remove existing results to start fresh., Run single-head baseline training., Run one four-head training session., run_four_head(), run_single_head()

## Knowledge Gaps
- **30 isolated node(s):** `Core Project Files`, `Source Code`, `Experiments & Results`, `Tools & Utilities`, `Documentation` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Path` connect `Community 7` to `Community 9`, `Community 11`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `Load single-head baseline results.`, `Load all four-head runs.`, `Find grokking epoch (test acc > 90%).` to the rest of the system?**
  _60 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.11462450592885376 - nodes in this community are weakly interconnected._
- **Should `Community 19` be split into smaller, more focused modules?**
  _Cohesion score 0.10476190476190476 - nodes in this community are weakly interconnected._