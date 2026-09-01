# Graph Report - grokking-benchmark  (2026-09-01)

## Corpus Check
- 38 files · ~396,459 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 111 nodes · 118 edges · 19 communities (15 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `706d11c8`
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

## God Nodes (most connected - your core abstractions)
1. `L2 Norm Predictor — Easy Notes for Revision and Viva` - 11 edges
2. `Directory Structure` - 7 edges
3. `Python Project Compilation` - 7 edges
4. `ModularArithmeticDataset` - 7 edges
5. `main()` - 5 edges
6. `extract_metrics()` - 5 edges
7. `ModularArithmeticDataset` - 5 edges
8. `Python Project Compilation` - 4 edges
9. `compile_python_files_to_pdf()` - 4 edges
10. `load_run_data()` - 4 edges

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

## Communities (19 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (6): get_dataloaders(), ModularArithmeticDataset, generate_pairs(), get_dataloaders(), ModularArithmeticDataset, Dataset

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

## Knowledge Gaps
- **30 isolated node(s):** `Core Project Files`, `Source Code`, `Experiments & Results`, `Tools & Utilities`, `Documentation` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Path` connect `Community 7` to `Community 9`, `Community 11`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 9` to `Community 7`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Path` (e.g. with `main()` and `compile_python_files_to_pdf()`) actually correct?**
  _`Path` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Core Project Files`, `Source Code`, `Experiments & Results` to the rest of the system?**
  _38 weakly-connected nodes found - possible documentation gaps or missing edges._