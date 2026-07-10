# Graph Report - grokking-benchmark  (2026-07-10)

## Corpus Check
- 10 files · ~211,949 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 24 nodes · 27 edges · 5 communities (2 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `04dcf313`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]

## God Nodes (most connected - your core abstractions)
1. `ModularArithmeticDataset` - 6 edges
2. `ModularArithmeticDataset` - 5 edges
3. `Python Project Compilation` - 4 edges
4. `Transformer` - 3 edges
5. `generate_pairs()` - 3 edges
6. `get_dataloaders()` - 2 edges
7. `get_dataloaders()` - 2 edges
8. `data/modular_arithmetic.py` - 1 edges
9. `models/transformer.py` - 1 edges
10. `train.py` - 1 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (5 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.40
Nodes (3): get_dataloaders(), ModularArithmeticDataset, Dataset

### Community 4 - "Community 4"
Cohesion: 0.40
Nodes (4): data/modular_arithmetic.py, models/transformer.py, Python Project Compilation, train.py

## Knowledge Gaps
- **3 isolated node(s):** `data/modular_arithmetic.py`, `models/transformer.py`, `train.py`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModularArithmeticDataset` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
- **Why does `ModularArithmeticDataset` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **What connects `data/modular_arithmetic.py`, `models/transformer.py`, `train.py` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._