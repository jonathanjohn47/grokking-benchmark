# Graph Report - grokking-benchmark  (2026-07-01)

## Corpus Check
- 4 files · ~159 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6 nodes · 7 edges · 2 communities (1 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `774ccf4c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 1|Community 1]]

## God Nodes (most connected - your core abstractions)
1. `ModularArithmeticDataset` - 2 edges
2. `generate_pairs()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (2 total, 1 thin omitted)

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModularArithmeticDataset` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `generate_pairs()` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._