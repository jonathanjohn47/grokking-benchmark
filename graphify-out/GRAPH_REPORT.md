# Graph Report - grokking-benchmark  (2026-07-06)

## Corpus Check
- 7 files · ~270 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 18 nodes · 21 edges · 4 communities (1 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8426aa95`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]

## God Nodes (most connected - your core abstractions)
1. `ModularArithmeticDataset` - 6 edges
2. `ModularArithmeticDataset` - 5 edges
3. `generate_pairs()` - 3 edges
4. `get_dataloaders()` - 2 edges
5. `Transformer` - 2 edges
6. `get_dataloaders()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (4 total, 3 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.33
Nodes (3): get_dataloaders(), ModularArithmeticDataset, Dataset

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModularArithmeticDataset` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.286) - this node is a cross-community bridge._
- **Why does `ModularArithmeticDataset` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.208) - this node is a cross-community bridge._
- **Why does `generate_pairs()` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._