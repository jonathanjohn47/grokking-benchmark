# Graph Report - .  (2026-06-23)

## Corpus Check
- Corpus is ~562 words - fits in a single context window. You may not need a graph.

## Summary
- 8 nodes · 9 edges · 2 communities
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 2,500 input · 800 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Grokking Thesis & Baseline|Grokking Thesis & Baseline]]
- [[_COMMUNITY_Academic Administration & Supervision|Academic Administration & Supervision]]

## God Nodes (most connected - your core abstractions)
1. `A Unified Benchmark of Grokking Predictors in Neural Networks` - 6 edges
2. `Jonathan John` - 2 edges
3. `Prof. Dr.-Ing. Sheikh Faisal Rashid` - 2 edges
4. `IU Internationale Hochschule` - 2 edges
5. `Nanda et al. Grokking` - 2 edges
6. `Canonical Grokking Replication Gate` - 2 edges
7. `Thesis Gantt Chart` - 1 edges
8. `Grokking Predictors` - 1 edges

## Surprising Connections (you probably didn't know these)
- `A Unified Benchmark of Grokking Predictors in Neural Networks` --conceptually_related_to--> `Jonathan John`  [EXTRACTED]
  context.md → context.md  _Bridges community 0 → community 1_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Thesis Organization and Direction** — grokking_benchmark_context_thesis_unified_benchmark, grokking_benchmark_context_jonathan_john, grokking_benchmark_context_sheikh_faisal_rashid, grokking_benchmark_context_iu_internationale_hochschule [EXTRACTED 1.00]

## Communities (2 total, 0 thin omitted)

### Community 0 - "Grokking Thesis & Baseline"
Cohesion: 0.50
Nodes (5): Canonical Grokking Replication Gate, Thesis Gantt Chart, Grokking Predictors, Nanda et al. Grokking, A Unified Benchmark of Grokking Predictors in Neural Networks

### Community 1 - "Academic Administration & Supervision"
Cohesion: 0.67
Nodes (3): IU Internationale Hochschule, Jonathan John, Prof. Dr.-Ing. Sheikh Faisal Rashid

## Knowledge Gaps
- **2 isolated node(s):** `Thesis Gantt Chart`, `Grokking Predictors`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A Unified Benchmark of Grokking Predictors in Neural Networks` connect `Grokking Thesis & Baseline` to `Academic Administration & Supervision`?**
  _High betweenness centrality (0.833) - this node is a cross-community bridge._
- **Why does `Jonathan John` connect `Academic Administration & Supervision` to `Grokking Thesis & Baseline`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `Prof. Dr.-Ing. Sheikh Faisal Rashid` connect `Academic Administration & Supervision` to `Grokking Thesis & Baseline`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **What connects `Thesis Gantt Chart`, `Grokking Predictors` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._