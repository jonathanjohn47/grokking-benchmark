# Archived — single-head experiment

**Archived on:** 2 September 2026
**Reason:** The benchmark has been consolidated onto the four-head model
(`d_model=128`, 4 attention heads), which is the faithful Nanda et al.
architecture. Single-head was an earlier from-scratch variant used for
harness/pipeline validation. It is **not** part of the benchmark protocol
any more.

See `context.md` (session dated 2 September 2026) for the full reasoning.

## What is here

| Path | Was at | Purpose |
|---|---|---|
| `train.py` | `src/train.py` | Single-head training loop (10000 epochs, no seeding) |
| `models/transformer.py` | `src/models/transformer.py` | Single-head `Transformer` class (one attention head over the full `d_model`) |
| `plot_results.py` | `src/plot_results.py` | Single-head L2-Norm plotting script |
| `results/` | `results/single_head/` | The one pilot run (grok epoch 4568). **Not seeded — cannot be regenerated exactly.** |

## Nothing was deleted

Every file above was **moved**, not removed. The single-head code still works
as written.

## How to bring it back

The scripts assume `src/` is on the import path (they do
`from models.transformer import ...`, `from data.modular_arithmetic import ...`,
`from predictors.l2_norm import ...`, `from unified_measurements import ...`).
The shared modules (`data/`, `predictors/`, `unified_measurements.py`) are
still in `src/`. To run single-head again:

1. Move `train.py` back to `src/train.py`.
2. Move `models/transformer.py` back to `src/models/transformer.py`.
3. (Optional) Move `plot_results.py` back to `src/plot_results.py`.
4. Run `python src/train.py` from the project root.

If it is being restored as a deliberate robustness check (architecture-change
test), first make it match the four-head protocol: same `num_epochs`, record a
fresh seed per run (`torch.seed()`), and use the same `run_N/` folder scheme.

## Open item (still postponed)

The single-head L2-Norm reliability anomaly flagged by the professor is still
open. Archiving this code does not close that question.
