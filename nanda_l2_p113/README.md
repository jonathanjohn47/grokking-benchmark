# nanda_l2_p113 — L2-Norm predictor on (a + b) mod 113

A secondary, self-contained run of the four-head grokking benchmark. It is a
**fully faithful Nanda et al. (2023) replication**, kept separate from the main
experiment in `src/`.

## What this is

A copy of the four-head experiment (`src/train_four_head.py` +
`transformer_four_head.py` + `predictors/l2_norm.py` + the data pipeline +
`unified_measurements.py` + `run_full_benchmark.py`), trimmed down to the
L2-Norm predictor and pointed at the prime Nanda uses.

Nothing here imports from `src/`. The folder is standalone.

## The four deliberate differences from `src/`

| | Main experiment (`src/`) | This folder |
|---|---|---|
| Prime `p` | 97 | **113** (Nanda's mainline prime) |
| Predictors | L2-Norm + Dropout multi-rate sweep | **L2-Norm only** — no Dropout code, imports, or files |
| AdamW `betas` | `(0.9, 0.999)` (PyTorch default) | **`(0.9, 0.98)`** (Nanda et al. / Power et al.) |
| Weight init | PyTorch defaults (`nn.Embedding` = `N(0, 1)`, `nn.Linear` = Kaiming-uniform) | **`N(0, 0.8 / sqrt(d_model))`** on every matrix, embeddings included (TransformerLens / Nanda) |

Everything else is identical: `TransformerFourHead` (4 heads, `d_model=128`,
`d_mlp=512`, ReLU MLP, no LayerNorm, no Linear biases), AdamW `lr=1e-3`
`weight_decay=1.0`, 40000 epochs, full-batch training on a 30/70 random split,
fresh `torch.seed()` per run, the same L2-Norm predictor code and windows
(`fast_window=50`, `slow_window=200`, MA-of-MA `fast_window=20`,
`skip_epochs=100`, noise-floor `quiet_epoch_cutoff=90`), 3 seeded runs.

For `p = 113`, number tokens are `0..112` and the `=` token id is `113`, so
`vocab_size = 114`.

## Why `p = 113`, `betas = (0.9, 0.98)`, and the small init

- `p = 113` is the prime in Nanda's mainline experiment. Nanda's Figure 7
  (`Literature/nanda_figures/`) — the only weight-norm panel in the whole
  paper — is a `p = 113` curve, so the weight-norm curve produced here can be
  read directly against it. The main experiment stays on `p = 97` for
  continuity with Power et al. and the existing runs.
- `betas = (0.9, 0.98)` is the value Nanda et al. and Power et al. both use.
  Per `context.md` (Sep 3, 2026 session, item 9.2) this is expected to shrink
  the slingshot-type weight-norm spike seen at the grok transition in the
  `p = 97` runs, which came from `beta2 = 0.999` on the long low-loss plateau.
- **Small init** (`std = 0.8 / sqrt(d_model)` ≈ 0.0707 for `d_model = 128`) is
  the TransformerLens scheme Nanda uses. With PyTorch's default `nn.Embedding`
  init (`N(0, 1)`) the token-embedding table is ~94% of the starting weight
  norm, and `weight_decay = 1.0` crushes it in the first ~2000 epochs — a
  large early collapse in the L2 curve that has nothing to do with grokking
  and is absent from Nanda's Figure 7. Sum of squared weights at init drops
  from ≈ 15,500 (PyTorch defaults) to ≈ 1,050 with this change; Nanda's
  Figure 7 curve starts at ≈ 1,800. The model then has to *grow* its weights
  to memorise, matching Nanda's memorisation-phase rise.

## How to run

```
python nanda_l2_p113/run_benchmark.py     # 3 seeded runs + analysis, resumable
python nanda_l2_p113/train.py             # a single run
```

`run_benchmark.py` resolves its own paths from `__file__`, so it can be run
from anywhere.

## Outputs

- `nanda_l2_p113/runs/run_<N>/`
  - `training/` — `train_acc_history.npy`, `test_acc_history.npy`,
    `loss_history.npy`
  - `l2_norm/` — raw + smoothed L2 norm, moving averages, MA-of-MA,
    acceleration derivatives, detection epoch, 4 PNGs
  - `reports/training_report.pdf` — 3 pages (grokking curve, loss, L2 norm + MAs)
  - `seed.npy`
- `nanda_l2_p113/benchmark_analysis/` — `01_grokking_curves.png`,
  `02_l2_norm_comparison.png`, `benchmark_report.pdf` (cross-run)

`nanda_l2_p113/runs/` is gitignored (it matches the repo-wide `runs/` rule).
