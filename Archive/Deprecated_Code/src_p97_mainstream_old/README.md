# archive/src_p97_mainstream_old/ — the pre-Nanda-Unified `src/` snapshot

This is a **full copy of `src/` as it stood at commit `6005a9f`**, i.e.
immediately before the September 4, 2026 switch to the Nanda-Unified
substrate (commit `029b739`). It is a read-only historical record — the
live code is `src/`.

## Why it is kept

`src/` is now the **Nanda-Unified mainstream** (see `context.md` and
`configs/nanda_unified.yaml`). This folder preserves the earlier
"p97 mainstream" so the old baseline is inspectable on disk without
`git checkout`.

## What the old code did (differs from the current `src/`)

| | old (this folder) | current `src/` |
|---|---|---|
| prime `p` | 97 (hardcoded) | 113 (from `configs/nanda_unified.yaml`) |
| `=` token id | hardcoded `97` in `data/modular_arithmetic.py` | `= modulus` (113) |
| weight init | PyTorch defaults — `nn.Embedding` = `N(0,1)`, `nn.Linear` = Kaiming-uniform | `N(0, 0.8/sqrt(d_model))` for every matrix (`_init_weights`) |
| MLP biases | `bias=False` | `bias=True`, zero-init |
| AdamW `betas` | `(0.9, 0.999)` (PyTorch default) | `(0.9, 0.98)` |
| weight-norm logging | `l2` only | `l2` + `sum_w2` + per-module `sum_w2` (5 groups) |
| protocol source | hardcoded constants in `train_four_head.py` | `configs/nanda_unified.yaml` (asserted) |

## Not a runnable tree

Nothing imports from here. It has no `__init__` wiring to the repo and is
not on `sys.path`. Do not point experiments at it. Recover any single old
file with `git show 6005a9f:src/<path>` instead of copying from here.

The runs this old code produced are separately archived at
`archive/p97_default/` (gitignored, local disk only).
