from pathlib import Path

context_path = Path("/Users/jonathanjohn/Documents/grokking-benchmark/context.md")

addition = """

---
### 2026-09-25: Switch from 24 log-spaced to linear EVAL_EVERY=100 (derived) and SAVE_EVERY=50 (GCD justification) - Viva Defense Note

#### Problem with old design
Old benchmark-v4 used 24 log-spaced checkpoints [0,1,2,3,5,9,15,...,39999]. This had no scientific basis - it was just compute budget for Dropout-Variance predictor. Log spacing oversamples first 2k epochs where token-embedding transient is crushed by weight-decay (w=1.0, alpha=0.01) and undersamples the actual grokking region (10k-20k epochs for p=113).

#### Why SAVE_EVERY=50 - Mathematical necessity
We want to evaluate ablation on eval grids DESIRED_EVAL_GRIDS = [100, 200, 250] without retraining.
Rule: You can only evaluate a grid if you have saved it. Save spacing must divide all desired eval grids.

GCD(100,200,250) = 50
LCM(100,250) = 500 - So if we save every 100, a 250 grid would silently become 500 grid.

Therefore SAVE_EVERY = 50 is mathematically forced. Storage cost is negligible: ~0.85 MB per checkpoint, 801 checkpoints (range(0,40000,50)+final) = ~680 MB per seed vs 401 checkpoints = ~340 MB for 100 spacing.

#### Why EVAL_EVERY=100 - Empirically derived (not from literature)
Not taken from Nanda et al. or any paper. Derived from our own M1 gate measurements for p=113:

1.  **Transition width measurement:** Test accuracy 0.1 -> 0.9 occurs in ~800-1200 epochs. Grok epoch itself is ~10k-15k for p=113.
2.  **Sampling requirement:** To detect early-warning, need at least 3-4 samples inside transition (Nyquist-like criterion). 
    - 50 spacing = 16-24 points inside transition (oversampling, diminishing returns)
    - 100 spacing = 8-12 points (sufficient)
    - 200 spacing = 4-6 points (borderline)
    - 250 spacing = 3-4 points (minimal)
3.  **MA window compatibility:** L2 predictor uses fast_window=50, slow_window=200. Eval spacing must be <= slow_window to avoid aliasing. 100 is half of slow window.
4.  **Compute-resolution trade-off:**
    - Spectral ~9.3s + Dropout-Variance ~1.8s = ~11s per checkpoint on MPS
    - 50 grid = 801 points = ~2.5h per seed
    - 100 grid = 401 points = ~1.2h per seed
    - 200 grid = 201 points = ~0.6h per seed
    For 9 predictors x 5 seeds, 50 grid = ~50 hours, 100 grid = ~25 hours. 100 fits in thesis timeline (weekend run on M1).
5.  **Literature resolution match:** Nanda Fig 7 uses ~400 points. Our 100 grid gives 401 points (linear) - comparable resolution but uniform coverage where grok happens.

#### Engineering principle
Save densely once, evaluate coarser later. Code allows --eval_every 50/100/200/250 without retraining because checkpoints already exist at 50 spacing. This decouples storage (cheap) from compute (expensive).

#### Validation for Viva
Lead fraction and Spearman rho are stable across 100, 200, 250 grids, so conclusion does not depend on 100. 100 is not a universal constant, it is a design choice for this benchmark justified by transition-width measurement and compute budget.
If examiner asks for higher resolution, we can rerun with --eval_every 50 without retraining.
---

"""

# Read existing and append if not already present
text = context_path.read_text(encoding="utf-8", errors="ignore")
if "Switch from 24 log-spaced to linear EVAL_EVERY=100" not in text:
    context_path.write_text(text + addition, encoding="utf-8")
    print("Added reason to context.md")
else:
    print("Reason already exists in context.md, skipping")
