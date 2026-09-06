# NANDA-UNIFIED BENCHMARK v4 — 4 Predictors CLOSED

**Config:** `configs/nanda_unified.yaml` — (a+b) mod 113, vocab 114, d_model=128, 4-head, full-batch 3830, AdamW lr=0.001 betas [0.9,0.98] weight_decay=1.0, init_std 0.0707, 40000 epochs, 24 log-uniform checkpoints
**Seeds:** 5 — grok_epochs [14474, 6988, 10418, 10193, 24021] — mean 13218.8 std 5900.6 — 5/5 grokked, 0/5 limit cycle

## Benchmark Result — All 4 CLOSED Negative (Valid Falsification)

| # | Predictor | Paper | Signal | Result per seed | Verdict |
|---|-----------|-------|--------|-----------------|---------|
| 1 | **L2 Norm** | Nanda et al. Fig 7 | MA-crossover epoch 109,121,119,112,103 vs grok | crossover far before grok 5/5 | **CLOSED negative** — tracks early optimisation transient |
| 2 | **Dropout** | Salah & Yevick 2025 | variance_peak_epoch 39999,15917,39999,39999,25232 — peak/grok ratio 2.76,2.27,3.83,3.92,1.05 | peak after grok 5/5 | **CLOSED negative** — robust only after circuit forms (k=30, rate=0.5 deviation documented) |
| 3 | **Spectral** | Canatar et al. 2021 Nature Comm | k_90 3448->3369, k_90_min_epoch 25232,10040,25232,39999,39999 — alignment_max 25232,15917,39999,39999,39999 | min/max after grok 5/5, k_90 drop only 2-3% | **CLOSED negative** — kernel regime theory, grokking is feature-learning, no rank collapse before grok |
| 4 | **AGE** | Papyan et al. 2020 PNAS NC1 + Beaglehole 2024 + Mallinar 2024 | NC1 45.37->0.06, 44.64->0.05, 44.58->0.06, 47.85->0.06, 46.09->0.04 — nc1_min_epoch 25232,15917,39999,39999,39999 — ratio 1.74,2.27,3.83,3.92,1.66 | min after grok 5/5, collapse real but late | **CLOSED negative** — NC1 collapse 660x happens but is post-grok consequence |

**Two-criteria test used for all:** (1) does signal extremum lead grok_epoch? No in 5/5 for all 4. (2) consistent across seeds? N/A — fails (1).

**Files:**
- `results/nanda_unified/aggregate.json` — this file, 5 seeds x 4 predictors
- `results/nanda_unified/seed_*/l2_norm/`, `dropout/`, `spectral/`, `age/`, `checkpoints/` — 24 checkpoints per seed
- `results/nanda_unified/reports/` — plots (run plot_nanda_results.py)

**Reproduce:**
```bash
python run_nanda_benchmark.py --seeds 5 --epochs 40000 --output_dir results/nanda_unified --config configs/nanda_unified.yaml --predictors l2,dropout_gap,dropout_variance,spectral,age
python plot_nanda_results.py --results_dir results/nanda_unified --output_dir results/nanda_unified/reports
```

**Next:** Predictor 5 HTSR Alpha — same checkpoint-only plugin pattern.
