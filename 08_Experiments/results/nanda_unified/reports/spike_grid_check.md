# Higher-MI Spike Grid Diagnosis Report

**Question to settle:** Are the spikes in $\Omega_{pop}$ locked to the checkpoint grid (same epoch in every seed), or do they follow each model's own trajectory (different epochs in different seeds)?

## 1. Cross-Check with Recorded Values

| Seed | Grok Epoch | Recorded Event Epoch (Omega max) | Detected Max Peak Epoch | Match? |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 12987 | 15500 | 15500 | YES |
| 1 | 6921 | 8500 | 8500 | YES |
| 2 | 9510 | 20600 | 20600 | YES |
| 3 | 11019 | 12900 | 12900 | YES |
| 4 | 23747 | 32500 | 32500 | YES |

## 2. Summary Table: Peaks per Seed

| Seed | Grok | N Peaks | First 8 Peak Epochs |
|:---:|:---:|:---:|:---|
| 0 | 12987 | 27 | [1900, 3500, 4900, 6200, 7600, 9100, 10500, 11900] |
| 1 | 6921 | 24 | [1900, 3400, 5000, 6600, 8500, 10300, 12200, 13700] |
| 2 | 9510 | 13 | [1900, 3600, 5200, 6900, 8300, 10100, 20300, 20600] |
| 3 | 11019 | 33 | [1900, 3400, 4900, 6400, 7900, 9300, 10800, 12900] |
| 4 | 23747 | 29 | [1900, 3600, 5000, 6500, 7800, 9300, 11000, 12300] |

## 3. Test Results

### Test 1: Coincidence Across Seeds
- **Condition:** For every peak in a seed, is there a peak in every other seed within $\pm 3$ checkpoints ($\pm 300$ epochs)?
- **Pooled Coincidence:** **15 / 126** peaks (**11.90%**)
- **Per-Seed Breakdown:**
  - **Seed 0:** 3/27 coincident (11.1%) — coincident epochs: `[1900, 3500, 4900]`
  - **Seed 1:** 3/24 coincident (12.5%) — coincident epochs: `[1900, 3400, 5000]`
  - **Seed 2:** 3/13 coincident (23.1%) — coincident epochs: `[1900, 3600, 5200]`
  - **Seed 3:** 3/33 coincident (9.1%) — coincident epochs: `[1900, 3400, 4900]`
  - **Seed 4:** 3/29 coincident (10.3%) — coincident epochs: `[1900, 3600, 5000]`

### Test 2: Period and Spread Across Seeds
- **Condition:** Median gap between consecutive peaks per seed, and spread (max minus min median) across seeds.
  - **Seed 0:** Median gap = **14.0** checkpoints (1400 epochs)
  - **Seed 1:** Median gap = **16.0** checkpoints (1600 epochs)
  - **Seed 2:** Median gap = **26.5** checkpoints (2650 epochs)
  - **Seed 3:** Median gap = **15.0** checkpoints (1500 epochs)
  - **Seed 4:** Median gap = **14.5** checkpoints (1450 epochs)
- **Spread Across Seeds:** **12.50** checkpoints (1250 epochs)

## 4. Verdict

Decision rules:
- `spread <= 2 and coincidence >= 50%` $\rightarrow$ **LOCKED TO THE GRID**
- `coincidence <= 30%` $\rightarrow$ **DRIFTING (spikes belong to the model)**
- `otherwise` $\rightarrow$ **MIXED**

**Verdict:** **`DRIFTING (spikes belong to the model)`**

*Evaluation details:* Coincidence is **11.90%** ($\le 30\%$). Median gap spread is **12.50** checkpoints.

### Physical Mechanism and Observations:
1. **Early Phase (Epochs 0 - 5200):** All 5 seeds exhibit coincident peaks at epoch 1900, ~3500, and ~5000. In this initial phase, training accuracy rapidly approaches 100% (memorization phase) across all seeds around epoch 205, followed by the peak of L2 norm at epoch ~1050 under weight decay. Representation geometry undergoes an initial reorganization common to modular addition before individual circuit formation diverges.
2. **Post-Grokking Phase (Epochs 5200 - 40000):** As each model undergoes grokking at wildly different epochs (Seed 1 at 6921, Seed 2 at 9510, Seed 3 at 11019, Seed 0 at 12987, Seed 4 at 23747), the spike patterns completely desynchronize. For instance, Seed 2 undergoes a long quiescent period between epochs 10100 and 20300 ($\Omega_{pop} < 0.20$), while Seed 3 exhibits high-frequency oscillations (33 peaks).
3. **Conclusion:** Spikes are not artifacts locked to a static checkpoint grid or fixed evaluation data ordering across the entire training run; they drift with each model's representation evolution.

## 5. Full Peak Epoch Lists per Seed

### Seed 0 (27 peaks)
```json
[1900, 3500, 4900, 6200, 7600, 9100, 10500, 11900, 13300, 15500, 17800, 20400, 20700, 22500, 24200, 25300, 27400, 27700, 28900, 29900, 30200, 32400, 32700, 34400, 37100, 37400, 39700]
```

### Seed 1 (24 peaks)
```json
[1900, 3400, 5000, 6600, 8500, 10300, 12200, 13700, 15700, 16900, 17700, 19300, 21000, 24500, 26300, 28100, 29600, 31000, 31400, 33100, 34400, 34900, 36600, 38400]
```

### Seed 2 (13 peaks)
```json
[1900, 3600, 5200, 6900, 8300, 10100, 20300, 20600, 24300, 27800, 32400, 35900, 39600]
```

### Seed 3 (33 peaks)
```json
[1900, 3400, 4900, 6400, 7900, 9300, 10800, 12900, 13200, 15000, 17000, 17300, 19200, 21000, 21300, 22800, 23200, 23500, 25400, 25700, 27500, 27800, 29700, 30000, 31800, 32100, 33400, 34000, 36100, 36400, 38300, 38600, 39800]
```

### Seed 4 (29 peaks)
```json
[1900, 3600, 5000, 6500, 7800, 9300, 11000, 12300, 13800, 15500, 16900, 18300, 19800, 21500, 22900, 24800, 26900, 28600, 30500, 30800, 32500, 33800, 34500, 36200, 37500, 37900, 39100, 39600, 39900]
```
