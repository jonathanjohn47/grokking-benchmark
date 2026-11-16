# Predictor Experimental Validation Master Tracker

This tracker records the experimental validation status of each predictor as of November 16, 2026. It serves as the definitive checklist to ensure every predictor is experiment-complete and viva-ready during the redesign phase.

## 1. L2 Norm Predictor
**Status Snapshot Date: November 16, 2026**
- Phase 1 (Signal and Trigger Definition): COMPLETED
- Phase 2 (Implementation and Integration): COMPLETED
- Phase 3 (Synthetic Detector Validation): COMPLETED
- Phase 4 (Synthetic Scorer Validation): COMPLETED
- Phase 5 (Empirical Execution): COMPLETED
- Phase 6 (Formal Scoring and Verdict): COMPLETED

## 2. Dropout Predictor
**Status Snapshot Date: November 16, 2026**
- Phase 1 (Signal and Trigger Definition): COMPLETED
- Phase 2 (Implementation and Integration): COMPLETED
- Phase 3 (Synthetic Detector Validation): COMPLETED
- Phase 4 (Synthetic Scorer Validation): COMPLETED
- Phase 5 (Empirical Execution): COMPLETED
- Phase 6 (Formal Scoring and Verdict): COMPLETED

## 3. Spectral Predictor
**Status Snapshot Date: November 16, 2026**
- Phase 1 (Signal and Trigger Definition): COMPLETED
- Phase 2 (Implementation and Integration): COMPLETED
- Phase 3 (Synthetic Detector Validation): COMPLETED
- Phase 4 (Synthetic Scorer Validation): COMPLETED
- Phase 5 (Empirical Execution): COMPLETED
- Phase 6 (Formal Scoring and Verdict): COMPLETED

Evidence — Pre-registration: docs/spectral_preregistration.md
Evidence — Detector tests: tests/test_spectral_signal_dependence.py (9 passed)
Evidence — Scorer tests: tests/test_spectral_scorer_synthetics.py (6 passed)
Evidence — Artifacts: 08_Experiments/results/nanda_unified/spectral/
Empirical P_i: [None, None, 9200, None, None]
Ground-truth G_i: [13457, 7237, 9852, 11138, 23913]
Lead fraction / Spearman rho / Noise margin: 0.0000 / None / 0.0000
Superiority Index: 0.0000
Verdict: FAIL (Note: All phases are COMPLETED as executed and evidenced; the empirical verdict is FAIL)
Failure mode: (c) never fires / insufficient firing count (1 of 5 seeds fired: seed 2 at 9200; seeds 0, 1, 3, 4 never satisfied joint persistence)

## 4. AGE Predictor
**Status Snapshot Date: November 16, 2026**
- Phase 1 (Signal and Trigger Definition): COMPLETED
- Phase 2 (Implementation and Integration): COMPLETED
- Phase 3 (Synthetic Detector Validation): COMPLETED
- Phase 4 (Synthetic Scorer Validation): COMPLETED
- Phase 5 (Empirical Execution): COMPLETED
- Phase 6 (Formal Scoring and Verdict): COMPLETED

Evidence — Implementation: `06_Code/src/predictors/age.py`
Evidence — Detector tests: `06_Code/tests/test_age_detector.py` (2 passed)
Evidence — Scorer tests: `06_Code/tests/test_age_scorer.py` (2 passed)
Evidence — Artifacts: `08_Experiments/results/results_per_nanda_power/seed_*/age/` (401 checkpoints per seed)
Empirical P_i (NC1 min): [13000, 28100, 30600, 35900, 34400]
Empirical P_i (log10 NC1 10% fall): [5100, 3700, 900, 5400, 3700]
Ground-truth G_i (cut 0.99): [8021, 8253, 4094, 7831, 6637]
Official Metrics (NC1 min): lead_frac = 0.0000, rho = -0.5000, mean_rel_lead = -3.4535
Official Metrics (log10 NC1 10% fall): lead_frac = 1.0000, rho = 0.4617, mean_rel_lead = 0.4898
Verdict: FAIL (0 of 3 pre-declared rows passing; Note: All 6 phases COMPLETED)
Physical interpretation: NC1 within-class variability collapse is postdictive. The global NC1 minimum occurs significantly after grokking (1.62x - 7.47x G_i) during terminal representation crystallization, functioning as a consequence of generalization rather than an ex-ante early warning indicator. Log10 NC1 10% fall leads but fails the Spearman rank correlation threshold (rho = 0.4617 < 0.90).

## 5. HTSR Alpha Predictor
**Status Snapshot Date: November 16, 2026**
- Phase 1 (Signal and Trigger Definition): PENDING
- Phase 2 (Implementation and Integration): PENDING
- Phase 3 (Synthetic Detector Validation): PENDING
- Phase 4 (Synthetic Scorer Validation): PENDING
- Phase 5 (Empirical Execution): PENDING
- Phase 6 (Formal Scoring and Verdict): PENDING
