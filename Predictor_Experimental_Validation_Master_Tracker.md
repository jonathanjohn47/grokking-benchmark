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
- Phase 1 (Signal and Trigger Definition): PENDING
- Phase 2 (Implementation and Integration): PENDING
- Phase 3 (Synthetic Detector Validation): PENDING
- Phase 4 (Synthetic Scorer Validation): PENDING
- Phase 5 (Empirical Execution): PENDING
- Phase 6 (Formal Scoring and Verdict): PENDING

## 5. HTSR Alpha Predictor
**Status Snapshot Date: November 16, 2026**
- Phase 1 (Signal and Trigger Definition): PENDING
- Phase 2 (Implementation and Integration): PENDING
- Phase 3 (Synthetic Detector Validation): PENDING
- Phase 4 (Synthetic Scorer Validation): PENDING
- Phase 5 (Empirical Execution): PENDING
- Phase 6 (Formal Scoring and Verdict): PENDING
