# Commutator Defect Predictor: Points an Examiner Could Attack

**Thesis:** A Unified Benchmark of Grokking Predictors in Neural Networks · Predictor 8 (Commutator Defect) · Prepared 4 November 2026

This document lists weak points in the Commutator Defect implementation and experiment that a thesis examiner could raise. Each point says what was done and why it could be challenged. Solutions are deliberately not included. Points are ordered from most to least serious.

**Status key:** ❌ = open, ✅ = resolved. Replace ❌ with ✅ as each point is fixed.

**Progress:** 0 / 16 issues resolved.

**Verdict note:** Under the frozen protocol the PRIMARY run (η = 1e−3) is already **FAIL** (Spearman ρ = 0.6325, mean relative lead = 0.9535). None of the points below can flip that FAIL into a PASS unless a point explicitly says so. None of them does.

---

## Big worries

### ❌ 1. The alarm fires during memorisation, not before grok
PRIMARY events land at epochs 400–600. Grok epochs are 6921–23747. In the Stage 6 figures (`figures/commutator_defect/`, seed 0 especially), D̃ rises from about 13 to about 1,700 as train accuracy reaches 1. So the Xu onset is catching the memorisation jump, not a build-up that sits just before generalisation. An examiner can say the predictor is timing "train locked in", not "grok is coming".

### ❌ 2. Spearman ρ and mean relative lead both fail the frozen band
PRIMARY: ρ = 0.6325 (need ≥ 0.9) and mean relative lead = 0.9535 (need ≤ 0.9). Lead fraction is 1.0, but that only means every seed fired far too early to be informative about when that seed will grok. SECONDARY η = 1e−4 is worse still on correlation (ρ = 0.1581) and is unscored. Changing the story around these numbers does not change the numbers, so this cannot turn FAIL into PASS.

### ❌ 3. D̃ spikes track the recurring gradient-norm spikes, not grok
Stage 6 reading: large pre-grok D̃ spikes line up with ‖g‖ cycling between about 1e−6 and about 1. Those gradient-norm spikes continue after grok as well. So D is reacting to a landscape / optimiser-cycle feature that is not special to the grokking transition. An examiner can say D is not a grokking-specific signal.

### ❌ 4. D can rise just because gradients shrink
The formula divides by ‖η g_A‖ · ‖η g_B‖. On flat stretches those norms collapse, so D grows even if the true order-of-steps gap is tiny. That alone can push D̃ over the Xu onset thresholds during quiet training, without any grokking mechanism. This weakens the claim that a large D means "transverse curvature linked to grok".

### ❌ 5. The "first 3 values" baseline sits on the initial transient
b is the median of D̃ at epochs 0, 100 and 200. Seed 0 PRIMARY medians there are 59.96, 12.90, 43.91 — still in the early transient, before training has settled. So the onset threshold (10 · max(b, 0.1), and the floor of 20) is fixed before the curve has a stable early regime. An examiner can say the baseline is not a fair "quiet" reference.

---

## Moderate worries

### ❌ 6. η dependence at epoch 12000; Check 2 fails at the frozen η
At epoch 12000, pairs 0 and 4 jump non-monotonically as η shrinks (about 141, 247, 81). That is likely the probe step crossing activation kinks. Sanity Check 2 (halving η) fails there at the frozen η = 1e−3. So even the measurement itself is not stable in η at every checkpoint. Stage 3d kept η frozen anyway; this is a property of Xu's probe, not a licence to retune.

### ❌ 7. Float64 rounding made the naive numerator noise (fixed, but a Xu hazard)
Subtracting full weight vectors for ‖θ_AB − θ_BA‖ was dominated by float64 rounding on flat checkpoints (epoch 6000: naive ≈ 82.7 vs gradient-difference form ≈ 59.4). Stage 3b switched to the algebraically identical gradient-difference numerator. Anyone reproducing Xu with the naive path can get a wrong D curve and not notice. The fix does not change our FAIL verdict; it only made the scored D trustworthy.

### ❌ 8. Xu's own lead fraction was only 43% at lr 1e−3
Even in the source paper, at learning rate 1e−3 the lead fraction is only about 43%. Our frozen training uses AdamW lr = 1e−3. An examiner can ask why a signal that was already weak in Xu's own tables was expected to clear a strict early-warning protocol here.

### ❌ 9. Non-grokking runs also show 30–50× rises in Xu's data
Xu reports large rises in D on non-grokking controls as well. Separation from those controls is only clear in hindsight. So a big rise alone does not mark grokking. Our benchmark never claimed otherwise, but an examiner can say the raw magnitude story was never discriminative enough to support a PASS-style early warning.

### ❌ 10. Weight decay matters in Xu; our probe uses plain loss
Xu's effect depends on weight decay in training. Our frozen probe takes plain gradients of the cross-entropy only — no weight-decay term — as required by the Stage 1 rule. That is faithful to the freeze, but an examiner can say we measured a different object from the one Xu's narrative leans on.

### ❌ 11. Full-batch training forced artificial A / B batches
Nanda-Unified trains full-batch. Xu's story is about the order of two optimiser batches. We draw A and B as 512-example resamples with replacement from the train split. They are not batches the optimiser actually saw. The adaptation is frozen and necessary, but it is still an adaptation an examiner can challenge.

### ❌ 12. Probe uses plain gradient steps; training uses AdamW
Training is AdamW (betas 0.9, 0.98, weight decay 1.0). The probe does two plain gradient steps of size η. D therefore measures a loss-landscape commutator, not the actual optimiser's step-order effect under AdamW. An examiner can say the measured D is one step removed from the dynamics that produce grokking in this setup.

---

## Small slips

### ❌ 13. Paper says K = 9; code uses K = 5; we froze K = 5
Xu's text and code disagree. We followed the code (K = 5), as recorded in Stage 1. That is a defensible choice, but an examiner can still ask which of Xu's two statements we claim to test.

### ❌ 14. The floor of 20 and the 10× factor may not transfer
The onset rule (D̃ > 10 · max(b, 0.1) and D̃ > 20) is copied from Xu's setup. On Nanda-Unified the early medians and the memorisation jump sit on a different scale. Those constants were frozen before any D curve, which is correct protocol — but they may simply be the wrong scale for this task. Retuning them after seeing results would be post-hoc; leaving them gives the FAIL we have.

### ❌ 15. Single-author preprints, not peer reviewed
Sources are Y. Xu (2026), arXiv:2602.16746 and arXiv:2602.16967. Neither is peer reviewed. An examiner can ask why this predictor earned a slot on the strength of those preprints alone.

### ❌ 16. A post-hoc η change to 1e−4 was reverted (Stage 3c → 3d)
After seeing D values, Stage 3c moved the default probe η to 1e−4 so Check 2 would pass. Stage 3d restored the frozen PRIMARY η = 1e−3; 1e−4 stayed SECONDARY and unscored. The process is now clean, but the episode itself is an examiner question: thresholds and η must not be tuned to force a pass. PRIMARY stays FAIL at the frozen η.

---

## What checks out

- PRIMARY events fire in all 5 seeds: [600, 500, 500, 400, 600]; lead_frac = 1.0 under the frozen lead rule.
- Stage 5 recompute: 401 snapshots × 5 seeds × 2 η values, zero NaNs in raw D and medians; exit code 0.
- Stage 7 unchanged-check: event epochs and frozen scores for the other 7 predictors match before/after; only Commutator Defect is new.
- The gradient-difference numerator removes the float64 flat-checkpoint artefact; Stage 3 sanity Checks 1, 3, 4 pass at the frozen η.
- SECONDARY η = 1e−4 also FAIL and was never mixed into the PRIMARY verdict.

---

## Summary table

| # | Item | Category | Could it change the verdict? |
|---|---|---|---|
| 1 | Alarm during memorisation, not before grok | Result | No |
| 2 | ρ = 0.6325 and mean rel. lead = 0.9535 fail bands | Result | No |
| 3 | D̃ tracks gradient-norm spikes (incl. after grok) | Result | No |
| 4 | Denominator collapses when gradients shrink | Result | No |
| 5 | First-3 baseline sits on the initial transient | Result | No |
| 6 | η dependence / Check 2 fails at epoch 12000 | Result | No |
| 7 | Float64 naive-numerator hazard (fixed) | Result | No |
| 8 | Xu lead fraction only ~43% at lr 1e−3 | Source / protocol | No |
| 9 | Non-grokking runs also rise 30–50× in Xu | Source / protocol | No |
| 10 | Probe omits weight decay (per freeze) | Source / protocol | No |
| 11 | Artificial A/B resamples under full-batch training | Source / protocol | No |
| 12 | Plain GD probe vs AdamW training | Source / protocol | No |
| 13 | Paper K = 9 vs code / freeze K = 5 | Source / protocol | No |
| 14 | Floor of 20 and 10× may not transfer | Source / protocol | No |
| 15 | Single-author preprints, not peer reviewed | Source / protocol | No |
| 16 | Post-hoc η→1e−4 reverted in Stage 3d | Process | No |

---

## Bottom line

The frozen PRIMARY verdict is FAIL, and that FAIL is robust. The alarm is early and memorisation-tied, correlation with grok epoch is too weak, mean relative lead is too large, and D also reacts to gradient-norm cycling and shrinking denominators that are not grok-specific. The source-paper and protocol adaptations (weight decay, full-batch A/B, AdamW vs plain steps, K mismatch, onset constants, preprint status) explain why a PASS was never likely; none of them rewrites ρ = 0.6325 or mean relative lead = 0.9535 into the PASS band. Secondary η = 1e−4 fails too and stays unscored. No item on this list turns FAIL into PASS.
