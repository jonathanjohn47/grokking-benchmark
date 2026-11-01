# Higher-MI Predictor: Points an Examiner Could Attack

**Thesis:** A Unified Benchmark of Grokking Predictors in Neural Networks · Predictor 7 (Higher-MI) · Prepared 1 November 2026

This document lists weak points in the Higher-MI implementation and experiment that a thesis examiner could raise. Each point says what was done and why it could be challenged. Solutions are deliberately not included. Points are ordered from most to least serious.

**Status key:** ❌ = open, ✅ = resolved. Replace ❌ with ✅ as each point is fixed.

**Progress:** 0 / 22 issues resolved, 0 / 3 unverified items checked.

---

## Big worries

### ❌ 1. Ω may be tracking a harmless shift in the scores
The model gives 113 scores per input. If you add the same amount to all 113, the answer doesn't change and neither does the loss, so training doesn't control that shared level and it drifts up and down. F0, F1 and F2 all ride on that drift together. Ω's peaks fall on the same epochs as the drift's peaks, so an examiner can say Ω is just picking up this drift, which has nothing to do with grokking.

### ❌ 2. Your Ω isn't the paper's Ω
Pomarico had a 3-class model, took its 3 fixed outputs, and tracked them across training steps. You have 113 classes, so you built three new quantities (correct score, best wrong score, average of the rest) and measured them across test examples at a single moment. That's a reasonable adaptation, but it's a different measurement. If it fails, an examiner can say your adaptation failed, not Pomarico's idea.

### ❌ 3. You tested a claim the paper never made
Pomarico says the Ω peak happens at grokking. Your benchmark asks whether it happens clearly before grokking. If a method was only ever claimed to coincide with grokking, failing an early-warning test is no surprise and could look like an unfair test.

### ❌ 4. The test examples might not be the original ones
Training randomly split the 113×113 sums into 30% train and 70% test. For Higher-MI, the split was rebuilt later on a different computer using the same seed. A seed only reproduces the same split when the software version is the same, and that version wasn't recorded. If the split changed, some examples called test data were actually training data.

### ❌ 5. The rule could look chosen after the fact
The benchmark says to take the highest peak as the event. But that rule, the code and the 'failed' verdict were all saved together in one commit, and a code comment still calls the rule an assumption awaiting confirmation. An examiner can ask how you can show the rule wasn't chosen after seeing the results.

### ❌ 6. The rule was likely to fail from the start
Ω's ups and downs get bigger as training goes on, so the tallest one is almost always near the end. Taking the tallest peak therefore almost guarantees an event after grokking. The test may be measuring the rule's bias rather than the predictor.

### ❌ 7. The chosen peak is a coin toss in some seeds
In seed 4 the tallest and second-tallest peaks are almost the same height but thousands of epochs apart. A tiny bit of noise could swap them and move P a long way. How much the curve wobbles from noise was never measured, so you can't say the chosen peak is the real one.

### ❌ 8. The estimator may overestimate
KSG estimates information by looking at how close each point is to its neighbours. After grokking, inputs like 3+5 and 5+3 probably give almost the same three numbers, so points stack on top of each other. KSG is known to overestimate when that happens, and the code doesn't guard against it.

---

## Moderate worries

### ❌ 9. The scale of the scores changes a lot
Early on the scores are small, and later they're about ten times bigger. In theory Ω doesn't care about scale, but with a finite sample KSG's estimate does, so some of Ω's change over time could come from scale alone.

### ❌ 10. The neighbour count is untested
KSG needs a choice of how many neighbours to look at (k = 4). That number was taken from a different part of the paper, and nobody checked whether k = 3 or k = 8 gives the same answer.

### ❌ 11. The three quantities are tied together by how they're defined
'Best wrong score' is the maximum of the wrong scores, and 'average of the rest' is whatever's left after removing it. So they're linked by construction, even in a random model, and part of the shared information Ω measures comes from the definitions, not from learning.

### ❌ 12. Easy versus hard examples could explain Ω
On easy inputs all three numbers might be high, and on hard ones all three low. That alone would make them move together across examples. A second curve was built to check for this, but the two curves were only compared by eye on one seed, which doesn't settle it.

### ❌ 13. The backup curve uses the future
The temporal version of Ω looks at a window centred on each epoch, so each value includes training that hasn't happened yet. A real early warning can't do that, and the window also means it can't produce a value before epoch 5,000.

### ❌ 14. The zero-crossing result is meaningless
Early in training Ω flickers just above and below zero, so 'the first time it crosses zero' is random. That variant passing the early test shouldn't count for anything.

### ❌ 15. Five seeds can't tell weak from useless
Ranking five numbers allows only a handful of possible correlations, and even random data hits 0.7 fairly often. So 'ρ = 0.70, fails' might really mean 'there aren't enough runs to know'.

### ❌ 16. No comparison runs
Ω was never computed on training data, or on a deliberately random model, to see what 'no grokking' looks like. Without that baseline it's hard to say what Ω's peaks mean.

### ❌ 17. The source paper is weak backing
It hasn't been peer reviewed, it uses a very different model and loss (one where the shared score level from point 1 does matter), and its grokking is mild. An examiner could ask why this paper deserved a slot in the benchmark.

---

## Small slips

### ❌ 18. One wrong fact in the write-up
It says Spearman's ρ has 11 possible values with five seeds. It actually has 21.

### ❌ 19. Checkpoints are described slightly wrong
The write-up says every 100 epochs up to 40,000. The code saves every 50, evaluates every 100, and stops at 39,999.

### ❌ 20. Two different run times
The write-up says 27 minutes and a code comment says about 22.

### ❌ 21. The point 4 problem also hits Dropout Variance
That predictor uses the same rebuilt test set whenever it's recomputed from saved checkpoints.

### ❌ 22. Nobody could rerun it exactly
The cloud run's log and software versions weren't kept, and the 'train fraction' setting in the config is ignored because the code hard-codes 30%.

---

## What checks out

Every number in the write-up's results tables and the 27 October progress report matches the saved result files. The grok epoch moves only 50–500 epochs when the accuracy threshold is raised to 0.95 or 0.99, so the 'peak after grokking' finding doesn't depend on the 0.90 cutoff.

---

## Not verified

Three things need the per-example arrays or a fresh run to check:

- ❌ The margins in Example 1.
- ❌ The 0.998 correlation between F0 and F1.
- ❌ Whether the rebuilt test set is truly separate from the training set.
