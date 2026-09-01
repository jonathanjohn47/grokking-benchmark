# L2 Norm Predictor — Easy Notes for Revision and Viva

*Based on: "L2 Norm Predictor Investigation Report", Phase 2 – Predictor 1 of 9, dated 7 August 2026.*

---

## 1. What This Report Is About, in One Line

Jonathan tried five different ways of using the L2 norm of the model's weights to predict grokking *before* it happens, and none of the five ways passed all the tests he had set for a "valid predictor." The report records this honestly as a negative result, and recommends moving ahead to the next predictor, namely Dropout.

Let us now go through the report section by section, in simple language.

---

## 2. Objective — In Simple Words

Grokking is the phenomenon where a neural network first memorises the training data (train accuracy becomes 100%, but test accuracy stays low), and then, after a long wait, suddenly generalises (test accuracy also jumps to nearly 100%).

The **L2 norm** of the weights is simply a single number that tells us how large the model's weights are, taken together. It is computed by squaring every weight, adding all these squares, and taking the square root of the total. As training goes on, this number keeps changing, and researchers have observed that its behaviour is related to grokking in some way.

The objective of this predictor was straightforward: **can we look only at the L2 norm curve, during training, and fire a signal that reliably comes before the test accuracy jump — not after it?** If yes, L2 norm becomes a usable "early warning system" for grokking.

This is Predictor 1 out of the 9 predictors in the thesis's evaluation order:

> L2 Norm → Dropout → Spectral → AGE → HTSR Alpha → Correlation Traps → Weight-PCA → Higher-MI → Commutator Defect

---

## 3. Experimental Setup — Table

| Item | Value |
|---|---|
| Task | Modular addition, (a + b) mod 97 (the standard Nanda et al. grokking task) |
| Input format | Combined token sequence [a, b, "="], prediction taken from the "=" position |
| Model | Single-head, single-layer Transformer; token + position embedding; Q/K/V self-attention with softmax; residual connection; MLP with 4× expansion and ReLU, also with residual; linear output head; d_model = 128; **no LayerNorm used** |
| Optimizer | AdamW, learning rate 1e-3, weight_decay = 1.0 (a strong regularisation, needed to induce grokking) |
| Data split | 30% train / 70% test → 2822 train pairs, 6587 test pairs, out of 9409 total |
| Training length | 10,000 epochs, run on CPU (MPS and CUDA were not available in that environment) |
| Grokking criterion | The first epoch at which test accuracy crosses 90% |

Note this point carefully for viva: the model deliberately has **no LayerNorm** and only **one attention head**. This detail becomes important later, in the side finding about the test accuracy plateau.

---

## 4. Five Detection Strategies Tried, in Order

Jonathan did not just try one idea and stop — he tried five different strategies, one after another, and each failure taught something useful for the next attempt. This chronological journey is itself a good viva story to tell.

| # | Strategy | What it does | Why it was abandoned / result |
|---|---|---|---|
| 1 | Raw rate-of-decline threshold | Fire when the L2 norm is falling fast | **Signal was inverted.** The L2 norm falls fastest during early memorisation, not during the actual grok transition. So this rule cannot tell the two phases apart. |
| 2 | Second-derivative inflection (acceleration sign flip) | Fire where the *rate of change of the rate of change* flips sign | Raw acceleration was too noisy; the rule kept firing at an artificial boundary (the "skip-epoch" boundary in the recorded data), not at any real feature of training. |
| 3 | Double-smoothed inflection | Smooth the curve twice before looking for the inflection point | Looked cleaner visually, but this line of attack was set aside in favour of a different family of methods (moving-average crossover), before being fully tested. |
| 4 | Fast/Slow moving-average (MA) crossover, windows 50 and 200 | Fire at the first point where a fast-moving average crosses a slow-moving average | The very first crossover always happened due to early-training noise. Lead time (gap before grokking) was 3000–5000+ epochs on every run — far too early to be a useful, tight signal. |
| 5a | MA-of-MA: fixed threshold trigger (fire at 10× the noise floor) | Take a moving average of the moving average, and fire when it crosses a fixed multiple of the "quiet" baseline level | A fixed multiplier does not generalise across runs. On a second run, two early humps (12.8× and 19× the noise floor) crossed the threshold well before the real signal, so the rule fired far too early — epoch 205 against an actual grok epoch of 4806. |
| 5b | MA-of-MA: first zero-crossing trigger | Fire at the point where (fast MA of slow MA) minus (slow MA) first goes from positive to negative | This is the strategy that was finally **adopted and tested properly across three runs.** It looked clean on two runs but **failed on the third run** — see Section 5 below for the full evidence. |

---

## 5. Three Formal Criteria for Judging Any Trigger

Before judging strategy 5b (or any strategy) as good or bad, Jonathan first fixed three formal rules. This is an important point for viva, because it shows methodological discipline — the strategies were not judged "by eye," but against a written standard, decided in advance.

**Criterion 1 — Always predictive, never postdictive.**
The rule must fire *before* grokking, on every single run, without any exception. Even one run where it fires after grokking is enough to disqualify the rule — because at that point, it is not really predicting anything; it is only coincidentally correlated some of the time.

**Criterion 2 — Consistent, tight relationship to the actual event.**
The gap between "trigger epoch" and "grok epoch" should stay small, and this smallness should hold as a *proportion* of the run length, not just in absolute epoch numbers. This point is subtle: if two different runs both trigger near "epoch 2000," that alone proves nothing, because the grok epoch itself is different in each run. The trigger must move together with the grok epoch, not stay fixed at some absolute epoch.

**Criterion 3 — Clearly above the noise floor.**
The signal value at the trigger point must be clearly larger than the ordinary jitter seen during quiet training. If the "signal" at the trigger point cannot be told apart from background noise, it is not a real signal.

---

## 6. Cross-Run Evidence — What the Numbers Actually Mean

Three separate training runs were used (each run has a different random seed, so a different grok epoch comes out each time — this is expected in grokking experiments).

**Table 1 — the adopted strategy (zero-crossing trigger):**

| Run | Grok Epoch | Trigger Epoch | Lead Time | Trigger/Grok Ratio | Verdict |
|---|---|---|---|---|---|
| 1 | 5739 | 2007.8 | +3731 (37% of run) | 0.350 | Weak, but early |
| 2 | 4806 | 1688.6 | +3117 (31% of run) | 0.351 | Weak, but early |
| 3 | 3760 | 5643.8 | −1884 (fires **after**) | 1.501 | **Fails Criterion 1** |

In plain words: on Runs 1 and 2, the trigger fired quite early — nearly a third of the whole run before grokking — which is not very tight (fails Criterion 2, even though it technically satisfies Criterion 1). But on Run 3, the trigger fired 1884 epochs *after* the model had already grokked. That single failure is enough to disqualify this strategy outright, because Criterion 1 allows no exceptions.

**Table 2 — the peak-of-difference candidate (for reference only, not a working detector):**

| Run | Grok Epoch | Peak Epoch | Lead Time | Gap as % of Run |
|---|---|---|---|---|
| 1 | 5739 | 5633.5 | +105.5 | 1.05% |
| 2 | 4806 | 4764.0 | +42.0 | 0.42% |
| 3 | 3760 | 3742.5 | +17.5 | 0.18% |

This candidate is excellent on paper — it always comes before grokking, and the gap shrinks nicely as a percentage of the run, across all three runs. The catch is that finding "the peak" of a curve requires already knowing the entire future of that curve — you cannot know where a curve's highest point is until you have seen the whole curve. So this is **not a causal (live) detector**; it is only included as evidence that the real, useful signal does live somewhere near this peak, and a proper causal version of it has not yet been built or tested.

**Figures 1 and 2** in the report show Run 3 visually: the trigger (red star) fires at epoch 5644, well after test accuracy (the green line) has already reached its plateau near epoch 3760. The strong peak in the MA-of-MA difference curve sits much closer to the true grok point than the zero-crossing trigger does — this picture is exactly what Table 1 and Table 2 describe in numbers.

---

## 7. Side Finding — Test Accuracy Plateau in Run 3

This is a separate, smaller discovery, not directly about the predictor, but worth noting because examiners often like side findings — they show you pay attention to your own data.

- Run 3 never reached 100% test accuracy. It plateaued at exactly **99.5446%** from around epoch 5500 onward, and did not move at all through epoch 10,000.
- 99.5446% of 6587 test pairs works out to exactly 6557 correct — meaning **30 pairs stayed permanently wrong** for more than 4500 epochs.
- Training loss froze near 0.0000834 — an extremely small, stable value, showing the model had truly converged, not merely slowed down.
- Train accuracy stayed at a perfect 1.0 the entire time.

**Likely cause:** the model used here is deliberately minimal — a single attention head, a single layer, and importantly, **no LayerNorm** (whereas Nanda et al.'s original setup usually does include LayerNorm). Combined with the strong weight_decay of 1.0, this can leave behind a small, stable set of "hard" input pairs that the model never manages to resolve, even after it has otherwise fully converged.

This is described as a property of the model and training setup, not a bug caused by the predictor investigation itself — the training loop and model code were not touched during this work. It has been flagged as an optional thing to look into later (which exact (a, b) pairs fail, and whether the same 30 pairs fail again on other runs).

---

## 8. Conclusion and Recommendation

No L2-Norm-based rule, out of all five tried, satisfied all three criteria together, across all three runs:

- The zero-crossing trigger (5b) looked promising on two runs but **failed Criterion 1 outright** on the third — this alone rules it out as a dependable predictor.
- The fixed-threshold trigger (5a) failed on **robustness**: the size of the early "noise humps" varies too much between runs for one fixed multiplier to reliably separate real signal from noise.
- The peak-of-difference idea is the **strongest evidence found** — it passes Criteria 1 and 2 on all three runs — but it needs to be rebuilt as a causal, "rising-edge" rule before it can actually be used live. This rebuild has not been done yet.

**Important framing point for viva:** the thesis is a benchmark that *compares* nine predictors under one shared protocol — it is not a claim that all nine predictors must succeed. A carefully tested and clearly documented negative result, exactly like this one, is a legitimate and defensible row in that comparison table, **as long as the failure is shown to come from the signal itself** (which has been demonstrated here with data and criteria), **and not from some coding mistake**.

**Recommended next step:** record the L2 Norm result exactly as above, keep the peak-based causal variant flagged as "untested, not ruled out" for possible future work, and move forward to the next predictor in the evaluation order — **Dropout**.

---

## 9. Important Terms to Remember (Quick Glossary)

- **Grokking:** sudden jump in test accuracy long after train accuracy has already reached 100%.
- **L2 norm:** a single number summarising the overall size of the model's weights (square all weights, add them, take the square root).
- **Lead time:** how many epochs *before* (positive) or *after* (negative) the grok epoch a trigger fires.
- **Causal / live detector:** a rule that can decide "fire now" using only data seen *so far*, without looking into the future. A predictor must be causal to be genuinely useful during real training.
- **Non-causal candidate:** a rule (like "the peak") that needs the entire curve, including future values, to be computed — useful only as a reference or upper bound, not as a live tool.
- **Noise floor:** the normal, small up-and-down jitter seen in a signal during quiet, uneventful training, against which any real signal must stand out clearly.
- **Zero-crossing:** the exact point where a curve changes sign, from positive to negative or the other way.
- **Moving average (MA):** a smoothed version of a curve, calculated by averaging nearby points, used to reduce noise before analysis.

---

## 10. Likely Viva Questions and Model Answers

An examiner reading this report will typically probe in a few predictable directions: definitions, methodology discipline, the meaning of the numbers, and whether the "failure" is being handled honestly. Below are the questions most likely to come up, with short model answers.

**Q1. What is grokking, and why is it interesting enough to build a whole thesis around it?**
Grokking is the delayed generalisation phenomenon — the network reaches perfect training accuracy quickly, but test accuracy stays poor for a long time, and then suddenly rises to near-perfect. It is interesting because it challenges the usual assumption that generalisation improves smoothly and together with training performance; here the two are separated in time by thousands of epochs.

**Q2. Why was L2 norm chosen as the first predictor to study, out of nine?**
It is listed first in the thesis's evaluation order because it is conceptually the simplest signal to compute — a single scalar tracked over training — and prior literature has already observed a relationship between weight norm behaviour and grokking, making it a natural, low-complexity starting point before moving to more elaborate predictors like Spectral or HTSR Alpha.

**Q3. Why does the report call this a "benchmark result" and not a "blocked task" or a "failure"?**
Because the goal of the thesis is to compare nine predictors under one shared, fair protocol — not to prove that every single predictor must work. A negative result, properly tested against fixed criteria and shown to come from the nature of the signal itself, is exactly the kind of outcome a benchmark is supposed to record. Calling it "blocked" would incorrectly suggest that work has stopped due to some obstacle, when actually the investigation was completed and a conclusion was reached.

**Q4. Explain, in your own words, why the "raw rate-of-decline" strategy (strategy 1) failed.**
Because the L2 norm actually falls fastest during early memorisation — the phase before grokking — not during the grok transition itself. So a rule that fires "when decline is fast" ends up firing during ordinary early training, not near the interesting event; the signal direction is essentially the opposite of what was expected, hence "inverted."

**Q5. What exactly is the difference between the zero-crossing trigger and the peak-of-difference candidate?**
The zero-crossing trigger only needs data up to the present moment — the point where a computed difference curve crosses zero — so it can, in principle, be used live during training. The peak-of-difference candidate needs the entire curve, including epochs that have not happened yet, to know where the highest point is; so it can only be computed after training is finished, and cannot be used as a live detector without being rebuilt into a causal, forward-looking rule.

**Q6. Why did the zero-crossing trigger get rejected, even though it worked on two out of three runs?**
Because Criterion 1 was defined as "always predictive, never postdictive," with no exceptions allowed. On Run 3, the trigger fired 1884 epochs after the model had already grokked — one clear counter-example is sufficient to disqualify a rule as unreliable, since a predictor that sometimes fires after the event cannot be trusted going forward on unseen runs.

**Q7. What is the practical meaning of "lead time" and the "trigger/grok ratio"?**
Lead time is simply trigger epoch minus grok epoch, in epochs; a positive lead time means the trigger fired early (good), a negative one means it fired late (bad). The trigger/grok ratio expresses the same gap relative to how far training had progressed at the grok point, which is useful because it lets you compare consistency across runs that grok at very different absolute epochs.

**Q8. Why were three separate training runs used instead of just one?**
Because training a neural network is stochastic — different random weight initialisation and different data shuffling can shift the grok epoch considerably from run to run. Testing a candidate rule on only one run cannot show whether the rule generalises; three independent runs let the criteria be checked for consistency, not just for a single lucky (or unlucky) outcome.

**Q9. What caused the 99.5446% test accuracy plateau in Run 3, and is it a bug?**
It is explained as a likely consequence of the deliberately minimal model design — single attention head, single layer, and no LayerNorm — combined with strong weight_decay of 1.0. This combination can leave a small, stable set of "hard" pairs that the model never resolves, even after full convergence. The report is explicit that this is a property of the model and training setup, and not a bug introduced while building the predictor, since the training loop and model code were left untouched during this investigation.

**Q10. What would you do differently, or what is the recommended next step?**
The report recommends recording the L2 Norm result exactly as found — with the peak-based idea flagged as a promising but untested causal candidate for possible future work — and then moving on to the next predictor in the fixed evaluation order, Dropout, rather than spending further time tuning L2 Norm indefinitely.

**Q11. Why does weight_decay = 1.0 matter here, and why is it called "strong"?**
Weight decay is a regularisation term that pulls weights toward zero during training; a value as high as 1.0 is unusually strong compared to typical training settings (often 0.01 or smaller). This strong pull is what makes the model prefer very large weights during early memorisation and then forces it toward simpler, generalising solutions later — it is a standard technique used deliberately to induce grokking within a reasonable number of epochs.

**Q12. Why is a 30/70 train-test split used, instead of the more common 80/20?**
A small train fraction is intentional for grokking experiments — it is exactly this scarcity of training data that creates the gap between fast memorisation and slow generalisation. With a larger train split, the model might generalise almost immediately, and the grokking phenomenon itself would be far less pronounced or might not appear at all.

---

*End of notes. Keep this file alongside the original PDF report for quick revision before the viva.*
