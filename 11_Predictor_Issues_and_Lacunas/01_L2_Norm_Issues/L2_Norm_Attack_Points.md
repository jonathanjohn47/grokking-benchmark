# L2 Norm Predictor: Points an Examiner Could Attack

**Thesis:** A Unified Benchmark of Grokking Predictors in Neural Networks · Predictor 1 (L2 Norm) · Prepared 5 November 2026

This document lists weak points in the L2 Norm implementation and experiment that a thesis examiner could raise. Each point says what was done and why it could be challenged. Solutions are deliberately not included. Points are ordered from most to least serious.

**Status key:** ❌ = open, ✅ = resolved. Replace ❌ with ✅ as each point is fixed.

**Progress:** 0 / 21 issues resolved, 0 / 4 unverified items checked.

---

## Big worries

### ❌ 1. L2 was never scored under the frozen protocol
The official verdict code only scores the rules listed in `PRIMARY_SIGNALS`: HTSR, Weight-PCA, Higher-MI and Commutator Defect. L2 isn't on that list, so no results file holds an L2 PASS/FAIL. Its "closed, negative" label comes from August runs on the old single-head and four-head models, before the Nanda-Unified setup existed. When I recomputed it myself, MA crossover gave $\rho = 0.70$ and a mean relative lead of $0.989$, and MA-of-MA zero crossing gave $\rho = 0.00$. So both fail, but nothing in the project records that officially.

### ❌ 2. You tested a claim the paper never made
Nanda et al. only use the sum of squared weights to describe what happens. They say it "decreases smoothly during circuit formation and more sharply during cleanup", and the sharp fall happens at the same time as grokking. They never claim it gives early warning, and their conclusion says they can't predict the transition ahead of time. Your own notes say the weight-norm-as-predictor idea really comes from Liu et al. ("Omnigrok"), and that paper isn't in `04_Literature`. An examiner can ask whose prediction you are actually testing.

### ❌ 3. The MA-crossover event comes from a setting you chose
The crossover fires at epochs 113, 113, 120, 131 and 137. The rule ignores everything before `skip_epochs` = 100, so it fires almost as soon as it's allowed to. That's early memorisation, about 100 times earlier than grok. Its timing comes from the cutoff you picked, not from grokking, and the "5/5 before grok" result means nothing.

### ❌ 4. The MA-of-MA event runs on a fixed clock
The zero crossing fires between epoch 1767 and 1814 in every seed (spread about 1%), while grok ranges from 6921 to 23747. Its $\rho$ with grok is $0.00$. It marks a fixed point in the training schedule, not anything about when grokking happens.

### ❌ 5. The only rule that passes looks into the future
"Halfway down from peak" passes the thresholds exactly: lead fraction $0.8$, $\rho = 0.9$, mean relative lead $0.162$. But its target is $\text{peak} - 0.5\,(\text{peak} - y_{\text{final}})$, and $y_{\text{final}}$ is the value at epoch 40,000. No live monitor could know that. It was also added in the same commit as the results (`baa1167`), the write-up calls the 50% level "my own illustrative choice (arbitrary)", and it passes only at the edges: $\rho$ equals the cutoff, and 4/5 is the lowest lead fraction that passes. It has to be explicitly excluded, and the thesis has to explain why.

### ❌ 6. The "causal" smoothing also uses the future
Both implemented rules smooth the curve with `uniform_filter1d`, which is a centred moving average. Each smoothed value therefore includes epochs that come after it, and the edges are padded by copying values. The code comment calls the MA-of-MA rule "causal", but it isn't. It would behave differently in a truly live setting.

### ❌ 7. The rules were picked after looking at the curves
Git history and `context.md` show the rule changing several times: a rate threshold, then an inflection point, then MA crossover, then an MA-of-MA threshold, which was abandoned in favour of the zero crossing (commit `2f8cd3b`). These choices were made by looking at plots on the old $p = 97$ models, and `skip_epochs` was lowered from 200 to 100 "for safety" after inspecting the curves. An examiner can say the rule was tuned to the data, and on a different model at that.

---

## Moderate worries

### ❌ 8. Weight decay forces the fall
Weight decay is set to $\lambda = 1.0$. Once training loss is near zero, weight decay is the main thing changing the weights, so $\sum w^2$ will fall whether or not the model is about to generalise. $\lambda$ was never varied, so you can't separate "falls before grokking" from "falls because of the optimiser setting".

### ❌ 9. The model isn't exactly Nanda's
The shared transformer has query, key and value matrices but no attention output matrix $W_O$, which Nanda's model has. So $\sum w^2$ is a sum over a different set of weights from his. The config calls the setup a "faithful replication", which isn't quite true.

### ❌ 10. Every weight is lumped together
The signal is $\sum w^2$ over all parameters: both embedding tables, Q/K/V, the MLP including its biases, and the output head. The per-module split is saved (`per_module_sum_w2.npy`), but no rule uses it. A change in one part of the model can be hidden by the rest, and an examiner can ask which weights are supposed to carry the signal.

### ❌ 11. Window sizes were never tested
Fast/slow windows of 50/200, the MA-of-MA window of 20, `skip_epochs` = 100 and `quiet_epoch_cutoff` = 90 are all your own choices, and the config says so ("Not from a paper"). Nobody checked whether other values change the result. Point 3 shows that at least one of them decides it.

### ❌ 12. A near-pass that shows how fragile the cutoffs are
Using the peak of $\sum w^2$ as the event gives $\rho = 0.9$ and a mean relative lead of $0.904$. That fails only because $0.904 > 0.9$. If verdicts can flip on the third decimal, an examiner can question how stable the protocol is for L2.

### ❌ 13. Five seeds can't tell weak from useless
With $n = 5$, $\rho = 0.70$ isn't significant, and only $\rho \ge 0.9$ is. Seed 4 (grok 23747) is the extreme point for everything, so it drives the Pearson values.

### ❌ 14. No comparison runs
There's no run without grokking, and no run with weight decay turned off or changed, to show what $\sum w^2$ does when grokking doesn't happen. Without that, the falling curve can't be credited to grokking.

---

## Small slips

### ❌ 15. "Before grok" counts the wrong thing
The summary table's `seeds_before_grok` counts leads above 0 epochs instead of the protocol's 100-epoch margin. That's why `context.md` says the halfway rule is 5/5. It's really 4/5, because seed 2 leads by only 13 epochs.

### ❌ 16. The halfway rule can crash
`l2_decline_epoch` takes `idx[0]` without checking for an empty result (line 102), so a run that never falls halfway would crash the analysis. This corrects the earlier audit: `put()` now handles None safely, so a None from the implemented rules is fine.

### ❌ 17. Leftover code that does nothing
`detect_l2_norm_drop` and `detect_inflection` (which return None at lines 71 and 94) and `detect_ma_of_ma_trigger` are never called by the benchmark. `noise_floor` is computed and saved but doesn't affect any decision. This makes it unclear which rule is "the" L2 predictor.

### ❌ 18. The 30% split is hard-coded
`get_dataloaders` uses `0.3` directly, so `train_fraction` in the config only sets the batch size and can't change the split.

### ❌ 19. Old numbers in `context.md`
The 4 September entry gives grok epochs 12686, 7721, 11376, 12678 and 25505, crossover at 101–144 and zero crossing at 1774–1834. The saved files now say 12987, 6921, 9510, 11019, 23747, then 113–137 and 1767–1814. The older numbers come from an earlier run and were never marked as out of date.

### ❌ 20. Documents disagree on run counts and order
The config says `per_predictor_min_runs: 3` while the benchmark uses 5 seeds, and `AGENTS.md` still lists the old nine-predictor order.

### ❌ 21. Spearman ties are mishandled
`rank_corr` ranks with `argsort`, which breaks ties arbitrarily. The L2 events have no ties, so L2's numbers aren't affected, but the function is wrong in general.

---

## What checks out

I recomputed every L2 number from the saved files, and they match `predictor_events.json` and the 25 September table in `context.md`. The grok epochs match `summary.json`. The saved L2 norm squared equals the saved $\sum w^2$ (relative error $< 10^{-6}$). Because the MAs are taken on a log scale, and $\log\sqrt{x} = \tfrac12 \log x$, using the norm instead of $\sum w^2$ doesn't change any event. During training the seed is set before the split is made. The scatter-plot marker bug doesn't affect L2: L2 comes first in the list, and the current file has 10 markers for 10 names.

---

## Not verified

- ❌ Whether Nanda's $\sum w^2$ includes embeddings and biases, and how his magnitudes compare (his Fig. 7 axis tops out around 3,500, while the peaks here are about 4,800–5,000).
- ❌ Whether Nanda's output layer has 113 classes, compared with 114 here (the "=" token is included).
- ❌ Whether the saved results were made with exactly the current code (no commit hash is stored with the run).
- ❌ What Liu et al. / Omnigrok actually claim, since the paper isn't in the library.
