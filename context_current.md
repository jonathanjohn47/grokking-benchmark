# Current State of the Grokking Predictor Benchmark (`context_current.md`)

This document summarizes the active, in-force state of the thesis research project *"A Unified Benchmark of Grokking Predictors in Neural Networks"*, conducted by Jonathan John under the supervision of Prof. Dr.-Ing. Sheikh Faisal Rashid (AI, Berlin campus, IU Internationale Hochschule). All superseded architectures, temporary debugging trials, abandoned trigger heuristics, and legacy data configurations have been stripped out. Each decision is accompanied by its underlying scientific or methodological rationale and citation to the original research log.

---

## 1. Frozen Experimental Setup

The experimental substrate is configured in `06_Code/configs/nanda_unified.yaml` and executed via `06_Code/scripts/run_nanda_benchmark.py`. It is the Nanda et al. (2023) modular-addition task. It does **not** strictly replicate Nanda Section 3, and it does **not** match that section exactly. The code was aligned with Nanda's released training code in November 2026. The paragraph "Training setup (retrain, Nov 2026)" below is the description of that code. Every \(G_i\) and every predictor result printed later in this file was computed on the **old** setup (no \(W_O\), no causal mask, no warmup, init \(0.8/\sqrt{d_{\mathrm{model}}}\), checkpoints every 50 epochs, primary grok cut 0.9). Those numbers must be recomputed after the 40,000-epoch retrain. They are not results of the new setup.

### Training setup (retrain, Nov 2026)

1. **\(W_O\)**. Attention output projection, shape \(128 \times 128\), no bias, applied after the four heads are concatenated and before the residual add. Source: Nanda's released `transformers.py` (`out = einsum('df,bqf->bqd', W_O, z_flat)`).
2. **Causal mask**. Lower-triangular mask on the \(3 \times 3\) attention scores. The upper triangle is set with `tril` and \(-10^{10}\), then the scores are divided by \(\sqrt{d_{\mathrm{head}}}\), then softmax. Source: Nanda's released `Attention.forward`.
3. **Learning-rate warmup**. `LambdaLR` with factor \(\min(\mathrm{step}/10, 1)\), and `scheduler.step()` after each `optimizer.step()`. Yaml `warmup_steps: 10`. Source: Nanda's released `Trainer`; Power et al. 2022 Appendix A also warms up for 10 updates.
4. **Initialisation**. \(W_E, W_{\mathrm{pos}}, W_Q, W_K, W_V, W_O, W_{\mathrm{in}}, W_{\mathrm{out}}\) drawn as `randn` divided by \(\sqrt{d_{\mathrm{model}}}\) (standard deviation \(1/\sqrt{128}\)). \(W_U\) drawn as `randn` divided by \(\sqrt{d_{\mathrm{vocab}}}\) with \(d_{\mathrm{vocab}} = 114\). \(b_{\mathrm{in}}\) and \(b_{\mathrm{out}}\) are zeros. No attention, embedding, or unembedding bias. The old \(0.8/\sqrt{d_{\mathrm{model}}}\) scheme is not used. Source: Nanda's released constructors. The paper text does not state this scale.
5. **Init checkpoint**. `model_init.pt` is written before any training step. Source: Nanda's `initial_save_if_appropriate` (`init.pth`).
6. **Loss**. Float32 cross entropy on the final position, over all 114 logits, on the model device: `logits.to(torch.float32)`, then `log_softmax`, then minus the mean of the gathered label log-probability. This is the reduction in `helpers.cross_entropy_high_precision` as executed. The comment in that file says float64, but the code casts to float32. Gradients flow back through that cast. The unembedding stays 114 wide (Nanda's code uses \(d_{\mathrm{vocab}} = p+1\); the paper's footnote 2 is the same vocabulary). Source: Nanda's `full_loss` and `cross_entropy_high_precision`.
7. **What is logged each epoch**. Train loss, test loss, train accuracy, and test accuracy are computed **before** the optimizer step, every epoch. Source: Nanda's `do_a_training_step` computes both losses before `optimizer.step()`. Accuracy at that same pre-step point is our addition so the benchmark can still read a grok epoch.
8. **Train fraction**. `get_dataloaders` reads `task.train_fraction` from the yaml. It does not hardcode 0.3. Source: the yaml, which is set to 0.3 to match Nanda and Power.
9. **Data split**. `pairs = [(i, j, p) for i in range(p) for j in range(p)]`, then `random.seed(seed)`, `random.shuffle(pairs)`, train = the first `int(0.3 * len(pairs))` pairs, test = the rest. One full batch. Source: Nanda's `gen_train_test`.
10. **Checkpoints**. Init, then every 100 epochs after that epoch's optimizer step (`model_epoch_0.pt`, `model_epoch_100.pt`, …, `model_epoch_39900.pt`). This replaces the old every-50 schedule (801 files, including epoch 39999). Source: Nanda's `save_every = 100`. Analysis and predictor code that assumed a 50-epoch spacing or 801 checkpoints now takes the spacing from this schedule. The 250-epoch eval grid is no longer a subset of the saved epochs.
11. **No dropout modules in the shared model**. `dropout1` and `dropout2` (rate 0) were removed. Nanda's model has none. This is not the Dropout predictor. The predictor still measures stochastic forward passes; it attaches its own hooks on `attn_out` and `mlp_out` only during that measurement.
12. **Left as they already were, because they already match Nanda's released code**. 40,000 epochs (Nanda's paper). Seeds 0, 1, 2, 3, 4. AdamW, learning rate \(10^{-3}\), weight decay 1.0, betas \((0.9, 0.98)\), Adam `eps` left at the library default, weight decay on all parameters, no gradient clipping.
13. **Primary grok cut**. \(G_i\) is the first epoch where test accuracy is strictly above 0.99. Source: Power et al. 2022 Section 3.1, which times generalization at 99%. The cuts 0.9 and 0.95 are still computed and reported as secondary cuts. The lead rule and the analysis use this 0.99 \(G_i\).

**Exception — seed.** One seed in \(\{0,1,2,3,4\}\) sets both the data split and the initial weights (`torch.manual_seed(seed)` and `random.seed(seed)`). Nanda's code seeds only the split. We keep the seed's influence on the weights as well because we want the experiment to be repeatable.

**Exception — hardware.** Training targets Apple MPS, not Nanda's CUDA GPU. The loss stays on the model device in float32, matching the executed cast in Nanda's `helpers.py` (the comment says float64; the code casts to float32). MPS has no float64, and this loss no longer needs one.

Scripts updated for the new checkpoint spacing or the new \(G_i\): `run_nanda_benchmark.py` (save every 100, init checkpoint, primary threshold 0.99), `analyze_nanda_unified.py` (grok wording 0.99), `analyze_threshold.py`, `debug_detection.py`, `plot_commutator_defect_stage6.py`, `plot_results_four_head.py`, `train_four_head.py`. `htsr_alpha.py` and `weight_pca.py` now include `attn_out` (\(W_O\)). `l2_norm.py` records an `attention_out` group. `dropout.py` no longer sets `dropout1` / `dropout2`; the measurement (rates, sample count, variance of accuracy) is unchanged. The Dropout predictor's own rule was not replaced.

- **Task**: Modular addition $(a + b) \pmod{113}$.
  - *Modulus*: $p = 113$.
  - *Input Representation*: Sequence of three tokens `[a, b, =]`.
  - *Vocabulary*: $114$ tokens total (integers $0 \dots 112$ represent numerical tokens; token ID $113$ represents the '=' operator). The target label is $(a + b) \pmod{113} \in \{0, \dots, 112\}$, evaluated at the '=' position (sequence index 2).
  - *Dataset Split*: Random split of $30\%$ training data ($N = \lfloor 0.3 \times 113^2 \rfloor = 3830$ pairs) and $70\%$ validation/test data ($8939$ pairs).
  - *Reason & Reference*: Modulus $p = 113$ matches Nanda et al. (2023) mainline. The earlier p=97 setup (Power et al. 2022) was replaced on 2026-09-04 because it sat closer to the critical weight-decay boundary, exhibiting severe optimization transients that confounded weight-space predictor signals (*2026-09-04: DECISION: whole benchmark moved to Nanda-Unified substrate*).
- **Model Architecture**: One-layer Transformer with four attention heads (`TransformerFourHead` in `06_Code/src/models/transformer_four_head.py`).
  - *Dimensions*: Model embedding dimension $d_{\text{model}} = 128$; attention head dimension $d_{\text{head}} = d_{\text{model}} / 4 = 32$; MLP hidden layer dimension $d_{\text{mlp}} = 512$ ($4 \times d_{\text{model}}$).
  - *Components*: Learned positional embeddings, query/key/value, $W_O$, causal mask, residuals, ReLU MLP, untied unembedding, no dropout modules. Details are in "Training setup (retrain, Nov 2026)".
  - *Normalization*: **No LayerNorm**.
  - *Biases*: Linear projections for queries, keys, values, and output head are unbiased (`bias=False`). The MLP linear layers retain learnable biases $b_{\text{in}}$ and $b_{\text{out}}$, initialized to zero.
  - *Reason & Reference*: Does not match Nanda Section 3 exactly. See Training setup (retrain, Nov 2026). No LayerNorm remains correct. A previous assumption that Nanda et al. used LayerNorm was audited on 2026-09-03 and proven factually wrong; an isolated shadow experiment confirmed that adding LayerNorm under high weight decay destabilized training with recurrent collapses (*2026-08-17: Shadow LayerNorm experiment*; *2026-09-03: Nanda paper read in full; CORRECTION on record — Nanda does NOT use LayerNorm*).
- **Weight Initialisation**:
  - November 2026 code: `randn/sqrt(d_model)` for every matrix except $W_U$, which is `randn/sqrt(d_vocab)`. The $0.8/\sqrt{d_{\mathrm{model}}}$ numbers in this bullet's history belong to the old saved run only. See item 4 of Training setup (retrain, Nov 2026).
- **Optimizer & Hyperparameters**:
  - *Optimizer*: AdamW with decoupled weight decay.
  - *Hyperparameters*: Learning rate $\eta = 10^{-3}$, weight decay $\lambda = 1.0$, betas $\beta = (0.9, 0.98)$, $\epsilon = 10^{-8}$.
  - *Batching & Horizon*: Full-batch gradient descent (batch size $3830$), linear warmup of 10 updates, trained for $40,000$ epochs.
  - *Reason & Reference*: Full-batch training with $\lambda = 1.0$ is the exact grokking configuration from Nanda et al. (2023). PyTorch's default $\beta_2 = 0.999$ created a long memory buffer that induced stale-moment slingshot spikes; setting $\beta_2 = 0.98$ matches Nanda et al. and eliminates slingshot artifacts (*2026-09-03: WHY the L2 behaviour differs*; *2026-09-04: Nanda-Unified protocol*).
- **Hardware Target**: PyTorch on Apple Silicon MPS (`mps`), with automatic CPU fallback for operations lacking MPS kernel support (such as float64 operations and specific SVD/eigh routines) (*2026-07-10: MPS device support*; *2026-09-02: Training now runs on MPS*).
- **Checkpoint Schedule**:
  - *Saving Schedule (code, Nov 2026)*: `model_init.pt`, then every 100 epochs (`model_epoch_0.pt` through `model_epoch_39900.pt`, 400 files). The 801-file every-50 grid is the old saved run and has not been overwritten.
  - *Evaluation Schedule*: Predictors are evaluated on a uniform linear grid spaced every $100$ epochs plus the final epoch ($39,999$): $k \in \{0, 100, 200, \dots, 39900, 39999\}$, producing $401$ evaluated snapshots per seed. Fallback grids of every $200$ epochs ($201$ snapshots) and every $250$ epochs ($161$ snapshots) are natively supported.
  - *Reason & Reference*: Replaced the arbitrary 24 log-spaced schedule. The save spacing $\Delta k_{\text{save}} = 50$ is mathematically forced by $\gcd(100, 200, 250) = 50$, allowing evaluation on $100$, $200$, or $250$ epoch grids without retraining. Across the baseline runs, the narrowest grokking transition (test accuracy $10\% \to 90\%$) was $1,570$ epochs (Seed 1), and earliest grokking occurred at epoch $6,988$ (Seed 1). Spacing $\Delta k = 100$ yields $15.7$ samples across the narrowest transition and a maximum ratio error of $1.43\%$, easily satisfying the project design criteria of $\ge 6$ transition samples and $\le 4\%$ ratio error (*2026-09-25: Switch from 24 log-spaced to save-every-100*; *2026-09-25: Shift from 100 to 50 to support 250 grid (GCD justification)*).
- **Random Seeds & Grokking Definition**:
  - Exactly 5 deterministic seeds: `seed = 0, 1, 2, 3, 4`.
  - *Reason for 5 Seeds*: Matches Nanda et al. (2023). In addition, $n = 5$ is the minimal sample size for Spearman's rank correlation $\rho$ to achieve one-sided statistical significance ($p < 0.05$, where $P(\rho \ge 0.9) = 5/120 \approx 0.0417$) while tolerating up to 1 non-firing seed ($n_{\text{firing}} \ge 4$, where $P(\rho = 1.0) = 1/24 \approx 0.0417$). Seed counts below 5 cannot achieve statistical significance (*2026-09-01: Methodology: Justifying 3-run protocol*; *2026-11-09: Why the benchmark uses 5 seeds*).
  - *Grokking Definition (code, Nov 2026)*: $G_i$ is the first epoch where test accuracy strictly exceeds $0.99$. Cuts $0.90$ and $0.95$ are secondary. The list below is the **old** primary cut of $0.90$ from the saved run and must be recomputed after the retrain.
  - *Old-setup epochs at the old $0.90$ cut (not the new $G_i$)*:
    - Seed 0: $G_0 = 12987$ (accuracy $>0.95$: 13133; $>0.99$: 13457)
    - Seed 1: $G_1 = 6921$ (accuracy $>0.95$: 7031; $>0.99$: 7237)
    - Seed 2: $G_2 = 9510$ (accuracy $>0.95$: 9622; $>0.99$: 9852)
    - Seed 3: $G_3 = 11019$ (accuracy $>0.95$: 11069; $>0.99$: 11138)
    - Seed 4: $G_4 = 23747$ (accuracy $>0.95$: 23803; $>0.99$: 23913)
    - *Aggregate*: Mean grokking epoch $\bar{G} = 12836.8$, standard deviation $\sigma_G = 5803.9$. All 5 seeds achieve $100\%$ training and test accuracy. Sustained limit-cycle count is zero ($0/5$) (*2026-09-25: 5-seed Nanda-Unified run finished*; *2026-11-11: Official numbers for the L2 viva-ready answers*).

---

## 2. Benchmark Rules & Evaluation Protocol

The evaluation protocol scores whether an early-warning predictor can forecast the onset of grokking across independent seeds.

- **Scope Decision (No Online Detection Engineering)**:
  - The study evaluates established, published grokking predictors according to their original published methodologies.
  - It strictly excludes researcher-engineered online detection rules, arbitrary moving thresholds, percentage-of-change heuristics, or predictor-specific parameter tuning designed to force a pass.
  - *Reason & Reference*: Designing custom detection rules would convert the study from a benchmark of literature claims into a method-development exercise with unverified assumptions (*2026-09-25: Scope decision — no online detection rule*).
  - *Predictors with no published detection rule*: Where the source papers report only an observation and give no rule that turns the signal into a firing epoch, the benchmark uses the simplest rule that directly encodes the published observation, fixes it before scoring, and labels it openly as the project's own operationalisation. This is not a new detection rule in the sense excluded above, because no published rule exists to replace. The other predictors are handled the same way (extreme-value or onset events chosen by the project). (*2026-11-12: L2 open point 2 closed*).
- **The Lead Test**:
  - A predictor event $P_i$ on seed $i$ is defined as leading if:
    $$P_i < c \cdot G_i$$
  - *Default Parameter*: $c = 0.8$, with sensitivity checks at $c = 0.7$ and $c = 0.9$.
  - *Reason & Reference*: Lead times scale proportionally with grokking time $G_i$ (Xu 2026, Golwala 2026). A constant margin (such as 100 epochs) represents a negligible fraction of late-grokking runs. Setting $c = 0.8$ demands an advance warning of at least $20\%$ of the grokking time. Furthermore, $c \le 0.94$ is bounded above by Nanda et al.'s cleanup phase starting at $\approx 0.94 G$; alarms firing after $0.94 G$ occur during accuracy takeoff. Empirical verdicts for all predictors are invariant across $c \in \{0.7, 0.8, 0.9\}$ (*2026-11-10: Fractional lead rule (c sweep)*; *2026-11-10: Justification for the lead-fraction values c = 0.7, 0.8, 0.9*).
- **PASS / FAIL Criteria**:
  A candidate predictor row passes the frozen benchmark protocol if and only if all four conditions are satisfied:
  1. *Minimum Firing Count*: The predictor must fire on at least 4 of the 5 seeds ($n_{\text{firing}} \ge 4$). Non-firing seeds are recorded as `None` and dropped prior to calculating correlation and relative lead. If fewer than 4 seeds fire, the predictor fails immediately.
  2. *Lead Fraction*:
     $$\text{lead\_frac} = \frac{1}{n_{\text{firing}}} \sum_{i \in \text{firing}} \mathbb{I}(P_i < c \cdot G_i) \ge 0.80$$
     At least $80\%$ of firing seeds must strictly lead. Anchored to Cohen's $0.80$ statistical power convention, permitting at most 1 failure across 5 seeds (*2026-11-10: Justification for lead_frac >= 0.8*).
  3. *Rank Correlation*:
     $$\rho(P_i, G_i) \ge \rho_{\text{thresh}} = 0.8999999999999998 \approx 0.90$$
     Spearman's rank correlation $\rho$ between event times and grokking times must be $\ge 0.90$. The threshold was calibrated as the 99th percentile of $10,000$ trials of an uninformative random predictor $P_i \sim \text{Uniform}[0, T_{\max}]$, capping the false-positive rate at $1\%$ (*2026-09-29: Controlled Experiment: Synthetic-Control Validation*; *2026-11-11: L2 open attack point 12 closed: attainable Spearman rho values for n = 5*).
     $\rho$ is set to $0$ only when all firing $P_i$ are exactly equal, because Spearman $\rho$ is undefined when the spread is exactly zero. No numerical tolerance is used. Fewer than 3 firing points, or a NaN from `spearmanr`, also store $\rho = 0$ (*2026-11-12: Constant-signal rho is an exact equality check*).
  4. *Mean Relative Lead*:
     $$\bar{r} = \frac{1}{n_{\text{firing}}} \sum_{i \in \text{firing}} \frac{G_i - P_i}{G_i} \in [0.05, 0.90]$$
     The average relative lead must be between $5\%$ and $90\%$ of grokking time. Negative relative leads remain signed in the average. Alarms firing earlier than $0.10 G_i$ ($\bar{r} > 0.90$) fail for being decoupled initialisation transients.
  - *Predictor-Level Verdict*: A predictor passes if and only if at least one pre-declared official row passes all criteria. Rows designated as "exploratory" are reported for diagnostic completeness but cannot qualify a predictor as passing (*2026-11-10: Official scorer now verdicts all eight predictors*).

---

## 3. Predictor Benchmark Evaluations

Predictors are evaluated in the frozen order established in `CLAUDE.md`. Official scoring is executed by `06_Code/scripts/analyze_nanda_unified.py` and output to `08_Experiments/results/nanda_unified/analysis/frozen_protocol_verdicts.json`.

```
========================================================================================================
PREDICTOR STATUS SUMMARY (OFFICIAL FROZEN PROTOCOL, c = 0.8)
========================================================================================================
1. L2 Norm            : FAIL (0 of 2 official pre-declared rows passing)
2. Dropout            : FAIL (0 of 3 official pre-declared rows passing)
3. Spectral           : FAIL (0 of 5 official pre-declared rows passing)
4. AGE                : FAIL (0 of 3 official pre-declared rows passing)
5. HTSR Alpha         : FAIL (0 of 1 official pre-declared rows passing)
6. Weight-PCA         : FAIL (0 of 1 official pre-declared rows passing)
7. Higher-MI          : FAIL (0 of 1 official pre-declared rows passing; 2 exploratory rows reported)
8. Commutator Defect  : FAIL (0 of 1 official pre-declared rows passing; 1 secondary unscored row)
--------------------------------------------------------------------------------------------------------
Correlation Traps     : REMOVED from line-up on 2026-09-30 (detects post-grok anti-grokking collapse)
========================================================================================================
```

### Predictor 1: L2 Norm
- **Paper**: Nanda et al. (2023, Figure 7) / Liu et al. (2022, *Towards Understanding Grokking* / *Omnigrok*).
- **Signal & Firing Rules**:
  - Evaluates the total L2 norm of model parameters $\|w\|_2 = \sqrt{\sum w_i^2}$.
  - *Rule 1 (MA Crossover)*: Fast moving average (window 50) crossing above slow moving average (window 200) on $\log_{10}\|w\|_2$, skipping the first 100 epochs. Project's own choice.
  - *Rule 2 (MA-of-MA Zero-Crossing)*: Fast MA (window 20) of slow MA crossing below slow MA, skipping the first 100 epochs. Project's own choice.
  - *(Discarded Rule)*: The retrospective "sum of squared weights half-way down from peak" was formally removed on 2026-11-11 because it requires the run's final terminal value, making it invalid as an early-warning rule (*2026-11-11: L2 half-way-down-from-peak row removed*).
  - *Why the project chose these rules*: No published paper gives a detection rule for the weight norm as a grokking predictor. Nanda et al. (2023) only describe the norm and state that grokking timing cannot be predicted ex ante. Liu et al. (2022, Omnigrok) give a delay law and a norm-control method, not a detection rule. A literature search on 2026-11-12 found no such rule. Khanh et al. (2026, arXiv:2606.13753, "The Weight Norm Sets the Grokking Timescale") define the critical norm $\|W\|_c$ only as the norm at the grokking step, which is post hoc. So there is no published rule that L2 skipped. The two rules are the simplest encodings of the published observation that the norm rises during memorisation and falls before grokking. L2's FAIL means this published observation, turned into a rule, gives no early warning (*2026-11-12: L2 open point 2 closed*).
  - *Why weight decay is not varied (no comparison runs)*: Every run uses the frozen weight decay λ = 1.0, and once training loss is near zero weight decay shrinks the weights whether or not grokking follows, so a movement in the norm cannot by itself be credited to grokking. This confound would matter only if L2 had passed, because only a pass would need the norm to move because of grokking. L2 fails: its two triggers fire at about epoch 113 and about epoch 1800 on every seed while $G_i$ ranges from 6,921 to 23,747 ($\rho = 0.0$ for the MA-of-MA row), and this document already attributes them to early weight-decay settling. The confound therefore supports the negative verdict rather than weakening it. λ = 1.0 is fixed because it is the exact Nanda et al. (2023) configuration, and changing λ changes when grokking happens at all (Liu et al. 2022, Omnigrok delay law), so a λ sweep or a no-grokking control would be a different experiment outside the benchmark's scope (*2026-11-12: L2 open point 3 closed*).
- **Official Results ($c = 0.8$)**:
  - *MA Crossover*: $P_i = [113.41, 113.20, 120.12, 130.61, 136.80]$ (mean $122.8$). Firing count: $5/5$. $\text{lead\_frac} = 1.0000$, $\rho = 0.7000$, $\bar{r} = 0.9889$. **FAIL** (fails $\rho \ge 0.90$ and $\bar{r} \le 0.90$).
  - *MA-of-MA Zero-Crossing*: $P_i = [1802.2, 1767.2, 1814.2, 1810.8, 1801.3]$ (mean $1799.1$). Firing count: $5/5$. $\text{lead\_frac} = 1.0000$, $\rho = 0.0000$, $\bar{r} = 0.8350$. **FAIL** (fails $\rho \ge 0.90$).
- **Verdict**: **CLOSED, NEGATIVE** (0 of 2 rows passing).
- **Scientific Reason & Notes**: L2 norm triggers fire exclusively on early optimization transients ($\sim 113$ and $\sim 1800$) associated with initial weight-decay settling. The trigger epochs remain invariant while grokking varies between $6,921$ and $23,747$ epochs ($\rho = 0.0000$). Nanda et al. presented weight decay descriptively during cleanup and explicitly noted that grokking timing cannot be predicted ex-ante (*2026-09-04: Discussion: why L2 Norm cannot be used*; *2026-09-17: Defense note: four negative closes justified*).

### Predictor 2: Dropout
- **Paper**: Ahmed Salah & David Yevick (2025/2026, arXiv:2507.11645 / *Neural Processing Letters*).
- **Signal & Firing Rules**:
  - Tracks the variance of test accuracy across $k = 100$ independent stochastic forward passes under evaluation-time dropout ($p = 0.5$ on attention and MLP dropout layers). Firing rule: Global maximum of test-accuracy variance across checkpoints. Derived from Salah & Yevick (adapted to $k=100, p=0.5$ based on the paper's Dropout Robustness Curve).
  - Also tracked: Retrospective 10% and 50% rise points of the variance curve.
- **Official Results ($c = 0.8$)**:
  - *Variance Peak*: $P_i = [24200, 25400, 16700, 18800, 39000]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = 0.4000$, $\bar{r} = -1.1276$. **FAIL** (fails lead, correlation, and relative lead).
  - *Variance 10% Rise*: $P_i = [11500, 5800, 9000, 9100, 22200]$. $\text{lead\_frac} = 0.0000$, $\rho = 1.0000$, $\bar{r} = 0.1139$. **FAIL** (fails lead fraction; all $P_i \ge 0.8 G_i$).
  - *Variance 50% Rise*: $P_i = [13700, 7200, 9700, 11300, 23900]$. $\text{lead\_frac} = 0.0000$, $\rho = 1.0000$, $\bar{r} = -0.0294$. **FAIL**.
- **Verdict**: **CLOSED, NEGATIVE** (0 of 3 rows passing).
- **Scientific Reason & Notes**: The variance peak occurs post-grokking in all seeds (peak/grok ratios $1.64 - 3.67 \times G_i$). Robustness to stochastic dropout emerges as a mechanistic consequence of redundant Fourier circuit consolidation, not as a precursor (*2026-09-05: Aggregate Analysis & Dropout Closure Decision*; *2026-09-17: Defense note*).

### Predictor 3: Spectral
- **Paper**: Abdulkadir Canatar, Blake Bordelon, Cengiz Pehlevan (2021, *Nature Communications*).
- **Signal & Firing Rules**:
  - Computes task-model alignment on the centered representation Gram kernel $K = \Phi_c \Phi_c^T \in \mathbb{R}^{3830 \times 3830}$ at the '=' token position. Kernel eigenvalues $\eta_k$ are obtained via `eigh`, and target task power $p_k$ is computed from one-hot training labels.
  - *Rule 1*: Global minimum of $k_{90}$ (number of eigenmodes required to capture $90\%$ of task power).
  - *Rule 2*: Global maximum of alignment score $\sum_k \bar{\eta}_k p_k$. Faithful to Canatar et al. (Also tracked: retrospective 10% and 50% alignment rise, and 50% $k_{90}$ fall).
- **Official Results ($c = 0.8$)**:
  - *$k_{90}$ Minimum*: $P_i = [29700, 30700, 14100, 38000, 33900]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = 0.3000$, $\bar{r} = -1.6163$. **FAIL**.
  - *Alignment Maximum*: $P_i = [31600, 37800, 29300, 37400, 32000]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = -0.3000$, $\bar{r} = -2.1435$. **FAIL**.
  - *Alignment 10% Rise*: $P_i = [8500, 4100, 6400, 2900, 19600]$. $\text{lead\_frac} = 0.8000$, $\rho = 0.7000$, $\bar{r} = 0.3983$. **FAIL** (fails $\rho \ge 0.90$).
  - *Alignment 50% Rise*: $P_i = [12500, 6300, 9500, 10200, 23200]$. $\text{lead\_frac} = 0.0000$, $\rho = 1.0000$, $\bar{r} = 0.0451$. **FAIL**.
  - *$k_{90}$ 50% Fall*: $P_i = [13800, 7500, 7500, 11800, 25100]$. $\text{lead\_frac} = 0.2000$, $\rho = 0.9747$, $\bar{r} = -0.0126$. **FAIL**.
- **Verdict**: **CLOSED, NEGATIVE** (0 of 5 rows passing).
- **Scientific Reason & Notes**: Canatar et al.'s theory applies to the lazy/kernel regime where kernel eigenspectra remain nearly static. Transformer grokking is in the feature-learning regime. Furthermore, because $\text{rank}(K) \le 128 \ll 3830$, at least $3702$ eigenvalues are identically zero; $k_{90} \approx 3400$ falls entirely within the arbitrary numerical null space and cannot reflect true feature rank collapse (*2026-09-06: Spectral predictor REBUILT*; *2026-09-27: Correction note: Spectral predictor*).

### Predictor 4: AGE (Adaptive Grokking Epoch via Neural Collapse)
- **Paper**: Vardan Papyan, X.Y. Han, David L. Donoho (2020, *PNAS*); mechanism from Beaglehole et al. (2024) / Mallinar et al. (2024).
- **Signal & Firing Rules**:
  - Measures Neural Collapse variability collapse:
    $$\text{NC1} = \frac{\text{Tr}(\Sigma_W)}{\text{Tr}(\Sigma_B)}$$
    where $\Sigma_W$ is within-class covariance and $\Sigma_B$ is between-class covariance of penultimate representations at the '=' position across the $C = 113$ classes. Firing rule: Global minimum of NC1 across checkpoints. From Papyan et al. (Also tracked: retrospective 10% and 50% fall of $\log_{10}\text{NC1}$).
- **Official Results ($c = 0.8$)**:
  - *NC1 Minimum*: $P_i = [31500, 37800, 29300, 37400, 30100]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = -0.4000$, $\bar{r} = -2.1260$. **FAIL**.
  - *$\log_{10}\text{NC1}$ 10% Fall*: $P_i = [8200, 3000, 4700, 2500, 19000]$. $\text{lead\_frac} = 0.8000$, $\rho = 0.7000$, $\bar{r} = 0.4828$. **FAIL** (fails $\rho \ge 0.90$).
  - *$\log_{10}\text{NC1}$ 50% Fall*: $P_i = [12900, 7000, 10300, 11100, 23600]$. $\text{lead\_frac} = 0.0000$, $\rho = 1.0000$, $\bar{r} = -0.0178$. **FAIL**.
- **Verdict**: **CLOSED, NEGATIVE** (0 of 3 rows passing).
- **Scientific Reason & Notes**: Neural collapse is defined by Papyan et al. as a terminal-phase phenomenon occurring after training loss has reached near-zero. NC1 exhibits a massive $\sim 660\times$ collapse (from $\sim 45$ to $\sim 0.05$), but the minimum occurs long after grokking ($1.27 - 5.46 \times G_i$) (*2026-09-06: AGE predictor implemented*; *2026-09-22: AGE Calculation Explanation*).

### Predictor 5: HTSR Alpha
- **Paper**: Charles H. Martin & Michael W. Mahoney (2021, *JMLR* / WeightWatcher).
- **Signal & Firing Rules**:
  - Fits power-law exponent $\alpha$ via Maximum Likelihood with automatic Kolmogorov-Smirnov $x_{\min}$ selection (`powerlaw.Fit`) on the empirical spectral density of the six `nn.Linear` weight matrices (`query`, `key`, `value`, `mlp_in`, `mlp_out`, `output_head`). Firing rule: Global minimum of layer-averaged mean $\alpha$. From Martin & Mahoney.
- **Official Results ($c = 0.8$)**:
  - *Mean $\alpha$ Minimum*: $P_i = [34300, 16000, 24100, 19500, 39600]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = 0.9000$, $\bar{r} = -1.1849$. **FAIL** (fails $\text{lead\_frac} \ge 0.80$ and $\bar{r} \ge 0.05$).
- **Verdict**: **CLOSED, NEGATIVE** (0 of 1 rows passing).
- **Scientific Reason & Notes**: Mean $\alpha$ falls monotonically from $\sim 2.7$ to $\sim 1.25$ and plateaus post-grokking, reaching its minimum after grokking ($1.67 - 2.64 \times G_i$). In $85\% - 94\%$ of layer-checkpoint fits, $\alpha < 2$, outside Martin & Mahoney's heavy-tailed regime $[2, 6]$, caused by small matrix dimensions ($128 \times 128$) and compact Fourier representations (*2026-09-25: HTSR Alpha implemented*; *2026-09-25: Formal scoring of HTSR Alpha*).

### Predictor 6: Weight-PCA
- **Paper**: David Yunis et al. (2024, arXiv:2408.11804).
- **Signal & Firing Rules**:
  - Normalized Effective Rank:
    $$\text{NormEffRank}(W) = \frac{-\sum_i p_i \log p_i}{R}, \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$
    averaged across weight matrices. Firing rule: Global minimum of NormEffRank. From Yunis et al.
- **Official Results ($c = 0.8$)**:
  - *NormEffRank Minimum*: $P_i = [37100, 27900, 32400, 29700, 36200]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = 0.8000$, $\bar{r} = -1.9029$. **FAIL**.
- **Verdict**: **CLOSED, NEGATIVE** (0 of 1 rows passing).
- **Scientific Reason & Notes**: NormEffRank falls monotonically from $\sim 0.037$ to $\sim 0.016 - 0.019$, bottoming out well after grokking ($1.52 - 4.03 \times G_i$), confirming that rank collapse accompanies post-grok cleanup rather than serving as an early indicator (*2026-10-02: Weight-PCA implemented and run*).

### Predictor 7: Higher-MI
- **Paper**: Pomarico et al. (2025, arXiv:2507.23346).
- **Signal & Firing Rules**:
  - Population O-information $\Omega_{\text{pop}}(t) = I(F_0; F_1) - I(F_0; F_1 | F_2)$ averaged across the 6 permutations of correct, runner-up, and rest-mean logits pooled across the test set, estimated via the KSG mutual information estimator ($k = 4$, with strict-inequality tree correction). Firing rule: Global maximum of $\Omega_{\text{pop}}(t)$. Project adaptation from continuous time series to population logit triplet. (Exploratory zero-crossing and windowed temporal control are reported but not counted).
- **Official Results ($c = 0.8$)**:
  - *Population $\Omega$ Maximum*: $P_i = [15500, 8500, 20600, 12900, 32500]$. Firing count: $5/5$. $\text{lead\_frac} = 0.0000$, $\rho = 0.7000$, $\bar{r} = -0.4254$. **FAIL**.
  - *Exploratory Zero-Crossing*: $P_i = [200, 1600, 500, 1100, 400]$. Firing count: $5/5$. $\text{lead\_frac} = 1.0000$, $\rho = -0.7000$, $\bar{r} = 0.9181$. FAIL (reported, not counted).
  - *Exploratory Temporal Control Max*: $P_i = [9300, 7100, 9200, 9400, 23700]$. Firing count: $5/5$. $\text{lead\_frac} = 0.8000$, $\rho = 0.1118$, $\bar{r} = 0.1703$. FAIL (reported, not counted).
- **Verdict**: **CLOSED, NEGATIVE** (0 of 1 rows passing).
- **Scientific Reason & Notes**: $\Omega_{\text{pop}}(t)$ displays quasi-periodic oscillatory spikes across training, with maximum peak occurring post-grok ($1.17 - 2.17 \times G_i$). Spike grid analysis confirmed that spikes drift with model-specific trajectories ($11.90\% \le 30\%$ cross-seed coincidence), reflecting logit margin expansion rather than a transition trigger (*2026-10-03: Higher-MI implemented*; *2026-10-30: Higher-MI spike grid diagnosis settled*).

### Predictor 8: Commutator Defect
- **Paper**: Y. Xu (2026, arXiv:2602.16746).
- **Signal & Firing Rules**:
  - Commutator defect:
    $$D = \frac{\|\theta_{AB} - \theta_{BA}\|}{\|\eta g_A(\theta_0)\| \cdot \|\eta g_B(\theta_0)\|}$$
    evaluated on CPU float64 using the gradient-difference numerator $\eta \|(g_A(\theta_B) - g_A(\theta_0)) - (g_B(\theta_A) - g_B(\theta_0))\|$, probe step $\eta = 10^{-3}$, $K = 5$ fresh pairs of batches ($A, B$ of size 512 sampled with replacement from train split, generator seed 20261103). Firing rule: Xu onset rule — baseline $b = \text{median}(\tilde{D}_0, \tilde{D}_1, \tilde{D}_2)$, first snapshot $k \ge 2$ where $\tilde{D}_k > 10 \cdot \max(b, 0.1)$ and $\tilde{D}_k > 20$. From Xu (2026).
- **Official Results ($c = 0.8$)**:
  - *Primary ($\eta = 10^{-3}$) Onset*: $P_i = [600, 500, 500, 400, 600]$. Firing count: $5/5$. $\text{lead\_frac} = 1.0000$, $\rho = 0.6325$, $\bar{r} = 0.9535$. **FAIL** (fails $\rho \ge 0.90$ and $\bar{r} \le 0.90$).
  - *Secondary ($\eta = 10^{-4}$, unscored)*: $P_i = [2500, 400, 500, 500, 400]$. $\text{lead\_frac} = 1.0000$, $\rho = 0.1581$, $\bar{r} = 0.9270$. **FAIL**.
- **Verdict**: **CLOSED, NEGATIVE** (0 of 1 rows passing).
- **Scientific Reason & Notes**: Defect alarms fire during early memorization (epochs $400 - 600$) when gradient norms contract and $\tilde{D}$ jumps from $\sim 13$ to $\sim 1700$. It exhibits weak rank correlation with grokking time ($\rho = 0.6325$), and mean relative lead ($\bar{r} = 0.9535$) exceeds the $0.90$ ceiling (*2026-11-03: Commutator Defect frozen rules*; *2026-11-04: Commutator Defect Stage 7 frozen verdict*).

### Note on Removed Predictor: Correlation Traps
- **Status**: Removed from the benchmark evaluation sequence on 2026-09-30.
- **Reason & Reference**: Correlation Traps (Prakash & Martin 2026, arXiv:2602.02859) is specifically an *anti-grokking* metric designed to detect late-stage generalization collapse occurring after grokking over long horizons ($10^7$ steps), not an ex-ante early warning indicator for grokking onset (*2026-09-30: Correlation Traps removed from predictor evaluation order*).

---

## 4. Current Repository State

The repository is organized according to the supervisor-mandated structure:

- **Repository Root**:
  - `context.md`: The append-only historical research log.
  - `context_current.md`: The present document (current state of the project).
  - `CLAUDE.md`: Non-negotiable agent instructions (Indian English, direct implementation, math pronunciation, teacher-first mode).
  - `project_compilation.pdf`: Compiled documentation, code, and result summaries generated by `06_Code/scripts/compile_python_files.py`.
  - `project_compilation_npy_dump.md`: Full uncompressed Markdown serialization of all 285 `.npy` arrays across the benchmark ($76.8$ MB).
- **`06_Code/` (Codebase)**:
  - `06_Code/src/models/transformer_four_head.py`: The single canonical 4-head Transformer model.
  - `06_Code/src/predictors/`: The 8 implemented predictor modules (`l2_norm.py`, `dropout.py`, `spectral.py`, `age.py`, `htsr_alpha.py`, `weight_pca.py`, `higher_mi.py`, `commutator_defect.py`).
  - `06_Code/src/unified_measurements.py`: Centralized persistence handler for all predictor data.
  - `06_Code/scripts/run_nanda_benchmark.py`: Master benchmark runner supporting full multi-seed training, checkpoint saving (init, then every 100 epochs; the folders on disk still have the old 801 files), and checkpoint-only recomputation.
  - `06_Code/scripts/analyze_nanda_unified.py`: Official scoring script evaluating all 8 predictors under the fractional lead rule ($c = 0.8$, with $0.7$ and $0.9$ sensitivity sweeps).
  - `06_Code/scripts/run_control_experiment.py`: Synthetic-control validation script for benchmark PASS criteria.
  - `06_Code/configs/nanda_unified.yaml`: Protocol configuration specifying task, architecture, optimizer, and evaluation thresholds.
  - `06_Code/tests/test_frozen_protocol_verdicts.py`: Automated test suite asserting that all 8 predictors fail under the official protocol.
- **`08_Experiments/results/nanda_unified/` (Experiment Data)**:
  - `aggregate.json` and `run.log`: Summary metrics across all 5 seeds.
  - `seed_0/` through `seed_4/`: Per-seed folders holding `summary.json`, training histories, predictor directories (`l2_norm/`, `dropout/`, `spectral/`, `age/`, `htsr/`, `weight_pca/`, `higher_mi/`, `commutator_defect/`), and `checkpoints/` ($801$ model state dict `.pt` files per seed, committed to git).
  - `analysis/`: Official scoring outputs, including `frozen_protocol_verdicts.json` ($c = 0.8$), `predictor_events.json`, four overview figures (`01_grokking_curves.png`, `02_signals_vs_grok.png`, `03_event_vs_grok_scatter.png`, `04_lead_times.png`), and subfolders `c_0.7/` and `c_0.9/`.
- **`11_Predictor_Issues_and_Lacunas/` (Attack Points)**:
  - Detailed checklists of open theoretical and empirical attack points per predictor (e.g., `08_Commutator_Defect_Issues/Commutator_Defect_Attack_Points.md`).
- **`12_Study/` (Textbooks & Study Material)**:
  - `12_Study/Textbook_Common_Concepts_v4.tex` and `.pdf`: The active, canonical common-concepts study textbook (55 pages, zero predictor-specific names, H.K. Dass pedagogical style).
  - `12_Study/Fixed_Values_Base_Experiment.md`: Complete audit table of all fixed constants in the baseline experiment.

---

## 5. Unresolved — Needs Jonathan's Check

The following four items remain active open issues requiring Jonathan's direction or supervisor confirmation:

### 1. Conditional Noise Stress Test False-Positive Rate ($P_i \sim \text{Uniform}[0, G_i]$)
- *Entry 2026-09-29 ("Controlled Experiment: Synthetic-Control Validation")* vs *Entry 2026-11-10 ("Open issue: noise stress test")*:
- *Context*: In the synthetic-control experiment, when an uninformative random predictor is drawn conditionally such that it always leads ($P_i \sim \text{Uniform}[0, G_i]$), it still passes the full PASS rule $14.90\%$ of the time at $c = 0.8$ ($9.74\%$ at $c = 0.7$, $19.51\%$ at $c = 0.9$, and $21.64\%$ under the 100-epoch fixed margin). The target false-positive rate is $\le 5\%$.
- *Issue*: Because $P_i$ is sampled from $[0, G_i]$, larger $G_i$ values naturally produce larger $P_i$ values on average, inducing a structural rank correlation ($\bar{\rho} \approx 0.47$) that inflates pass rates at $n = 5$ seeds.
- *Status*: Open. Jonathan must confirm whether to report this as an acknowledged sample-size limitation in the thesis or address it by running 5 additional seeds ($n = 10$).

### 2. Treatment of HTSR Alpha Values Below 2 ($\alpha < 2$)
- *Entry 2026-09-25 ("HTSR Alpha implemented")* vs *Entry 2026-09-25 ("Progress report PDF")*:
- *Context*: Across the 5 seeds and 401 checkpoints, $85\% - 94\%$ of power-law fits on individual layer empirical spectral densities yield $\alpha < 2$ (layer-averaged mean $\alpha$ terminates at $\approx 1.25$).
- *Issue*: In Martin & Mahoney's heavy-tailed self-regularization framework, the valid heavy-tailed regime is defined as $\alpha \in [2, 6]$; values of $\alpha < 2$ indicate extremely heavy tails, non-standard scaling, or tail-fitting noise due to small layer dimensions ($128 \times 128$).
- *Status*: Open. Awaiting final supervisor confirmation on whether to report $\alpha$ as-is with the small-matrix caveat or include supplementary diagnostic distributions of Kolmogorov-Smirnov distance $D$ and tail cutoff $x_{\min}$.

### 3. Auxiliary Script `REPO_ROOT` Definitions
- *Entry 2026-09-25 ("Path bug in run_nanda_benchmark.py FIXED")*:
- *Context*: On 2026-09-25, the path bug in `06_Code/scripts/run_nanda_benchmark.py` was resolved by setting `REPO_ROOT = dirname(dirname(dirname(__file__)))`. However, auxiliary scripts like `06_Code/scripts/compile_context_bundle.py` and legacy scripts in `06_Code/src/` still compute `REPO_ROOT` using two `dirname` calls, resolving to `06_Code/` instead of the repository root.
- *Issue*: Running these auxiliary scripts directly from subfolders may cause file lookup errors if they attempt to locate `08_Experiments/` or `06_Code/`.
- *Status*: Open. Requires Jonathan's confirmation on whether auxiliary maintenance scripts should be patched or left deprecated.

### 4. Commutator Defect Open Attack Points
- *Entry 2026-11-04 ("Commutator Defect: Stage 8 issues checklist")*:
- *Context*: An audit of Predictor 8 identified 16 theoretical and methodological attack points documented in `11_Predictor_Issues_and_Lacunas/08_Commutator_Defect_Issues/Commutator_Defect_Attack_Points.md`. Key items include: probe step $\eta$ dependence at late epochs, gradient evaluations using batch sampling with replacement on a full-batch model, and the arbitrary floor of 20 in Xu's onset rule.
- *Issue*: While none of these items can flip the overall verdict from FAIL to PASS, they represent potential colloquium defense questions.
- *Status*: Open. Requires Jonathan's review during viva preparation.
