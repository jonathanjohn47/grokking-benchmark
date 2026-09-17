# Literature

Source papers for the thesis *A Unified Benchmark of Grokking Predictors in Neural Networks*.

All PDFs sit flat in this folder. This README is the index. Filenames are kept
as supplied, with one exception: `Paper et al.pdf` (a download placeholder name)
was renamed to `Alethea Power - Grokking Generalization beyond overfitting on
small algorithmic datasets [2022].pdf`.

Total: 25 PDFs.

---

## 1. Grokking — direct

| File | Note |
|---|---|
| `Alethea Power - Grokking Generalization beyond overfitting on small algorithmic datasets [2022].pdf` | Power, Burda, Edwards, Babuschkin, Misra (2022). **The paper that first reported grokking.** This project follows it for the modulus `p = 97` and the 3-seeds-per-predictor protocol. (Was supplied as `Paper et al.pdf`.) |
| `Nanda et al.pdf` | Nanda, Chan, Lieberum, Smith, Steinhardt (2023), *Progress measures for grokking via mechanistic interpretability*, ICLR. **The architecture/setup reference for this project's four-head model.** Defines restricted loss and excluded loss, the three phases (memorisation / circuit formation / cleanup), and states plainly (Sec. 3) that no LayerNorm is used. |
| `Domenico Pomarico - Transfer Entropy and O-Information to Detect Grokking in Tensor Network Multi-Class Classification Problem.pdf` | Grokking detection via information-theoretic measures. Relevant to the **Higher-MI** predictor. |

## 2. Heavy-tailed self-regularisation / weight-matrix spectra

| File | Note |
|---|---|
| `Charles H. Martin - Predicting trends in the quality of state-of-the-art neural networks without access to training or testing.pdf` | Martin & Mahoney — HTSR / `weightwatcher`. **Source for the HTSR Alpha predictor**; also feeds the Spectral predictor. |
| `Chunheng Jiang - Network properties determine neural network performance [2024].pdf` | Network-science view of what predicts NN performance. |

## 3. Neural collapse (terminal phase of training)

| File | Note |
|---|---|
| `Vardan Papyan - Prevalence of neural collapse during the terminal phase of deep learning training [2020].pdf` | The original neural-collapse paper. **Source for the AGE predictor** — AGE (Adaptive Grokking Epoch via Neural Collapse) reads off the epoch of NC1 "within-class variability collapse", `Tr(Sigma_W)/Tr(Sigma_B) -> 0`. |
| `D. Mixon - Neural collapse with unconstrained features [2020].pdf` | Unconstrained-features model of neural collapse. |
| `Cong Fang - Exploring deep neural networks via layer-peeled model Minority collapse in imbalanced training [2021].pdf` | Layer-peeled model; minority collapse. |
| `Mengjia Xu - Dynamics in Deep Classifiers Trained with the Square Loss Normalization, Low Rank, Neural Collapse, and Generaliz.pdf` | Square-loss training dynamics linking normalisation, low rank, neural collapse, generalisation. |

## 4. Generalisation dynamics / double descent / interpolation

| File | Note |
|---|---|
| `Chiyuan Zhang - Understanding deep learning (still) requires rethinking generalization [2021].pdf` | CACM update of the 2017 "rethinking generalization" paper. |
| `J. Rocks - Memorizing without overfitting Bias, variance, and interpolation in overparameterized models [2020].pdf` | Rocks & Mehta — arXiv preprint (2020). |
| `PhysRevResearch.4.013201.pdf` | **Same paper as the line above** — the published version, Phys. Rev. Research 4, 013201 (2022). Duplicate; keep one. |
| `S. Spigler - A jamming transition from under- to over-parametrization affects generalization in deep learning [2018].pdf` | Jamming transition at the interpolation threshold. |
| `Madhu S. Advani - High-dimensional dynamics of generalization error in neural networks [2017].pdf` | Analytic generalisation-error dynamics. |
| `P. Baldi - Temporal Evolution of Generalization during Learning in Linear Networks [1991].pdf` | Early linear-network analysis of the generalisation curve over training time. |

## 5. Kernel / random-features / mean-field theory

| File | Note |
|---|---|
| `Song Mei - A mean field view of the landscape of two-layer neural networks [2018].pdf` | Mean-field limit of two-layer nets. |
| `Song Mei - The Generalization Error of Random Features Regression Precise Asymptotics and the Double Descent Curve [2019].pdf` | Random-features regression; precise double-descent asymptotics. |
| `Abdulkadir Canatar - Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide ne.pdf` | Spectral bias / task-model alignment in kernel regression and wide nets. Relevant to the **Spectral** predictor. |
| `Convex_Formulation_of_Overparameterized_Deep_Neural_Networks.pdf` | Convex reformulation of overparameterised DNN training. |
| `Jean Barbier - Optimal errors and phase transitions in high-dimensional generalized linear models [2017].pdf` | Phase transitions in high-dimensional GLMs. |

## 6. Scaling laws / representations

| File | Note |
|---|---|
| `Yasaman Bahri - Explaining neural scaling laws [2021].pdf` | Mechanisms behind power-law scaling. |
| `Qianyi Li - Representations and generalization in artificial and brain neural networks [2024].pdf` | Representation geometry vs generalisation, ANN and brain. (Large file, ~19 MB.) |

## 7. Robustness / margins / shortcut learning

| File | Note |
|---|---|
| `Jure Sokolić - Robust Large Margin Deep Neural Networks [2016].pdf` | Margin-based generalisation bounds. |
| `Robert Geirhos - Shortcut learning in deep neural networks [2020].pdf` | Shortcut learning as a failure of generalisation. |

## 8. Other

| File | Note |
|---|---|
| `Giulio Biroli - Dynamical regimes of diffusion models [2024].pdf` | Dynamical regimes / phase behaviour in diffusion models. |

---

## Housekeeping notes

- **Duplicate:** `J. Rocks - Memorizing without overfitting ... [2020].pdf` and
  `PhysRevResearch.4.013201.pdf` are the preprint and published version of the
  same Rocks & Mehta paper. Decide which one to keep.
- **Predictor → paper map.**

  | Predictor | Primary paper | Status |
  |---|---|---|
  | L2 Norm | Nanda et al. 2023 (Fig. 7 sum of squared weights) | CLOSED, negative |
  | Dropout | Salah & Yevick, arXiv:2507.11645 (variance under stochastic dropout) | CLOSED, negative |
  | Spectral | Canatar et al. 2021, *Nat. Commun.* 12, 2914 (spectral bias / task-model alignment) | CLOSED, negative |
  | AGE | Papyan et al. 2020 PNAS (NC1 variability collapse) — **mechanism:** Beaglehole et al. 2024 (AGOP → deep neural collapse, arXiv:2402.13728) + Mallinar et al. 2024 (grokking mod-p via AGOP/RFM, arXiv:2407.20199) | CLOSED, negative (2026-09-06) — NC1 collapses ~45 → ~0.05 in 5/5 seeds but `nc1_min_epoch` lags grok in 5/5 (ratio 1.67–3.92) |
  | HTSR Alpha | Martin & Mahoney (`weightwatcher`) | not started |
  | Higher-MI | Pomarico et al. 2025 (transfer entropy / O-information) | not started |
  | Correlation Traps, Weight-PCA, Commutator Defect | no source in this set yet | not started |

  **Note on AGE:** AGE is defined as an NC1-based *adaptive grokking-epoch
  detector* — the checkpoint epoch at which within-class feature
  variability collapses (`Tr(Sigma_W)/Tr(Sigma_B)` bottoms out). It is
  checkpoint-only, loss-agnostic, and independent of the Nanda training
  substrate. The Beaglehole and Mallinar papers are the mechanism
  citations (why last-layer neural collapse should track grokking on
  `(a+b) mod p` at all); Papyan is the quantity itself.
- These PDFs are large (~90 MB total). Decide before committing whether they
  belong in git history or should be `.gitignore`d and kept locally / in shared
  storage.
