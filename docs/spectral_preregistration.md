# Spectral Predictor Pre-Registration

Pre-registered before any empirical Spectral evaluation. Snapshot date: November 16, 2026. Immutable thereafter.

=====================================================================
SECTION 2 — EXACT MATHEMATICAL BLUEPRINT (SPECTRAL SETUP)
=====================================================================
The Spectral setup is the empirical, finite-sample operationalization of kernel spectral bias and task-model alignment. Implement exactly the following. All symbols are defined; no external reading is required.

2.1 REPRESENTATION MATRIX
    - For a given checkpoint, run the model in evaluation mode with gradients disabled.
    - Input set: TRAINING examples only. Never test/validation examples.
    - From the model's final hidden layer, take the hidden state at the READ position, i.e. the last token position of the input sequence. Call this vector h(x) with dimension D.
    - Deterministic subsampling: select exactly N_SPEC = 2048 training examples using a fixed numpy RNG with seed 12345, sampling WITHOUT replacement from the full training index range, then sort the selected indices ascending. If the training set has fewer than 2048 examples, use all of them sorted ascending. The identical index set must be reused for every checkpoint and every seed. Persist the chosen index array to disk once per seed and reload it, so drift is impossible.
    - Stack rows to form Phi with shape (N, D), where N = number of selected examples.
    - Center columns: Phi_c = Phi - mean(Phi, axis=0).

2.2 GRAM MATRIX AND EIGENDECOMPOSITION
    - K = (Phi_c @ Phi_c.T) / N, shape (N, N). This normalization makes trace(K) equal to the mean squared centered representation norm.
    - Cast K to float64. Move to CPU. Eigendecomposition MUST run on CPU in float64 (MPS/CUDA float16/float32 paths are forbidden for this step).
    - Use a symmetric eigensolver (eigh-equivalent), not a general eigensolver.
    - Symmetrize before solving: K = 0.5 * (K + K.T).
    - Sort eigenvalues DESCENDING. Reorder eigenvectors with the same permutation. Call them eta_k (eigenvalues) and v_k (eigenvectors, columns), k = 1..N.
    - Clamp small negatives: eta_k = max(eta_k, 0.0).
    - Numerical rank cutoff: keep modes with eta_k > 1e-12 * eta_1. Call the kept count R. Report R. All mode sums below run over kept modes only.
    - Kernel trace power: T = sum_k eta_k. Normalized kernel spectrum: q_k = eta_k / T.

2.3 TASK POWER (TASK-MODEL ALIGNMENT COEFFICIENTS)
    - The task is multi-class. Build the one-hot target matrix Y with shape (N, C), C = number of classes, using the SAME subsampled indices and the SAME order as Phi.
    - Center each target column: Y_c = Y - mean(Y, axis=0).
    - Project onto kernel eigenmodes: W = V.T @ Y_c, shape (N, C), where V holds the kept eigenvectors.
    - Raw per-mode task power: P_k = sum over classes c of W[k, c]^2.
    - Total task power: S = sum_k P_k. If S == 0, mark the checkpoint invalid and record NaN for alignment-derived metrics (do not divide by zero).
    - Normalized task power: w2_k = P_k / S. By construction sum_k w2_k = 1.
    - Cumulative task power: C_task(k) = sum_{j<=k} w2_j, monotonically non-decreasing, C_task(R) = 1.
    - Cumulative kernel power: C_kern(k) = sum_{j<=k} q_j.

2.4 SCALAR METRICS PER CHECKPOINT (EXACT DEFINITIONS)
    (a) k_90 = smallest integer k >= 1 such that C_task(k) >= 0.90. Primary signal.
    (b) k_50 = smallest k with C_task(k) >= 0.50. Diagnostic only.
    (c) k_95 = smallest k with C_task(k) >= 0.95. Diagnostic only.
    (d) K_kern90 = smallest k with C_kern(k) >= 0.90. Diagnostic.
    (e) alignment_score A = C_task(K_kern90). This is the fraction of total task power captured by the kernel modes that carry 90 percent of kernel trace power. Primary co-signal. Range [0, 1].
    (f) alignment_top10 = C_task(max(1, floor(0.10 * R))). Secondary diagnostic only.
    (g) spectral_entropy H = - sum_k q_k * ln(q_k), natural logarithm, terms with q_k = 0 contribute 0.
    (h) effective_rank = exp(H). Diagnostic.
    (i) kernel_trace = T. Diagnostic.
    (j) numerical_rank = R. Diagnostic.

2.5 SECONDARY REFERENCE-THEORY DIAGNOSTIC (NON-SCORING, MUST NOT DRIVE THE TRIGGER)
    Purpose: produce a Canatar-style mode-resolved analytic learning curve from the measured spectrum, as supporting evidence only.
    - Inputs: measured eta_k (kept modes, raw, not normalized), measured w2_k, ridge lambda = 1e-4 (fixed), sample counts P in the fixed grid [64, 128, 256, 512, 1024, 2048, 4096].
    - Effective regularization kappa solves the scalar self-consistency relation:
          kappa = lambda + kappa * sum_k ( eta_k / (P * eta_k + kappa) )
      Solve for kappa > 0 by bisection on the residual f(kappa) = lambda + kappa * sum_k(eta_k/(P*eta_k+kappa)) - kappa, bracket [1e-12, 1e6 * (1 + T)], tolerance 1e-12 relative, maximum 500 iterations. If the bracket does not change sign, record NaN and continue; never crash the run.
    - Mode susceptibility: gamma = P * sum_k ( eta_k^2 / (P * eta_k + kappa)^2 ).
    - Mode error: E_k = (1 / (1 - gamma)) * ( kappa^2 / (P * eta_k + kappa)^2 ) * w2_k.
    - Predicted generalization error: E_g(P) = sum_k E_k. If gamma >= 1, record NaN for that P.
    - Store the (P, E_g) pairs. These numbers are reported as theory-side context only. They are NEVER used as the firing signal and NEVER passed to the Scorer.

2.6 WHAT IS EXPLICITLY FORBIDDEN IN THE METRIC
    - No test-set representations.
    - No PCA on uncentered features.
    - No weight-matrix spectra instead of representation-kernel spectra.
    - No float32 eigendecomposition.
    - No per-checkpoint renormalization that changes the meaning of k_90 across checkpoints (k_90 is always an absolute mode count under 2.4a).
    - No use of ground-truth grokking epochs anywhere inside metric or trigger computation.

=====================================================================
SECTION 3 — EXACT PRE-REGISTERED TRIGGER RULE (IMMUTABLE)
=====================================================================
Let checkpoints for a seed be ordered by epoch e_1 < e_2 < ... < e_M. Let k(t) = k_90 at checkpoint t and A(t) = alignment_score at checkpoint t.

3.1 Burn-in exclusion: discard all checkpoints with e_t < 200 from firing eligibility. They may still be used for smoothing history. Fixed a priori.
3.2 Causal smoothing (trailing moving average, width 3, no centered or future window):
        k_s(t) = mean of k(t-2..t), A_s(t) = mean of A(t-2..t).
    For t < 3, the smoothed value is undefined and the checkpoint is ineligible to fire.
3.3 Baseline: b_k = median of the RAW k(t) values over the first 5 checkpoints of the run (by epoch order), computed once per seed. If fewer than 5 checkpoints exist, use all available. If b_k is 0 or NaN, the seed cannot fire.
3.4 Joint condition at checkpoint t (both must hold):
        C1: k_s(t) <= 0.90 * b_k
        C2: A_s(t) >= 0.50
3.5 Persistence: the joint condition must hold at 3 CONSECUTIVE checkpoints t, t+1, t+2, all with epoch >= 200.
3.6 Firing epoch: P_i = e_t of the FIRST checkpoint of the first qualifying persistence window. Not the window end. Not the global extremum.
3.7 If no qualifying window exists, P_i = None (recorded as null / no-fire). Never substitute a fallback epoch, never clamp to the last checkpoint, never impute.
3.8 All constants above (200, 3, 5, 0.90, 0.50, 3) are frozen. No sweep, no grid search, no sensitivity-based retuning. A separate read-only sensitivity report is permitted only in Phase 6.5 and must not change P_i.

=====================================================================
SECTION 6 — GATE THRESHOLDS (FROZEN PROTOCOL & SCORER)
=====================================================================
6.1 Lead fraction (protocol constant c = 0.8):
    lead condition: P_i < c * G_i = 0.8 * G_i
    threshold: lead_frac >= 0.80 (at least 80% of firing seeds strictly lead)
6.2 Rank correlation:
    Spearman rho(P_i, G_i) >= 0.90 (specifically rho >= 0.8999999999999998)
6.3 Noise margin (mean relative lead):
    r_bar = mean((G_i - P_i) / G_i) in [0.05, 0.90]
6.4 Minimum firing count:
    At least 4 of 5 seeds must fire (n_firing >= 4). If fewer than 4 fire, FAIL immediately.
6.5 Superiority Index:
    If any validity gate condition fails, SI = 0.0.
    If validity gate passes:
        m_lead = (lead_frac - 0.8) / 0.2
        m_corr = (rho - 0.9) / 0.1
        m_noise = (r_bar - 0.05) / 0.45
        SI = (clip(m_lead, 0, 1) + clip(m_corr, 0, 1) + clip(m_noise, 0, 1)) / 3.0
6.6 Final verdict:
    PASS if and only if all validity gate conditions are satisfied (lead_frac >= 0.80, rho >= 0.90, noise_margin in [0.05, 0.90], n_firing >= 4). FAIL otherwise.
