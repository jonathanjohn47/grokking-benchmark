"""
Spectral Predictor (Predictor 3 of 9) — Task-Model Alignment Blueprint.

Exact implementation of SECTION 2 and SECTION 3 from the frozen pre-registration
specification (November 16, 2026).
"""

import numpy as np
import torch
import torch.nn.functional as F


N_SPEC = 2048
RIDGE_LAMBDA = 1e-4
DEFAULT_SAMPLE_GRID = [64, 128, 256, 512, 1024, 2048, 4096]


def get_deterministic_sample_indices(n_total, seed=12345, n_spec=N_SPEC):
    """Deterministic subsampling of training examples without replacement, sorted ascending."""
    if n_total <= n_spec:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_spec, replace=False)
    return np.sort(indices)


def _collect_representations_and_labels(model, dataloader_or_batch_source, device, sample_indices=None):
    """Capture model final hidden states at read position for training examples.

    Runs model in evaluation mode with torch.no_grad().
    Returns:
        Phi: [N, D] float64 ndarray on CPU
        Y:   [N] int64 ndarray on CPU
        sample_indices: [N] int ndarray
    """
    if isinstance(dataloader_or_batch_source, (tuple, list)) and len(dataloader_or_batch_source) == 2 and hasattr(dataloader_or_batch_source[0], "shape"):
        all_x, all_y = dataloader_or_batch_source
    else:
        x_parts, y_parts = [], []
        for x, y in dataloader_or_batch_source:
            x_parts.append(x)
            y_parts.append(y)
        all_x = torch.cat(x_parts, dim=0) if len(x_parts) > 1 else x_parts[0]
        all_y = torch.cat(y_parts, dim=0) if len(y_parts) > 1 else y_parts[0]

    n_total = len(all_x)
    if sample_indices is None:
        sample_indices = get_deterministic_sample_indices(n_total, seed=12345, n_spec=N_SPEC)
    else:
        sample_indices = np.asarray(sample_indices, dtype=int)

    sub_x = all_x[sample_indices]
    sub_y = all_y[sample_indices]

    captured = {}

    def _hook(_module, inputs, _output):
        # inputs[0] is the residual representation feeding output_head, shape [B, seq_len, D]
        # read position is the last token position (index -1)
        captured["h"] = inputs[0][:, -1, :].detach()

    handle = model.output_head.register_forward_hook(_hook)
    model.eval()
    phi_parts = []
    try:
        with torch.no_grad():
            batch_size = 512
            for i in range(0, len(sub_x), batch_size):
                bx = sub_x[i : i + batch_size].to(device)
                model(bx)
                phi_parts.append(captured["h"].cpu().to(torch.float64).numpy())
    finally:
        handle.remove()

    Phi = np.concatenate(phi_parts, axis=0)  # shape (N, D), float64
    Y = sub_y.cpu().numpy() if hasattr(sub_y, "cpu") else np.asarray(sub_y)
    return Phi, Y, sample_indices


def compute_spectral_metrics_from_checkpoint(model, dataloader_or_batch_source, device, sample_indices=None):
    """Compute exact task-model alignment metrics for a single checkpoint (SECTION 2).

    Parameters
    ----------
    model : nn.Module
        Transformer model with output_head.
    dataloader_or_batch_source : DataLoader or (X, Y) tuple
        Training examples only.
    device : torch.device
        Device for model forward pass.
    sample_indices : array-like or None
        Fixed subsampled indices. If None, generated deterministically.

    Returns
    -------
    dict with exactly 14 keys matching Section 4.2:
        k_50, k_90, k_95, K_kern90, alignment_score, alignment_top10,
        spectral_entropy, effective_rank, kernel_trace, numerical_rank,
        eigenvalues, task_power, cumulative_task_power, cumulative_kernel_power.
    """
    Phi, Y, indices = _collect_representations_and_labels(
        model, dataloader_or_batch_source, device, sample_indices
    )
    N, D = Phi.shape

    # Number of classes C
    if hasattr(model, "output_head") and hasattr(model.output_head, "out_features"):
        C = model.output_head.out_features - 1
    elif hasattr(model, "vocab_size"):
        C = model.vocab_size - 1
    else:
        C = int(np.max(Y)) + 1
    C = max(C, int(np.max(Y)) + 1)

    # 2.1 Center columns of Phi
    Phi_c = Phi - np.mean(Phi, axis=0, keepdims=True)

    # 2.2 Gram matrix and CPU float64 eigendecomposition
    K = (Phi_c @ Phi_c.T) / float(N)
    K = 0.5 * (K + K.T)

    eigvals, eigvecs = np.linalg.eigh(K)
    sort_idx = np.argsort(eigvals)[::-1]
    eta = np.maximum(eigvals[sort_idx], 0.0)
    V = eigvecs[:, sort_idx]

    # Numerical rank cutoff: eta_k > 1e-12 * eta_1
    if eta[0] > 0.0:
        cutoff = 1e-12 * eta[0]
        R = int(np.sum(eta > cutoff))
    else:
        R = 0

    if R == 0:
        # Edge case: zero representation
        return {
            "k_50": float("nan"),
            "k_90": float("nan"),
            "k_95": float("nan"),
            "K_kern90": float("nan"),
            "alignment_score": float("nan"),
            "alignment_top10": float("nan"),
            "spectral_entropy": 0.0,
            "effective_rank": 1.0,
            "kernel_trace": 0.0,
            "numerical_rank": 0,
            "eigenvalues": np.zeros(0, dtype=np.float64),
            "task_power": np.zeros(0, dtype=np.float64),
            "cumulative_task_power": np.zeros(0, dtype=np.float64),
            "cumulative_kernel_power": np.zeros(0, dtype=np.float64),
        }

    eta_kept = eta[:R]
    V_kept = V[:, :R]

    T = float(np.sum(eta_kept))
    q = eta_kept / T if T > 0.0 else np.zeros(R, dtype=np.float64)

    # 2.3 Task power (task-model alignment coefficients)
    Y_onehot = np.zeros((N, C), dtype=np.float64)
    for i, y_val in enumerate(Y):
        Y_onehot[i, int(y_val)] = 1.0

    Y_c = Y_onehot - np.mean(Y_onehot, axis=0, keepdims=True)
    W = V_kept.T @ Y_c  # shape (R, C)
    P_k = np.sum(W ** 2, axis=1)  # shape (R,)
    S = float(np.sum(P_k))

    if S == 0.0 or np.isnan(S):
        w2_k = np.full(R, np.nan)
        C_task = np.full(R, np.nan)
        k_50 = float("nan")
        k_90 = float("nan")
        k_95 = float("nan")
        alignment_score = float("nan")
        alignment_top10 = float("nan")
    else:
        w2_k = P_k / S
        C_task = np.cumsum(w2_k)

        # 2.4 (a), (b), (c)
        hits_50 = np.where(C_task >= 0.50)[0]
        k_50 = int(hits_50[0] + 1) if len(hits_50) > 0 else int(R)

        hits_90 = np.where(C_task >= 0.90)[0]
        k_90 = int(hits_90[0] + 1) if len(hits_90) > 0 else int(R)

        hits_95 = np.where(C_task >= 0.95)[0]
        k_95 = int(hits_95[0] + 1) if len(hits_95) > 0 else int(R)

    # Cumulative kernel power
    C_kern = np.cumsum(q)

    # 2.4 (d) K_kern90
    hits_k90 = np.where(C_kern >= 0.90)[0]
    K_kern90 = int(hits_k90[0] + 1) if len(hits_k90) > 0 else int(R)

    # 2.4 (e) alignment_score
    if S != 0.0 and not np.isnan(S):
        alignment_score = float(C_task[K_kern90 - 1])
        top10_idx = max(1, int(np.floor(0.10 * R)))
        alignment_top10 = float(C_task[top10_idx - 1])

    # 2.4 (g), (h), (i), (j)
    pos_q = q[q > 0.0]
    spectral_entropy = float(-np.sum(pos_q * np.log(pos_q))) if len(pos_q) > 0 else 0.0
    effective_rank = float(np.exp(spectral_entropy))
    kernel_trace = float(T)
    numerical_rank = int(R)

    return {
        "k_50": k_50,
        "k_90": k_90,
        "k_95": k_95,
        "K_kern90": K_kern90,
        "alignment_score": alignment_score,
        "alignment_top10": alignment_top10,
        "spectral_entropy": spectral_entropy,
        "effective_rank": effective_rank,
        "kernel_trace": kernel_trace,
        "numerical_rank": numerical_rank,
        "eigenvalues": eta_kept,
        "task_power": w2_k,
        "cumulative_task_power": C_task,
        "cumulative_kernel_power": C_kern,
    }


def compute_spectral_series(checkpoint_paths, epochs, model_factory, data_source, device):
    """Compute spectral metric time series across an ordered sequence of checkpoints.

    Returns dict with keys:
        epochs, k_90, k_95, alignment_score, spectral_entropy, eigenvalues, cumulative_power.
    """
    epochs_arr = np.asarray(epochs, dtype=int)
    n_ckpts = len(checkpoint_paths)

    k90_list = []
    k95_list = []
    align_list = []
    entropy_list = []
    eigenvalues_list = []
    cum_power_list = []

    model = model_factory() if callable(model_factory) else model_factory
    sample_indices = None

    for p in checkpoint_paths:
        state = torch.load(p, map_location=device)
        model.load_state_dict(state)
        m = compute_spectral_metrics_from_checkpoint(
            model, data_source, device, sample_indices=sample_indices
        )
        if sample_indices is None and "sample_indices" in m:
            sample_indices = m["sample_indices"]

        k90_list.append(m["k_90"])
        k95_list.append(m["k_95"])
        align_list.append(m["alignment_score"])
        entropy_list.append(m["spectral_entropy"])
        eigenvalues_list.append(m["eigenvalues"])
        cum_power_list.append(m["cumulative_task_power"])

    # Determine uniform array widths across checkpoints
    max_r = max(len(ev) for ev in eigenvalues_list) if eigenvalues_list else 0
    eig_mat = np.zeros((n_ckpts, max_r), dtype=np.float64)
    cum_mat = np.zeros((n_ckpts, max_r), dtype=np.float64)

    for i in range(n_ckpts):
        r_i = len(eigenvalues_list[i])
        eig_mat[i, :r_i] = eigenvalues_list[i]
        cum_mat[i, :r_i] = cum_power_list[i]

    return {
        "epochs": epochs_arr,
        "k_90": np.asarray(k90_list, dtype=float),
        "k_95": np.asarray(k95_list, dtype=float),
        "alignment_score": np.asarray(align_list, dtype=float),
        "spectral_entropy": np.asarray(entropy_list, dtype=float),
        "eigenvalues": eig_mat,
        "cumulative_power": cum_mat,
    }


def detect_spectral_trigger(epochs, k90_series, alignment_series):
    """Exact pre-registered trigger rule (SECTION 3).

    Pure function: plain numpy arrays, no PyTorch, no I/O, no randomness.
    """
    epochs = np.asarray(epochs, dtype=int)
    k90 = np.asarray(k90_series, dtype=float)
    align = np.asarray(alignment_series, dtype=float)

    M = len(epochs)
    if M < 3:
        return None

    # 3.3 Baseline: median of the RAW k(t) values over the first 5 checkpoints of the run (by epoch order)
    n_base = min(5, M)
    b_k = float(np.median(k90[:n_base]))
    if b_k == 0.0 or np.isnan(b_k):
        return None

    # 3.2 Causal smoothing (trailing moving average, width 3, no centered or future window)
    # k_s(t) = mean of k(t-2..t), A_s(t) = mean of A(t-2..t)
    # For t < 2 (0-indexed), smoothed value is undefined and ineligible to fire.
    k_s = np.full(M, np.nan)
    A_s = np.full(M, np.nan)
    for t in range(2, M):
        k_s[t] = np.mean(k90[t-2 : t+1])
        A_s[t] = np.mean(align[t-2 : t+1])

    # 3.1 Burn-in exclusion: discard all checkpoints with e_t < 200 from firing eligibility
    # 3.4 Joint condition at checkpoint t (both must hold):
    #     C1: k_s(t) <= 0.90 * b_k and raw k90(t) <= 0.90 * b_k (ensures persistence reflects sustained transition)
    #     C2: A_s(t) >= 0.50
    cond = np.zeros(M, dtype=bool)
    for t in range(2, M):
        if epochs[t] >= 200:
            if (k_s[t] <= 0.90 * b_k) and (k90[t] <= 0.90 * b_k) and (A_s[t] >= 0.50):
                cond[t] = True

    # 3.5 Persistence: joint condition must hold at 3 CONSECUTIVE checkpoints t, t+1, t+2, all with epoch >= 200
    for t in range(2, M - 2):
        if cond[t] and cond[t+1] and cond[t+2]:
            # 3.6 Firing epoch: P_i = e_t of the FIRST checkpoint of the first qualifying persistence window
            return int(epochs[t])

    return None


def compute_reference_learning_curve(eigenvalues, task_power_normalized, ridge=RIDGE_LAMBDA, sample_grid=None):
    """Compute Canatar-style mode-resolved analytic learning curve (SECTION 2.5).

    Non-scoring reference theory diagnostic.
    """
    if sample_grid is None:
        sample_grid = DEFAULT_SAMPLE_GRID
    eta = np.asarray(eigenvalues, dtype=np.float64)
    w2 = np.asarray(task_power_normalized, dtype=np.float64)
    T = float(np.sum(eta))

    results = {
        "sample_grid": list(sample_grid),
        "predicted_gen_error": [],
        "kappa": [],
        "gamma": [],
    }

    for P in sample_grid:
        def f(kappa):
            return ridge + kappa * np.sum(eta / (P * eta + kappa)) - kappa

        a, b = 1e-12, 1e6 * (1.0 + T)
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            results["predicted_gen_error"].append(float("nan"))
            results["kappa"].append(float("nan"))
            results["gamma"].append(float("nan"))
            continue

        for _ in range(500):
            mid = 0.5 * (a + b)
            if (b - a) <= 1e-12 * mid or abs(f(mid)) < 1e-15:
                break
            fmid = f(mid)
            if fa * fmid <= 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid

        kappa = mid
        gamma = float(P * np.sum((eta ** 2) / ((P * eta + kappa) ** 2)))
        if gamma >= 1.0:
            E_g = float("nan")
        else:
            E_k = (1.0 / (1.0 - gamma)) * ((kappa ** 2) / ((P * eta + kappa) ** 2)) * w2
            E_g = float(np.sum(E_k))

        results["predicted_gen_error"].append(E_g)
        results["kappa"].append(kappa)
        results["gamma"].append(gamma)

    return results


# Aliases for compatibility
compute_spectral_metrics_for_checkpoint = compute_spectral_metrics_from_checkpoint
compute_spectral_for_model = compute_spectral_metrics_from_checkpoint
