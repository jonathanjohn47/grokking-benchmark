"""
Spectral predictor (Predictor 3 of 9) — FAITHFUL Canatar et al. 2021.

Paper: Abdulkadir Canatar, Blake Bordelon, Cengiz Pehlevan, "Spectral bias
and task-model alignment explain generalization in kernel regression and
infinitely wide neural networks", Nature Communications 12, 2914 (2021).
PDF in repo: literature/Abdulkadir Canatar - Spectral bias and task-model
alignment ... .pdf

WHAT THIS NOW COMPUTES (and what it replaced)
---------------------------------------------
The previous version of this file computed per-weight-matrix singular
value statistics (spectral_norm, fro_norm, stable_rank, effective_rank).
That was a cheap proxy inspired by Nanda et al.'s low-rank Fourier
circuit, NOT anything from Canatar et al. It has been deleted entirely —
no `stable_rank`, no `svdvals(W)` on weight matrices anywhere below.

This file is now a direct implementation of Canatar et al. 2021,
Eq. (3)-(6), applied to the trained transformer's own representation
kernel:

  1. Representation kernel.  Take the model's final hidden state at the
     read position (the vector that feeds `output_head`), one per
     TRAINING example: Phi in R^{N x d_model}, N = |train split| = 3830
     for p = 113 (floor(0.3 * 113 * 113)). Centre it and form the Gram
     matrix K = Phi_c @ Phi_c.T  (N x N). This is the empirical, sampled
     version of the kernel whose Mercer spectrum Canatar's theory needs
     (their Eq. 3: the eigenproblem of the kernel under the data
     measure).

  2. Kernel eigenvalues eta_k  (Canatar Eq. 3).  eigh(K), sorted
     descending. "Spectral bias" = the model learns the eigenfunctions
     with the largest eta_k first.

  3. Task power w_k^2  (Canatar Eq. 4-5).  Project the centred one-hot
     label matrix Y_c in R^{N x p} onto the kernel eigenvectors U:
     W = U.T @ Y_c, and w_k^2 = ||W_k||^2 (row-norm squared, summed over
     the p classes). Normalised: p_k = w_k^2 / sum_j w_j^2 is the
     fraction of the target function's power carried by eigenmode k.

  4. Cumulative power C(k)  (Canatar Eq. 6, "task-model alignment").
     C(k) = sum_{j<=k} p_j. If the target's power is concentrated in the
     top (large-eta) modes, C(k) rises to 1 for small k and the task is
     "aligned" with the kernel — Canatar's condition for sample-efficient
     generalization.

  5. Predictor signals per checkpoint:
       - k_90 / k_95 : smallest k with C(k) >= 0.90 / 0.95 (how many
         leading eigenmodes hold 90 / 95 % of the task power).
       - alignment_score = sum_k eta_k_norm * p_k : task power weighted
         by normalised eigenvalue. High => task power sits in the large
         eigenvalue directions.
       - entropy = -sum_k p_k log p_k : spread of the task power over
         eigenmodes.

THE PREDICTION.  Canatar's mechanism says generalization becomes possible
once the target aligns with the top of the kernel spectrum. So the
grokking signal this predictor tests is: **k_90 should DROP (and
alignment_score should RISE) at or before the grok epoch** — task-model
alignment forming ahead of the test-accuracy jump. k_90_min_epoch vs
grok_epoch is the head-to-head number the benchmark records.

MPS NOTE.  torch.linalg.eigh does not run on the "mps" backend. The Gram
matrix is moved to CPU (and promoted to float64) before eigh — cheap,
since this is one eigendecomposition per saved checkpoint, not per
training epoch.
"""

import torch
import torch.nn.functional as F

# How many leading eigenvalues / cumulative-power points to persist per
# checkpoint (full N=3830 vectors are not needed downstream; the signal
# lives in the head of the spectrum).
N_EIGENVALUES_SAVED = 50
N_CUMULATIVE_SAVED = 100

_EPS = 1e-12


def _collect_representations_and_labels(model, train_loader, device):
    """Run the model over the full training split and capture, per example,
    the hidden state that feeds `output_head` at the read position
    (sequence index 2, the "=" token). Returns:
        Phi : [N, d_model]  float64, CPU
        Y   : [N]           int64,  CPU   (class labels 0..p-1)
    """
    captured = {}

    def _hook(_module, inputs, _output):
        # inputs[0] is mlp_output, shape [B, seq_len, d_model]; the vector
        # that goes into output_head. read_position = 2 (configs/nanda_unified.yaml).
        captured["h"] = inputs[0][:, 2, :].detach()

    handle = model.output_head.register_forward_hook(_hook)
    model.eval()
    phi_parts, y_parts = [], []
    try:
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                model.forward(x)
                # move off MPS FIRST, then promote — MPS has no float64.
                phi_parts.append(captured["h"].cpu().double())
                y_parts.append(y.detach().cpu())
    finally:
        handle.remove()

    Phi = torch.cat(phi_parts, dim=0)
    Y = torch.cat(y_parts, dim=0).long()
    return Phi, Y


def compute_spectral_metrics_for_checkpoint(model, train_loader, device):
    """Canatar et al. 2021 task-model-alignment metrics for ONE frozen
    model checkpoint. `train_loader` must iterate the full training split
    (for p=113 that is N = floor(0.3 * 113 * 113) = 3830 examples; the
    Nanda-Unified run is full-batch, so one iteration already covers it).

    Returns a dict:
        eigenvalues      : list[float]  top N_EIGENVALUES_SAVED of eta_k (descending)
        k_90, k_95       : int          smallest k with C(k) >= 0.90 / 0.95
        alignment_score  : float        sum_k eta_k_norm * p_k
        entropy          : float        -sum_k p_k log p_k
        cumulative_power : list[float]  C(1..N_CUMULATIVE_SAVED)
        N                : int          number of training examples used
    """
    Phi, Y = _collect_representations_and_labels(model, train_loader, device)
    N, _d_model = Phi.shape
    p = model.output_head.out_features - 1  # vocab = p + 1 ("=" token id is p)

    # ---- (1) centred representation Gram matrix K = Phi_c Phi_c^T ----
    Phi_c = Phi - Phi.mean(dim=0, keepdim=True)
    K = Phi_c @ Phi_c.T  # [N, N], float64, CPU

    # ---- (2) kernel eigenvalues eta_k (Canatar Eq. 3) ----
    # eigh runs on CPU only (MPS has no aten::_linalg_eigh); K is already CPU.
    eigvals, eigvecs = torch.linalg.eigh(K)
    # eigh returns ascending -> flip to descending.
    eta = eigvals.flip(0).clamp_min(0.0)           # [N]
    U = eigvecs.flip(1)                             # [N, N], column k <-> eta[k]

    # ---- (3) task power w_k^2 = || (U^T Y_c)_k ||^2  (Canatar Eq. 4-5) ----
    Y_onehot = F.one_hot(Y, num_classes=p).to(torch.float64)   # [N, p]
    Y_c = Y_onehot - Y_onehot.mean(dim=0, keepdim=True)
    W = U.T @ Y_c                                   # [N, p]
    power = (W ** 2).sum(dim=1)                     # [N]  w_k^2
    power_norm = power / (power.sum() + _EPS)       # p_k

    # ---- (4) cumulative power C(k)  (Canatar Eq. 6) ----
    C = torch.cumsum(power_norm, dim=0)            # [N]

    def _first_k_at_least(threshold):
        hits = (C >= threshold).nonzero()
        return int(hits[0].item()) + 1 if len(hits) else int(N)

    k_90 = _first_k_at_least(0.90)
    k_95 = _first_k_at_least(0.95)

    # ---- (5) scalar signals ----
    eta_norm = eta / (eta.sum() + _EPS)
    alignment_score = float((eta_norm * power_norm).sum())
    entropy = float(-(power_norm * torch.log(power_norm + _EPS)).sum())

    n_eig = min(N_EIGENVALUES_SAVED, int(N))
    n_cum = min(N_CUMULATIVE_SAVED, int(N))

    return {
        "eigenvalues": [float(v) for v in eta[:n_eig]],
        "k_90": k_90,
        "k_95": k_95,
        "alignment_score": alignment_score,
        "entropy": entropy,
        "cumulative_power": [float(v) for v in C[:n_cum]],
        "N": int(N),
    }


def compute_spectral_for_model(model, train_loader, device):
    """Thin wrapper kept for run_nanda_benchmark.py's import. One frozen
    checkpoint in -> one Canatar-metrics dict out (see
    compute_spectral_metrics_for_checkpoint)."""
    return compute_spectral_metrics_for_checkpoint(model, train_loader, device)
