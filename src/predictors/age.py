"""
AGE predictor (Predictor 4 of 9) — Adaptive Grokking Epoch via Neural Collapse.

DEFINITION (locked)
-------------------
AGE = the epoch at which NC1 "within-class variability collapse" occurs.
NC1 is the neural-collapse quantity from Papyan, Han & Donoho 2020:

  Papyan, Vardan; Han, X. Y.; Donoho, David L. "Prevalence of neural
  collapse during the terminal phase of deep learning training."
  Proceedings of the National Academy of Sciences 117 (40), 24652-24663
  (2020).  PDF in repo:
  literature/Vardan Papyan - Prevalence of neural collapse during the
  terminal phase of deep learning training [2020].pdf

Papyan's NC1 ("variability collapse"): the last-layer features of examples
from the same class stop varying — the within-class scatter shrinks
relative to the between-class scatter, i.e.

        NC1  =  Tr(Sigma_W) / Tr(Sigma_B)  ->  0

  Sigma_W = within-class covariance of the penultimate features,
  Sigma_B = between-class covariance of the class means.

AGE reads this collapse epoch off frozen checkpoints and compares it to
the grok epoch. The prediction under test: NC1 falls sharply at or
BEFORE the test-accuracy jump, so the epoch of minimum NC1 is a leading
(or coincident) marker of grokking.

MECHANISM SUPPORT
-----------------
Why last-layer neural collapse is expected to track grokking on this
substrate at all:

  - Beaglehole, Daniel; Radhakrishnan, Adityanarayanan; Pandit, Parthe;
    Belkin, Mikhail. "Average gradient outer product as a mechanism for
    deep neural collapse." arXiv:2402.13728 (2024). NC1 collapse is
    driven by the Average Gradient Outer Product (AGOP) reshaping
    features class-wise.

  - Mallinar, Neil; Beaglehole, Daniel; Zhu, Libin; Radhakrishnan,
    Adityanarayanan; Pandit, Parthe; Belkin, Mikhail. "Emergence in
    non-neural models: grokking modular arithmetic via average gradient
    outer product." arXiv:2407.20199 (2024). The same AGOP mechanism
    produces grokking on (a+b) mod p specifically — the task this
    benchmark runs.

Threshold / collapse-epoch reading follows Paul & Rupa 2026, "Neural
Collapse Dynamics".

WHAT THIS FILE COMPUTES (and what it does NOT)
---------------------------------------------
Per frozen checkpoint, on the model's own penultimate representation at
the read position (the vector that feeds `output_head`), one per TRAINING
example: Phi in R^{N x d_model}, N = |train split| = floor(0.3 * p * p)
= 3830 for p = 113.

  mu_c    = mean of Phi over examples of class c
  mu_G    = global mean of Phi
  Tr_W    = mean_c  mean_{i in c} || h_i - mu_c ||^2      (within-class)
  Tr_B    = mean_c  || mu_c - mu_G ||^2                   (between-class)
  NC1     = Tr_W / (Tr_B + 1e-12)
  fn      = mean_i || h_i ||_2                            (feature norm)

There is NO weight-matrix SVD here. AGE is a representation-space,
loss-agnostic, checkpoint-only signal — completely independent of the
Nanda training substrate, the optimiser, and the loss curve. It needs
only frozen weights and the training inputs.

MPS NOTE
--------
Representations are moved off the "mps" backend BEFORE promotion to
float64 (MPS has no float64): captured["h"].cpu().double(). All the
class-mean / scatter arithmetic then runs on CPU in double precision.
"""

import torch

_EPS = 1e-12


def _collect_representations_and_labels(model, train_loader, device):
    """Run the model over the full training split and capture, per example,
    the hidden state that feeds `output_head` at the read position
    (sequence index 2, the "=" token). Returns:
        Phi : [N, d_model]  float64, CPU
        Y   : [N]           int64,  CPU   (class labels 0..p-1)

    Identical collection logic to src/predictors/spectral.py — hook on
    model.output_head, take inputs[0][:, 2, :], move off MPS first, then
    promote to float64.
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


def compute_age_metrics_for_checkpoint(model, train_loader, device):
    """Papyan et al. 2020 NC1 variability-collapse metrics for ONE frozen
    model checkpoint. `train_loader` must iterate the full training split
    (for p=113 that is N = floor(0.3 * 113 * 113) = 3830 examples; the
    Nanda-Unified run is full-batch, so one iteration already covers it).

    Returns a dict:
        nc1 : float   Tr(Sigma_W) / (Tr(Sigma_B) + 1e-12)  -- lower = more collapsed
        fn  : float   mean_i || h_i ||_2                    -- feature-norm scale
        N   : int     number of training examples used
    """
    Phi, Y = _collect_representations_and_labels(model, train_loader, device)
    N, _d_model = Phi.shape
    p = model.output_head.out_features - 1  # vocab = p + 1 ("=" token id is p)

    mu_G = Phi.mean(dim=0)  # [d_model]

    within_sq = []   # per-class mean ||h_i - mu_c||^2
    between_sq = []   # per-class ||mu_c - mu_G||^2
    for c in range(p):
        mask = (Y == c)
        n_c = int(mask.sum())
        if n_c == 0:
            continue  # class unseen in this train split — skip, do not bias the mean
        Phi_c = Phi[mask]              # [n_c, d_model]
        mu_c = Phi_c.mean(dim=0)       # [d_model]
        within_sq.append(((Phi_c - mu_c) ** 2).sum(dim=1).mean())
        between_sq.append(((mu_c - mu_G) ** 2).sum())

    tr_w = float(torch.stack(within_sq).mean())
    tr_b = float(torch.stack(between_sq).mean())
    nc1 = tr_w / (tr_b + _EPS)

    fn = float(Phi.norm(dim=1).mean())

    return {"nc1": nc1, "fn": fn, "N": int(N)}


def compute_age_for_model(model, train_loader, device):
    """Thin wrapper kept parallel to
    src/predictors/spectral.py::compute_spectral_for_model. One frozen
    checkpoint in -> one NC1-metrics dict out (see
    compute_age_metrics_for_checkpoint)."""
    return compute_age_metrics_for_checkpoint(model, train_loader, device)
