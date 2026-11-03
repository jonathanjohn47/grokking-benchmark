"""
Commutator Defect predictor (Predictor 8 of 8) — Xu (2026).

SOURCE (locked)
---------------
  Y. Xu (2026), "Low-Dimensional and Transversely Curved Optimization
  Dynamics in Grokking", arXiv:2602.16746 (formula, eta, K = 5, onset).
  Extra reporting percentiles: arXiv:2602.16967.
  Second-order step-order theory: Smith, Dherin, Barrett & De (2021),
  arXiv:2101.12176.

FROZEN RULES
------------
  context.md section dated 2026-11-03
  ("Commutator Defect (predictor 8): frozen rules, set before any code
  or results"). That section is the authority for definition,
  measurement, event rule, and PASS criteria. This file implements
  ONLY the per-snapshot measurement of D. Event detection and
  benchmark wiring come later (Stage 4). Do not change these numbers
  after any D curve has been seen.

WHAT THIS FILE COMPUTES (one frozen checkpoint)
-----------------------------------------------
From snapshot weights theta_k, with two mini-batches A and B:

  theta_AB = theta_k - eta * g_A(theta_k)
                      - eta * g_B(theta_k - eta * g_A(theta_k))
  theta_BA = theta_k - eta * g_B(theta_k)
                      - eta * g_A(theta_k - eta * g_B(theta_k))
  D = ||theta_AB - theta_BA||
      / ( ||eta * g_A(theta_k)|| * ||eta * g_B(theta_k)|| )

g_X is the gradient of the training loss on batch X, over ALL trainable
parameters. Steps are plain gradient steps only — no AdamW, no momentum,
no weight decay, no clipping. eta = 1e-4 is a probe step size; training
itself is not changed.

Because Nanda-Unified training is full-batch, A = B would give D = 0.
So A and B are each 512 examples drawn WITH REPLACEMENT from the
seeded training split (3,830 examples for p = 113). K = 5 fresh
(A, B) pairs per snapshot. We store median D-tilde, 25th / 75th
percentiles, the 5 raw D values, and ||g_A||, ||g_B||.

DEVICE / DTYPE
--------------
CPU, float64 only. The commutator gap is O(eta^2) in size, and Apple
Silicon MPS has no float64. Never run this measurement on MPS.

LOSS
----
Same answer-token cross-entropy as the training loop in
06_Code/scripts/run_nanda_benchmark.py (logits at sequence index 2,
nn.CrossEntropyLoss / F.cross_entropy). Weight decay is NOT added to
this probe loss — the frozen rule asks for the plain training loss
gradient only.

EXPLORATORY (not implemented here, not scored)
----------------------------------------------
  weights-only ||W_Q W_K - W_K W_Q|| — recorded in the frozen rules as
  exploratory only. This module does not compute it.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


def _trainable_param_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect every trainable parameter, by name, as a plain dict.

    We keep the names so torch.func.functional_call can put each tensor
    back in the right place without touching the stored module weights.
    """
    return {name: param for name, param in model.named_parameters()
            if param.requires_grad}


def _flatten(tensors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Stack several tensors into one long vector. Needed for the L2
    norms in the D formula, which are over the FULL parameter vector.
    """
    return torch.cat([t.reshape(-1) for t in tensors])


def _vector_norm(tensors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """L2 norm of a tuple of tensors, treated as one flat vector."""
    return torch.linalg.vector_norm(_flatten(tensors))


def _clone_detached(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fresh leaf copies. Gradients must not flow back into the probe
    model's stored weights, and each pair should start from the same
    theta_0.
    """
    return {name: tensor.detach().clone() for name, tensor in params.items()}


def _answer_token_loss(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Training loss used by run_nanda_benchmark.py, reused here.

    Exact training-loop form (see that file, around the train pass):
        logits = model.forward(x)[:, 2, :]
        loss = CrossEntropyLoss()(logits, y)

    Here we call the same thing through functional_call, so the forward
    uses the parameter dict we pass in, not the module's stored tensors.
    No weight-decay term is added — the frozen rule wants the plain
    loss gradient only.
    """
    outputs = functional_call(model, params, (inputs,))
    logits = outputs[:, 2, :]
    return F.cross_entropy(logits, labels)


def _grads_at(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Gradient of the answer-token loss at the given parameter dict.

    We mark a detached copy of each tensor as a leaf with requires_grad,
    run the loss, then take autograd.grad. The probe model's own
    .parameters() are never updated.
    """
    leaves = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in params.items()
    }
    loss = _answer_token_loss(model, leaves, inputs, labels)
    grads = torch.autograd.grad(
        loss,
        tuple(leaves.values()),
        allow_unused=False,
    )
    # Detach so later arithmetic (theta - eta * g) does not keep a tall
    # graph hanging around for the next pair.
    return tuple(g.detach() for g in grads)


def _params_minus_eta_grad(
    params: Dict[str, torch.Tensor],
    grads: Tuple[torch.Tensor, ...],
    eta: float,
) -> Dict[str, torch.Tensor]:
    """One plain gradient step: theta' = theta - eta * g. No optimiser."""
    names = list(params.keys())
    return {
        name: params[name] - eta * grad
        for name, grad in zip(names, grads)
    }


def _params_difference_norm(
    params_left: Dict[str, torch.Tensor],
    params_right: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """||theta_left - theta_right|| over the full parameter vector."""
    names = list(params_left.keys())
    diffs = tuple(params_left[name] - params_right[name] for name in names)
    return _vector_norm(diffs)


def _json_float(value: torch.Tensor) -> float:
    """Convert a 0-dim tensor to a plain Python float (JSON-safe)."""
    return float(value.detach().cpu().item())


def _nan_summary(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Median / 25th / 75th percentile, ignoring NaNs.

    If every value is NaN, return (None, None, None) — there is nothing
    honest to report as a central value.
    """
    tensor = torch.tensor(values, dtype=torch.float64)
    if torch.all(torch.isnan(tensor)):
        return None, None, None
    # nanquantile(0.5) averages the two middle values when a NaN leaves an even count; nanmedian would pick the lower one.
    median = _json_float(torch.nanquantile(tensor, 0.5))
    q25 = _json_float(torch.nanquantile(tensor, 0.25))
    q75 = _json_float(torch.nanquantile(tensor, 0.75))
    return median, q25, q75


def defect_one_pair(
    theta0: Dict[str, torch.Tensor],
    grad_a_fn,
    grad_b_fn,
    eta: float = 1e-4,
) -> Dict[str, float]:
    """Xu (2026) commutator defect D for one (A, B) pair.

    Parameters
    ----------
    theta0:
        Snapshot weights as a name -> tensor dict. Order of values is the
        order that grad_a_fn / grad_b_fn must return.
    grad_a_fn, grad_b_fn:
        Callables of the form ``fn(params_dict) -> Tuple[Tensor, ...]``.
        Each returns the gradient of the probe loss on its batch, at the
        given parameter dict, in the same key order as theta0.
    eta:
        Probe step size. Default 1e-4 (see compute_commutator_defect).

    Returns
    -------
    dict with keys:
      D            — float; NaN if a gradient norm is zero or non-finite
      grad_norm_A  — ||g_A(theta0)|| (unscaled)
      grad_norm_B  — ||g_B(theta0)|| (unscaled)

    Exact arithmetic (frozen definition):
      g_A, g_B at theta0;
      theta_AB = (theta0 - eta g_A) - eta g_B(theta0 - eta g_A);
      theta_BA likewise with A and B swapped;
      D = ||theta_AB - theta_BA|| / (||eta g_A|| * ||eta g_B||).

    The numerator is evaluated in the algebraically identical gradient-
    difference form (see comment at the numer= line below). The frozen
    definition of D is unchanged.
    """
    g_a = grad_a_fn(theta0)
    g_b = grad_b_fn(theta0)

    norm_a = _vector_norm(g_a)
    norm_b = _vector_norm(g_b)
    grad_norm_a = _json_float(norm_a)
    grad_norm_b = _json_float(norm_b)

    # If a gradient vanishes or is non-finite, D is undefined for this
    # pair. Record NaN — do not crash the run.
    if (
        (not torch.isfinite(norm_a))
        or (not torch.isfinite(norm_b))
        or float(norm_a.item()) == 0.0
        or float(norm_b.item()) == 0.0
    ):
        return {
            "D": float("nan"),
            "grad_norm_A": grad_norm_a,
            "grad_norm_B": grad_norm_b,
        }

    # One-step points needed only to evaluate the OTHER batch's gradient.
    # We do NOT build theta_AB / theta_BA for the numerator (see below).
    theta_after_a = _params_minus_eta_grad(theta0, g_a, eta)
    theta_after_b = _params_minus_eta_grad(theta0, g_b, eta)

    g_b_at_after_a = grad_b_fn(theta_after_a)
    g_a_at_after_b = grad_a_fn(theta_after_b)

    # ----------------------------------------------------------------
    # WHY we do not form ||theta_AB - theta_BA|| from the two weight
    # vectors directly:
    #
    #   theta_AB = (theta0 - eta g_A) - eta g_B(theta0 - eta g_A)
    #   theta_BA = (theta0 - eta g_B) - eta g_A(theta0 - eta g_B)
    #
    # Expand the difference. The bare theta0 terms cancel exactly:
    #
    #   theta_AB - theta_BA
    #     = eta * [ (g_A(theta0 - eta g_B) - g_A(theta0))
    #               - (g_B(theta0 - eta g_A) - g_B(theta0)) ]
    #
    # So ||theta_AB - theta_BA|| = eta * ||(g_A_after_B - g_A)
    #                                      - (g_B_after_A - g_B)||.
    #
    # On a flat-gradient checkpoint (e.g. epoch 6000, ||g|| ~ 7e-6) the
    # true gap is ~1e-15. Weight entries are ~0.1, so float64 rounding
    # of theta_AB / theta_BA is ~1e-14 and SWAMPS the gap if we subtract
    # the two big vectors. Subtracting gradients (size ~||g||) instead
    # of ~0.1-sized weights keeps the cancellation exact and leaves the
    # O(eta^2) signal intact. Frozen D definition is unchanged; only
    # the arithmetic path to the same number changes.
    # ----------------------------------------------------------------
    grad_diff = tuple(
        (g_a_at_b - g_a_0) - (g_b_at_a - g_b_0)
        for g_a_at_b, g_a_0, g_b_at_a, g_b_0 in zip(
            g_a_at_after_b, g_a, g_b_at_after_a, g_b,
        )
    )
    numer = eta * _vector_norm(grad_diff)
    denom = (eta * norm_a) * (eta * norm_b)
    if (not torch.isfinite(numer)) or (not torch.isfinite(denom)) or float(denom.item()) == 0.0:
        return {
            "D": float("nan"),
            "grad_norm_A": grad_norm_a,
            "grad_norm_B": grad_norm_b,
        }

    d_value = numer / denom
    if not torch.isfinite(d_value):
        d_float = float("nan")
    else:
        d_float = _json_float(d_value)

    return {
        "D": d_float,
        "grad_norm_A": grad_norm_a,
        "grad_norm_B": grad_norm_b,
    }


def compute_commutator_defect(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    # At late epochs (e.g. 12000) eta=1e-3 leaves higher-order eta
    # dependence in some pairs; the eta sweep shows the median
    # converging near 91.62 by eta<=2.5e-4, so 1e-4 measures the
    # small-eta limit.
    eta: float = 1e-4,
    k_pairs: int = 5,
    batch_size: int = 512,
    seed: int = 20261103,
) -> Dict[str, Any]:
    """Measure the Xu (2026) commutator defect D at one frozen snapshot.

    Parameters
    ----------
    model:
        Trained (or checkpoint-loaded) model. A deep copy is taken; the
        caller's model is never modified.
    train_inputs, train_labels:
        Full seeded training split tensors (for p = 113 this is 3,830
        examples). Batches A and B are drawn from these with replacement.
    eta:
        Probe step size. Default 1e-4 (small-eta limit). Does not change
        training.
    k_pairs:
        Number of fresh (A, B) pairs. Frozen default K = 5.
    batch_size:
        Examples per batch A or B. Frozen default 512.
    seed:
        Seed for a fresh torch.Generator used only to draw batch indices.
        Frozen default 20261103. Same seed => same five pairs.

    Returns
    -------
    dict with keys:
      median, q25, q75   — over the K values of D (NaNs ignored);
                           None if every D is NaN
      values             — list of K floats (D per pair; may contain NaN)
      grad_norm_A,
      grad_norm_B        — lists of K floats, ||g_A|| and ||g_B|| at theta_0
                           (unscaled; not multiplied by eta)
      eta, k_pairs, batch_size, seed
    """
    # ------------------------------------------------------------------
    # 1. Private CPU float64 copy. Caller model stays untouched.
    #    MPS is never used here — float64 is required for an O(eta^2) gap.
    # ------------------------------------------------------------------
    probe = copy.deepcopy(model).cpu().to(dtype=torch.float64)
    probe.eval()

    # theta_0: all trainable parameters, as a name -> tensor dict.
    theta0 = _clone_detached(_trainable_param_dict(probe))

    # Training split lives on CPU next to the probe. Token ids stay
    # integer; labels become int64 for cross_entropy.
    inputs_cpu = train_inputs.detach().cpu()
    labels_cpu = train_labels.detach().cpu().long()
    n_examples = int(inputs_cpu.shape[0])
    if n_examples < 1:
        raise ValueError(
            "train_inputs is empty; commutator defect needs the training split."
        )

    # Fresh Generator inside the function so the same seed always redraws
    # the same K pairs, and no global RNG state is disturbed.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    values: List[float] = []
    grad_norm_A: List[float] = []
    grad_norm_B: List[float] = []

    for _pair_index in range(int(k_pairs)):
        # --------------------------------------------------------------
        # 2. Draw A and B with replacement (torch.randint). Full-batch
        #    training would make A = B and force D = 0, so this
        #    with-replacement mini-batch draw is the frozen adaptation.
        # --------------------------------------------------------------
        idx_a = torch.randint(
            0, n_examples, (int(batch_size),), generator=generator,
        )
        idx_b = torch.randint(
            0, n_examples, (int(batch_size),), generator=generator,
        )
        x_a, y_a = inputs_cpu[idx_a], labels_cpu[idx_a]
        x_b, y_b = inputs_cpu[idx_b], labels_cpu[idx_b]

        # --------------------------------------------------------------
        # 3–5. Per-pair maths live in defect_one_pair. Grad callables
        #      wrap _grads_at for batch A / batch B (default-arg capture
        #      so the loop variables do not leak across iterations).
        # --------------------------------------------------------------
        pair = defect_one_pair(
            theta0,
            (lambda params, _x=x_a, _y=y_a: _grads_at(probe, params, _x, _y)),
            (lambda params, _x=x_b, _y=y_b: _grads_at(probe, params, _x, _y)),
            eta,
        )
        values.append(pair["D"])
        grad_norm_A.append(pair["grad_norm_A"])
        grad_norm_B.append(pair["grad_norm_B"])

    median, q25, q75 = _nan_summary(values)

    return {
        "median": median,
        "q25": q25,
        "q75": q75,
        "values": values,
        "grad_norm_A": grad_norm_A,
        "grad_norm_B": grad_norm_B,
        "eta": float(eta),
        "k_pairs": int(k_pairs),
        "batch_size": int(batch_size),
        "seed": int(seed),
    }
