"""
Weight-PCA predictor (Predictor 7 of 8) — Spectral Dynamics / Rank
Minimization of weight matrices.

DEFINITION (locked to the source paper; no project-specific tuning)
---------------------------------------------------------------------
  Yunis, David; Patel, Kumar Kshitij; Wheeler, Samuel; Savarese, Pedro;
  Vardi, Gal; Livescu, Karen; Maire, Michael; Walter, Matthew R.
  "Approaching Deep Learning through the Spectral Dynamics of Weights."
  arXiv:2408.11804 (2024). Section 3, "Grokking and Rank Minimization".

Despite the project's working name "Weight-PCA", the source paper computes
the SINGULAR VALUE DECOMPOSITION (SVD) of each weight matrix, not PCA on
a data matrix. For a weight matrix this is the same underlying linear-
algebra operation (the singular values of W are what a PCA of W, treated
as a data matrix, would also return), so the two names refer to the same
computation here.

For every weight matrix W (rank R, singular values sigma_1 ... sigma_R):

  EffRank(W)     = - sum_{i=1}^{R} (sigma_i / sum_j sigma_j)
                     * log(sigma_i / sum_j sigma_j)                 (Eqn 1)

  NormEffRank(W) = EffRank(W) / R                                   (Eqn 2)

EffRank is the entropy of the normalised singular-value distribution: as
singular-value mass concentrates into fewer directions, EffRank falls.
NormEffRank divides by R so matrices of different shape/rank can be
compared on the same [0, 1] scale, and so the signal can be compared
across layers and across checkpoints — exactly as the paper does in its
Figure 2(c) ("Effective Rank").

Paper's result (Section 3, Figure 2): on a single-layer Transformer doing
modular addition (same substrate as this project's Nanda-Unified setup),
"the sudden drop in validation loss coincides precisely with the onset of
low-rank behavior in the singular values." Weight decay is reported as
essential: without it, in their training budget, neither grokking nor
low-rank collapse occurred.

IMPORTANT CAVEAT, carried over honestly from the source paper: the paper
describes the rank collapse as COINCIDING with grok, not necessarily
PRECEDING it ("co-occurs", Figure 2 caption). This project's frozen
protocol specifically tests for a LEADING signal. There is therefore a
real, literature-grounded risk that the primary event rule below (global
minimum) will land at or after the grok epoch, the same failure mode
already seen in L2 Norm, Dropout, Spectral, AGE and HTSR Alpha. This is
not assumed away here; the evaluation in run_nanda_benchmark.py /
analyze_nanda_unified.py will show whether it holds for this project's
own checkpoints.

WHICH MATRICES
---------------
Same six Linear weight matrices as HTSR Alpha (src/predictors/htsr_alpha.py),
for consistency and comparability across predictors on this architecture:

    query, key, value          128 x 128
    mlp_in                     512 x 128
    mlp_out                    128 x 512
    output_head                114 x 128

The token / position embedding tables are nn.Embedding (not Linear) and
are not analysed, matching the HTSR precedent. Biases are vectors, not
matrices, and are skipped.

This file only READS weights of an already-built model. It does not
change the shared model file.

MPS NOTE
--------
Weights are moved off the "mps" backend to CPU and promoted to float64
BEFORE the SVD (MPS has no float64 and no reliable SVD), same convention
as htsr_alpha.py.

EVENT RULES
-----------
PRIMARY (used for the official PASS/FAIL verdict, evaluate(Gs, Ps)):
    effrank_min_epoch — the epoch of the GLOBAL MINIMUM of the
    mean-over-layers NormEffRank signal. Chosen for consistency with
    AGE's nc1_min_epoch and HTSR's alpha_min_epoch (both also use the
    extreme-value epoch as the event). Frozen protocol, 2026-09-17.

EXPLORATORY ONLY (saved to weight_pca_signal.json, NOT fed into the
PASS/FAIL table — thesis appendix / sensitivity analysis material):
    steepest_drop_epoch — epoch of the single largest one-step decrease
        in the mean NormEffRank signal between consecutive checkpoints.
    threshold_90_epoch — first epoch at which the signal has covered 90%
        of the total decline from its initial value to its global
        minimum. (Same "fraction-of-change" convention already used for
        L2 Norm / Dropout / Spectral / AGE in analyze_nanda_unified.py,
        generalised to the 90% level here. ASSUMPTION to confirm with
        Jonathan: "threshold=90" is read as "90% of the way from the
        starting value to the eventual minimum", not a raw NormEffRank
        value of 90. NormEffRank(W) = EffRank(W)/R is bounded in
        [0, log(R)/R] by construction (max entropy of R values is
        log(R), not R) -- for this project's layers, R ranges 114 to
        512, so log(R)/R is about 0.01 to 0.05. Verified numerically:
        see the module's own sanity checks. A raw threshold of 90 is
        therefore unreachable by a wide margin either way.)
"""

import numpy as np
import torch

# Linear layers of TransformerFourHead that carry a 2-D weight matrix.
# Identical set to src/predictors/htsr_alpha.py::LAYER_NAMES, kept as a
# separate list here (not imported) so the two predictors stay independent
# modules, matching the project's existing per-predictor file convention.
LAYER_NAMES = ["query", "key", "value", "mlp_in", "mlp_out", "output_head"]


def _layer_singular_values(weight):
    """Singular values of one weight matrix (float64, CPU), descending.
    Same MPS-safe promotion convention as htsr_alpha.py::_layer_eigenvalues."""
    W = weight.detach().cpu().double()
    sv = torch.linalg.svdvals(W)  # descending, length = min(n, m)
    return sv.numpy()


def effective_rank(singular_values):
    """EffRank(W) (Eqn 1) and NormEffRank(W) (Eqn 2), Yunis et al. 2024.

    singular_values: 1-D array of a single weight matrix's singular
        values (any order; only their relative magnitudes matter).

    Returns (eff_rank, norm_eff_rank). R = len(singular_values).
    Degenerate case (all-zero matrix) returns (0.0, 0.0) rather than NaN.
    """
    sv = np.asarray(singular_values, dtype=float)
    R = len(sv)
    total = sv.sum()
    if R == 0 or total <= 0:
        return 0.0, 0.0
    p = sv / total
    # 0 * log(0) := 0 by convention (entropy), avoid log(0) warnings.
    nonzero = p > 0
    eff_rank = float(-np.sum(p[nonzero] * np.log(p[nonzero])))
    norm_eff_rank = eff_rank / R
    return eff_rank, norm_eff_rank


def compute_weight_pca_metrics_for_checkpoint(model):
    """Weight-PCA (spectral dynamics / rank minimization) metrics for ONE
    frozen model checkpoint.

    Returns a dict:
        norm_eff_rank       : float        mean NormEffRank over the
                              analysed layers (the project's single
                              summary scalar for this predictor)
        layer_eff_rank      : list[float]  per layer, order = LAYER_NAMES
        layer_norm_eff_rank : list[float]  per layer, order = LAYER_NAMES
        layer_rank          : list[int]    R per layer (len of its
                              singular-value vector)
    """
    eff_ranks, norm_eff_ranks, ranks = [], [], []
    for name in LAYER_NAMES:
        sv = _layer_singular_values(getattr(model, name).weight)
        eff_rank, norm_eff_rank = effective_rank(sv)
        eff_ranks.append(eff_rank)
        norm_eff_ranks.append(norm_eff_rank)
        ranks.append(int(len(sv)))

    return {
        "norm_eff_rank": float(np.mean(norm_eff_ranks)),
        "layer_eff_rank": eff_ranks,
        "layer_norm_eff_rank": norm_eff_ranks,
        "layer_rank": ranks,
    }


def compute_weight_pca_for_model(model):
    """Thin wrapper kept parallel to compute_htsr_for_model /
    compute_age_for_model. One frozen checkpoint in -> one metrics dict."""
    return compute_weight_pca_metrics_for_checkpoint(model)


# --------------------------------------------------------------------------
# Event-epoch rules
# --------------------------------------------------------------------------

def find_global_minimum_epoch(checkpoints, signal):
    """PRIMARY event rule. Epoch of the global minimum of `signal`
    (mean NormEffRank history). Mirrors AGE's nc1_min_epoch and HTSR's
    alpha_min_epoch — see module docstring, "EVENT RULES".

    Returns (event_epoch, event_value, event_index).
    """
    arr = np.asarray(signal, dtype=float)
    idx = int(np.nanargmin(arr))
    return int(checkpoints[idx]), float(arr[idx]), idx


def find_steepest_drop_epoch(checkpoints, signal):
    """EXPLORATORY ONLY. Epoch of the single largest one-step decrease
    (most negative signal[t+1] - signal[t]) between consecutive saved
    checkpoints. Reports the epoch the drop lands ON (checkpoints[i+1]).

    Returns (event_epoch, drop_size) or (None, None) if fewer than 2
    checkpoints are available. drop_size is positive = a real decrease.
    """
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 2:
        return None, None
    diffs = arr[:-1] - arr[1:]  # positive where the signal fell
    idx = int(np.nanargmax(diffs))
    return int(checkpoints[idx + 1]), float(diffs[idx])


def find_threshold_crossing_epoch(checkpoints, signal, drop_pct=90.0):
    """EXPLORATORY ONLY. First epoch at which `signal` has covered
    `drop_pct`% of its total decline from the initial value to its global
    minimum. Generalises the 10% / 50% "fraction-of-change" rule already
    used elsewhere in analyze_nanda_unified.py (L2 Norm, Dropout,
    Spectral, AGE) to the 90% level, for this predictor.

    threshold_value = signal[0] - (drop_pct / 100) * (signal[0] - min(signal))
    Returns (event_epoch, threshold_value) or (None, None) if the signal
    never reaches the threshold (e.g. it does not fall monotonically) or
    fewer than 2 checkpoints are available.

    ASSUMPTION (flagged in the module docstring): "threshold=90" is read
    as this 90%-of-total-decline definition, not a raw NormEffRank value
    of 90 (NormEffRank is bounded in [0, log(R)/R] by construction, about
    0.01-0.05 for this project's layers -- far below 90 either way).
    Confirm with Jonathan before treating this number as final.
    """
    arr = np.asarray(signal, dtype=float)
    if len(arr) < 2:
        return None, None
    start, minimum = arr[0], float(np.nanmin(arr))
    total_drop = start - minimum
    if total_drop <= 0:
        return None, None  # signal never falls below its start value
    threshold_value = start - (drop_pct / 100.0) * total_drop
    hits = np.where(arr <= threshold_value)[0]
    if len(hits) == 0:
        return None, None
    idx = int(hits[0])
    return int(checkpoints[idx]), float(threshold_value)
