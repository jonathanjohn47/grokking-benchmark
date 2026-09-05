"""
Spectral predictor (Predictor 3 of 9).

Signal: per-weight-matrix singular-value statistics (spectral norm,
Frobenius norm, stable rank, effective/entropy rank), tracked across
training checkpoints. The Fourier-based modular-addition circuit Nanda
et al. describe is LOW-RANK relative to a full d_model x d_model weight
matrix (only a handful of frequency components actually matter once the
circuit forms), so the expectation this predictor is testing is that
stable_rank (and effective_rank) DROPS around/before grokking as the
network prunes away the high-rank, memorization-era structure in favour
of the compact circuit — the mirror image of L2-Norm's "weight norm
falls post-grok" story, but at the level of individual weight matrices
rather than the aggregate norm.

MPS fix: torch.linalg.svdvals (and other aten::_linalg_svd* ops) does
not run on the "mps" backend as of this session (confirmed by hitting
the actual NotImplementedError once, same gap already noted in
run_nanda_benchmark.py from the throwaway Spectral stub two sessions
ago). Every SVD call here therefore moves the weight to CPU first via
W.detach().cpu() before calling torch.linalg.svdvals — cheap, since
these are one-off per-checkpoint computations, not something done every
training epoch.
"""

import torch


def compute_spectral_metrics_for_weight(W):
    """W: a 2D weight tensor (any device). Returns a dict of singular-value
    statistics computed on CPU (see module docstring for the MPS reason)."""
    assert W.dim() == 2, f"compute_spectral_metrics_for_weight expects a 2D tensor, got shape {tuple(W.shape)}"

    s = torch.linalg.svdvals(W.detach().cpu())

    spectral_norm = float(s[0])
    fro_norm = float(torch.norm(s))
    stable_rank = float((fro_norm ** 2) / (spectral_norm ** 2 + 1e-12))

    p = s / s.sum()
    ent = float(-(p * torch.log(p + 1e-12)).sum())
    eff_rank = float(torch.exp(torch.tensor(ent)))

    return {
        "spectral_norm": spectral_norm,
        "fro_norm": fro_norm,
        "stable_rank": stable_rank,
        "effective_rank": eff_rank,
        "top5_singular": [float(v) for v in s[:5]],
    }


# The specific 2D weight matrices this predictor tracks, matching
# TransformerFourHead's module names (src/models/transformer_four_head.py).
# position_embedding is deliberately excluded — it is only (3, d_model),
# too small for a meaningful rank signal.
SPECTRAL_MODULE_NAMES = [
    "token_embedding",
    "query",
    "key",
    "value",
    "mlp_in",
    "mlp_out",
    "output_head",
]


def compute_spectral_for_model(model):
    """Returns {module_name: compute_spectral_metrics_for_weight(...)} for
    every module in SPECTRAL_MODULE_NAMES present on `model`."""
    result = {}
    for name in SPECTRAL_MODULE_NAMES:
        module = getattr(model, name, None)
        if module is None:
            continue
        result[name] = compute_spectral_metrics_for_weight(module.weight)
    return result
