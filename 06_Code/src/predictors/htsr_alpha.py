"""
HTSR Alpha predictor (Predictor 5 of 9) — Heavy-Tailed Self-Regularization.

DEFINITION (locked to the original papers; no project-specific tuning)
---------------------------------------------------------------------
  Martin, Charles H.; Mahoney, Michael W. "Implicit self-regularization in
  deep neural networks: evidence from random matrix theory and
  implications for learning." Journal of Machine Learning Research 22
  (165), 1-73 (2021).  (HTSR theory; the WeightWatcher method.)

For every weight matrix W (N x M, N >= M after transposing if needed):

  1. X = W^T W / N            correlation matrix of the layer
  2. lambda_i                 eigenvalues of X  (= squared singular values
                              of W, divided by N)
  3. alpha                    exponent of the power-law tail of the
                              empirical spectral density (ESD):
                                  rho(lambda) ~ lambda^(-alpha),
                                  lambda >= xmin

The tail is fitted exactly as WeightWatcher does it (Clauset, Shalizi &
Newman 2009, "Power-law distributions in empirical data", SIAM Review):

  - alpha  by MAXIMUM LIKELIHOOD (not a log-log regression line),
  - xmin   chosen AUTOMATICALLY as the tail start whose fitted power law
           has the smallest Kolmogorov-Smirnov distance D to the data.

There is NO hand-picked quantile, threshold, or cut-off in this file. The
fit is delegated to the `powerlaw` package, the same engine WeightWatcher
uses. The model's summary alpha is the MEAN of alpha over the analysed
layers (Martin & Mahoney's "average alpha" quality metric); per-layer
alpha, xmin, D and lambda_max are also returned so any other published
aggregate (e.g. the alpha-hat = alpha * log10(lambda_max) weighting of
Martin et al. 2021, Nature Communications 12:4122) can be rebuilt later.

WHICH MATRICES
--------------
Like WeightWatcher (which analyses nn.Linear / Conv layers), only the
Linear weight matrices are analysed. For the shared TransformerFourHead:

    query, key, value          128 x 128
    mlp_in                     512 x 128
    mlp_out                    128 x 512
    output_head                114 x 128

The token / position embedding tables are nn.Embedding (not Linear) and
are not analysed. Biases are vectors, not matrices, and are skipped.

This file only READS weights of an already-built model. It does not
change the shared model file.

MPS NOTE
--------
Weights are moved off the "mps" backend to CPU and promoted to float64
BEFORE the SVD (MPS has no float64 and no reliable SVD). Everything here
runs on CPU in double precision.
"""

import warnings

import numpy as np
import torch

# Linear layers of TransformerFourHead that carry a 2-D weight matrix.
LAYER_NAMES = ["query", "key", "value", "mlp_in", "mlp_out", "output_head"]


def _layer_eigenvalues(weight):
    """Eigenvalues of X = W^T W / N for one weight matrix (float64, CPU),
    sorted ascending. N is the larger dimension, so X is the smaller
    (M x M) correlation matrix — same convention as WeightWatcher."""
    W = weight.detach().cpu().double()
    n, m = W.shape
    if n < m:
        W = W.T
        n, m = m, n
    sv = torch.linalg.svdvals(W)              # [m], descending
    evals = (sv ** 2) / n
    return np.sort(evals.numpy())


def fit_powerlaw_alpha(evals):
    """Fit the ESD tail of ONE layer (Clauset-Shalizi-Newman via the
    `powerlaw` package: ML alpha, xmin picked by minimum KS distance).

    Returns (alpha, xmin, D, n_tail). All NaN / 0 if the fit fails."""
    import powerlaw  # local import: keeps module import cheap and optional

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                fit = powerlaw.Fit(evals, xmin=None, verbose=False)
                alpha = float(fit.alpha)
                xmin = float(fit.xmin)
                D = float(fit.D)
    except Exception:
        return float("nan"), float("nan"), float("nan"), 0
    n_tail = int((evals >= xmin).sum())
    return alpha, xmin, D, n_tail


def compute_htsr_metrics_for_checkpoint(model):
    """HTSR Alpha metrics for ONE frozen model checkpoint.

    Returns a dict:
        alpha       : float        mean alpha over the analysed layers
        layer_alpha : list[float]  per layer, order = LAYER_NAMES
        layer_xmin  : list[float]  fitted tail start per layer
        layer_D     : list[float]  KS distance of the fit per layer
        layer_lmax  : list[float]  largest eigenvalue per layer
        layer_ntail : list[int]    number of eigenvalues in the tail
    """
    alphas, xmins, Ds, lmaxs, ntails = [], [], [], [], []
    for name in LAYER_NAMES:
        evals = _layer_eigenvalues(getattr(model, name).weight)
        alpha, xmin, D, n_tail = fit_powerlaw_alpha(evals)
        alphas.append(alpha)
        xmins.append(xmin)
        Ds.append(D)
        lmaxs.append(float(evals[-1]))
        ntails.append(n_tail)

    return {
        "alpha": float(np.nanmean(alphas)),
        "layer_alpha": alphas,
        "layer_xmin": xmins,
        "layer_D": Ds,
        "layer_lmax": lmaxs,
        "layer_ntail": ntails,
    }


def compute_htsr_for_model(model):
    """Thin wrapper kept parallel to compute_spectral_for_model and
    compute_age_for_model. One frozen checkpoint in -> one metrics dict."""
    return compute_htsr_metrics_for_checkpoint(model)
