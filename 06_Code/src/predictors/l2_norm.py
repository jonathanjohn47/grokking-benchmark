"""L2 Norm predictor: Nanda et al. (2023) Figure 7, plus one own-choice rule.

S(t) is the total sum of squared weights over every parameter at epoch t.
Nanda et al. plot that total. They do not split it by layer, and they do
not use the parameter L2 norm (the square root of S).

No published paper gives a detection rule for this curve. The confirmed-peak
rule below is this project's own reading of "S rises, then stays below the
highest value it has reached." k = 100 was fixed before scoring.
"""

import numpy as np
import torch

# No published paper gives a confirmation window for the weight-norm peak.
# 100 epochs is this project's own choice. It equals Nanda et al.'s save_every.
CONFIRMED_PEAK_K = 100


def compute_sum_of_squared_weights(model):
    """Total sum of squared weights over all parameters (Nanda et al. 2023, Figure 7)."""
    total = 0.0
    for p in model.parameters():
        total += torch.sum(p.detach() ** 2).item()
    return total


def sum_w2_peak(sum_w2):
    """Global peak of S(t): first epoch of the maximum, and that value."""
    series = np.asarray(sum_w2, dtype=float)
    if series.size == 0:
        return None, None
    peak_epoch = int(np.argmax(series))
    return peak_epoch, float(series[peak_epoch])


def detect_confirmed_sum_w2_peak(sum_w2, k=CONFIRMED_PEAK_K):
    """First epoch when S has stayed strictly below its running maximum for k epochs.

    S(t) is the total sum of squared weights at epoch t.
    M(t) is the running maximum of S over epochs 0..t.
    The returned epoch is the first t such that S(t-j) < M(t-j) for every j = 0..k-1.
    A later higher peak does not move this epoch: the first completed window is the event.
    Returns None when the window never completes.
    k is CONFIRMED_PEAK_K unless a test passes another window.
    """
    if k < 1:
        raise ValueError("confirmation window k must be at least 1")
    series = np.asarray(sum_w2, dtype=float)
    if series.size < k:
        return None
    running_max = np.maximum.accumulate(series)
    below = series < running_max
    run = 0
    for t, is_below in enumerate(below):
        if is_below:
            run += 1
            if run >= k:
                return int(t)
        else:
            run = 0
    return None
