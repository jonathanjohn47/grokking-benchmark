"""Dropout variance + Dropout Robustness Curve (Salah & Yevick).

Paper: Ahmed Salah & David Yevick, arXiv:2507.11645 / Neural Processing Letters.

This module extracts ONLY the raw signals:
  1. Dropout Variance — variance of test accuracy across 100 stochastic
     forward passes at dropout rate 0.3.
  2. Dropout Robustness Curve (DRC) — test accuracy vs dropout rate in
     [0.0, 0.1, ..., 0.9].

It is written for the isolated MLPTwoLayer substrate (p = 53). It does
not implement validity gates or the Superiority Index. The old
Transformer hook-based measurement (30 passes / rate 0.5 / 24 log
checkpoints) has been removed.
"""

import numpy as np
import torch

# Paper defaults for the primary variance signal (Fig. 1).
VARIANCE_DROPOUT_RATE = 0.3
VARIANCE_N_SAMPLES = 100

# Paper DRC sweep (Fig. 2): rates from 0.0 to 0.9.
DRC_RATES = [round(0.1 * i, 1) for i in range(10)]  # 0.0, 0.1, ..., 0.9


def compute_accuracy(model, data_loader, device=None):
    """One forward pass over the loader; returns fraction correct."""
    if device is None:
        device = next(model.parameters()).device
    total_correct = 0
    total_samples = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
        predicted = logits.argmax(dim=1)
        total_correct += (predicted == y).sum().item()
        total_samples += len(y)
    return total_correct / total_samples if total_samples > 0 else 0.0


def compute_dropout_variance(
    model,
    data_loader,
    n_samples=VARIANCE_N_SAMPLES,
    dropout_rate=VARIANCE_DROPOUT_RATE,
    device=None,
):
    """Variance of test accuracy across `n_samples` stochastic passes.

    Weights stay frozen. Dropout is active at `dropout_rate` (paper: 0.3).
    Returns (mean_accuracy, variance_of_accuracy). Restores dropout rate
    to 0.0 and model.eval() before returning.
    """
    if device is None:
        device = next(model.parameters()).device

    previous_rate = float(model.dropout.p)
    model.set_dropout_rate(dropout_rate)
    model.train()  # so nn.Dropout samples a fresh mask each pass

    accuracies = []
    with torch.no_grad():
        for _ in range(n_samples):
            accuracies.append(compute_accuracy(model, data_loader, device=device))

    model.set_dropout_rate(previous_rate)
    model.eval()

    accuracies = np.asarray(accuracies, dtype=float)
    return float(accuracies.mean()), float(accuracies.var())


def compute_dropout_robustness_curve(
    model,
    data_loader,
    dropout_rates=None,
    device=None,
):
    """Dropout Robustness Curve: test accuracy at each dropout rate.

    For rate 0.0 the forward pass is deterministic (eval mode). For rate
    > 0 a single stochastic pass is used (paper plots accuracy vs rate).
    Returns a dict {rate: test_accuracy}. Restores dropout to 0.0 and
    model.eval() before returning.
    """
    if dropout_rates is None:
        dropout_rates = list(DRC_RATES)
    if device is None:
        device = next(model.parameters()).device

    previous_rate = float(model.dropout.p)
    results = {}

    for rate in dropout_rates:
        rate = float(rate)
        model.set_dropout_rate(rate)
        if rate <= 0.0:
            model.eval()
        else:
            model.train()
        results[rate] = float(compute_accuracy(model, data_loader, device=device))

    model.set_dropout_rate(previous_rate)
    model.eval()
    return results
