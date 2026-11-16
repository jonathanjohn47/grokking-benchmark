"""Dropout signals measured on a frozen Nanda transformer.

The shared model (``transformer_four_head.py``) is not changed. It has no
``nn.Dropout`` modules, matching Nanda (2023) and Power et al. (2022).
Dropout is applied only while these functions run, by forward hooks on the
attention-block output (``attn_out``) and the MLP-block output (``mlp_out``).
The hooks are removed before the function returns, and the weights are not
written.

Two signals, both from Salah & Yevick (arXiv:2507.11645):

  1. Dropout variance (their Figure 1). Dropout rate 0.3. Variance of test
     accuracy across 100 independent stochastic passes.
  2. Dropout Robustness Curve area (their Figure 2). Test accuracy at rates
     0.0, 0.1, ..., 0.9. The scalar is the area under that curve
     (trapezoidal rule). A grokked model keeps a high area because accuracy
     survives large dropout. A memorising model collapses at small dropout,
     so the area stays small.

PyTorch ``F.dropout(..., training=True)`` uses inverted dropout: kept units
are scaled by ``1 / (1 - p)``. That is the scaling Nanda used in the dropout
ablations. The module can stay in ``eval`` mode; the hook passes
``training=True`` itself, so the mask is still sampled.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Salah & Yevick Figure 1. Do not change these to chase a pass.
VARIANCE_DROPOUT_RATE = 0.3
VARIANCE_N_SAMPLES = 100

# Salah & Yevick Figure 2. Rates from 0.0 through 0.9 in steps of 0.1.
DRC_RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Rate 0.0 is deterministic, so one pass is enough. Every positive rate is
# the mean of this many independent passes, to smooth the Bernoulli mask.
DRC_PASSES_AT_POSITIVE_RATE = 5

# Logits are read at the "=" token. Training does the same: ``logits[:, 2, :]``.
READ_POSITION = 2

# How many stochastic copies of the test batch go through one forward.
# Each copy receives its own mask, because dropout is elementwise. The
# accuracies are the same random variable as separate passes. The chunk
# only limits activation memory.
_COPIES_PER_FORWARD = 10


class _HookDropout:
    """Inject ``F.dropout`` on the attention block and the MLP block.

    The hook replaces the module output. In ``TransformerFourHead`` that
    output is added back by the residual connection in ``forward``, so the
    dropped tensor is what the residual sees. Nothing is registered on the
    class itself.
    """

    def __init__(self, model, rate):
        self.model = model
        self.rate = float(rate)
        self.handles = []

    def _drop(self, module, inputs, output):
        # ``training=True`` samples a new mask and applies the 1/(1-p) scale
        # even when the surrounding module is in eval mode.
        return F.dropout(output, p=self.rate, training=True)

    def __enter__(self):
        for name in ("attn_out", "mlp_out"):
            if not hasattr(self.model, name):
                raise AttributeError(
                    f"Dropout hooks need model.{name}. "
                    "This measurement is for the Nanda transformer "
                    "(attention block output and MLP block output)."
                )
            module = getattr(self.model, name)
            self.handles.append(module.register_forward_hook(self._drop))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def _batches(data_loader, device):
    """Materialise the loader on ``device``. The Nanda test loader is one batch."""
    batches = []
    for x, y in data_loader:
        batches.append((x.to(device), y.to(device)))
    if not batches:
        raise ValueError("data_loader produced no batches")
    return batches


def _position_logits(logits):
    """Class logits at the '=' position.

    The transformer returns ``(batch, sequence, vocab)``. The MLP used by
    the separate Salah & Yevick run returns ``(batch, vocab)`` already.
    """
    if logits.dim() == 3:
        return logits[:, READ_POSITION, :]
    if logits.dim() == 2:
        return logits
    raise ValueError(f"expected logits of rank 2 or 3, got shape {tuple(logits.shape)}")


def _chunked_batch_accuracy(model, x, y, n_samples):
    """``n_samples`` accuracies on one batch.

    The batch is repeated inside one forward. Dropout then draws an
    independent mask on each copy. Splitting the predictions recovers one
    accuracy per copy.
    """
    batch_size = y.shape[0]
    accuracies = []
    done = 0
    while done < n_samples:
        copies = min(_COPIES_PER_FORWARD, n_samples - done)
        x_rep = x.repeat(copies, *([1] * (x.dim() - 1)))
        y_rep = y.repeat(copies)
        logits = _position_logits(model(x_rep))
        correct = (logits.argmax(dim=-1) == y_rep).view(copies, batch_size)
        accuracies.append(correct.float().mean(dim=1).detach().cpu())
        done += copies
    return torch.cat(accuracies).numpy()


def _accuracies(model, batches, n_samples):
    """Accuracy of each of ``n_samples`` full passes over ``batches``."""
    if len(batches) == 1:
        x, y = batches[0]
        return _chunked_batch_accuracy(model, x, y, n_samples)

    # Several batches: one pass is a loop over the loader, so the copies
    # cannot be paired across batches inside a single forward.
    accuracies = np.empty(n_samples, dtype=np.float64)
    for sample in range(n_samples):
        correct = 0.0
        total = 0
        for x, y in batches:
            logits = _position_logits(model(x))
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += int(y.shape[0])
        accuracies[sample] = correct / total
    return accuracies


def _forward_hook_count(model):
    return sum(len(module._forward_hooks) for module in model.modules())


def _measure(model, data_loader, rate, n_samples, device):
    """Mean test accuracy and the per-pass accuracies, then restore the model.

    Restoration:
      * every forward hook is removed
      * ``train()`` / ``eval()`` is put back
      * if the caller asked for a different device, the module is moved back
    Weights are only read. The forward runs under ``inference_mode``.
    """
    target = torch.device(device)
    original_device = next(model.parameters()).device
    was_training = model.training
    moved = original_device != target
    if moved:
        model.to(target)
    model.eval()
    hooks_before = _forward_hook_count(model)
    try:
        batches = _batches(data_loader, target)
        with _HookDropout(model, rate):
            with torch.inference_mode():
                accuracies = _accuracies(model, batches, n_samples)
        if _forward_hook_count(model) != hooks_before:
            raise RuntimeError("dropout forward hooks were not removed")
    finally:
        model.train(was_training)
        if moved:
            model.to(original_device)
    accuracies = np.asarray(accuracies, dtype=np.float64)
    return float(accuracies.mean()), accuracies


def compute_dropout_variance(
    model,
    data_loader,
    rate=VARIANCE_DROPOUT_RATE,
    n_samples=VARIANCE_N_SAMPLES,
    device="cpu",
):
    """Variance of test accuracy across ``n_samples`` passes at ``rate``.

    Salah & Yevick Figure 1 uses rate 0.3 and 100 passes. The variance is
    the population variance (divide by ``n_samples``, NumPy's default).
    A constant scale factor does not move the epoch of the peak.

    Returns one float. The model is back in its previous train/eval mode,
    on its previous device, with no hooks left attached.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    _mean, accuracies = _measure(model, data_loader, rate, n_samples, device)
    return float(accuracies.var())


def compute_drc_curve(
    model,
    data_loader,
    rates=None,
    device="cpu",
):
    """Mean test accuracy at each dropout rate.

    Rate 0.0 uses one pass (the mask is empty, so the pass is
    deterministic). Each rate above 0.0 is the mean of
    ``DRC_PASSES_AT_POSITIVE_RATE`` passes.

    Returns ``(rates, accuracies)`` as 1-d float arrays, rates ascending.
    ``compute_drc_auc`` integrates this curve. The snapshot plot uses the
    same arrays, so the area and the figure come from one definition.
    """
    if rates is None:
        rates = DRC_RATES
    pairs = []
    for rate in rates:
        rate = float(rate)
        n_samples = 1 if rate <= 0.0 else DRC_PASSES_AT_POSITIVE_RATE
        mean_acc, _accuracies = _measure(
            model, data_loader, rate, n_samples, device
        )
        pairs.append((rate, mean_acc))
    pairs.sort(key=lambda item: item[0])
    rate_arr = np.asarray([item[0] for item in pairs], dtype=np.float64)
    acc_arr = np.asarray([item[1] for item in pairs], dtype=np.float64)
    return rate_arr, acc_arr


def compute_drc_auc(
    model,
    data_loader,
    rates=None,
    device="cpu",
):
    """Area under the Dropout Robustness Curve.

    The curve is accuracy against dropout rate. The area is ``np.trapz``
    (trapezoidal rule) of that curve. One scalar. A larger area means the
    frozen model kept more accuracy as dropout was raised.
    """
    rate_arr, acc_arr = compute_drc_curve(
        model, data_loader, rates=rates, device=device
    )
    # NumPy 2 renamed trapz to trapezoid and kept trapz as an alias.
    trapz = getattr(np, "trapz", None)
    if trapz is None:
        trapz = np.trapezoid
    return float(trapz(acc_arr, rate_arr))
