import numpy as np
import torch
import torch.nn.functional as F


def _measurement_dropout(model, dropout_rate):
    """Apply dropout on the attention-output and MLP-output branches for one
    measurement. The shared model has no nn.Dropout (Nanda's model has none).
    These hooks are the Dropout predictor's own measurement, at the same two
    residual branches the predictor used before. Rate 0 attaches nothing.
    """
    handles = []
    if dropout_rate <= 0:
        return handles

    def _hook(_module, _inputs, output):
        return F.dropout(output, p=dropout_rate, training=True)

    handles.append(model.attn_out.register_forward_hook(_hook))
    handles.append(model.mlp_out.register_forward_hook(_hook))
    return handles


def _remove_hooks(handles):
    for handle in handles:
        handle.remove()


def compute_accuracy(model, data_loader):
    device = next(model.parameters()).device
    total_correct = 0
    total_samples = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logit = model.forward(x)
        equal_sign_logit = logit[:, 2, :]
        predicted = equal_sign_logit.argmax(dim=1)
        total_correct += (predicted == y).sum().item() 
        total_samples += len(y)
    return total_correct / total_samples if total_samples > 0 else 0.0


# NOTE: the old single-rate compute_dropout_gap(model, data_loader, dropout_rate)
# has been removed. The Dropout predictor is now a full multi-rate sweep only —
# no hardcoded p=0.9 anywhere. Use compute_dropout_gap_multi_rate.


def compute_dropout_gap_multi_rate(model, data_loader, dropout_rates):
    # Step 1: START (inputs: model, data_loader, dropout_rates)

    # Step 2: Setup Evaluation Mode
    model.eval()

    # Step 3: Compute Clean Accuracy (execute ONLY ONCE)
    clean_accuracy = compute_accuracy(model, data_loader)

    # Step 4: Initialize Results Storage
    results = {}

    # Step 5: FOR each rate in dropout_rates
    for dropout_rate in dropout_rates:
        # --- Switch to train mode with dropout active ---
        model.train()
        handles = _measurement_dropout(model, dropout_rate)

        # Compute train accuracy (with dropout)
        train_accuracy = compute_accuracy(model, data_loader)
        _remove_hooks(handles)

        # Compute dropout gap (train_accuracy - clean_accuracy)
        dropout_gap = train_accuracy - clean_accuracy

        # Store results
        results[dropout_rate] = {
            "train_accuracy": train_accuracy,
            "eval_accuracy": clean_accuracy,      # == clean_accuracy (dropout=0, eval mode)
            "dropout_gap": dropout_gap,
            "clean_accuracy": clean_accuracy
        }

    # Step 6: Loop Decision — "More rates?" (handled by for-loop)

    # Step 7: Final Cleanup — restore model to clean eval state
    model.eval()

    # Step 8: RETURN results — END
    return results


def compute_dropout_variance(model, data_loader, n_samples=30, dropout_rate=0.5, device=None):
    """
    Dropout-variance predictor signal (Salah & Yevick, arXiv:2507.11645):
    at a frozen checkpoint, run `n_samples` independent stochastic forward
    passes over the test set with dropout ACTIVE, and look at how much the
    resulting test accuracy jumps around from pass to pass.

    Why rate=0.5: this is the rate at which Salah & Yevick's own Dropout
    Robustness Curve shows the clearest pre-/post-grok separation (their
    Fig. 2 — post-grok accuracy survives dropout up to about rate 0.5),
    and it keeps the primary signal to one rate for compute-cost reasons
    (n_samples stochastic passes at every checkpoint, for every seed, adds
    up fast).

    "Dropout active during eval" means a fresh mask on every forward pass,
    wrapped in torch.no_grad() (a measurement, not a training step). The
    shared model has no dropout layers, so the masks are applied by
    forward hooks on attn_out and mlp_out for this call only.

    Returns (mean_accuracy, variance_of_accuracy) over the n_samples
    passes. Restores model.eval() and removes the hooks before returning.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    handles = _measurement_dropout(model, dropout_rate)

    accuracies = []
    with torch.no_grad():
        for _ in range(n_samples):
            total_correct = 0
            total_samples = 0
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                logit = model.forward(x)
                equal_sign_logit = logit[:, 2, :]
                predicted = equal_sign_logit.argmax(dim=1)
                total_correct += (predicted == y).sum().item()
                total_samples += len(y)
            accuracies.append(total_correct / total_samples if total_samples > 0 else 0.0)

    _remove_hooks(handles)
    model.eval()

    accuracies = np.array(accuracies, dtype=float)
    return float(accuracies.mean()), float(accuracies.var())