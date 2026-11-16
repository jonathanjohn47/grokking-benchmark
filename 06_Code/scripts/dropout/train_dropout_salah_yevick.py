#!/usr/bin/env python3
"""Isolated Salah & Yevick Dropout training + signal extraction.

Trains the paper's 2-layer MLP on (a + b) mod 53 and, every epoch, records:
  - Dropout Variance (100 stochastic passes at rate 0.3)
  - Dropout Robustness Curve (test accuracy for rates 0.0 … 0.9)

No validity gates. No Superiority Index. Does not touch the Nanda-Unified
Transformer runner or any other predictor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "06_Code"
SRC_ROOT = CODE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.modular_arithmetic_p53 import get_dataloaders_p53  # noqa: E402
from models.mlp_two_layer import MLPTwoLayer, cross_entropy_loss  # noqa: E402

# This script trains the paper's MLP, which owns an nn.Dropout module.
# The Nanda transformer predictor in predictors/dropout.py is a different
# measurement (forward hooks, no module in the transformer). These helpers
# stay here so the two substrates do not share a dropout implementation.
VARIANCE_DROPOUT_RATE = 0.3
VARIANCE_N_SAMPLES = 100
DRC_RATES = [round(0.1 * i, 1) for i in range(10)]


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
    """Variance of test accuracy across ``n_samples`` stochastic passes.

    Weights stay frozen. Dropout is active at ``dropout_rate`` (paper: 0.3).
    Returns (mean_accuracy, variance_of_accuracy). Restores the dropout rate
    and model.eval() before returning.
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

    For rate 0.0 the forward pass is deterministic (eval mode). For a rate
    above 0.0 a single stochastic pass is used. Returns ``{rate: accuracy}``.
    Restores dropout to the previous rate and model.eval() before returning.
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


DEFAULT_CONFIG = CODE_ROOT / "configs" / "dropout_salah_yevick.yaml"
DEFAULT_OUTPUT = (
    REPO_ROOT / "08_Experiments" / "results" / "dropout_salah_yevick"
)


def load_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def grok_epoch_from(test_acc_history, threshold=0.99):
    """First epoch where test accuracy strictly exceeds threshold, else None."""
    for epoch, acc in enumerate(test_acc_history):
        if acc > threshold:
            return epoch
    return None


def seed_dir(output_dir, seed):
    path = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "dropout"), exist_ok=True)
    os.makedirs(os.path.join(path, "training"), exist_ok=True)
    return path


def train_one_seed(seed, cfg, args, device):
    out_dir = seed_dir(args.output_dir, seed)
    drop_dir = os.path.join(out_dir, "dropout")
    train_dir = os.path.join(out_dir, "training")

    modulus = int(cfg["task"]["modulus"])
    train_fraction = float(cfg["task"]["train_fraction"])
    embed_dim = int(cfg["architecture"]["embed_dim"])
    hidden_dim = int(cfg["architecture"]["hidden_dim"])
    init_scale = float(cfg["init"]["init_scale"])
    lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    betas = tuple(cfg["optimizer"]["betas"])
    epochs = int(args.epochs)
    variance_rate = float(cfg["dropout"]["variance_rate"])
    n_samples = int(cfg["dropout"]["n_samples"])
    drc_rates = [float(r) for r in cfg["dropout"]["drc_rates"]]
    grok_threshold = float(cfg["evaluation"]["grok_threshold"])

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, test_loader = get_dataloaders_p53(
        seed=seed, train_fraction=train_fraction, modulus=modulus
    )

    model = MLPTwoLayer(
        modulus=modulus,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout_rate=0.0,
        init_scale=init_scale,
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
    )

    train_acc_history = []
    test_acc_history = []
    loss_history = []
    variance_history = []
    variance_mean_acc_history = []
    drc_history = {rate: [] for rate in drc_rates}
    epoch_index = []

    started = time.time()
    log_every = max(1, epochs // 40)

    for epoch in range(epochs):
        # ---- training step (full batch, dropout off) ----
        model.set_dropout_rate(0.0)
        model.train()
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        train_correct = (logits.argmax(dim=1) == y).sum().item()
        train_acc = train_correct / len(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- clean test accuracy (dropout off) ----
        model.eval()
        model.set_dropout_rate(0.0)
        with torch.no_grad():
            test_acc = compute_accuracy(model, test_loader, device=device)

        # ---- every-epoch Dropout Variance (paper Fig. 1) ----
        mean_acc, variance = compute_dropout_variance(
            model,
            test_loader,
            n_samples=n_samples,
            dropout_rate=variance_rate,
            device=device,
        )

        # ---- every-epoch DRC sweep (paper Fig. 2) ----
        drc = compute_dropout_robustness_curve(
            model, test_loader, dropout_rates=drc_rates, device=device
        )

        train_acc_history.append(float(train_acc))
        test_acc_history.append(float(test_acc))
        loss_history.append(float(loss.item()))
        variance_history.append(float(variance))
        variance_mean_acc_history.append(float(mean_acc))
        epoch_index.append(epoch)
        for rate in drc_rates:
            drc_history[rate].append(float(drc[rate]))

        if epoch % log_every == 0 or epoch == epochs - 1:
            elapsed = time.time() - started
            print(
                f"[seed {seed}] epoch {epoch:>5}/{epochs}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  "
                f"var@{variance_rate}={variance:.6f}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    grok_epoch = grok_epoch_from(test_acc_history, threshold=grok_threshold)
    peak_idx = int(np.argmax(variance_history)) if variance_history else None
    variance_peak_epoch = epoch_index[peak_idx] if peak_idx is not None else None
    variance_peak_value = (
        float(variance_history[peak_idx]) if peak_idx is not None else None
    )

    # ---- persist training histories ----
    np.save(os.path.join(train_dir, "train_acc_history.npy"), np.array(train_acc_history))
    np.save(os.path.join(train_dir, "test_acc_history.npy"), np.array(test_acc_history))
    np.save(os.path.join(train_dir, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(out_dir, "seed.npy"), np.array([seed]))

    # ---- persist Dropout signals (raw only) ----
    np.save(os.path.join(drop_dir, "epochs.npy"), np.array(epoch_index, dtype=int))
    np.save(
        os.path.join(drop_dir, "dropout_variance_history.npy"),
        np.array(variance_history, dtype=float),
    )
    np.save(
        os.path.join(drop_dir, "dropout_variance_mean_acc_history.npy"),
        np.array(variance_mean_acc_history, dtype=float),
    )
    np.save(
        os.path.join(drop_dir, "drc_rates.npy"),
        np.array(drc_rates, dtype=float),
    )
    # shape: [n_rates, n_epochs]
    drc_matrix = np.array([drc_history[r] for r in drc_rates], dtype=float)
    np.save(os.path.join(drop_dir, "drc_test_acc_by_rate.npy"), drc_matrix)

    signal = {
        "paper": "Salah & Yevick, arXiv:2507.11645",
        "substrate": "mlp_two_layer_p53",
        "modulus": modulus,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "variance_rate": variance_rate,
        "n_samples": n_samples,
        "drc_rates": drc_rates,
        "variance_peak_epoch": variance_peak_epoch,
        "variance_peak_value": variance_peak_value,
        "grok_epoch": grok_epoch,
        "grok_threshold": grok_threshold,
        "final_train_acc": float(train_acc_history[-1]),
        "final_test_acc": float(test_acc_history[-1]),
        "note": (
            "Signal extraction only. Validity gates and Superiority Index "
            "are not computed in this pipeline."
        ),
    }
    with open(os.path.join(drop_dir, "dropout_signal.json"), "w") as handle:
        json.dump(signal, handle, indent=2)

    summary = {
        "seed": seed,
        "epochs": epochs,
        "modulus": modulus,
        "architecture": "mlp_two_layer",
        "grok_epoch": grok_epoch,
        "final_train_acc": float(train_acc_history[-1]),
        "final_test_acc": float(test_acc_history[-1]),
        "dropout_variance_peak_epoch": variance_peak_epoch,
        "dropout_variance_peak_value": variance_peak_value,
        "wall_time_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[seed {seed}] done in {summary['wall_time_sec']}s  "
        f"grok_epoch={grok_epoch}  "
        f"variance_peak_epoch={variance_peak_epoch}  "
        f"final_test_acc={summary['final_test_acc']:.4f}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Isolated Salah & Yevick Dropout pipeline "
            "(MLP p=53, every-epoch variance + DRC). "
            "Does not run the Nanda-Unified benchmark."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of seeds; seed i uses torch.manual_seed(i)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override yaml training.epochs")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is None:
        args.epochs = int(cfg["training"]["epochs"])

    # Sanity: paper constants must match what we claim to replicate.
    assert int(cfg["task"]["modulus"]) == 53
    assert float(cfg["dropout"]["variance_rate"]) == VARIANCE_DROPOUT_RATE
    assert int(cfg["dropout"]["n_samples"]) == VARIANCE_N_SAMPLES
    assert [float(r) for r in cfg["dropout"]["drc_rates"]] == [
        float(r) for r in DRC_RATES
    ]

    device = pick_device()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(
        f"Substrate: MLPTwoLayer  p=53  train_fraction="
        f"{cfg['task']['train_fraction']}  epochs={args.epochs}"
    )
    print(
        f"Signal: variance rate={cfg['dropout']['variance_rate']}  "
        f"n_samples={cfg['dropout']['n_samples']}  "
        f"DRC rates={cfg['dropout']['drc_rates']}  "
        f"(every epoch)"
    )

    summaries = []
    for seed in range(args.seeds):
        summaries.append(train_one_seed(seed, cfg, args, device))

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as handle:
        json.dump(
            {
                "n_seeds": len(summaries),
                "epochs": args.epochs,
                "config": os.path.relpath(args.config, REPO_ROOT),
                "paper": "Salah & Yevick, arXiv:2507.11645",
                "seeds": summaries,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
