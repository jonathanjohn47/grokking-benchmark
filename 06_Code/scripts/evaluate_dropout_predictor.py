#!/usr/bin/env python3
"""Score the Dropout predictor on the saved Nanda-Unified checkpoints.

The transformer is not trained here and its class is not edited. Each
checkpoint is loaded, the two Salah & Yevick signals are measured with
forward hooks, and the hooks are gone again before the next checkpoint.

Signal 1. Variance of test accuracy at dropout rate 0.3, 100 passes.
          Firing epoch = checkpoint where that variance is highest.
Signal 2. Area under the Dropout Robustness Curve (rates 0.0 to 0.9).
          Firing epoch = checkpoint where the derivative of that area
          with respect to epoch is highest (the steepest rise).

Both epochs are scored with the frozen protocol already used for the
other predictors (``evaluate_frozen_protocol`` + ``attach_pass_checks``):

  * a seed leads when P_i < 0.8 * G_i
  * lead_frac = (number of leading seeds) / (number of seeds)
  * lead_frac >= 0.8, Spearman rho at or above the frozen threshold,
    and mean relative lead (G_i - P_i) / G_i inside [0.05, 0.9]
  * at least 4 of 5 seeds must fire

Nothing in those cuts is adjusted if a signal peaks after grokking.
That case is a FAIL.

Run from the repository root:

    python 06_Code/scripts/evaluate_dropout_predictor.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NumPy 2 kept np.trapz as the trapezoidal rule and renamed the public
# function to np.trapezoid. The call stays np.trapz, as specified. The
# deprecation warning would otherwise print once per checkpoint.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"`trapz` is deprecated",
)
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "06_Code"
SRC_ROOT = CODE_ROOT / "src"
SCRIPTS_ROOT = CODE_ROOT / "scripts"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from analyze_nanda_unified import (  # noqa: E402
    RHO_THRESH,
    attach_pass_checks,
    evaluate_frozen_protocol,
)
from data.modular_arithmetic import get_dataloaders  # noqa: E402
from models.transformer_four_head import TransformerFourHead  # noqa: E402
from predictors.dropout import (  # noqa: E402
    DRC_PASSES_AT_POSITIVE_RATE,
    DRC_RATES,
    VARIANCE_DROPOUT_RATE,
    VARIANCE_N_SAMPLES,
    compute_drc_curve,
    compute_dropout_variance,
)

RESULTS_DIR = REPO_ROOT / "08_Experiments" / "results" / "results_per_nanda_power"
MODULUS = 113
D_MODEL = 128
NUM_HEADS = 4
# Saved grid: model_epoch_0.pt, model_epoch_100.pt, ..., model_epoch_40000.pt.
# model_init.pt is the draw before any step, so it is not on this grid.
CHECKPOINT_EPOCHS = list(range(0, 40001, 100))


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_grok_epoch(seed_dir):
    """Primary grok epoch: first epoch with test accuracy strictly above 0.99."""
    with open(seed_dir / "summary.json") as handle:
        summary = json.load(handle)
    grok = summary.get("grok_epoch")
    if grok is None:
        raise ValueError(f"{seed_dir}: summary.json has no grok_epoch")
    return int(grok)


def auc_from_curve(rates, accuracies):
    """Same trapezoidal area as ``compute_drc_auc``, without a second sweep."""
    trapz = getattr(np, "trapz", None)
    if trapz is None:
        trapz = np.trapezoid
    return float(trapz(accuracies, rates))


def nearest_epoch(epochs, target):
    epochs = np.asarray(epochs)
    return int(epochs[int(np.argmin(np.abs(epochs - target)))])


def measure_seed(seed, device):
    """Variance and DRC area at every saved checkpoint of one seed."""
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    ckpt_dir = seed_dir / "checkpoints"
    out_dir = seed_dir / "dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    grok_epoch = load_grok_epoch(seed_dir)
    missing = [
        epoch
        for epoch in CHECKPOINT_EPOCHS
        if not (ckpt_dir / f"model_epoch_{epoch}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"seed {seed}: missing {len(missing)} checkpoints, "
            f"first missing epoch {missing[0]}"
        )

    # Same split the training run used. The test loader is one full batch.
    _train_loader, test_loader = get_dataloaders(
        number=MODULUS,
        batch_size=int(0.3 * MODULUS * MODULUS),
        seed=seed,
    )
    model = TransformerFourHead(
        vocab_size=MODULUS + 1,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
    ).to(device)
    model.eval()

    n_ckpt = len(CHECKPOINT_EPOCHS)
    variance = np.empty(n_ckpt, dtype=np.float64)
    areas = np.empty(n_ckpt, dtype=np.float64)
    # One accuracy per rate, at every checkpoint. Plot 3 reads three rows.
    curve = np.empty((n_ckpt, len(DRC_RATES)), dtype=np.float64)
    rates_saved = None

    started = time.time()
    for index, epoch in enumerate(CHECKPOINT_EPOCHS):
        path = ckpt_dir / f"model_epoch_{epoch}.pt"
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Masks depend on the seed and the checkpoint, not on loop position.
        torch.manual_seed(1_000_003 * seed + int(epoch))

        variance[index] = compute_dropout_variance(
            model,
            test_loader,
            rate=VARIANCE_DROPOUT_RATE,
            n_samples=VARIANCE_N_SAMPLES,
            device=device,
        )
        rates, accuracies = compute_drc_curve(
            model, test_loader, rates=DRC_RATES, device=device
        )
        if rates_saved is None:
            rates_saved = rates
        areas[index] = auc_from_curve(rates, accuracies)
        curve[index] = accuracies

        if device.type == "mps" and index % 50 == 0:
            torch.mps.empty_cache()
        if index % 20 == 0 or index == n_ckpt - 1:
            elapsed = time.time() - started
            print(
                f"[seed {seed}] checkpoint {index + 1}/{n_ckpt}  "
                f"epoch {epoch}  var={variance[index]:.6f}  "
                f"auc={areas[index]:.4f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    epochs = np.asarray(CHECKPOINT_EPOCHS, dtype=np.int64)
    # First maximum. If the highest variance is after grokking, that epoch
    # is still the firing epoch. The gate then fails. We do not pick an
    # earlier bump in order to pass.
    variance_peak_epoch = int(epochs[int(np.argmax(variance))])
    variance_peak_value = float(variance[int(np.argmax(variance))])

    # Derivative of AUC with respect to epoch, one value per checkpoint.
    # The firing epoch is where this derivative is largest.
    derivative = np.gradient(areas, epochs.astype(np.float64))
    inflection_epoch = int(epochs[int(np.argmax(derivative))])
    inflection_value = float(derivative[int(np.argmax(derivative))])

    np.save(out_dir / "epochs.npy", epochs)
    np.save(out_dir / "dropout_variance_history.npy", variance)
    np.save(out_dir / "drc_auc_history.npy", areas)
    np.save(out_dir / "drc_auc_derivative.npy", derivative)
    np.save(out_dir / "drc_rates.npy", rates_saved)
    np.save(out_dir / "drc_accuracy.npy", curve)

    signal = {
        "seed": seed,
        "grok_epoch": grok_epoch,
        "grok_cut": 0.99,
        "variance_rate": VARIANCE_DROPOUT_RATE,
        "variance_n_samples": VARIANCE_N_SAMPLES,
        "drc_rates": [float(r) for r in rates_saved],
        "drc_passes_at_positive_rate": DRC_PASSES_AT_POSITIVE_RATE,
        "variance_peak_epoch": variance_peak_epoch,
        "variance_peak_value": variance_peak_value,
        "drc_inflection_epoch": inflection_epoch,
        "drc_inflection_derivative": inflection_value,
        "n_checkpoints": n_ckpt,
    }
    with open(out_dir / "dropout_signal.json", "w") as handle:
        json.dump(signal, handle, indent=2)

    plot_variance(seed, epochs, variance, variance_peak_epoch, grok_epoch, out_dir)
    plot_auc(seed, epochs, areas, inflection_epoch, grok_epoch, out_dir)
    plot_snapshots(
        seed, epochs, rates_saved, curve, grok_epoch, out_dir
    )
    print(
        f"[seed {seed}] peak variance epoch {variance_peak_epoch}  "
        f"DRC inflection epoch {inflection_epoch}  "
        f"grok epoch {grok_epoch}",
        flush=True,
    )
    return signal


def _log_line(ax, epochs, values, label, colour):
    """Log epoch axis. Epoch 0 cannot sit on a log axis, so the line starts at 1."""
    epochs = np.asarray(epochs, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = epochs >= 1
    ax.plot(epochs[keep], values[keep], color=colour, lw=1.4, label=label)
    ax.set_xscale("log")
    positive = epochs[keep]
    if len(positive):
        ax.set_xlim(float(positive.min()), float(positive.max()))
    ax.set_xlabel("Checkpoint epoch (log scale)")


def _mark(ax, epoch, colour, label):
    if epoch is None:
        return
    if float(epoch) < 1:
        ax.annotate(
            f"{label}: epoch {epoch} (not on the log axis)",
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            color=colour,
            fontsize=8,
        )
        return
    ax.axvline(float(epoch), color=colour, ls="--", lw=1.2, label=f"{label} ({epoch})")


def plot_variance(seed, epochs, variance, peak_epoch, grok_epoch, out_dir):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _log_line(
        ax,
        epochs,
        variance,
        f"Variance of test accuracy (dropout rate {VARIANCE_DROPOUT_RATE})",
        "#6a3d9a",
    )
    _mark(ax, peak_epoch, "#333333", "Variance peak")
    _mark(ax, grok_epoch, "#b00020", "Grok epoch")
    ax.set_ylabel("Variance of test accuracy")
    ax.set_title(f"Seed {seed}: dropout variance")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "dropout_variance_vs_epoch.png", dpi=150)
    plt.close(fig)


def plot_auc(seed, epochs, areas, inflection_epoch, grok_epoch, out_dir):
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    _log_line(ax, epochs, areas, "DRC area (accuracy vs dropout rate)", "#1f4e79")
    _mark(ax, inflection_epoch, "#333333", "DRC inflection")
    _mark(ax, grok_epoch, "#b00020", "Grok epoch")
    ax.set_ylabel("Area under the Dropout Robustness Curve")
    ax.set_title(f"Seed {seed}: dropout robustness area")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "drc_auc_vs_epoch.png", dpi=150)
    plt.close(fig)


def plot_snapshots(seed, epochs, rates, curve, grok_epoch, out_dir):
    """Accuracy against dropout rate at three checkpoints.

    Pre-grok is the saved epoch nearest 0.10 * G. At-grok is the saved
    epoch nearest G. Post-grok is the saved epoch nearest 2 * G, or the
    last checkpoint if that lies past the run.
    """
    targets = {
        "pre-grok": nearest_epoch(epochs, 0.10 * grok_epoch),
        "at-grok": nearest_epoch(epochs, grok_epoch),
        "post-grok": nearest_epoch(epochs, min(2.0 * grok_epoch, float(epochs[-1]))),
    }
    colours = {"pre-grok": "#e07a00", "at-grok": "#b00020", "post-grok": "#1b7f4e"}
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    epoch_to_row = {int(epoch): index for index, epoch in enumerate(epochs)}
    for name, epoch in targets.items():
        row = curve[epoch_to_row[epoch]]
        ax.plot(
            rates,
            row,
            color=colours[name],
            marker="o",
            lw=1.4,
            label=f"{name} (epoch {epoch})",
        )
    ax.set_xlabel("Dropout rate")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Seed {seed}: Dropout Robustness Curve at three epochs")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "drc_snapshots.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "drc_snapshot_epochs.json", "w") as handle:
        json.dump(targets, handle, indent=2)


def _plain_verdict(name, grok_epochs, signal_epochs, per_seed):
    """Frozen protocol. Thresholds come from the shared scorer, not from this file."""
    scored = evaluate_frozen_protocol(grok_epochs, signal_epochs)
    scored = attach_pass_checks(scored)
    rho = scored["rho"]
    return {
        "signal": name,
        "per_seed": per_seed,
        "n_firing": scored["n_firing"],
        "lead_frac": scored["lead_frac"],
        "spearman_rho": rho,
        "mean_relative_lead": scored["mean_rel_lead"],
        "rho_threshold": RHO_THRESH,
        "checks": scored["checks"],
        "verdict": scored["verdict"],
    }


def score_all(signals):
    signals = sorted(signals, key=lambda row: row["seed"])
    grok = [row["grok_epoch"] for row in signals]

    def rows(epoch_key):
        per_seed = []
        epochs = []
        for row in signals:
            signal_epoch = row[epoch_key]
            grok_epoch = row["grok_epoch"]
            relative_lead = (grok_epoch - signal_epoch) / grok_epoch
            per_seed.append(
                {
                    "seed": row["seed"],
                    "grok_epoch": grok_epoch,
                    "signal_epoch": signal_epoch,
                    "relative_lead": relative_lead,
                    "leads": bool(signal_epoch < 0.8 * grok_epoch),
                }
            )
            epochs.append(signal_epoch)
        return per_seed, epochs

    variance_rows, variance_epochs = rows("variance_peak_epoch")
    drc_rows, drc_epochs = rows("drc_inflection_epoch")
    verdict = {
        "results_dir": str(RESULTS_DIR),
        "protocol": {
            "lead_rule": "P_i < 0.8 * G_i",
            "lead_frac": "number of leading seeds divided by number of seeds",
            "relative_lead": "(grok_epoch - signal_epoch) / grok_epoch",
            "mean_relative_lead_window": [0.05, 0.9],
            "spearman_rho_threshold": RHO_THRESH,
            "minimum_firing_seeds": 4,
            "note": (
                "Cuts were not changed after seeing the peaks. "
                "A peak after grokking fails the lead test."
            ),
        },
        "signal_1_dropout_variance": _plain_verdict(
            "dropout_variance_peak", grok, variance_epochs, variance_rows
        ),
        "signal_2_drc_auc": _plain_verdict(
            "drc_inflection", grok, drc_epochs, drc_rows
        ),
    }
    path = RESULTS_DIR / "dropout_verdict.json"
    with open(path, "w") as handle:
        json.dump(verdict, handle, indent=2)
    print(f"Wrote {path}", flush=True)
    for key in ("signal_1_dropout_variance", "signal_2_drc_auc"):
        block = verdict[key]
        print(
            f"{key}: {block['verdict']}  "
            f"lead_frac={block['lead_frac']:.4f}  "
            f"rho={block['spearman_rho']}  "
            f"mean_relative_lead={block['mean_relative_lead']:.4f}",
            flush=True,
        )
    return verdict


def main():
    device = pick_device()
    print(f"Device: {device}", flush=True)
    print(f"Checkpoints per seed: {len(CHECKPOINT_EPOCHS)}", flush=True)
    print(
        f"Variance: rate {VARIANCE_DROPOUT_RATE}, {VARIANCE_N_SAMPLES} passes. "
        f"DRC rates: {list(DRC_RATES)}.",
        flush=True,
    )
    signals = []
    for seed in range(5):
        signals.append(measure_seed(seed, device))
    score_all(signals)


if __name__ == "__main__":
    main()
