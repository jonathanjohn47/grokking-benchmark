#!/usr/bin/env python3
"""
run_nanda_benchmark.py
======================

Master runner for the Nanda-Unified grokking-predictor benchmark, for the
predictors that actually exist so far: ONLY L2-Norm and Dropout.

Every constant comes from configs/nanda_unified.yaml (the single source of
truth). Nothing is hardcoded here about the task or the optimiser. The
"=" token id is the modulus from the config (p = 113), never a fixed
value. Predictors 3-9 (Spectral, HTSR Alpha, AGE, Weight-PCA, Higher-MI,
Commutator Defect, ...) are NOT built yet and are NOT called.

Per seed it:
  - sets torch.manual_seed(seed)
  - builds the dataloaders for p from the config (full-batch)
  - builds TransformerFourHead with the small init (N(0, 0.8/sqrt(d)))
  - trains full-batch with AdamW(lr, betas, weight_decay) from the config
  - logs every epoch via unified_measurements.PredictorMeasurements:
      train_acc, test_acc, loss, l2_norm, sum_w2, per-module sum_w2
      (5 groups: token_embedding, position_embedding, attention_qkv,
       mlp, output_head)
  - after training: computes the L2-Norm predictor signals (MA crossover,
    MA-of-MA zero crossing, noise floor) and the Dropout gap sweep, and
    saves them under output_dir/seed_{seed}/

After all seeds it aggregates: mean grok epoch, per-seed predictor
signals, and a post-grok limit-cycle check across the seeds.

Resume: a seed whose output_dir/seed_{seed}/summary.json exists is
skipped and its summary reloaded.

Usage:
    python run_nanda_benchmark.py --seeds 5 --epochs 40000 \
        --output_dir results/nanda_unified --config configs/nanda_unified.yaml

    # quick smoke:
    python run_nanda_benchmark.py --seeds 1 --epochs 100 \
        --output_dir results/test_smoke
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data.modular_arithmetic import get_dataloaders                       # noqa: E402
from models.transformer_four_head import TransformerFourHead              # noqa: E402
from predictors.l2_norm import (                                          # noqa: E402
    compute_l2_norm,
    compute_sum_of_squared_weights,
    compute_per_module_sum_of_squared_weights,
    compute_fast_slow_moving_averages,
    detect_ma_crossover,
    compute_ma_of_slow_ma,
    compute_noise_floor,
    detect_ma_of_ma_zero_crossing,
)
from predictors.dropout import (                                          # noqa: E402
    compute_dropout_gap_multi_rate,
    compute_dropout_variance,
)
from predictors.spectral import compute_spectral_for_model                # noqa: E402
from unified_measurements import PredictorMeasurements                    # noqa: E402

# The grok epoch is the first epoch test accuracy exceeds this (same
# threshold nanda_l2_p113 uses).
GROK_ACC_THRESHOLD = 0.9
# Dropout predictor: full multi-rate sweep, no single "primary" rate.
DROPOUT_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]
# Console cadence (task: print every 1000 epochs).
LOG_EVERY = 1000
# L2-Norm predictor window parameters (identical to src/train_four_head.py).
L2_FAST_WINDOW = 50
L2_SLOW_WINDOW = 200
L2_MA_OF_MA_FAST_WINDOW = 20
L2_SKIP_EPOCHS = 100
L2_QUIET_EPOCH_CUTOFF = 90

# Dropout-Variance predictor (Salah & Yevick, arXiv:2507.11645): at a
# checkpoint, run n_samples stochastic forward passes with dropout active
# and look at the variance of test accuracy across those passes. rate=0.5
# is where their Dropout Robustness Curve shows the clearest pre/post-grok
# separation (see src/predictors/dropout.py's compute_dropout_variance
# docstring for the full rationale).
#
# Deviations from the paper, and why (all for MPS wall-clock budget, same
# spirit as this project's other predictors — no change to what is being
# measured, only how densely/expensively):
#   - paper: 100 stochastic passes per checkpoint, rate=0.3 for their
#     Figure 1 sweep -> here: n_samples=30, rate=0.5 (their own Figure 2
#     DRC shows rate=0.5 gives the clearest pre-/post-grok separation, so
#     the primary signal uses that rate rather than 0.3).
#   - paper: variance measured at (implicitly) every logged epoch -> here:
#     DROPOUT_VARIANCE_NUM_CHECKPOINTS=24 log-uniform checkpoints across
#     training (see dropout_variance_checkpoint_schedule below), since
#     40000 epochs x 30 passes at every epoch is not affordable on MPS.
DROPOUT_VARIANCE_RATE = 0.5
DROPOUT_VARIANCE_N_SAMPLES = 30
DROPOUT_VARIANCE_NUM_CHECKPOINTS = 24
DROPOUT_VARIANCE_NUM_DRC_CHECKPOINTS = 5

# Names understood by --predictors / --overwrite, and the summary.json key
# each one's presence is checked against (see get_done_predictors below).
PREDICTOR_SUMMARY_KEY = {
    "l2": "l2_predictor",
    "dropout_gap": "dropout_final_gap_by_rate",
    "dropout_variance": "dropout_variance_predictor",
    "spectral": "spectral_predictor",
}
ALL_PREDICTORS = list(PREDICTOR_SUMMARY_KEY.keys())


def dropout_variance_checkpoint_schedule(total_epochs, num_points=DROPOUT_VARIANCE_NUM_CHECKPOINTS):
    """
    Log-uniform-spaced epoch indices for the dropout-variance checkpoints,
    same spirit as l2_norm.resample_to_log_uniform_grid's log-epoch grid —
    equal VISUAL spacing on a log-x plot, so early training (where things
    change fast) gets proportionally more checkpoints than the long flat
    tail. Returns a sorted list of 0-based epoch indices (matching the
    training loop's `epoch` variable, range(total_epochs)), always
    including epoch 0 and the final epoch (total_epochs - 1).
    """
    if total_epochs <= 1:
        return [0]
    log_grid = np.linspace(0, np.log10(total_epochs), num_points)
    raw = np.round(10 ** log_grid).astype(int) - 1  # 1..total_epochs -> 0..total_epochs-1
    raw = np.clip(raw, 0, total_epochs - 1)
    checkpoints = sorted(set(int(v) for v in raw))
    if checkpoints[0] != 0:
        checkpoints.insert(0, 0)
    if checkpoints[-1] != total_epochs - 1:
        checkpoints.append(total_epochs - 1)
    return checkpoints


def pick_drc_checkpoint_subset(checkpoints, num_drc=DROPOUT_VARIANCE_NUM_DRC_CHECKPOINTS):
    """Evenly-spaced-by-position subset of an existing checkpoint list, for
    the cheap 5-rate DRC-style snapshot (qualitative companion plot only —
    NOT the k=30 variance sweep, which runs at every checkpoint)."""
    if len(checkpoints) <= num_drc:
        return list(checkpoints)
    idx = np.linspace(0, len(checkpoints) - 1, num_drc).round().astype(int)
    return sorted(set(checkpoints[i] for i in idx))


def _num_or_none(value):
    return None if value is None else float(value)


def load_config(path):
    """Read configs/nanda_unified.yaml -> flat dict of the constants used here."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle)

    d_model = cfg["architecture"]["d_model"]
    init_std = cfg["init"]["init_std"]
    if init_std is None:
        init_std = 0.8 / (d_model ** 0.5)

    parsed = {
        "modulus": cfg["task"]["modulus"],
        "vocab_size": cfg["task"]["vocab_size"],
        "train_fraction": cfg["task"]["train_fraction"],
        "d_model": d_model,
        "num_heads": cfg["architecture"]["num_heads"],
        "init_std": float(init_std),
        "lr": float(cfg["optimizer"]["lr"]),
        "weight_decay": float(cfg["optimizer"]["weight_decay"]),
        "betas": tuple(cfg["optimizer"]["betas"]),
        "epochs_default": int(cfg["training"]["epochs"]),
    }
    # Fail loud if the config is internally inconsistent.
    assert parsed["vocab_size"] == parsed["modulus"] + 1, \
        "config: vocab_size must be modulus + 1"
    assert parsed["d_model"] % parsed["num_heads"] == 0, \
        "config: d_model must be divisible by num_heads"
    return parsed


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_dir_for(output_dir, seed):
    return os.path.join(output_dir, f"seed_{seed}")


def checkpoints_dir_for(output_dir, seed):
    return os.path.join(seed_dir_for(output_dir, seed), "checkpoints")


def summary_path_for(output_dir, seed):
    return os.path.join(seed_dir_for(output_dir, seed), "summary.json")


def load_summary(output_dir, seed):
    path = summary_path_for(output_dir, seed)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def get_done_predictors(output_dir, seed):
    """Which predictors already have a real result in this seed's
    summary.json, keyed by PREDICTOR_SUMMARY_KEY -> returns a set of
    predictor names (e.g. {"l2", "dropout_gap"}). Empty set if
    summary.json does not exist yet, or exists but is missing a key
    entirely (e.g. an old run from before dropout_variance existed)."""
    summary = load_summary(output_dir, seed)
    if summary is None:
        return set()
    done = set()
    for name, key in PREDICTOR_SUMMARY_KEY.items():
        if summary.get(key) is not None:
            done.add(name)
    return done


def is_predictor_done(output_dir, seed, predictor):
    return predictor in get_done_predictors(output_dir, seed)


def has_saved_checkpoints(output_dir, seed):
    ckpt_dir = checkpoints_dir_for(output_dir, seed)
    if not os.path.isdir(ckpt_dir):
        return False
    return any(name.endswith(".pt") for name in os.listdir(ckpt_dir))


def grok_epoch_from(test_acc_history):
    arr = np.asarray(test_acc_history, dtype=float)
    hits = np.where(arr > GROK_ACC_THRESHOLD)[0]
    return int(hits[0]) if len(hits) else None


def limit_cycle_check(test_acc_history, grok_epoch, settle_epochs=500):
    """Detect the post-grok test-accuracy oscillation seen in the p=113
    run_1 (test acc swinging ~0.5-1.0 for the rest of training). Looks at
    the test-acc tail starting settle_epochs after grok."""
    arr = np.asarray(test_acc_history, dtype=float)
    if grok_epoch is None:
        return {"applicable": False, "limit_cycle": False,
                "reason": "no grok"}
    start = grok_epoch + settle_epochs
    if start >= len(arr) - 10:
        return {"applicable": False, "limit_cycle": False,
                "reason": "not enough epochs after grok"}
    tail = arr[start:]
    post_min = float(tail.min())
    post_std = float(tail.std())
    n_dips = int(np.sum(tail < 0.9))
    is_limit_cycle = bool((post_min < 0.9) and (post_std > 0.05))
    return {
        "applicable": True,
        "window_start_epoch": int(start),
        "post_grok_min": post_min,
        "post_grok_std": post_std,
        "post_grok_final": float(tail[-1]),
        "epochs_below_0.9_post_grok": n_dips,
        "limit_cycle": is_limit_cycle,
    }


def train_one_seed(seed, args, cfg, device, predictors_to_compute, old_summary,
                    save_checkpoints=True):
    """Train seed from scratch. predictors_to_compute (a set, subset of
    ALL_PREDICTORS) controls which expensive per-predictor work actually
    runs this call; a predictor NOT in that set falls back to its block
    in old_summary (if any) in the final summary.json, so a predictor
    that was already done is never silently dropped or recomputed just
    because a different predictor triggered this retrain. l2_norm_history
    itself is always collected during training regardless (needed for the
    training-data plots either way, and is cheap), but the L2-predictor
    *signal* block still only overwrites old_summary's when "l2" is
    actually requested."""
    out_dir = seed_dir_for(args.output_dir, seed)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = checkpoints_dir_for(args.output_dir, seed)
    if save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)
    compute_dropout_gap = "dropout_gap" in predictors_to_compute
    compute_dv = "dropout_variance" in predictors_to_compute
    compute_l2 = "l2" in predictors_to_compute

    # (a) deterministic per-seed RNG
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = cfg["modulus"]
    batch_size = int(cfg["train_fraction"] * p * p)  # full-batch

    # (b) data — "=" token id must equal the config modulus p, not any
    # value carried over from the old p=<old prime> code path
    train_loader, test_loader = get_dataloaders(number=p, batch_size=batch_size)
    probe_x, _ = next(iter(train_loader))
    eq_token_ids = {int(v) for v in probe_x[:, 2].tolist()}
    assert eq_token_ids == {p}, \
        f"'=' token ids {sorted(eq_token_ids)} != {{{p}}} (config modulus)"

    # (c) model — small-init _init_weights (init_std = 0.8/sqrt(d_model))
    model = TransformerFourHead(
        vocab_size=p + 1,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        init_std=cfg["init_std"],
    ).to(device)

    # (d) optimiser — AdamW(lr, betas, weight_decay) from the config
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()

    measurements = PredictorMeasurements(out_dir, model_type="four_head")
    np.save(os.path.join(out_dir, "seed.npy"), np.array([seed]))

    train_acc_history, test_acc_history, loss_history = [], [], []
    l2_norm_history, sum_w2_history, per_module_sum_w2_history = [], [], []

    # Dropout-Variance predictor: checkpoint schedule fixed up front, so
    # every checkpoint's variance measurement uses this seed's actual
    # weights at that exact epoch (cannot be recovered after the fact from
    # saved histories, unlike L2 Norm's post-hoc-computable signals).
    dv_checkpoint_schedule = dropout_variance_checkpoint_schedule(args.epochs)
    dv_checkpoint_set = set(dv_checkpoint_schedule)
    dv_drc_checkpoints = pick_drc_checkpoint_subset(dv_checkpoint_schedule)
    dv_drc_checkpoint_set = set(dv_drc_checkpoints)
    dropout_variance_checkpoints, dropout_variance_history = [], []
    dropout_variance_mean_acc_history = []
    dropout_drc_snapshots = {}

    started = time.time()
    last_log = started
    for epoch in range(args.epochs):
        # ---- train pass (full-batch: one iteration) ----
        model.train()
        train_correct = train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)[:, 2, :]
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)

        # ---- test pass ----
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for x_t, y_t in test_loader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                pred_t = model.forward(x_t)[:, 2, :].argmax(dim=1)
                test_correct += (pred_t == y_t).sum().item()
                test_total += len(y_t)

        # (e) per-epoch logging quantities
        train_acc_history.append(train_correct / train_total)
        test_acc_history.append(test_correct / test_total)
        loss_history.append(loss.item())
        l2_norm_history.append(compute_l2_norm(model))
        sum_w2_history.append(compute_sum_of_squared_weights(model))
        per_module_sum_w2_history.append(
            compute_per_module_sum_of_squared_weights(model))

        # ---- checkpoint save + Dropout-Variance predictor (post-epoch, model frozen) ----
        if epoch in dv_checkpoint_set:
            # Model-weight checkpointing is independent of whether the
            # Dropout-Variance predictor is being computed THIS run — a
            # future predictor (e.g. Spectral) can reuse these .pt files
            # without needing to retrain, as long as save_checkpoints is on.
            if save_checkpoints:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

            if compute_dv:
                mean_acc, variance = compute_dropout_variance(
                    model, test_loader, n_samples=DROPOUT_VARIANCE_N_SAMPLES,
                    dropout_rate=DROPOUT_VARIANCE_RATE, device=device)
                dropout_variance_checkpoints.append(epoch)
                dropout_variance_history.append(variance)
                dropout_variance_mean_acc_history.append(mean_acc)

                if epoch in dv_drc_checkpoint_set:
                    drc_results = compute_dropout_gap_multi_rate(model, test_loader, DROPOUT_RATES)
                    dropout_drc_snapshots[epoch] = {
                        str(r): drc_results[r]["train_accuracy"] for r in DROPOUT_RATES}

        if epoch % LOG_EVERY == 0 or epoch == args.epochs - 1:
            now = time.time()
            elapsed = now - started          # total wall time this seed
            since_last = now - last_log       # wall time for the last LOG_EVERY block
            last_log = now
            print(f"[seed {seed}] epoch {epoch:>6}/{args.epochs}  "
                  f"train_acc={train_acc_history[-1]:.4f}  "
                  f"test_acc={test_acc_history[-1]:.4f}  "
                  f"sum_w2={sum_w2_history[-1]:.1f}  "
                  f"elapsed={elapsed:10.3f}s  "
                  f"d{LOG_EVERY}={since_last:8.3f}s", flush=True)

    # (f) save the per-epoch histories
    measurements.save_training_data(train_acc_history, test_acc_history, loss_history)

    # grok_epoch only needs test_acc_history — compute it once here so both
    # the Dropout-Variance signal below and the final summary can use it.
    grok_epoch = grok_epoch_from(test_acc_history)

    # (g) L2-Norm predictor signals (post-training)
    epoch_grid, fast_ma, slow_ma = compute_fast_slow_moving_averages(
        l2_norm_history, fast_window=L2_FAST_WINDOW, slow_window=L2_SLOW_WINDOW)
    ma_crossover_epoch = detect_ma_crossover(
        epoch_grid, fast_ma, slow_ma, skip_epochs=L2_SKIP_EPOCHS)
    fast_ma_of_slow_ma, ma_of_ma_diff = compute_ma_of_slow_ma(
        slow_ma, fast_window=L2_MA_OF_MA_FAST_WINDOW)
    noise_floor = compute_noise_floor(
        ma_of_ma_diff, epoch_grid, quiet_epoch_cutoff=L2_QUIET_EPOCH_CUTOFF)
    ma_of_ma_zero_crossing_epoch = detect_ma_of_ma_zero_crossing(
        epoch_grid, ma_of_ma_diff, skip_epochs=L2_SKIP_EPOCHS)

    measurements.save_l2_norm_data(
        l2_norm_history, epoch_grid, fast_ma, slow_ma,
        fast_ma_of_slow_ma, ma_of_ma_diff, ma_crossover_epoch,
        sum_w2_history=sum_w2_history,
        per_module_sum_w2_history=per_module_sum_w2_history,
    )

    l2_signals = {
        "ma_crossover_epoch": _num_or_none(ma_crossover_epoch),
        "ma_of_ma_zero_crossing_epoch": _num_or_none(ma_of_ma_zero_crossing_epoch),
        "noise_floor": float(noise_floor),
        "windows": {
            "fast": L2_FAST_WINDOW, "slow": L2_SLOW_WINDOW,
            "ma_of_ma_fast": L2_MA_OF_MA_FAST_WINDOW,
            "skip_epochs": L2_SKIP_EPOCHS,
            "quiet_epoch_cutoff": L2_QUIET_EPOCH_CUTOFF,
        },
    }
    with open(os.path.join(measurements.l2_norm_dir,
                           "l2_predictor_signals.json"), "w") as handle:
        json.dump(l2_signals, handle, indent=2)

    # (g) Dropout predictor — one multi-rate sweep on the final model
    old_dropout_gap_block = (old_summary or {}).get("dropout_final_gap_by_rate")
    if compute_dropout_gap:
        dropout_results = compute_dropout_gap_multi_rate(model, test_loader, DROPOUT_RATES)
        model.train()  # compute_dropout_gap_multi_rate leaves the model in eval()

        final_epoch = args.epochs
        measurements.save_dropout_data(
            dropout_gap_epochs=[final_epoch],
            dropout_gap_history_by_rate={
                r: [dropout_results[r]["dropout_gap"]] for r in DROPOUT_RATES},
            dropout_train_acc_by_rate={
                r: [dropout_results[r]["train_accuracy"]] for r in DROPOUT_RATES},
            dropout_eval_acc_by_rate={
                r: [dropout_results[r]["eval_accuracy"]] for r in DROPOUT_RATES},
            dropout_rates=DROPOUT_RATES,
        )
        dropout_json = {
            str(r): {k: float(v) for k, v in dropout_results[r].items()}
            for r in DROPOUT_RATES
        }
        with open(os.path.join(measurements.dropout_dir,
                               "dropout_gap_final.json"), "w") as handle:
            json.dump(dropout_json, handle, indent=2)
        dropout_gap_block = {
            str(r): float(dropout_results[r]["dropout_gap"]) for r in DROPOUT_RATES}
    else:
        # not requested this call -> carry the old result forward unchanged
        dropout_gap_block = old_dropout_gap_block

    # (h) Dropout-Variance predictor — per-checkpoint arrays + signal
    old_dv_block = (old_summary or {}).get("dropout_variance_predictor")
    if compute_dv:
        np.save(os.path.join(measurements.dropout_dir, "dropout_variance_checkpoints.npy"),
                np.array(dropout_variance_checkpoints, dtype=int))
        np.save(os.path.join(measurements.dropout_dir, "dropout_variance_history.npy"),
                np.array(dropout_variance_history, dtype=float))
        np.save(os.path.join(measurements.dropout_dir, "dropout_variance_mean_acc_history.npy"),
                np.array(dropout_variance_mean_acc_history, dtype=float))
        with open(os.path.join(measurements.dropout_dir,
                               "dropout_drc_snapshots.json"), "w") as handle:
            json.dump({str(e): v for e, v in dropout_drc_snapshots.items()}, handle, indent=2)

        peak_idx = int(np.argmax(dropout_variance_history))
        variance_peak_epoch = dropout_variance_checkpoints[peak_idx]
        dropout_variance_block = {
            "variance_peak_epoch": variance_peak_epoch,
            "variance_peak_value": float(dropout_variance_history[peak_idx]),
            "grok_epoch": grok_epoch,
            "peak_to_grok_ratio": (
                float(variance_peak_epoch) / grok_epoch if grok_epoch else None),
            "rate": DROPOUT_VARIANCE_RATE,
            "n_samples": DROPOUT_VARIANCE_N_SAMPLES,
            "num_checkpoints": len(dropout_variance_checkpoints),
        }
        with open(os.path.join(measurements.dropout_dir,
                               "dropout_variance_signal.json"), "w") as handle:
            json.dump(dropout_variance_block, handle, indent=2)
    else:
        dropout_variance_block = old_dv_block

    # L2-predictor block: reuse the old one unless "l2" was requested.
    old_l2_block = (old_summary or {}).get("l2_predictor")
    l2_block = l2_signals if compute_l2 else (old_l2_block or l2_signals)

    # Spectral is checkpoint-only (see CHECKPOINT_PREDICTOR_FUNCS below) —
    # train_one_seed never computes it live, it only carries forward
    # whatever recompute_from_checkpoints already wrote for this seed.
    spectral_block = (old_summary or {}).get("spectral_predictor")

    # per-seed summary + resume sentinel. Old blocks for predictors NOT
    # recomputed this call are carried forward, never dropped.
    summary = {
        "seed": seed,
        "epochs": args.epochs,
        "modulus": p,
        "grok_epoch": grok_epoch,
        "final_train_acc": float(train_acc_history[-1]),
        "final_test_acc": float(test_acc_history[-1]),
        "l2_norm_init": float(l2_norm_history[0]),
        "l2_norm_final": float(l2_norm_history[-1]),
        "sum_w2_init": float(sum_w2_history[0]),
        "sum_w2_final": float(sum_w2_history[-1]),
        "token_embedding_share_init": float(
            per_module_sum_w2_history[0]["token_embedding"] / sum_w2_history[0]),
        "l2_predictor": l2_block,
        "dropout_final_gap_by_rate": dropout_gap_block,
        "dropout_variance_predictor": dropout_variance_block,
        "spectral_predictor": spectral_block,
        "limit_cycle_check": limit_cycle_check(test_acc_history, grok_epoch),
        "wall_time_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[seed {seed}] done in {summary['wall_time_sec']}s  "
          f"grok_epoch={grok_epoch}  "
          f"final_test_acc={summary['final_test_acc']:.4f}", flush=True)
    return summary


def _checkpoint_predictor_dropout_variance(model, test_loader, ckpt_dir, ckpt_epochs,
                                            measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["dropout_variance"] — loads each saved
    checkpoint into `model` in turn and rebuilds the same variance-peak
    signal that train_one_seed computes live during training. Returns the
    dropout_variance_predictor block; also re-writes the same .npy/.json
    files train_one_seed writes, so downstream (plot_nanda_results.py)
    cannot tell a live-trained result from a checkpoint-recomputed one.

    This is the reference shape for any FUTURE checkpoint-only predictor
    (e.g. Spectral): take (model, test_loader, ckpt_dir, ckpt_epochs,
    measurements, grok_epoch, device), load each checkpoint's
    state_dict() into `model` yourself, compute your signal, save your own
    files under measurements.<predictor>_dir, and return one block dict —
    then register the function in CHECKPOINT_PREDICTOR_FUNCS below and add
    the predictor's name to PREDICTOR_SUMMARY_KEY / ALL_PREDICTORS /
    CHECKPOINT_ONLY_PREDICTORS. No other part of this file needs to
    change."""
    drc_checkpoints = set(pick_drc_checkpoint_subset(ckpt_epochs))
    dropout_variance_checkpoints, dropout_variance_history = [], []
    dropout_variance_mean_acc_history = []
    dropout_drc_snapshots = {}

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        mean_acc, variance = compute_dropout_variance(
            model, test_loader, n_samples=DROPOUT_VARIANCE_N_SAMPLES,
            dropout_rate=DROPOUT_VARIANCE_RATE, device=device)
        dropout_variance_checkpoints.append(epoch)
        dropout_variance_history.append(variance)
        dropout_variance_mean_acc_history.append(mean_acc)
        if epoch in drc_checkpoints:
            drc_results = compute_dropout_gap_multi_rate(model, test_loader, DROPOUT_RATES)
            dropout_drc_snapshots[epoch] = {
                str(r): drc_results[r]["train_accuracy"] for r in DROPOUT_RATES}

    np.save(os.path.join(measurements.dropout_dir, "dropout_variance_checkpoints.npy"),
            np.array(dropout_variance_checkpoints, dtype=int))
    np.save(os.path.join(measurements.dropout_dir, "dropout_variance_history.npy"),
            np.array(dropout_variance_history, dtype=float))
    np.save(os.path.join(measurements.dropout_dir, "dropout_variance_mean_acc_history.npy"),
            np.array(dropout_variance_mean_acc_history, dtype=float))
    with open(os.path.join(measurements.dropout_dir,
                           "dropout_drc_snapshots.json"), "w") as handle:
        json.dump({str(e): v for e, v in dropout_drc_snapshots.items()}, handle, indent=2)

    peak_idx = int(np.argmax(dropout_variance_history))
    variance_peak_epoch = dropout_variance_checkpoints[peak_idx]
    block = {
        "variance_peak_epoch": variance_peak_epoch,
        "variance_peak_value": float(dropout_variance_history[peak_idx]),
        "grok_epoch": grok_epoch,
        "peak_to_grok_ratio": (
            float(variance_peak_epoch) / grok_epoch if grok_epoch else None),
        "rate": DROPOUT_VARIANCE_RATE,
        "n_samples": DROPOUT_VARIANCE_N_SAMPLES,
        "num_checkpoints": len(dropout_variance_checkpoints),
    }
    with open(os.path.join(measurements.dropout_dir,
                           "dropout_variance_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


def _checkpoint_predictor_spectral(model, test_loader, ckpt_dir, ckpt_epochs,
                                    measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["spectral"] — same shape as
    _checkpoint_predictor_dropout_variance above: loads each saved
    checkpoint into `model` in turn, computes
    predictors.spectral.compute_spectral_for_model(model) at that
    checkpoint, collects the per-module singular-value history across all
    checkpoints, saves it via measurements.save_spectral_data, and
    returns the spectral_predictor block for summary.json.

    test_loader is unused (spectral metrics only need the frozen weights,
    not the data), but is kept in the signature to match the shared
    CHECKPOINT_PREDICTOR_FUNCS shape every registered function follows."""
    spectral_checkpoints = []
    spectral_history_by_module = {name: {"spectral_norm": [], "fro_norm": [],
                                          "stable_rank": [], "effective_rank": []}
                                   for name in compute_spectral_for_model(model)}

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        metrics_by_module = compute_spectral_for_model(model)
        spectral_checkpoints.append(epoch)
        for name, metrics in metrics_by_module.items():
            for metric_name in ("spectral_norm", "fro_norm", "stable_rank", "effective_rank"):
                spectral_history_by_module[name][metric_name].append(metrics[metric_name])

    measurements.save_spectral_data(spectral_checkpoints, spectral_history_by_module)

    # Aggregate signal: stable_rank should drop toward the low-rank Fourier
    # circuit (see src/predictors/spectral.py docstring). Track, per
    # module, the epoch where stable_rank first reaches its post-checkpoint
    # minimum, and compare it against grok_epoch.
    stable_rank_min_epoch_by_module = {}
    stable_rank_first_last_by_module = {}
    for name, metrics in spectral_history_by_module.items():
        sr = metrics["stable_rank"]
        min_idx = int(np.argmin(sr))
        stable_rank_min_epoch_by_module[name] = spectral_checkpoints[min_idx]
        stable_rank_first_last_by_module[name] = {
            "first": float(sr[0]), "last": float(sr[-1]),
        }

    block = {
        "spectral_checkpoints": spectral_checkpoints,
        "spectral_history_by_module": spectral_history_by_module,
        "module_names": list(spectral_history_by_module.keys()),
        "grok_epoch": grok_epoch,
        "stable_rank_min_epoch_by_module": stable_rank_min_epoch_by_module,
        "stable_rank_first_last_by_module": stable_rank_first_last_by_module,
        "num_checkpoints": len(spectral_checkpoints),
    }
    with open(os.path.join(measurements.spectral_dir,
                           "spectral_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


# Predictors this runner knows how to (re)compute from already-saved
# checkpoints/model_epoch_*.pt alone, with NO retraining. Adding a future
# predictor here (once it exists) is the ONLY change needed for it to gain
# the same "compute later, no retrain" resume behaviour dropout_variance
# already has — see _checkpoint_predictor_dropout_variance's docstring for
# the exact function shape expected.
CHECKPOINT_PREDICTOR_FUNCS = {
    "dropout_variance": _checkpoint_predictor_dropout_variance,
    "spectral": _checkpoint_predictor_spectral,
}
CHECKPOINT_ONLY_PREDICTORS = set(CHECKPOINT_PREDICTOR_FUNCS.keys())


def recompute_from_checkpoints(seed, args, cfg, device, predictors_to_compute, old_summary):
    """Compute predictor(s) that only need frozen model weights (currently
    just "dropout_variance", via CHECKPOINT_PREDICTOR_FUNCS) from this
    seed's already-saved checkpoints/model_epoch_*.pt, with NO retraining.
    Every other block in old_summary is carried forward unchanged. Only
    called when predictors_to_compute is a subset of
    CHECKPOINT_ONLY_PREDICTORS and has_saved_checkpoints() is true for
    this seed."""
    assert predictors_to_compute <= CHECKPOINT_ONLY_PREDICTORS, (
        f"recompute_from_checkpoints can only rebuild "
        f"{sorted(CHECKPOINT_ONLY_PREDICTORS)} from saved weights; the rest "
        f"need the full per-epoch training history")

    out_dir = seed_dir_for(args.output_dir, seed)
    ckpt_dir = checkpoints_dir_for(args.output_dir, seed)
    p = cfg["modulus"]
    batch_size = int(cfg["train_fraction"] * p * p)
    _, test_loader = get_dataloaders(number=p, batch_size=batch_size)

    model = TransformerFourHead(
        vocab_size=p + 1, d_model=cfg["d_model"], num_heads=cfg["num_heads"],
        init_std=cfg["init_std"],
    ).to(device)

    ckpt_epochs = sorted(
        int(name[len("model_epoch_"):-len(".pt")])
        for name in os.listdir(ckpt_dir) if name.startswith("model_epoch_") and name.endswith(".pt"))
    measurements = PredictorMeasurements(out_dir, model_type="four_head")
    grok_epoch = old_summary.get("grok_epoch")

    summary = dict(old_summary)  # carry every other block forward unchanged
    for predictor in sorted(predictors_to_compute):
        func = CHECKPOINT_PREDICTOR_FUNCS[predictor]
        block = func(model, test_loader, ckpt_dir, ckpt_epochs, measurements, grok_epoch, device)
        summary[PREDICTOR_SUMMARY_KEY[predictor]] = block

    with open(summary_path_for(args.output_dir, seed), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[seed {seed}] {', '.join(sorted(predictors_to_compute))} recomputed from "
          f"{len(ckpt_epochs)} saved checkpoints (no retrain)", flush=True)
    return summary


def aggregate(summaries, args):
    print("\n" + "=" * 72)
    print(f"NANDA-UNIFIED BENCHMARK - AGGREGATE  "
          f"({len(summaries)} seeds x {args.epochs} epochs)")
    print("=" * 72)

    groks = [s["grok_epoch"] for s in summaries if s["grok_epoch"] is not None]
    if groks:
        print(f"Grok epoch (test acc > {GROK_ACC_THRESHOLD}): "
              f"mean={np.mean(groks):.1f}  std={np.std(groks):.1f}  "
              f"min={min(groks)}  max={max(groks)}  "
              f"({len(groks)}/{len(summaries)} seeds grokked)")
    else:
        print("Grok epoch: no seed reached the grok threshold")

    print("\nL2-Norm predictor (per seed):")
    for s in summaries:
        lp = s.get("l2_predictor")
        if lp is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        print(f"  seed {s['seed']}: MA-crossover={lp['ma_crossover_epoch']}  "
              f"MA-of-MA zero-cross={lp['ma_of_ma_zero_crossing_epoch']}  "
              f"(grok={s['grok_epoch']})")

    print("\nDropout-Variance predictor (per seed):")
    for s in summaries:
        dv = s.get("dropout_variance_predictor")
        if dv is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        ratio = dv["peak_to_grok_ratio"]
        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        print(f"  seed {s['seed']}: variance_peak_epoch={dv['variance_peak_epoch']}  "
              f"peak/grok={ratio_str}  (grok={s['grok_epoch']})")

    print("\nSpectral predictor (per seed, stable_rank first->last by module):")
    for s in summaries:
        sp = s.get("spectral_predictor")
        if sp is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        fl = sp["stable_rank_first_last_by_module"]
        print(f"  seed {s['seed']}: " +
              "  ".join(f"{m}={fl[m]['first']:.1f}->{fl[m]['last']:.1f}" for m in fl) +
              f"  (grok={s['grok_epoch']})")

    print("\nDropout gap at final epoch (per seed, by rate):")
    for s in summaries:
        gaps = s.get("dropout_final_gap_by_rate")
        if gaps is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        print(f"  seed {s['seed']}: " +
              "  ".join(f"p{r}={gaps[r]:+.3f}" for r in gaps))

    print("\nLimit-cycle check (post-grok test-acc oscillation):")
    n_limit_cycle = 0
    for s in summaries:
        lc = s["limit_cycle_check"]
        if not lc.get("applicable"):
            print(f"  seed {s['seed']}: n/a ({lc.get('reason', 'n/a')})")
            continue
        n_limit_cycle += int(lc["limit_cycle"])
        label = "LIMIT CYCLE" if lc["limit_cycle"] else "stable"
        print(f"  seed {s['seed']}: {label}  "
              f"post-grok min={lc['post_grok_min']:.3f}  "
              f"std={lc['post_grok_std']:.3f}  "
              f"final={lc['post_grok_final']:.3f}  "
              f"dips<0.9={lc['epochs_below_0.9_post_grok']}")
    print(f"\n  => {n_limit_cycle}/{len(summaries)} seeds show a post-grok limit cycle")

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as handle:
        json.dump({
            "n_seeds": len(summaries),
            "epochs": args.epochs,
            "grok_epoch_mean": float(np.mean(groks)) if groks else None,
            "grok_epoch_std": float(np.std(groks)) if groks else None,
            "grok_epochs": groks,
            "n_limit_cycle": n_limit_cycle,
            "seeds": summaries,
        }, handle, indent=2)
    print(f"\nWrote {agg_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Nanda-Unified benchmark runner (L2-Norm + Dropout only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds; seed i uses torch.manual_seed(i)")
    parser.add_argument("--epochs", type=int, default=40000,
                        help="full-batch epochs per seed")
    parser.add_argument("--output_dir", type=str, default="results/nanda_unified")
    parser.add_argument("--config", type=str, default="configs/nanda_unified.yaml")
    parser.add_argument("--predictors", type=str, default="l2,dropout_gap,dropout_variance",
                        help="comma-separated subset of " + ",".join(ALL_PREDICTORS))
    parser.add_argument("--overwrite", type=str, default="",
                        help="comma-separated predictors to force-recompute even if "
                             "already present in summary.json")
    parser.add_argument("--no_save_checkpoints", action="store_true",
                        help="disable saving checkpoints/model_epoch_*.pt during training")
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(REPO_ROOT, args.output_dir)
    if not os.path.isabs(args.config):
        args.config = os.path.join(REPO_ROOT, args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    requested_predictors = {p.strip() for p in args.predictors.split(",") if p.strip()}
    unknown = requested_predictors - set(ALL_PREDICTORS)
    assert not unknown, f"--predictors: unknown predictor(s) {unknown}, valid: {ALL_PREDICTORS}"
    overwrite_predictors = {p.strip() for p in args.overwrite.split(",") if p.strip()}
    unknown_ow = overwrite_predictors - set(ALL_PREDICTORS)
    assert not unknown_ow, f"--overwrite: unknown predictor(s) {unknown_ow}, valid: {ALL_PREDICTORS}"

    cfg = load_config(args.config)
    device = pick_device()

    print("=" * 72)
    print("NANDA-UNIFIED BENCHMARK RUNNER   (predictors built so far: L2-Norm, Dropout)")
    print("=" * 72)
    print(f"config     : {args.config}")
    print(f"output_dir : {args.output_dir}")
    print(f"device     : {device}")
    print(f"task       : (a + b) mod {cfg['modulus']}   "
          f"'=' token id {cfg['modulus']}   vocab {cfg['vocab_size']}")
    print(f"model      : 4-head, d_model={cfg['d_model']}, "
          f"init_std={cfg['init_std']:.5f}  (small init)")
    print(f"optimiser  : AdamW lr={cfg['lr']} betas={cfg['betas']} "
          f"weight_decay={cfg['weight_decay']}")
    print(f"run        : seeds={args.seeds}  epochs={args.epochs}  "
          f"full-batch={int(cfg['train_fraction'] * cfg['modulus'] * cfg['modulus'])}")
    print("=" * 72)

    summaries = []
    for seed in range(args.seeds):
        old_summary = load_summary(args.output_dir, seed)
        done = get_done_predictors(args.output_dir, seed)
        effective_done = done - overwrite_predictors

        missing = set()
        for predictor in sorted(requested_predictors):
            if predictor in effective_done:
                print(f"[seed {seed}] {predictor} complete - skipping")
            else:
                missing.add(predictor)

        if not missing:
            print(f"[seed {seed}] all predictors complete - skipping")
            summaries.append(old_summary)
            continue

        can_skip_retrain = (
            missing <= CHECKPOINT_ONLY_PREDICTORS
            and old_summary is not None
            and has_saved_checkpoints(args.output_dir, seed)
        )
        if can_skip_retrain:
            summaries.append(
                recompute_from_checkpoints(seed, args, cfg, device, missing, old_summary))
        else:
            summary = train_one_seed(
                seed, args, cfg, device, predictors_to_compute=missing,
                old_summary=old_summary, save_checkpoints=not args.no_save_checkpoints)
            # Some CHECKPOINT_ONLY_PREDICTORS (e.g. spectral) are never
            # computed live inside train_one_seed's epoch loop — they only
            # need the checkpoints that fresh training just saved. Run
            # those now, straight off this seed's brand-new checkpoints,
            # so a first-time "--predictors spectral" run does not silently
            # finish with spectral_predictor still None.
            leftover = {p for p in (missing & CHECKPOINT_ONLY_PREDICTORS)
                        if summary.get(PREDICTOR_SUMMARY_KEY[p]) is None}
            if leftover and not args.no_save_checkpoints and has_saved_checkpoints(args.output_dir, seed):
                summary = recompute_from_checkpoints(seed, args, cfg, device, leftover, summary)
            summaries.append(summary)

    aggregate(summaries, args)


if __name__ == "__main__":
    main()
