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
    python 06_Code/scripts/run_nanda_benchmark.py --seeds 5 --epochs 40000 \
        --output_dir 08_Experiments/results/nanda_unified --config 06_Code/configs/nanda_unified.yaml

    # quick smoke:
    python 06_Code/scripts/run_nanda_benchmark.py --seeds 1 --epochs 100 \
        --output_dir 08_Experiments/results/test_smoke
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

# This file lives at <repo>/06_Code/scripts/, so the repo root is two folders
# up. (Before the 2026-09-17 reorganisation the script sat at the repo root,
# and REPO_ROOT was simply its own folder; that broke after the move.) The
# relative --output_dir / --config defaults below are resolved from here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "06_Code", "src")
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
from predictors.spectral import compute_spectral_metrics_for_checkpoint   # noqa: E402
from predictors.age import compute_age_metrics_for_checkpoint             # noqa: E402
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
#     measured on the EVAL_EVERY grid (default every 100 epochs, see the
#     checkpoint-schedule block below), since 40000 epochs x 30 passes at
#     every epoch is not affordable on MPS (about 1.8 s per measurement).
DROPOUT_VARIANCE_RATE = 0.5
DROPOUT_VARIANCE_N_SAMPLES = 30
DROPOUT_VARIANCE_NUM_CHECKPOINTS = 24   # legacy default of the old log-spaced generator only
DROPOUT_VARIANCE_NUM_DRC_CHECKPOINTS = 5

# ---------------------------------------------------------------------------
# Checkpoint schedule: SAVING is decoupled from EVALUATING.
# (Updates 2026-09-25; full derivations are in context.md, appended sections
#  "Switch from 24 log-spaced to save-every-100 (derived)" and
#  "Shift from 100 to 50 to support 250 grid (GCD justification)".)
#
# Two requirements fix the spacing (thresholds are the project's own design
# choice, NOT a literature standard):
#   a) transition coverage : samples inside grok transition = width / spacing
#        (narrowest measured 10%->90% test-acc width = 1570 epochs; want >= 6)
#   b) ratio precision     : worst ratio error = spacing / smallest grok epoch
#        (smallest measured grok epoch = 6988; want <= 4%)
#   spacing  50 -> 31.40 samples, 0.72% error     (SAVE grid)
#   spacing 100 -> 15.70 samples, 1.43% error     (default EVAL grid)
#   spacing 200 ->  7.85 samples, 2.86% error     (fallback EVAL grid)
#   spacing 250 ->  6.28 samples, 3.58% error     (coarsest fallback, just inside both thresholds)
#   spacing 500 ->  3.14 samples, 7.16% error     (breaks both thresholds; what a "250 grid"
#                                                  would silently become on a 100-grid save)
#
# WHY SAVE EVERY 50: the evaluation grids we want (100, 200, 250) have
# gcd(100, 200, 250) = 50. A grid can only be evaluated if every one of its
# epochs was saved, so SAVE_EVERY must divide all of them. 50 does; 100 does
# not divide 250 (saved multiples of 100 that are also multiples of 250 are
# only multiples of 500, i.e. aliasing to a 500 grid). Save densely once,
# evaluate on any coarser grid later without retraining.
#
# SAVE   : a checkpoint every SAVE_EVERY = 50 epochs, PLUS the final epoch:
#          range(0, 40000, 50) = 800 points (0..39950) + epoch 39999 = 801.
#          Storage: ~0.85 MB per checkpoint -> ~680 MB per seed, ~3.4 GB for 5 seeds.
# EVAL   : every predictor evaluates only the saved epochs with
#          epoch % EVAL_EVERY == 0, PLUS the final saved epoch (the fully
#          trained model; kept so results stay comparable with the earlier
#          runs, whose grids always ended on epoch 39999). Saving is nearly
#          free, evaluating is not: Spectral ~9.3 s and Dropout-Variance
#          ~1.8 s per checkpoint on this machine (~11 s together), so the
#          default 100 grid (401 points) costs about 1.2 h per seed for those
#          two, the 200 grid (201 points) ~0.6 h, the 250 grid (161 points)
#          ~0.5 h, and the full 50 grid (801 points) ~2.5 h.
# --eval_every accepts 50, 100, 200 or 250 (any multiple of SAVE_EVERY);
#          select_eval_epochs() raises if the requested grid is not on disk.
# EVAL_EVERY = 0 means "use every saved checkpoint as-is" (needed to
#          re-run predictors on the legacy 24-point directories).
TOTAL_STEPS = 40000          # default --epochs
SAVE_EVERY = 50
EVAL_EVERY = 100
EVAL_EVERY_FALLBACK = 200
EVAL_EVERY_FALLBACK_COARSE = 250

# Legacy schedule (used up to benchmark v4, tag benchmark-v4-4predictors):
# 24 log-spaced epochs, chosen as a compute budget for Dropout-Variance, with
# no scientific justification for the count (see context.md, 2026-09-25).
# Kept, NOT deleted, so the old grid can be reproduced for ablation
# comparison, e.g. via --eval_every 0 on a legacy results directory.
OLD_24_LOG_SPACED_CHECKPOINTS = [
    0, 1, 2, 3, 5, 9, 15, 24, 39, 62, 99, 158, 251, 398, 632, 1002, 1589,
    2520, 3995, 6333, 10040, 15917, 25232, 39999,
]

# Names understood by --predictors / --overwrite, and the summary.json key
# each one's presence is checked against (see get_done_predictors below).
PREDICTOR_SUMMARY_KEY = {
    "l2": "l2_predictor",
    "dropout_gap": "dropout_final_gap_by_rate",
    "dropout_variance": "dropout_variance_predictor",
    "spectral": "spectral_predictor",
    "age": "age_predictor",
}
ALL_PREDICTORS = list(PREDICTOR_SUMMARY_KEY.keys())


def save_checkpoint_schedule(total_epochs, save_every=SAVE_EVERY):
    """Epoch indices (0-based, matching the training loop's `epoch`) at which
    a model checkpoint is SAVED: 0, save_every, 2*save_every, ... below
    total_epochs, plus the final epoch total_epochs - 1. For 40000 epochs and
    save_every=50 that is 800 + 1 = 801 points."""
    if total_epochs <= 1:
        return [0]
    schedule = list(range(0, total_epochs, save_every))
    if schedule[-1] != total_epochs - 1:
        schedule.append(total_epochs - 1)
    return schedule


SAVE_CHECKPOINTS = save_checkpoint_schedule(TOTAL_STEPS)


def select_eval_epochs(saved_epochs, eval_every=EVAL_EVERY):
    """Subsample the evaluation grid from the SAVED checkpoint epochs.
    Keeps saved epochs with epoch % eval_every == 0, plus the final saved
    epoch. eval_every <= 0 returns every saved epoch unchanged (legacy
    directories). Raises if the saved files cannot supply the requested
    grid, instead of silently evaluating on a coarser one."""
    saved = sorted(saved_epochs)
    if not saved:
        return []
    if eval_every <= 0:
        return saved
    saved_set = set(saved)
    missing = [e for e in range(0, saved[-1] + 1, eval_every) if e not in saved_set]
    if missing:
        raise ValueError(
            f"eval_every={eval_every} needs saved checkpoints at epochs such as "
            f"{missing[:5]}, which are not on disk (saved grid spacing is not a "
            f"divisor of {eval_every}). Use a multiple of the save spacing "
            f"({SAVE_EVERY}; e.g. 100, {EVAL_EVERY_FALLBACK} or {EVAL_EVERY_FALLBACK_COARSE}), "
            f"or --eval_every 0 to use "
            f"all saved checkpoints as they are.")
    chosen = [e for e in saved if e % eval_every == 0]
    if chosen[-1] != saved[-1]:
        chosen.append(saved[-1])
    return chosen


def dropout_variance_checkpoint_schedule(total_epochs, num_points=DROPOUT_VARIANCE_NUM_CHECKPOINTS):
    """
    LEGACY (24 log-spaced points; no longer used by default — see
    OLD_24_LOG_SPACED_CHECKPOINTS and the checkpoint-schedule block above).
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

    # Checkpoint SAVE grid (every SAVE_EVERY epochs + final epoch) and the
    # EVAL grid (subsampled from it, every args.eval_every epochs + final
    # epoch) are fixed up front. Saving and evaluating are decoupled: the
    # Dropout-Variance measurement (which needs live weights) runs only on
    # the eval grid, but every save-grid checkpoint is written to disk so
    # any other predictor can later be evaluated on any coarser grid
    # without retraining.
    save_checkpoint_set = set(save_checkpoint_schedule(args.epochs))
    dv_checkpoint_schedule = select_eval_epochs(
        sorted(save_checkpoint_set), args.eval_every)
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
        # Model-weight checkpointing (SAVE grid) is independent of whether the
        # Dropout-Variance predictor is being computed THIS run — other
        # predictors (Spectral, AGE, HTSR, ...) reuse these .pt files
        # without retraining, as long as save_checkpoints is on.
        if save_checkpoints and epoch in save_checkpoint_set:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

        # Dropout-Variance evaluation (EVAL grid, a subset of the SAVE grid).
        if compute_dv and epoch in dv_checkpoint_set:
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

    # AGE is checkpoint-only too (CHECKPOINT_PREDICTOR_FUNCS["age"]) —
    # train_one_seed never computes it live, it only carries forward
    # whatever recompute_from_checkpoints already wrote for this seed.
    age_block = (old_summary or {}).get("age_predictor")

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
        "age_predictor": age_block,
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
    """CHECKPOINT_PREDICTOR_FUNCS["spectral"] — Canatar et al. 2021
    task-model alignment (see src/predictors/spectral.py). Loads each
    saved checkpoint into `model` in turn, computes
    compute_spectral_metrics_for_checkpoint(model, train_loader, device)
    — kernel eigenvalues eta_k, task power w_k^2, cumulative power C(k),
    k_90 / k_95, alignment_score, entropy — collects the per-checkpoint
    history, saves it via measurements.save_spectral_data, and returns the
    spectral_predictor block for summary.json.

    test_loader is IGNORED — Canatar's metrics are built on the TRAINING
    representations and training labels, not the test split. This function
    rebuilds the exact training split this seed used (same per-seed RNG
    seeding as train_one_seed step (a)) so the sampled kernel matches the
    one the model was actually trained on. The signature keeps
    `test_loader` only to match the shared CHECKPOINT_PREDICTOR_FUNCS
    shape every registered function follows."""
    # Recover this seed and rebuild its training split deterministically.
    seed = int(np.load(os.path.join(measurements.output_dir, "seed.npy"))[0])
    p = model.output_head.out_features - 1
    batch_size = int(0.3 * p * p)  # full-batch training split (matches get_dataloaders)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _ = get_dataloaders(number=p, batch_size=batch_size)

    spectral_checkpoints = []
    hist = {
        "k_90": [], "k_95": [], "alignment_score": [], "entropy": [],
        "eigenvalues_history": [], "cumulative_power_history": [],
    }

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        m = compute_spectral_metrics_for_checkpoint(model, train_loader, device)
        spectral_checkpoints.append(epoch)
        hist["k_90"].append(m["k_90"])
        hist["k_95"].append(m["k_95"])
        hist["alignment_score"].append(m["alignment_score"])
        hist["entropy"].append(m["entropy"])
        hist["eigenvalues_history"].append(m["eigenvalues"])
        hist["cumulative_power_history"].append(m["cumulative_power"])

    measurements.save_spectral_data(spectral_checkpoints, hist)

    # Aggregate signal (Canatar): task-model alignment should form at or
    # before grokking, i.e. k_90 DROPS and alignment_score RISES around
    # grok_epoch. Record the checkpoint epoch of minimum k_90 / maximum
    # alignment and compare against grok_epoch.
    k90 = hist["k_90"]
    align = hist["alignment_score"]
    k_90_min_idx = int(np.argmin(k90))
    k_95_min_idx = int(np.argmin(hist["k_95"]))
    alignment_max_idx = int(np.argmax(align))
    k_90_min_epoch = spectral_checkpoints[k_90_min_idx]
    alignment_max_epoch = spectral_checkpoints[alignment_max_idx]

    block = {
        "spectral_checkpoints": spectral_checkpoints,
        "k_90_history": [int(v) for v in k90],
        "k_95_history": [int(v) for v in hist["k_95"]],
        "alignment_history": [float(v) for v in align],
        "entropy_history": [float(v) for v in hist["entropy"]],
        "grok_epoch": grok_epoch,
        "k_90_min_epoch": k_90_min_epoch,
        "k_95_min_epoch": spectral_checkpoints[k_95_min_idx],
        "alignment_max_epoch": alignment_max_epoch,
        "k_90_first_last": {"first": int(k90[0]), "last": int(k90[-1])},
        "k_95_first_last": {"first": int(hist["k_95"][0]), "last": int(hist["k_95"][-1])},
        "alignment_first_last": {"first": float(align[0]), "last": float(align[-1])},
        "k_90_min_to_grok_ratio": (
            float(k_90_min_epoch) / grok_epoch if grok_epoch else None),
        "alignment_max_to_grok_ratio": (
            float(alignment_max_epoch) / grok_epoch if grok_epoch else None),
        "num_checkpoints": len(spectral_checkpoints),
    }
    with open(os.path.join(measurements.spectral_dir,
                           "spectral_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


def _checkpoint_predictor_age(model, test_loader, ckpt_dir, ckpt_epochs,
                              measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["age"] — AGE, Adaptive Grokking Epoch via
    Neural Collapse (Papyan et al. 2020 NC1; see src/predictors/age.py).
    Loads each saved checkpoint into `model` in turn, computes
    compute_age_metrics_for_checkpoint(model, train_loader, device) — the
    NC1 variability-collapse ratio Tr(Sigma_W)/Tr(Sigma_B) and the mean
    feature norm — collects the per-checkpoint history, saves it via
    measurements.save_age_data, and returns the age_predictor block for
    summary.json.

    test_loader is IGNORED — NC1 is a property of the TRAINING
    representations (within-/between-class scatter of the penultimate
    features on the train split). This function rebuilds the exact
    training split this seed used (same per-seed RNG seeding as
    train_one_seed step (a) and _checkpoint_predictor_spectral) so the
    class means match the data the model was trained on. The signature
    keeps `test_loader` only to match the shared
    CHECKPOINT_PREDICTOR_FUNCS shape.

    AGE prediction under test: NC1 falls sharply at or BEFORE grok, so the
    checkpoint epoch of minimum NC1 is a leading/coincident grokking
    marker. nc1_min_to_grok_ratio = nc1_min_epoch / grok_epoch is the
    head-to-head number the benchmark records."""
    # Recover this seed and rebuild its training split deterministically.
    seed = int(np.load(os.path.join(measurements.output_dir, "seed.npy"))[0])
    p = model.output_head.out_features - 1
    batch_size = int(0.3 * p * p)  # full-batch training split (matches get_dataloaders)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _ = get_dataloaders(number=p, batch_size=batch_size)

    age_checkpoints = []
    nc1_history, fn_history = [], []

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        m = compute_age_metrics_for_checkpoint(model, train_loader, device)
        age_checkpoints.append(epoch)
        nc1_history.append(m["nc1"])
        fn_history.append(m["fn"])

    measurements.save_age_data(age_checkpoints, {"nc1": nc1_history, "fn": fn_history})

    nc1_min_idx = int(np.argmin(nc1_history))
    nc1_min_epoch = age_checkpoints[nc1_min_idx]
    fn_arr = np.asarray(fn_history, dtype=float)

    block = {
        "age_checkpoints": age_checkpoints,
        "nc1_history": [float(v) for v in nc1_history],
        "fn_history": [float(v) for v in fn_history],
        "grok_epoch": grok_epoch,
        "nc1_min_epoch": nc1_min_epoch,
        "nc1_min_value": float(nc1_history[nc1_min_idx]),
        "nc1_first_last": {"first": float(nc1_history[0]), "last": float(nc1_history[-1])},
        "fn_first_last": {"first": float(fn_history[0]), "last": float(fn_history[-1])},
        "fn_min": float(fn_arr.min()),
        "fn_max": float(fn_arr.max()),
        "nc1_min_to_grok_ratio": (
            float(nc1_min_epoch) / grok_epoch if grok_epoch else None),
        "num_checkpoints": len(age_checkpoints),
    }
    with open(os.path.join(measurements.age_dir, "age_signal.json"), "w") as handle:
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
    "age": _checkpoint_predictor_age,
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

    saved_epochs = sorted(
        int(name[len("model_epoch_"):-len(".pt")])
        for name in os.listdir(ckpt_dir) if name.startswith("model_epoch_") and name.endswith(".pt"))
    # Evaluate on the EVAL grid, subsampled from what is saved on disk.
    ckpt_epochs = select_eval_epochs(saved_epochs, args.eval_every)
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
          f"{len(ckpt_epochs)} of {len(saved_epochs)} saved checkpoints "
          f"(eval_every={args.eval_every}, no retrain)", flush=True)
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

    print("\nSpectral predictor (Canatar task-model alignment, per seed):")
    for s in summaries:
        sp = s.get("spectral_predictor")
        if sp is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        k90 = sp["k_90_first_last"]
        al = sp["alignment_first_last"]
        print(f"  seed {s['seed']}: k_90 {k90['first']}->{k90['last']}  "
              f"k_90_min_epoch={sp['k_90_min_epoch']}  "
              f"alignment {al['first']:.4f}->{al['last']:.4f}  "
              f"alignment_max_epoch={sp['alignment_max_epoch']}  "
              f"(grok={s['grok_epoch']})")

    print("\nAGE predictor (NC1 variability collapse, per seed):")
    for s in summaries:
        ap = s.get("age_predictor")
        if ap is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        nc1 = ap["nc1_first_last"]
        ratio = ap["nc1_min_to_grok_ratio"]
        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        print(f"  seed {s['seed']}: NC1 {nc1['first']:.4f}->{nc1['last']:.4f}  "
              f"nc1_min_epoch={ap['nc1_min_epoch']}  "
              f"nc1_min/grok={ratio_str}  (grok={s['grok_epoch']})")

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
    parser.add_argument("--output_dir", type=str, default="08_Experiments/results/nanda_unified")
    parser.add_argument("--config", type=str, default="06_Code/configs/nanda_unified.yaml")
    parser.add_argument("--predictors", type=str, default="l2,dropout_gap,dropout_variance",
                        help="comma-separated subset of " + ",".join(ALL_PREDICTORS))
    parser.add_argument("--overwrite", type=str, default="",
                        help="comma-separated predictors to force-recompute even if "
                             "already present in summary.json")
    parser.add_argument("--no_save_checkpoints", action="store_true",
                        help="disable saving checkpoints/model_epoch_*.pt during training")
    parser.add_argument("--eval_every", type=int, default=EVAL_EVERY,
                        help=f"evaluate predictors on saved checkpoints with epoch %% eval_every == 0 "
                             f"(plus the final epoch). Must be a multiple of the save spacing "
                             f"({SAVE_EVERY}); fallbacks if too slow: {EVAL_EVERY_FALLBACK}, {EVAL_EVERY_FALLBACK_COARSE}. "
                             f"0 = use every saved checkpoint (legacy 24-point directories).")
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
