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
  - sets torch.manual_seed(seed) and passes the same seed to the data split
  - builds the dataloaders for p from the config (full-batch)
  - builds TransformerFourHead with Nanda's randn/sqrt init
  - trains full-batch with AdamW and a 10-step linear warmup
  - logs every epoch, before the optimizer step:
      train loss, test loss, train acc, test acc, and the total sum of
      squared weights (Nanda et al. 2023 Figure 7)
  - after training: records the confirmed-peak epoch of that sum, and the
    Dropout gap sweep, and
    saves them under output_dir/seed_{seed}/

After all seeds it aggregates: mean grok epoch, per-seed predictor
signals, and a post-grok limit-cycle check across the seeds.

Resume: a seed whose output_dir/seed_{seed}/summary.json exists is
skipped and its summary reloaded.

Usage:
    python 06_Code/scripts/run_nanda_benchmark.py --seeds 5 --epochs 40000 \
        --output_dir 08_Experiments/results/results_per_nanda_power --config 06_Code/configs/nanda_unified.yaml

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# This file lives at <repo>/06_Code/scripts/, so the repo root is two folders
# up. (Before the 2026-09-17 reorganisation the script sat at the repo root,
# and REPO_ROOT was simply its own folder; that broke after the move.) The
# relative --output_dir / --config defaults below are resolved from here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "06_Code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data.modular_arithmetic import get_dataloaders                       # noqa: E402
from models.transformer_four_head import (                                 # noqa: E402
    TransformerFourHead,
    cross_entropy_high_precision,
)
from predictors.l2_norm import (                                          # noqa: E402
    compute_sum_of_squared_weights,
    detect_confirmed_sum_w2_peak,
    sum_w2_peak,
)
from predictors.spectral import (                                          # noqa: E402
    compute_spectral_metrics_from_checkpoint,
    compute_spectral_metrics_for_checkpoint,
    detect_spectral_trigger,
    compute_reference_learning_curve,
    get_deterministic_sample_indices,
)
from predictors.age import compute_age_metrics_for_checkpoint             # noqa: E402
from predictors.htsr_alpha import (                                       # noqa: E402
    compute_htsr_metrics_for_checkpoint, LAYER_NAMES as HTSR_LAYER_NAMES)
from predictors.weight_pca import (                                       # noqa: E402
    compute_weight_pca_metrics_for_checkpoint,
    find_global_minimum_epoch, find_steepest_drop_epoch,
    find_threshold_crossing_epoch, LAYER_NAMES as WEIGHT_PCA_LAYER_NAMES)
from predictors.higher_mi import (                                        # noqa: E402
    compute_higher_mi_logit_triplet_for_checkpoint, o_information_three_way,
    compute_temporal_o_information_signal, find_global_maximum_epoch,
    find_zero_crossing_epoch, transfer_entropy_table)
from predictors.commutator_defect import (                                # noqa: E402
    compute_commutator_defect, find_commutator_defect_event_epoch)
from unified_measurements import PredictorMeasurements                    # noqa: E402

# Secondary unscored Commutator Defect sensitivity (eta=1e-4). Off by
# default; enable via --commutator_secondary_eta. Never feeds the PRIMARY
# verdict; writes only under commutator_defect_eta1e-4/.
COMMUTATOR_RUN_SECONDARY_ETA = False
COMMUTATOR_SECONDARY_ETA = 1e-4
COMMUTATOR_PRIMARY_ETA = 1e-3
COMMUTATOR_BATCH_SEED = 20261103
COMMUTATOR_K_PAIRS = 5
COMMUTATOR_BATCH_SIZE = 512

# ---------------------------------------------------------------------------
# Science constants live in configs/nanda_unified.yaml (the single source of
# truth; every value has its source written next to it there). They are copied
# into the module-level names below ONCE, by apply_config_constants(cfg) at the
# start of main(), so every helper (including the checkpoint-predictor plugins,
# whose signature is shared) can read them without a cfg argument.
GROK_THRESHOLDS = None            # e.g. [0.9, 0.95, 0.99]; grok epoch reported for each
GROK_ACC_THRESHOLD = None         # primary threshold = GROK_THRESHOLDS[0]
LOG_LINES_PER_RUN = None          # console lines per run; log_every = epochs // this
LIMIT_CYCLE_SETTLE_FRACTION = None
LIMIT_CYCLE_STD_ABOVE = None

# Dropout measurement has been removed from this Nanda-Unified runner.
# The Salah & Yevick signal (MLP p=53, every-epoch variance + DRC) lives in
# scripts/dropout/train_dropout_salah_yevick.py and predictors/dropout.py. This
# runner no longer computes dropout_gap or dropout_variance on the
# Transformer. Old summary.json keys are still carried forward if present.

# ---------------------------------------------------------------------------
# Checkpoint schedule: SAVING is decoupled from EVALUATING.
# (Updates 2026-09-25; the full derivations, the measured transition widths and
#  the design criteria used to compare spacings are in context.md, appended
#  sections "Switch from 24 log-spaced to save-every-100 (derived)" and
#  "Shift from 100 to 50 to support 250 grid (GCD justification)". They are
#  thesis analysis criteria, so they are deliberately NOT repeated as code
#  constants or checks here.)
#
# SAVE EVERY 100: Nanda's Config.save_every = 100. An init checkpoint is
# written separately, before any optimizer step (model_init.pt). Epoch
# files are written after that epoch's optimizer step, when
# epoch % 100 == 0 (0, 100, ..., 39900), plus model_epoch_<total>.pt
# after the final step (model_epoch_40000.pt when total is 40000).
# The loop index of the last step is total-1, so that final file is
# written after the loop. The old every-50 grid (801 files, including
# epoch 39999) is replaced by this schedule. A 250-epoch eval grid is
# no longer a subset of the saved epochs.
#
# EVAL: predictors use saved epochs with epoch % EVAL_EVERY == 0.
# --eval_every accepts 100 or 200 (multiples of SAVE_EVERY).
# EVAL_EVERY = 0 means "use every saved checkpoint as-is".
TOTAL_STEPS = 40000          # default --epochs
DESIRED_EVAL_GRIDS = [100, 200]   # first = default, the rest = fallbacks
SAVE_EVERY = 100             # Nanda save_every, not gcd(100, 200, 250)
EVAL_EVERY = DESIRED_EVAL_GRIDS[0]
EVAL_EVERY_FALLBACK = DESIRED_EVAL_GRIDS[1]
EVAL_EVERY_FALLBACK_COARSE = DESIRED_EVAL_GRIDS[1]

# Names understood by --predictors / --overwrite, and the summary.json key
# each one's presence is checked against (see get_done_predictors below).
# Dropout is not listed: it is measured on the isolated Salah & Yevick
# MLP pipeline, not on this Transformer runner.
PREDICTOR_SUMMARY_KEY = {
    "l2": "l2_predictor",
    "spectral": "spectral_predictor",
    "age": "age_predictor",
    "htsr": "htsr_predictor",
    "weight_pca": "weight_pca_predictor",
    "higher_mi": "higher_mi_predictor",
    "commutator_defect": "commutator_defect_predictor",
}
ALL_PREDICTORS = list(PREDICTOR_SUMMARY_KEY.keys())


def save_checkpoint_schedule(total_epochs, save_every=SAVE_EVERY):
    """Epoch labels at which a model checkpoint is SAVED.

    0, save_every, 2*save_every, ... strictly below total_epochs are the
    training-loop indices (the file is written after that epoch's optimizer
    step). total_epochs itself is included once: it is the weights after
    total_epochs completed steps, saved as model_epoch_<total_epochs>.pt.
    For 40000 epochs and save_every=100 that is 401 points
    (0, 100, ..., 39900, 40000). The init checkpoint is separate and is
    not in this list. total_epochs is not inserted twice when it is already
    the last grid point."""
    if total_epochs <= 0:
        return []
    schedule = list(range(0, total_epochs, save_every))
    if not schedule or schedule[-1] != total_epochs:
        schedule.append(total_epochs)
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
            f"({SAVE_EVERY}; e.g. 100 or {EVAL_EVERY_FALLBACK}), "
            f"or --eval_every 0 to use "
            f"all saved checkpoints as they are.")
    chosen = [e for e in saved if e % eval_every == 0]
    if chosen[-1] != saved[-1]:
        chosen.append(saved[-1])
    return chosen


def _num_or_none(value):
    return None if value is None else float(value)


def load_config(path):
    """Read configs/nanda_unified.yaml -> flat dict of the constants used here."""
    with open(path) as handle:
        cfg = yaml.safe_load(handle)

    d_model = cfg["architecture"]["d_model"]

    parsed = {
        "modulus": cfg["task"]["modulus"],
        "vocab_size": cfg["task"]["vocab_size"],
        "train_fraction": cfg["task"]["train_fraction"],
        "d_model": d_model,
        "num_heads": cfg["architecture"]["num_heads"],
        "lr": float(cfg["optimizer"]["lr"]),
        "weight_decay": float(cfg["optimizer"]["weight_decay"]),
        "betas": tuple(cfg["optimizer"]["betas"]),
        "warmup_steps": int(cfg["training"]["warmup_steps"]),
        "epochs_default": int(cfg["training"]["epochs"]),
        # evaluation / predictor constants (each has its source in the yaml)
        "grok_thresholds": [float(t) for t in cfg["evaluation"]["grok_thresholds"]],
        "log_lines_per_run": int(cfg["evaluation"]["log_lines_per_run"]),
        "limit_cycle_settle_fraction": float(cfg["evaluation"]["limit_cycle"]["settle_fraction"]),
        "limit_cycle_std_above": float(cfg["evaluation"]["limit_cycle"]["std_above"]),
    }
    # Fail loud if the config is internally inconsistent.
    assert parsed["vocab_size"] == parsed["modulus"] + 1, \
        "config: vocab_size must be modulus + 1"
    assert parsed["d_model"] % parsed["num_heads"] == 0, \
        "config: d_model must be divisible by num_heads"
    assert parsed["grok_thresholds"], "config: evaluation.grok_thresholds must not be empty"
    return parsed


def apply_config_constants(cfg):
    """Copy the yaml-derived constants into the module-level names declared
    near the top of this file (see the comment there for why). Called once,
    at the start of main(), before any training or recompute."""
    global GROK_THRESHOLDS, GROK_ACC_THRESHOLD, LOG_LINES_PER_RUN
    global LIMIT_CYCLE_SETTLE_FRACTION, LIMIT_CYCLE_STD_ABOVE
    GROK_THRESHOLDS = list(cfg["grok_thresholds"])
    GROK_ACC_THRESHOLD = GROK_THRESHOLDS[0]      # primary
    LOG_LINES_PER_RUN = cfg["log_lines_per_run"]
    LIMIT_CYCLE_SETTLE_FRACTION = cfg["limit_cycle_settle_fraction"]
    LIMIT_CYCLE_STD_ABOVE = cfg["limit_cycle_std_above"]


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
    predictor names (e.g. {"l2", "spectral"}). Empty set if
    summary.json does not exist yet, or exists but is missing a key
    entirely."""
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


def grok_epoch_from(test_acc_history, threshold=None):
    """First epoch at which test accuracy exceeds `threshold` (default: the
    primary threshold, GROK_THRESHOLDS[0]); None if it never does."""
    if threshold is None:
        threshold = GROK_ACC_THRESHOLD
    arr = np.asarray(test_acc_history, dtype=float)
    hits = np.where(arr > threshold)[0]
    return int(hits[0]) if len(hits) else None


def grok_epochs_by_threshold(test_acc_history):
    """{"0.99": epoch, "0.9": epoch, "0.95": epoch} for every threshold in
    GROK_THRESHOLDS (None where a threshold is never reached)."""
    return {str(t): grok_epoch_from(test_acc_history, t) for t in GROK_THRESHOLDS}


def limit_cycle_check(test_acc_history, grok_epoch):
    """Detect the post-grok test-accuracy oscillation seen in the p=113
    run_1 (test acc swinging ~0.5-1.0 for the rest of training). Looks at
    the test-acc tail starting settle_epochs after grok, where settle_epochs
    is LIMIT_CYCLE_SETTLE_FRACTION of the run length (5% = 2000 of 40000
    epochs), so it scales with the run instead of being a fixed number. A
    "dip" is a fall below the primary grok threshold."""
    arr = np.asarray(test_acc_history, dtype=float)
    if grok_epoch is None:
        return {"applicable": False, "limit_cycle": False,
                "reason": "no grok"}
    settle_epochs = int(LIMIT_CYCLE_SETTLE_FRACTION * len(arr))
    start = grok_epoch + settle_epochs
    if start >= len(arr) - 10:
        return {"applicable": False, "limit_cycle": False,
                "reason": "not enough epochs after grok"}
    tail = arr[start:]
    post_min = float(tail.min())
    post_std = float(tail.std())
    n_dips = int(np.sum(tail < GROK_ACC_THRESHOLD))
    is_limit_cycle = bool((post_min < GROK_ACC_THRESHOLD) and (post_std > LIMIT_CYCLE_STD_ABOVE))
    return {
        "applicable": True,
        "window_start_epoch": int(start),
        "post_grok_min": post_min,
        "post_grok_std": post_std,
        "post_grok_final": float(tail[-1]),
        dips_key(): n_dips,
        "limit_cycle": is_limit_cycle,
    }


def dips_key():
    """Key under which limit_cycle_check stores its dip count. With the
    primary grok threshold 0.99 this is 'epochs_below_0.99_post_grok'."""
    return f"epochs_below_{GROK_ACC_THRESHOLD}_post_grok"


def train_one_seed(seed, args, cfg, device, predictors_to_compute, old_summary,
                    save_checkpoints=True):
    """Train seed from scratch. predictors_to_compute (a set, subset of
    ALL_PREDICTORS) controls which expensive per-predictor work actually
    runs this call; a predictor NOT in that set falls back to its block
    in old_summary (if any) in the final summary.json, so a predictor
    that was already done is never silently dropped or recomputed just
    because a different predictor triggered this retrain. sum_w2_history
    (total sum of squared weights) is always collected during training
    (it is cheap), but the L2-predictor *signal* block still only
    overwrites old_summary's when "l2" is actually requested."""
    out_dir = seed_dir_for(args.output_dir, seed)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = checkpoints_dir_for(args.output_dir, seed)
    if save_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)
    compute_l2 = "l2" in predictors_to_compute

    # (a) One seed sets both RNGs. random.seed is applied inside
    # get_dataloaders. torch.manual_seed is applied immediately before the
    # model is built, so the initial weights depend on the seed as well.
    # Nanda's released code seeds only the split. We keep the seed's
    # influence on the weights because we want the experiment to be repeatable.
    np.random.seed(seed)

    p = cfg["modulus"]
    batch_size = int(cfg["train_fraction"] * p * p)  # full-batch

    # (b) data — Nanda gen_train_test, train_fraction read from the yaml
    train_loader, test_loader = get_dataloaders(number=p, batch_size=batch_size, seed=seed)
    probe_x, _ = next(iter(train_loader))
    eq_token_ids = {int(v) for v in probe_x[:, 2].tolist()}
    assert eq_token_ids == {p}, \
        f"'=' token ids {sorted(eq_token_ids)} != {{{p}}} (config modulus)"

    # (c) model — Nanda randn/sqrt init. Seed the torch RNG here so nothing
    # earlier in this function consumes it.
    torch.manual_seed(seed)
    model = TransformerFourHead(
        vocab_size=p + 1,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
    ).to(device)
    if save_checkpoints:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_init.pt"))

    # (d) optimiser — AdamW on all parameters, no grad clipping.
    # LambdaLR warmup matches Nanda: min(step/10, 1). PyTorch calls
    # scheduler.step() once inside LambdaLR.__init__, so the first
    # optimizer step sees lambda(0) = 0, and the factor reaches 1 at step 10.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )
    warmup_steps = cfg["warmup_steps"]
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: min(step / float(warmup_steps), 1.0)
    )

    measurements = PredictorMeasurements(out_dir, model_type="four_head")
    np.save(os.path.join(out_dir, "seed.npy"), np.array([seed]))

    train_acc_history, test_acc_history, loss_history = [], [], []
    test_loss_history = []
    sum_w2_history = []

    # Checkpoint SAVE grid (every SAVE_EVERY epochs + final epoch).
    # Predictors that need frozen weights re-read these files later.
    save_checkpoint_set = set(save_checkpoint_schedule(args.epochs))
    # Console cadence derived from the run length: LOG_LINES_PER_RUN lines
    # per run (40 lines = every 1000 epochs for the default 40000).
    log_every = max(1, args.epochs // LOG_LINES_PER_RUN)

    started = time.time()
    last_log = started
    for epoch in range(args.epochs):
        # Losses and accuracies are computed BEFORE the optimizer step,
        # as in Nanda's do_a_training_step. One full batch each.
        model.eval()
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        train_logits = model.forward(x)[:, 2, :]
        train_loss = cross_entropy_high_precision(train_logits, y)
        train_correct = (train_logits.argmax(dim=1) == y).sum().item()
        train_total = len(y)

        with torch.no_grad():
            x_t, y_t = next(iter(test_loader))
            x_t, y_t = x_t.to(device), y_t.to(device)
            test_logits = model.forward(x_t)[:, 2, :]
            test_loss = cross_entropy_high_precision(test_logits, y_t)
            test_correct = (test_logits.argmax(dim=1) == y_t).sum().item()
            test_total = len(y_t)

        train_acc_history.append(train_correct / train_total)
        test_acc_history.append(test_correct / test_total)
        loss_history.append(train_loss.item())
        test_loss_history.append(test_loss.item())
        sum_w2_history.append(compute_sum_of_squared_weights(model))

        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Model-weight checkpointing (SAVE grid). Predictors such as
        # Spectral, AGE, HTSR reuse these .pt files without retraining.
        if save_checkpoints and epoch in save_checkpoint_set:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

        if epoch % log_every == 0 or epoch == args.epochs - 1:
            now = time.time()
            elapsed = now - started          # total wall time this seed
            since_last = now - last_log       # wall time for the last log_every block
            last_log = now
            print(f"[seed {seed}] epoch {epoch:>6}/{args.epochs}  "
                  f"train_acc={train_acc_history[-1]:.4f}  "
                  f"test_acc={test_acc_history[-1]:.4f}  "
                  f"sum_w2={sum_w2_history[-1]:.1f}  "
                  f"elapsed={elapsed:10.3f}s  "
                  f"d{log_every}={since_last:8.3f}s", flush=True)

    # The loop index stops at epochs-1. model_epoch_<epochs>.pt is the
    # weights after that many completed steps. It is not also written
    # inside the loop, so the file is saved once.
    if save_checkpoints and args.epochs in save_checkpoint_set:
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"model_epoch_{args.epochs}.pt"))

    # (f) save the per-epoch histories
    measurements.save_training_data(
        train_acc_history, test_acc_history, loss_history, test_loss=test_loss_history)

    # grok epochs only need test_acc_history.
    # grok_epoch is the primary threshold (GROK_THRESHOLDS[0]); the epochs for
    # every threshold are also stored, so no conclusion rests on one cut.
    grok_epochs_all = grok_epochs_by_threshold(test_acc_history)
    grok_epoch = grok_epoch_from(test_acc_history)

    # L2 predictor: confirmed peak of the total sum of squared weights.
    measurements.save_l2_norm_data(sum_w2_history)
    peak_epoch, peak_value = sum_w2_peak(sum_w2_history)
    confirmed_epoch = detect_confirmed_sum_w2_peak(sum_w2_history)
    l2_signals = {
        "signal": "sum of squared weights",
        "rule": "confirmed peak (own choice)",
        "k": 100,
        "peak_epoch": peak_epoch,
        "peak_S": peak_value,
        "confirmed_peak_epoch": confirmed_epoch,
    }
    with open(os.path.join(measurements.l2_norm_dir,
                           "l2_predictor_signals.json"), "w") as handle:
        json.dump(l2_signals, handle, indent=2)

    # Dropout is no longer measured on this Transformer path. Carry any
    # previously saved blocks forward so old result folders stay intact.
    dropout_gap_block = (old_summary or {}).get("dropout_final_gap_by_rate")
    dropout_variance_block = (old_summary or {}).get("dropout_variance_predictor")

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

    # HTSR Alpha is checkpoint-only as well (CHECKPOINT_PREDICTOR_FUNCS["htsr"]).
    htsr_block = (old_summary or {}).get("htsr_predictor")

    # Higher-MI is checkpoint-only too (CHECKPOINT_PREDICTOR_FUNCS["higher_mi"]) --
    # train_one_seed never computes it live, it only carries forward whatever
    # recompute_from_checkpoints already wrote for this seed.
    higher_mi_block = (old_summary or {}).get("higher_mi_predictor")

    # Commutator Defect is checkpoint-only
    # (CHECKPOINT_PREDICTOR_FUNCS["commutator_defect"]) — train_one_seed
    # never computes it live, it only carries forward whatever
    # recompute_from_checkpoints already wrote for this seed.
    commutator_defect_block = (old_summary or {}).get("commutator_defect_predictor")

    # per-seed summary + resume sentinel. Old blocks for predictors NOT
    # recomputed this call are carried forward, never dropped.
    summary = {
        "seed": seed,
        "epochs": args.epochs,
        "modulus": p,
        "grok_epoch": grok_epoch,
        "grok_epochs_by_threshold": grok_epochs_all,
        "final_train_acc": float(train_acc_history[-1]),
        "final_test_acc": float(test_acc_history[-1]),
        "sum_w2_init": float(sum_w2_history[0]),
        "sum_w2_final": float(sum_w2_history[-1]),
        "l2_predictor": l2_block,
        "dropout_final_gap_by_rate": dropout_gap_block,
        "dropout_variance_predictor": dropout_variance_block,
        "spectral_predictor": spectral_block,
        "age_predictor": age_block,
        "htsr_predictor": htsr_block,
        "higher_mi_predictor": higher_mi_block,
        "commutator_defect_predictor": commutator_defect_block,
        "limit_cycle_check": limit_cycle_check(test_acc_history, grok_epoch),
        "wall_time_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[seed {seed}] done in {summary['wall_time_sec']}s  "
          f"grok_epoch={grok_epoch}  "
          f"final_test_acc={summary['final_test_acc']:.4f}", flush=True)
    return summary


def _checkpoint_predictor_spectral(model, test_loader, ckpt_dir, ckpt_epochs,
                                    measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["spectral"] — Canatar et al. 2021
    task-model alignment (see src/predictors/spectral.py). Loads each
    saved checkpoint into `model` in turn, computes
    compute_spectral_metrics_from_checkpoint(model, train_loader, device)
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
    train_loader, _ = get_dataloaders(number=p, batch_size=batch_size, seed=seed)

    # 4.7 Per-seed artifacts directory under <output_dir>/spectral/seed_<seed>/
    # (Sole home for Spectral artifacts; no duplicate under seed_<seed>/spectral)
    spectral_artifact_dir = os.path.join(
        os.path.dirname(measurements.output_dir), "spectral", f"seed_{seed}"
    )
    os.makedirs(spectral_artifact_dir, exist_ok=True)

    # Persist or reload fixed deterministic sample indices
    sample_indices_path = os.path.join(spectral_artifact_dir, "spectral_sample_indices.npy")
    if os.path.exists(sample_indices_path):
        sample_indices = np.load(sample_indices_path)
    else:
        sample_indices = get_deterministic_sample_indices(batch_size, seed=12345, n_spec=2048)
        np.save(sample_indices_path, sample_indices)

    spectral_checkpoints = []
    hist = {
        "k_90": [], "k_95": [], "alignment_score": [], "entropy": [],
        "eigenvalues_history": [], "cumulative_power_history": [],
    }
    last_metrics = None

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        if "attn_out.weight" not in state and hasattr(model, "attn_out"):
            model.attn_out.weight.data.copy_(torch.eye(model.d_model, device=device))
            if hasattr(model, "mask"):
                model.mask.data.copy_(torch.ones_like(model.mask))
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state)
        m = compute_spectral_metrics_from_checkpoint(
            model, train_loader, device, sample_indices=sample_indices
        )
        last_metrics = m
        spectral_checkpoints.append(epoch)
        hist["k_90"].append(m["k_90"])
        hist["k_95"].append(m["k_95"])
        hist["alignment_score"].append(m["alignment_score"])
        hist["entropy"].append(m["spectral_entropy"])
        hist["eigenvalues_history"].append(m["eigenvalues"])
        hist["cumulative_power_history"].append(m["cumulative_task_power"])

    k90 = hist["k_90"]
    align = hist["alignment_score"]
    P_i = detect_spectral_trigger(spectral_checkpoints, k90, align)

    n_base = min(5, len(k90))
    b_k = float(np.median(k90[:n_base])) if n_base > 0 else float("nan")
    eligible_epochs = [e for e in spectral_checkpoints if e >= 200]
    n_eligible = len(eligible_epochs)

    first_window_start_epoch = P_i
    if P_i is None:
        if b_k == 0.0 or np.isnan(b_k):
            reason_if_none = "baseline b_k is zero or NaN"
        elif n_eligible < 3:
            reason_if_none = "fewer than 3 eligible checkpoints with epoch >= 200"
        else:
            reason_if_none = "no 3-consecutive persistence window satisfying k_s <= 0.9*b_k and A_s >= 0.50"
    else:
        reason_if_none = None

    trigger_data = {
        "seed": seed,
        "baseline_b_k": b_k,
        "P_i": P_i,
        "fired": bool(P_i is not None),
        "first_window_start_epoch": first_window_start_epoch,
        "n_eligible_checkpoints": n_eligible,
        "reason_if_none": reason_if_none,
    }

    if last_metrics is not None and len(last_metrics["eigenvalues"]) > 0:
        ref_curve = compute_reference_learning_curve(
            last_metrics["eigenvalues"], last_metrics["task_power"]
        )
    else:
        ref_curve = {}

    # Persist the exact 10 artifacts required by Section 4.7
    np.save(os.path.join(spectral_artifact_dir, "spectral_checkpoints.npy"), np.array(spectral_checkpoints, dtype=int))
    np.save(os.path.join(spectral_artifact_dir, "spectral_k90.npy"), np.array(hist["k_90"], dtype=float))
    np.save(os.path.join(spectral_artifact_dir, "spectral_k95.npy"), np.array(hist["k_95"], dtype=float))
    np.save(os.path.join(spectral_artifact_dir, "spectral_alignment.npy"), np.array(hist["alignment_score"], dtype=float))
    np.save(os.path.join(spectral_artifact_dir, "spectral_entropy.npy"), np.array(hist["entropy"], dtype=float))

    max_r = max(len(ev) for ev in hist["eigenvalues_history"]) if hist["eigenvalues_history"] else 0
    eig_mat = np.zeros((len(spectral_checkpoints), max_r), dtype=np.float64)
    cum_mat = np.zeros((len(spectral_checkpoints), max_r), dtype=np.float64)
    for i in range(len(spectral_checkpoints)):
        r_i = len(hist["eigenvalues_history"][i])
        eig_mat[i, :r_i] = hist["eigenvalues_history"][i]
        cum_mat[i, :r_i] = hist["cumulative_power_history"][i]

    np.save(os.path.join(spectral_artifact_dir, "spectral_eigenvalues.npy"), eig_mat)
    np.save(os.path.join(spectral_artifact_dir, "spectral_cumulative_power.npy"), cum_mat)
    with open(os.path.join(spectral_artifact_dir, "spectral_reference_learning_curve.json"), "w") as f:
        json.dump(ref_curve, f, indent=2)
    with open(os.path.join(spectral_artifact_dir, "spectral_trigger.json"), "w") as f:
        json.dump(trigger_data, f, indent=2)

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
        "firing_epoch": P_i,
        "P_i": P_i,
        "baseline_b_k": b_k,
        "fired": bool(P_i is not None),
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
    with open(os.path.join(spectral_artifact_dir,
                           "spectral_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


def _checkpoint_predictor_age(model, test_loader, ckpt_dir, ckpt_epochs,
                              measurements, grok_epoch, device):
    """Checkpoint-only AGE computation."""
    import os
    seed = int(np.load(os.path.join(measurements.output_dir, "seed.npy"))[0])
    p = model.output_head.out_features - 1

    # Rebuild train_loader with correct seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    from data.modular_arithmetic import get_dataloaders
    train_loader, _ = get_dataloaders(number=p, batch_size=int(0.3 * p * p), seed=seed)

    age_checkpoints = []
    age_history = {"nc1": [], "fn": []}

    for epoch in ckpt_epochs:
        ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        metrics = compute_age_metrics_for_checkpoint(model, train_loader, device)
        age_checkpoints.append(epoch)
        age_history["nc1"].append(metrics["nc1"])
        age_history["fn"].append(metrics["fn"])

    # Save data
    measurements.save_age_data(age_checkpoints, age_history)

    # Find trigger epochs
    nc1_arr = np.array(age_history["nc1"])
    nc1_min_epoch = int(age_checkpoints[np.argmin(nc1_arr)])

    # 10% drop rule
    nc1_start = nc1_arr[0]
    nc1_min = nc1_arr.min()
    total_drop = nc1_start - nc1_min
    threshold_10 = nc1_start - 0.1 * total_drop
    hits = np.where(nc1_arr <= threshold_10)[0]
    threshold_10_epoch = int(age_checkpoints[hits[0]]) if len(hits) > 0 else None

    block = {
        "age_checkpoints": age_checkpoints,
        "nc1_history": age_history["nc1"],
        "fn_history": age_history["fn"],
        "nc1_min_epoch": nc1_min_epoch,
        "nc1_min_value": float(nc1_min),
        "nc1_first_last": {"first": float(age_history["nc1"][0]), "last": float(age_history["nc1"][-1])},
        "fn_first_last": {"first": float(age_history["fn"][0]), "last": float(age_history["fn"][-1])},
        "nc1_min_to_grok_ratio": (
            float(nc1_min_epoch) / grok_epoch if grok_epoch else None),
        "threshold_10_epoch": threshold_10_epoch,
        "grok_epoch": grok_epoch,
        "num_checkpoints": len(age_checkpoints)
    }

    with open(os.path.join(measurements.age_dir, "age_signal.json"), "w") as f:
        import json
        json.dump(block, f, indent=2)

    return block


def _checkpoint_predictor_htsr(model, test_loader, ckpt_dir, ckpt_epochs,
                               measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["htsr"] — HTSR Alpha (Martin & Mahoney;
    see src/predictors/htsr_alpha.py). Loads each saved checkpoint into
    `model`, computes compute_htsr_metrics_for_checkpoint(model) — the
    power-law tail exponent alpha of every Linear weight matrix's spectrum
    (ML fit, automatic xmin) and its mean over layers — collects the
    per-checkpoint history, saves it via measurements.save_htsr_data, and
    returns the htsr_predictor block for summary.json.

    test_loader is IGNORED: alpha depends on the weights only. Signature
    kept to match the shared CHECKPOINT_PREDICTOR_FUNCS shape.

    alpha_min_epoch (epoch of the lowest mean alpha) is recorded in the
    same "extreme value" form as the other checkpoint predictors, so the
    head-to-head number alpha_min_to_grok_ratio is comparable. No new
    detection rule or threshold is introduced here."""
    htsr_checkpoints = []
    history = {k: [] for k in ["alpha", "layer_alpha", "layer_xmin",
                               "layer_D", "layer_lmax", "layer_ntail"]}

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        m = compute_htsr_metrics_for_checkpoint(model)
        htsr_checkpoints.append(epoch)
        for k in history:
            history[k].append(m[k])

    measurements.save_htsr_data(htsr_checkpoints, history)

    alpha_arr = np.asarray(history["alpha"], dtype=float)
    alpha_min_idx = int(np.nanargmin(alpha_arr))
    alpha_min_epoch = htsr_checkpoints[alpha_min_idx]

    block = {
        "htsr_checkpoints": htsr_checkpoints,
        "layer_names": list(HTSR_LAYER_NAMES),
        "alpha_history": [float(v) for v in alpha_arr],
        "grok_epoch": grok_epoch,
        "alpha_min_epoch": alpha_min_epoch,
        "alpha_min_value": float(alpha_arr[alpha_min_idx]),
        "alpha_first_last": {"first": float(alpha_arr[0]), "last": float(alpha_arr[-1])},
        "layer_alpha_first": [float(v) for v in history["layer_alpha"][0]],
        "layer_alpha_last": [float(v) for v in history["layer_alpha"][-1]],
        "alpha_min_to_grok_ratio": (
            float(alpha_min_epoch) / grok_epoch if grok_epoch else None),
        "num_checkpoints": len(htsr_checkpoints),
    }
    with open(os.path.join(measurements.htsr_dir, "htsr_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


def _checkpoint_predictor_weight_pca(model, test_loader, ckpt_dir, ckpt_epochs,
                                     measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["weight_pca"] — Weight-PCA / Spectral
    Dynamics (Yunis et al. 2024, arXiv:2408.11804; see
    src/predictors/weight_pca.py). Loads each saved checkpoint into `model`,
    computes compute_weight_pca_metrics_for_checkpoint(model) — the
    normalised effective rank (Eqn 1/2 of the source paper) of every Linear
    weight matrix's singular-value spectrum, and its mean over layers —
    collects the per-checkpoint history, saves it via
    measurements.save_weight_pca_data, and returns the weight_pca_predictor
    block for summary.json.

    test_loader is IGNORED: the signal depends on the weights only, same as
    HTSR Alpha. Signature kept to match the shared CHECKPOINT_PREDICTOR_FUNCS
    shape.

    Primary event rule fixed to global minimum for consistency with
    AGE/HTSR, frozen protocol 2026-09-17. norm_eff_rank_min_epoch is the
    ONLY field fed into the official PASS/FAIL table (via evaluate(Gs, Ps)
    in analyze_nanda_unified.py). steepest_drop_epoch and
    threshold_90_epoch are computed and saved for the thesis appendix /
    sensitivity analysis ONLY — they are not scored.
    """
    weight_pca_checkpoints = []
    history = {k: [] for k in ["norm_eff_rank", "layer_eff_rank",
                               "layer_norm_eff_rank", "layer_rank"]}

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        m = compute_weight_pca_metrics_for_checkpoint(model)
        weight_pca_checkpoints.append(epoch)
        for k in history:
            history[k].append(m[k])

    measurements.save_weight_pca_data(weight_pca_checkpoints, history)

    signal_arr = np.asarray(history["norm_eff_rank"], dtype=float)

    # PRIMARY event rule — this is the ONLY epoch scored against grok.
    min_epoch, min_value, min_idx = find_global_minimum_epoch(
        weight_pca_checkpoints, signal_arr)

    # EXPLORATORY ONLY — saved for the record, excluded from PASS/FAIL.
    steepest_epoch, steepest_drop = find_steepest_drop_epoch(
        weight_pca_checkpoints, signal_arr)
    threshold_epoch, threshold_value = find_threshold_crossing_epoch(
        weight_pca_checkpoints, signal_arr, drop_pct=90.0)

    block = {
        "weight_pca_checkpoints": weight_pca_checkpoints,
        "layer_names": list(WEIGHT_PCA_LAYER_NAMES),
        "norm_eff_rank_history": [float(v) for v in signal_arr],
        "grok_epoch": grok_epoch,
        # --- PRIMARY (scored) ---
        "norm_eff_rank_min_epoch": min_epoch,
        "norm_eff_rank_min_value": min_value,
        "norm_eff_rank_min_to_grok_ratio": (
            float(min_epoch) / grok_epoch if grok_epoch else None),
        # --- EXPLORATORY (NOT scored — thesis appendix / sensitivity analysis) ---
        "exploratory": {
            "steepest_drop_epoch": steepest_epoch,
            "steepest_drop_size": steepest_drop,
            "threshold_90_epoch": threshold_epoch,
            "threshold_90_value": threshold_value,
            "note": ("Not fed into the PASS/FAIL table. See "
                     "src/predictors/weight_pca.py module docstring, "
                     "'EVENT RULES', for definitions and the threshold=90 "
                     "assumption to confirm."),
        },
        "norm_eff_rank_first_last": {"first": float(signal_arr[0]), "last": float(signal_arr[-1])},
        "num_checkpoints": len(weight_pca_checkpoints),
    }
    with open(os.path.join(measurements.weight_pca_dir, "weight_pca_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


def _checkpoint_predictor_higher_mi(model, test_loader, ckpt_dir, ckpt_epochs,
                                    measurements, grok_epoch, device):
    """CHECKPOINT_PREDICTOR_FUNCS["higher_mi"] — O-information (Pomarico et
    al. 2025, arXiv:2507.23346; see src/predictors/higher_mi.py). Loads
    each saved checkpoint into `model`, evaluates it on the FULL held-out
    test set via compute_higher_mi_logit_triplet_for_checkpoint (the
    project's 113->3 class reduction: correct / runner-up / rest-mean
    logit per test example), and computes TWO O-information signals from
    the resulting per-checkpoint histories:
      - population O-information (PRIMARY): one Omega per checkpoint,
        estimated across the ~8939 test examples AT that checkpoint.
      - temporal/windowed O-information (EXPLORATORY CONTROL, W=100,
        stride=10): one Omega per window of per-checkpoint MEAN logits,
        to empirically test the population signal's conceptual risk
        against the source paper's own (temporal) sense of
        O-information. See higher_mi.py module docstring, "MAJOR
        ADAPTATION" / "EXPLORATORY CONTROL", for the full reasoning.
    Also computes the Transfer Entropy table (EXPLORATORY ONLY, no event
    epoch — Table-I shape, tau = 1..10, both directions) on the same
    per-checkpoint mean-logit trajectories.

    IMPLEMENTATION NOTE — re-seeding test_loader, not reusing the one
    passed in: the `test_loader` argument this function receives (built
    once in recompute_from_checkpoints / main(), WITHOUT a
    torch.manual_seed(seed) call first) is NOT the same train/test split
    train_one_seed actually trained and evaluated this seed on —
    The split is Nanda's random.shuffle under random.seed(seed), read from
    seed.npy. Passing that seed into get_dataloaders rebuilds the same split.
    This is
    harmless for every OTHER registered checkpoint predictor (spectral
    and age ignore test_loader entirely and rebuild their own TRAIN
    split with an explicit torch.manual_seed(seed) first — see their
    own docstrings above — htsr and weight_pca ignore test_loader too).
    It is NOT harmless here: Higher-MI's whole point is population
    structure across the TEST set specifically, so evaluating it on a
    resampled split that likely includes points the model WAS trained on
    (pre-grok, where train_acc != test_acc, this would bias
    correct/runner_up/rest_mean toward higher-confidence "seen" examples)
    would quietly corrupt the signal, worst on the early checkpoints that
    matter most for a LEADING-indicator claim. Fix, same precedent as
    _checkpoint_predictor_spectral / _checkpoint_predictor_age's
    train_loader rebuild: IGNORE the passed-in test_loader and rebuild a
    freshly, correctly-seeded one here instead."""
    seed = int(np.load(os.path.join(measurements.output_dir, "seed.npy"))[0])
    p = model.output_head.out_features - 1
    batch_size = int(0.3 * p * p)  # full-batch training split (matches get_dataloaders)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, seeded_test_loader = get_dataloaders(number=p, batch_size=batch_size, seed=seed)

    higher_mi_checkpoints = []
    correct_mean_hist, runner_up_mean_hist, rest_mean_mean_hist = [], [], []
    population_omega_hist, population_omega_std_hist = [], []

    for epoch in ckpt_epochs:
        state = torch.load(os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"), map_location=device)
        model.load_state_dict(state)
        triplet = compute_higher_mi_logit_triplet_for_checkpoint(model, seeded_test_loader, device)
        omega_mean, omega_std = o_information_three_way(
            triplet["correct"], triplet["runner_up"], triplet["rest_mean"])
        higher_mi_checkpoints.append(epoch)
        correct_mean_hist.append(float(triplet["correct"].mean()))
        runner_up_mean_hist.append(float(triplet["runner_up"].mean()))
        rest_mean_mean_hist.append(float(triplet["rest_mean"].mean()))
        population_omega_hist.append(omega_mean)
        population_omega_std_hist.append(omega_std)

    higher_mi_checkpoints_arr = np.asarray(higher_mi_checkpoints)
    population_omega_arr = np.asarray(population_omega_hist, dtype=float)

    # EXPLORATORY CONTROL — windowed/temporal O-information on the SAME
    # per-checkpoint mean trajectories, W=100/stride=10 (confirmed
    # defaults). A run with fewer than 100 evaluated checkpoints (e.g. a
    # smoke test) cannot form even one window — guarded, not crashed.
    try:
        temporal_epochs, temporal_omega, temporal_omega_std = compute_temporal_o_information_signal(
            higher_mi_checkpoints_arr, correct_mean_hist, runner_up_mean_hist, rest_mean_mean_hist,
            window=100, stride=10)
    except ValueError as exc:
        temporal_epochs = np.array([])
        temporal_omega = np.array([])
        temporal_omega_std = np.array([])
        print(f"[higher_mi] temporal control signal skipped: {exc}", flush=True)

    measurements.save_higher_mi_data(
        higher_mi_checkpoints_arr, population_omega_arr, temporal_omega,
        {
            "correct_mean": correct_mean_hist,
            "runner_up_mean": runner_up_mean_hist,
            "rest_mean_mean": rest_mean_mean_hist,
            "population_omega_std": population_omega_std_hist,
            "temporal_checkpoints": temporal_epochs,
            "temporal_omega_std": temporal_omega_std,
        })

    # PRIMARY event rule — this is the ONLY epoch scored against grok.
    max_epoch, max_value, _ = find_global_maximum_epoch(
        higher_mi_checkpoints_arr, population_omega_arr)

    # EXPLORATORY ONLY — population signal's zero-crossing (may be None;
    # see higher_mi.py docstring, "EVENT RULE", for why this is not PRIMARY).
    zero_crossing_epoch = find_zero_crossing_epoch(higher_mi_checkpoints_arr, population_omega_arr)

    # EXPLORATORY CONTROL — same two event rules, on the temporal signal.
    if len(temporal_omega) > 0:
        ctrl_max_epoch, ctrl_max_value, _ = find_global_maximum_epoch(temporal_epochs, temporal_omega)
        ctrl_zero_crossing_epoch = find_zero_crossing_epoch(temporal_epochs, temporal_omega)
    else:
        ctrl_max_epoch = ctrl_max_value = ctrl_zero_crossing_epoch = None

    # EXPLORATORY ONLY — Transfer Entropy table, tau=1..10, both
    # directions, full (non-windowed) trajectory of per-checkpoint mean
    # logits. No event epoch; not fed into the PASS/FAIL table.
    te_table = transfer_entropy_table(correct_mean_hist, runner_up_mean_hist)

    block = {
        "higher_mi_checkpoints": higher_mi_checkpoints,
        "population_omega_history": [float(v) for v in population_omega_arr],
        "correct_mean_history": [float(v) for v in correct_mean_hist],
        "runner_up_mean_history": [float(v) for v in runner_up_mean_hist],
        "rest_mean_mean_history": [float(v) for v in rest_mean_mean_hist],
        "grok_epoch": grok_epoch,
        # --- PRIMARY (scored) ---
        "population_omega_max_epoch": max_epoch,
        "population_omega_max_value": max_value,
        "population_omega_max_to_grok_ratio": (
            float(max_epoch) / grok_epoch if grok_epoch else None),
        # --- EXPLORATORY (NOT scored — thesis appendix / sensitivity analysis) ---
        "exploratory": {
            "population_omega_zero_crossing_epoch": zero_crossing_epoch,
            "temporal_control": {
                "window": 100,
                "stride": 10,
                "window_center_epochs": [float(v) for v in temporal_epochs],
                "omega_history": [float(v) for v in temporal_omega],
                "omega_max_epoch": ctrl_max_epoch,
                "omega_max_value": ctrl_max_value,
                "omega_zero_crossing_epoch": ctrl_zero_crossing_epoch,
            },
            "transfer_entropy": {str(tau): v for tau, v in te_table.items()},
            "note": ("Not fed into the PASS/FAIL table. See "
                     "src/predictors/higher_mi.py module docstring, "
                     "'EVENT RULE' and 'EXPLORATORY CONTROL', for the "
                     "reasoning behind every exploratory signal here."),
        },
        "population_omega_first_last": {
            "first": float(population_omega_arr[0]), "last": float(population_omega_arr[-1])},
        "num_checkpoints": len(higher_mi_checkpoints),
    }
    with open(os.path.join(measurements.higher_mi_dir, "higher_mi_signal.json"), "w") as handle:
        json.dump(block, handle, indent=2)
    return block


def _run_commutator_defect_pass(
    model, train_inputs, train_labels, ckpt_dir, ckpt_epochs, device, eta,
):
    """Per-checkpoint Commutator Defect measurement at one probe eta.

    Returns (checkpoints, history_dict) ready for save_commutator_defect_data.
    """
    checkpoints = []
    hist = {
        "median": [], "q25": [], "q75": [],
        "values": [], "grad_norm_A": [], "grad_norm_B": [],
        "eta": float(eta),
        "k_pairs": int(COMMUTATOR_K_PAIRS),
        "batch_size": int(COMMUTATOR_BATCH_SIZE),
        "seed": int(COMMUTATOR_BATCH_SEED),
    }
    for epoch in ckpt_epochs:
        state = torch.load(
            os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"),
            map_location=device,
        )
        model.load_state_dict(state)
        result = compute_commutator_defect(
            model, train_inputs, train_labels,
            eta=float(eta),
            k_pairs=COMMUTATOR_K_PAIRS,
            batch_size=COMMUTATOR_BATCH_SIZE,
            seed=COMMUTATOR_BATCH_SEED,
        )
        checkpoints.append(epoch)
        hist["median"].append(result["median"])
        hist["q25"].append(result["q25"])
        hist["q75"].append(result["q75"])
        hist["values"].append(result["values"])
        hist["grad_norm_A"].append(result["grad_norm_A"])
        hist["grad_norm_B"].append(result["grad_norm_B"])
    return checkpoints, hist


def _checkpoint_predictor_commutator_defect(
    model, test_loader, ckpt_dir, ckpt_epochs, measurements, grok_epoch, device,
):
    """CHECKPOINT_PREDICTOR_FUNCS["commutator_defect"] — Xu (2026)
    Commutator Defect (see src/predictors/commutator_defect.py).

    PRIMARY scored run: frozen eta=1e-3, K=5, batch_size=512,
    batch-draw seed=20261103. Saves under measurements.commutator_defect_dir
    and returns the commutator_defect_predictor block for summary.json.

    test_loader is IGNORED — the probe draws A/B from the TRAINING split.
    recompute_from_checkpoints builds an unseeded loader; this function
    re-seeds from seed.npy and rebuilds the training split exactly as
    Spectral / AGE do (must give 3,830 examples for p=113).

    Optional SECONDARY unscored sensitivity (eta=1e-4): enabled by the
    module flag COMMUTATOR_RUN_SECONDARY_ETA / --commutator_secondary_eta.
    Writes only under measurements.commutator_defect_eta1e4_dir and into
    block["secondary_eta_1e-4"]. Never overwrites the primary arrays and
    never feeds the PRIMARY verdict.
    """
    seed = int(np.load(os.path.join(measurements.output_dir, "seed.npy"))[0])
    p = model.output_head.out_features - 1
    batch_size = int(0.3 * p * p)  # full-batch training split
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _ = get_dataloaders(number=p, batch_size=batch_size, seed=seed)
    train_inputs, train_labels = next(iter(train_loader))
    n_train = int(train_inputs.shape[0])
    if n_train != 3830:
        raise RuntimeError(
            f"Commutator Defect expected 3830 training examples after "
            f"seeded rebuild, got {n_train}"
        )

    # --- PRIMARY (scored): frozen eta=1e-3 ---
    checkpoints, hist = _run_commutator_defect_pass(
        model, train_inputs, train_labels, ckpt_dir, ckpt_epochs, device,
        eta=COMMUTATOR_PRIMARY_ETA,
    )
    measurements.save_commutator_defect_data(
        checkpoints, hist, output_dir=measurements.commutator_defect_dir,
    )

    event_epoch = find_commutator_defect_event_epoch(checkpoints, hist["median"])
    block = {
        "commutator_defect_checkpoints": checkpoints,
        "median_history": hist["median"],
        "q25_history": hist["q25"],
        "q75_history": hist["q75"],
        "values_history": hist["values"],
        "grad_norm_A_history": hist["grad_norm_A"],
        "grad_norm_B_history": hist["grad_norm_B"],
        "eta": float(COMMUTATOR_PRIMARY_ETA),
        "k_pairs": int(COMMUTATOR_K_PAIRS),
        "batch_size": int(COMMUTATOR_BATCH_SIZE),
        "batch_draw_seed": int(COMMUTATOR_BATCH_SEED),
        "grok_epoch": grok_epoch,
        # PRIMARY event (Xu onset). None = miss for this seed.
        "event_epoch": event_epoch,
        "event_to_grok_ratio": (
            float(event_epoch) / grok_epoch
            if (event_epoch is not None and grok_epoch) else None
        ),
        "num_checkpoints": len(checkpoints),
        "note": (
            "PRIMARY scored run uses eta=1e-3 (frozen Stage 1 / Xu 2026). "
            "Secondary eta=1e-4 is unscored sensitivity only."
        ),
    }

    # --- SECONDARY (unscored): eta=1e-4, separate dir, never for verdict ---
    if COMMUTATOR_RUN_SECONDARY_ETA:
        sec_ckpts, sec_hist = _run_commutator_defect_pass(
            model, train_inputs, train_labels, ckpt_dir, ckpt_epochs, device,
            eta=COMMUTATOR_SECONDARY_ETA,
        )
        measurements.save_commutator_defect_data(
            sec_ckpts, sec_hist,
            output_dir=measurements.commutator_defect_eta1e4_dir,
        )
        sec_event = find_commutator_defect_event_epoch(sec_ckpts, sec_hist["median"])
        block["secondary_eta_1e-4"] = {
            "eta": float(COMMUTATOR_SECONDARY_ETA),
            "median_history": sec_hist["median"],
            "event_epoch": sec_event,
            "output_dir": "commutator_defect_eta1e-4",
            "note": (
                "Unscored sensitivity. Same Xu onset rule. Never used for "
                "the PRIMARY PASS/FAIL verdict."
            ),
        }
        with open(
            os.path.join(
                measurements.commutator_defect_eta1e4_dir,
                "commutator_defect_signal.json",
            ),
            "w",
        ) as handle:
            json.dump(block["secondary_eta_1e-4"], handle, indent=2)

    with open(
        os.path.join(
            measurements.commutator_defect_dir, "commutator_defect_signal.json",
        ),
        "w",
    ) as handle:
        # Primary signal.json must not embed the secondary arrays (keep
        # the two runs separate on disk). Write a copy without secondary.
        primary_json = {k: v for k, v in block.items() if k != "secondary_eta_1e-4"}
        json.dump(primary_json, handle, indent=2)
    return block


# Predictors this runner knows how to (re)compute from already-saved
# checkpoints/model_epoch_*.pt alone, with NO retraining. See
# _checkpoint_predictor_spectral for the shared function shape.
# Dropout is not registered here; it runs on the isolated Salah & Yevick
# MLP pipeline (scripts/dropout/train_dropout_salah_yevick.py).
CHECKPOINT_PREDICTOR_FUNCS = {
    "spectral": _checkpoint_predictor_spectral,
    "age": _checkpoint_predictor_age,
    "htsr": _checkpoint_predictor_htsr,
    "weight_pca": _checkpoint_predictor_weight_pca,
    "higher_mi": _checkpoint_predictor_higher_mi,
    "commutator_defect": _checkpoint_predictor_commutator_defect,
}
CHECKPOINT_ONLY_PREDICTORS = set(CHECKPOINT_PREDICTOR_FUNCS.keys())


def recompute_from_checkpoints(seed, args, cfg, device, predictors_to_compute, old_summary):
    """Compute predictor(s) that only need frozen model weights (via
    CHECKPOINT_PREDICTOR_FUNCS) from this seed's already-saved
    checkpoints/model_epoch_*.pt, with NO retraining. Every other block
    in old_summary is carried forward unchanged. Only called when
    predictors_to_compute is a subset of CHECKPOINT_ONLY_PREDICTORS and
    has_saved_checkpoints() is true for this seed."""
    assert predictors_to_compute <= CHECKPOINT_ONLY_PREDICTORS, (
        f"recompute_from_checkpoints can only rebuild "
        f"{sorted(CHECKPOINT_ONLY_PREDICTORS)} from saved weights; the rest "
        f"need the full per-epoch training history")

    out_dir = seed_dir_for(args.output_dir, seed)
    ckpt_dir = checkpoints_dir_for(args.output_dir, seed)
    p = cfg["modulus"]
    batch_size = int(cfg["train_fraction"] * p * p)
    _, test_loader = get_dataloaders(number=p, batch_size=batch_size, seed=seed)

    model = TransformerFourHead(
        vocab_size=p + 1, d_model=cfg["d_model"], num_heads=cfg["num_heads"],
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

    # Grok epoch for EVERY threshold (primary first). Older summaries written
    # before the multi-threshold change have no such block; they are skipped.
    grok_by_threshold = {}
    for t in GROK_THRESHOLDS:
        per_seed = [(s.get("grok_epochs_by_threshold") or {}).get(str(t)) for s in summaries]
        reached = [e for e in per_seed if e is not None]
        grok_by_threshold[str(t)] = {
            "per_seed": per_seed,
            "mean": float(np.mean(reached)) if reached else None,
            "std": float(np.std(reached)) if reached else None,
            "n_grokked": len(reached),
        }
    print("\nGrok epoch by threshold (first epoch test acc > threshold):")
    for t, info in grok_by_threshold.items():
        mean_str = f"mean={info['mean']:.1f}  std={info['std']:.1f}" if info["mean"] is not None else "mean=n/a"
        print(f"  > {t:<5}: {mean_str}  ({info['n_grokked']}/{len(summaries)} seeds)  "
              f"per seed={info['per_seed']}")

    print("\nL2-Norm predictor (per seed):")
    for s in summaries:
        lp = s.get("l2_predictor")
        if lp is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        print(f"  seed {s['seed']}: confirmed peak P_i={lp.get('confirmed_peak_epoch')}  "
              f"global peak epoch={lp.get('peak_epoch')}  "
              f"(grok={s['grok_epoch']})")

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
        nc1 = ap.get("nc1_first_last")
        if nc1 is None and "nc1_history" in ap:
            nc1 = {"first": float(ap["nc1_history"][0]), "last": float(ap["nc1_history"][-1])}
        ratio = ap.get("nc1_min_to_grok_ratio")
        if ratio is None and ap.get("nc1_min_epoch") is not None and s.get("grok_epoch"):
            ratio = float(ap["nc1_min_epoch"]) / s["grok_epoch"]
        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        print(f"  seed {s['seed']}: NC1 {nc1['first']:.4f}->{nc1['last']:.4f}  "
              f"nc1_min_epoch={ap['nc1_min_epoch']}  "
              f"nc1_min/grok={ratio_str}  (grok={s['grok_epoch']})")

    print("\nHTSR Alpha predictor (Martin & Mahoney power-law exponent, per seed):")
    for s in summaries:
        hp = s.get("htsr_predictor")
        if hp is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        al = hp["alpha_first_last"]
        ratio = hp["alpha_min_to_grok_ratio"]
        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        print(f"  seed {s['seed']}: alpha {al['first']:.3f}->{al['last']:.3f}  "
              f"alpha_min_epoch={hp['alpha_min_epoch']}  "
              f"alpha_min/grok={ratio_str}  (grok={s['grok_epoch']})")

    print("\nCommutator Defect predictor (Xu onset, per seed):")
    for s in summaries:
        cd = s.get("commutator_defect_predictor")
        if cd is None:
            print(f"  seed {s['seed']}: n/a (not computed for this seed)")
            continue
        event = cd.get("event_epoch")
        event_str = "None (miss)" if event is None else str(event)
        ratio = cd.get("event_to_grok_ratio")
        ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
        print(f"  seed {s['seed']}: event_epoch={event_str}  "
              f"event/grok={ratio_str}  eta={cd.get('eta')}  "
              f"(grok={s['grok_epoch']})")

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
              f"dips<{GROK_ACC_THRESHOLD}={lc.get(dips_key())}")
    print(f"\n  => {n_limit_cycle}/{len(summaries)} seeds show a post-grok limit cycle")

    agg_path = os.path.join(args.output_dir, "aggregate.json")
    with open(agg_path, "w") as handle:
        json.dump({
            "n_seeds": len(summaries),
            "epochs": args.epochs,
            "grok_epoch_mean": float(np.mean(groks)) if groks else None,
            "grok_epoch_std": float(np.std(groks)) if groks else None,
            "grok_epochs": groks,
            "grok_thresholds": GROK_THRESHOLDS,
            "grok_epochs_by_threshold": grok_by_threshold,
            "n_limit_cycle": n_limit_cycle,
            "seeds": summaries,
        }, handle, indent=2)
    print(f"\nWrote {agg_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Nanda-Unified benchmark runner (Transformer substrate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds; seed i uses torch.manual_seed(i)")
    parser.add_argument("--epochs", type=int, default=TOTAL_STEPS,
                        help="full-batch epochs per seed")
    parser.add_argument(
        "--output_dir", type=str,
        default="08_Experiments/results/results_per_nanda_power",
        help="Results folder for this run. Default is the canonical "
             "results_per_nanda_power folder. The old run is "
             "08_Experiments/results/nanda_unified and is used only when "
             "that path is passed explicitly. This runner never falls "
             "back to it.",
    )
    parser.add_argument("--config", type=str, default="06_Code/configs/nanda_unified.yaml")
    parser.add_argument("--predictors", type=str, default="l2",
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
                             f"0 = use every saved checkpoint as-is.")
    parser.add_argument(
        "--commutator_secondary_eta", action="store_true",
        help="also run unscored Commutator Defect sensitivity at eta=1e-4 "
             "into commutator_defect_eta1e-4/ (never overwrites primary "
             "eta=1e-3; never feeds the PRIMARY verdict)",
    )
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

    # Optional secondary eta=1e-4 pass for Commutator Defect (unscored).
    global COMMUTATOR_RUN_SECONDARY_ETA
    COMMUTATOR_RUN_SECONDARY_ETA = bool(args.commutator_secondary_eta)

    cfg = load_config(args.config)
    apply_config_constants(cfg)
    device = pick_device()

    print("=" * 72)
    print("NANDA-UNIFIED BENCHMARK RUNNER   (Transformer substrate)")
    print("=" * 72)
    print(f"config     : {args.config}")
    print(f"output_dir : {args.output_dir}")
    print(f"device     : {device}")
    print(f"task       : (a + b) mod {cfg['modulus']}   "
          f"'=' token id {cfg['modulus']}   vocab {cfg['vocab_size']}")
    print(f"model      : 4-head, d_model={cfg['d_model']}, "
          f"Nanda randn/sqrt init, W_O, causal mask, warmup {cfg['warmup_steps']}")
    print(f"optimiser  : AdamW lr={cfg['lr']} betas={cfg['betas']} "
          f"weight_decay={cfg['weight_decay']}")
    print(f"run        : seeds={args.seeds}  epochs={args.epochs}  "
          f"full-batch={int(cfg['train_fraction'] * cfg['modulus'] * cfg['modulus'])}")
    print(f"grok thr.  : {GROK_THRESHOLDS} (primary {GROK_ACC_THRESHOLD})")
    print(f"predictors : {sorted(requested_predictors)}")
    print(f"checkpoints: init, then every {SAVE_EVERY} (Nanda save_every), "
          f"plus model_epoch_{args.epochs}.pt after the final step; "
          f"evaluate every {args.eval_every}")
    print("note       : Dropout signal is isolated "
          "(scripts/dropout/train_dropout_salah_yevick.py); not measured here.")
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
