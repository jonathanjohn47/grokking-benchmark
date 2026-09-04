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
from predictors.dropout import compute_dropout_gap_multi_rate             # noqa: E402
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


def seed_is_complete(output_dir, seed):
    return os.path.isfile(os.path.join(seed_dir_for(output_dir, seed), "summary.json"))


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


def train_one_seed(seed, args, cfg, device):
    out_dir = seed_dir_for(args.output_dir, seed)
    os.makedirs(out_dir, exist_ok=True)

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

    # per-seed summary + resume sentinel
    grok_epoch = grok_epoch_from(test_acc_history)
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
        "l2_predictor": l2_signals,
        "dropout_final_gap_by_rate": {
            str(r): float(dropout_results[r]["dropout_gap"]) for r in DROPOUT_RATES},
        "limit_cycle_check": limit_cycle_check(test_acc_history, grok_epoch),
        "wall_time_sec": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[seed {seed}] done in {summary['wall_time_sec']}s  "
          f"grok_epoch={grok_epoch}  "
          f"final_test_acc={summary['final_test_acc']:.4f}", flush=True)
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
        lp = s["l2_predictor"]
        print(f"  seed {s['seed']}: MA-crossover={lp['ma_crossover_epoch']}  "
              f"MA-of-MA zero-cross={lp['ma_of_ma_zero_crossing_epoch']}  "
              f"(grok={s['grok_epoch']})")

    print("\nDropout gap at final epoch (per seed, by rate):")
    for s in summaries:
        gaps = s["dropout_final_gap_by_rate"]
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
    args = parser.parse_args()

    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(REPO_ROOT, args.output_dir)
    if not os.path.isabs(args.config):
        args.config = os.path.join(REPO_ROOT, args.config)
    os.makedirs(args.output_dir, exist_ok=True)

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
        if seed_is_complete(args.output_dir, seed):
            with open(os.path.join(seed_dir_for(args.output_dir, seed),
                                   "summary.json")) as handle:
                summaries.append(json.load(handle))
            print(f"[seed {seed}] complete - skipping (found summary.json)")
            continue
        summaries.append(train_one_seed(seed, args, cfg, device))

    aggregate(summaries, args)


if __name__ == "__main__":
    main()
