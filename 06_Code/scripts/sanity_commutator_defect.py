#!/usr/bin/env python3
"""
Stage 3 / 3b / 3c sanity checks for the Commutator Defect predictor.

Does NOT touch unified_measurements.py, run_nanda_benchmark.py, or
analyze_nanda_unified.py. Does NOT commit.

Loading pattern (same as Spectral / AGE recompute paths):
  seed = int(np.load(seed_0/seed.npy)[0])
  torch.manual_seed(seed); np.random.seed(seed)
  train_loader, _ = get_dataloaders(number=p, batch_size=int(0.3*p*p))

recompute_from_checkpoints builds an unseeded loader; Spectral/AGE ignore
that and rebuild with the seeded split. We copy Spectral/AGE here.

Stage 3b adds: eta sweep table + old-vs-new numerator comparison.
Stage 3c: default eta 1e-4; Check 2 compares 1e-4 vs 5e-5; sweep
includes 5e-5 and prints per-pair D at epoch 12000.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "06_Code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from data.modular_arithmetic import get_dataloaders  # noqa: E402
from models.transformer_four_head import TransformerFourHead  # noqa: E402
from predictors.commutator_defect import (  # noqa: E402
    _clone_detached,
    _grads_at,
    _nan_summary,
    _params_difference_norm,
    _params_minus_eta_grad,
    _trainable_param_dict,
    _vector_norm,
    compute_commutator_defect,
    defect_one_pair,
)

SEED_INDEX = 0  # first of the 5 seeds; grok epoch 12987
OUTPUT_DIR = os.path.join(REPO_ROOT, "08_Experiments", "results", "nanda_unified")
SEED_DIR = os.path.join(OUTPUT_DIR, f"seed_{SEED_INDEX}")
CKPT_DIR = os.path.join(SEED_DIR, "checkpoints")
P = 113
D_MODEL = 128
NUM_HEADS = 4
INIT_STD = 0.8 / math.sqrt(D_MODEL)
BATCH_DRAW_SEED = 20261103
WANTED_EPOCHS = (1000, 6000, 12000)


def _nearest_saved_epoch(wanted: int, saved: list[int]) -> int:
    return min(saved, key=lambda e: (abs(e - wanted), e))


def _load_train_split():
    """Rebuild the seeded training split exactly as Spectral / AGE do."""
    seed = int(np.load(os.path.join(SEED_DIR, "seed.npy"))[0])
    batch_size = int(0.3 * P * P)  # full-batch training split
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, _ = get_dataloaders(number=P, batch_size=batch_size)
    train_inputs, train_labels = next(iter(train_loader))
    return seed, train_inputs, train_labels


def _build_model():
    return TransformerFourHead(
        vocab_size=P + 1,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        init_std=INIT_STD,
    )


def _load_checkpoint(epoch: int) -> torch.nn.Module:
    model = _build_model()
    path = os.path.join(CKPT_DIR, f"model_epoch_{epoch}.pt")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _dicts_exactly_equal(a, b) -> bool:
    """Exact equality, treating NaN == NaN as equal."""
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        if a is None and b is None:
            return True
        if (a is None) != (b is None):
            return False
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_dicts_exactly_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_dicts_exactly_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def _caller_weights_unchanged(model, snapshot, device, dtype) -> bool:
    if next(model.parameters()).device != device:
        return False
    if next(model.parameters()).dtype != dtype:
        return False
    for name, param in model.named_parameters():
        if name not in snapshot:
            return False
        if not torch.equal(param, snapshot[name]):
            return False
    return True


def check1_same_batch_zero(model_6000, train_inputs, train_labels) -> bool:
    print("\n=== Check 1 — Same batch twice gives zero ===")
    probe = copy.deepcopy(model_6000).cpu().to(dtype=torch.float64)
    probe.eval()
    theta0 = _clone_detached(_trainable_param_dict(probe))

    inputs_cpu = train_inputs.detach().cpu()
    labels_cpu = train_labels.detach().cpu().long()
    n = int(inputs_cpu.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BATCH_DRAW_SEED)
    idx = torch.randint(0, n, (512,), generator=generator)
    x, y = inputs_cpu[idx], labels_cpu[idx]

    grad_fn = lambda params, _x=x, _y=y: _grads_at(probe, params, _x, _y)
    result = defect_one_pair(theta0, grad_fn, grad_fn, eta=1e-3)
    d_value = result["D"]
    passed = (d_value == 0.0)
    print(f"  D = {d_value!r}")
    print(f"  Check 1: {'PASS' if passed else 'FAIL'}")
    return passed


def check2_halving_eta(models_by_epoch, train_inputs, train_labels, timings) -> bool:
    print("\n=== Check 2 — Halving eta keeps D the same (1e-4 vs 5e-5) ===")
    all_ok = True
    for epoch, model in models_by_epoch.items():
        t0 = time.perf_counter()
        out1 = compute_commutator_defect(
            model, train_inputs, train_labels, eta=1e-4,
        )
        timings.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        out2 = compute_commutator_defect(
            model, train_inputs, train_labels, eta=5e-5,
        )
        timings.append(time.perf_counter() - t0)

        d1 = out1["median"]
        d2 = out2["median"]
        if d1 is None or d1 == 0.0:
            rel = float("inf")
            ok = False
        else:
            rel = abs(d1 - d2) / abs(d1)
            ok = rel < 0.05
        all_ok = all_ok and ok
        print(
            f"  epoch {epoch}: median(eta=1e-4)={d1!r}  "
            f"median(eta=5e-5)={d2!r}  |D1-D2|/D1={rel:.6g}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print(f"  Check 2: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def check3_reproducible(model_6000, train_inputs, train_labels, timings) -> bool:
    print("\n=== Check 3 — Reproducible + caller model untouched ===")
    device = next(model_6000.parameters()).device
    dtype = next(model_6000.parameters()).dtype
    snapshot = {
        name: param.detach().clone()
        for name, param in model_6000.named_parameters()
    }

    t0 = time.perf_counter()
    out1 = compute_commutator_defect(model_6000, train_inputs, train_labels)
    timings.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    out2 = compute_commutator_defect(model_6000, train_inputs, train_labels)
    timings.append(time.perf_counter() - t0)

    equal = _dicts_exactly_equal(out1, out2)
    untouched = _caller_weights_unchanged(model_6000, snapshot, device, dtype)
    passed = equal and untouched
    print(f"  dicts exactly equal: {equal}")
    print(f"  caller weights/device/dtype unchanged: {untouched}")
    print(f"  median={out1['median']!r}")
    print(f"  Check 3: {'PASS' if passed else 'FAIL'}")
    return passed


def check4_two_param_toy() -> bool:
    print("\n=== Check 4 — 2-parameter toy (D = sqrt(2.5)) ===")
    H_A = torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    c_A = torch.tensor([1.0, 0.0], dtype=torch.float64)
    H_B = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    c_B = torch.tensor([0.0, 1.0], dtype=torch.float64)

    theta0 = {"theta": torch.zeros(2, dtype=torch.float64)}

    def grad_a_fn(params):
        theta = params["theta"]
        return (H_A @ (theta - c_A),)

    def grad_b_fn(params):
        theta = params["theta"]
        return (H_B @ (theta - c_B),)

    expected = math.sqrt(2.5)
    all_ok = True
    for eta in (0.1, 1e-3):
        result = defect_one_pair(theta0, grad_a_fn, grad_b_fn, eta=eta)
        d_value = result["D"]
        err = abs(d_value - expected)
        ok = err < 1e-6
        all_ok = all_ok and ok
        print(
            f"  eta={eta}: D={d_value!r}  |D - sqrt(2.5)|={err:.6g}  "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # theta_AB / theta_BA are NOT used for D any more; build them only to print.
    eta = 0.1
    g_a = grad_a_fn(theta0)
    g_b = grad_b_fn(theta0)
    after_a = _params_minus_eta_grad(theta0, g_a, eta)
    after_b = _params_minus_eta_grad(theta0, g_b, eta)
    theta_ab = _params_minus_eta_grad(after_a, grad_b_fn(after_a), eta)
    theta_ba = _params_minus_eta_grad(after_b, grad_a_fn(after_b), eta)
    print(f"  eta=0.1 theta_AB = {theta_ab['theta'].tolist()}")
    print(f"  eta=0.1 theta_BA = {theta_ba['theta'].tolist()}")
    print(f"  expected D = sqrt(2.5) = {expected!r}")
    print(f"  Check 4: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def _defect_one_pair_old_style(theta0, grad_a_fn, grad_b_fn, eta: float) -> float:
    """Pre-Stage-3b numerator: ||theta_AB - theta_BA|| from the two weight vectors.

    Kept only inside this sanity script so we can show old-vs-new medians.
    Production code no longer uses this path.
    """
    g_a = grad_a_fn(theta0)
    g_b = grad_b_fn(theta0)
    norm_a = _vector_norm(g_a)
    norm_b = _vector_norm(g_b)
    if (
        (not torch.isfinite(norm_a))
        or (not torch.isfinite(norm_b))
        or float(norm_a.item()) == 0.0
        or float(norm_b.item()) == 0.0
    ):
        return float("nan")

    theta_after_a = _params_minus_eta_grad(theta0, g_a, eta)
    theta_after_b = _params_minus_eta_grad(theta0, g_b, eta)
    g_b_at_after_a = grad_b_fn(theta_after_a)
    g_a_at_after_b = grad_a_fn(theta_after_b)
    theta_ab = _params_minus_eta_grad(theta_after_a, g_b_at_after_a, eta)
    theta_ba = _params_minus_eta_grad(theta_after_b, g_a_at_after_b, eta)

    numer = _params_difference_norm(theta_ab, theta_ba)
    denom = (eta * norm_a) * (eta * norm_b)
    if (not torch.isfinite(numer)) or (not torch.isfinite(denom)) or float(denom.item()) == 0.0:
        return float("nan")
    d_value = numer / denom
    if not torch.isfinite(d_value):
        return float("nan")
    return float(d_value.detach().cpu().item())


def _old_style_median(model, train_inputs, train_labels, eta: float = 1e-3):
    """Recompute the Stage-3a median with the old weight-subtraction numerator."""
    probe = copy.deepcopy(model).cpu().to(dtype=torch.float64)
    probe.eval()
    theta0 = _clone_detached(_trainable_param_dict(probe))
    inputs_cpu = train_inputs.detach().cpu()
    labels_cpu = train_labels.detach().cpu().long()
    n_examples = int(inputs_cpu.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BATCH_DRAW_SEED)

    values = []
    for _ in range(5):
        idx_a = torch.randint(0, n_examples, (512,), generator=generator)
        idx_b = torch.randint(0, n_examples, (512,), generator=generator)
        x_a, y_a = inputs_cpu[idx_a], labels_cpu[idx_a]
        x_b, y_b = inputs_cpu[idx_b], labels_cpu[idx_b]
        d_value = _defect_one_pair_old_style(
            theta0,
            (lambda params, _x=x_a, _y=y_a: _grads_at(probe, params, _x, _y)),
            (lambda params, _x=x_b, _y=y_b: _grads_at(probe, params, _x, _y)),
            eta,
        )
        values.append(d_value)
    median, _, _ = _nan_summary(values)
    return median


def eta_sweep_table(models_by_epoch, train_inputs, train_labels) -> None:
    print("\n=== Eta sweep (new numerator) — medians ===")
    etas = (1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5)
    header = f"  {'epoch':>6} |" + "".join(
        f"{('eta=' + format(e, 'g')):>14}" for e in etas
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Cache full outputs so we can reuse values for the 12000 per-pair
    # print and the epoch-1000 float-rounding guard without recompute.
    cache: dict[tuple[int, float], dict] = {}
    for epoch, model in models_by_epoch.items():
        medians = []
        for eta in etas:
            out = compute_commutator_defect(
                model, train_inputs, train_labels, eta=eta,
            )
            cache[(epoch, eta)] = out
            medians.append(out["median"])
        cells = [
            f"{m:12.6f}" if m is not None else f"{'nan':>12}"
            for m in medians
        ]
        print(f"  {epoch:>6} |" + "".join(f"  {c}" for c in cells))

    epoch_12000 = min(models_by_epoch, key=lambda e: (abs(e - 12000), e))
    print(f"\n=== Per-pair D at epoch {epoch_12000} (all 5 pairs) ===")
    for eta in etas:
        vals = cache[(epoch_12000, eta)]["values"]
        formatted = ", ".join(
            f"{v:.6f}" if v == v else "nan" for v in vals
        )
        print(f"  eta={format(eta, 'g')}: [{formatted}]")

    # Float-rounding guard: at epoch 1000, 5e-5 must still match 1e-4
    # to within 1e-4 relative (eta not yet so small that rounding returns).
    epoch_1000 = min(models_by_epoch, key=lambda e: (abs(e - 1000), e))
    d_1e4 = cache[(epoch_1000, 1e-4)]["median"]
    d_5e5 = cache[(epoch_1000, 5e-5)]["median"]
    print(
        f"\n=== Float-rounding guard (epoch {epoch_1000}: "
        f"eta=5e-5 vs 1e-4) ==="
    )
    if d_1e4 is None or d_1e4 == 0.0 or d_5e5 is None:
        rel = float("nan")
        ok = False
    else:
        rel = abs(d_1e4 - d_5e5) / abs(d_1e4)
        ok = rel < 1e-4
    print(
        f"  median(1e-4)={d_1e4!r}  median(5e-5)={d_5e5!r}  "
        f"|D1-D2|/D1={rel:.6g}  (need < 1e-4)  "
        f"{'PASS' if ok else 'FAIL'}"
    )


def old_vs_new_table(models_by_epoch, train_inputs, train_labels) -> None:
    print("\n=== Old vs new numerator (eta = 1e-3) ===")
    print(
        f"  {'epoch':>6} | {'old median':>14} | {'new median':>14} | "
        f"{'|old-new|/|old|':>16}"
    )
    print("  " + "-" * 60)
    for epoch, model in models_by_epoch.items():
        old_med = _old_style_median(model, train_inputs, train_labels, eta=1e-3)
        new_out = compute_commutator_defect(
            model, train_inputs, train_labels, eta=1e-3,
        )
        new_med = new_out["median"]
        if old_med is None or old_med == 0.0 or new_med is None:
            rel = float("nan")
        else:
            rel = abs(old_med - new_med) / abs(old_med)
        print(
            f"  {epoch:>6} | {old_med:14.6f} | {new_med:14.6f} | {rel:16.6g}"
        )
        if epoch == 1000:
            note = "expect close agreement (gradients not tiny)"
        elif epoch == 6000:
            note = "expect LARGE change (old value was float noise)"
        else:
            note = "mixed; some pairs were already stable"
        print(f"           note: {note}")


def main() -> int:
    print("Commutator Defect — Stage 3c sanity checks (default eta=1e-4)")
    print(f"SEED_DIR = {SEED_DIR}")

    seed, train_inputs, train_labels = _load_train_split()
    n_train = int(train_inputs.shape[0])
    print(f"training seed (from seed.npy) = {seed}")
    print(f"rebuilt train split size = {n_train}")
    if n_train != 3830:
        print(f"FAIL: expected 3830 training examples, got {n_train}")
        return 1

    saved = sorted(
        int(name[len("model_epoch_"):-len(".pt")])
        for name in os.listdir(CKPT_DIR)
        if name.startswith("model_epoch_") and name.endswith(".pt")
    )
    epochs_used = [_nearest_saved_epoch(w, saved) for w in WANTED_EPOCHS]
    print(f"wanted epochs {list(WANTED_EPOCHS)} -> using {epochs_used}")

    models_by_epoch = {e: _load_checkpoint(e) for e in epochs_used}
    model_6000 = models_by_epoch[epochs_used[1]]  # nearest to 6000

    timings: list[float] = []

    r1 = check1_same_batch_zero(model_6000, train_inputs, train_labels)
    r2 = check2_halving_eta(models_by_epoch, train_inputs, train_labels, timings)
    r3 = check3_reproducible(model_6000, train_inputs, train_labels, timings)
    r4 = check4_two_param_toy()

    eta_sweep_table(models_by_epoch, train_inputs, train_labels)
    old_vs_new_table(models_by_epoch, train_inputs, train_labels)

    if timings:
        avg = sum(timings) / len(timings)
        print("\n=== Timing ===")
        print(f"  compute_commutator_defect calls: {len(timings)}")
        print(f"  wall-clock seconds per call (average): {avg:.4f}")
        print(f"  individual (s): {[round(t, 4) for t in timings]}")
        print(
            f"  Stage 5 estimate for 2005 snapshots: "
            f"~{avg * 2005 / 3600:.2f} hours "
            f"(single-threaded, one seed's 401 x 5 would differ)"
        )

    print("\n=== Summary ===")
    for name, ok in (
        ("Check 1 same-batch zero", r1),
        ("Check 2 halving eta", r2),
        ("Check 3 reproducible", r3),
        ("Check 4 2-param toy", r4),
    ):
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    return 0 if (r1 and r2 and r3 and r4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
