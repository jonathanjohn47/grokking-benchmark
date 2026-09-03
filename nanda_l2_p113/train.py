import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
import os
import re
from predictors.l2_norm import (
    compute_l2_norm,
    compute_fast_slow_moving_averages,
    detect_ma_crossover,
    compute_ma_of_slow_ma,
    compute_noise_floor,
    detect_ma_of_ma_zero_crossing,
)
from data.modular_arithmetic import get_dataloaders
from models.transformer_four_head import TransformerFourHead
from measurements import PredictorMeasurements

# ======================================================================
# nanda_l2_p113 — L2-Norm-only training run on (a + b) mod 113.
#
# This is a trimmed copy of src/train_four_head.py. It is a fully faithful
# Nanda et al. replication. Four deliberate differences from the main
# experiment in src/:
#
#   1. MODULUS = 113 (Nanda's mainline prime), not 97.
#   2. L2-Norm predictor only — the Dropout multi-rate sweep is gone.
#   3. AdamW betas = (0.9, 0.98) — the value Nanda et al. and Power et al.
#      both use — instead of PyTorch's default (0.9, 0.999). See
#      context.md, Sep 3 2026 session, item 9.2.
#   4. Weight init: TransformerFourHead now draws every matrix (embeddings
#      included) from N(0, 0.8/sqrt(d_model)) — the TransformerLens scheme
#      Nanda uses — instead of PyTorch's defaults. This removes the large
#      early L2 collapse (a weight-decay transient on the oversized N(0,1)
#      embedding table, not a grokking feature). See the model file.
#
# Everything else matches src/train_four_head.py: same architecture
# (TransformerFourHead, 4 heads, d_model=128), same lr=1e-3,
# weight_decay=1.0, same 40000 epochs, same full-batch training on a 30/70
# split, same L2-Norm predictor code and windows.
#
# Each run is saved into nanda_l2_p113/runs/run_<N>/ (a NEW numbered
# folder every time), so several independent seeded runs sit side by side
# and are never overwritten.
# ======================================================================

MODULUS = 113


def get_next_run_number(runs_base):
    if not os.path.isdir(runs_base):
        return 1
    existing_numbers = []
    for name in os.listdir(runs_base):
        match = re.fullmatch(r"run_(\d+)", name)
        if match and os.path.isdir(os.path.join(runs_base, name)):
            existing_numbers.append(int(match.group(1)))
    return max(existing_numbers, default=0) + 1


package_root = os.path.dirname(os.path.abspath(__file__))
runs_base = os.path.join(package_root, "runs")
os.makedirs(runs_base, exist_ok=True)

run_number = get_next_run_number(runs_base)
output_dir = os.path.join(runs_base, f"run_{run_number}")
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

# Initialize measurements system (L2 Norm + training only)
measurements = PredictorMeasurements(output_dir, model_type="four_head")

print(f"This is Run {run_number}. All outputs will be saved to nanda_l2_p113/runs/run_{run_number}/")

# NEW random seed on every run. torch.seed() picks a fresh, non-deterministic
# seed and returns the exact number it used, so we both use it and record it in
# one step. Grokking is stochastic (grok epoch moves with the seed), so the
# thesis needs several genuinely independent runs, not one run repeated.
seed = torch.seed()
print(f"Seed: {seed}")
np.save("seed.npy", seed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

data_loader = get_dataloaders(MODULUS, batch_size=int(0.3 * MODULUS * MODULUS))
model = TransformerFourHead(vocab_size=MODULUS + 1, d_model=128, num_heads=4).to(device)

# betas=(0.9, 0.98) matches Nanda et al. and Power et al. (PyTorch's AdamW
# default is (0.9, 0.999)). This is one of the three deliberate differences
# from src/train_four_head.py.
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

print("Model: TransformerFourHead (num_heads=4, matches Nanda et al.)")
print(f"Task: (a + b) mod {MODULUS}, vocab_size = {MODULUS + 1}")
print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 40000
train_acc_history = []
test_acc_history = []
loss_history = []
l2_norm_history = []

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for x, y in data_loader[0]:
        x, y = x.to(device), y.to(device)
        logit = model.forward(x)
        equal_sign_logit = logit[:, 2, :]

        loss = cross_entropy_loss(equal_sign_logit, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = equal_sign_logit.argmax(dim=1)
        total_correct += (predicted == y).sum().item()
        total_samples += len(y)

    test_total_correct = 0
    test_total_samples = 0

    for x_test, y_test in data_loader[1]:
        x_test, y_test = x_test.to(device), y_test.to(device)
        logit_test = model.forward(x_test)
        equal_sign_logit_test = logit_test[:, 2, :]
        predicted_test = equal_sign_logit_test.argmax(dim=1)
        test_total_correct += (predicted_test == y_test).sum().item()
        test_total_samples += len(y_test)

    train_acc_history.append(total_correct / total_samples)
    test_acc_history.append(test_total_correct / test_total_samples)
    loss_history.append(loss.item())
    l2_norm_history.append(compute_l2_norm(model))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc_history[-1]:.4f}, Test Accuracy: {test_acc_history[-1]:.4f}, L2 Norm: {l2_norm_history[-1]:.4f}")

# Save training data
measurements.save_training_data(train_acc_history, test_acc_history, loss_history)

grok_epoch = np.argmax(np.array(test_acc_history) > 0.9)


# L2 Norm Predictor: Moving Average Crossover Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR (4-head model, p=113): Moving Average Crossover")
print("="*60)

epoch_grid, fast_ma, slow_ma = compute_fast_slow_moving_averages(l2_norm_history, fast_window=50, slow_window=200)

detection_epoch = detect_ma_crossover(epoch_grid, fast_ma, slow_ma, skip_epochs=100)

print(f"\nDetection Results:")
if detection_epoch is not None:
    lead_time = grok_epoch - detection_epoch
    print(f"  MA Crossover epoch: {detection_epoch:.1f}")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"  Lead time: {lead_time:.1f} epochs")
else:
    print("  No MA crossover detected")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")

print(f"\nL2 Norm Stats:")
print(f"  Min: {np.min(l2_norm_history):.4f}")
print(f"  Max: {np.max(l2_norm_history):.4f}")
print(f"  Final: {l2_norm_history[-1]:.4f}")


# L2 Norm Predictor: MA-of-MA Zero-Crossing Trigger Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR (4-head model, p=113): MA of Slow MA — Zero-Crossing Trigger")
print("="*60)

fast_ma_of_slow_ma, ma_of_ma_diff = compute_ma_of_slow_ma(slow_ma, fast_window=20)

noise_floor = compute_noise_floor(ma_of_ma_diff, epoch_grid, quiet_epoch_cutoff=90)

trigger_epoch = detect_ma_of_ma_zero_crossing(epoch_grid, ma_of_ma_diff, skip_epochs=100)

print(f"\nNoise floor: {noise_floor:.6f}")
if trigger_epoch is not None:
    trigger_lead_time = grok_epoch - trigger_epoch
    print(f"  Trigger epoch (first zero-crossing): {trigger_epoch:.1f}")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"  Lead time: {trigger_lead_time:.1f} epochs")
else:
    print("  No trigger detected")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")

# Save all L2 Norm measurements
measurements.save_l2_norm_data(
    l2_norm_history, epoch_grid, fast_ma, slow_ma,
    fast_ma_of_slow_ma, ma_of_ma_diff, detection_epoch
)


# ======================================================================
# VISUALIZATION & REPORT GENERATION
# ======================================================================
print("\n" + "="*60)
print("VISUALIZATION & REPORT GENERATION (4-head model, p=113)")
print("="*60)

# Generate standalone L2 Norm visualizations
measurements.generate_l2_norm_visualizations(
    l2_norm_history, epoch_grid, fast_ma, slow_ma,
    fast_ma_of_slow_ma, ma_of_ma_diff, test_acc_history,
    grok_epoch, detection_epoch, trigger_epoch
)
print("[OK] L2 Norm visualizations saved to nanda_l2_p113/runs/run_{}/l2_norm/".format(run_number))

# Generate combined PDF report (training + L2 Norm pages only)
report_path = measurements.generate_combined_report(
    train_acc_history, test_acc_history, loss_history, l2_norm_history,
    epoch_grid, fast_ma, slow_ma, grok_epoch, detection_epoch
)
print(f"[OK] PDF report saved to {report_path} (3 pages: training + L2 Norm)")

print("\n" + "="*60)
print("RUN {} COMPLETE".format(run_number))
print("="*60)
print(f"Grok epoch: {grok_epoch}")
print(f"L2 Norm MA crossover: {detection_epoch if detection_epoch else 'Not detected'}")
print(f"L2 Norm MA-of-MA trigger: {trigger_epoch if trigger_epoch else 'Not detected'}")
print(f"\nAll measurements saved to nanda_l2_p113/runs/run_{run_number}/")
print(f"  - training/ : raw training data")
print(f"  - l2_norm/ : L2 norm measurements + visualizations")
print(f"  - reports/ : combined PDF report")
