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
from predictors.dropout import compute_dropout_gap, compute_dropout_gap_multi_rate
from unified_measurements import PredictorMeasurements

# ======================================================================
# 4-HEAD VARIANT of train.py.
#
# This script is a separate copy of the main training loop — it does NOT
# modify train.py or any of its saved outputs. The only real difference
# from train.py is the model: TransformerFourHead(num_heads=4) instead of
# the original single-head Transformer. Same task (p=97), same optimiser
# (AdamW, lr=1e-3, weight_decay=1.0), same full-batch training, same 30/70
# split, same L2 Norm and Dropout Gap predictor code (reused as-is from
# src/predictors/ — nothing predictor-specific needed to change, since
# both predictors work on any model that exposes .parameters() and
# .dropout1/.dropout2, which this model does too).
#
# Outputs are saved under results/four_head/run_<N>/ (created if it does not
# exist yet), NOT in the project root, so they never overwrite train.py's
# own train_acc_history.npy / l2_norm_history.npy / dropout_gap_history.npy
# / etc., which belong to the original single-head run.
#
# RUN NUMBERING: every time this script is run, it saves into a NEW
# numbered subfolder (run_1, run_2, run_3, ...) instead of overwriting the
# previous run's results. This is needed because grokking is stochastic —
# the grok epoch moves with the random seed — so a real comparison needs
# several independent runs kept side by side, not one run's numbers
# overwritten by the next. plot_results_four_head.py reads every run_<N>
# folder it finds and plots them together for comparison.
# ======================================================================

LEGACY_FILENAMES = [
    "train_acc_history.npy", "test_acc_history.npy", "loss_history.npy",
    "l2_norm_history.npy", "dropout_gap_epochs.npy", "dropout_gap_history.npy",
    "dropout_train_acc_history.npy", "dropout_eval_acc_history.npy",
    "epoch_grid.npy", "fast_ma.npy", "slow_ma.npy", "fast_ma_of_slow_ma.npy",
    "ma_of_ma_diff.npy", "training_report.pdf",
    "ma_of_slow_ma_crossover.png", "ma_of_slow_ma_diff.png",
    "ma_of_slow_ma_diff_linear.png", "ma_of_ma_diff_vs_grokking_linear.png",
    "grokking_curve.png", "loss_curve.png", "l2_norm_curve.png", "dropout_gap_curve.png",
]


def migrate_legacy_flat_run(four_head_dir):
    """
    Earlier versions of this script saved directly into results/four_head/
    instead of results/four_head/run_<N>/. If such files are found here, and
    run_1/ does not exist yet, move them into run_1/ so they are counted
    as the first run instead of silently sitting outside the new
    numbering scheme (and instead of being silently overwritten by this
    run, which is what would have happened before this change).
    """
    run_1_dir = os.path.join(four_head_dir, "run_1")
    if os.path.isdir(run_1_dir):
        return
    legacy_present = any(
        os.path.isfile(os.path.join(four_head_dir, name)) for name in LEGACY_FILENAMES
    )
    if not legacy_present:
        return
    os.makedirs(run_1_dir, exist_ok=True)
    for name in LEGACY_FILENAMES:
        src = os.path.join(four_head_dir, name)
        if os.path.isfile(src):
            os.rename(src, os.path.join(run_1_dir, name))
    print("[MIGRATION] Found results saved directly in results/four_head/ by an earlier "
          "version of this script — moved them into results/four_head/run_1/ so they are "
          "counted as Run 1 instead of being overwritten.")


def get_next_run_number(four_head_dir):
    if not os.path.isdir(four_head_dir):
        return 1
    existing_numbers = []
    for name in os.listdir(four_head_dir):
        match = re.fullmatch(r"run_(\d+)", name)
        if match and os.path.isdir(os.path.join(four_head_dir, name)):
            existing_numbers.append(int(match.group(1)))
    return max(existing_numbers, default=0) + 1


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

four_head_dir = os.path.join(project_root, "runs", "four_head")
migrate_legacy_flat_run(four_head_dir)
run_number = get_next_run_number(four_head_dir)

output_dir = os.path.join(four_head_dir, f"run_{run_number}")
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

# Initialize unified measurements system
measurements = PredictorMeasurements(output_dir, model_type="four_head")

print(f"This is Run {run_number}. All outputs will be saved to runs/four_head/run_{run_number}/")

# NEW random seed on every run, instead of a fixed one. torch.seed() picks
# a fresh, non-deterministic seed itself and also returns the exact number
# it used, so we can both use it and record it in the same step. This is
# needed because grokking is stochastic (grok epoch moves with the seed —
# see the L2 Norm predictor's 3-run history) and the thesis needs several
# genuinely independent 4-head runs, not the same run repeated.
seed = torch.seed()
print(f"Seed: {seed}")
np.save("seed.npy", seed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = TransformerFourHead(vocab_size=98, d_model=128, num_heads=4).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Model: TransformerFourHead (num_heads=4, matches Nanda et al.)")
print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 40000
train_acc_history = []
test_acc_history = []
loss_history = []
l2_norm_history = []
dropout_gap_epochs = []
dropout_gap_history = []
dropout_train_acc_history = []
dropout_eval_acc_history = []
dropout_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
dropout_gap_history_by_rate = {rate: [] for rate in dropout_rates}

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

    # Dropout Gap Predictor: Multi-rate sweep across 5 rates (0.1, 0.3, 0.5, 0.7, 0.9)
    # to determine if the gap-narrowing pattern is robust across rates or rate-dependent.
    results = compute_dropout_gap_multi_rate(model, data_loader[1], dropout_rates)

    # Store results for each rate
    for rate in dropout_rates:
        dropout_gap_history_by_rate[rate].append(results[rate]["dropout_gap"])

    # Also keep the p=0.9 results in the single-rate arrays for compatibility
    dropout_gap_epochs.append(epoch + 1)
    dropout_gap_history.append(results[0.9]["dropout_gap"])
    dropout_train_acc_history.append(results[0.9]["train_accuracy"])
    dropout_eval_acc_history.append(results[0.9]["eval_accuracy"])

    # compute_dropout_gap() leaves the model in eval() mode with p reset
    # to 0.0. Training must resume in train() mode, so we restore it here
    # before the next epoch's training pass begins.
    model.train()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc_history[-1]:.4f}, Test Accuracy: {test_acc_history[-1]:.4f}, L2 Norm: {l2_norm_history[-1]:.4f}, Dropout Gap (p=0.9): {results[0.9]['dropout_gap']:.4f}")

# Save all measurements using unified system
measurements.save_training_data(train_acc_history, test_acc_history, loss_history)

grok_epoch = np.argmax(np.array(test_acc_history) > 0.9)



# L2 Norm Predictor: Moving Average Crossover Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR (4-head model): Moving Average Crossover")
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
print("L2 NORM PREDICTOR (4-head model): MA of Slow MA — Zero-Crossing Trigger")
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

# Dropout Predictor: gap tracked at every epoch during training
print("\n" + "="*60)
print("DROPOUT PREDICTOR (4-head model): Gap tracked every epoch")
print("="*60)
print(f"\nFinal Dropout Gap Check (rate=0.9, epoch {dropout_gap_epochs[-1]}):")
print(f"  Accuracy with dropout (train mode, p=0.9): {dropout_train_acc_history[-1]:.4f}")
print(f"  Accuracy without dropout (eval mode, p=0.0): {dropout_eval_acc_history[-1]:.4f}")
print(f"  Gap: {dropout_gap_history[-1]:.4f}")

# Save all Dropout measurements
measurements.save_dropout_data(
    dropout_gap_epochs, dropout_gap_history, dropout_gap_history_by_rate,
    dropout_train_acc_history, dropout_eval_acc_history, dropout_rates
)


# ======================================================================
# VISUALIZATION & REPORT GENERATION
# ======================================================================
print("\n" + "="*60)
print("VISUALIZATION & REPORT GENERATION (4-head model)")
print("="*60)

# Generate standalone L2 Norm visualizations
measurements.generate_l2_norm_visualizations(
    l2_norm_history, epoch_grid, fast_ma, slow_ma,
    fast_ma_of_slow_ma, ma_of_ma_diff, test_acc_history,
    grok_epoch, detection_epoch, trigger_epoch
)
print("[OK] L2 Norm visualizations saved to runs/four_head/run_{}/l2_norm/".format(run_number))

# Generate standalone Dropout visualizations
measurements.generate_dropout_visualizations(
    dropout_gap_epochs, dropout_gap_history_by_rate,
    dropout_rates, test_acc_history, grok_epoch
)
print("[OK] Dropout visualizations saved to runs/four_head/run_{}/dropout/".format(run_number))

# Generate combined PDF report
report_path = measurements.generate_combined_report(
    train_acc_history, test_acc_history, loss_history, l2_norm_history,
    dropout_gap_epochs, dropout_gap_history_by_rate, dropout_rates,
    epoch_grid, fast_ma, slow_ma, grok_epoch, detection_epoch
)
print(f"[OK] PDF report saved to {report_path} (4 pages: all measurements)")

print("\n" + "="*60)
print("RUN {} COMPLETE".format(run_number))
print("="*60)
print(f"Grok epoch: {grok_epoch}")
print(f"L2 Norm MA crossover: {detection_epoch if detection_epoch else 'Not detected'}")
print(f"L2 Norm MA-of-MA trigger: {trigger_epoch if trigger_epoch else 'Not detected'}")
print(f"\nAll measurements saved to runs/four_head/run_{}/".format(run_number))
print(f"  - training/ : raw training data")
print(f"  - l2_norm/ : L2 norm measurements + visualizations")
print(f"  - dropout/ : dropout measurements + visualizations")
print(f"  - reports/ : combined PDF report")
