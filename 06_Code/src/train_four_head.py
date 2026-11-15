import torch
from torch.optim import AdamW
from models.transformer_four_head import cross_entropy_high_precision
import numpy as np
import os
import re
import yaml
from predictors.l2_norm import (
    compute_sum_of_squared_weights,
    detect_confirmed_sum_w2_peak,
    sum_w2_peak,
)
from data.modular_arithmetic import get_dataloaders
from models.transformer_four_head import TransformerFourHead
from unified_measurements import PredictorMeasurements

# ======================================================================
# 4-HEAD training loop — Nanda-Unified benchmark protocol.
#
# The substrate is defined ONCE in configs/nanda_unified.yaml and loaded
# below. This script asserts its own constants against that file so the
# two cannot drift silently.
#
# Dropout measurement has been removed from this script. The Salah &
# Yevick Dropout signal is measured on the isolated MLP pipeline
# (scripts/train_dropout_salah_yevick.py), not on this Transformer.
# ======================================================================

LEGACY_FILENAMES = [
    "train_acc_history.npy", "test_acc_history.npy", "loss_history.npy",
    "dropout_gap_epochs.npy", "dropout_gap_history.npy",
    "dropout_train_acc_history.npy", "dropout_eval_acc_history.npy",
    "training_report.pdf",
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

# ---- Load the single common benchmark protocol -----------------------
CONFIG_PATH = os.path.join(project_root, "configs", "nanda_unified.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MODULUS = CFG["task"]["modulus"]
VOCAB_SIZE = CFG["task"]["vocab_size"]
TRAIN_FRACTION = CFG["task"]["train_fraction"]
D_MODEL = CFG["architecture"]["d_model"]
NUM_HEADS = CFG["architecture"]["num_heads"]
LR = CFG["optimizer"]["lr"]
WEIGHT_DECAY = CFG["optimizer"]["weight_decay"]
BETAS = tuple(CFG["optimizer"]["betas"])
NUM_EPOCHS = CFG["training"]["epochs"]
BATCH_SIZE = int(TRAIN_FRACTION * MODULUS * MODULUS)

# Fail loud if this script and the protocol file ever disagree.
assert VOCAB_SIZE == MODULUS + 1, "vocab_size must be modulus + 1"
assert D_MODEL % NUM_HEADS == 0, "d_model must be divisible by num_heads"
print(f"Protocol: {CONFIG_PATH}")
print(f"  p={MODULUS}  vocab={VOCAB_SIZE}  d_model={D_MODEL}  heads={NUM_HEADS}")
print(f"  init=Nanda randn/sqrt  betas={BETAS}  wd={WEIGHT_DECAY}  lr={LR}")
print(f"  epochs={NUM_EPOCHS}  full-batch size={BATCH_SIZE}")

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

data_loader = get_dataloaders(MODULUS, batch_size=BATCH_SIZE, seed=seed)
model = TransformerFourHead(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS
).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
WARMUP_STEPS = int(CFG["training"]["warmup_steps"])
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: min(step / float(WARMUP_STEPS), 1.0)
)

print("Model: TransformerFourHead (num_heads=4, Nanda-Unified substrate)")
print("Optimizer:", optimizer)

num_epochs = NUM_EPOCHS
train_acc_history = []
test_acc_history = []
loss_history = []
sum_w2_history = []

for epoch in range(num_epochs):
    model.eval()
    x, y = next(iter(data_loader[0]))
    x, y = x.to(device), y.to(device)
    equal_sign_logit = model.forward(x)[:, 2, :]
    loss = cross_entropy_high_precision(equal_sign_logit, y)
    total_correct = (equal_sign_logit.argmax(dim=1) == y).sum().item()
    total_samples = len(y)

    with torch.no_grad():
        x_test, y_test = next(iter(data_loader[1]))
        x_test, y_test = x_test.to(device), y_test.to(device)
        equal_sign_logit_test = model.forward(x_test)[:, 2, :]
        test_total_correct = (equal_sign_logit_test.argmax(dim=1) == y_test).sum().item()
        test_total_samples = len(y_test)

    train_acc_history.append(total_correct / total_samples)
    test_acc_history.append(test_total_correct / test_total_samples)
    loss_history.append(loss.item())

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    sum_w2_history.append(compute_sum_of_squared_weights(model))

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
        f"Train Accuracy: {train_acc_history[-1]:.4f}, "
        f"Test Accuracy: {test_acc_history[-1]:.4f}, "
        f"Sum w^2: {sum_w2_history[-1]:.1f}"
    )

# Save all measurements using unified system
measurements.save_training_data(train_acc_history, test_acc_history, loss_history)

grok_epoch = np.argmax(np.array(test_acc_history) > 0.99)

print("\n" + "=" * 60)
print("L2 NORM PREDICTOR: confirmed peak of the total sum of squared weights")
print("=" * 60)

peak_epoch, peak_value = sum_w2_peak(sum_w2_history)
confirmed_epoch = detect_confirmed_sum_w2_peak(sum_w2_history)
print(f"  Global peak epoch: {peak_epoch}  (S = {peak_value})")
print(f"  Confirmed-peak epoch P_i: {confirmed_epoch}")
print(f"  Grok epoch (test acc > 99%): {grok_epoch}")

measurements.save_l2_norm_data(sum_w2_history)

print("\n" + "=" * 60)
print("VISUALIZATION & REPORT GENERATION (4-head model)")
print("=" * 60)

measurements.generate_l2_norm_visualizations(
    sum_w2_history, grok_epoch, peak_epoch, confirmed_epoch
)
print("[OK] L2 Norm visualizations saved to runs/four_head/run_{}/l2_norm/".format(run_number))

# Combined PDF: Dropout page is left empty (rates=[]) because Dropout is
# no longer measured on this Transformer path.
report_path = measurements.generate_combined_report(
    train_acc_history, test_acc_history, loss_history, sum_w2_history,
    [], {}, [],
    grok_epoch, peak_epoch, confirmed_epoch
)
print(f"[OK] PDF report saved to {report_path}")

print("\n" + "=" * 60)
print("RUN {} COMPLETE".format(run_number))
print("=" * 60)
print(f"Grok epoch: {grok_epoch}")
print(f"L2 confirmed-peak epoch: {confirmed_epoch}")
print(f"L2 global peak epoch: {peak_epoch}")
print(f"\nAll measurements saved to runs/four_head/run_{run_number}/")
print(f"  - training/ : raw training data")
print(f"  - l2_norm/ : L2 norm measurements + visualizations")
print(f"  - reports/ : combined PDF report")
print("Note: Dropout is measured by train_dropout_salah_yevick.py, not here.")
