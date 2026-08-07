import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
import os
from predictors.l2_norm import (
    compute_l2_norm,
    compute_fast_slow_moving_averages,
    detect_ma_crossover,
    compute_ma_of_slow_ma,
    compute_noise_floor,
    detect_ma_of_ma_zero_crossing,
)
from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

# Save results to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = Transformer(vocab_size=98, d_model=128).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 10000
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
    if (epoch + 1) % 100 == 0:
        compute_l2 = l2_norm_history[-1]
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc_history[-1]:.4f}, Test Accuracy: {test_acc_history[-1]:.4f}, L2 Norm: {compute_l2:.4f}")

np.save("train_acc_history.npy", train_acc_history)
np.save("test_acc_history.npy", test_acc_history)
np.save("loss_history.npy", loss_history)
np.save("l2_norm_history.npy", l2_norm_history)
print("Training data saved:")
print("  - train_acc_history.npy")
print("  - test_acc_history.npy")
print("  - loss_history.npy")
print("  - l2_norm_history.npy")



# L2 Norm Predictor: Moving Average Crossover Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR: Moving Average Crossover")
print("="*60)

# Compute fast and slow moving averages on a log-epoch-uniform grid
# (matches the log-x plot's visual spacing, so the window covers equal
# visual width everywhere instead of scribbling at high epochs)
# Fast MA (window=50 grid points): responsive to recent changes
# Slow MA (window=200 grid points): captures overall trend
epoch_grid, fast_ma, slow_ma = compute_fast_slow_moving_averages(l2_norm_history, fast_window=50, slow_window=200)

# Detect crossover point: where fast MA crosses above slow MA
detection_epoch = detect_ma_crossover(epoch_grid, fast_ma, slow_ma, skip_epochs=100)

# Find when test accuracy actually jumps (crosses 90%)
grok_epoch = np.argmax(np.array(test_acc_history) > 0.9)

# Save data for plotting
np.save("epoch_grid.npy", epoch_grid)
np.save("fast_ma.npy", fast_ma)
np.save("slow_ma.npy", slow_ma)

# Report results
print(f"\nDetection Results:")
if detection_epoch is not None:
    lead_time = grok_epoch - detection_epoch
    print(f"  MA Crossover epoch: {detection_epoch:.1f}")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"  Lead time: {lead_time:.1f} epochs")
else:
    print("  No MA crossover detected")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")

# Debug stats
print(f"\nL2 Norm Stats:")
print(f"  Min: {np.min(l2_norm_history):.4f}")
print(f"  Max: {np.max(l2_norm_history):.4f}")
print(f"  Final: {l2_norm_history[-1]:.4f}")


# L2 Norm Predictor: MA-of-MA Zero-Crossing Trigger Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR: MA of Slow MA — Zero-Crossing Trigger")
print("="*60)

# Apply a second, faster-window MA on top of slow_ma itself. slow_ma has
# already filtered out raw L2 norm noise, so this reveals slow_ma's own
# turning points cleanly instead of chasing noise in the raw signal.
fast_ma_of_slow_ma, ma_of_ma_diff = compute_ma_of_slow_ma(slow_ma, fast_window=20)

# Noise floor: how big this difference is during the quiet early-training
# region, before any real dynamics begin — printed for context only, no
# longer used to decide the trigger (see detect_ma_of_ma_zero_crossing).
noise_floor = compute_noise_floor(ma_of_ma_diff, epoch_grid, quiet_epoch_cutoff=90)

# Trigger: first epoch (after skip_epochs) where the difference crosses from
# positive to negative. Confirmed on two runs to land in a consistent,
# unambiguous spot even though the crossing itself is weak in magnitude.
trigger_epoch = detect_ma_of_ma_zero_crossing(epoch_grid, ma_of_ma_diff, skip_epochs=100)

np.save("fast_ma_of_slow_ma.npy", fast_ma_of_slow_ma)
np.save("ma_of_ma_diff.npy", ma_of_ma_diff)

print(f"\nNoise floor: {noise_floor:.6f}")
if trigger_epoch is not None:
    trigger_lead_time = grok_epoch - trigger_epoch
    print(f"  Trigger epoch (first zero-crossing): {trigger_epoch:.1f}")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"  Lead time: {trigger_lead_time:.1f} epochs")
else:
    print("  No trigger detected")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
