import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Look for files in project root (parent of src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Load saved training data
train_acc = np.load("train_acc_history.npy")
test_acc = np.load("test_acc_history.npy")
loss = np.load("loss_history.npy")
l2_norm = np.load("l2_norm_history.npy")

# Load MA crossover data (if available)
# epoch_grid is uniform in log10(epoch) — always plot fast_ma/slow_ma against
# epoch_grid, never against range(1, num_epochs+1), otherwise the log-uniform
# resampling that fixed the smoothing is undone.
try:
    epoch_grid = np.load("epoch_grid.npy")
    fast_ma = np.load("fast_ma.npy")
    slow_ma = np.load("slow_ma.npy")
    has_ma_data = True
except FileNotFoundError:
    has_ma_data = False
    print("Warning: MA data not found, plotting without it")

l2_norm_history = np.load("l2_norm_history.npy")
print("First 20 L2 norms:", l2_norm_history[:20])
print("Shape:", l2_norm_history.shape)

num_epochs = len(train_acc)

# Create figure with subplots (3x2 if MA data available, else 2x2)
if has_ma_data:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
else:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Grokking curve (Accuracy)
axes[0, 0].plot(range(1, num_epochs + 1), train_acc, label="Train Accuracy", linewidth=2)
axes[0, 0].plot(range(1, num_epochs + 1), test_acc, label="Test Accuracy", linewidth=2)
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlabel("Epoch (log scale)")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Grokking Curve: Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(range(1, num_epochs + 1), loss, color="red", linewidth=2)
axes[0, 1].set_xscale("log")
axes[0, 1].set_xlabel("Epoch (log scale)")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("Training Loss")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: L2 Norm (Raw + Moving Averages)
# fast_ma/slow_ma are plotted against epoch_grid (log-uniform), NOT
# range(1, num_epochs+1), because they were computed on the log-uniform grid.
if has_ma_data:
    axes[1, 0].plot(range(1, num_epochs + 1), l2_norm, color="lightgreen", linewidth=1, alpha=0.5, label="Raw")
    axes[1, 0].plot(epoch_grid, fast_ma, color="orange", linewidth=2, label="Fast MA (w=50)")
    axes[1, 0].plot(epoch_grid, slow_ma, color="purple", linewidth=2.5, label="Slow MA (w=200)")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Epoch (log scale)")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].set_title("L2 Norm: Raw + Moving Averages (Crossover Strategy)")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].plot(range(1, num_epochs + 1), l2_norm, color="green", linewidth=2)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Epoch (log scale)")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].set_title("Model Weight L2 Norm")
    axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Train-Test Accuracy Gap
gap = np.array(train_acc) - np.array(test_acc)
axes[1, 1].plot(range(1, num_epochs + 1), gap, color="purple", linewidth=2)
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("Epoch (log scale)")
axes[1, 1].set_ylabel("Train Acc - Test Acc")
axes[1, 1].set_title("Generalization Gap")
axes[1, 1].grid(True, alpha=0.3)

# Plot 5 & 6: MA Crossover Details (if data available)
# Both plotted against epoch_grid — already log-uniform, no extra smoothing needed.
if has_ma_data:
    # Plot 5: Fast vs Slow MA
    ax5 = axes[2, 0]
    ax5.plot(epoch_grid, fast_ma, color="orange", linewidth=2.5, label="Fast MA (w=50)")
    ax5.plot(epoch_grid, slow_ma, color="purple", linewidth=2.5, label="Slow MA (w=200)")
    ax5.set_xscale("log")
    ax5.set_xlabel("Epoch (log scale)")
    ax5.set_ylabel("L2 Norm")
    ax5.set_title("Moving Average Crossover (Full View)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Difference (Fast MA - Slow MA) to visualize crossover
    ax6 = axes[2, 1]
    ma_diff = fast_ma - slow_ma

    ax6.plot(epoch_grid, ma_diff, color="darkred", linewidth=2.5)
    ax6.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax6.fill_between(epoch_grid, 0, ma_diff, where=(ma_diff > 0), alpha=0.2, color="green")
    ax6.fill_between(epoch_grid, 0, ma_diff, where=(ma_diff <= 0), alpha=0.2, color="red")
    ax6.set_xscale("log")
    ax6.set_xlabel("Epoch (log scale)")
    ax6.set_ylabel("Fast MA - Slow MA")
    ax6.set_title("MA Difference (Crossover at Zero)")
    ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grokking_analysis.png", dpi=150, bbox_inches="tight")
print("[OK] Plot saved to grokking_analysis.png")
print(f"  - Epochs trained: {num_epochs}")
print(f"  - Final train accuracy: {train_acc[-1]:.4f}")
print(f"  - Final test accuracy: {test_acc[-1]:.4f}")
print(f"  - Final loss: {loss[-1]:.6f}")
print(f"  - Final L2 norm: {l2_norm[-1]:.4f}")
