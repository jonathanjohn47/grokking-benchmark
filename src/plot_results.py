import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from predictors.l2_norm import compute_noise_floor, detect_ma_of_ma_zero_crossing

# Set up results directory paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_base = os.path.join(project_root, "results", "single_head")
training_dir = os.path.join(results_base, "training")
l2_norm_dir = os.path.join(results_base, "l2_norm")

# Load data already computed and saved by train.py — no re-run needed.
epoch_grid = np.load(os.path.join(l2_norm_dir, "epoch_grid.npy"))
slow_ma = np.load(os.path.join(l2_norm_dir, "slow_ma.npy"))
fast_ma_of_slow_ma = np.load(os.path.join(l2_norm_dir, "fast_ma_of_slow_ma.npy"))
diff = np.load(os.path.join(l2_norm_dir, "ma_of_ma_diff.npy"))
train_acc = np.load(os.path.join(training_dir, "train_acc_history.npy"))
test_acc = np.load(os.path.join(training_dir, "test_acc_history.npy"))
num_epochs = len(train_acc)

SKIP_EPOCHS = 100
QUIET_EPOCH_CUTOFF = 90

# Noise floor printed for context only — the trigger itself is a plain zero
# crossing now, not a magnitude threshold.
noise_floor = compute_noise_floor(diff, epoch_grid, quiet_epoch_cutoff=QUIET_EPOCH_CUTOFF)
trigger_epoch = detect_ma_of_ma_zero_crossing(epoch_grid, diff, skip_epochs=SKIP_EPOCHS)

# Plot 1: Slow MA vs. Fast MA of Slow MA, with the trigger point marked
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(epoch_grid, slow_ma, color="purple", linewidth=3.5, label="Slow MA (w=200)")
ax.plot(epoch_grid, fast_ma_of_slow_ma, color="orange", linewidth=3.5, label="Fast MA of Slow MA (w=20)")

if trigger_epoch is not None:
    trigger_idx = np.argmin(np.abs(epoch_grid - trigger_epoch))
    ax.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2)
    ax.scatter(trigger_epoch, slow_ma[trigger_idx], color="red", zorder=5, s=150, marker="*",
               label=f"Trigger (epoch {trigger_epoch:.0f})")

ax.set_xscale("log")
ax.set_xlabel("Epoch (log scale)", fontsize=12)
ax.set_ylabel("L2 Norm", fontsize=12)
ax.set_title("Slow MA vs. Fast MA of Slow MA — Zero-Crossing Trigger", fontsize=13)
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(l2_norm_dir, "ma_of_slow_ma_crossover.png"), dpi=150, bbox_inches="tight")
print("[OK] Plot saved to ma_of_slow_ma_crossover.png")

# Plot 2: The difference curve itself, with the zero-crossing trigger marked
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.plot(epoch_grid, diff, color="darkred", linewidth=3.5)
ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
ax2.fill_between(epoch_grid, 0, diff, where=(diff > 0), alpha=0.2, color="green")
ax2.fill_between(epoch_grid, 0, diff, where=(diff <= 0), alpha=0.2, color="red")

if trigger_epoch is not None:
    trigger_idx = np.argmin(np.abs(epoch_grid - trigger_epoch))
    ax2.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2)
    ax2.scatter(trigger_epoch, diff[trigger_idx], color="red", zorder=5, s=150, marker="*",
                label=f"Trigger (epoch {trigger_epoch:.0f})")

ax2.set_xscale("log")
ax2.set_xlabel("Epoch (log scale)", fontsize=12)
ax2.set_ylabel("Fast MA of Slow MA − Slow MA", fontsize=12)
ax2.set_title("Difference: Fast MA of Slow MA minus Slow MA — Zero-Crossing Trigger", fontsize=13)
ax2.legend(loc="upper right", fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(l2_norm_dir, "ma_of_slow_ma_diff.png"), dpi=150, bbox_inches="tight")
print("[OK] Plot saved to ma_of_slow_ma_diff.png")

# Plot 3: Same difference curve as Plot 2, but on a LINEAR x-axis instead of log.
fig3, ax3 = plt.subplots(figsize=(12, 7))

ax3.plot(epoch_grid, diff, color="darkred", linewidth=3.5)
ax3.axhline(y=0, color="black", linestyle="--", linewidth=1)
ax3.fill_between(epoch_grid, 0, diff, where=(diff > 0), alpha=0.2, color="green")
ax3.fill_between(epoch_grid, 0, diff, where=(diff <= 0), alpha=0.2, color="red")

if trigger_epoch is not None:
    trigger_idx = np.argmin(np.abs(epoch_grid - trigger_epoch))
    ax3.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2)
    ax3.scatter(trigger_epoch, diff[trigger_idx], color="red", zorder=5, s=150, marker="*",
                label=f"Trigger (epoch {trigger_epoch:.0f})")

ax3.set_xlabel("Epoch (linear scale)", fontsize=12)
ax3.set_ylabel("Fast MA of Slow MA − Slow MA", fontsize=12)
ax3.set_title("Difference: Fast MA of Slow MA minus Slow MA — Linear Scale", fontsize=13)
ax3.legend(loc="upper right", fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(l2_norm_dir, "ma_of_slow_ma_diff_linear.png"), dpi=150, bbox_inches="tight")
print("[OK] Plot saved to ma_of_slow_ma_diff_linear.png")

# Plot 4: Difference curve (linear x-axis) overlaid on the grokking curve
# (train/test accuracy). Different scales (accuracy 0-1 vs. diff ~0.00-0.02),
# so accuracy uses the left y-axis and the diff uses a twin right y-axis.
grok_epoch = int(np.argmax(test_acc > 0.9))

fig4, ax4 = plt.subplots(figsize=(12, 7))

ax4.plot(range(1, num_epochs + 1), train_acc, color="steelblue", linewidth=2.5, label="Train Accuracy")
ax4.plot(range(1, num_epochs + 1), test_acc, color="seagreen", linewidth=2.5, label="Test Accuracy")
ax4.set_xlabel("Epoch (linear scale)", fontsize=12)
ax4.set_ylabel("Accuracy", fontsize=12)
ax4.set_ylim(-0.05, 1.05)

ax4b = ax4.twinx()
ax4b.plot(epoch_grid, diff, color="darkred", linewidth=3, label="Fast MA of Slow MA − Slow MA", alpha=0.85)
ax4b.axhline(y=0, color="black", linestyle="--", linewidth=1)
ax4b.set_ylabel("MA-of-MA Difference", fontsize=12)

ax4.axvline(x=grok_epoch, color="seagreen", linestyle=":", linewidth=2, alpha=0.7)
if trigger_epoch is not None:
    ax4.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2)
    ax4b.scatter(trigger_epoch, diff[np.argmin(np.abs(epoch_grid - trigger_epoch))],
                 color="red", zorder=5, s=150, marker="*", label=f"Trigger (epoch {trigger_epoch:.0f})")

ax4.set_title(f"MA-of-MA Difference vs. Grokking Curve (Linear Scale) — Grok epoch {grok_epoch}", fontsize=13)
lines4, labels4 = ax4.get_legend_handles_labels()
lines4b, labels4b = ax4b.get_legend_handles_labels()
ax4.legend(lines4 + lines4b, labels4 + labels4b, loc="center right", fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(l2_norm_dir, "ma_of_ma_diff_vs_grokking_linear.png"), dpi=150, bbox_inches="tight")
print("[OK] Plot saved to ma_of_ma_diff_vs_grokking_linear.png")

print(f"\nNoise floor (context only): {noise_floor:.6f}")
if trigger_epoch is not None:
    print(f"Trigger epoch (first zero-crossing): {trigger_epoch:.1f}")
else:
    print("No zero-crossing detected")
