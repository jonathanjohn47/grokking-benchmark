import numpy as np
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

l2_norm_history = np.load("l2_norm_history.npy")
print("First 20 L2 norms:", l2_norm_history[:20])
print("Shape:", l2_norm_history.shape)

num_epochs = len(train_acc)

# Create figure with subplots
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

# Plot 3: L2 Norm
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

plt.tight_layout()
plt.savefig("grokking_analysis.png", dpi=150, bbox_inches="tight")
print("✓ Plot saved to grokking_analysis.png")
print(f"  - Epochs trained: {num_epochs}")
print(f"  - Final train accuracy: {train_acc[-1]:.4f}")
print(f"  - Final test accuracy: {test_acc[-1]:.4f}")
print(f"  - Final loss: {loss[-1]:.6f}")
print(f"  - Final L2 norm: {l2_norm[-1]:.4f}")
