import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Look for files in project root (parent of src/), same pattern as plot_results.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Known plateau value from the original (no-LayerNorm) model, recorded in
# context.md (Aug 7, 2026 plateau-investigation session): 6557/6587 correct.
ORIGINAL_PLATEAU_ACC = 0.995446

shadow_train_acc = np.load("shadow_ln_train_acc_history.npy")
shadow_test_acc = np.load("shadow_ln_test_acc_history.npy")
num_epochs = len(shadow_train_acc)
epochs_axis = range(1, num_epochs + 1)

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(epochs_axis, shadow_train_acc, color="steelblue", linewidth=2.5, label="Shadow (LayerNorm) Train Accuracy")
ax.plot(epochs_axis, shadow_test_acc, color="seagreen", linewidth=2.5, label="Shadow (LayerNorm) Test Accuracy")

# Reference line: the original no-LayerNorm model's known plateau value.
ax.axhline(y=ORIGINAL_PLATEAU_ACC, color="darkred", linestyle="--", linewidth=2,
           label=f"Original plateau (no LayerNorm): {ORIGINAL_PLATEAU_ACC:.4%}")

# If the original run's test_acc_history.npy is present in the project root,
# overlay it too, so both curves can be compared directly on one graph.
# This is optional — the shadow curve above still plots fine without it.
try:
    original_test_acc = np.load("test_acc_history.npy")
    original_epochs_axis = range(1, len(original_test_acc) + 1)
    ax.plot(original_epochs_axis, original_test_acc, color="darkorange", linewidth=2,
            linestyle=":", label="Original (no LayerNorm) Test Accuracy", alpha=0.85)
except FileNotFoundError:
    print("Note: test_acc_history.npy (original run) not found — skipping overlay, "
          "shadow curve plotted alone.")

ax.set_xscale("log")
ax.set_xlabel("Epoch (log scale)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.set_title("Shadow Model (with LayerNorm) — Grokking Curve vs. Original Plateau", fontsize=13)
ax.legend(loc="center right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("shadow_ln_grokking_curve.png", dpi=150, bbox_inches="tight")
print("[OK] Plot saved to shadow_ln_grokking_curve.png")

final_shadow_test_acc = shadow_test_acc[-1]
print(f"\nFinal shadow (LayerNorm) test accuracy: {final_shadow_test_acc:.6f}")
print(f"Original (no LayerNorm) plateau:         {ORIGINAL_PLATEAU_ACC:.6f}")
if final_shadow_test_acc > ORIGINAL_PLATEAU_ACC:
    print("Shadow model beat the original plateau.")
elif final_shadow_test_acc == ORIGINAL_PLATEAU_ACC:
    print("Shadow model matched the original plateau exactly.")
else:
    print("Shadow model did not reach the original plateau.")
