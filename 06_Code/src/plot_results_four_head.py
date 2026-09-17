import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re

from predictors.l2_norm import compute_noise_floor, detect_ma_of_ma_zero_crossing, detect_ma_crossover

# ======================================================================
# 4-HEAD VARIANT of plot_results.py — now MULTI-RUN aware.
#
# train_four_head.py saves every run into its own numbered subfolder,
# results/four_head/run_1/, results/four_head/run_2/, and so on (see that
# file's RUN NUMBERING comment for why — grokking is stochastic, so a
# real comparison needs several independent runs kept side by side, not
# one run's numbers overwritten by the next).
#
# This script does two things:
#   1. For every run_<N> folder it finds, regenerates that run's own 8
#      plots (same as before), saved inside that run's own folder.
#   2. Builds 4 NEW comparison plots at the results/four_head/ level,
#      overlaying every run's grokking curve / loss curve / L2 norm
#      curve / Dropout Gap curve together, so all previous runs can be
#      read on one chart instead of opening each run's folder one by one.
# ======================================================================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
four_head_dir = os.path.join(project_root, "runs", "four_head")

SKIP_EPOCHS = 100
QUIET_EPOCH_CUTOFF = 90

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

# Distinct colors cycled across runs, so Run 1 / Run 2 / Run 3 ... always
# look different from each other on the comparison plots.
RUN_COLORS = ["steelblue", "darkorange", "seagreen", "crimson", "purple",
              "goldenrod", "teal", "indianred", "slategray", "mediumvioletred"]


def migrate_legacy_flat_run(base_dir):
    """Same migration as train_four_head.py — kept here too so this
    script can be re-run on its own and still pick up an old flat run
    even if train_four_head.py has not been re-run since the update."""
    run_1_dir = os.path.join(base_dir, "run_1")
    if os.path.isdir(run_1_dir):
        return
    legacy_present = any(
        os.path.isfile(os.path.join(base_dir, name)) for name in LEGACY_FILENAMES
    )
    if not legacy_present:
        return
    os.makedirs(run_1_dir, exist_ok=True)
    for name in LEGACY_FILENAMES:
        src = os.path.join(base_dir, name)
        if os.path.isfile(src):
            os.rename(src, os.path.join(run_1_dir, name))
    print("[MIGRATION] Found results saved directly in results/four_head/ by an earlier "
          "version of these scripts — moved them into results/four_head/run_1/ so they "
          "are counted as Run 1.")


def discover_run_dirs(base_dir):
    """Finds every run_<N> folder inside base_dir, sorted by run number
    (not alphabetically, so run_2 comes before run_10)."""
    if not os.path.isdir(base_dir):
        return []
    found = []
    for name in os.listdir(base_dir):
        match = re.fullmatch(r"run_(\d+)", name)
        if match and os.path.isdir(os.path.join(base_dir, name)):
            found.append((int(match.group(1)), os.path.join(base_dir, name)))
    found.sort(key=lambda pair: pair[0])
    return found


def load_run_data(run_dir):
    # unified_measurements.PredictorMeasurements writes into per-predictor
    # subfolders: training/, l2_norm/, dropout/. Each entry below is
    # (key, relative_path).
    required_files = [
        ("train_acc", "training/train_acc_history.npy"), ("test_acc", "training/test_acc_history.npy"),
        ("loss", "training/loss_history.npy"), ("l2_norm", "l2_norm/l2_norm_history.npy"),
        ("epoch_grid", "l2_norm/epoch_grid.npy"), ("fast_ma", "l2_norm/fast_ma.npy"),
        ("slow_ma", "l2_norm/slow_ma.npy"), ("fast_ma_of_slow_ma", "l2_norm/fast_ma_of_slow_ma.npy"),
        ("ma_of_ma_diff", "l2_norm/ma_of_ma_diff.npy"),
        ("dropout_gap_epochs", "dropout/dropout_gap_epochs.npy"),
        ("dropout_gap_by_rate", "dropout/dropout_gap_by_rate.npy"),
        ("dropout_rates", "dropout/dropout_rates.npy"),
    ]

    for _, rel in required_files:
        if not os.path.exists(os.path.join(run_dir, rel)):
            return None

    data = {}
    for key, rel in required_files:
        data[key] = np.load(os.path.join(run_dir, rel))
    data["num_epochs"] = len(data["train_acc"])
    return data


def plot_single_run(run_number, run_dir, data):
    """Recreates this run's own 8 plots, saved inside results/four_head/run_<N>/ itself."""
    epoch_grid = data["epoch_grid"]
    fast_ma = data["fast_ma"]
    slow_ma = data["slow_ma"]
    fast_ma_of_slow_ma = data["fast_ma_of_slow_ma"]
    diff = data["ma_of_ma_diff"]
    train_acc = data["train_acc"]
    test_acc = data["test_acc"]
    loss_history = data["loss"]
    l2_norm_history = data["l2_norm"]
    dropout_gap_epochs = data["dropout_gap_epochs"]
    dropout_gap_by_rate = data["dropout_gap_by_rate"]   # shape [rate, epoch]
    dropout_rates = data["dropout_rates"]
    num_epochs = data["num_epochs"]
    epochs_axis = range(1, num_epochs + 1)

    noise_floor = compute_noise_floor(diff, epoch_grid, quiet_epoch_cutoff=QUIET_EPOCH_CUTOFF)
    trigger_epoch = detect_ma_of_ma_zero_crossing(epoch_grid, diff, skip_epochs=SKIP_EPOCHS)
    detection_epoch = detect_ma_crossover(epoch_grid, fast_ma, slow_ma, skip_epochs=SKIP_EPOCHS)
    grokked = bool((test_acc > 0.9).any())
    grok_epoch = int(np.argmax(test_acc > 0.9)) if grokked else None

    def save(fig, filename):
        path = os.path.join(run_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Run {run_number}: plot saved to results/four_head/run_{run_number}/{filename}")

    # Plot 1: Slow MA vs. Fast MA of Slow MA
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
    ax.set_title(f"4-head model, Run {run_number}: Slow MA vs. Fast MA of Slow MA", fontsize=13)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "ma_of_slow_ma_crossover.png")

    # Plot 2: Difference curve, log-x
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
    ax2.set_title(f"4-head model, Run {run_number}: Difference — Fast MA of Slow MA minus Slow MA", fontsize=13)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig2, "ma_of_slow_ma_diff.png")

    # Plot 3: Same difference curve, linear x-axis
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
    ax3.set_title(f"4-head model, Run {run_number}: Difference — Linear Scale", fontsize=13)
    ax3.legend(loc="upper right", fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig3, "ma_of_slow_ma_diff_linear.png")

    # Plot 4: Difference curve overlaid on the grokking curve
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
    if grokked:
        ax4.axvline(x=grok_epoch, color="seagreen", linestyle=":", linewidth=2, alpha=0.7)
    if trigger_epoch is not None:
        ax4.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2)
        ax4b.scatter(trigger_epoch, diff[np.argmin(np.abs(epoch_grid - trigger_epoch))],
                     color="red", zorder=5, s=150, marker="*", label=f"Trigger (epoch {trigger_epoch:.0f})")
    title_suffix = f"Grok epoch {grok_epoch}" if grokked else "did not grok in this run"
    ax4.set_title(f"4-head model, Run {run_number}: MA-of-MA Difference vs. Grokking Curve — {title_suffix}", fontsize=13)
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines4b, labels4b = ax4b.get_legend_handles_labels()
    ax4.legend(lines4 + lines4b, labels4 + labels4b, loc="center right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig4, "ma_of_ma_diff_vs_grokking_linear.png")

    # Plot 5: Grokking curve
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    ax5.plot(epochs_axis, train_acc, label="Train Accuracy", color="steelblue", linewidth=2)
    ax5.plot(epochs_axis, test_acc, label="Test Accuracy", color="seagreen", linewidth=2)
    ax5.set_xscale("log")
    ax5.set_xlabel("Epoch (log scale)")
    ax5.set_ylabel("Accuracy")
    ax5.set_title(f"4-head model, Run {run_number}: Grokking Curve — Train vs. Test Accuracy")
    ax5.legend(loc="center right")
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig5, "grokking_curve.png")

    # Plot 6: Loss curve
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    ax6.plot(epochs_axis, loss_history, color="darkorange", linewidth=2)
    ax6.set_xscale("log")
    ax6.set_xlabel("Epoch (log scale)")
    ax6.set_ylabel("Loss")
    ax6.set_title(f"4-head model, Run {run_number}: Training Loss")
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig6, "loss_curve.png")

    # Plot 7: L2 Norm curve, with MA Crossover marked
    fig7, ax7 = plt.subplots(figsize=(12, 7))
    ax7.plot(epochs_axis, l2_norm_history, color="purple", linewidth=2, label="L2 Norm")
    if detection_epoch is not None:
        ax7.axvline(x=detection_epoch, color="red", linestyle="--", linewidth=2,
                    label=f"MA Crossover (epoch {detection_epoch:.0f})")
    ax7.set_xscale("log")
    ax7.set_xlabel("Epoch (log scale)")
    ax7.set_ylabel("L2 Norm")
    ax7.set_title(f"4-head model, Run {run_number}: L2 Norm of Model Weights")
    ax7.legend(loc="upper right")
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig7, "l2_norm_curve.png")

    # Plot 8: Dropout Gap — full multi-rate sweep
    fig8, ax8 = plt.subplots(figsize=(12, 7))
    for r_idx, rate in enumerate(dropout_rates):
        ax8.plot(dropout_gap_epochs, dropout_gap_by_rate[r_idx], linewidth=2,
                 label=f"p={rate:g}")
    ax8.axhline(y=0, color="black", linewidth=0.6, alpha=0.5)
    ax8.set_xscale("log")
    ax8.set_xlabel("Epoch (log scale)")
    ax8.set_ylabel("Dropout Gap")
    ax8.set_title(f"4-head model, Run {run_number}: Dropout Gap — Multi-Rate Sweep")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig8, "dropout_gap_curve.png")

    final_gap_by_rate = {float(rate): float(dropout_gap_by_rate[r_idx][-1])
                         for r_idx, rate in enumerate(dropout_rates)}
    return {
        "run_number": run_number, "num_epochs": num_epochs, "grokked": grokked,
        "grok_epoch": grok_epoch, "final_test_acc": float(test_acc[-1]),
        "final_l2_norm": float(l2_norm_history[-1]),
        "final_dropout_gap_by_rate": final_gap_by_rate,
    }


def plot_comparison(all_runs_data, out_dir):
    """Builds 4 plots overlaying every discovered run together, saved
    directly in results/four_head/ (not inside any single run's folder),
    so all previous runs can be read on one chart."""

    if not all_runs_data:
        print("[SKIP] No complete runs to compare — comparison plots not generated.")
        return

    def color_for(i):
        return RUN_COLORS[i % len(RUN_COLORS)]

    # Comparison Plot A: Test Accuracy (grokking curve) across all runs
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (run_number, data) in enumerate(all_runs_data):
        epochs_axis = range(1, data["num_epochs"] + 1)
        color = color_for(i)
        ax.plot(epochs_axis, data["test_acc"], color=color, linewidth=2.2,
                label=f"Run {run_number} ({data['num_epochs']} epochs)")
        if (data["test_acc"] > 0.9).any():
            grok_epoch = int(np.argmax(data["test_acc"] > 0.9))
            ax.axvline(x=grok_epoch, color=color, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("4-head model: Test Accuracy Across All Runs", fontsize=13)
    ax.legend(loc="center right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_grokking_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Comparison plot saved to results/four_head/comparison_grokking_curve.png")

    # Comparison Plot B: Loss across all runs
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (run_number, data) in enumerate(all_runs_data):
        epochs_axis = range(1, data["num_epochs"] + 1)
        ax.plot(epochs_axis, data["loss"], color=color_for(i), linewidth=2,
                label=f"Run {run_number} ({data['num_epochs']} epochs)")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("4-head model: Training Loss Across All Runs", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Comparison plot saved to results/four_head/comparison_loss_curve.png")

    # Comparison Plot C: L2 Norm across all runs
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (run_number, data) in enumerate(all_runs_data):
        epochs_axis = range(1, data["num_epochs"] + 1)
        ax.plot(epochs_axis, data["l2_norm"], color=color_for(i), linewidth=2,
                label=f"Run {run_number} ({data['num_epochs']} epochs)")
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title("4-head model: L2 Norm of Model Weights Across All Runs", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_l2_norm_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Comparison plot saved to results/four_head/comparison_l2_norm_curve.png")

    # Comparison Plot D: Dropout Gap sweep — one panel per run, all rates
    n = len(all_runs_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, (run_number, data) in enumerate(all_runs_data):
        ax = axes[0][i]
        rates = data["dropout_rates"]
        gap_by_rate = data["dropout_gap_by_rate"]
        for r_idx, rate in enumerate(rates):
            ax.plot(data["dropout_gap_epochs"], gap_by_rate[r_idx], linewidth=1.8,
                    alpha=0.9, label=f"p={rate:g}")
        ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)", fontsize=11)
        ax.set_ylabel("Dropout Gap", fontsize=11)
        ax.set_title(f"Run {run_number} ({data['num_epochs']} epochs)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("4-head model: Dropout Gap Multi-Rate Sweep Across All Runs", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_dropout_gap_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Comparison plot saved to results/four_head/comparison_dropout_gap_curve.png")


migrate_legacy_flat_run(four_head_dir)
run_dirs = discover_run_dirs(four_head_dir)

if not run_dirs:
    print(f"No runs found under results/four_head/. Run train_four_head.py at least once first.")
else:
    print(f"Found {len(run_dirs)} run(s): {', '.join(f'Run {n}' for n, _ in run_dirs)}\n")

    all_runs_data = []
    summaries = []
    for run_number, run_dir in run_dirs:
        data = load_run_data(run_dir)
        if data is None:
            print(f"  Skipping Run {run_number} (incomplete or empty)")
            continue
        all_runs_data.append((run_number, data))
        summaries.append(plot_single_run(run_number, run_dir, data))

    plot_comparison(all_runs_data, four_head_dir)

    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL RUNS")
    print("=" * 60)
    for s in summaries:
        grok_text = f"grokked at epoch {s['grok_epoch']}" if s["grokked"] else "did NOT grok"
        gap_text = ", ".join(f"p{rate:g}={gap:+.4f}"
                             for rate, gap in sorted(s["final_dropout_gap_by_rate"].items()))
        print(f"  Run {s['run_number']} ({s['num_epochs']} epochs): {grok_text}, "
              f"final test acc = {s['final_test_acc']:.4f}, "
              f"final L2 norm = {s['final_l2_norm']:.4f}")
        print(f"      final dropout gap sweep: {gap_text}")

    print(f"\n[OK] Per-run plots saved inside each results/four_head/run_<N>/ folder.")
    print(f"[OK] 4 comparison plots saved directly in results/four_head/:")
    print("  - comparison_grokking_curve.png")
    print("  - comparison_loss_curve.png")
    print("  - comparison_l2_norm_curve.png")
    print("  - comparison_dropout_gap_curve.png")
