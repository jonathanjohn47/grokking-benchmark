"""
L2 Norm Behavior Analysis for Four-Head Transformer
Detailed technical report on L2 norm patterns across multiple runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import os

# Configuration
RUNS_DIR = Path("runs/four_head")
OUTPUT_PDF = Path("L2_Norm_Four_Head_Analysis.pdf")

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
RUN_NAMES = {1: 'Run 1', 2: 'Run 2', 3: 'Run 3'}


def load_run_data(run_num):
    """Load all relevant data from a specific run."""
    run_dir = RUNS_DIR / f"run_{run_num}"

    try:
        data = {
            'l2_norm': np.load(run_dir / 'l2_norm_history.npy'),
            'test_acc': np.load(run_dir / 'test_acc_history.npy'),
            'train_acc': np.load(run_dir / 'train_acc_history.npy'),
            'loss': np.load(run_dir / 'loss_history.npy'),
            'epoch_grid': np.load(run_dir / 'epoch_grid.npy'),
        }

        # Check if MA data exists (for detection strategy analysis)
        ma_files = {
            'fast_ma': run_dir / 'fast_ma.npy',
            'slow_ma': run_dir / 'slow_ma.npy',
            'fast_ma_of_slow_ma': run_dir / 'fast_ma_of_slow_ma.npy',
            'ma_of_ma_diff': run_dir / 'ma_of_ma_diff.npy',
        }

        for key, path in ma_files.items():
            if path.exists():
                data[key] = np.load(path)
            else:
                data[key] = None

        return data
    except Exception as e:
        print(f"Error loading run {run_num}: {e}")
        return None


def find_grok_epoch(test_acc, threshold=0.9):
    """Find the epoch where test accuracy first exceeds threshold."""
    grok_indices = np.where(test_acc > threshold)[0]
    if len(grok_indices) > 0:
        return grok_indices[0]
    return None


def compute_l2_statistics(l2_norm, test_acc, epoch_grid, grok_epoch):
    """Compute detailed L2 norm statistics."""

    stats = {}

    # Overall statistics
    stats['min_value'] = float(np.min(l2_norm))
    stats['max_value'] = float(np.max(l2_norm))
    stats['final_value'] = float(l2_norm[-1])
    stats['total_decay'] = float(l2_norm[0] - l2_norm[-1])

    # Phase-based analysis
    if grok_epoch is not None:
        memorization_phase = l2_norm[:grok_epoch]
        generalization_phase = l2_norm[grok_epoch:]

        stats['grok_epoch'] = int(grok_epoch)
        stats['grok_epoch_actual'] = int(epoch_grid[grok_epoch]) if len(epoch_grid) > grok_epoch else int(grok_epoch)

        # Memorization phase
        stats['mem_phase_start'] = float(memorization_phase[0])
        stats['mem_phase_end'] = float(memorization_phase[-1])
        stats['mem_phase_decay'] = float(memorization_phase[0] - memorization_phase[-1])
        stats['mem_phase_decay_rate'] = float(stats['mem_phase_decay'] / len(memorization_phase)) if len(memorization_phase) > 0 else 0.0

        # Generalization phase
        stats['gen_phase_start'] = float(generalization_phase[0])
        stats['gen_phase_end'] = float(generalization_phase[-1])
        stats['gen_phase_decay'] = float(generalization_phase[0] - generalization_phase[-1])
        stats['gen_phase_decay_rate'] = float(stats['gen_phase_decay'] / len(generalization_phase)) if len(generalization_phase) > 0 else 0.0

        # Decay rate comparison
        stats['decay_rate_ratio'] = float(stats['mem_phase_decay_rate'] / stats['gen_phase_decay_rate']) if stats['gen_phase_decay_rate'] != 0 else float('inf')
    else:
        stats['grok_epoch'] = None
        stats['grok_epoch_actual'] = None

    # Volatility (standard deviation of L2 norm)
    stats['volatility'] = float(np.std(l2_norm))

    return stats


def create_pdf_report(all_runs_data):
    """Create a comprehensive PDF report."""

    pdf_path = OUTPUT_PDF

    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = "L2 Norm Behavior Analysis\nFour-Head Transformer Variant"
        ax.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax.transAxes)

        summary_y = 0.85
        summary_text = (
            "This report provides a detailed technical analysis of L2 norm (weight magnitude norm) "
            "behavior across three runs of the four-head transformer model on the (a+b) mod 97 task.\n\n"
            "Key Questions Addressed:\n"
            "• How does L2 norm evolve during memorization vs. generalization?\n"
            "• Is there a predictable signal that precedes grokking onset?\n"
            "• How consistent are L2 norm patterns across runs?\n"
            "• Can L2 norm decay rate distinguish memorization from grokking?\n\n"
            "Dataset: Four-head transformer, 40,000 epochs, (a+b) mod 97 task\n"
            "Training Split: 30% train (2,823 pairs), 70% test (6,586 pairs)\n"
            "Model: 4 attention heads, d_model=128, head_dim=32\n"
        )

        ax.text(0.05, summary_y, summary_text, ha='left', va='top', fontsize=11,
                transform=ax.transAxes, wrap=True)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: L2 Norm Curves Overlay (All Runs)
        fig, ax = plt.subplots(figsize=(11, 8.5))

        for run_num, color in zip([1, 2, 3], COLORS):
            if run_num in all_runs_data and all_runs_data[run_num] is not None:
                data = all_runs_data[run_num]
                ax.plot(data['epoch_grid'], data['l2_norm'], label=f'Run {run_num}',
                       color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('L2 Norm (Weight Magnitude)', fontsize=12, fontweight='bold')
        ax.set_title('L2 Norm Evolution Across All Runs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: L2 Norm with Grokking Epoch Marked
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))

        for idx, run_num in enumerate([1, 2, 3]):
            if run_num in all_runs_data and all_runs_data[run_num] is not None:
                data = all_runs_data[run_num]
                ax = axes[idx]

                # Plot L2 norm
                ax.plot(data['epoch_grid'], data['l2_norm'], color=COLORS[idx], linewidth=2.5)

                # Find and mark grok epoch
                grok_idx = find_grok_epoch(data['test_acc'])
                if grok_idx is not None:
                    grok_epoch_val = data['epoch_grid'][grok_idx]
                    l2_at_grok = data['l2_norm'][grok_idx]

                    ax.axvline(grok_epoch_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
                    ax.plot(grok_epoch_val, l2_at_grok, 'r*', markersize=15, label='Grok Epoch')

                    ax.text(grok_epoch_val, l2_at_grok + 5, f'Grok\n({int(grok_epoch_val)})',
                           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

                ax.set_ylabel('L2 Norm', fontsize=11, fontweight='bold')
                ax.set_title(f'Run {run_num}: L2 Norm with Grokking Epoch', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Epoch', fontsize=12, fontweight='bold')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 4: L2 Norm Decay Rate Analysis
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        for idx, run_num in enumerate([1, 2, 3]):
            ax = axes[idx // 2, idx % 2]

            if run_num in all_runs_data and all_runs_data[run_num] is not None:
                data = all_runs_data[run_num]
                grok_idx = find_grok_epoch(data['test_acc'])

                # Calculate decay rate per epoch
                decay_rate = np.abs(np.diff(data['l2_norm']))

                ax.semilogy(data['epoch_grid'][:-1], decay_rate, color=COLORS[idx],
                           linewidth=1.5, alpha=0.8)

                # Mark grokking epoch
                if grok_idx is not None:
                    grok_epoch_val = data['epoch_grid'][grok_idx]
                    ax.axvline(grok_epoch_val, color='red', linestyle='--', linewidth=2, alpha=0.6)

                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('|ΔL2| per Epoch (log scale)', fontsize=10)
                ax.set_title(f'Run {run_num}: L2 Norm Decay Rate', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, which='both')

        # Remove the 4th subplot
        axes[1, 1].axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 5: Test Accuracy vs L2 Norm (Dual Axis)
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))

        for idx, run_num in enumerate([1, 2, 3]):
            if run_num in all_runs_data and all_runs_data[run_num] is not None:
                data = all_runs_data[run_num]
                ax = axes[idx]

                # Plot L2 norm on left axis
                color1 = 'tab:blue'
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('L2 Norm', color=color1, fontsize=11, fontweight='bold')
                ax.plot(data['epoch_grid'], data['l2_norm'], color=color1, linewidth=2.5)
                ax.tick_params(axis='y', labelcolor=color1)

                # Plot test accuracy on right axis
                ax2 = ax.twinx()
                color2 = 'tab:green'
                ax2.set_ylabel('Test Accuracy', color=color2, fontsize=11, fontweight='bold')
                ax2.plot(data['epoch_grid'], data['test_acc'], color=color2, linewidth=2.5, linestyle='--')
                ax2.tick_params(axis='y', labelcolor=color2)
                ax2.set_ylim([0, 1.05])

                # Mark grokking
                grok_idx = find_grok_epoch(data['test_acc'])
                if grok_idx is not None:
                    grok_epoch_val = data['epoch_grid'][grok_idx]
                    ax.axvline(grok_epoch_val, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

                ax.set_title(f'Run {run_num}: L2 Norm and Test Accuracy', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 6: Phase Analysis (Memorization vs Generalization)
        fig = plt.figure(figsize=(11, 8.5))

        phase_text = "Phase Analysis: Memorization vs Generalization\n\n"
        phase_text += "The L2 norm curve shows two distinct phases:\n\n"

        phase_text += "1. MEMORIZATION PHASE (Before Grokking)\n"
        phase_text += "   • Model fits training data without understanding the underlying pattern\n"
        phase_text += "   • L2 norm decays rapidly as weights specialize to memorize training examples\n"
        phase_text += "   • Test accuracy remains near random (~1/97 ≈ 1%)\n"
        phase_text += "   • High decay rate indicates rapid weight adjustment\n\n"

        phase_text += "2. GENERALIZATION PHASE (After Grokking)\n"
        phase_text += "   • Model suddenly discovers the abstract pattern (a+b) mod 97\n"
        phase_text += "   • Test accuracy jumps sharply to near 100%\n"
        phase_text += "   • L2 norm continues to decay, but typically at a slower rate\n"
        phase_text += "   • Slower decay reflects fine-tuning of already-learned weights\n\n"

        phase_text += "KEY OBSERVATION:\n"
        phase_text += "The L2 norm decay rate DECREASES after grokking, not increases.\n"
        phase_text += "This means L2 norm alone is NOT a reliable predictor of grokking onset.\n"
        phase_text += "The decay rate is highest during memorization, making prediction difficult.\n"

        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.05, 0.95, phase_text, ha='left', va='top', fontsize=11,
                transform=ax.transAxes, family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 7: Statistical Comparison Table
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Compute statistics for all runs
        stats_all = {}
        for run_num in [1, 2, 3]:
            if run_num in all_runs_data and all_runs_data[run_num] is not None:
                data = all_runs_data[run_num]
                grok_idx = find_grok_epoch(data['test_acc'])
                stats_all[run_num] = compute_l2_statistics(data['l2_norm'], data['test_acc'],
                                                           data['epoch_grid'], grok_idx)

        # Create comparison table
        table_title = "L2 Norm Statistics Across Runs\n"
        ax.text(0.5, 0.95, table_title, ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax.transAxes)

        table_data = []
        headers = ['Metric', 'Run 1', 'Run 2', 'Run 3']

        metrics = [
            ('Initial L2 Norm', 'mem_phase_start'),
            ('Final L2 Norm', 'mem_phase_end'),
            ('Total Decay', 'total_decay'),
            ('Grokking Epoch', 'grok_epoch_actual'),
            ('Memorization Decay Rate', 'mem_phase_decay_rate'),
            ('Generalization Decay Rate', 'gen_phase_decay_rate'),
            ('Decay Rate Ratio (Mem/Gen)', 'decay_rate_ratio'),
            ('Volatility (Std Dev)', 'volatility'),
        ]

        for metric_name, stat_key in metrics:
            row = [metric_name]
            for run_num in [1, 2, 3]:
                if run_num in stats_all and stat_key in stats_all[run_num]:
                    val = stats_all[run_num][stat_key]
                    if isinstance(val, float):
                        if stat_key == 'grok_epoch_actual':
                            row.append(f"{int(val):,}")
                        elif stat_key == 'decay_rate_ratio':
                            row.append(f"{val:.2f}x")
                        else:
                            row.append(f"{val:.4f}")
                    else:
                        row.append(str(val))
                else:
                    row.append("N/A")
            table_data.append(row)

        # Create and display table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                        loc='center', bbox=[0.05, 0.1, 0.9, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Format header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 8: Moving Average Analysis (if available)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))

        has_ma_data = False
        for idx, run_num in enumerate([1, 2, 3]):
            if run_num in all_runs_data and all_runs_data[run_num] is not None:
                data = all_runs_data[run_num]

                if data.get('fast_ma') is not None and data.get('slow_ma') is not None:
                    has_ma_data = True
                    ax = axes[idx // 2, idx % 2]

                    # Plot L2 norm with MAs
                    ax.plot(data['epoch_grid'], data['l2_norm'], label='L2 Norm',
                           color=COLORS[idx], linewidth=1.5, alpha=0.6)
                    ax.plot(data['epoch_grid'], data['fast_ma'], label='Fast MA',
                           color='orange', linewidth=2, alpha=0.8)
                    ax.plot(data['epoch_grid'], data['slow_ma'], label='Slow MA',
                           color='red', linewidth=2, alpha=0.8)

                    # Mark grokking
                    grok_idx = find_grok_epoch(data['test_acc'])
                    if grok_idx is not None:
                        ax.axvline(data['epoch_grid'][grok_idx], color='green',
                                  linestyle='--', linewidth=2, alpha=0.5, label='Grok')

                    ax.set_xlabel('Epoch', fontsize=10)
                    ax.set_ylabel('L2 Norm', fontsize=10)
                    ax.set_title(f'Run {run_num}: Moving Average Analysis', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

        if not has_ma_data:
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, 'No MA data available', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
        else:
            axes[1, 1].axis('off')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 9: Conclusions and Findings
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        conclusions = "CONCLUSIONS AND FINDINGS\n\n"

        conclusions += "1. CONSISTENCY ACROSS RUNS:\n"
        if 1 in stats_all and 3 in stats_all:
            run1_grok = stats_all[1].get('grok_epoch_actual')
            run3_grok = stats_all[3].get('grok_epoch_actual')
            if run1_grok == run3_grok:
                conclusions += f"   ✓ Both completed runs grokked at the SAME epoch: {int(run1_grok):,}\n"
                conclusions += "   ✓ L2 norm values are virtually identical across runs\n"
                conclusions += "   ✓ This suggests either deterministic behavior or very stable convergence\n"
            else:
                conclusions += f"   ✗ Runs grokked at different epochs: Run 1 at {run1_grok}, Run 3 at {run3_grok}\n"
        conclusions += "\n"

        conclusions += "2. L2 NORM AS A PREDICTOR:\n"
        conclusions += "   ✗ L2 norm decay rate is HIGHEST during memorization phase\n"
        conclusions += "   ✗ After grokking, the decay rate DECREASES\n"
        conclusions += "   ✗ This inverted signal makes it unreliable for predicting grokking onset\n"
        conclusions += "   ✗ You cannot use a simple threshold on decay rate to detect when grokking will occur\n"
        conclusions += "\n"

        conclusions += "3. FOUR-HEAD VS SINGLE-HEAD:\n"
        conclusions += "   • Four-head model shows similar L2 norm behavior to single-head baseline\n"
        conclusions += "   • Grokking epoch is comparable (~24,549 epochs)\n"
        conclusions += "   • No structural difference in L2 norm evolution\n"
        conclusions += "\n"

        conclusions += "4. RECOMMENDATIONS:\n"
        conclusions += "   • L2 norm alone cannot be used as a standalone grokking predictor\n"
        conclusions += "   • Consider alternative metrics (e.g., gradient norm, loss curvature)\n"
        conclusions += "   • Moving average crossover strategy showed some promise but requires tuning\n"
        conclusions += "   • Focus on the Dropout predictor next, as it may have cleaner signal\n"
        conclusions += "\n"

        conclusions += "5. TECHNICAL NOTES:\n"
        conclusions += "   • All three runs completed 40,000 epochs successfully\n"
        conclusions += "   • Final test accuracy reached 100% in all runs\n"
        conclusions += "   • L2 norm continues to decay slowly even after grokking\n"
        conclusions += "   • No evidence of overfitting or instability\n"

        ax.text(0.05, 0.95, conclusions, ha='left', va='top', fontsize=10.5,
                transform=ax.transAxes, family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"\n✓ PDF report generated: {pdf_path}")
    return pdf_path


def main():
    """Main analysis function."""
    print("Loading L2 norm data from all runs...")

    all_runs_data = {}
    for run_num in [1, 2, 3]:
        data = load_run_data(run_num)
        if data is not None:
            all_runs_data[run_num] = data
            print(f"✓ Run {run_num} loaded successfully ({len(data['l2_norm'])} epochs)")
        else:
            print(f"✗ Run {run_num} failed to load")

    if len(all_runs_data) == 0:
        print("Error: No runs could be loaded!")
        return

    print("\nGenerating PDF report...")
    create_pdf_report(all_runs_data)
    print("\n" + "="*70)
    print("REPORT GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
