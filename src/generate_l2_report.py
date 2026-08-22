"""
Professional L2 Norm Analysis Report for Four-Head Transformer
Comprehensive analysis with raw data tables and high-quality visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd

# Configuration
RUNS_DIR = Path("runs/four_head")
OUTPUT_PDF = Path("L2_Norm_Comprehensive_Report.pdf")

plt.style.use('default')
COLORS = ['#0173B2', '#DE8F05', '#CC78BC']  # Blue, Orange, Purple


def load_run_data(run_num):
    """Load all data from a run."""
    run_dir = RUNS_DIR / f"run_{run_num}"
    try:
        data = {
            'l2_norm': np.load(run_dir / 'l2_norm_history.npy'),
            'test_acc': np.load(run_dir / 'test_acc_history.npy'),
            'train_acc': np.load(run_dir / 'train_acc_history.npy'),
            'loss': np.load(run_dir / 'loss_history.npy'),
            'epoch_grid': np.load(run_dir / 'epoch_grid.npy'),
            'dropout_gap': np.load(run_dir / 'dropout_gap_history.npy'),
        }
        return data
    except Exception as e:
        print(f"Error loading run {run_num}: {e}")
        return None


def find_grok_epoch(test_acc, epoch_grid, threshold=0.9):
    """Find exact epoch when test accuracy exceeds threshold."""
    grok_indices = np.where(test_acc > threshold)[0]
    if len(grok_indices) > 0:
        return epoch_grid[grok_indices[0]]
    return None


def extract_metrics(data, run_num):
    """Extract all key metrics from run data."""
    l2_norm = data['l2_norm']
    test_acc = data['test_acc']
    train_acc = data['train_acc']
    loss = data['loss']
    epoch_grid = data['epoch_grid']
    dropout_gap = data['dropout_gap']

    grok_epoch = find_grok_epoch(test_acc, epoch_grid)
    grok_idx = np.where(test_acc > 0.9)[0][0] if len(np.where(test_acc > 0.9)[0]) > 0 else None

    metrics = {
        'run': run_num,
        'total_epochs': len(test_acc),
        'l2_initial': float(l2_norm[0]),
        'l2_final': float(l2_norm[-1]),
        'l2_min': float(np.min(l2_norm)),
        'l2_max': float(np.max(l2_norm)),
        'l2_decay': float(l2_norm[0] - l2_norm[-1]),
        'l2_decay_pct': float((l2_norm[0] - l2_norm[-1]) / l2_norm[0] * 100),
        'test_acc_initial': float(test_acc[0]),
        'test_acc_final': float(test_acc[-1]),
        'train_acc_initial': float(train_acc[0]),
        'train_acc_final': float(train_acc[-1]),
        'loss_initial': float(loss[0]),
        'loss_final': float(loss[-1]),
        'grok_epoch': float(grok_epoch) if grok_epoch is not None else None,
        'dropout_gap_final': float(dropout_gap[-1]),
        'dropout_gap_min': float(np.min(dropout_gap)),
        'dropout_gap_max': float(np.max(dropout_gap)),
    }

    # Add phase analysis if grokked
    if grok_idx is not None:
        mem_phase = l2_norm[:grok_idx]
        gen_phase = l2_norm[grok_idx:]

        metrics['mem_phase_length'] = len(mem_phase)
        metrics['gen_phase_length'] = len(gen_phase)
        metrics['l2_mem_start'] = float(mem_phase[0])
        metrics['l2_mem_end'] = float(mem_phase[-1])
        metrics['l2_mem_decay'] = float(mem_phase[0] - mem_phase[-1])
        metrics['l2_gen_start'] = float(gen_phase[0])
        metrics['l2_gen_end'] = float(gen_phase[-1])
        metrics['l2_gen_decay'] = float(gen_phase[0] - gen_phase[-1])

    return metrics


def create_report(all_metrics):
    """Create professional PDF report."""
    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Title and Executive Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        title = "L2 Norm Analysis Report\nFour-Head Transformer Variant"
        ax.text(0.5, 0.93, title, ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax.transAxes)

        summary = (
            "Dataset: Four-Head Transformer on (a+b) mod 97\n"
            "Training Configuration: 40,000 epochs, 30/70 train/test split\n"
            "Model: 4 attention heads, d_model=128, head_dim=32\n"
            "Dropout predictor: rate=0.9 (post-training stress test)\n\n"
            "REPORT PURPOSE:\n"
            "This report presents comprehensive analysis of L2 norm (weight magnitude)\n"
            "behavior across three independent runs of the four-head transformer.\n\n"
            "KEY FINDINGS:\n"
            "• Grokking occurs at significantly different epochs across runs (149, 2815, 671)\n"
            "• L2 norm values show high variability despite identical architecture\n"
            "• Dropout Gap is consistently negative across all runs (-0.97 to -0.98)\n"
            "• L2 norm decay pattern differs substantially between runs\n"
            "• Single-run metrics cannot be trusted for prediction without averaging"
        )

        ax.text(0.05, 0.85, summary, ha='left', va='top', fontsize=9.5,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 2: Raw Metrics Table
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.97, "Table 1: Raw Metrics Across All Runs",
                ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)

        table_data = []
        for m in all_metrics:
            table_data.append([
                f"Run {int(m['run'])}",
                f"{int(m['total_epochs']):,}",
                f"{m['l2_initial']:.4f}",
                f"{m['l2_final']:.4f}",
                f"{m['l2_decay']:.4f}",
                f"{m['l2_decay_pct']:.2f}%",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Run', 'Total Epochs', 'Initial L2', 'Final L2', 'Total Decay', 'Decay %'],
            cellLoc='center',
            loc='upper center',
            bbox=[0.05, 0.65, 0.9, 0.28]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        for i in range(6):
            table[(0, i)].set_facecolor('#0173B2')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, 4):
            for j in range(6):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        # Accuracy and Loss table
        ax.text(0.5, 0.60, "Table 2: Test Accuracy and Loss",
                ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

        table_data2 = []
        for m in all_metrics:
            table_data2.append([
                f"Run {int(m['run'])}",
                f"{m['test_acc_initial']:.6f}",
                f"{m['test_acc_final']:.6f}",
                f"{m['loss_initial']:.6f}",
                f"{m['loss_final']:.9f}",
                f"{m['grok_epoch']:.0f}" if m['grok_epoch'] else "N/A",
            ])

        table2 = ax.table(
            cellText=table_data2,
            colLabels=['Run', 'Initial Test Acc', 'Final Test Acc', 'Initial Loss', 'Final Loss', 'Grok Epoch'],
            cellLoc='center',
            loc='center',
            bbox=[0.05, 0.30, 0.9, 0.28]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2.5)

        for i in range(6):
            table2[(0, i)].set_facecolor('#DE8F05')
            table2[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, 4):
            for j in range(6):
                table2[(i, j)].set_facecolor('#f5f5f5' if i % 2 == 0 else 'white')

        # Dropout Gap table
        ax.text(0.5, 0.25, "Table 3: Dropout Gap (Stress Test Results)",
                ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

        table_data3 = []
        for m in all_metrics:
            table_data3.append([
                f"Run {int(m['run'])}",
                f"{m['dropout_gap_min']:.6f}",
                f"{m['dropout_gap_final']:.6f}",
                f"{m['dropout_gap_max']:.6f}",
            ])

        table3 = ax.table(
            cellText=table_data3,
            colLabels=['Run', 'Min Dropout Gap', 'Final Dropout Gap', 'Max Dropout Gap'],
            cellLoc='center',
            loc='lower center',
            bbox=[0.05, 0.02, 0.9, 0.20]
        )
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 2.0)

        for i in range(4):
            table3[(0, i)].set_facecolor('#CC78BC')
            table3[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, 4):
            for j in range(4):
                table3[(i, j)].set_facecolor('#f5f5f5' if i % 2 == 0 else 'white')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 3: L2 Norm Curves Overlay
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('white')

        all_data = []
        for run_num in [1, 2, 3]:
            data = load_run_data(run_num)
            if data:
                all_data.append((run_num, data))

        for idx, (run_num, data) in enumerate(all_data):
            ax.plot(data['epoch_grid'], data['l2_norm'],
                   color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                   alpha=0.85)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('L2 Norm (Weight Magnitude)', fontsize=12, fontweight='bold')
        ax.set_title('L2 Norm Evolution: All Runs Overlay', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='upper right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 4: Test Accuracy with Grokking Marked
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('white')

        for idx, (run_num, data) in enumerate(all_data):
            ax.plot(data['epoch_grid'], data['test_acc'],
                   color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                   alpha=0.85)

            # Mark grokking epoch
            grok_idx = np.where(data['test_acc'] > 0.9)[0]
            if len(grok_idx) > 0:
                grok_epoch = data['epoch_grid'][grok_idx[0]]
                ax.plot(grok_epoch, 0.95, marker='*', markersize=20,
                       color=COLORS[idx], markeredgecolor='black', markeredgewidth=1)

        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Grokking Threshold (0.9)')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Test Accuracy with Grokking Epoch Marked (★)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='lower right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 5: Loss Curves
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('white')

        for idx, (run_num, data) in enumerate(all_data):
            ax.semilogy(data['epoch_grid'], data['loss'],
                       color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                       alpha=0.85)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss: All Runs (Logarithmic Scale)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='upper right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 6: Dropout Gap Curves
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('white')

        for idx, (run_num, data) in enumerate(all_data):
            ax.plot(np.arange(1, len(data['dropout_gap']) + 1), data['dropout_gap'],
                   color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                   alpha=0.85)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dropout Gap', fontsize=12, fontweight='bold')
        ax.set_title('Dropout Gap (Post-Training Stress Test at rate=0.9)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='lower right')

        note = "NOTE: Negative gap indicates dropout INCREASES accuracy (unexpected).\nThis suggests a bug in the dropout gap calculation or model has reached perfect accuracy."
        ax.text(0.5, -0.15, note, ha='center', fontsize=9, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 7: Dual Axis - L2 Norm vs Test Accuracy
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))
        fig.patch.set_facecolor('white')

        for idx, (run_num, data) in enumerate(all_data):
            ax = axes[idx]

            # L2 norm on left
            ax.plot(data['epoch_grid'], data['l2_norm'],
                   color=COLORS[idx], linewidth=2.5, label='L2 Norm')
            ax.set_ylabel('L2 Norm', color=COLORS[idx], fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=COLORS[idx])

            # Test accuracy on right
            ax2 = ax.twinx()
            ax2.plot(data['epoch_grid'], data['test_acc'],
                    color='green', linewidth=2.5, linestyle='--', label='Test Accuracy')
            ax2.set_ylabel('Test Accuracy', color='green', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim([0, 1.05])

            # Mark grokking
            grok_idx = np.where(data['test_acc'] > 0.9)[0]
            if len(grok_idx) > 0:
                grok_epoch = data['epoch_grid'][grok_idx[0]]
                ax.axvline(grok_epoch, color='red', linestyle=':', linewidth=2, alpha=0.6)

            ax.set_title(f'Run {run_num}: L2 Norm vs Test Accuracy', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3, which='both')

        axes[-1].set_xlabel('Epoch (log scale)', fontsize=12, fontweight='bold')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 8: Key Insights and Conclusions
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        conclusions = (
            "KEY INSIGHTS AND CONCLUSIONS\n\n"
            "1. STOCHASTIC GROKKING BEHAVIOR:\n"
            "   • Grokking epochs vary dramatically: Run 1 at 149, Run 2 at 2,815, Run 3 at 671\n"
            "   • This is NOT consistent despite identical architecture and hyperparameters\n"
            "   • Suggests random seed strongly influences grokking timing\n"
            "   • Single-run predictions would be unreliable\n\n"
            "2. L2 NORM VARIABILITY:\n"
            "   • Initial L2 norm: 115.7 to 116.4 (consistent)\n"
            "   • Final L2 norm: 39.2 to 65.2 (highly variable!)\n"
            "   • Total decay: 50.6 to 77.5 (different decay rates)\n"
            "   • L2 norm alone cannot predict grokking onset reliably\n\n"
            "3. DROPOUT GAP SIGNAL (CONCERNING):\n"
            "   • All runs show NEGATIVE dropout gap (-0.97 to -0.98)\n"
            "   • This means dropout INCREASES accuracy after training\n"
            "   • Expected: dropout should DECREASE accuracy (stress test)\n"
            "   • This indicates either:\n"
            "     - Bug in dropout gap calculation, OR\n"
            "     - Model has reached perfect accuracy (ceiling effect)\n"
            "   • Dropout predictor is NOT currently usable\n\n"
            "4. LOSS DECAY:\n"
            "   • All runs reach extremely low final loss (< 0.001)\n"
            "   • Indicates perfect or near-perfect training fit\n"
            "   • Loss curve does not clearly predict grokking\n\n"
            "5. RECOMMENDATIONS:\n"
            "   ✗ L2 Norm: Not a reliable standalone predictor\n"
            "   ✗ Dropout Gap: Requires debugging before use\n"
            "   → Need to fix dropout gap calculation\n"
            "   → Consider next predictor in evaluation order (Spectral)\n"
            "   → Ensemble multiple runs for any prediction attempt\n\n"
            "6. EXPERIMENTAL RIGOR:\n"
            "   • Use minimum 3-5 runs before drawing conclusions\n"
            "   • Report mean, std, min, max of metrics\n"
            "   • Consider that grokking is inherently stochastic\n"
            "   • Average L2 norm behavior across seeds\n"
        )

        ax.text(0.05, 0.97, conclusions, ha='left', va='top', fontsize=9,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"✓ Professional report generated: {OUTPUT_PDF}")


def main():
    print("Loading data from all runs...")
    all_metrics = []

    for run_num in [1, 2, 3]:
        data = load_run_data(run_num)
        if data:
            metrics = extract_metrics(data, run_num)
            all_metrics.append(metrics)
            print(f"✓ Run {run_num} loaded and processed")
        else:
            print(f"✗ Run {run_num} failed to load")

    if not all_metrics:
        print("Error: No runs loaded!")
        return

    print(f"\nGenerating comprehensive PDF report...")
    create_report(all_metrics)

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nSummary Statistics:")
    for m in all_metrics:
        print(f"\nRun {int(m['run'])}:")
        print(f"  Grokking epoch: {m['grok_epoch']:.0f}" if m['grok_epoch'] else "  Grokking: N/A")
        print(f"  L2 Norm: {m['l2_initial']:.4f} → {m['l2_final']:.4f} (decay: {m['l2_decay_pct']:.2f}%)")
        print(f"  Dropout Gap: {m['dropout_gap_final']:.6f}")


if __name__ == "__main__":
    main()
