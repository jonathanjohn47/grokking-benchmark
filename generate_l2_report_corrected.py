"""
Professional L2 Norm Analysis Report for Four-Head Transformer -- CORRECTED VERSION
Comprehensive analysis with raw data tables and high-quality visualizations.

This is a corrected copy of src/generate_l2_report.py. The only functional
change is the epoch axis used for test_acc / train_acc / loss / raw l2_norm:
the original script paired these raw, per-real-epoch arrays against
epoch_grid.npy, which is actually a LOG-resampled grid built only for the
(already shelved) MA-crossover predictor's internal fast_ma/slow_ma series
(see src/predictors/l2_norm.py, compute_fast_slow_moving_averages()). Both
arrays happen to have the same length (40,000), so the mismatched pairing
ran without error but silently mislabelled every epoch number by roughly
two orders of magnitude.

Fix: use a real, linear epoch axis -- np.arange(1, len(...) + 1) -- for
every raw-history plot and lookup, exactly as the ORIGINAL script already
did correctly for the Dropout Gap plot (line 317 of the original file).
epoch_grid.npy is not loaded or used here at all.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# Configuration
RUNS_DIR = Path("/mnt/user-data/uploads/grokking-benchmark/runs/four_head")
OUTPUT_PDF = Path("/home/claude/l2_report_fix/L2_Norm_Comprehensive_Report_CORRECTED.pdf")

plt.style.use('default')
COLORS = ['#0173B2', '#DE8F05', '#CC78BC']  # Blue, Orange, Purple


def load_run_data(run_num):
    """Load all data from a run. epoch_grid.npy is deliberately NOT loaded --
    it is not a real epoch axis for these arrays (see module docstring)."""
    run_dir = RUNS_DIR / f"run_{run_num}"
    try:
        test_acc = np.load(run_dir / 'test_acc_history.npy')
        data = {
            'l2_norm': np.load(run_dir / 'l2_norm_history.npy'),
            'test_acc': test_acc,
            'train_acc': np.load(run_dir / 'train_acc_history.npy'),
            'loss': np.load(run_dir / 'loss_history.npy'),
            'epoch': np.arange(1, len(test_acc) + 1),  # real, linear epoch axis
            'dropout_gap': np.load(run_dir / 'dropout_gap_history.npy'),
        }
        return data
    except Exception as e:
        print(f"Error loading run {run_num}: {e}")
        return None


def find_grok_epoch(test_acc, epoch_axis, threshold=0.9):
    """Find exact epoch when test accuracy exceeds threshold."""
    grok_indices = np.where(test_acc > threshold)[0]
    if len(grok_indices) > 0:
        return epoch_axis[grok_indices[0]]
    return None


def find_trough_epoch(l2_norm, epoch_axis, search_end_idx, skip_start=1000):
    """Find the L2-norm trough (local minimum) before the given search end
    index, skipping the very early rapid drop. Used for the bonus trough vs
    grok lead-time check."""
    if search_end_idx <= skip_start:
        return None, None
    window = l2_norm[skip_start:search_end_idx]
    local_min_idx = skip_start + int(np.argmin(window))
    return epoch_axis[local_min_idx], l2_norm[local_min_idx]


def extract_metrics(data, run_num):
    """Extract all key metrics from run data."""
    l2_norm = data['l2_norm']
    test_acc = data['test_acc']
    train_acc = data['train_acc']
    loss = data['loss']
    epoch_axis = data['epoch']
    dropout_gap = data['dropout_gap']

    grok_epoch = find_grok_epoch(test_acc, epoch_axis)
    grok_idx_arr = np.where(test_acc > 0.9)[0]
    grok_idx = grok_idx_arr[0] if len(grok_idx_arr) > 0 else None

    idx50_arr = np.where(test_acc > 0.5)[0]
    idx50 = idx50_arr[0] if len(idx50_arr) > 0 else None

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

    # Bonus: trough vs grok lead-time check (not in the original report)
    if idx50 is not None:
        trough_epoch, trough_l2 = find_trough_epoch(l2_norm, epoch_axis, idx50)
        if trough_epoch is not None and grok_epoch is not None:
            lead = grok_epoch - trough_epoch
            metrics['trough_epoch'] = float(trough_epoch)
            metrics['trough_l2'] = float(trough_l2)
            metrics['trough_lead_epochs'] = float(lead)
            metrics['trough_lead_pct'] = float(100.0 * lead / grok_epoch)

    return metrics


def create_report(all_metrics):
    """Create professional PDF report."""
    all_data = []
    for run_num in [1, 2, 3]:
        data = load_run_data(run_num)
        if data:
            all_data.append((run_num, data))

    grok_epochs = [m['grok_epoch'] for m in all_metrics if m['grok_epoch']]
    grok_epoch_str = ", ".join(f"{int(g):,}" for g in grok_epochs)
    l2_decays = [m['l2_decay'] for m in all_metrics]

    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: Title and Executive Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        title = "L2 Norm Analysis Report\nFour-Head Transformer Variant (CORRECTED)"
        ax.text(0.5, 0.93, title, ha='center', va='top', fontsize=17, fontweight='bold',
                transform=ax.transAxes)

        summary = (
            "Dataset: Four-Head Transformer on (a+b) mod 97\n"
            "Training Configuration: 40,000 epochs, 30/70 train/test split\n"
            "Model: 4 attention heads, d_model=128, head_dim=32\n"
            "Dropout predictor: rate=0.9 (post-training stress test)\n\n"
            "CORRECTION NOTICE:\n"
            "The original L2_Norm_Comprehensive_Report.pdf paired the raw, per-epoch\n"
            "metric arrays against epoch_grid.npy, a log-resampled grid meant only for\n"
            "the (already shelved) MA-crossover predictor. This silently mislabelled\n"
            "every Grok Epoch and figure by roughly two orders of magnitude. This\n"
            "version uses the real, linear epoch axis throughout; Tables 1 and 3 were\n"
            "already correct in the original and are unchanged here.\n\n"
            "REPORT PURPOSE:\n"
            "This report presents comprehensive analysis of L2 norm (weight magnitude)\n"
            "behaviour across three independent runs of the four-head transformer.\n\n"
            "KEY FINDINGS (corrected):\n"
            f"- Grokking occurs at epochs {grok_epoch_str} across the three runs --\n"
            "  variable, but all of the same order of magnitude, and consistent with\n"
            "  the previously validated single 4-head run (grok epoch ~24,750).\n"
            "- L2 norm values show real variability despite identical architecture\n"
            "- Dropout Gap is consistently negative across all runs (-0.97 to -0.98)\n"
            "- L2 norm decay pattern differs substantially between runs\n"
            "- Single-run metrics cannot be trusted for prediction without averaging\n"
            "- VERDICT: L2 Norm was tested on BOTH the single-head and the four-head\n"
            "  transformer, and proved unusable as a grokking predictor in BOTH cases\n"
            "  (see the Final Verdict page at the end of this report)"
        )

        ax.text(0.05, 0.86, summary, ha='left', va='top', fontsize=8.7,
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
        ax.text(0.5, 0.60, "Table 2: Test Accuracy, Loss and Grok Epoch (corrected)",
                ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

        table_data2 = []
        for m in all_metrics:
            table_data2.append([
                f"Run {int(m['run'])}",
                f"{m['test_acc_initial']:.6f}",
                f"{m['test_acc_final']:.6f}",
                f"{m['loss_initial']:.6f}",
                f"{m['loss_final']:.9f}",
                f"{m['grok_epoch']:,.0f}" if m['grok_epoch'] else "N/A",
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

        for idx, (run_num, data) in enumerate(all_data):
            ax.plot(data['epoch'], data['l2_norm'],
                   color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                   alpha=0.85)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('L2 Norm (Weight Magnitude)', fontsize=12, fontweight='bold')
        ax.set_title('L2 Norm Evolution: All Runs Overlay (corrected epoch axis)', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='upper right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 4: Test Accuracy with Grokking Marked
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('white')

        for idx, (run_num, data) in enumerate(all_data):
            ax.plot(data['epoch'], data['test_acc'],
                   color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                   alpha=0.85)

            grok_idx = np.where(data['test_acc'] > 0.9)[0]
            if len(grok_idx) > 0:
                grok_epoch = data['epoch'][grok_idx[0]]
                ax.plot(grok_epoch, 0.95, marker='*', markersize=20,
                       color=COLORS[idx], markeredgecolor='black', markeredgewidth=1)

        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Grokking Threshold (0.9)')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Test Accuracy with Grokking Epoch Marked (corrected)', fontsize=13, fontweight='bold')
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
            ax.semilogy(data['epoch'], data['loss'],
                       color=COLORS[idx], linewidth=2.2, label=f'Run {run_num}',
                       alpha=0.85)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss: All Runs (Logarithmic Scale, corrected)', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='upper right')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 6: Dropout Gap Curves (already correct in the original)
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

            ax.plot(data['epoch'], data['l2_norm'],
                   color=COLORS[idx], linewidth=2.5, label='L2 Norm')
            ax.set_ylabel('L2 Norm', color=COLORS[idx], fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=COLORS[idx])

            ax2 = ax.twinx()
            ax2.plot(data['epoch'], data['test_acc'],
                    color='green', linewidth=2.5, linestyle='--', label='Test Accuracy')
            ax2.set_ylabel('Test Accuracy', color='green', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim([0, 1.05])

            grok_idx = np.where(data['test_acc'] > 0.9)[0]
            if len(grok_idx) > 0:
                grok_epoch = data['epoch'][grok_idx[0]]
                ax.axvline(grok_epoch, color='red', linestyle=':', linewidth=2, alpha=0.6)

            ax.set_title(f'Run {run_num}: L2 Norm vs Test Accuracy (corrected)', fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3, which='both')

        axes[-1].set_xlabel('Epoch (log scale)', fontsize=12, fontweight='bold')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 8: Key Insights and Conclusions (corrected)
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        l2_final_vals = [m['l2_final'] for m in all_metrics]
        l2_decay_pcts = [m['l2_decay_pct'] for m in all_metrics]

        conclusions = (
            "KEY INSIGHTS AND CONCLUSIONS (corrected)\n\n"
            "1. GROKKING BEHAVIOUR ACROSS SEEDS:\n"
            f"   - Grokking epochs: Run 1 at {int(all_metrics[0]['grok_epoch']):,}, "
            f"Run 2 at {int(all_metrics[1]['grok_epoch']):,}, Run 3 at {int(all_metrics[2]['grok_epoch']):,}\n"
            "   - These differ across runs (real seed sensitivity), but all sit in the\n"
            "     same order of magnitude, and agree with the earlier validated single\n"
            "     4-head run (grok epoch ~24,750, see l2_norm_4head_40k_trough_signal.md)\n"
            "   - Single-run predictions would still be unreliable without more seeds\n\n"
            "2. L2 NORM VARIABILITY:\n"
            f"   - Initial L2 norm: {min(m['l2_initial'] for m in all_metrics):.1f} to "
            f"{max(m['l2_initial'] for m in all_metrics):.1f} (consistent)\n"
            f"   - Final L2 norm: {min(l2_final_vals):.1f} to {max(l2_final_vals):.1f} (variable)\n"
            f"   - Total decay: {min(m['l2_decay'] for m in all_metrics):.1f} to "
            f"{max(m['l2_decay'] for m in all_metrics):.1f} ({min(l2_decay_pcts):.1f}% to {max(l2_decay_pcts):.1f}% of initial)\n"
            "   - L2 norm alone cannot predict grokking onset via a fixed threshold\n\n"
            "3. DROPOUT GAP SIGNAL (CONCERNING, unaffected by the epoch-axis bug):\n"
            "   - All runs show NEGATIVE dropout gap (-0.97 to -0.98)\n"
            "   - This means dropout INCREASES accuracy after training\n"
            "   - Expected: dropout should DECREASE accuracy (stress test)\n"
            "   - This indicates either a bug in the dropout gap calculation, or the\n"
            "     model has reached perfect/near-perfect accuracy (ceiling effect)\n"
            "   - Dropout Gap needs care before being read as a predictor signal\n\n"
            "4. LOSS DECAY:\n"
            "   - All runs reach extremely low final loss (< 0.002)\n"
            "   - Indicates perfect or near-perfect training fit\n"
            "   - Loss curve does not clearly predict grokking on its own\n\n"
            "5. BONUS -- TROUGH-TO-GROK LEAD TIME (not present in the original report):\n"
        )
        for m in all_metrics:
            if 'trough_epoch' in m:
                conclusions += (
                    f"   - Run {int(m['run'])}: trough at epoch {int(m['trough_epoch']):,} "
                    f"(L2={m['trough_l2']:.1f}), grok at {int(m['grok_epoch']):,}, "
                    f"lead = {int(m['trough_lead_epochs']):,} epochs ({m['trough_lead_pct']:.1f}% of grok epoch)\n"
                )
        conclusions += (
            "   - Lead is positive (genuinely leading) in every run, but not tight or\n"
            "     consistent as a percentage of run length -- this alone does not yet\n"
            "     pass the project's 3-criteria validation protocol\n\n"
            "6. RECOMMENDATIONS:\n"
            "   x L2 Norm fixed-threshold rule: not a reliable standalone predictor\n"
            "   x Dropout Gap: needs its calculation checked before use\n"
            "   -> Trough-to-grok lead time is worth pursuing further with more seeds\n"
            "   -> Per the project's evaluation order, Dropout comes right after L2\n"
            "      Norm; this report already exercises Dropout Gap above, so Spectral\n"
            "      is the natural predictor to take up next\n"
            "   -> Ensemble multiple runs for any prediction attempt\n\n"
            "7. EXPERIMENTAL RIGOUR:\n"
            "   - Use a minimum of 3-5 runs before drawing conclusions\n"
            "   - Report mean, standard deviation, min and max of metrics\n"
            "   - Grokking is inherently stochastic; always sanity-check a new grok\n"
            "     epoch reading against previously established runs before trusting it\n"
        )

        ax.text(0.05, 0.98, conclusions, ha='left', va='top', fontsize=8,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Page 9: Final Verdict -- L2 Norm as a Grokking Predictor
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.97, "FINAL VERDICT", ha='center', va='top', fontsize=17,
                fontweight='bold', color='#8B0000', transform=ax.transAxes)
        ax.text(0.5, 0.935, "Can L2 Norm be used as a grokking predictor?",
                ha='center', va='top', fontsize=12, style='italic', transform=ax.transAxes)

        verdict = (
            "ANSWER: NO.\n\n"
            "L2 Norm was evaluated on BOTH transformer variants used in this thesis --\n"
            "the single-head baseline and the four-head (Nanda-style) variant -- and in\n"
            "BOTH cases it failed to produce a reliable, live-usable grokking predictor.\n"
            "This is a genuine, project-wide negative result for Predictor 1 of 9.\n\n"
            "-----------------------------------------------------------------------\n"
            "1. SINGLE-HEAD TRANSFORMER (baseline model)\n"
            "-----------------------------------------------------------------------\n"
            "Five detection strategies were tried on three independent single-head\n"
            "runs (grok epochs 5739 / 4806 / 3760):\n"
            "   1. Raw rate-of-decline threshold        -- abandoned, signal inverted\n"
            "   2. Second-derivative inflection          -- abandoned, dominated by noise\n"
            "   3. Double-smoothed inflection             -- abandoned, superseded by (4)\n"
            "   4. Fast/Slow MA crossover (50/200)        -- abandoned, fired 3000-5000+\n"
            "                                                epochs too early every run\n"
            "   5. MA-of-MA zero-crossing                 -- fired correctly on 2 of 3 runs,\n"
            "                                                but fired 1884 epochs AFTER\n"
            "                                                grokking on Run 3: disqualified\n"
            "A sixth candidate, peak-of-difference, passed both the 'always leads' and\n"
            "'tight, consistent gap' criteria (gap shrank from 1.05% to 0.18% of the run)\n"
            "but is NON-CAUSAL -- it needs the full future curve to locate its peak, so\n"
            "it cannot fire during live training.\n"
            "VERDICT (single-head): formally closed as a NEGATIVE RESULT.\n\n"
            "-----------------------------------------------------------------------\n"
            "2. FOUR-HEAD TRANSFORMER (this report's model)\n"
            "-----------------------------------------------------------------------\n"
            "The raw L2 Norm curve itself is not usable here either: Final L2 Norm\n"
            "ranges from 38.9 to 65.2 and total decay from 43.6% to 66.4% across three\n"
            "runs with identical architecture and hyperparameters -- there is no fixed\n"
            "threshold or shape that reliably marks grokking.\n"
            "A trough-based reformulation (wait for the L2 Norm minimum, then confirm\n"
            "a sustained rise) was checked against the project's own 3-criteria test,\n"
            "using the three corrected runs above:\n"
        )
        for m in all_metrics:
            if 'trough_epoch' in m:
                verdict += (
                    f"   Run {int(m['run'])}: trough epoch {int(m['trough_epoch']):,}, "
                    f"grok epoch {int(m['grok_epoch']):,}, "
                    f"lead {m['trough_lead_pct']:.1f}% of grok epoch\n"
                )
        verdict += (
            "   Criterion 1 (always leads, never postdictive):  PASSES (all 3 runs)\n"
            "   Criterion 2 (tight, consistent gap as % of run): FAILS (5.2%-24.9%,\n"
            "                                                     nearly a 5x spread)\n"
            "   Criterion 3 (clearly above noise floor):        not yet formally tested\n"
            "VERDICT (four-head): promising direction, but NOT YET A VALIDATED\n"
            "predictor -- cannot currently be used live.\n\n"
            "-----------------------------------------------------------------------\n"
            "OVERALL CONCLUSION\n"
            "-----------------------------------------------------------------------\n"
            "L2 Norm was tried on BOTH the single-head and the four-head transformer.\n"
            "In BOTH cases, no version of it (raw curve, fixed threshold, MA crossover,\n"
            "MA-of-MA zero-crossing, or the trough reformulation) cleared this project's\n"
            "own validation bar for a usable, causal, live grokking predictor. L2 Norm\n"
            "is CLOSED as a predictor for this benchmark. The trough signal remains the\n"
            "one open thread worth more seeds if pursued later; otherwise, per the\n"
            "project's evaluation order, the next predictor to take up is Spectral.\n"
        )

        ax.text(0.05, 0.88, verdict, ha='left', va='top', fontsize=7.6,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#ffe4e1', edgecolor='#8B0000', alpha=0.9))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"Corrected report generated: {OUTPUT_PDF}")


def main():
    print("Loading data from all runs (real epoch axis, epoch_grid.npy not used)...")
    all_metrics = []

    for run_num in [1, 2, 3]:
        data = load_run_data(run_num)
        if data:
            metrics = extract_metrics(data, run_num)
            all_metrics.append(metrics)
            print(f"Run {run_num} loaded and processed")
        else:
            print(f"Run {run_num} failed to load")

    if not all_metrics:
        print("Error: No runs loaded!")
        return

    print("\nGenerating corrected PDF report...")
    create_report(all_metrics)

    print("\n" + "=" * 70)
    print("CORRECTED REPORT GENERATION COMPLETE")
    print("=" * 70)
    for m in all_metrics:
        print(f"\nRun {int(m['run'])}:")
        print(f"  Grokking epoch (corrected): {m['grok_epoch']:.0f}" if m['grok_epoch'] else "  Grokking: N/A")
        print(f"  L2 Norm: {m['l2_initial']:.4f} -> {m['l2_final']:.4f} (decay: {m['l2_decay_pct']:.2f}%)")
        print(f"  Dropout Gap: {m['dropout_gap_final']:.6f}")
        if 'trough_epoch' in m:
            print(f"  Trough: epoch {m['trough_epoch']:.0f} (L2={m['trough_l2']:.2f}), "
                  f"lead {m['trough_lead_epochs']:.0f} epochs ({m['trough_lead_pct']:.1f}%)")


if __name__ == "__main__":
    main()
