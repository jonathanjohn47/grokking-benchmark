"""
L2 Norm + Dropout analysis report for the four-head transformer.

Standalone tool. Reads every runs/four_head/run_<N>/ folder (the
unified_measurements subdir layout: training/, l2_norm/, dropout/) and
builds a multi-page PDF. All figures and tables use COMPUTED numbers from
the loaded runs — nothing about the results is hardcoded here.

Dropout is reported as the full multi-rate sweep. There is no single
"primary" rate.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

RUNS_DIR = Path("runs/four_head")
OUTPUT_PDF = Path("L2_Norm_Comprehensive_Report.pdf")

plt.style.use('default')
COLORS = ['#0173B2', '#DE8F05', '#CC78BC', '#029E73', '#D55E00',
          '#CC79A7', '#56B4E9', '#E69F00']


def discover_run_numbers(base_dir):
    if not base_dir.is_dir():
        return []
    nums = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit():
            if (p / "training" / "test_acc_history.npy").exists():
                nums.append(int(p.name[4:]))
    return sorted(nums)


def load_run_data(run_num):
    """Load one run from the per-predictor subdir layout."""
    run_dir = RUNS_DIR / f"run_{run_num}"
    try:
        data = {
            'l2_norm': np.load(run_dir / 'l2_norm' / 'l2_norm_history.npy'),
            'test_acc': np.load(run_dir / 'training' / 'test_acc_history.npy'),
            'train_acc': np.load(run_dir / 'training' / 'train_acc_history.npy'),
            'loss': np.load(run_dir / 'training' / 'loss_history.npy'),
            'dropout_gap_epochs': np.load(run_dir / 'dropout' / 'dropout_gap_epochs.npy'),
            'dropout_gap_by_rate': np.load(run_dir / 'dropout' / 'dropout_gap_by_rate.npy'),
            'dropout_rates': np.load(run_dir / 'dropout' / 'dropout_rates.npy'),
        }
        return data
    except Exception as e:
        print(f"Error loading run {run_num}: {e}")
        return None


def find_grok_epoch(test_acc, threshold=0.9):
    """First 1-indexed epoch where test accuracy exceeds threshold."""
    idx = np.where(np.asarray(test_acc) > threshold)[0]
    return int(idx[0] + 1) if len(idx) > 0 else None


def extract_metrics(data, run_num):
    l2_norm = data['l2_norm']
    test_acc = data['test_acc']
    train_acc = data['train_acc']
    loss = data['loss']
    rates = [float(r) for r in data['dropout_rates']]
    gap_by_rate = data['dropout_gap_by_rate']  # [rate, epoch]

    metrics = {
        'run': run_num,
        'total_epochs': len(test_acc),
        'l2_initial': float(l2_norm[0]),
        'l2_final': float(l2_norm[-1]),
        'l2_min': float(np.min(l2_norm)),
        'l2_max': float(np.max(l2_norm)),
        'l2_decay': float(l2_norm[0] - l2_norm[-1]),
        'l2_decay_pct': float((l2_norm[0] - l2_norm[-1]) / l2_norm[0] * 100) if l2_norm[0] != 0 else 0.0,
        'test_acc_initial': float(test_acc[0]),
        'test_acc_final': float(test_acc[-1]),
        'train_acc_initial': float(train_acc[0]),
        'train_acc_final': float(train_acc[-1]),
        'loss_initial': float(loss[0]),
        'loss_final': float(loss[-1]),
        'grok_epoch': find_grok_epoch(test_acc),
        'rates': rates,
        'dropout_gap_final_by_rate': {r: float(gap_by_rate[i][-1]) for i, r in enumerate(rates)},
        'dropout_gap_min_by_rate': {r: float(np.min(gap_by_rate[i])) for i, r in enumerate(rates)},
        'dropout_gap_max_by_rate': {r: float(np.max(gap_by_rate[i])) for i, r in enumerate(rates)},
    }
    return metrics


def _style_header(table, ncols, color):
    for i in range(ncols):
        table[(0, i)].set_facecolor(color)
        table[(0, i)].set_text_props(weight='bold', color='white')


def create_report(all_data, all_metrics):
    n = len(all_data)
    rates = all_metrics[0]['rates']

    with PdfPages(OUTPUT_PDF) as pdf:

        # ---- Page 1: Title + computed executive summary ----
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111); ax.axis('off')
        ax.text(0.5, 0.95, "L2 Norm + Dropout Analysis Report\nFour-Head Transformer",
                ha='center', va='top', fontsize=18, fontweight='bold', transform=ax.transAxes)

        grok_list = [m['grok_epoch'] for m in all_metrics]
        grok_known = [g for g in grok_list if g is not None]
        grok_txt = ", ".join(str(g) if g is not None else "none" for g in grok_list)
        l2_finals = [m['l2_final'] for m in all_metrics]
        rate_str = ", ".join(f"{r:g}" for r in rates)

        summary = (
            f"Task: four-head transformer on (a+b) mod 113, 30/70 train/test split  [Nanda-Unified]\n"
            f"Model: 4 attention heads, d_model=128, head_dim=32\n"
            f"Runs analysed: {n}  (run numbers: {[m['run'] for m in all_metrics]})\n"
            f"Dropout predictor: full multi-rate sweep, rates = [{rate_str}]\n\n"
            f"COMPUTED SUMMARY:\n"
            f"- Grokking epoch per run: {grok_txt}\n"
        )
        if len(grok_known) >= 2:
            summary += (f"  mean {np.mean(grok_known):.0f}, std {np.std(grok_known):.0f}, "
                        f"min {min(grok_known)}, max {max(grok_known)}\n")
        summary += (
            f"- Final L2 norm per run: {', '.join(f'{v:.2f}' for v in l2_finals)}\n"
            f"  range {min(l2_finals):.2f} to {max(l2_finals):.2f}\n"
            f"- Final dropout gap by rate (mean over runs):\n"
        )
        for r in rates:
            vals = [m['dropout_gap_final_by_rate'][r] for m in all_metrics]
            summary += f"    p={r:g}: {np.mean(vals):+.4f}\n"

        ax.text(0.05, 0.86, summary, ha='left', va='top', fontsize=9.5,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 2: metric tables ----
        fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111); ax.axis('off')

        ax.text(0.5, 0.98, "Table 1: L2 Norm Metrics", ha='center', fontsize=13,
                fontweight='bold', transform=ax.transAxes)
        t1 = ax.table(
            cellText=[[f"Run {m['run']}", f"{m['total_epochs']:,}", f"{m['l2_initial']:.4f}",
                       f"{m['l2_final']:.4f}", f"{m['l2_decay']:.4f}", f"{m['l2_decay_pct']:.2f}%"]
                      for m in all_metrics],
            colLabels=['Run', 'Epochs', 'Initial L2', 'Final L2', 'Decay', 'Decay %'],
            cellLoc='center', loc='upper center', bbox=[0.05, 0.68, 0.9, 0.26])
        t1.auto_set_font_size(False); t1.set_fontsize(10); t1.scale(1, 2.2)
        _style_header(t1, 6, '#0173B2')

        ax.text(0.5, 0.62, "Table 2: Accuracy & Loss", ha='center', fontsize=13,
                fontweight='bold', transform=ax.transAxes)
        t2 = ax.table(
            cellText=[[f"Run {m['run']}", f"{m['test_acc_final']:.6f}", f"{m['train_acc_final']:.6f}",
                       f"{m['loss_initial']:.6f}", f"{m['loss_final']:.3e}",
                       str(m['grok_epoch']) if m['grok_epoch'] else "N/A"]
                      for m in all_metrics],
            colLabels=['Run', 'Final Test Acc', 'Final Train Acc', 'Initial Loss', 'Final Loss', 'Grok Epoch'],
            cellLoc='center', loc='center', bbox=[0.05, 0.34, 0.9, 0.24])
        t2.auto_set_font_size(False); t2.set_fontsize(10); t2.scale(1, 2.2)
        _style_header(t2, 6, '#DE8F05')

        ax.text(0.5, 0.28, "Table 3: Final Dropout Gap by Rate", ha='center', fontsize=13,
                fontweight='bold', transform=ax.transAxes)
        t3 = ax.table(
            cellText=[[f"Run {m['run']}"] + [f"{m['dropout_gap_final_by_rate'][r]:+.4f}" for r in rates]
                      for m in all_metrics],
            colLabels=['Run'] + [f"p={r:g}" for r in rates],
            cellLoc='center', loc='lower center', bbox=[0.05, 0.02, 0.9, 0.22])
        t3.auto_set_font_size(False); t3.set_fontsize(10); t3.scale(1, 2.0)
        _style_header(t3, 1 + len(rates), '#CC78BC')

        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 3: L2 norm overlay ----
        fig, ax = plt.subplots(figsize=(11, 7)); fig.patch.set_facecolor('white')
        for idx, (run_num, data) in enumerate(all_data):
            x = np.arange(1, len(data['l2_norm']) + 1)
            ax.plot(x, data['l2_norm'], color=COLORS[idx % len(COLORS)], linewidth=2.2,
                    label=f'Run {run_num}', alpha=0.85)
        ax.set_xlabel('Epoch (log scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('L2 Norm', fontsize=12, fontweight='bold')
        ax.set_title('L2 Norm Evolution: All Runs', fontsize=14, fontweight='bold')
        ax.set_xscale('log'); ax.grid(True, alpha=0.3, which='both'); ax.legend(fontsize=11)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 4: test accuracy + grok markers ----
        fig, ax = plt.subplots(figsize=(11, 7)); fig.patch.set_facecolor('white')
        for idx, (run_num, data) in enumerate(all_data):
            x = np.arange(1, len(data['test_acc']) + 1)
            c = COLORS[idx % len(COLORS)]
            ax.plot(x, data['test_acc'], color=c, linewidth=2.2, label=f'Run {run_num}', alpha=0.85)
            g = find_grok_epoch(data['test_acc'])
            if g is not None:
                ax.plot(g, 0.95, marker='*', markersize=18, color=c,
                        markeredgecolor='black', markeredgewidth=1)
        ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Grok threshold (0.9)')
        ax.set_xlabel('Epoch (log scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Test Accuracy with Grokking Epoch (star)', fontsize=14, fontweight='bold')
        ax.set_xscale('log'); ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='lower right')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 5: loss ----
        fig, ax = plt.subplots(figsize=(11, 7)); fig.patch.set_facecolor('white')
        for idx, (run_num, data) in enumerate(all_data):
            x = np.arange(1, len(data['loss']) + 1)
            ax.semilogy(x, data['loss'], color=COLORS[idx % len(COLORS)], linewidth=2.2,
                        label=f'Run {run_num}', alpha=0.85)
        ax.set_xlabel('Epoch (log scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Training Loss: All Runs', fontsize=14, fontweight='bold')
        ax.set_xscale('log'); ax.grid(True, alpha=0.3, which='both'); ax.legend(fontsize=11)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 6: dropout gap sweep, one panel per run ----
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        fig.patch.set_facecolor('white')
        for idx, (run_num, data) in enumerate(all_data):
            ax = axes[0][idx]
            ep = data['dropout_gap_epochs']
            rr = [float(r) for r in data['dropout_rates']]
            gbr = data['dropout_gap_by_rate']
            for j, r in enumerate(rr):
                ax.plot(ep, gbr[j], linewidth=1.8, alpha=0.9, label=f'p={r:g}')
            ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
            ax.set_xscale('log')
            ax.set_xlabel('Epoch (log scale)', fontsize=11)
            ax.set_ylabel('Dropout Gap', fontsize=11)
            ax.set_title(f'Run {run_num}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both'); ax.legend(fontsize=8)
        fig.suptitle('Dropout Gap: Multi-Rate Sweep', fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 7: L2 norm vs test accuracy, dual axis, one row per run ----
        fig, axes = plt.subplots(n, 1, figsize=(11, 3.3 * n), squeeze=False)
        fig.patch.set_facecolor('white')
        for idx, (run_num, data) in enumerate(all_data):
            ax = axes[idx][0]
            c = COLORS[idx % len(COLORS)]
            x = np.arange(1, len(data['l2_norm']) + 1)
            ax.plot(x, data['l2_norm'], color=c, linewidth=2.2, label='L2 Norm')
            ax.set_ylabel('L2 Norm', color=c, fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=c)
            ax2 = ax.twinx()
            xt = np.arange(1, len(data['test_acc']) + 1)
            ax2.plot(xt, data['test_acc'], color='green', linewidth=2.0, linestyle='--',
                     label='Test Accuracy')
            ax2.set_ylabel('Test Accuracy', color='green', fontsize=11, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='green'); ax2.set_ylim([0, 1.05])
            g = find_grok_epoch(data['test_acc'])
            if g is not None:
                ax.axvline(g, color='red', linestyle=':', linewidth=2, alpha=0.6)
            ax.set_title(f'Run {run_num}: L2 Norm vs Test Accuracy', fontsize=12, fontweight='bold')
            ax.set_xscale('log'); ax.grid(True, alpha=0.3, which='both')
        axes[-1][0].set_xlabel('Epoch (log scale)', fontsize=12, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ---- Page 8: computed conclusions ----
        fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111); ax.axis('off')

        lines = ["COMPUTED CONCLUSIONS", ""]
        lines.append("1. GROKKING TIMING")
        lines.append(f"   grok epochs: {grok_txt}")
        if len(grok_known) >= 2:
            lines.append(f"   mean {np.mean(grok_known):.0f} +/- {np.std(grok_known):.0f}  "
                         f"(min {min(grok_known)}, max {max(grok_known)})")
            spread = max(grok_known) - min(grok_known)
            lines.append(f"   spread {spread} epochs -> "
                         f"{'high' if spread > 1000 else 'moderate' if spread > 300 else 'low'} seed sensitivity")
        lines.append("")
        lines.append("2. L2 NORM")
        _init = ", ".join(f"{m['l2_initial']:.2f}" for m in all_metrics)
        _final = ", ".join(f"{m['l2_final']:.2f}" for m in all_metrics)
        _decay = ", ".join(f"{m['l2_decay_pct']:.1f}" for m in all_metrics)
        lines.append(f"   initial: {_init}")
        lines.append(f"   final:   {_final}")
        lines.append(f"   decay %: {_decay}")
        lines.append("")
        lines.append("3. DROPOUT GAP SWEEP (final value, sign per rate)")
        for r in rates:
            vals = [m['dropout_gap_final_by_rate'][r] for m in all_metrics]
            sign = "negative" if all(v < 0 for v in vals) else "positive" if all(v > 0 for v in vals) else "mixed"
            lines.append(f"   p={r:g}: {', '.join(f'{v:+.4f}' for v in vals)}   ({sign} across runs)")
        lines.append("")
        lines.append("4. NOTE")
        lines.append("   A negative dropout gap means dropout did not reduce accuracy")
        lines.append("   at that rate (often a ceiling effect once the model has grokked).")
        lines.append("   Read the sign and trend per rate, not a single number.")

        ax.text(0.05, 0.97, "\n".join(lines), ha='left', va='top', fontsize=9,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.8))
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

    print(f"[OK] Report generated: {OUTPUT_PDF}")


def main():
    run_nums = discover_run_numbers(RUNS_DIR)
    if not run_nums:
        print(f"No completed runs found under {RUNS_DIR}/. Run train_four_head.py first.")
        return

    print(f"Found runs: {run_nums}")
    all_data, all_metrics = [], []
    for rn in run_nums:
        data = load_run_data(rn)
        if data is None:
            print(f"  skipping run {rn} (incomplete)")
            continue
        all_data.append((rn, data))
        all_metrics.append(extract_metrics(data, rn))
        print(f"  loaded run {rn}")

    if not all_metrics:
        print("Error: no runs loaded.")
        return

    create_report(all_data, all_metrics)

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    for m in all_metrics:
        grok = f"{m['grok_epoch']}" if m['grok_epoch'] else "N/A"
        print(f"\nRun {m['run']}:")
        print(f"  grok epoch: {grok}")
        print(f"  L2 norm: {m['l2_initial']:.4f} -> {m['l2_final']:.4f} "
              f"(decay {m['l2_decay_pct']:.2f}%)")
        gap_txt = ", ".join(f"p{r:g}={m['dropout_gap_final_by_rate'][r]:+.4f}" for r in m['rates'])
        print(f"  final dropout gap sweep: {gap_txt}")


if __name__ == "__main__":
    main()
