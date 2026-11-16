"""plot_bespoke_spectral_age.py
================================
Bespoke diagnostic plots for Predictor 3 (Spectral) and Predictor 4 (AGE)
using canonical artifacts under 08_Experiments/results/results_per_nanda_power/.

Reads from the canonical rebuild output directory structure:
- Global grokking epochs: 08_Experiments/results/results_per_nanda_power/aggregate.json ("grok_epochs")
  strictly [8021, 8253, 4094, 7831, 6637]
- AGE per-seed:
    Trajectory: 08_Experiments/results/results_per_nanda_power/seed_{i}/age/age_nc1.npy
    Metrics: 08_Experiments/results/results_per_nanda_power/seed_{i}/age/age_signal.json ("nc1_min_epoch")
- Spectral predictor-first:
    Trajectories: 08_Experiments/results/results_per_nanda_power/spectral/seed_{i}/alignment_trajectory.npy
                  08_Experiments/results/results_per_nanda_power/spectral/seed_{i}/k90_trajectory.npy
    Metrics: 08_Experiments/results/results_per_nanda_power/spectral/spectral_summary.json
- Epoch grid:
    08_Experiments/results/results_per_nanda_power/seed_{i}/age/age_checkpoints.npy or spectral_checkpoints.npy
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def set_plotting_style():
    """Apply standard clean styling."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.5


def load_epoch_grid(seed, base_dir, trajectory_len):
    """Load actual epoch array from saved checkpoint arrays or derive if length matches 401."""
    epoch_paths = [
        os.path.join(base_dir, f"seed_{seed}", "age", "age_checkpoints.npy"),
        os.path.join(base_dir, f"seed_{seed}", "spectral", "spectral_checkpoints.npy"),
        os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_checkpoints.npy"),
        os.path.join(base_dir, f"seed_{seed}", "training", "epochs.npy"),
    ]
    for p in epoch_paths:
        if os.path.exists(p):
            epochs = np.load(p)
            if len(epochs) == trajectory_len:
                return epochs

    # Derive only if length perfectly matches 401 checkpoints
    if trajectory_len == 401:
        return np.linspace(0, 40000, 401)

    raise RuntimeError(
        f"Could not load or derive exact epoch grid for seed {seed} (len={trajectory_len}). "
        f"Halted per strict constraint."
    )


def load_all_data(base_dir="08_Experiments/results/results_per_nanda_power"):
    """Load trajectories, exact checkpoint epochs, and metrics for all seeds."""
    agg_path = os.path.join(base_dir, "aggregate.json")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Missing required aggregate file: {agg_path}")

    with open(agg_path, "r") as f:
        agg_data = json.load(f)

    grok_epochs = agg_data.get("grok_epochs")
    if grok_epochs is None:
        raise KeyError(f"'grok_epochs' key not found in {agg_path}. Keys: {list(agg_data.keys())}")

    spec_summary = {}
    spec_summary_path = os.path.join(base_dir, "spectral", "spectral_summary.json")
    if os.path.exists(spec_summary_path):
        with open(spec_summary_path, "r") as f:
            spec_summary = json.load(f)

    seeds_data = []

    for seed in range(5):
        # 1. AGE: Trajectory & Metrics
        nc1_path = os.path.join(base_dir, f"seed_{seed}", "age", "age_nc1.npy")
        if not os.path.exists(nc1_path):
            fallback_nc1 = os.path.join(base_dir, f"seed_{seed}", "age", "nc1_trajectory.npy")
            if os.path.exists(fallback_nc1):
                nc1_path = fallback_nc1
            else:
                raise FileNotFoundError(f"Missing required AGE trajectory: {nc1_path}")

        nc1 = np.load(nc1_path)

        age_signal_path = os.path.join(base_dir, f"seed_{seed}", "age", "age_signal.json")
        if not os.path.exists(age_signal_path):
            fallback_age_sig = os.path.join(base_dir, f"seed_{seed}", "age", "age_metrics.json")
            if os.path.exists(fallback_age_sig):
                age_signal_path = fallback_age_sig
            else:
                raise FileNotFoundError(f"Missing required AGE metrics file: {age_signal_path}")

        with open(age_signal_path, "r") as f:
            age_metrics = json.load(f)

        nc1_min_epoch = age_metrics.get("nc1_min_epoch")
        if nc1_min_epoch is None:
            nc1_min_epoch = age_metrics.get("min_nc1_epoch")
        if nc1_min_epoch is None:
            raise KeyError(
                f"Seed {seed}: 'nc1_min_epoch' not found in {age_signal_path}. Keys: {list(age_metrics.keys())}"
            )

        # 2. Spectral: Trajectories & Metrics
        align_path = os.path.join(base_dir, "spectral", f"seed_{seed}", "alignment_trajectory.npy")
        if not os.path.exists(align_path):
            candidate_paths = [
                os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_alignment.npy"),
                os.path.join(base_dir, f"seed_{seed}", "spectral", "spectral_alignment.npy"),
                os.path.join(base_dir, f"seed_{seed}", "spectral", "alignment_trajectory.npy"),
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    align_path = cp
                    break
            else:
                raise FileNotFoundError(f"Missing required Spectral alignment trajectory for seed {seed}")

        align = np.load(align_path)

        k90_path = os.path.join(base_dir, "spectral", f"seed_{seed}", "k90_trajectory.npy")
        if not os.path.exists(k90_path):
            candidate_paths = [
                os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_k90.npy"),
                os.path.join(base_dir, f"seed_{seed}", "spectral", "spectral_k90.npy"),
                os.path.join(base_dir, f"seed_{seed}", "spectral", "k90_trajectory.npy"),
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    k90_path = cp
                    break
            else:
                raise FileNotFoundError(f"Missing required Spectral k90 trajectory for seed {seed}")

        k90 = np.load(k90_path)

        # 3. Epoch Grid
        epochs = load_epoch_grid(seed, base_dir, len(nc1))

        if len(nc1) != len(epochs) or len(align) != len(epochs) or len(k90) != len(epochs):
            raise ValueError(
                f"Seed {seed}: Mismatched array lengths: epochs={len(epochs)}, nc1={len(nc1)}, "
                f"align={len(align)}, k90={len(k90)}"
            )

        # Extract spectral metrics for this seed
        alignment_max_epoch = None
        k_90_min_epoch = None
        if f"seed_{seed}" in spec_summary:
            seed_block = spec_summary[f"seed_{seed}"]
            alignment_max_epoch = seed_block.get("alignment_max_epoch")
            k_90_min_epoch = seed_block.get("k_90_min_epoch") or seed_block.get("k90_min_epoch")
        elif "diagnostics" in spec_summary and "alignment_global_max_epochs" in spec_summary["diagnostics"]:
            diag = spec_summary["diagnostics"]
            alignment_max_epoch = diag["alignment_global_max_epochs"][seed]
            k_90_min_epoch = diag["k90_global_min_epochs"][seed]
        else:
            # Check spectral_signal.json in seed or summary.json
            sig_path = os.path.join(base_dir, f"seed_{seed}", "spectral", "spectral_signal.json")
            if not os.path.exists(sig_path):
                sig_path = os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_signal.json")
            if os.path.exists(sig_path):
                with open(sig_path, "r") as sf:
                    sig_data = json.load(sf)
                    alignment_max_epoch = sig_data.get("alignment_max_epoch")
                    k_90_min_epoch = sig_data.get("k_90_min_epoch") or sig_data.get("k90_min_epoch")
            if alignment_max_epoch is None or k_90_min_epoch is None:
                seed_summary_path = os.path.join(base_dir, f"seed_{seed}", "summary.json")
                if os.path.exists(seed_summary_path):
                    with open(seed_summary_path, "r") as ssf:
                        ss_data = json.load(ssf)
                        sp_block = ss_data.get("spectral_predictor") or {}
                        alignment_max_epoch = sp_block.get("alignment_max_epoch")
                        k_90_min_epoch = sp_block.get("k_90_min_epoch") or sp_block.get("k90_min_epoch")

        # Fallback to direct computation from trajectory if not in summary files
        if alignment_max_epoch is None:
            alignment_max_epoch = int(epochs[np.argmax(align)])
        if k_90_min_epoch is None:
            k_90_min_epoch = int(epochs[np.argmin(k90)])

        seeds_data.append({
            "seed": seed,
            "epochs": epochs,
            "grok_epoch": grok_epochs[seed],
            "nc1": nc1,
            "nc1_min_epoch": nc1_min_epoch,
            "align": align,
            "k90": k90,
            "alignment_max_epoch": alignment_max_epoch,
            "k_90_min_epoch": k_90_min_epoch,
        })

    return grok_epochs, seeds_data


def plot_age_per_seed(s_data, out_path):
    """Generate AGE Diagnostic Plot for a single seed (size 10x6)."""
    seed = s_data["seed"]
    epochs = s_data["epochs"]
    nc1 = s_data["nc1"]
    grok_epoch = s_data["grok_epoch"]
    nc1_min_epoch = s_data["nc1_min_epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Y-axis: log10(nc1)
    log_nc1 = np.log10(nc1)
    ax.plot(epochs, log_nc1, color="#1f77b4", lw=1.8, label=r"$\log_{10}(\mathrm{NC1})$")

    # Annotations
    ax.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")
    ax.axvline(x=nc1_min_epoch, color="purple", linestyle=":", lw=2.0, label=f"NC1 Min ({nc1_min_epoch})")

    max_epoch = epochs[-1]
    ax.axvspan(grok_epoch, max_epoch, color="red", alpha=0.1, label="Post-Grok Phase")

    ax.set_xlim(epochs[0], max_epoch)
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel(r"$\log_{10}(\mathrm{NC1})$", fontsize=11)
    ax.set_title(f"Seed {seed}: AGE (NC1) Trajectory", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spectral_per_seed(s_data, out_path):
    """Generate Spectral Diagnostic Plot for a single seed (size 12x6, 2 subplots sharing X-axis)."""
    seed = s_data["seed"]
    epochs = s_data["epochs"]
    align = s_data["align"]
    k90 = s_data["k90"]
    grok_epoch = s_data["grok_epoch"]
    alignment_max_epoch = s_data["alignment_max_epoch"]
    k_90_min_epoch = s_data["k_90_min_epoch"]
    max_epoch = epochs[-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Subplot 1 (Top): Alignment
    ax1.plot(epochs, align, color="#2ca02c", lw=1.8, label="Alignment Score")
    ax1.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")
    ax1.axvline(x=alignment_max_epoch, color="blue", linestyle=":", lw=2.0, label=f"Alignment Max ({alignment_max_epoch})")
    ax1.set_ylabel("Task-Model Alignment", fontsize=11)
    ax1.set_title(f"Seed {seed}: Spectral Predictors", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    # Subplot 2 (Bottom): Effective Rank (k90)
    ax2.plot(epochs, k90, color="#ff7f0e", lw=1.8, label="Effective Rank (k90)")
    ax2.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")
    ax2.axvline(x=k_90_min_epoch, color="green", linestyle=":", lw=2.0, label=f"k90 Min ({k_90_min_epoch})")
    ax2.set_xlim(epochs[0], max_epoch)
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Effective Rank (k90)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_age_all_seeds_pdf(seeds_data, out_path):
    """Combined AGE plot across all 5 seeds on a single set of axes."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(11, 7))

    all_grok = [s_data["grok_epoch"] for s_data in seeds_data]
    mean_grok = float(np.mean(all_grok))
    max_epoch = max(s_data["epochs"][-1] for s_data in seeds_data)
    min_epoch = min(s_data["epochs"][0] for s_data in seeds_data)

    for s_data in seeds_data:
        s = s_data["seed"]
        c = colors[s % len(colors)]
        epochs = s_data["epochs"]
        log_nc1 = np.log10(s_data["nc1"])
        ax.plot(epochs, log_nc1, color=c, lw=1.5, label=f"Seed {s} log10(NC1)")

        # Individual grok line
        ax.axvline(x=s_data["grok_epoch"], color=c, linestyle="--", alpha=0.5, lw=1.0)
        # Individual min NC1 marker
        ax.axvline(x=s_data["nc1_min_epoch"], color=c, linestyle=":", alpha=0.8, lw=1.2)

    ax.axvline(x=mean_grok, color="red", linestyle="--", lw=2.0, label=f"Mean Grok ({mean_grok:.0f})")
    ax.axvspan(min(all_grok), max_epoch, color="red", alpha=0.08, label="Post-Grok Span (All Seeds)")

    ax.set_xlim(min_epoch, max_epoch)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(\mathrm{NC1})$", fontsize=12)
    ax.set_title("AGE (NC1) Trajectories Across All 5 Seeds", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_spectral_all_seeds_pdf(seeds_data, out_path):
    """Combined Spectral plot across all 5 seeds with shared X-axis."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    all_grok = [s_data["grok_epoch"] for s_data in seeds_data]
    mean_grok = float(np.mean(all_grok))
    max_epoch = max(s_data["epochs"][-1] for s_data in seeds_data)
    min_epoch = min(s_data["epochs"][0] for s_data in seeds_data)

    # Top: Alignment
    for s_data in seeds_data:
        s = s_data["seed"]
        c = colors[s % len(colors)]
        epochs = s_data["epochs"]
        ax1.plot(epochs, s_data["align"], color=c, lw=1.4, label=f"Seed {s}")
        ax1.axvline(x=s_data["alignment_max_epoch"], color=c, linestyle=":", alpha=0.7, lw=1.0)
    ax1.axvline(x=mean_grok, color="red", linestyle="--", lw=1.8, label=f"Mean Grok ({mean_grok:.0f})")
    ax1.set_ylabel("Task-Model Alignment", fontsize=11)
    ax1.set_title("Spectral Alignment Across All 5 Seeds", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    # Bottom: k90
    for s_data in seeds_data:
        s = s_data["seed"]
        c = colors[s % len(colors)]
        epochs = s_data["epochs"]
        ax2.plot(epochs, s_data["k90"], color=c, lw=1.4, label=f"Seed {s}")
        ax2.axvline(x=s_data["k_90_min_epoch"], color=c, linestyle=":", alpha=0.7, lw=1.0)
    ax2.axvline(x=mean_grok, color="red", linestyle="--", lw=1.8, label=f"Mean Grok ({mean_grok:.0f})")
    ax2.set_xlim(min_epoch, max_epoch)
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Effective Rank (k90)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    fig.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main():
    set_plotting_style()
    grok_epochs, seeds_data = load_all_data()

    out_dirs = [
        "08_Experiments/results/nanda_unified/analysis/bespoke_diagnostics",
        "08_Experiments/results/results_per_nanda_power/analysis/bespoke_diagnostics",
    ]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

    generated_files = []

    for out_dir in out_dirs:
        # Per-seed plots
        for s_data in seeds_data:
            seed = s_data["seed"]
            # AGE plot
            age_png = os.path.join(out_dir, f"age_seed_{seed}.png")
            plot_age_per_seed(s_data, age_png)
            generated_files.append(age_png)

            # Spectral plot
            spec_png = os.path.join(out_dir, f"spectral_seed_{seed}.png")
            plot_spectral_per_seed(s_data, spec_png)
            generated_files.append(spec_png)

        # Combined PDFs
        age_pdf = os.path.join(out_dir, "age_all_seeds.pdf")
        plot_age_all_seeds_pdf(seeds_data, age_pdf)
        generated_files.append(age_pdf)

        spec_pdf = os.path.join(out_dir, "spectral_all_seeds.pdf")
        plot_spectral_all_seeds_pdf(seeds_data, spec_pdf)
        generated_files.append(spec_pdf)

    print("Successfully generated bespoke diagnostic plots:")
    for f in generated_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
