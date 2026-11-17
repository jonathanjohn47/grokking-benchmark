"""plot_bespoke_spectral_age.py
================================
Strict bespoke diagnostic plots for Predictor 3 (Spectral) and Predictor 4 (AGE)
using strictly canonical artifacts under 08_Experiments/results/results_per_nanda_power/.

Strict requirements:
- No fallback paths, no np.linspace.
- Global grokking epochs: 08_Experiments/results/results_per_nanda_power/aggregate.json ("grok_epochs")
- For each seed:
  - AGE:
      Trajectory: 08_Experiments/results/results_per_nanda_power/seed_{i}/age/age_nc1.npy
      Checkpoints: 08_Experiments/results/results_per_nanda_power/seed_{i}/age/age_checkpoints.npy
      Metrics: 08_Experiments/results/results_per_nanda_power/seed_{i}/age/age_signal.json ("nc1_min_epoch")
  - Spectral:
      Trajectories: 08_Experiments/results/results_per_nanda_power/spectral/seed_{i}/spectral_alignment.npy
                    08_Experiments/results/results_per_nanda_power/spectral/seed_{i}/spectral_k90.npy
      Checkpoints:  08_Experiments/results/results_per_nanda_power/spectral/seed_{i}/spectral_checkpoints.npy
      Metrics:      08_Experiments/results/results_per_nanda_power/spectral/spectral_summary.json
- Output directory:
  08_Experiments/results/results_per_nanda_power/analysis/bespoke_diagnostics/
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANONICAL_BASE_DIR = os.path.join(REPO_ROOT, "08_Experiments", "results", "results_per_nanda_power")


def set_plotting_style():
    """Apply standard clean styling."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.5


def load_all_data(base_dir=CANONICAL_BASE_DIR):
    """Load trajectories, exact checkpoint epochs, and metrics for all seeds strictly."""
    agg_path = os.path.join(base_dir, "aggregate.json")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Missing required aggregate file: {agg_path}")

    with open(agg_path, "r") as f:
        agg_data = json.load(f)

    grok_epochs = agg_data.get("grok_epochs")
    if grok_epochs is None:
        raise KeyError(f"'grok_epochs' key not found in {agg_path}. Keys: {list(agg_data.keys())}")

    spec_summary_path = os.path.join(base_dir, "spectral", "spectral_summary.json")
    if not os.path.exists(spec_summary_path):
        raise FileNotFoundError(f"Missing required Spectral summary file: {spec_summary_path}")

    with open(spec_summary_path, "r") as f:
        spec_summary = json.load(f)

    seeds_data = []

    for seed in range(5):
        # 1. AGE: Trajectory, Epochs & Metrics
        nc1_path = os.path.join(base_dir, f"seed_{seed}", "age", "age_nc1.npy")
        if not os.path.exists(nc1_path):
            raise FileNotFoundError(f"Missing required AGE trajectory: {nc1_path}")
        nc1 = np.load(nc1_path)

        age_ep_path = os.path.join(base_dir, f"seed_{seed}", "age", "age_checkpoints.npy")
        if not os.path.exists(age_ep_path):
            raise FileNotFoundError(f"Missing required AGE checkpoints: {age_ep_path}")
        age_epochs = np.load(age_ep_path)

        if len(nc1) != len(age_epochs):
            raise ValueError(
                f"Seed {seed}: Mismatched AGE lengths: nc1={len(nc1)}, age_checkpoints={len(age_epochs)}"
            )

        age_signal_path = os.path.join(base_dir, f"seed_{seed}", "age", "age_signal.json")
        if not os.path.exists(age_signal_path):
            raise FileNotFoundError(f"Missing required AGE metrics file: {age_signal_path}")

        with open(age_signal_path, "r") as f:
            age_metrics = json.load(f)

        nc1_min_epoch = age_metrics.get("nc1_min_epoch")
        if nc1_min_epoch is None:
            raise KeyError(
                f"Seed {seed}: 'nc1_min_epoch' not found in {age_signal_path}. Keys: {list(age_metrics.keys())}"
            )

        # 2. Spectral: Trajectories, Epochs & Metrics
        align_path = os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_alignment.npy")
        if not os.path.exists(align_path):
            raise FileNotFoundError(f"Missing required Spectral alignment trajectory: {align_path}")
        align = np.load(align_path)

        k90_path = os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_k90.npy")
        if not os.path.exists(k90_path):
            raise FileNotFoundError(f"Missing required Spectral k90 trajectory: {k90_path}")
        k90 = np.load(k90_path)

        spec_ep_path = os.path.join(base_dir, "spectral", f"seed_{seed}", "spectral_checkpoints.npy")
        if not os.path.exists(spec_ep_path):
            raise FileNotFoundError(f"Missing required Spectral checkpoints: {spec_ep_path}")
        spec_epochs = np.load(spec_ep_path)

        if len(align) != len(spec_epochs) or len(k90) != len(spec_epochs):
            raise ValueError(
                f"Seed {seed}: Mismatched Spectral lengths: align={len(align)}, k90={len(k90)}, spec_epochs={len(spec_epochs)}"
            )

        # Spectral metrics from spectral_summary.json
        seed_key = f"seed_{seed}"
        if seed_key not in spec_summary:
            raise KeyError(f"Missing '{seed_key}' block in {spec_summary_path}")

        seed_block = spec_summary[seed_key]
        alignment_max_epoch = seed_block.get("alignment_max_epoch")
        k_90_min_epoch = seed_block.get("k_90_min_epoch")

        if alignment_max_epoch is None or k_90_min_epoch is None:
            raise KeyError(
                f"Seed {seed}: missing alignment_max_epoch or k_90_min_epoch in {spec_summary_path}"
            )

        # 3. HTSR: Trajectory, Epochs & Metrics
        htsr_signal_path = os.path.join(base_dir, f"seed_{seed}", "htsr", "htsr_signal.json")
        if not os.path.exists(htsr_signal_path):
            raise FileNotFoundError(f"Missing required HTSR metrics file: {htsr_signal_path}")

        with open(htsr_signal_path, "r") as f:
            htsr_metrics = json.load(f)

        htsr_epochs = np.array(htsr_metrics["htsr_checkpoints"])
        alpha = np.array(htsr_metrics["alpha_history"])
        alpha_min_epoch = htsr_metrics["alpha_min_epoch"]

        # 4. Weight-PCA: Trajectory & Metrics
        w_pca_path = os.path.join(base_dir, f"seed_{seed}", "weight_pca", "weight_pca_signal.json")
        if not os.path.exists(w_pca_path):
            raise FileNotFoundError(f"Missing required Weight-PCA signal file: {w_pca_path}")
        with open(w_pca_path, "r") as f:
            w_pca_metrics = json.load(f)

        # 5. Higher-MI: Trajectory & Metrics
        hmi_path = os.path.join(base_dir, f"seed_{seed}", "higher_mi", "higher_mi_signal.json")
        if not os.path.exists(hmi_path):
            raise FileNotFoundError(f"Missing required Higher-MI signal file: {hmi_path}")
        with open(hmi_path, "r") as f:
            hmi_metrics = json.load(f)

        seeds_data.append({
            "seed": seed,
            "grok_epoch": int(grok_epochs[seed]),
            "age_epochs": age_epochs,
            "nc1": nc1,
            "nc1_min_epoch": int(nc1_min_epoch),
            "spec_epochs": spec_epochs,
            "align": align,
            "k90": k90,
            "alignment_max_epoch": int(alignment_max_epoch),
            "k_90_min_epoch": int(k_90_min_epoch),
            "htsr_epochs": htsr_epochs,
            "alpha": alpha,
            "alpha_min_epoch": int(alpha_min_epoch),
            "w_pca_epochs": w_pca_metrics["weight_pca_checkpoints"],
            "w_pca_rank_history": w_pca_metrics["norm_eff_rank_history"],
            "w_pca_min_epoch": w_pca_metrics["norm_eff_rank_min_epoch"],
            "hmi_epochs": hmi_metrics["higher_mi_checkpoints"],
            "hmi_omega_history": hmi_metrics["population_omega_history"],
            "hmi_max_epoch": hmi_metrics["population_omega_max_epoch"]
        })

    return grok_epochs, seeds_data


def plot_age_per_seed(s_data, out_path):
    """Generate AGE Diagnostic Plot for a single seed (size 10x6)."""
    seed = s_data["seed"]
    epochs = s_data["age_epochs"]
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

    ax.set_xscale("log")
    ax.set_xlim(max(1, epochs[0]), max_epoch)
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
    epochs = s_data["spec_epochs"]
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
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(max(1, epochs[0]), max_epoch)
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
    max_epoch = max(s_data["age_epochs"][-1] for s_data in seeds_data)
    min_epoch = min(s_data["age_epochs"][0] for s_data in seeds_data)

    for s_data in seeds_data:
        s = s_data["seed"]
        c = colors[s % len(colors)]
        epochs = s_data["age_epochs"]
        log_nc1 = np.log10(s_data["nc1"])
        ax.plot(epochs, log_nc1, color=c, lw=1.5, label=f"Seed {s} log10(NC1)")

        # Individual grok line
        ax.axvline(x=s_data["grok_epoch"], color=c, linestyle="--", alpha=0.5, lw=1.0)
        # Individual min NC1 marker
        ax.axvline(x=s_data["nc1_min_epoch"], color=c, linestyle=":", alpha=0.8, lw=1.2)

    ax.axvline(x=mean_grok, color="red", linestyle="--", lw=2.0, label=f"Mean Grok ({mean_grok:.0f})")
    ax.axvspan(min(all_grok), max_epoch, color="red", alpha=0.08, label="Post-Grok Span (All Seeds)")

    ax.set_xscale("log")
    ax.set_xlim(max(1, min_epoch), max_epoch)
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
    max_epoch = max(s_data["spec_epochs"][-1] for s_data in seeds_data)
    min_epoch = min(s_data["spec_epochs"][0] for s_data in seeds_data)

    # Top: Alignment
    for s_data in seeds_data:
        s = s_data["seed"]
        c = colors[s % len(colors)]
        epochs = s_data["spec_epochs"]
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
        epochs = s_data["spec_epochs"]
        ax2.plot(epochs, s_data["k90"], color=c, lw=1.4, label=f"Seed {s}")
        ax2.axvline(x=s_data["k_90_min_epoch"], color=c, linestyle=":", alpha=0.7, lw=1.0)
    ax2.axvline(x=mean_grok, color="red", linestyle="--", lw=1.8, label=f"Mean Grok ({mean_grok:.0f})")
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(max(1, min_epoch), max_epoch)
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Effective Rank (k90)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=8, ncol=2)

    fig.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_htsr_per_seed(s_data, out_path):
    """Generate HTSR Alpha Diagnostic Plot for a single seed."""
    seed = s_data["seed"]
    epochs = s_data["htsr_epochs"]
    alpha = s_data["alpha"]
    grok_epoch = s_data["grok_epoch"]
    alpha_min_epoch = s_data["alpha_min_epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, alpha, color="#1b7f4e", lw=1.8, label=r"$\alpha$ (HTSR)")
    
    ax.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")
    ax.axvline(x=alpha_min_epoch, color="purple", linestyle=":", lw=2.0, label=f"Alpha Min ({alpha_min_epoch})")

    ax.set_xscale("log")
    ax.set_xlim(max(1, epochs[0]), epochs[-1])
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel(r"Heavy-Tailed Power-Law Exponent ($\alpha$)", fontsize=11)
    ax.set_title(f"Seed {seed}: HTSR Alpha Trajectory", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_weight_pca_per_seed(s_data, out_path):
    """Generate Weight-PCA Diagnostic Plot for a single seed."""
    seed = s_data["seed"]
    epochs = s_data["w_pca_epochs"]
    rank = s_data["w_pca_rank_history"]
    grok_epoch = s_data["grok_epoch"]
    min_epoch = s_data["w_pca_min_epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, rank, color="#1f77b4", lw=1.8, label="Normalized Effective Rank")
    
    ax.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")
    ax.axvline(x=min_epoch, color="purple", linestyle=":", lw=2.0, label=f"Min Rank ({min_epoch})")

    ax.set_xscale("log")
    ax.set_xlim(max(1, epochs[0]), epochs[-1])
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Norm. Effective Rank", fontsize=11)
    ax.set_title(f"Seed {seed}: Weight-PCA Trajectory", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_higher_mi_per_seed(s_data, out_path):
    """Generate Higher-MI Diagnostic Plot for a single seed."""
    seed = s_data["seed"]
    epochs = s_data["hmi_epochs"]
    omega = s_data["hmi_omega_history"]
    grok_epoch = s_data["grok_epoch"]
    max_epoch = s_data["hmi_max_epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, omega, color="#d62728", lw=1.8, label="Population O-information")
    
    ax.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")
    ax.axvline(x=max_epoch, color="purple", linestyle=":", lw=2.0, label=f"Max Omega ({max_epoch})")

    ax.set_xscale("log")
    ax.set_xlim(max(1, epochs[0]), epochs[-1])
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("O-information", fontsize=11)
    ax.set_title(f"Seed {seed}: Higher-MI Trajectory", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    set_plotting_style()
    grok_epochs, seeds_data = load_all_data(CANONICAL_BASE_DIR)

    out_dir = os.path.join(CANONICAL_BASE_DIR, "analysis", "bespoke_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    generated_files = []

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

        # HTSR plot
        htsr_png = os.path.join(out_dir, f"htsr_seed_{seed}.png")
        plot_htsr_per_seed(s_data, htsr_png)
        generated_files.append(htsr_png)

        w_pca_png = os.path.join(out_dir, f"weight_pca_seed_{seed}.png")
        plot_weight_pca_per_seed(s_data, w_pca_png)
        generated_files.append(w_pca_png)

        hmi_png = os.path.join(out_dir, f"higher_mi_seed_{seed}.png")
        plot_higher_mi_per_seed(s_data, hmi_png)
        generated_files.append(hmi_png)

    # Combined PDFs
    age_pdf = os.path.join(out_dir, "age_all_seeds.pdf")
    plot_age_all_seeds_pdf(seeds_data, age_pdf)
    generated_files.append(age_pdf)

    spec_pdf = os.path.join(out_dir, "spectral_all_seeds.pdf")
    plot_spectral_all_seeds_pdf(seeds_data, spec_pdf)
    generated_files.append(spec_pdf)

    print(f"Successfully generated {len(generated_files)} bespoke diagnostic plots in {out_dir}:")
    for f in generated_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
