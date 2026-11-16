"""plot_bespoke_spectral_age.py
================================
Bespoke diagnostic plots for Predictor 3 (Spectral) and Predictor 4 (AGE)
using artifacts under 08_Experiments/results/nanda_unified/.

Uses exact loaded checkpoint epoch grid (no np.linspace guessing)
and correct grokking epochs for seeds 0–4: [8021, 8253, 4094, 7831, 6637].
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Correct grokking epochs for seeds 0-4 (primary cut 0.99)
GROK_EPOCHS = {
    0: 8021,
    1: 8253,
    2: 4094,
    3: 7831,
    4: 6637,
}


def set_plotting_style():
    """Apply standard clean styling."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.5


def find_epoch_grid(seed, seed_dir):
    """Locate and return the exact checkpoint epoch array without guessing."""
    candidate_paths = [
        os.path.join(seed_dir, "spectral", "spectral_checkpoints.npy"),
        os.path.join(seed_dir, "age", "age_checkpoints.npy"),
        os.path.join(seed_dir, "epochs.npy"),
        os.path.join(seed_dir, "spectral_checkpoints.npy"),
        os.path.join(seed_dir, "checkpoints.npy"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            epochs = np.load(p)
            return epochs

    # If not found, halt and report per strict constraint
    age_dir = os.path.join(seed_dir, "age")
    spectral_dir = os.path.join(seed_dir, "spectral")
    age_contents = os.listdir(age_dir) if os.path.exists(age_dir) else "DOES NOT EXIST"
    spec_contents = os.listdir(spectral_dir) if os.path.exists(spectral_dir) else "DOES NOT EXIST"
    raise RuntimeError(
        f"Exact epoch array not found for seed {seed} in {seed_dir}!\n"
        f"Directory contents:\n"
        f"  {age_dir}: {age_contents}\n"
        f"  {spectral_dir}: {spec_contents}\n"
        "Halted according to strict constraint: do not guess epoch grid."
    )


def load_seed_data(base_dir="08_Experiments/results/nanda_unified"):
    """Load trajectories, exact checkpoint epochs, and metrics for all seeds."""
    seeds_data = []

    for seed in range(5):
        seed_dir = os.path.join(base_dir, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            raise FileNotFoundError(f"Missing required seed directory: {seed_dir}")

        epochs = find_epoch_grid(seed, seed_dir)

        # 1. AGE trajectory & metrics paths
        nc1_path = os.path.join(seed_dir, "age", "nc1_trajectory.npy")
        if not os.path.exists(nc1_path):
            raise FileNotFoundError(f"Missing required file: {nc1_path}")

        age_metrics_path = os.path.join(seed_dir, "age", "age_metrics.json")
        if not os.path.exists(age_metrics_path):
            raise FileNotFoundError(f"Missing required file: {age_metrics_path}")

        # 2. Spectral trajectories & metrics paths
        align_path = os.path.join(seed_dir, "spectral", "alignment_trajectory.npy")
        if not os.path.exists(align_path):
            raise FileNotFoundError(f"Missing required file: {align_path}")

        k90_path = os.path.join(seed_dir, "spectral", "k90_trajectory.npy")
        if not os.path.exists(k90_path):
            raise FileNotFoundError(f"Missing required file: {k90_path}")

        spec_metrics_path = os.path.join(seed_dir, "spectral", "spectral_metrics.json")
        if not os.path.exists(spec_metrics_path):
            raise FileNotFoundError(f"Missing required file: {spec_metrics_path}")

        # Load arrays and json
        nc1 = np.load(nc1_path)
        with open(age_metrics_path, "r") as f:
            age_metrics = json.load(f)

        align = np.load(align_path)
        k90 = np.load(k90_path)
        with open(spec_metrics_path, "r") as f:
            spec_metrics = json.load(f)

        if len(nc1) != len(epochs):
            raise ValueError(
                f"Seed {seed}: NC1 trajectory length ({len(nc1)}) != epoch grid length ({len(epochs)})"
            )
        if len(align) != len(epochs):
            raise ValueError(
                f"Seed {seed}: Alignment trajectory length ({len(align)}) != epoch grid length ({len(epochs)})"
            )
        if len(k90) != len(epochs):
            raise ValueError(
                f"Seed {seed}: k90 trajectory length ({len(k90)}) != epoch grid length ({len(epochs)})"
            )

        seeds_data.append({
            "seed": seed,
            "epochs": epochs,
            "grok_epoch": GROK_EPOCHS[seed],
            "nc1": nc1,
            "age_metrics": age_metrics,
            "align": align,
            "k90": k90,
            "spec_metrics": spec_metrics,
        })

    return seeds_data


def plot_age_per_seed(s_data, out_path):
    """Generate AGE Diagnostic Plot for a single seed (size 10x6)."""
    seed = s_data["seed"]
    epochs = s_data["epochs"]
    nc1 = s_data["nc1"]
    grok_epoch = s_data["grok_epoch"]
    age_metrics = s_data["age_metrics"]
    min_nc1_epoch = age_metrics["min_nc1_epoch"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Y-axis: log10(nc1)
    log_nc1 = np.log10(nc1)
    ax.plot(epochs, log_nc1, color="#1f77b4", lw=1.8, label=r"$\log_{10}(\mathrm{NC1})$")

    # 1. Vertical dashed red line at grokking epoch
    ax.axvline(x=grok_epoch, color="red", linestyle="--", lw=1.8, label=f"Grokking ({grok_epoch})")

    # 2. Vertical dotted purple line at NC1 Global Min
    ax.axvline(x=min_nc1_epoch, color="purple", linestyle=":", lw=2.0, label=f"NC1 Min ({min_nc1_epoch})")

    # 3. Red alpha span from grokking epoch to max epoch
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
    spec_metrics = s_data["spec_metrics"]
    alignment_max_epoch = spec_metrics["alignment_max_epoch"]
    k90_min_epoch = spec_metrics["k90_min_epoch"]
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
    ax2.axvline(x=k90_min_epoch, color="green", linestyle=":", lw=2.0, label=f"k90 Min ({k90_min_epoch})")
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
        min_nc1_epoch = s_data["age_metrics"]["min_nc1_epoch"]
        ax.axvline(x=min_nc1_epoch, color=c, linestyle=":", alpha=0.8, lw=1.2)

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
        ax1.axvline(x=s_data["spec_metrics"]["alignment_max_epoch"], color=c, linestyle=":", alpha=0.7, lw=1.0)
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
        ax2.axvline(x=s_data["spec_metrics"]["k90_min_epoch"], color=c, linestyle=":", alpha=0.7, lw=1.0)
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
    seeds_data = load_seed_data()

    out_dir = "08_Experiments/results/nanda_unified/analysis/bespoke_diagnostics"
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
