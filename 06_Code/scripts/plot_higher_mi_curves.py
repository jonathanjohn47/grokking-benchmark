#!/usr/bin/env python3
"""
plot_higher_mi_curves.py
=========================

RAW-CURVE plots for Higher-MI (Pomarico et al. 2025, arXiv:2507.23346; see
src/predictors/higher_mi.py) -- generated BEFORE wiring the signal into
analyze_nanda_unified.py's scored table, per Jonathan's own stated practice:
"Will run both and report raw Omega(t) before trusting event epochs." These
two plots are exactly that raw look, nothing more -- no event-epoch lines,
no PASS/FAIL, no frozen-protocol scoring here.

Plot 1 (higher_mi_population_omega_all_seeds.png): Omega_population(t) vs.
epoch, one subplot per seed, with a vertical line at that seed's
grok_epoch -- same "stacked per-seed subplot" convention as
analyze_nanda_unified.py's own plot_signals.

Plot 2 (higher_mi_population_vs_temporal_seed0.png): seed 0's population
signal side by side with its temporal/windowed EXPLORATORY CONTROL signal
(W=100, stride=10) -- the direct comparison meant to test the population
signal's shared-example-difficulty confound risk (see higher_mi.py module
docstring, "MAJOR ADAPTATION" / "EXPLORATORY CONTROL").

Usage (from the repo root):
    python 06_Code/scripts/plot_higher_mi_curves.py
Outputs go to 08_Experiments/results/nanda_unified/reports/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO_ROOT, "08_Experiments", "results", "nanda_unified")
REPORTS = os.path.join(RESULTS, "reports")


def load_seed_higher_mi(seed):
    d = os.path.join(RESULTS, f"seed_{seed}")
    with open(os.path.join(d, "summary.json")) as handle:
        summary = json.load(handle)
    hd = os.path.join(d, "higher_mi")
    return {
        "seed": seed,
        "grok_epoch": summary["grok_epoch"],
        "checkpoints": np.load(os.path.join(hd, "higher_mi_checkpoints.npy")),
        "population_omega": np.load(os.path.join(hd, "higher_mi_population_omega.npy")),
        "temporal_checkpoints": np.load(os.path.join(hd, "higher_mi_temporal_checkpoints.npy")),
        "temporal_omega": np.load(os.path.join(hd, "higher_mi_temporal_omega.npy")),
    }


def plot_population_all_seeds(runs, path):
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    cols = plt.cm.tab10(np.arange(n))
    for s, r in enumerate(runs):
        ax = axes[s]
        ax.plot(r["checkpoints"], r["population_omega"], color=cols[s], lw=1.3,
                label=f"seed {r['seed']}")
        ax.axhline(0, color="grey", lw=0.6, ls="-")
        ax.axvline(r["grok_epoch"], color="crimson", ls="--", lw=1.2,
                   label=f"grok ({r['grok_epoch']})")
        ax.set_ylabel("Omega_pop")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("epoch")
    fig.suptitle("Higher-MI: population O-information Omega(t) per seed "
                 "(dashed red = grok epoch, grey line = Omega=0)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_population_vs_temporal_seed0(run0, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(run0["checkpoints"], run0["population_omega"], color="tab:blue", lw=1.3)
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].axvline(run0["grok_epoch"], color="crimson", ls="--", lw=1.2, label="grok")
    ax[0].set(title="PRIMARY: Population O-information", xlabel="epoch", ylabel="Omega")
    ax[0].legend()

    ax[1].plot(run0["temporal_checkpoints"], run0["temporal_omega"], color="tab:orange", lw=1.3)
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].axvline(run0["grok_epoch"], color="crimson", ls="--", lw=1.2, label="grok")
    ax[1].set(title="EXPLORATORY CONTROL: Temporal O-information (W=100, stride=10)",
              xlabel="epoch (window center)", ylabel="Omega")
    ax[1].legend()

    fig.suptitle("Seed 0 -- population vs. temporal O-information "
                 "(testing the shared-example-difficulty confound risk)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    seeds = sorted(int(n.split("_")[1]) for n in os.listdir(RESULTS) if n.startswith("seed_"))
    runs = [load_seed_higher_mi(s) for s in seeds]

    p1 = os.path.join(REPORTS, "higher_mi_population_omega_all_seeds.png")
    plot_population_all_seeds(runs, p1)
    print(f"Wrote {p1}")

    p2 = os.path.join(REPORTS, "higher_mi_population_vs_temporal_seed0.png")
    plot_population_vs_temporal_seed0(runs[0], p2)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
