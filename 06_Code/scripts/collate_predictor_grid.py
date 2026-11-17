#!/usr/bin/env python3
"""Collate L2, Spectral, AGE, HTSR, Weight-PCA, Higher-MI, Commutator, and Dropout plots into a single 5x8 grid."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(REPO_ROOT, "08_Experiments", "results", "results_per_nanda_power")
FIG_DIR = os.path.join(BASE_DIR, "figures")
DIAG_DIR = os.path.join(BASE_DIR, "analysis", "bespoke_diagnostics")
OUT_PATH = os.path.join(BASE_DIR, "analysis", "predictor_grid_5x8.png")

def get_paths(seed):
    return [
        os.path.join(FIG_DIR, f"seed_{seed}_l2_norm.png"),
        os.path.join(DIAG_DIR, f"spectral_seed_{seed}.png"),
        os.path.join(DIAG_DIR, f"age_seed_{seed}.png"),
        os.path.join(DIAG_DIR, f"htsr_seed_{seed}.png"),
        os.path.join(DIAG_DIR, f"weight_pca_seed_{seed}.png"),
        os.path.join(DIAG_DIR, f"higher_mi_seed_{seed}.png"),
        os.path.join(DIAG_DIR, f"commutator_defect_seed_{seed}.png"),
        os.path.join(DIAG_DIR, f"dropout_seed_{seed}.png")
    ]

def main():
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(48, 25))
    col_titles = ["L2 Norm", "Spectral", "AGE", "HTSR", "Weight-PCA", "Higher-MI", "Commutator", "Dropout"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=16, fontweight="bold", pad=15)

    for row, seed in enumerate(range(5)):
        paths = get_paths(seed)
        for col, path in enumerate(paths):
            ax = axes[row, col]
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.axis("off")
            else:
                ax.text(0.5, 0.5, f"Missing:\n{os.path.basename(path)}", 
                        ha="center", va="center", fontsize=12, color="red")
                ax.axis("off")
            
            if col == 0:
                ax.text(-0.05, 0.5, f"Seed {seed}", transform=ax.transAxes, 
                        fontsize=14, fontweight="bold", va="center", ha="right", rotation=90)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved grid to: {OUT_PATH}")

if __name__ == "__main__":
    main()
