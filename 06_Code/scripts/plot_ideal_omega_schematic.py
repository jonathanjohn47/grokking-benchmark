"""Schematic (NOT real data) of the ideal Higher-MI population O-information curve.

Purpose: thesis figure showing what a working Higher-MI predictor was expected to look like:
Omega starts in the synergy zone (<0), rises smoothly into the redundancy zone (>0),
and peaks BEFORE the grok epoch G. The real measured curves are in
08_Experiments/results/nanda_unified/reports/measured/measured_higher_mi_population_omega_all_seeds.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(
    ROOT,
    "05_Thesis_Document",
    "Figures",
    "schematics",
    "schematic_not_measured_ideal_higher_mi_omega.png",
)

t = np.linspace(0, 40000, 401)
G = 13000
peak = 0.85 * G
omega = np.where(
    t < peak,
    -0.6 + 1.9 * np.exp(-((t - peak) / 5500) ** 2),
    0.1 + 1.2 * np.exp(-((t - peak) / 9000) ** 2),
)

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(t, omega, color="tab:blue", lw=2.5, label=r"$\Omega_{\mathrm{pop}}(t)$ (ideal)")
ax.axvline(G, color="red", ls="--", label="grok epoch $G$")
ax.axhline(0, color="gray", lw=1)
ax.axhspan(-1.2, 0, color="tab:purple", alpha=0.08)
ax.axhspan(0, 1.6, color="tab:green", alpha=0.06)
ax.text(300, -0.95, r"synergy zone ($\Omega<0$)", color="purple")
ax.text(300, 1.4, r"redundancy zone ($\Omega>0$)", color="green")
ax.annotate("peak before $G$\n(= early warning)", xy=(peak, omega.max()),
            xytext=(21000, 1.0), arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("training epoch")
ax.set_ylabel(r"O-information $\Omega$")
ax.set_ylim(-1.2, 1.6)
ax.set_title("Expected behaviour of a working Higher-MI predictor (schematic, not measured data)", fontsize=9)
ax.legend(loc="lower right")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200)
print("saved", OUT)
