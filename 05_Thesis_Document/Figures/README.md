# Figure organisation

Higher-MI figures are separated by evidence type so a conceptual illustration cannot be mistaken for an experimental result.

## Measured experiment figures

The measured Higher-MI plots are stored beside the experiment outputs in `08_Experiments/results/nanda_unified/reports/measured/`:

- `measured_higher_mi_population_omega_all_seeds.png` - population O-information from the saved checkpoint evaluations for Seeds 0-4.
- `measured_higher_mi_population_vs_temporal_seed0.png` - measured population signal and exploratory temporal control for Seed 0.

These plots are generated from the saved `.npy` checkpoint measurements by running `python 06_Code/scripts/plot_higher_mi_curves.py` from the repository root.

## Schematics and hypothetical figures

The hypothetical ideal-behaviour diagram is kept under `schematics/`:

- `schematic_not_measured_ideal_higher_mi_omega.png` - illustrative expectation only; its curve is generated from chosen values and is not derived from checkpoints.

Regenerate it with `python 06_Code/scripts/plot_ideal_omega_schematic.py`. Do not cite it as measured evidence or place it among measured result plots without a prominent schematic label.

## General rule

Use `measured_` for figures produced from experimental measurements and `schematic_` for conceptual or hypothetical figures. Keep the two types in separate folders and label them in captions.
