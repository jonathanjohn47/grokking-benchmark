"""First full frozen-protocol score of every stored row for all 8 predictors.

Reads predictor_events.json only. Writes a new JSON file. Does not
overwrite predictor_events.json or lead_fraction_sweep_2026-11-10.json.

Default verdict is c = 0.8 (P_i < 0.8 * G_i). Also stores c = 0.7, c = 0.9,
and the margin-100 comparison. L2, Dropout, Spectral and AGE had no
frozen-protocol verdict on file before this run; these numbers are their
first ones. Every stored row is scored, including exploratory rows.
"""

import json
import os

from rescore_lead_fraction import REPO, RULES, analysis_paths, resolve_results_dir, score_one


def all_eight_out(results_dir=None):
    analysis = os.path.join(resolve_results_dir(results_dir), "analysis")
    return os.path.join(analysis, "lead_fraction_all_eight_2026-11-10.json")

PREFIXES = (
    "L2:",
    "Dropout:",
    "Spectral:",
    "AGE:",
    "HTSR:",
    "Weight-PCA:",
    "Higher-MI:",
    "Commutator Defect:",
)
FIRST_VERDICT = ("L2:", "Dropout:", "Spectral:", "AGE:")


def family(signal):
    for prefix in PREFIXES:
        if signal.startswith(prefix):
            return prefix[:-1]
    return None


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Score every stored row for all eight predictors.")
    parser.add_argument(
        "--results-dir", default=None,
        help="Results folder. Default: 08_Experiments/results/results_per_nanda_power. "
             "Pass 08_Experiments/results/nanda_unified for the legacy run.",
    )
    args = parser.parse_args(argv)
    events_path, _sweep = analysis_paths(args.results_dir)
    out_path = all_eight_out(args.results_dir)
    with open(events_path) as handle:
        events = json.load(handle)
    grok = events["grok_epochs"]
    rows = []
    for row in events["signals"]:
        name = row["signal"]
        group = family(name)
        if group is None:
            continue
        per_rule = [score_one(grok, row["epochs"], spec) for spec in RULES]
        by_name = {item["rule_name"]: item["passed"] for item in per_rule}
        rows.append({
            "predictor": group,
            "signal": name,
            "first_frozen_protocol_verdict": name.startswith(FIRST_VERDICT),
            "G_i": grok,
            "P_i": row["epochs"],
            "rules": per_rule,
            "verdict_changes_at_c_0.7": by_name["c=0.7"] != by_name["c=0.8"],
            "verdict_changes_at_c_0.9": by_name["c=0.9"] != by_name["c=0.8"],
        })
    payload = {
        "date": "2026-11-10",
        "source": os.path.relpath(events_path, REPO),
        "default_c": 0.8,
        "note": (
            "All stored rows for the eight predictors, scored from saved "
            "P_i and G_i. No retraining. Older result files were not overwritten. "
            "L2, Dropout, Spectral and AGE did not have a frozen-protocol "
            "verdict before this file; these are their first ones. "
            "Lead test: P_i < c * G_i, default c = 0.8. Margin-100 is a comparison."
        ),
        "rows": rows,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {out_path}")
    print(f"{'signal':<62} {'n':>3} {'lead':>8} {'rho':>8} {'mean_r':>8} verdict")
    for row in rows:
        main_rule = next(item for item in row["rules"] if item["rule_name"] == "c=0.8")
        rho = main_rule["rho"]
        rho_s = "nan" if rho != rho else f"{rho:.4f}"
        print(
            f"{row['signal']:<62} {main_rule['n_firing']:>3} "
            f"{main_rule['lead_frac']:>8.4f} {rho_s:>8} "
            f"{main_rule['mean_rel_lead']:>8.4f} "
            f"{'PASS' if main_rule['passed'] else 'FAIL'}"
            f"  d0.7={row['verdict_changes_at_c_0.7']} d0.9={row['verdict_changes_at_c_0.9']}"
        )


if __name__ == "__main__":
    main()
