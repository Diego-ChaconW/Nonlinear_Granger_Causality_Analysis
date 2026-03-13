#!/usr/bin/env python
"""
Step 3: Analyze experiment results.

Loads a .pkl file from Step 2, computes summary statistics (mean ± std),
and generates histograms (NAR vs NARX, p-value, Cohen's d).

Usage:
    python scripts/run_analysis.py \\
        --pkl results/experiments/results_henon_GRU_Y_to_X_final.pkl \\
        --output results/figures/
"""

import argparse
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (  # noqa: E402
    compute_mean_std, print_summary_table,
    plot_rss_histogram, plot_pvalue_cohensd,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results and generate figures."
    )
    parser.add_argument("--pkl", required=True,
                        help="Path to the _final.pkl results file.")
    parser.add_argument("--output", default="results/figures/",
                        help="Output directory for figures.")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save figures to disk.")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def detect_rss_keys(lag_data):
    """Detect which key naming convention the .pkl uses."""
    if "RSS_restricted" in lag_data:
        return "RSS_restricted", "RSS_full"
    elif "RSS_X" in lag_data:
        return "RSS_X", "RSS_XY"
    elif "RSS_Y" in lag_data:
        return "RSS_Y", "RSS_YX"
    else:
        raise ValueError(
            f"Cannot detect RSS keys. Available: {list(lag_data.keys())}"
        )


def main():
    args = parse_args()

    if not os.path.exists(args.pkl):
        print(f"ERROR: File not found: {args.pkl}")
        sys.exit(1)

    with open(args.pkl, "rb") as f:
        results = pickle.load(f)

    config = results.get("config", {})
    chaotic_map = config.get("chaotic_map", "unknown")
    direction = config.get("causality_direction", "unknown")
    direction_str = config.get("direction_str", "")
    target_var = config.get("target_variable", "")
    cause_var = config.get("cause_variable", "")
    architecture = config.get("nn_architecture", "unknown")

    # Find lag key
    lag_keys = [k for k in results if k.startswith("lag_")]
    if not lag_keys:
        print("ERROR: No lag data found in .pkl")
        sys.exit(1)

    lag_key = lag_keys[0]
    lag_data = results[lag_key]

    key_nar, key_narx = detect_rss_keys(lag_data)
    nar_label = f"NAR ({target_var} only)" if target_var else "NAR"
    narx_label = (f"NARX ({target_var} + {cause_var})"
                  if target_var else "NARX")

    # Print summary
    print_summary_table(
        lag_data, lag_key, chaotic_map, direction_str, architecture,
        key_nar, key_narx,
    )

    # Generate figures
    os.makedirs(args.output, exist_ok=True)

    rss_nar = np.asarray(lag_data.get(key_nar, []), dtype=float)
    rss_narx = np.asarray(lag_data.get(key_narx, []), dtype=float)
    rss_nar = rss_nar[~np.isnan(rss_nar)]
    rss_narx = rss_narx[~np.isnan(rss_narx)]

    if len(rss_nar) > 0 and len(rss_narx) > 0:
        save_path = None if args.no_save else os.path.join(
            args.output,
            f"histogram_{chaotic_map}_{architecture}_{direction}_{lag_key}.png"
        )
        plot_rss_histogram(
            rss_nar, rss_narx, chaotic_map, direction, architecture, lag_key,
            nar_label=nar_label, narx_label=narx_label,
            save_path=save_path, dpi=args.dpi,
        )

        # NAR vs NARX comparison
        nar_m, nar_s, _ = compute_mean_std(rss_nar)
        narx_m, narx_s, _ = compute_mean_std(rss_narx)
        print(f"\n{key_nar} (NAR):  {nar_m:.4f} ± {nar_s:.4f}")
        print(f"{key_narx} (NARX): {narx_m:.4f} ± {narx_s:.4f}")
        if narx_m < nar_m:
            reduction = (1 - narx_m / nar_m) * 100
            print(f"NARX reduces RSS by {reduction:.1f}%")

    # p-value and Cohen's d histograms
    p_vals = np.asarray(lag_data.get("p_value", []), dtype=float)
    d_vals = np.asarray(lag_data.get("cohens_d", []), dtype=float)
    p_vals = p_vals[~np.isnan(p_vals)]
    d_vals = d_vals[~np.isnan(d_vals)]

    if len(p_vals) > 0 and len(d_vals) > 0:
        save_path2 = None if args.no_save else os.path.join(
            args.output,
            f"histogram_stats_{chaotic_map}_{architecture}_{direction}_"
            f"{lag_key}.png"
        )
        plot_pvalue_cohensd(
            p_vals, d_vals, chaotic_map, direction, architecture, lag_key,
            save_path=save_path2, dpi=args.dpi,
        )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
