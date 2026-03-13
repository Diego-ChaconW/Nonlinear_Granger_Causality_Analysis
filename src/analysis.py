"""
Results analysis: statistics computation and figure generation.
"""

import numpy as np
import matplotlib.pyplot as plt

from .config import MAP_DISPLAY_NAMES


def compute_mean_std(data):
    """
    Compute mean and standard deviation, ignoring NaN values.

    Parameters
    ----------
    data : array-like
        Input data.

    Returns
    -------
    mean : float
    std : float
    n_valid : int
        Number of non-NaN values.
    """
    arr = np.asarray(data, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.nan, np.nan, 0
    return np.mean(valid), np.std(valid), len(valid)


def print_summary_table(lag_data, lag_key, chaotic_map, direction_str,
                        architecture, key_nar="RSS_restricted",
                        key_narx="RSS_full"):
    """
    Print a formatted summary table of all metrics.

    Parameters
    ----------
    lag_data : dict
        Dictionary of metric arrays for a specific lag.
    lag_key : str
        Lag key name (e.g., "lag_20").
    chaotic_map : str
        Map name.
    direction_str : str
        Human-readable direction string (e.g., "Y → X").
    architecture : str
        NN architecture name.
    key_nar : str
        Key for restricted model RSS.
    key_narx : str
        Key for full model RSS.
    """
    map_name = MAP_DISPLAY_NAMES.get(chaotic_map, chaotic_map.capitalize())

    print("=" * 60)
    print(f"SUMMARY — {map_name} {direction_str} ({architecture})")
    print(f"Lag: {lag_key}")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Mean':>12} {'± Std':>12} {'N':>6}")
    print("-" * 55)

    for metric_key in lag_data.keys():
        values = lag_data[metric_key]
        mean, std, n = compute_mean_std(values)

        if metric_key == key_nar:
            label = f"{metric_key} (NAR)"
        elif metric_key == key_narx:
            label = f"{metric_key} (NARX)"
        else:
            label = metric_key

        print(f"{label:<25} {mean:>12.4f} {std:>12.4f} {n:>6}")

    print("-" * 55)

    # Causality inference
    p_values = np.asarray(lag_data.get("p_value", []), dtype=float)
    valid_p = p_values[~np.isnan(p_values)]
    if len(valid_p) > 0:
        mean_p = np.mean(valid_p)
        significant_runs = np.sum(valid_p < 0.05)
        pct = significant_runs / len(valid_p) * 100

        d_vals = np.asarray(lag_data.get("cohens_d", []), dtype=float)
        valid_d = d_vals[~np.isnan(d_vals)]
        mean_d = np.mean(valid_d) if len(valid_d) > 0 else np.nan

        print(f"\nCausality Inference ({direction_str}):")
        print(f"  Mean p-value:       {mean_p:.6f}")
        print(f"  Significant runs:   {significant_runs}/{len(valid_p)} "
              f"({pct:.1f}%)")
        print(f"  Mean Cohen's d:     {mean_d:.4f}")

        if pct >= 50:
            print(f"\n  ✅ Evidence SUPPORTS {direction_str} "
                  "Granger causality")
        else:
            print(f"\n  ❌ Evidence DOES NOT support {direction_str} "
                  "Granger causality")

    print("=" * 60)


def plot_rss_histogram(
    rss_nar, rss_narx, chaotic_map, direction, architecture, lag_key,
    nar_label="NAR", narx_label="NARX",
    fig_width=10, fig_height=6, n_bins=30, save_path=None, dpi=300,
):
    """
    Generate a histogram comparing NAR vs NARX RSS distributions.

    Parameters
    ----------
    rss_nar : np.ndarray
        RSS values from the restricted (NAR) model.
    rss_narx : np.ndarray
        RSS values from the full (NARX) model.
    chaotic_map : str
        Map name for the title.
    direction : str
        Causality direction ("Y_to_X" or "X_to_Y").
    architecture : str
        NN architecture name.
    lag_key : str
        Lag key for the title.
    nar_label, narx_label : str
        Legend labels.
    fig_width, fig_height : float
        Figure dimensions.
    n_bins : int
        Number of histogram bins.
    save_path : str or None
        If provided, save the figure to this path.
    dpi : int
        DPI for saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    map_name = MAP_DISPLAY_NAMES.get(chaotic_map, chaotic_map.capitalize())

    if direction == "Y_to_X":
        arrow = "X ← Y"
    elif direction == "X_to_Y":
        arrow = "X → Y"
    else:
        arrow = direction

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.hist(rss_nar, bins=n_bins, histtype="step", linewidth=1.5,
            label=nar_label, color="black")
    ax.hist(rss_narx, bins=n_bins, histtype="step", linewidth=1.5,
            label=narx_label, color="red")

    ax.set_title(
        f"{map_name} chaotic map {arrow}  "
        f"({architecture}, {lag_key.replace('_', '=')})",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlabel("RSS (Residual Sum of Squares)", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig


def plot_pvalue_cohensd(
    p_values, cohens_d, chaotic_map, direction, architecture, lag_key,
    fig_width=14, fig_height=6, n_bins=30, save_path=None, dpi=300,
):
    """
    Generate histograms for p-value and Cohen's d distributions.

    Parameters
    ----------
    p_values : np.ndarray
        p-value array.
    cohens_d : np.ndarray
        Cohen's d array.
    chaotic_map, direction, architecture, lag_key : str
        Labels for the plot title.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    map_name = MAP_DISPLAY_NAMES.get(chaotic_map, chaotic_map.capitalize())
    arrow = "X ← Y" if direction == "Y_to_X" else "X → Y"

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    axes[0].hist(p_values, bins=n_bins, color="#2196F3", alpha=0.8,
                 edgecolor="white")
    axes[0].axvline(x=0.05, color="red", linestyle="--", linewidth=1.5,
                    label="α = 0.05")
    axes[0].set_title("p-value Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("p-value", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].hist(cohens_d, bins=n_bins, color="#FF9800", alpha=0.8,
                 edgecolor="white")
    axes[1].set_title("Cohen's d Distribution", fontsize=14,
                      fontweight="bold")
    axes[1].set_xlabel("Cohen's d (effect size)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"{map_name} {arrow} — {architecture}, "
        f"{lag_key.replace('_', '=')}",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig
