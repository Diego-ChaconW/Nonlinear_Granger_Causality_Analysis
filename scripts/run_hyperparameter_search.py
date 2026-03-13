#!/usr/bin/env python
"""
Step 1: Hyperparameter grid search.

Evaluates all combinations of NN architecture, neurons, lag, and batch size
for a specified chaotic map and causality direction. Outputs a .pkl file
with the best hyperparameter configuration.

Usage:
    python scripts/run_hyperparameter_search.py \\
        --map henon --direction Y_to_X --arch GRU \\
        --output results/hyperparameters/
"""

import argparse
import os
import sys
import time
import pickle
from datetime import datetime
from itertools import product

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src  # noqa: E402 — sets TF_USE_LEGACY_KERAS
from src.config import (
    NN_CONFIG_MAP, validate_map, validate_direction, validate_architecture,
    get_direction_labels,
)
from src.data import generate_data, normalize_data, split_data
from src.causality import run_causality_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for nonlinear Granger "
                    "causality."
    )
    parser.add_argument("--map", required=True,
                        choices=["henon", "ikeda", "tinkerbell", "rulkov"],
                        help="Chaotic map to use.")
    parser.add_argument("--direction", required=True,
                        choices=["Y_to_X", "X_to_Y"],
                        help="Causality direction to test.")
    parser.add_argument("--arch", nargs="+", default=["GRU"],
                        help="NN architectures to try (default: GRU).")
    parser.add_argument("--neurons", nargs="+", type=int,
                        default=[10, 50, 100],
                        help="Neuron counts to try (default: 10 50 100).")
    parser.add_argument("--lags", nargs="+", type=int, default=[5, 10, 20],
                        help="Lag values to try (default: 5 10 20).")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32],
                        help="Batch sizes to try (default: 16 32).")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50).")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default: 0.0001).")
    parser.add_argument("--n-iter", type=int, default=500,
                        help="Time series length (default: 500).")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output", default="results/hyperparameters/",
                        help="Output directory for .pkl files.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate
    validate_map(args.map)
    validate_direction(args.direction)
    for a in args.arch:
        validate_architecture(a)

    labels = get_direction_labels(args.direction)

    # Generate and prepare data
    data = generate_data(args.map, args.direction, args.n_iter)
    data = normalize_data(data)
    data_train, data_val, data_test = split_data(
        data, args.train_ratio, args.val_ratio
    )

    # Grid search
    combos = list(product(
        args.arch, args.neurons, args.lags, args.batch_sizes
    ))
    total = len(combos)

    print("=" * 60)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 60)
    print(f"  Map: {args.map} | Direction: {labels['direction_str']}")
    print(f"  Architectures: {args.arch}")
    print(f"  Total combinations: {total}")
    print("=" * 60)

    best_error = float("inf")
    best_config = None
    all_results = {}

    start_time = time.time()

    for idx, (arch, neurons, lag, bs) in enumerate(combos, 1):
        print(f"Config {idx}/{total}: Arch={arch}, Neurons={neurons}, "
              f"Lag={lag}, BS={bs}")

        metrics, error_msg = run_causality_test(
            data_train, data_val, data_test,
            lag=lag, architecture=arch, neurons=neurons,
            epochs=args.epochs, learning_rate=args.lr, batch_size=bs,
        )

        result_entry = {
            "nn_architecture": arch, "neurons": neurons, "lag": lag,
            "batch_size": bs, "epochs": args.epochs,
            "learning_rate": args.lr,
        }

        if metrics is not None:
            result_entry.update({
                "p_value": metrics["p_value"],
                "test_statistic": metrics["test_statistic"],
                "RSS_restricted": metrics["RSS_restricted"],
                "RSS_full": metrics["RSS_full"],
                "cohens_d": metrics["cohens_d"],
                "total_error": metrics["total_error"],
                "error": "none",
            })
            # Only consider significant results (p < 0.05)
            if (metrics["p_value"] < 0.05
                    and metrics["total_error"] < best_error):
                best_error = metrics["total_error"]
                best_config = result_entry.copy()
                best_config["config_idx"] = idx
            print(f"  → p={metrics['p_value']:.4f}, "
                  f"error={metrics['total_error']:.6f}")
        else:
            result_entry["error"] = error_msg
            print(f"  → ERROR: {error_msg}")

        all_results[idx] = result_entry

    elapsed = time.time() - start_time

    # Save results
    os.makedirs(args.output, exist_ok=True)
    filename = (f"hyperparameters_{args.map}_"
                f"{args.direction}.pkl")
    filepath = os.path.join(args.output, filename)

    final = {
        "config": {
            "chaotic_map": args.map,
            "causality_direction": args.direction,
            "direction_str": labels["direction_str"],
            "target_variable": labels["target_label"],
            "cause_variable": labels["cause_label"],
            "nn_architectures": args.arch,
            "neurons_options": args.neurons,
            "lag_options": args.lags,
            "batch_size_options": args.batch_sizes,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "n_iterations": args.n_iter,
            "total_combinations": total,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": elapsed,
        },
        "all_results": all_results,
        "best_config": best_config,
    }

    with open(filepath, "wb") as f:
        pickle.dump(final, f)

    print(f"\nResults saved: {filepath}")
    print(f"Elapsed: {elapsed:.1f}s")

    if best_config:
        print(f"\nBest config: Arch={best_config['nn_architecture']}, "
              f"Neurons={best_config['neurons']}, "
              f"Lag={best_config['lag']}, "
              f"BS={best_config['batch_size']}, "
              f"Error={best_config['total_error']:.6f}")
    else:
        print("\nNo configuration with p < 0.05 found.")


if __name__ == "__main__":
    main()
