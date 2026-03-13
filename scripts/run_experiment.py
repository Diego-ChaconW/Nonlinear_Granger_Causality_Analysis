#!/usr/bin/env python
"""
Step 2: Run the Granger causality experiment with N initializations.

Uses fixed hyperparameters and runs the causality test N times to address
the neural network initialization problem. Saves intermediate checkpoints
and a final .pkl with all metrics.

Usage:
    python scripts/run_experiment.py \\
        --map henon --direction Y_to_X --arch GRU \\
        --neurons 100 --lag 5 --batch-size 16 --runs 100 \\
        --output results/experiments/
"""

import argparse
import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src  # noqa: E402
from src.config import (
    validate_map, validate_direction, validate_architecture,
    get_direction_labels,
)
from src.data import generate_data, normalize_data, split_data
from src.causality import run_causality_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nonlinear Granger causality experiment with "
                    "multiple initializations."
    )
    parser.add_argument("--map", required=True,
                        choices=["henon", "ikeda", "tinkerbell", "rulkov"])
    parser.add_argument("--direction", required=True,
                        choices=["Y_to_X", "X_to_Y"])
    parser.add_argument("--arch", required=True,
                        choices=["MLP", "LSTM", "GRU"])
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of initializations (default: 100).")
    parser.add_argument("--batch-runs", type=int, default=10,
                        help="Runs per checkpoint save (default: 10).")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output", default="results/experiments/")
    return parser.parse_args()


def main():
    args = parse_args()

    validate_map(args.map)
    validate_direction(args.direction)
    validate_architecture(args.arch)

    labels = get_direction_labels(args.direction)

    # Generate and prepare data
    data = generate_data(args.map, args.direction, args.n_iter)
    data = normalize_data(data)
    data_train, data_val, data_test = split_data(
        data, args.train_ratio, args.val_ratio
    )

    # Setup
    os.makedirs(args.output, exist_ok=True)
    prefix = f"results_{args.map}_{args.arch}_{args.direction}"

    n_batches = args.runs // args.batch_runs
    if args.runs % args.batch_runs > 0:
        n_batches += 1

    p_values = np.zeros(args.runs)
    test_statistics = np.zeros(args.runs)
    rss_restricted = np.zeros(args.runs)
    rss_full = np.zeros(args.runs)
    cohens_d = np.zeros(args.runs)

    print("=" * 60)
    print(f"EXPERIMENT: {prefix}")
    print(f"Runs: {args.runs} | Batches: {n_batches}")
    print("=" * 60)

    start_time = time.time()

    for batch in range(n_batches):
        start_idx = batch * args.batch_runs
        end_idx = min(start_idx + args.batch_runs, args.runs)

        print(f"\nBatch {batch + 1}/{n_batches} "
              f"(runs {start_idx + 1} to {end_idx})")

        for i in range(start_idx, end_idx):
            print(f"  Run {i + 1}/{args.runs}...", end=" ")

            metrics, error_msg = run_causality_test(
                data_train, data_val, data_test,
                lag=args.lag, architecture=args.arch, neurons=args.neurons,
                epochs=args.epochs, learning_rate=args.lr,
                batch_size=args.batch_size,
            )

            if metrics is not None:
                p_values[i] = metrics["p_value"]
                test_statistics[i] = metrics["test_statistic"]
                rss_restricted[i] = metrics["RSS_restricted"]
                rss_full[i] = metrics["RSS_full"]
                cohens_d[i] = metrics["cohens_d"]
                print(f"p={p_values[i]:.4f}, d={cohens_d[i]:.4f}")
            else:
                p_values[i] = np.nan
                test_statistics[i] = np.nan
                rss_restricted[i] = np.nan
                rss_full[i] = np.nan
                cohens_d[i] = np.nan
                print(f"ERROR: {error_msg}")

        # Checkpoint
        cp_path = os.path.join(
            args.output, f"{prefix}_batch_{batch + 1}.pkl"
        )
        cp_data = {
            "config": {"chaotic_map": args.map,
                       "causality_direction": args.direction,
                       "nn_architecture": args.arch,
                       "nn_neurons": [args.neurons], "max_lag": args.lag},
            f"lag_{args.lag}": {
                "p_value": p_values[:end_idx].copy(),
                "test_statistic": test_statistics[:end_idx].copy(),
                "RSS_restricted": rss_restricted[:end_idx].copy(),
                "RSS_full": rss_full[:end_idx].copy(),
                "cohens_d": cohens_d[:end_idx].copy(),
            },
        }
        with open(cp_path, "wb") as f:
            pickle.dump(cp_data, f)
        print(f"  → Checkpoint: {cp_path}")

    elapsed = time.time() - start_time

    # Final results
    final = {
        "config": {
            "chaotic_map": args.map,
            "causality_direction": args.direction,
            "direction_str": labels["direction_str"],
            "target_variable": labels["target_label"],
            "cause_variable": labels["cause_label"],
            "nn_architecture": args.arch,
            "nn_neurons": [args.neurons],
            "max_lag": args.lag,
            "batch_size_nn": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "n_iterations": args.n_iter,
            "n_runs": args.runs,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": elapsed,
        },
        f"lag_{args.lag}": {
            "p_value": p_values,
            "test_statistic": test_statistics,
            "RSS_restricted": rss_restricted,
            "RSS_full": rss_full,
            "cohens_d": cohens_d,
        },
    }

    final_path = os.path.join(args.output, f"{prefix}_final.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(final, f)

    # Cleanup checkpoints
    for batch in range(1, n_batches + 1):
        cp = os.path.join(args.output, f"{prefix}_batch_{batch}.pkl")
        try:
            os.remove(cp)
        except OSError:
            pass

    print(f"\nFinal results: {final_path}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
