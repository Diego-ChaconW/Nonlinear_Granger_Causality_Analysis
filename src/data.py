"""
Data generation, normalization, and splitting utilities.

This module provides the common data pipeline shared across the
hyperparameter search and experiment notebooks:
    1. Generate time series from a chaotic map
    2. Arrange columns by causality direction
    3. Normalize to [-1, 1]
    4. Split into train / validation / test sets
"""

import numpy as np

from .maps import MAP_GENERATORS
from .config import validate_map, validate_direction


def generate_data(chaotic_map, direction, n_iterations=500):
    """
    Generate and arrange time series data for Granger causality testing.

    The nonlincausality library expects a 2-column matrix [col_0, col_1]
    and tests whether col_1 Granger-causes col_0:
        - Y_to_X: data = [X, Y]  (Y potentially causes X)
        - X_to_Y: data = [Y, X]  (X potentially causes Y)

    Parameters
    ----------
    chaotic_map : str
        Name of the chaotic map ("henon", "ikeda", "tinkerbell", "rulkov").
    direction : str
        Causality direction ("Y_to_X" or "X_to_Y").
    n_iterations : int
        Length of the generated time series (default: 500).

    Returns
    -------
    data : np.ndarray, shape (n_iterations, 2)
        Data matrix arranged for the specified causality direction.
    """
    validate_map(chaotic_map)
    validate_direction(direction)

    generator = MAP_GENERATORS[chaotic_map]
    x_series, y_series = generator(n_iter=n_iterations)

    min_length = min(len(x_series), len(y_series))

    if direction == "Y_to_X":
        data = np.vstack([x_series[:min_length], y_series[:min_length]]).T
    else:  # X_to_Y
        data = np.vstack([y_series[:min_length], x_series[:min_length]]).T

    return data


def normalize_data(data):
    """
    Normalize each column of the data matrix to the interval [-1, 1].

    This follows the methodology: "the time series were normalized to the
    interval [-1, 1] before applying the methods used in the analysis."

    Parameters
    ----------
    data : np.ndarray, shape (n, 2)
        Input data matrix.

    Returns
    -------
    data : np.ndarray, shape (n, 2)
        Normalized data matrix (modified in-place and returned).
    """
    for col in range(data.shape[1]):
        col_min = data[:, col].min()
        col_max = data[:, col].max()
        if col_max - col_min > 0:
            data[:, col] = (
                2 * (data[:, col] - col_min) / (col_max - col_min) - 1
            )
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.2):
    """
    Split data into training, validation, and test sets.

    Parameters
    ----------
    data : np.ndarray, shape (n, 2)
        Full data matrix.
    train_ratio : float
        Fraction for training (default: 0.7).
    val_ratio : float
        Fraction for validation (default: 0.2).
        Test fraction = 1 - train_ratio - val_ratio.

    Returns
    -------
    data_train, data_val, data_test : np.ndarray
        The three splits.
    """
    n_samples = len(data)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)

    data_train = data[:train_size, :]
    data_val = data[train_size:train_size + val_size, :]
    data_test = data[train_size + val_size:, :]

    return data_train, data_val, data_test
