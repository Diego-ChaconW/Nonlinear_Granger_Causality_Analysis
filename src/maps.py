"""
Chaotic map generators for nonlinear Granger causality analysis.

Each function generates a pair of coupled time series (X, Y) from a
two-dimensional discrete-time dynamical system. The coupling structure
determines the true Granger causality relationships between variables.

Supported maps:
    - Hénon map
    - Ikeda map
    - Tinkerbell map
    - Rulkov map
"""

import numpy as np


def generate_henon_map(a=1.4, b=0.3, x0=0.1, y0=0.1, n_iter=500):
    """
    Generate time series from the Hénon map.

    The Hénon map is a 2D discrete-time dynamical system:
        x_{n+1} = 1 - a * x_n^2 + y_n
        y_{n+1} = b * x_n

    Parameters
    ----------
    a : float
        Parameter controlling the quadratic nonlinearity (default: 1.4).
    b : float
        Parameter controlling the coupling (default: 0.3).
    x0, y0 : float
        Initial conditions (default: 0.1, 0.1).
    n_iter : int
        Number of iterations (default: 500).

    Returns
    -------
    x, y : np.ndarray
        Time series of length n_iter.
    """
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    x[0], y[0] = x0, y0
    for i in range(1, n_iter):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x, y


def generate_ikeda_map(u=0.9, x0=0.1, y0=0.1, n_iter=500):
    """
    Generate time series from the Ikeda map.

    The Ikeda map is a 2D discrete-time dynamical system:
        t_n     = 0.4 - 6 / (1 + x_n^2 + y_n^2)
        x_{n+1} = 1 + u * (x_n * cos(t_n) - y_n * sin(t_n))
        y_{n+1} = u * (x_n * sin(t_n) + y_n * cos(t_n))

    Parameters
    ----------
    u : float
        Coupling parameter (default: 0.9).
    x0, y0 : float
        Initial conditions (default: 0.1, 0.1).
    n_iter : int
        Number of iterations (default: 500).

    Returns
    -------
    x, y : np.ndarray
        Time series of length n_iter.
    """
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    x[0], y[0] = x0, y0
    for i in range(1, n_iter):
        t = 0.4 - 6.0 / (1.0 + x[i - 1] ** 2 + y[i - 1] ** 2)
        x[i] = 1.0 + u * (x[i - 1] * np.cos(t) - y[i - 1] * np.sin(t))
        y[i] = u * (x[i - 1] * np.sin(t) + y[i - 1] * np.cos(t))
    return x, y


def generate_tinkerbell_map(
    a=0.9, b=-0.6013, c=2.0, d=0.50, x0=0.1, y0=0.0, n_iter=500
):
    """
    Generate time series from the Tinkerbell map.

    The Tinkerbell map is a 2D discrete-time dynamical system:
        x_{n+1} = x_n^2 - y_n^2 + a * x_n + b * y_n
        y_{n+1} = 2 * x_n * y_n + c * x_n + d * y_n

    Parameters
    ----------
    a, b, c, d : float
        Map parameters (defaults: 0.9, -0.6013, 2.0, 0.50).
    x0, y0 : float
        Initial conditions (default: 0.1, 0.0).
    n_iter : int
        Number of iterations (default: 500).

    Returns
    -------
    x, y : np.ndarray
        Time series of length n_iter.
    """
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    x[0], y[0] = x0, y0
    for i in range(1, n_iter):
        x[i] = x[i - 1] ** 2 - y[i - 1] ** 2 + a * x[i - 1] + b * y[i - 1]
        y[i] = 2 * x[i - 1] * y[i - 1] + c * x[i - 1] + d * y[i - 1]
    return x, y


def generate_rulkov_map(alpha=4.1, sigma=0.001, beta=0.001, n_iter=500):
    """
    Generate time series from the Rulkov map.

    The Rulkov map models neuronal bursting behavior:
        x_{n+1} = alpha / (1 + x_n^2) + y_n
        y_{n+1} = y_n - sigma * (x_n - beta)

    Parameters
    ----------
    alpha : float
        Controls fast dynamics amplitude (default: 4.1).
    sigma : float
        Controls slow dynamics time scale (default: 0.001).
    beta : float
        Offset parameter (default: 0.001).
    n_iter : int
        Number of iterations (default: 500).

    Returns
    -------
    x, y : np.ndarray
        Time series of length n_iter.
    """
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    x[0] = 0.0
    y[0] = -2.0
    for i in range(1, n_iter):
        x[i] = alpha / (1.0 + x[i - 1] ** 2) + y[i - 1]
        y[i] = y[i - 1] - sigma * (x[i - 1] - beta)
    return x, y


# Registry of available map generators
MAP_GENERATORS = {
    "henon": generate_henon_map,
    "ikeda": generate_ikeda_map,
    "tinkerbell": generate_tinkerbell_map,
    "rulkov": generate_rulkov_map,
}
