"""
Core nonlinear Granger causality testing wrapper.

Wraps the nonlincausality library with proper error handling,
memory cleanup, and metric extraction.
"""

import gc

import numpy as np
import nonlincausality as nlc

from .config import NN_CONFIG_MAP, validate_architecture


def run_causality_test(
    data_train,
    data_val,
    data_test,
    lag,
    architecture,
    neurons,
    epochs=50,
    learning_rate=0.0001,
    batch_size=16,
    run=1,
    verbose=False,
    plot=False,
):
    """
    Run a single nonlinear Granger causality test.

    Parameters
    ----------
    data_train : np.ndarray
        Training data, shape (n_train, 2).
    data_val : np.ndarray
        Validation data, shape (n_val, 2).
    data_test : np.ndarray
        Test data, shape (n_test, 2).
    lag : int
        Number of past time steps to consider.
    architecture : str
        Neural network type: "MLP", "LSTM", or "GRU".
    neurons : int or list
        Number of neurons in the hidden layer.
    epochs : int
        Number of training epochs (default: 50).
    learning_rate : float
        Optimizer learning rate (default: 0.0001).
    batch_size : int
        Mini-batch size (default: 16).
    run : int
        Number of random initializations per call (default: 1).
    verbose : bool
        Print training progress (default: False).
    plot : bool
        Show training plots (default: False).

    Returns
    -------
    dict
        Dictionary with keys: p_value, test_statistic, errors_restricted,
        errors_full, RSS_restricted, RSS_full, cohens_d, total_error.
        Returns None if an error occurs.
    error_msg : str or None
        Error message if the test failed, None otherwise.
    """
    validate_architecture(architecture)
    nn_code = NN_CONFIG_MAP[architecture]

    if isinstance(neurons, int):
        neurons = [neurons]

    try:
        results = nlc.nonlincausalityNN(
            x=data_train,
            maxlag=lag,
            NN_config=[nn_code],
            NN_neurons=neurons,
            x_test=data_test,
            run=run,
            epochs_num=[epochs],
            learning_rate=[learning_rate],
            batch_size_num=batch_size,
            x_val=data_val,
            reg_alpha=None,
            callbacks=None,
            verbose=verbose,
            plot=plot,
        )

        result = results[lag]

        errors_restricted = result.best_errors_X
        errors_full = result.best_errors_XY

        total_error = (
            np.mean(np.abs(errors_restricted))
            + np.mean(np.abs(errors_full))
        )

        cohens_d = np.abs(
            (np.mean(np.abs(errors_restricted)) - np.mean(np.abs(errors_full)))
            / np.std([errors_restricted, errors_full])
        )

        metrics = {
            "p_value": result.p_value,
            "test_statistic": result.test_statistic,
            "errors_restricted": errors_restricted,
            "errors_full": errors_full,
            "RSS_restricted": getattr(result, "_best_RSS_X", np.nan),
            "RSS_full": getattr(result, "_best_RSS_XY", np.nan),
            "cohens_d": cohens_d,
            "total_error": total_error,
        }

        del results
        gc.collect()

        return metrics, None

    except Exception as e:
        gc.collect()
        return None, str(e)
