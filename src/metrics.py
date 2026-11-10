
import numpy as np


def compute_rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_pred = np.maximum(y_pred, 0)

    return np.sqrt(
        np.mean(
            (np.log1p(y_true) - np.log1p(y_pred)) ** 2
        )
    )


def compute_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))
