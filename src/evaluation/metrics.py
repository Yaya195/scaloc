import numpy as np


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute root mean square error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
