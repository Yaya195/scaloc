# src/evaluation/metrics.py
#
# Localization evaluation metrics.
# All functions expect (N, 2) arrays of predicted and true positions.

import numpy as np
from typing import Dict


def euclidean_errors(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-sample Euclidean distance error (meters)."""
    return np.linalg.norm(y_pred - y_true, axis=1)


def mean_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(euclidean_errors(y_pred, y_true)))


def median_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.median(euclidean_errors(y_pred, y_true)))


def percentile_error(y_pred: np.ndarray, y_true: np.ndarray, p: float = 75) -> float:
    return float(np.percentile(euclidean_errors(y_pred, y_true), p))


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean(euclidean_errors(y_pred, y_true) ** 2)))


def cep(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 5.0) -> float:
    """Circular Error Probability: fraction of predictions within `threshold` meters."""
    errors = euclidean_errors(y_pred, y_true)
    return float(np.mean(errors <= threshold))


def compute_all_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute all localization metrics at once."""
    errors = euclidean_errors(y_pred, y_true)
    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "p75_error": float(np.percentile(errors, 75)),
        "p90_error": float(np.percentile(errors, 90)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "cep_5m": float(np.mean(errors <= 5.0)),
        "cep_10m": float(np.mean(errors <= 10.0)),
        "max_error": float(np.max(errors)),
        "std_error": float(np.std(errors)),
    }
