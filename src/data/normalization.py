from typing import Dict, List

import math
import numpy as np


EPS = 1e-6
BETA = math.e


def _safe_range(max_v: np.ndarray, min_v: np.ndarray) -> np.ndarray:
    return np.maximum(max_v - min_v, EPS)


def fit_domain_normalization_stats(train_queries: Dict[str, List[dict]]) -> Dict[str, dict]:
    """
    Fit per-domain normalization stats from training queries only.

    Returns:
        {
          domain_id: {
            "rssi_min": float,
            "rssi_max": float,
            "coord_min": np.ndarray shape (2,),
            "coord_max": np.ndarray shape (2,),
          }
        }
    """
    stats = {}

    for domain_id, queries in train_queries.items():
        if not queries:
            continue

        rssi_vals = [float(v) for q in queries for v in q.get("rssi", [])]
        if not rssi_vals:
            rssi_min = -110.0
            rssi_max = -30.0
        else:
            rssi_min = float(min(rssi_vals))
            rssi_max = float(max(rssi_vals))
            if abs(rssi_max - rssi_min) < EPS:
                rssi_max = rssi_min + 1.0

        coords = np.array([q["pos"] for q in queries], dtype=np.float32)
        coord_min = coords.min(axis=0)
        coord_max = coords.max(axis=0)

        stats[domain_id] = {
            "rssi_min": rssi_min,
            "rssi_max": rssi_max,
            "coord_min": coord_min,
            "coord_max": coord_max,
        }

    return stats


def fit_global_normalization_stats(train_queries: Dict[str, List[dict]]) -> dict:
    """
    Fit one global normalization stat set from aggregated training queries.

    Intended for centralized protocols trained on pooled selected domains.
    """
    all_queries = []
    for queries in train_queries.values():
        all_queries.extend(queries)

    if not all_queries:
        return {
            "rssi_min": -110.0,
            "rssi_max": -30.0,
            "coord_min": np.array([0.0, 0.0], dtype=np.float32),
            "coord_max": np.array([1.0, 1.0], dtype=np.float32),
        }

    rssi_vals = [float(v) for q in all_queries for v in q.get("rssi", [])]
    if not rssi_vals:
        rssi_min = -110.0
        rssi_max = -30.0
    else:
        rssi_min = float(min(rssi_vals))
        rssi_max = float(max(rssi_vals))
        if abs(rssi_max - rssi_min) < EPS:
            rssi_max = rssi_min + 1.0

    coords = np.array([q["pos"] for q in all_queries], dtype=np.float32)
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)

    return {
        "rssi_min": rssi_min,
        "rssi_max": rssi_max,
        "coord_min": coord_min,
        "coord_max": coord_max,
    }


def normalize_rssi_values(values, rssi_min: float, _rssi_max: float = None):
    arr = np.asarray(values, dtype=np.float32)
    denom = max(-float(rssi_min), EPS)
    base = (arr - float(rssi_min)) / denom
    base = np.maximum(base, 0.0)
    return np.power(base, BETA)


def normalize_rssi_matrix(matrix: np.ndarray, rssi_min: float, _rssi_max: float = None) -> np.ndarray:
    denom = max(-float(rssi_min), EPS)
    base = (matrix.astype(np.float32) - float(rssi_min)) / denom
    base = np.maximum(base, 0.0)
    return np.power(base, BETA)


def normalize_coords(coords: np.ndarray, coord_min: np.ndarray, coord_max: np.ndarray) -> np.ndarray:
    return (coords.astype(np.float32) - coord_min.astype(np.float32)) / _safe_range(
        coord_max.astype(np.float32), coord_min.astype(np.float32)
    )


def denormalize_coords(coords_norm: np.ndarray, coord_min: np.ndarray, coord_max: np.ndarray) -> np.ndarray:
    return coords_norm.astype(np.float32) * _safe_range(
        coord_max.astype(np.float32), coord_min.astype(np.float32)
    ) + coord_min.astype(np.float32)
