# src/baselines/knn_baseline.py
#
# k-NN fingerprinting baseline (spec §8).
# Uses raw RSSI vectors, matches by Euclidean distance in RSSI space,
# predicts position as weighted average of k nearest RPs.

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, List, Tuple

from src.data.normalization import (
    denormalize_coords,
    fit_domain_normalization_stats,
    normalize_coords,
    normalize_rssi_matrix,
)
from src.evaluation.metrics import compute_all_metrics


def _infer_num_aps(train_queries: Dict[str, List[dict]], val_queries: Dict[str, List[dict]]) -> int:
    max_ap_id = 0
    for domain_queries in list(train_queries.values()) + list(val_queries.values()):
        for q in domain_queries:
            for ap_id in q.get("ap_ids", []):
                if isinstance(ap_id, str):
                    ap_id = int(ap_id.replace("WAP", ""))
                max_ap_id = max(max_ap_id, int(ap_id))
    return max_ap_id if max_ap_id > 0 else 520


def build_rssi_vector(aps: List[dict], num_aps: int) -> np.ndarray:
    """Convert AP-wise fingerprint to fixed-length RSSI vector."""
    vec = np.full(num_aps, -110.0)  # default = very weak
    for ap in aps:
        ap_id = ap["ap_id"]
        if isinstance(ap_id, str):
            ap_id = int(ap_id.replace("WAP", ""))
        if 1 <= ap_id <= num_aps:
            vec[ap_id - 1] = ap["rssi"]
    return vec


def run_knn_baseline(
    train_queries: Dict[str, List[dict]],
    val_queries: Dict[str, List[dict]],
    k: int = 5,
    num_aps: int = None,
    weighted: bool = True,
) -> Dict[str, dict]:
    """
    Run k-NN fingerprinting baseline per domain and globally.

    Args:
        train_queries: {domain_id: list of query dicts with ap_ids, rssi, pos}
        val_queries: {domain_id: list of query dicts}
        k: number of neighbors
        num_aps: total AP count for vector size
        weighted: use distance-weighted averaging

    Returns:
        {domain_id: metrics_dict, "global": metrics_dict}
    """
    all_preds = []
    all_truths = []
    results = {}
    domain_stats = fit_domain_normalization_stats(train_queries)

    if num_aps is None:
        num_aps = _infer_num_aps(train_queries, val_queries)

    for domain_id, val_qs in val_queries.items():
        train_qs = train_queries.get(domain_id, [])
        if not train_qs or not val_qs:
            continue

        # Build RSSI matrices
        X_train = np.array([build_rssi_vector_from_query(q, num_aps) for q in train_qs])
        y_train_raw = np.array([q["pos"] for q in train_qs], dtype=np.float32)
        X_val = np.array([build_rssi_vector_from_query(q, num_aps) for q in val_qs])
        y_val_raw = np.array([q["pos"] for q in val_qs], dtype=np.float32)

        stats = domain_stats.get(domain_id)
        if stats is not None:
            X_train = normalize_rssi_matrix(X_train, stats["rssi_min"], stats["rssi_max"])
            X_val = normalize_rssi_matrix(X_val, stats["rssi_min"], stats["rssi_max"])
            y_train = normalize_coords(y_train_raw, stats["coord_min"], stats["coord_max"])
        else:
            y_train = y_train_raw

        # Fit k-NN
        weights = "distance" if weighted else "uniform"
        knn = KNeighborsRegressor(n_neighbors=min(k, len(X_train)), weights=weights)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        if stats is not None:
            y_pred = denormalize_coords(y_pred, stats["coord_min"], stats["coord_max"])

        metrics = compute_all_metrics(y_pred, y_val_raw)
        results[domain_id] = metrics

        all_preds.append(y_pred)
        all_truths.append(y_val_raw)

    if all_preds:
        results["global"] = compute_all_metrics(
            np.concatenate(all_preds), np.concatenate(all_truths)
        )

    return results


def build_rssi_vector_from_query(query: dict, num_aps: int) -> np.ndarray:
    """Convert a query dict {ap_ids, rssi, pos} to RSSI vector."""
    vec = np.full(num_aps, -110.0)
    for ap_id, rssi in zip(query["ap_ids"], query["rssi"]):
        if isinstance(ap_id, str):
            ap_id = int(ap_id.replace("WAP", ""))
        if 1 <= ap_id <= num_aps:
            vec[ap_id - 1] = rssi
    return vec
