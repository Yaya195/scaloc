# src/baselines/knn_baseline.py
#
# k-NN fingerprinting baseline (spec §8).
# Uses raw RSSI vectors, matches by Euclidean distance in RSSI space,
# predicts position as weighted average of k nearest RPs.

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, List, Tuple

from src.evaluation.metrics import compute_all_metrics


def build_rssi_vector(aps: List[dict], num_aps: int = 520) -> np.ndarray:
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
    num_aps: int = 520,
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

    for domain_id, val_qs in val_queries.items():
        train_qs = train_queries.get(domain_id, [])
        if not train_qs or not val_qs:
            continue

        # Build RSSI matrices
        X_train = np.array([build_rssi_vector_from_query(q, num_aps) for q in train_qs])
        y_train = np.array([q["pos"] for q in train_qs])
        X_val = np.array([build_rssi_vector_from_query(q, num_aps) for q in val_qs])
        y_val = np.array([q["pos"] for q in val_qs])

        # Fit k-NN
        weights = "distance" if weighted else "uniform"
        knn = KNeighborsRegressor(n_neighbors=min(k, len(X_train)), weights=weights)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        metrics = compute_all_metrics(y_pred, y_val)
        results[domain_id] = metrics

        all_preds.append(y_pred)
        all_truths.append(y_val)

    if all_preds:
        results["global"] = compute_all_metrics(
            np.concatenate(all_preds), np.concatenate(all_truths)
        )

    return results


def build_rssi_vector_from_query(query: dict, num_aps: int = 520) -> np.ndarray:
    """Convert a query dict {ap_ids, rssi, pos} to RSSI vector."""
    vec = np.full(num_aps, -110.0)
    for ap_id, rssi in zip(query["ap_ids"], query["rssi"]):
        if isinstance(ap_id, str):
            ap_id = int(ap_id.replace("WAP", ""))
        if 1 <= ap_id <= num_aps:
            vec[ap_id - 1] = rssi
    return vec
