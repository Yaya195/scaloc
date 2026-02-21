# src/data/build_graphs.py

from pathlib import Path
from typing import List, Tuple, Set
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

RPS_DIR = Path("data/processed/rps")
GRAPHS_DIR = Path("data/processed/graphs")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def list_train_rp_files() -> List[Path]:
    # Look for new parquet files first, fall back to pickle for backward compatibility
    parquet_files = sorted(RPS_DIR.glob("train_rps_*.parquet"))
    if parquet_files:
        return parquet_files
    return sorted(RPS_DIR.glob("train_rps_*.pkl"))


def adaptive_k(n_rps: int) -> int:
    if n_rps <= 3:
        return max(1, n_rps - 1)
    return max(3, int(round(np.log(n_rps))) + 2)


def build_knn_graph(coords: np.ndarray) -> List[Tuple[int, int, float]]:
    if len(coords) == 0:
        return []

    k = adaptive_k(len(coords))

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edges = set()
    for i in range(len(coords)):
        for j_idx, d in zip(indices[i][1:], distances[i][1:]):
            j = int(j_idx)
            d = float(d)
            edges.add((i, j, d))
            edges.add((j, i, d))

    return list(edges)


def get_components(n_nodes: int, edges: List[Tuple[int, int, float]]) -> List[Set[int]]:
    adj = {i: set() for i in range(n_nodes)}
    for u, v, _ in edges:
        adj[u].add(v)
        adj[v].add(u)

    visited = set()
    components = []

    for node in range(n_nodes):
        if node not in visited:
            stack = [node]
            comp = set()
            while stack:
                x = stack.pop()
                if x not in visited:
                    visited.add(x)
                    comp.add(x)
                    stack.extend(adj[x] - visited)
            components.append(comp)

    return components


def connect_components(coords: np.ndarray, edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    n_nodes = len(coords)
    components = get_components(n_nodes, edges)

    if len(components) <= 1:
        return edges

    edges_set = set(edges)

    while len(components) > 1:
        comp_a = components[0]
        comp_b = components[1]

        best_dist = float("inf")
        best_pair = None

        for i in comp_a:
            for j in comp_b:
                d = np.linalg.norm(coords[i] - coords[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j, float(d))

        i, j, d = best_pair
        edges_set.add((i, j, d))
        edges_set.add((j, i, d))

        components = get_components(n_nodes, list(edges_set))

    return list(edges_set)


def to_json_serializable(obj):
    """Recursively convert numpy arrays and types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


def main():
    for rp_path in list_train_rp_files():
        # Read from new parquet format
        rps_df = pd.read_parquet(rp_path)

        coords = rps_df[["x", "y"]].to_numpy()

        edges = build_knn_graph(coords)
        edges = connect_components(coords, edges)

        graph = {
            "rp_ids": rps_df["rp_id"].tolist(),
            "coords": coords.tolist(),
            "ap_ids": rps_df["ap_ids"].tolist(),
            "rssi": rps_df["rssi"].tolist(),
            "edges": [(int(u), int(v), float(d)) for u, v, d in edges],
        }

        # Ensure all nested structures are JSON-serializable
        graph = to_json_serializable(graph)

        domain_id = rp_path.stem.replace("train_rps_", "")
        out_path = GRAPHS_DIR / f"train_graph_{domain_id}.json"
        with open(out_path, "w") as f:
            json.dump(graph, f)

        print(f"[build_graphs] Saved CONNECTED graph for train {domain_id} to {out_path}")


if __name__ == "__main__":
    main()