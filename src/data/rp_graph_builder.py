# src/data/rp_graph_builder.py
#
# Build per-domain RP graph:
#   - Node features: AP-wise encoded RP fingerprints
#   - Node coords:   (x, y)
#   - Edges:         k-NN in coordinate space

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch

from torch_geometric.data import Data

from src.models.ap_wise_encoder import APWiseEncoder


def build_domain_graph(
    domain_id: str,
    rp_table: pd.DataFrame,
    encoder: APWiseEncoder,
    k: int = 5,
) -> Data:
    """
    Build a single domain graph G_k.

    Args:
        domain_id: string identifier, e.g. "2_3"
        rp_table:  DataFrame with columns:
                   - "rp_id"
                   - "x", "y"
                   - "ap_ids": list[int]
                   - "rssi":   list[float]
        encoder:   APWiseEncoder instance (shared)
        k:         number of neighbors for k-NN graph

    Returns:
        PyG Data object with:
            - x:   (N, latent_dim) node features
            - pos: (N, 2) normalized coordinates
            - edge_index: (2, E)
            - domain_id: str
            - coord_stats: dict with 'min' and 'max' for denormalization
    """
    rp_ids = rp_table["rp_id"].tolist()
    coords = rp_table[["x", "y"]].to_numpy().astype(float)
    
    # Store raw RP fingerprints for re-encoding when encoder updates
    rp_fingerprints = []
    for _, row in rp_table.iterrows():
        rp_fingerprints.append({
            "ap_ids": row["ap_ids"],
            "rssi": row["rssi"]
        })
    
    # Compute normalization stats (min-max per dimension)
    coord_min = coords.min(axis=0)
    coord_max = coords.max(axis=0)
    coord_range = coord_max - coord_min
    coord_range = np.maximum(coord_range, 1e-6)  # Avoid division by zero
    
    # Normalize coordinates to [0, 1]
    coords_normalized = (coords - coord_min) / coord_range

    # Encode each RP fingerprint with current encoder
    node_feats = []
    for fp in rp_fingerprints:
        ap_ids = torch.tensor(fp["ap_ids"], dtype=torch.long)
        rssi = torch.tensor(fp["rssi"], dtype=torch.float).unsqueeze(-1)
        z = encoder(ap_ids, rssi)  # (latent_dim,)
        node_feats.append(z.detach().cpu().numpy())

    x = np.stack(node_feats, axis=0)  # (N, latent_dim)

    # Build k-NN edges in coordinate space (use normalized coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords_normalized)
    distances, indices = nbrs.kneighbors(coords_normalized)

    edge_list = []
    for i in range(coords.shape[0]):
        for j_idx in indices[i, 1:]:  # skip self
            edge_list.append((i, int(j_idx)))

    # Make edges undirected
    edge_set = set()
    for u, v in edge_list:
        edge_set.add((u, v))
        edge_set.add((v, u))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t()

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        pos=torch.tensor(coords_normalized, dtype=torch.float),
        edge_index=edge_index,
        num_nodes=coords.shape[0],
        domain_id=domain_id,
        rp_ids=rp_ids,
        coord_min=torch.tensor(coord_min, dtype=torch.float),
        coord_max=torch.tensor(coord_max, dtype=torch.float),
        rp_fingerprints=rp_fingerprints,  # Store raw data for re-encoding
    )

    return data