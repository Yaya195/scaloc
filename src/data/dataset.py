# src/data/indoor_graph_dataset.py
#
# Purpose:
#   Load RP graphs (built earlier), normalize them consistently,
#   and convert them into PyTorch Geometric Data objects.
#
# Key features:
#   - Normalizes coordinates (per domain)
#   - Normalizes RSSI features (per domain)
#   - Normalizes edge weights (per domain)
#   - Computes normalization stats on TRAIN
#   - Reuses the same stats for VALIDATION
#   - Keeps raw graph files untouched
#
# This is the correct place for normalization:
#   - NOT in RP builder
#   - NOT in graph builder
#   - ONLY at model input time

import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
import numpy as np
import json
from pathlib import Path

GRAPHS_DIR = Path("data/processed/graphs")


class IndoorGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for indoor RP graphs.

    Each domain graph becomes one PyG Data object:
        - x: normalized fingerprint features
        - pos: normalized coordinates
        - edge_index: graph connectivity
        - edge_attr: normalized edge distances
        - domain_id: string identifier

    Normalization:
        - TRAIN computes normalization stats
        - VALIDATION reuses TRAIN stats
    """

    def __init__(self, split="train", transform=None, pre_transform=None, norm_stats=None):
        """
        Args:
            split: "train" or "val"
            norm_stats: (coord_stats, feature_stats) from TRAIN
        """
        self.split = split
        self.norm_stats = norm_stats  # will be filled for train
        super().__init__(None, transform, pre_transform)

        # Load processed dataset (created in process())
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # Saved as train_graphs.pt or val_graphs.pt
        return [f"{self.split}_graphs.pt"]

    # ---------------------------------------------------------
    # Normalization utilities (self-contained)
    # ---------------------------------------------------------

    def _fit_coord_norm(self, coords):
        """Compute min/max for coordinate normalization."""
        min_xy = coords.min(axis=0, keepdims=True)
        max_xy = coords.max(axis=0, keepdims=True)
        return (min_xy, max_xy)

    def _apply_coord_norm(self, coords, stats):
        """Apply min/max normalization to coordinates."""
        min_xy, max_xy = stats
        return (coords - min_xy) / (max_xy - min_xy + 1e-6)

    def _fit_feature_norm(self, features):
        """Compute mean/std for RSSI feature normalization."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        return (mean, std)

    def _apply_feature_norm(self, features, stats):
        """Apply z-score normalization to RSSI features."""
        mean, std = stats
        return (features - mean) / std

    def _normalize_edge_weights(self, edges, coords):
        """
        Normalize edge distances by domain diagonal.
        This makes edge weights comparable across domains.
        """
        diag = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) + 1e-6
        return [(u, v, d / diag) for (u, v, d) in edges]

    # ---------------------------------------------------------
    # Main processing logic
    # ---------------------------------------------------------

    def process(self):
        """
        Loads raw graph files, normalizes them, and converts them
        into PyTorch Geometric Data objects.
        """
        graph_files = sorted(GRAPHS_DIR.glob(f"{self.split}_graph_*.json"))
        data_list = []

        # -----------------------------------------------------
        # 1. Compute normalization stats on TRAIN only
        # -----------------------------------------------------
        if self.split == "train":
            all_coords = []
            all_features = []

            for path in graph_files:
                with open(path) as f:
                    g = json.load(f)
                all_coords.append(np.array(g["coords"]))
                # Build RSSI features from ap_ids and rssi lists
                # Features are the RSSI values indexed by AP ID
                features = np.array(g["rssi"]).reshape(-1, 1)
                all_features.append(features)

            all_coords = np.vstack(all_coords)
            all_features = np.vstack(all_features)

            coord_stats = self._fit_coord_norm(all_coords)
            feat_stats = self._fit_feature_norm(all_features)

            self.norm_stats = (coord_stats, feat_stats)

        # For validation: reuse train stats
        coord_stats, feat_stats = self.norm_stats

        # -----------------------------------------------------
        # 2. Build PyG Data objects
        # -----------------------------------------------------
        for path in graph_files:
            with open(path) as f:
                g = json.load(f)

            coords = np.array(g["coords"])
            # Build RSSI features from ap_ids and rssi lists
            features = np.array(g["rssi"]).reshape(-1, 1)
            edges = g["edges"]

            # Apply normalization
            coords = self._apply_coord_norm(coords, coord_stats)
            features = self._apply_feature_norm(features, feat_stats)
            edges = self._normalize_edge_weights(edges, coords)

            # Convert edges to PyG format
            edge_index = torch.tensor(
                [[u, v] for (u, v, _) in edges], dtype=torch.long
            ).t()

            edge_attr = torch.tensor(
                [d for (_, _, d) in edges], dtype=torch.float
            ).unsqueeze(1)

            # Create PyG Data object
            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                pos=torch.tensor(coords, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(coords),
                domain_id=path.stem.replace(f"{self.split}_graph_", "")
            )

            data_list.append(data)

        # Save processed dataset
        torch.save(self.collate(data_list), self.processed_paths[0])



# src/data/indoor_graph_dataset_fl.py
#
# Purpose:
#   Federated Learning–ready dataset class.
#   Each client loads exactly ONE domain graph.
#   Normalization is computed LOCALLY on that domain only.
#
# Why this design:
#   - In FL, clients must NOT see each other's data.
#   - Each domain has its own geometry + RSSI distribution.
#   - Local normalization preserves privacy and domain structure.
#   - Server only aggregates model weights, never raw data.

class IndoorGraphDatasetFL(InMemoryDataset):
    """
    Federated Learning dataset:
        - Loads ONE domain graph (one client = one domain)
        - Computes LOCAL normalization (coords, features, edges)
        - Returns a single PyG Data object

    This class is intentionally minimal and FL‑correct.
    """

    def __init__(self, domain_id, transform=None, pre_transform=None):
        """
        Args:
            domain_id: e.g. "2_3"
        """
        self.domain_id = domain_id
        super().__init__(None, transform, pre_transform)

        # Load processed dataset (created in process())
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # Saved as: fl_graph_2_3.pt
        return [f"fl_graph_{self.domain_id}.pt"]

    # ---------------------------------------------------------
    # Normalization utilities (local to each client)
    # ---------------------------------------------------------

    def _fit_coord_norm(self, coords):
        """Compute min/max for coordinate normalization."""
        min_xy = coords.min(axis=0, keepdims=True)
        max_xy = coords.max(axis=0, keepdims=True)
        return (min_xy, max_xy)

    def _apply_coord_norm(self, coords, stats):
        """Apply min/max normalization to coordinates."""
        min_xy, max_xy = stats
        return (coords - min_xy) / (max_xy - min_xy + 1e-6)

    def _fit_feature_norm(self, features):
        """Compute mean/std for RSSI feature normalization."""
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        return (mean, std)

    def _apply_feature_norm(self, features, stats):
        """Apply z-score normalization to RSSI features."""
        mean, std = stats
        return (features - mean) / std

    def _normalize_edge_weights(self, edges, coords):
        """
        Normalize edge distances by domain diagonal.
        This makes edge weights comparable across domains.
        """
        diag = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) + 1e-6
        return [(u, v, d / diag) for (u, v, d) in edges]

    # ---------------------------------------------------------
    # Main processing logic
    # ---------------------------------------------------------

    def process(self):
        """
        Loads ONE graph file, normalizes it locally,
        and converts it into a PyTorch Geometric Data object.
        """
        graph_path = GRAPHS_DIR / f"train_graph_{self.domain_id}.json"
        with open(graph_path) as f:
            g = json.load(f)

        coords = np.array(g["coords"])
        # Build RSSI features from ap_ids and rssi lists
        features = np.array(g["rssi"]).reshape(-1, 1)
        edges = g["edges"]

        # -----------------------------------------------------
        # 1. Compute LOCAL normalization stats
        # -----------------------------------------------------
        coord_stats = self._fit_coord_norm(coords)
        feat_stats = self._fit_feature_norm(features)

        # -----------------------------------------------------
        # 2. Apply normalization
        # -----------------------------------------------------
        coords = self._apply_coord_norm(coords, coord_stats)
        features = self._apply_feature_norm(features, feat_stats)
        edges = self._normalize_edge_weights(edges, coords)

        # -----------------------------------------------------
        # 3. Convert to PyG format
        # -----------------------------------------------------
        edge_index = torch.tensor(
            [[u, v] for (u, v, _) in edges], dtype=torch.long
        ).t()

        edge_attr = torch.tensor(
            [d for (_, _, d) in edges], dtype=torch.float
        ).unsqueeze(1)

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            pos=torch.tensor(coords, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(coords),
            domain_id=self.domain_id
        )

        # Save processed dataset
        torch.save(self.collate([data]), self.processed_paths[0])