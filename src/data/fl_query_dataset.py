# src/data/fl_query_dataset.py

import torch
from torch.utils.data import Dataset

from src.data.normalization import normalize_rssi_values


class FLQueryDataset(Dataset):
    """
    Dataset for one domain in FL:
        - Holds the domain graph (PyG Data)
        - Holds query fingerprints + true positions (normalized)
    """

    def __init__(self, graph_data, queries):
        """
        Args:
            graph_data: PyG Data object for the domain (with coord_min, coord_max)
            queries: list of dicts:
                {
                    "ap_ids": list[int],
                    "rssi":   list[float],
                    "pos":    (x, y)  # in original metric space
                }
        """
        self.graph = graph_data
        self.queries = queries
        
        # Extract normalization stats from graph
        self.coord_min = graph_data.coord_min.numpy()
        self.coord_max = graph_data.coord_max.numpy()
        self.coord_range = self.coord_max - self.coord_min
        self.rssi_min = float(graph_data.rssi_min.item()) if hasattr(graph_data, "rssi_min") else -110.0
        self.rssi_max = float(graph_data.rssi_max.item()) if hasattr(graph_data, "rssi_max") else -30.0

    def update_graph_features(self, encoder, device="cpu"):
        """
        Re-encode graph node features with the updated encoder (no gradients).
        Used to keep graph.x in sync after FL aggregation for inference
        and for `update_graph_features` calls in set_parameters.
        
        NOTE: During training, client.py re-encodes RP features WITH gradients
        inline in each forward pass (see train_one_epoch). This method is only
        used for inference / round-boundary bookkeeping.
        """
        encoder.eval()
        node_feats = []
        
        with torch.no_grad():
            for fp in self.graph.rp_fingerprints:
                ap_ids = torch.tensor(fp["ap_ids"], dtype=torch.long).to(device)
                rssi = torch.tensor(fp["rssi"], dtype=torch.float).unsqueeze(-1).to(device)
                z = encoder(ap_ids, rssi)  # (latent_dim,)
                node_feats.append(z.cpu().numpy())
        
        import numpy as np
        x = np.stack(node_feats, axis=0)
        self.graph.x = torch.tensor(x, dtype=torch.float)
    
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]

        ap_ids = torch.tensor(q["ap_ids"], dtype=torch.long)
        rssi_norm = normalize_rssi_values(q["rssi"], self.rssi_min, self.rssi_max)
        rssi = torch.tensor(rssi_norm, dtype=torch.float).unsqueeze(-1)
        
        # Normalize query position using the same stats as RPs
        pos_raw = torch.tensor(q["pos"], dtype=torch.float)
        pos_normalized = (pos_raw - torch.from_numpy(self.coord_min).float()) / torch.from_numpy(self.coord_range).float()

        return ap_ids, rssi, pos_normalized