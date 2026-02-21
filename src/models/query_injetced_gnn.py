# src/models/query_injected_gnn.py
#
# GNN that:
#   - takes an RP graph (Data)
#   - takes a query fingerprint
#   - injects query embedding into all nodes
#   - scores nodes
#   - outputs continuous position estimate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

from src.models.ap_wise_encoder import APWiseEncoder


class QueryInjectedGNN(nn.Module):
    """
    Indoor localization model:
        - AP-wise encoder for query
        - GNN over RP graph with query injection
        - Node scoring head
        - Position regression via weighted sum of RP coords
    """

    def __init__(
        self,
        encoder: APWiseEncoder,
        node_feat_dim: int,
        query_latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = encoder
        self.dropout = dropout

        in_dim = node_feat_dim + query_latent_dim

        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))

        # Node scoring head
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data, query_ap_ids, query_rssi):
        """
        Args:
            data:          PyG Data with x, pos, edge_index
            query_ap_ids:  (N_q,) long tensor
            query_rssi:    (N_q, 1) float tensor

        Returns:
            p_hat: (2,) predicted position
            s:     (num_nodes,) node scores (softmax)
        """
        # Encode query fingerprint
        z_q = self.encoder(query_ap_ids, query_rssi)  # (query_latent_dim,)

        # Inject query into all nodes
        N = data.x.shape[0]
        z_q_tiled = z_q.unsqueeze(0).repeat(N, 1)  # (N, query_latent_dim)
        x = torch.cat([data.x, z_q_tiled], dim=-1)  # (N, in_dim)

        edge_index = data.edge_index

        # GNN message passing
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Node scores
        s = self.scorer(x).squeeze(-1)  # (N,)
        s_soft = F.softmax(s, dim=0)    # (N,)

        # Continuous position estimate
        pos = data.pos  # (N, 2)
        p_hat = (s_soft.unsqueeze(-1) * pos).sum(dim=0)  # (2,)

        return p_hat, s_soft