# src/models/fl_localization_model.py

import torch.nn as nn

from src.models.ap_wise_encoder import APWiseEncoder
from src.models.query_injected_gnn import QueryInjectedGNN


class FLIndoorLocalizationModel(nn.Module):
    """
    Shared FL model:
        - AP-wise encoder (shared)
        - Query-injected GNN (shared)
    """

    def __init__(
        self,
        num_aps: int,
        ap_emb_dim: int = 32,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = APWiseEncoder(
            num_aps=num_aps,
            ap_emb_dim=ap_emb_dim,
            latent_dim=latent_dim,
        )

        self.gnn = QueryInjectedGNN(
            encoder=self.encoder,
            node_feat_dim=latent_dim,      # RP latent dim
            query_latent_dim=latent_dim,   # same encoder
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, data, query_ap_ids, query_rssi):
        return self.gnn(data, query_ap_ids, query_rssi)