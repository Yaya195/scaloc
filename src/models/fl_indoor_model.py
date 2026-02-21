# src/models/fl_indoor_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn_model import FlexibleGNN


class FLIndoorModel(nn.Module):
    """
    Shared FL model:
        - Takes RP graph with latent node features
        - Takes query latent embedding z_q
        - Injects z_q into all nodes
        - Uses FlexibleGNN as backbone
        - Scores nodes and regresses position
    """

    def __init__(
        self,
        latent_dim: int,
        arch: str = "sage",
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gnn = FlexibleGNN(
            in_dim=latent_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            arch=arch,
            num_layers=num_layers,
            dropout=dropout,
            pooling=False,
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data, z_q):
        """
        Args:
            data: PyG Data with x, pos, edge_index
            z_q:  (latent_dim,) query embedding

        Returns:
            p_hat: (2,) predicted position
            w:     (N,) node weights
        """
        N = data.x.size(0)
        z_q_tiled = z_q.unsqueeze(0).repeat(N, 1)

        data = data.clone()
        data.x = torch.cat([data.x, z_q_tiled], dim=-1)

        node_emb = self.gnn(data)                  # (N, hidden_dim)
        scores = self.scorer(node_emb).squeeze(-1) # (N,)
        w = F.softmax(scores, dim=0)               # (N,)

        pos = data.pos                             # (N, 2)
        p_hat = (w.unsqueeze(-1) * pos).sum(dim=0) # (2,)

        return p_hat, w