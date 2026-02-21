# src/models/ap_wise_encoder.py
#
# AP-wise set encoder:
#   Input:  variable-size set {(AP_j, r_j)}
#   Output: fixed-size latent vector z ∈ R^d_latent

import torch
import torch.nn as nn
import torch.nn.functional as F


class APWiseEncoder(nn.Module):
    """
    AP-wise encoder:
        - Learnable AP embeddings
        - Shared modulation f(r_j, e_j)
        - Set pooling over visible APs
    """

    def __init__(
        self,
        num_aps: int,
        ap_emb_dim: int = 32,
        latent_dim: int = 64,
        rssi_dim: int = 1,
        pooling: str = "mean",
    ):
        super().__init__()

        self.num_aps = num_aps
        self.ap_emb_dim = ap_emb_dim
        self.latent_dim = latent_dim
        self.pooling = pooling

        self.ap_emb = nn.Embedding(num_aps, ap_emb_dim)

        self.modulator = nn.Sequential(
            nn.Linear(ap_emb_dim + rssi_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # Initialize weights properly to avoid NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ap_ids: torch.Tensor, rssi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ap_ids: (N,) long tensor of AP indices
            rssi:   (N, 1) float tensor of RSSI values

        Returns:
            z: (latent_dim,) fingerprint embedding
        """
        # Handle empty fingerprint case
        if len(ap_ids) == 0:
            return torch.zeros(self.latent_dim, dtype=torch.float32, device=ap_ids.device)
        
        e = self.ap_emb(ap_ids)              # (N, ap_emb_dim)
        x = torch.cat([e, rssi], dim=-1)     # (N, ap_emb_dim + 1)
        z_j = self.modulator(x)              # (N, latent_dim)

        if self.pooling == "mean":
            z = z_j.mean(dim=0)
        elif self.pooling == "sum":
            z = z_j.sum(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return z