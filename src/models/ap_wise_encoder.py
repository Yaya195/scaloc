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
        
        # Attention pooling: learns a scalar score per AP embedding
        # Score is computed from z_j (the modulated vector), not raw RSSI,
        # so it captures both AP identity and signal strength jointly.
        if pooling == "attention":
            self.attention_scorer = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.Tanh(),
                nn.Linear(latent_dim // 2, 1),
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

    def encode_packed(
        self,
        ap_ids: torch.Tensor,
        rssi: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a packed batch of variable-length fingerprints.

        Args:
            ap_ids: (B, L) long tensor with AP ids (padding can be any value)
            rssi: (B, L, 1) float tensor
            mask: (B, L) bool tensor, True for valid elements

        Returns:
            z: (B, latent_dim) batch embedding
        """
        if ap_ids.ndim != 2:
            raise ValueError(f"Expected ap_ids shape (B, L), got {tuple(ap_ids.shape)}")
        if rssi.ndim != 3:
            raise ValueError(f"Expected rssi shape (B, L, 1), got {tuple(rssi.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (B, L), got {tuple(mask.shape)}")

        batch_size, seq_len = ap_ids.shape
        ap_ids_safe = ap_ids.clamp(min=0)

        e = self.ap_emb(ap_ids_safe)                 # (B, L, ap_emb_dim)
        x = torch.cat([e, rssi], dim=-1)             # (B, L, ap_emb_dim + 1)
        z_j = self.modulator(x)                      # (B, L, latent_dim)

        valid = mask.unsqueeze(-1).to(z_j.dtype)     # (B, L, 1)

        if self.pooling == "mean":
            summed = (z_j * valid).sum(dim=1)        # (B, latent_dim)
            counts = valid.sum(dim=1).clamp(min=1.0) # (B, 1)
            z = summed / counts
        elif self.pooling == "sum":
            z = (z_j * valid).sum(dim=1)
        elif self.pooling == "attention":
            scores = self.attention_scorer(z_j).squeeze(-1)               # (B, L)
            scores = scores.masked_fill(~mask, float("-inf"))
            empty_rows = ~mask.any(dim=1)
            if empty_rows.any():
                scores = scores.clone()
                scores[empty_rows] = 0.0
            weights = F.softmax(scores, dim=1).unsqueeze(-1)              # (B, L, 1)
            weights = weights * valid
            norm = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = weights / norm
            z = (weights * z_j).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}. Choose from: mean, sum, attention")

        if batch_size == 0 or seq_len == 0:
            return torch.zeros((batch_size, self.latent_dim), dtype=torch.float32, device=ap_ids.device)

        empty = ~mask.any(dim=1)
        if empty.any():
            z = z.clone()
            z[empty] = 0.0
        return z

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
        elif self.pooling == "attention":
            # scores: (N, 1) -> softmax over visible APs
            scores = self.attention_scorer(z_j)          # (N, 1)
            weights = F.softmax(scores, dim=0)           # (N, 1) sums to 1
            z = (weights * z_j).sum(dim=0)               # (latent_dim,)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}. Choose from: mean, sum, attention")

        return z