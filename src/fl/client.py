# src/fl/client.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    """
    One FL client = one domain.
    """

    def __init__(self, client_id, model, encoder, dataset, lr=1e-3, device="cpu"):
        self.client_id = client_id
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.dataset = dataset
        self.device = device

        params = list(self.model.parameters()) + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.criterion = nn.MSELoss()

    def get_parameters(self):
        return {
            "model": {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()},
            "encoder": {k: v.cpu().detach().clone() for k, v in self.encoder.state_dict().items()},
        }

    def set_parameters(self, params):
        # Move parameters to the correct device before loading
        model_params = {k: v.to(self.device) for k, v in params["model"].items()}
        encoder_params = {k: v.to(self.device) for k, v in params["encoder"].items()}
        
        self.model.load_state_dict(model_params)
        self.encoder.load_state_dict(encoder_params)
        
        # Re-encode graph node features with updated encoder
        # (no_grad — training re-encodes with gradients per batch in train_one_epoch)
        self.dataset.update_graph_features(self.encoder, self.device)
        
        # NOTE: optimizer state is intentionally preserved across rounds.
        # Resetting it every round destroys Adam's momentum, preventing convergence
        # with small local_epochs. Only model weights are synchronized in FedAvg.

    def train_one_epoch(self, batch_size=1):
        self.model.train()
        self.encoder.train()
        # Use batch_size=1 because encoder expects single queries with variable-length AP sets
        loader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        # Get domain graph structure (topology + raw fingerprints, same for all batches)
        graph = self.dataset.graph.to(self.device)

        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            ap_ids, rssi, true_pos = batch

            # batch_size=1, so squeeze the batch dimension
            ap_ids = ap_ids.squeeze(0).to(self.device)
            rssi = rssi.squeeze(0).to(self.device)
            true_pos = true_pos.squeeze(0).to(self.device)

            self.optimizer.zero_grad()

            # ----------------------------------------------------------------
            # Re-encode RP node features WITH gradients so the encoder
            # is jointly trained through both the RP path and the query path.
            # spec §5.1: x_i = phi_enc(r_bar_i), z_q = phi_enc(r_q)
            # Both must use the same encoder in the same forward pass.
            # ----------------------------------------------------------------
            rp_feats = []
            for fp in graph.rp_fingerprints:
                rp_ap_ids = torch.tensor(fp["ap_ids"], dtype=torch.long).to(self.device)
                rp_rssi = torch.tensor(fp["rssi"], dtype=torch.float).unsqueeze(-1).to(self.device)
                rp_feats.append(self.encoder(rp_ap_ids, rp_rssi))  # (latent_dim,)
            
            # Build a differentiable graph copy with live node features
            graph_live = graph.clone()
            graph_live.x = torch.stack(rp_feats, dim=0)  # (N, latent_dim) — has grad_fn

            z_q = self.encoder(ap_ids, rssi)  # (latent_dim,)
            
            # Check for NaN in encoder output
            if torch.isnan(z_q).any():
                print(f"  WARNING: NaN in encoder output for client {self.client_id}")
                print(f"    AP count: {len(ap_ids)}")
                continue
            
            p_hat, _ = self.model(graph_live, z_q)
            
            # Check for NaN in model output
            if torch.isnan(p_hat).any():
                print(f"  WARNING: NaN in model output for client {self.client_id}")
                continue

            loss = self.criterion(p_hat, true_pos)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss for client {self.client_id}")
                print(f"    p_hat: {p_hat}, true_pos: {true_pos}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.encoder.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)