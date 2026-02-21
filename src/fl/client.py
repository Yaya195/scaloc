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
        # This is critical: without this, graph has stale embeddings from old encoder
        self.dataset.update_graph_features(self.encoder, self.device)
        
        # Recreate optimizer to reset momentum/state when loading new global parameters
        params_list = list(self.model.parameters()) + list(self.encoder.parameters())
        lr = self.optimizer.param_groups[0]['lr']  # Preserve learning rate
        self.optimizer = torch.optim.Adam(params_list, lr=lr)

    def train_one_epoch(self, batch_size=1):
        self.model.train()
        self.encoder.train()
        # Use batch_size=1 because encoder expects single queries with variable-length AP sets
        loader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        # Get domain graph (same for all batches)
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

            z_q = self.encoder(ap_ids, rssi)  # (latent_dim,)
            
            # Check for NaN in encoder output
            if torch.isnan(z_q).any():
                print(f"  WARNING: NaN in encoder output for client {self.client_id}")
                print(f"    AP count: {len(ap_ids)}")
                continue
            
            p_hat, _ = self.model(graph, z_q)
            
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