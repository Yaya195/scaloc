# src/fl/client.py

import torch
import torch.nn as nn


class FLClient:
    """
    One FL client = one domain.
    """

    def __init__(self, client_id, model, encoder, dataset, lr=1e-3, device="cpu", batch_size=1):
        self.client_id = client_id
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._cached_rp_device = None
        self._cached_rp_ap_ids = None
        self._cached_rp_rssi = None
        self._cached_rp_mask = None

        params = list(self.model.parameters()) + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.criterion = nn.MSELoss()

    def get_parameters(self):
        return {
            "model": {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()},
            "encoder": {k: v.cpu().detach().clone() for k, v in self.encoder.state_dict().items()},
        }

    def set_parameters(self, params):
        self.model.load_state_dict(params["model"])
        self.encoder.load_state_dict(params["encoder"])
        
        # NOTE: optimizer state is intentionally preserved across rounds.
        # Resetting it every round destroys Adam's momentum, preventing convergence
        # with small local_epochs. Only model weights are synchronized in FedAvg.

    def _ensure_packed_rp_tensors(self):
        if self._cached_rp_device == self.device and self._cached_rp_ap_ids is not None:
            return

        fingerprints = self.dataset.graph.rp_fingerprints
        num_rps = len(fingerprints)
        max_len = max((len(fp["ap_ids"]) for fp in fingerprints), default=0)

        if num_rps == 0 or max_len == 0:
            self._cached_rp_ap_ids = torch.empty((0, 0), dtype=torch.long, device=self.device)
            self._cached_rp_rssi = torch.empty((0, 0, 1), dtype=torch.float, device=self.device)
            self._cached_rp_mask = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            self._cached_rp_device = self.device
            return

        ap_ids = torch.zeros((num_rps, max_len), dtype=torch.long)
        rssi = torch.zeros((num_rps, max_len, 1), dtype=torch.float)
        mask = torch.zeros((num_rps, max_len), dtype=torch.bool)

        for row, fp in enumerate(fingerprints):
            length = len(fp["ap_ids"])
            if length == 0:
                continue
            ap_ids[row, :length] = torch.tensor(fp["ap_ids"], dtype=torch.long)
            rssi[row, :length, 0] = torch.tensor(fp["rssi"], dtype=torch.float)
            mask[row, :length] = True

        self._cached_rp_ap_ids = ap_ids.to(self.device, non_blocking=True)
        self._cached_rp_rssi = rssi.to(self.device, non_blocking=True)
        self._cached_rp_mask = mask.to(self.device, non_blocking=True)
        self._cached_rp_device = self.device

    def _pack_query_batch(self, batch_indices):
        batch_items = [self.dataset[idx] for idx in batch_indices]
        batch_size = len(batch_items)
        max_len = max((len(ap_ids) for ap_ids, _rssi, _pos in batch_items), default=0)

        if batch_size == 0 or max_len == 0:
            return (
                torch.empty((0, 0), dtype=torch.long, device=self.device),
                torch.empty((0, 0, 1), dtype=torch.float, device=self.device),
                torch.empty((0, 0), dtype=torch.bool, device=self.device),
                torch.empty((0, 2), dtype=torch.float, device=self.device),
            )

        ap_ids_batch = torch.zeros((batch_size, max_len), dtype=torch.long)
        rssi_batch = torch.zeros((batch_size, max_len, 1), dtype=torch.float)
        mask_batch = torch.zeros((batch_size, max_len), dtype=torch.bool)
        pos_batch = torch.zeros((batch_size, 2), dtype=torch.float)

        for row, (ap_ids, rssi, pos) in enumerate(batch_items):
            length = len(ap_ids)
            if length > 0:
                ap_ids_batch[row, :length] = ap_ids
                rssi_batch[row, :length, :] = rssi
                mask_batch[row, :length] = True
            pos_batch[row] = pos

        return (
            ap_ids_batch.to(self.device, non_blocking=True),
            rssi_batch.to(self.device, non_blocking=True),
            mask_batch.to(self.device, non_blocking=True),
            pos_batch.to(self.device, non_blocking=True),
        )

    def train_one_epoch(self, batch_size=None):
        self.model.train()
        self.encoder.train()
        self._ensure_packed_rp_tensors()

        effective_batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))

        graph = self.dataset.graph.to(self.device)

        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(len(self.dataset)).tolist()
        for start in range(0, len(indices), effective_batch_size):
            batch_indices = indices[start : start + effective_batch_size]
            ap_ids_batch, rssi_batch, mask_batch, true_pos_batch = self._pack_query_batch(batch_indices)
            if ap_ids_batch.numel() == 0:
                continue

            self.optimizer.zero_grad(set_to_none=True)

            rp_feats = self.encoder.encode_packed(
                self._cached_rp_ap_ids,
                self._cached_rp_rssi,
                self._cached_rp_mask,
            )

            graph.x = rp_feats

            z_q_batch = self.encoder.encode_packed(ap_ids_batch, rssi_batch, mask_batch)
            if not torch.isfinite(z_q_batch).all():
                print(f"  WARNING: Non-finite encoder output for client {self.client_id}")
                print(f"    Batch size: {z_q_batch.size(0)}")
                continue

            p_hat_batch, _ = self.model(graph, z_q_batch)
            if not torch.isfinite(p_hat_batch).all():
                print(f"  WARNING: Non-finite model output for client {self.client_id}")
                continue

            loss = self.criterion(p_hat_batch, true_pos_batch)

            if not torch.isfinite(loss):
                print(f"  WARNING: Non-finite loss for client {self.client_id}")
                continue

            loss.backward()

            grad_finite = True
            for param in list(self.model.parameters()) + list(self.encoder.parameters()):
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                print(f"  WARNING: Non-finite gradients for client {self.client_id}; skipping optimizer step")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.encoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)