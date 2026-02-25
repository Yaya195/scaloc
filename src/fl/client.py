# src/fl/client.py

import torch
import torch.nn as nn
from contextlib import nullcontext


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
        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            try:
                self.scaler = torch.amp.GradScaler("cpu", enabled=False)
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)
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

    def train_one_epoch(self, batch_size=1):
        self.model.train()
        self.encoder.train()
        self._ensure_packed_rp_tensors()

        graph = self.dataset.graph.to(self.device)

        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(len(self.dataset)).tolist()
        for index in indices:
            ap_ids, rssi, true_pos = self.dataset[index]
            ap_ids = ap_ids.to(self.device, non_blocking=True)
            rssi = rssi.to(self.device, non_blocking=True)
            true_pos = true_pos.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )
            with autocast_ctx:
                rp_feats = self.encoder.encode_packed(
                    self._cached_rp_ap_ids,
                    self._cached_rp_rssi,
                    self._cached_rp_mask,
                )

                graph.x = rp_feats

                z_q = self.encoder(ap_ids, rssi)
                if not torch.isfinite(z_q).all():
                    print(f"  WARNING: Non-finite encoder output for client {self.client_id}")
                    print(f"    AP count: {len(ap_ids)}")
                    continue

                p_hat, _ = self.model(graph, z_q)

                if not torch.isfinite(p_hat).all():
                    print(f"  WARNING: Non-finite model output for client {self.client_id}")
                    continue

                loss = self.criterion(p_hat, true_pos)

                if not torch.isfinite(loss):
                    print(f"  WARNING: Non-finite loss for client {self.client_id}")
                    print(f"    p_hat: {p_hat}, true_pos: {true_pos}")
                    continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

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
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)