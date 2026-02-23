# src/baselines/centralized_mlp.py
#
# Centralized MLP baseline (spec §8).
# Trains a single MLP on all domains' data pooled together.
# Input: fixed-length RSSI vector. Output: (x, y) position.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List

from src.baselines.knn_baseline import build_rssi_vector_from_query
from src.data.normalization import (
    denormalize_coords,
    fit_global_normalization_stats,
    normalize_coords,
    normalize_rssi_matrix,
)
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tracker import ExperimentTracker
from src.utils.device import resolve_device


class LocalizationMLP(nn.Module):
    """Simple MLP for WiFi fingerprint localization."""

    def __init__(self, input_dim: int = 520, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def prepare_data(
    queries: List[dict],
    num_aps: int = 520,
    rssi_min: float = None,
    rssi_max: float = None,
    coord_min: np.ndarray = None,
    coord_max: np.ndarray = None,
):
    """Convert list of query dicts to (X, y) tensors, with optional normalization."""
    X = np.array([build_rssi_vector_from_query(q, num_aps) for q in queries], dtype=np.float32)
    y = np.array([q["pos"] for q in queries], dtype=np.float32)

    if rssi_min is not None and rssi_max is not None:
        X = normalize_rssi_matrix(X, rssi_min, rssi_max)
    if coord_min is not None and coord_max is not None:
        y = normalize_coords(y, coord_min, coord_max)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def run_centralized_mlp(
    train_queries: Dict[str, List[dict]],
    val_queries: Dict[str, List[dict]],
    num_aps: int = 520,
    hidden_dim: int = 256,
    num_layers: int = 3,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "auto",
    experiment_name: str = "centralized_mlp",
) -> Dict[str, dict]:
    """
    Train centralized MLP on pooled data from all domains.

    Returns:
        {domain_id: metrics_dict, "global": metrics_dict}
    """
    device = resolve_device(device)

    global_stats = fit_global_normalization_stats(train_queries)

    # Build pooled training data from per-domain normalized tensors
    train_x_parts = []
    train_y_parts = []
    val_domain_tensors = {}

    for domain_id, queries in train_queries.items():
        if not queries:
            continue
        X_d, y_d = prepare_data(
            queries,
            num_aps=num_aps,
            rssi_min=global_stats["rssi_min"],
            rssi_max=global_stats["rssi_max"],
            coord_min=global_stats["coord_min"],
            coord_max=global_stats["coord_max"],
        )
        train_x_parts.append(X_d)
        train_y_parts.append(y_d)

    for domain_id, queries in val_queries.items():
        if not queries:
            continue
        X_v, y_v_norm = prepare_data(
            queries,
            num_aps=num_aps,
            rssi_min=global_stats["rssi_min"],
            rssi_max=global_stats["rssi_max"],
            coord_min=global_stats["coord_min"],
            coord_max=global_stats["coord_max"],
        )
        y_v_raw = np.array([q["pos"] for q in queries], dtype=np.float32)
        val_domain_tensors[domain_id] = (X_v, y_v_norm, y_v_raw)

    if not train_x_parts:
        return {}

    X_train = torch.cat(train_x_parts, dim=0)
    y_train = torch.cat(train_y_parts, dim=0)
    
    X_train, y_train = X_train.to(device), y_train.to(device)

    model = LocalizationMLP(input_dim=num_aps, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Tracker
    tracker = ExperimentTracker(
        experiment_name,
        config={"type": "centralized_mlp", "epochs": epochs},
        tensorboard_log_dir="logs",
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        
        # Eval periodically
        eval_metrics = None
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            all_preds = []
            all_truths = []
            with torch.no_grad():
                for domain_id, (X_v, _y_v_norm, y_v_raw) in val_domain_tensors.items():
                    pred_norm = model(X_v.to(device)).cpu().numpy()
                    pred_raw = denormalize_coords(
                        pred_norm,
                        global_stats["coord_min"],
                        global_stats["coord_max"],
                    )
                    all_preds.append(pred_raw)
                    all_truths.append(y_v_raw)
            if all_preds:
                eval_metrics = compute_all_metrics(
                    np.concatenate(all_preds), np.concatenate(all_truths)
                )
            if eval_metrics is not None:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  "
                      f"mean_err={eval_metrics['mean_error']:.2f}m  "
                      f"median={eval_metrics['median_error']:.2f}m")
        
        tracker.log_round(epoch + 1, {"centralized": avg_loss}, eval_metrics)
    
    tracker.save()

    # Final per-domain evaluation
    model.eval()
    results = {}
    all_preds = []
    all_truths = []

    for domain_id, queries in val_queries.items():
        if not queries:
            continue
        X_d, _y_norm = prepare_data(
            queries,
            num_aps=num_aps,
            rssi_min=global_stats["rssi_min"],
            rssi_max=global_stats["rssi_max"],
            coord_min=global_stats["coord_min"],
            coord_max=global_stats["coord_max"],
        )
        X_d = X_d.to(device)
        with torch.no_grad():
            pred_norm = model(X_d).cpu().numpy()
        pred = denormalize_coords(
            pred_norm,
            global_stats["coord_min"],
            global_stats["coord_max"],
        )
        y_d_np = np.array([q["pos"] for q in queries], dtype=np.float32)

        metrics = compute_all_metrics(pred, y_d_np)
        results[domain_id] = metrics
        all_preds.append(pred)
        all_truths.append(y_d_np)

    if all_preds:
        results["global"] = compute_all_metrics(
            np.concatenate(all_preds), np.concatenate(all_truths)
        )

    return results
