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
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tracker import ExperimentTracker


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


def prepare_data(queries: List[dict], num_aps: int = 520):
    """Convert list of query dicts to (X, y) tensors."""
    X = np.array([build_rssi_vector_from_query(q, num_aps) for q in queries])
    y = np.array([q["pos"] for q in queries])
    
    # Normalize RSSI: shift to [0, 1] range
    X = (X + 110.0) / 110.0
    
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
    device: str = "cpu",
    experiment_name: str = "centralized_mlp",
) -> Dict[str, dict]:
    """
    Train centralized MLP on pooled data from all domains.

    Returns:
        {domain_id: metrics_dict, "global": metrics_dict}
    """
    # Pool all training data
    all_train = []
    for queries in train_queries.values():
        all_train.extend(queries)
    
    all_val = []
    for queries in val_queries.values():
        all_val.extend(queries)

    X_train, y_train = prepare_data(all_train, num_aps)
    X_val, y_val = prepare_data(all_val, num_aps)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    model = LocalizationMLP(input_dim=num_aps, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Tracker
    tracker = ExperimentTracker(experiment_name, config={"type": "centralized_mlp", "epochs": epochs})

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
            with torch.no_grad():
                pred_val = model(X_val).cpu().numpy()
            eval_metrics = compute_all_metrics(pred_val, y_val.cpu().numpy())
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
        X_d, y_d = prepare_data(queries, num_aps)
        X_d = X_d.to(device)
        with torch.no_grad():
            pred = model(X_d).cpu().numpy()
        y_d_np = y_d.numpy()

        metrics = compute_all_metrics(pred, y_d_np)
        results[domain_id] = metrics
        all_preds.append(pred)
        all_truths.append(y_d_np)

    if all_preds:
        results["global"] = compute_all_metrics(
            np.concatenate(all_preds), np.concatenate(all_truths)
        )

    return results
