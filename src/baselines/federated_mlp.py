# src/baselines/federated_mlp.py
#
# Federated MLP baseline (spec §8).
# Same MLP architecture as centralized, but trained via FedAvg
# with each domain as a separate client. No GNN, no graphs.

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List

from src.baselines.centralized_mlp import LocalizationMLP, prepare_data
from src.data.normalization import (
    denormalize_coords,
    fit_domain_normalization_stats,
)
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tracker import ExperimentTracker
from src.fl.utils import select_federated_client_ids
from src.utils.device import resolve_device


def run_federated_mlp(
    train_queries: Dict[str, List[dict]],
    val_queries: Dict[str, List[dict]],
    num_aps: int = 520,
    hidden_dim: int = 256,
    num_layers: int = 3,
    rounds: int = 50,
    local_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_clients_per_round: int = None,
    sampling_strategy: str = "random",
    seed: int = 42,
    device: str = "auto",
    experiment_name: str = "federated_mlp",
) -> Dict[str, dict]:
    """
    Train MLP via FedAvg across domain clients.
    Uses the same FL protocol as FedGNN for fair comparison:
      - same local_epochs, client sampling, optimizer preservation.

    Returns:
        {domain_id: metrics_dict, "global": metrics_dict}
    """
    device = resolve_device(device)
    global_model = LocalizationMLP(input_dim=num_aps, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    domain_stats = fit_domain_normalization_stats(train_queries)

    # Prepare per-domain data
    domain_data = {}
    for domain_id, queries in train_queries.items():
        if not queries or domain_id not in domain_stats:
            continue
        stats = domain_stats[domain_id]
        X, y = prepare_data(
            queries,
            num_aps=num_aps,
            rssi_min=stats["rssi_min"],
            rssi_max=stats["rssi_max"],
            coord_min=stats["coord_min"],
            coord_max=stats["coord_max"],
        )
        domain_data[domain_id] = (X.to(device), y.to(device))

    # Prepare validation
    val_data = {}
    for domain_id, queries in val_queries.items():
        if queries and domain_id in domain_stats:
            stats = domain_stats[domain_id]
            X, y_norm = prepare_data(
                queries,
                num_aps=num_aps,
                rssi_min=stats["rssi_min"],
                rssi_max=stats["rssi_max"],
                coord_min=stats["coord_min"],
                coord_max=stats["coord_max"],
            )
            y_raw = np.array([q["pos"] for q in queries], dtype=np.float32)
            val_data[domain_id] = (X.to(device), y_norm.to(device), y_raw)
    
    tracker = ExperimentTracker(experiment_name, config={
        "type": "federated_mlp", "rounds": rounds, "local_epochs": local_epochs
    }, tensorboard_log_dir="logs")

    # Client sampling: select ONCE per experiment (same as FedGNN)
    all_domain_ids = sorted(domain_data.keys())
    selected_ids = select_federated_client_ids(
        client_ids=all_domain_ids,
        num_clients_per_round=num_clients_per_round,
        sampling_strategy=sampling_strategy,
        seed=seed,
    )
    if sampling_strategy == "random" and num_clients_per_round is not None:
        print(
            f"  Federated MLP: sampled {len(selected_ids)}/{len(all_domain_ids)} clients: {selected_ids}"
        )

    # Per-client persistent optimizers (same as FedGNN: Adam state preserved)
    client_optimizers = {}
    client_models = {}
    for domain_id in selected_ids:
        client_models[domain_id] = copy.deepcopy(global_model).to(device)
        client_optimizers[domain_id] = torch.optim.Adam(
            client_models[domain_id].parameters(), lr=lr
        )

    for r in range(rounds):
        client_states = []
        client_sizes = []
        client_losses = {}

        for domain_id in selected_ids:
            X_train, y_train = domain_data[domain_id]
            
            # Load global weights into local model (preserve optimizer state)
            local_model = client_models[domain_id]
            local_model.load_state_dict(global_model.state_dict())
            optimizer = client_optimizers[domain_id]
            
            ds = TensorDataset(X_train, y_train)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            local_model.train()
            loss_val = 0.0
            for _ in range(local_epochs):
                epoch_loss = 0.0
                n = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = local_model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n += 1
                loss_val = epoch_loss / max(n, 1)

            client_losses[domain_id] = loss_val
            client_states.append({k: v.clone() for k, v in local_model.state_dict().items()})
            client_sizes.append(len(X_train))

        # FedAvg aggregation
        total = sum(client_sizes)
        agg_state = {}
        for key in global_model.state_dict().keys():
            agg_state[key] = sum(
                (client_sizes[i] / total) * client_states[i][key]
                for i in range(len(client_states))
            )
        global_model.load_state_dict(agg_state)

        # Eval periodically
        eval_metrics = None
        if (r + 1) % 5 == 0 or r == 0 or r == rounds - 1:
            global_model.eval()
            all_preds, all_truths = [], []
            for domain_id, (X_v, _y_v_norm, y_v_raw) in val_data.items():
                stats = domain_stats[domain_id]
                with torch.no_grad():
                    pred_norm = global_model(X_v).cpu().numpy()
                pred = denormalize_coords(pred_norm, stats["coord_min"], stats["coord_max"])
                all_preds.append(pred)
                all_truths.append(y_v_raw)
            if all_preds:
                eval_metrics = compute_all_metrics(
                    np.concatenate(all_preds), np.concatenate(all_truths)
                )
                avg_loss = sum(client_losses.values()) / len(client_losses)
                print(f"  Round {r+1}/{rounds}: avg_loss={avg_loss:.4f}  "
                      f"mean_err={eval_metrics['mean_error']:.2f}m  "
                      f"median={eval_metrics['median_error']:.2f}m")

        tracker.log_round(r + 1, client_losses, eval_metrics)

    tracker.save()

    # Final per-domain evaluation
    global_model.eval()
    results = {}
    all_preds, all_truths = [], []

    for domain_id, (X_v, _y_v_norm, y_raw) in val_data.items():
        stats = domain_stats[domain_id]
        with torch.no_grad():
            pred_norm = global_model(X_v).cpu().numpy()
        pred = denormalize_coords(pred_norm, stats["coord_min"], stats["coord_max"])
        y_np = y_raw
        metrics = compute_all_metrics(pred, y_np)
        results[domain_id] = metrics
        all_preds.append(pred)
        all_truths.append(y_np)

    if all_preds:
        results["global"] = compute_all_metrics(
            np.concatenate(all_preds), np.concatenate(all_truths)
        )

    return results
