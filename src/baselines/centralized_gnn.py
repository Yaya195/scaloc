# src/baselines/centralized_gnn.py
#
# Centralized GNN baseline (spec §8).
# Same APWiseEncoder + FLIndoorModel architecture, but trained
# centrally on all domain graphs pooled together (no FL).

import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.ap_wise_encoder import APWiseEncoder
from src.models.fl_indoor_model import FLIndoorModel
from src.data.rp_graph_builder import build_domain_graph
from src.data.fl_query_dataset import FLQueryDataset
from src.data.normalization import fit_global_normalization_stats
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tracker import ExperimentTracker
from src.utils.device import resolve_device


def run_centralized_gnn(
    rps_dir: str = "data/processed/rps",
    samples_dir: str = "data/processed/samples",
    num_aps: int = 521,
    latent_dim: int = 64,
    ap_emb_dim: int = 32,
    pooling: str = "attention",
    arch: str = "sage",
    hidden_dim: int = 64,
    gnn_layers: int = 3,
    epochs: int = 50,
    lr: float = 5e-4,
    device: str = "auto",
    eval_every: int = 5,
    allowed_domains=None,
    experiment_name: str = "centralized_gnn",
) -> dict:
    """
    Train an APWiseEncoder + FLIndoorModel centrally on all domains.

    The training iterates over all (domain, query) pairs each epoch,
    re-encoding RP features with gradients per query (same as FL client).

    Returns:
        {domain_id: metrics_dict, "global": metrics_dict}
    """
    device = resolve_device(device)
    rps_dir = Path(rps_dir)
    samples_dir = Path(samples_dir)
    allowed_set = set(allowed_domains) if allowed_domains is not None else None

    # --- Create model + encoder ---
    encoder = APWiseEncoder(
        num_aps=num_aps, latent_dim=latent_dim,
        ap_emb_dim=ap_emb_dim, pooling=pooling,
    ).to(device)

    model = FLIndoorModel(
        latent_dim=latent_dim, arch=arch,
        hidden_dim=hidden_dim, num_layers=gnn_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(encoder.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    # --- Load training data ---
    from run_fl_experiment import load_queries

    train_queries_all = load_queries(samples_dir / "train_samples.parquet", "train")
    if allowed_set is not None:
        selected_train_queries = {
            d: q for d, q in train_queries_all.items()
            if d in allowed_set and q
        }
    else:
        selected_train_queries = {d: q for d, q in train_queries_all.items() if q}
    global_norm_stats = fit_global_normalization_stats(selected_train_queries)

    train_datasets = {}
    for rp_path in sorted(rps_dir.glob("train_rps_*.parquet")):
        domain_id = rp_path.stem.replace("train_rps_", "")
        if allowed_set is not None and domain_id not in allowed_set:
            continue
        rp_table = pd.read_parquet(rp_path)
        queries = train_queries_all.get(domain_id, [])
        if not queries:
            continue
        graph = build_domain_graph(
            domain_id,
            rp_table,
            encoder,
            norm_stats=global_norm_stats,
        )
        dataset = FLQueryDataset(graph, queries)
        dataset.update_graph_features(encoder, device)
        train_datasets[domain_id] = dataset

    # --- Load validation data ---
    val_datasets = {}
    val_path = samples_dir / "val_samples.parquet"
    if val_path.exists():
        val_queries_all = load_queries(val_path, "val")
        for domain_id, queries in val_queries_all.items():
            if allowed_set is not None and domain_id not in allowed_set:
                continue
            if domain_id in train_datasets:
                graph = train_datasets[domain_id].graph
                val_datasets[domain_id] = FLQueryDataset(graph, queries)

    print(f"Centralized GNN: {len(train_datasets)} domains, "
          f"{sum(len(d) for d in train_datasets.values())} train queries, "
          f"{sum(len(d) for d in val_datasets.values())} val queries")

    # --- Build flat training list (domain_id, query_idx) ---
    train_items = []
    for domain_id, ds in train_datasets.items():
        for idx in range(len(ds)):
            train_items.append((domain_id, idx))

    tracker = ExperimentTracker(
        experiment_name,
        config={
            "type": "centralized_gnn",
            "epochs": epochs,
            "arch": arch,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        },
        tensorboard_log_dir="logs",
    )

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        encoder.train()

        # Shuffle training order
        indices = np.random.permutation(len(train_items))
        epoch_loss = 0.0
        n_batches = 0

        for i in indices:
            domain_id, q_idx = train_items[i]
            ds = train_datasets[domain_id]
            graph = ds.graph.to(device)

            ap_ids, rssi, true_pos = ds[q_idx]
            ap_ids = ap_ids.to(device)
            rssi = rssi.to(device)
            true_pos = true_pos.to(device)

            optimizer.zero_grad()

            # Re-encode RP node features WITH gradients
            rp_feats = []
            for fp in graph.rp_fingerprints:
                rp_ap = torch.tensor(fp["ap_ids"], dtype=torch.long, device=device)
                rp_r = torch.tensor(fp["rssi"], dtype=torch.float, device=device).unsqueeze(-1)
                rp_feats.append(encoder(rp_ap, rp_r))

            graph_live = graph.clone()
            graph_live.x = torch.stack(rp_feats, dim=0)

            z_q = encoder(ap_ids, rssi)

            if torch.isnan(z_q).any():
                continue

            p_hat, _ = model(graph_live, z_q)

            if torch.isnan(p_hat).any():
                continue

            loss = criterion(p_hat, true_pos)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(encoder.parameters()), max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Periodic eval
        eval_metrics = None
        if (epoch + 1) % eval_every == 0 or epoch == 0 or epoch == epochs - 1:
            eval_metrics = _evaluate(model, encoder, val_datasets, device)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  "
                  f"mean_err={eval_metrics['mean_error']:.2f}m  "
                  f"median={eval_metrics['median_error']:.2f}m")

        tracker.log_round(epoch + 1, {"centralized": avg_loss}, eval_metrics)

    tracker.save()

    # --- Final per-domain eval ---
    results = {}
    all_preds, all_truths = [], []

    model.eval()
    encoder.eval()

    for domain_id, val_ds in val_datasets.items():
        preds, truths = _predict_domain(model, encoder, val_ds, device)
        metrics = compute_all_metrics(preds, truths)
        results[domain_id] = metrics
        all_preds.append(preds)
        all_truths.append(truths)

    if all_preds:
        results["global"] = compute_all_metrics(
            np.concatenate(all_preds), np.concatenate(all_truths)
        )

    return results


def _evaluate(model, encoder, val_datasets, device):
    """Run evaluation on all validation domains, return global metrics."""
    model.eval()
    encoder.eval()
    all_preds, all_truths = [], []

    for domain_id, val_ds in val_datasets.items():
        preds, truths = _predict_domain(model, encoder, val_ds, device)
        all_preds.append(preds)
        all_truths.append(truths)

    model.train()
    encoder.train()

    if not all_preds:
        return {"mean_error": float("inf"), "median_error": float("inf")}

    return compute_all_metrics(
        np.concatenate(all_preds), np.concatenate(all_truths)
    )


def _predict_domain(model, encoder, val_ds, device):
    """Predict positions for all queries in a validation dataset.
    
    Returns positions in METRIC space (denormalized) for fair comparison
    with baselines that operate in raw coordinate space.
    """
    preds, truths = [], []
    graph = val_ds.graph.to(device)
    coord_min = val_ds.coord_min
    coord_max = val_ds.coord_max
    coord_range = coord_max - coord_min

    with torch.no_grad():
        # Encode RPs once
        rp_feats = []
        for fp in graph.rp_fingerprints:
            rp_ap = torch.tensor(fp["ap_ids"], dtype=torch.long, device=device)
            rp_r = torch.tensor(fp["rssi"], dtype=torch.float, device=device).unsqueeze(-1)
            rp_feats.append(encoder(rp_ap, rp_r))

        graph_live = graph.clone()
        graph_live.x = torch.stack(rp_feats, dim=0)

        for idx in range(len(val_ds)):
            ap_ids, rssi, true_pos = val_ds[idx]
            ap_ids = ap_ids.to(device)
            rssi = rssi.to(device)

            z_q = encoder(ap_ids, rssi)
            p_hat, _ = model(graph_live, z_q)

            # Denormalize to metric space
            p_hat_metric = p_hat.cpu().numpy() * coord_range + coord_min
            p_true_metric = true_pos.numpy() * coord_range + coord_min

            preds.append(p_hat_metric)
            truths.append(p_true_metric)

    return np.stack(preds), np.stack(truths)
