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
import sys
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

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))
from load_config import load_config


def run_centralized_gnn(
    rps_dir: str = "data/processed/rps",
    samples_dir: str = "data/processed/samples",
    num_aps: int = None,
    latent_dim: int = None,
    ap_emb_dim: int = None,
    pooling: str = None,
    arch: str = None,
    hidden_dim: int = None,
    gnn_layers: int = None,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    device: str = "auto",
    eval_every: int = None,
    allowed_domains=None,
    max_rps_per_domain: int = None,
    max_queries_per_domain: int = None,
    rp_sample_seed: int = 42,
    query_sample_seed: int = 42,
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

    model_cfg = load_config("model_config")
    fl_cfg = load_config("fl_config")
    train_cfg = load_config("train_config")
    if num_aps is None:
        num_aps = int(model_cfg["encoder"]["num_aps"])
    if latent_dim is None:
        latent_dim = int(model_cfg["encoder"]["latent_dim"])
    if ap_emb_dim is None:
        ap_emb_dim = int(model_cfg["encoder"]["ap_emb_dim"])
    if pooling is None:
        pooling = str(model_cfg["encoder"]["pooling"])
    if arch is None:
        arch = str(model_cfg["gnn"]["arch"])
    if hidden_dim is None:
        hidden_dim = int(model_cfg["gnn"]["hidden_dim"])
    if gnn_layers is None:
        gnn_layers = int(model_cfg["gnn"]["num_layers"])
    if epochs is None:
        epochs = int(fl_cfg["federated"]["rounds"]) * int(fl_cfg["federated"]["local_epochs"])
    if lr is None:
        lr = float(train_cfg["training"]["learning_rate"])
    if batch_size is None:
        batch_size = int(train_cfg["training"]["batch_size"])
    if eval_every is None:
        eval_every = int(train_cfg["logging"]["eval_every"])

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

        if max_rps_per_domain and len(rp_table) > max_rps_per_domain:
            rp_table = rp_table.sample(n=max_rps_per_domain, random_state=rp_sample_seed)

        queries = train_queries_all.get(domain_id, [])
        if max_queries_per_domain and len(queries) > max_queries_per_domain:
            rng = np.random.RandomState(query_sample_seed)
            idx = rng.choice(len(queries), size=max_queries_per_domain, replace=False)
            queries = [queries[i] for i in sorted(idx)]

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
                graph = copy.deepcopy(train_datasets[domain_id].graph)
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

        for start in range(0, len(indices), batch_size):
            batch_slice = indices[start : start + batch_size]
            optimizer.zero_grad()
            batch_losses = []
            grouped = {}

            for i in batch_slice:
                domain_id, q_idx = train_items[i]
                grouped.setdefault(domain_id, []).append(q_idx)

            for domain_id, q_indices in grouped.items():
                ds = train_datasets[domain_id]
                graph = ds.graph.to(device)

                # Re-encode RP node features WITH gradients
                rp_feats = []
                for fp in graph.rp_fingerprints:
                    rp_ap = torch.tensor(fp["ap_ids"], dtype=torch.long, device=device)
                    rp_r = torch.tensor(fp["rssi"], dtype=torch.float, device=device).unsqueeze(-1)
                    rp_feats.append(encoder(rp_ap, rp_r))

                graph_live = graph.clone()
                graph_live.x = torch.stack(rp_feats, dim=0)

                query_items = [ds[q_idx] for q_idx in q_indices]
                max_len = max((len(ap_ids) for ap_ids, _rssi, _pos in query_items), default=0)
                if max_len == 0:
                    continue

                q_ap = torch.zeros((len(query_items), max_len), dtype=torch.long, device=device)
                q_rssi = torch.zeros((len(query_items), max_len, 1), dtype=torch.float, device=device)
                q_mask = torch.zeros((len(query_items), max_len), dtype=torch.bool, device=device)
                q_pos = torch.zeros((len(query_items), 2), dtype=torch.float, device=device)

                for row, (ap_ids, rssi, true_pos) in enumerate(query_items):
                    ap_ids = ap_ids.to(device)
                    rssi = rssi.to(device)
                    true_pos = true_pos.to(device)
                    length = len(ap_ids)
                    if length > 0:
                        q_ap[row, :length] = ap_ids
                        q_rssi[row, :length, :] = rssi
                        q_mask[row, :length] = True
                    q_pos[row] = true_pos

                z_q = encoder.encode_packed(q_ap, q_rssi, q_mask)
                if torch.isnan(z_q).any():
                    continue

                p_hat, _ = model(graph_live, z_q)
                if torch.isnan(p_hat).any():
                    continue

                loss = criterion(p_hat, q_pos)
                if torch.isnan(loss):
                    continue

                batch_losses.append(loss)

            if not batch_losses:
                continue

            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(encoder.parameters()), max_norm=1.0
            )
            optimizer.step()

            epoch_loss += float(batch_loss.item())
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
    model = model.to(device)
    encoder = encoder.to(device)
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


def _predict_domain(model, encoder, val_ds, device, batch_size: int = 128):
    """Predict positions for all queries in a validation dataset.
    
    Returns positions in METRIC space (denormalized) for fair comparison
    with baselines that operate in raw coordinate space.
    """
    model = model.to(device)
    encoder = encoder.to(device)
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

        indices = list(range(len(val_ds)))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            query_items = [val_ds[idx] for idx in batch_indices]
            max_len = max((len(ap_ids) for ap_ids, _rssi, _pos in query_items), default=0)
            if max_len == 0:
                continue

            q_ap = torch.zeros((len(query_items), max_len), dtype=torch.long, device=device)
            q_rssi = torch.zeros((len(query_items), max_len, 1), dtype=torch.float, device=device)
            q_mask = torch.zeros((len(query_items), max_len), dtype=torch.bool, device=device)
            q_true = torch.zeros((len(query_items), 2), dtype=torch.float, device=device)

            for row, (ap_ids, rssi, true_pos) in enumerate(query_items):
                ap_ids = ap_ids.to(device)
                rssi = rssi.to(device)
                true_pos = true_pos.to(device)
                length = len(ap_ids)
                if length > 0:
                    q_ap[row, :length] = ap_ids
                    q_rssi[row, :length, :] = rssi
                    q_mask[row, :length] = True
                q_true[row] = true_pos

            z_q = encoder.encode_packed(q_ap, q_rssi, q_mask)
            p_hat, _ = model(graph_live, z_q)

            p_hat_metric = p_hat.cpu().numpy() * coord_range + coord_min
            p_true_metric = q_true.cpu().numpy() * coord_range + coord_min

            preds.extend(list(p_hat_metric))
            truths.extend(list(p_true_metric))

    return np.stack(preds), np.stack(truths)
