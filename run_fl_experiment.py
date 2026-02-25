# run_fl_experiment.py

import copy
import json
import pandas as pd
from pathlib import Path
import sys

# Add configs to path
sys.path.insert(0, str(Path(__file__).parent / "configs"))

from load_config import load_config
from src.models.ap_wise_encoder import APWiseEncoder
from src.models.fl_indoor_model import FLIndoorModel
from src.fl.server import FLServer
from src.fl.client import FLClient
from src.fl.run_federated_training import run_federated_training
from src.data.rp_graph_builder import build_domain_graph
from src.data.fl_query_dataset import FLQueryDataset
from src.evaluation.tracker import ExperimentTracker
from src.utils.device import resolve_device

RPS_DIR = Path("data/processed/rps")
GRAPHS_DIR = Path("data/processed/graphs")
SAMPLES_DIR = Path("data/processed/samples")


def load_queries(samples_path, split_prefix="train"):
    """Load and parse query data from a parquet samples file."""
    samples = pd.read_parquet(samples_path)
    
    domains = {}
    for domain_id in samples["domain_id"].unique():
        q_df = samples[samples["domain_id"] == domain_id]
        queries = []
        skipped = 0
        for _, row in q_df.iterrows():
            aps = row["aps"]
            if isinstance(aps, str):
                aps = json.loads(aps)
            
            if len(aps) == 0:
                skipped += 1
                continue
            
            ap_ids = [a["ap_id"] for a in aps]
            rssi = [a["rssi"] for a in aps]
            ap_ids = [int(ap.replace('WAP', '')) if isinstance(ap, str) else ap for ap in ap_ids]
            
            queries.append({
                "ap_ids": ap_ids,
                "rssi": rssi,
                "pos": (row["x"], row["y"]),
            })
        
        if skipped > 0:
            print(f"  Domain {domain_id}: Skipped {skipped} {split_prefix} queries with empty AP lists")
        
        domains[domain_id] = queries
    
    return domains


def load_domains():
    """Load training data: RP tables + query sets per domain."""
    train_queries = load_queries(SAMPLES_DIR / "train_samples.parquet", "train")
    
    domains = {}
    for rp_path in sorted(RPS_DIR.glob("train_rps_*.parquet")):
        domain_id = rp_path.stem.replace("train_rps_", "")
        rp_table = pd.read_parquet(rp_path)
        queries = train_queries.get(domain_id, [])
        domains[domain_id] = (rp_table, queries)
    
    return domains


def main():
    # Load configs
    model_cfg = load_config("model_config")
    fl_cfg = load_config("fl_config")
    train_cfg = load_config("train_config")
    
    # Extract parameters from configs
    num_aps = model_cfg["encoder"]["num_aps"]
    latent_dim = model_cfg["encoder"]["latent_dim"]
    ap_emb_dim = model_cfg["encoder"]["ap_emb_dim"]
    pooling = model_cfg["encoder"]["pooling"]
    arch = model_cfg["gnn"]["arch"]
    gnn_hidden_dim = model_cfg["gnn"]["hidden_dim"]
    gnn_layers = model_cfg["gnn"]["num_layers"]
    
    rounds = fl_cfg["federated"]["rounds"]
    local_epochs = fl_cfg["federated"]["local_epochs"]
    num_clients_per_round = fl_cfg["federated"]["num_clients_per_round"]
    sampling_strategy = fl_cfg["federated"]["sampling_strategy"]
    seed = fl_cfg["federated"]["seed"]
    parallel_cfg = fl_cfg.get("parallel", {})
    parallel_clients = bool(parallel_cfg.get("enabled", False))
    max_workers = parallel_cfg.get("max_workers", None)
    parallel_backend = parallel_cfg.get("backend", "thread")
    
    lr = train_cfg["training"]["learning_rate"]
    device = resolve_device(train_cfg["training"].get("device", "auto"))
    print(f"[device] Using {device}")
    eval_every = train_cfg["logging"].get("eval_every", 5)

    # Create global model and encoder for the server
    encoder_global = APWiseEncoder(num_aps=num_aps, latent_dim=latent_dim, ap_emb_dim=ap_emb_dim, pooling=pooling)
    model_global = FLIndoorModel(
        latent_dim=latent_dim,
        arch=arch,
        hidden_dim=gnn_hidden_dim,
        num_layers=gnn_layers,
    )

    server = FLServer(global_model=model_global, global_encoder=encoder_global)

    # ------- Load training data -------
    domains = load_domains()
    domain_norm_stats = {}
    for domain_id, (rp_table, queries) in domains.items():
        coords = rp_table[["x", "y"]].to_numpy().astype(float)
        coord_min = coords.min(axis=0).astype("float32")
        coord_max = coords.max(axis=0).astype("float32")

        rssi_vals = [float(v) for q in queries for v in q.get("rssi", [])]
        if not rssi_vals:
            rssi_vals = [float(v) for row in rp_table["rssi"] for v in row]

        if rssi_vals:
            rssi_min = float(min(rssi_vals))
            rssi_max = float(max(rssi_vals))
            if abs(rssi_max - rssi_min) < 1e-6:
                rssi_max = rssi_min + 1.0
        else:
            rssi_min = -110.0
            rssi_max = -30.0

        domain_norm_stats[domain_id] = {
            "coord_min": coord_min,
            "coord_max": coord_max,
            "rssi_min": rssi_min,
            "rssi_max": rssi_max,
        }
    clients = []
    
    graph_data_dict = {}
    for domain_id, (rp_table, queries) in domains.items():
        graph_data = build_domain_graph(
            domain_id,
            rp_table,
            encoder_global,
            norm_stats=domain_norm_stats.get(domain_id),
        )
        graph_data_dict[domain_id] = (graph_data, queries)
    
    for domain_id, (graph_data, queries) in graph_data_dict.items():
        dataset = FLQueryDataset(graph_data, queries)
        
        client_model = copy.deepcopy(model_global)
        client_encoder = copy.deepcopy(encoder_global)

        client = FLClient(
            client_id=domain_id,
            model=client_model,
            encoder=client_encoder,
            dataset=dataset,
            lr=lr,
            device=device,
        )
        dataset.update_graph_features(client_encoder, device)
        clients.append(client)
    
    # ------- Load validation data -------
    val_datasets = {}
    val_samples_path = SAMPLES_DIR / "val_samples.parquet"
    if val_samples_path.exists():
        val_queries = load_queries(val_samples_path, "val")
        for domain_id, queries in val_queries.items():
            if domain_id in graph_data_dict:
                graph_data = copy.deepcopy(graph_data_dict[domain_id][0])
                val_datasets[domain_id] = FLQueryDataset(graph_data, queries)
        print(f"Loaded validation data: {sum(len(d) for d in val_datasets.values())} samples across {len(val_datasets)} domains\n")
    else:
        print("No validation data found, skipping evaluation during training.\n")
    
    # ------- Setup tracker -------
    experiment_config = {
        "model": model_cfg,
        "fl": fl_cfg,
        "training": train_cfg,
        "effective_device": device,
        "num_clients": len(clients),
        "num_domains": len(domains),
    }
    tracker = ExperimentTracker(
        experiment_name="fedgnn",
        results_dir="results",
        config=experiment_config,
        tensorboard_log_dir="logs",
    )

    # ------- Run FL training -------
    run_federated_training(
        server=server,
        clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
        num_clients_per_round=num_clients_per_round,
        sampling_strategy=sampling_strategy,
        seed=seed,
        tracker=tracker,
        val_datasets=val_datasets if val_datasets else None,
        eval_every=eval_every,
        device=device,
        parallel_clients=parallel_clients,
        max_workers=max_workers,
        parallel_backend=parallel_backend,
    )


if __name__ == "__main__":
    main()