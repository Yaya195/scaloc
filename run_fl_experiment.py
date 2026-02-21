# run_fl_experiment.py

import copy
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

RPS_DIR = Path("data/processed/rps")
GRAPHS_DIR = Path("data/processed/graphs")
SAMPLES_DIR = Path("data/processed/samples")


def load_domains():
    train_samples = pd.read_parquet(SAMPLES_DIR / "train_samples.parquet")

    domains = {}
    for rp_path in sorted(RPS_DIR.glob("train_rps_*.parquet")):
        domain_id = rp_path.stem.replace("train_rps_", "")
        rp_table = pd.read_parquet(rp_path)

        q_df = train_samples[train_samples["domain_id"] == domain_id]
        queries = []
        skipped = 0
        for _, row in q_df.iterrows():
            aps = row["aps"]
            # Handle case where aps might be a string (JSON serialized)
            if isinstance(aps, str):
                import json
                aps = json.loads(aps)
            
            # Skip queries with empty AP lists
            if len(aps) == 0:
                skipped += 1
                continue
            
            ap_ids = [a["ap_id"] for a in aps]
            rssi = [a["rssi"] for a in aps]
            
            # Extract numeric part from ap_id (e.g., 'WAP007' -> 7)
            ap_ids = [int(ap.replace('WAP', '')) if isinstance(ap, str) else ap for ap in ap_ids]
            
            queries.append(
                {
                    "ap_ids": ap_ids,
                    "rssi": rssi,
                    "pos": (row["x"], row["y"]),
                }
            )
        
        if skipped > 0:
            print(f"  Domain {domain_id}: Skipped {skipped} queries with empty AP lists")

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
    arch = model_cfg["gnn"]["arch"]
    
    rounds = fl_cfg["federated"]["rounds"]
    local_epochs = fl_cfg["federated"]["local_epochs"]
    num_clients_per_round = fl_cfg["federated"]["num_clients_per_round"]
    sampling_strategy = fl_cfg["federated"]["sampling_strategy"]
    seed = fl_cfg["federated"]["seed"]
    
    lr = train_cfg["training"]["learning_rate"]
    device = train_cfg["training"]["device"]

    # Create global model and encoder for the server
    encoder_global = APWiseEncoder(num_aps=num_aps, latent_dim=latent_dim)
    model_global = FLIndoorModel(latent_dim=latent_dim, arch=arch)

    server = FLServer(global_model=model_global, global_encoder=encoder_global)

    domains = load_domains()
    clients = []
    
    # Build graphs using the global encoder (before creating client copies)
    graph_data_dict = {}
    for domain_id, (rp_table, queries) in domains.items():
        graph_data = build_domain_graph(domain_id, rp_table, encoder_global)
        graph_data_dict[domain_id] = (graph_data, queries)
    
    # Create independent model copies for each client
    for domain_id, (graph_data, queries) in graph_data_dict.items():
        dataset = FLQueryDataset(graph_data, queries)
        
        # Each client gets its own deep copy of model and encoder
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
        
        # Re-encode graph with client's own encoder copy for consistency
        dataset.update_graph_features(client_encoder, device)
        
        clients.append(client)

    run_federated_training(
        server=server,
        clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
        num_clients_per_round=num_clients_per_round,
        sampling_strategy=sampling_strategy,
        seed=seed,
    )


if __name__ == "__main__":
    main()