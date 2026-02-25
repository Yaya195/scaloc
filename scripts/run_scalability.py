# scripts/run_scalability.py
#
# Scalability evaluation (spec §9).
# Varies key parameters and measures accuracy, convergence, and timing:
#   1. Graph size (number of RPs per domain)
#   2. Number of domains/clients (K)
#   3. Data volume (samples per domain)
#   4. Number of FL rounds

import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

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
from run_fl_experiment import load_queries, load_domains

RPS_DIR = Path("data/processed/rps")
SAMPLES_DIR = Path("data/processed/samples")
RESULTS_DIR = Path("results/scalability")


def build_experiment(
    domains_subset: dict,
    val_queries_all: dict,
    model_cfg: dict,
    fl_cfg: dict,
    train_cfg: dict,
    device: str,
    max_rps_per_domain: int = None,
    max_queries_per_domain: int = None,
    seed: int = 42,
    rp_sample_seed: int = None,
    query_sample_seed: int = None,
):
    """Build server, clients, val_datasets for a scalability experiment."""
    num_aps = model_cfg["encoder"]["num_aps"]
    latent_dim = model_cfg["encoder"]["latent_dim"]
    ap_emb_dim = model_cfg["encoder"]["ap_emb_dim"]
    pooling = model_cfg["encoder"]["pooling"]
    arch = model_cfg["gnn"]["arch"]
    gnn_hidden_dim = model_cfg["gnn"]["hidden_dim"]
    gnn_layers = model_cfg["gnn"]["num_layers"]
    lr = train_cfg["training"]["learning_rate"]

    encoder = APWiseEncoder(num_aps=num_aps, latent_dim=latent_dim,
                            ap_emb_dim=ap_emb_dim, pooling=pooling)
    model = FLIndoorModel(
        latent_dim=latent_dim,
        arch=arch,
        hidden_dim=gnn_hidden_dim,
        num_layers=gnn_layers,
    )
    server = FLServer(global_model=model, global_encoder=encoder)

    clients = []
    graph_data_dict = {}
    domain_norm_stats = {}
    for domain_id, (rp_table_full, queries_full) in domains_subset.items():
        coords = rp_table_full[["x", "y"]].to_numpy().astype(float)
        coord_min = coords.min(axis=0).astype("float32")
        coord_max = coords.max(axis=0).astype("float32")

        rssi_vals = [float(v) for q in queries_full for v in q.get("rssi", [])]
        if not rssi_vals:
            rssi_vals = [float(v) for row in rp_table_full["rssi"] for v in row]

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

    for domain_id, (rp_table, queries) in domains_subset.items():
        # Optionally limit RPs
        if max_rps_per_domain and len(rp_table) > max_rps_per_domain:
            rp_table = rp_table.sample(
                n=max_rps_per_domain,
                random_state=rp_sample_seed if rp_sample_seed is not None else seed,
            )

        # Optionally limit queries
        if max_queries_per_domain and len(queries) > max_queries_per_domain:
            rng = np.random.RandomState(
                query_sample_seed if query_sample_seed is not None else seed
            )
            idx = rng.choice(len(queries), size=max_queries_per_domain, replace=False)
            queries = [queries[i] for i in sorted(idx)]

        if not queries:
            continue

        graph_data = build_domain_graph(
            domain_id,
            rp_table,
            encoder,
            norm_stats=domain_norm_stats.get(domain_id),
        )
        graph_data_dict[domain_id] = graph_data

        dataset = FLQueryDataset(graph_data, queries)
        client_model = copy.deepcopy(model)
        client_encoder = copy.deepcopy(encoder)

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

    # Validation
    val_datasets = {}
    for domain_id, queries in val_queries_all.items():
        if domain_id in graph_data_dict:
            val_datasets[domain_id] = FLQueryDataset(copy.deepcopy(graph_data_dict[domain_id]), queries)

    return server, clients, val_datasets


def run_one_experiment(
    name: str,
    server,
    clients,
    val_datasets,
    fl_cfg,
    train_cfg,
    device,
    rounds_override=None,
):
    """Run a single FL experiment and return results dict."""
    rounds = rounds_override or fl_cfg["federated"]["rounds"]
    local_epochs = fl_cfg["federated"]["local_epochs"]
    num_clients_per_round = None
    seed = fl_cfg["federated"]["seed"]
    eval_every = train_cfg["logging"].get("eval_every", 5)
    parallel_cfg = fl_cfg.get("parallel", {})

    tracker = ExperimentTracker(name, results_dir=str(RESULTS_DIR), config={
        "type": "scalability", "num_clients": len(clients),
    }, tensorboard_log_dir="logs")

    t_start = time.time()
    run_federated_training(
        server=server, clients=clients, rounds=rounds,
        local_epochs=local_epochs, num_clients_per_round=num_clients_per_round,
        sampling_strategy="all", seed=seed,
        tracker=tracker, val_datasets=val_datasets if val_datasets else None,
        eval_every=eval_every, device=device,
        parallel_clients=bool(parallel_cfg.get("enabled", False)),
        max_workers=parallel_cfg.get("max_workers", None),
        parallel_backend=parallel_cfg.get("backend", "thread"),
    )
    total_time = time.time() - t_start

    tracker.save()

    # Extract final metrics
    final_round = tracker.history["rounds"][-1] if tracker.history["rounds"] else {}
    return {
        "experiment": name,
        "num_clients": len(clients),
        "total_time_sec": total_time,
        "avg_loss_final": final_round.get("avg_loss"),
        "eval_metrics": final_round.get("eval_metrics", {}),
        "loss_history": tracker.get_loss_history(),
    }


def vary_num_domains(
    all_domains,
    val_queries_all,
    model_cfg,
    fl_cfg,
    train_cfg,
    device,
    domain_counts,
    rp_sample_seed,
    query_sample_seed,
):
    """Experiment 1: vary number of client domains K."""
    domain_ids = sorted(all_domains.keys())
    results = []
    for k_cfg in domain_counts:
        k = len(domain_ids) if k_cfg is None else min(int(k_cfg), len(domain_ids))
        subset = {d: all_domains[d] for d in domain_ids[:k]}
        print(f"\n=== Domains K={k} ===")
        server, clients, val_ds = build_experiment(
            subset,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            seed=fl_cfg["federated"]["seed"],
            rp_sample_seed=rp_sample_seed,
            query_sample_seed=query_sample_seed,
        )
        r = run_one_experiment(
            f"scale_domains_{k}", server, clients, val_ds, fl_cfg, train_cfg, device
        )
        r["K"] = k
        results.append(r)
    return results


def vary_graph_size(
    all_domains,
    val_queries_all,
    model_cfg,
    fl_cfg,
    train_cfg,
    device,
    max_rps_grid,
    rp_sample_seed,
    query_sample_seed,
):
    """Experiment 2: vary max RPs per domain."""
    results = []
    for max_rps in max_rps_grid:
        label = max_rps if max_rps else "all"
        print(f"\n=== Max RPs per domain: {label} ===")
        server, clients, val_ds = build_experiment(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            max_rps_per_domain=max_rps,
            seed=fl_cfg["federated"]["seed"],
            rp_sample_seed=rp_sample_seed,
            query_sample_seed=query_sample_seed,
        )
        r = run_one_experiment(
            f"scale_rps_{label}", server, clients, val_ds, fl_cfg, train_cfg, device
        )
        r["max_rps"] = label
        results.append(r)
    return results


def vary_data_volume(
    all_domains,
    val_queries_all,
    model_cfg,
    fl_cfg,
    train_cfg,
    device,
    max_queries_grid,
    rp_sample_seed,
    query_sample_seed,
):
    """Experiment 3: vary samples per domain."""
    results = []
    for max_q in max_queries_grid:
        label = max_q if max_q else "all"
        print(f"\n=== Max queries per domain: {label} ===")
        server, clients, val_ds = build_experiment(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            max_queries_per_domain=max_q,
            seed=fl_cfg["federated"]["seed"],
            rp_sample_seed=rp_sample_seed,
            query_sample_seed=query_sample_seed,
        )
        r = run_one_experiment(
            f"scale_queries_{label}", server, clients, val_ds, fl_cfg, train_cfg, device
        )
        r["max_queries"] = label
        results.append(r)
    return results


def vary_rounds(
    all_domains,
    val_queries_all,
    model_cfg,
    fl_cfg,
    train_cfg,
    device,
    rounds_grid,
    rp_sample_seed,
    query_sample_seed,
):
    """Experiment 4: convergence analysis — vary number of FL rounds."""
    results = []
    for rounds in rounds_grid:
        print(f"\n=== Rounds: {rounds} ===")
        server, clients, val_ds = build_experiment(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            seed=fl_cfg["federated"]["seed"],
            rp_sample_seed=rp_sample_seed,
            query_sample_seed=query_sample_seed,
        )
        r = run_one_experiment(
            f"scale_rounds_{rounds}", server, clients, val_ds, fl_cfg, train_cfg, device,
            rounds_override=rounds,
        )
        r["rounds"] = rounds
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(description="Scalability evaluation")
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["domains", "rps", "queries", "rounds", "all"],
        default=["all"],
        help="Which scalability experiments to run",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model_config")
    fl_cfg = load_config("fl_config")
    train_cfg = load_config("train_config")
    scalability_cfg = fl_cfg.get("scalability", {})

    domain_counts = scalability_cfg["domain_counts"]
    max_rps_grid = scalability_cfg["max_rps_per_domain"]
    max_queries_grid = scalability_cfg["max_queries_per_domain"]
    rounds_grid = scalability_cfg["rounds_grid"]
    rp_sample_seed = scalability_cfg["rp_sample_seed"]
    query_sample_seed = scalability_cfg["query_sample_seed"]
    device = resolve_device(train_cfg["training"].get("device", "auto"))
    print(f"[device] Using {device}")

    # Load all domains
    all_domains = load_domains()
    val_queries_all = load_queries(SAMPLES_DIR / "val_samples.parquet", "val")

    experiments = args.experiments
    if "all" in experiments:
        experiments = ["domains", "rps", "queries", "rounds"]

    all_results = {}

    if "domains" in experiments:
        print("\n" + "=" * 60)
        print("SCALABILITY: Varying number of domains (K)")
        print("=" * 60)
        all_results["domains"] = vary_num_domains(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            domain_counts,
            rp_sample_seed,
            query_sample_seed,
        )

    if "rps" in experiments:
        print("\n" + "=" * 60)
        print("SCALABILITY: Varying graph size (max RPs)")
        print("=" * 60)
        all_results["rps"] = vary_graph_size(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            max_rps_grid,
            rp_sample_seed,
            query_sample_seed,
        )

    if "queries" in experiments:
        print("\n" + "=" * 60)
        print("SCALABILITY: Varying data volume (queries per domain)")
        print("=" * 60)
        all_results["queries"] = vary_data_volume(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            max_queries_grid,
            rp_sample_seed,
            query_sample_seed,
        )

    if "rounds" in experiments:
        print("\n" + "=" * 60)
        print("SCALABILITY: Convergence analysis (rounds)")
        print("=" * 60)
        all_results["rounds"] = vary_rounds(
            all_domains,
            val_queries_all,
            model_cfg,
            fl_cfg,
            train_cfg,
            device,
            rounds_grid,
            rp_sample_seed,
            query_sample_seed,
        )

    # Save summary
    summary_path = RESULTS_DIR / "scalability_summary.json"
    # Convert for JSON serialization
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nScalability summary saved to {summary_path}")


if __name__ == "__main__":
    main()
