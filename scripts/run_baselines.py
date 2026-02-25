# scripts/run_baselines.py
#
# Unified runner for all baselines (spec §8).
# Runs k-NN, Centralized MLP, Federated MLP, and Centralized GNN,
# then saves a comparison summary.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

import numpy as np
import pandas as pd

from load_config import load_config
from run_fl_experiment import load_queries
from src.fl.utils import select_federated_client_ids
from src.utils.device import resolve_device

RPS_DIR = Path("data/processed/rps")
SAMPLES_DIR = Path("data/processed/samples")
RESULTS_DIR = Path("results")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def load_all_data():
    """Load training and validation queries per domain."""
    train_queries = load_queries(SAMPLES_DIR / "train_samples.parquet", "train")
    val_queries = load_queries(SAMPLES_DIR / "val_samples.parquet", "val")
    return train_queries, val_queries


def run_knn(train_queries, val_queries, num_aps):
    """Run k-NN baseline."""
    print("\n" + "=" * 60)
    print("BASELINE: k-NN Fingerprinting")
    print("=" * 60)
    from src.baselines.knn_baseline import run_knn_baseline
    model_cfg = load_config("model_config")
    baseline_cfg = model_cfg.get("baselines", {})
    return run_knn_baseline(
        train_queries=train_queries,
        val_queries=val_queries,
        num_aps=num_aps,
        k=int(baseline_cfg.get("knn_k", 5)),
    )


def run_centralized_mlp(train_queries, val_queries, num_aps, device, train_cfg, fl_cfg):
    """Run Centralized MLP baseline."""
    print("\n" + "=" * 60)
    print("BASELINE: Centralized MLP")
    print("=" * 60)
    from src.baselines.centralized_mlp import run_centralized_mlp as _run
    model_cfg = load_config("model_config")
    baseline_cfg = model_cfg.get("baselines", {})
    return _run(
        train_queries=train_queries,
        val_queries=val_queries,
        num_aps=num_aps,
        hidden_dim=int(baseline_cfg.get("mlp_hidden_dim", model_cfg["gnn"]["hidden_dim"])),
        num_layers=int(baseline_cfg.get("mlp_num_layers", model_cfg["gnn"]["num_layers"])),
        dropout=float(baseline_cfg.get("mlp_dropout", model_cfg["encoder"].get("dropout", 0.2))),
        epochs=fl_cfg["federated"]["rounds"] * fl_cfg["federated"]["local_epochs"],  # total epochs across all rounds
        lr=float(train_cfg["training"]["learning_rate"]),
        batch_size=int(train_cfg["training"].get("batch_size", 128)),
        eval_every=int(train_cfg["logging"].get("eval_every", 5)),
        device=device,
    )


def run_federated_mlp(train_queries, val_queries, num_aps, fl_cfg, train_cfg, device):
    """Run Federated MLP baseline with same FL protocol as FedGNN."""
    print("\n" + "=" * 60)
    print("BASELINE: Federated MLP")
    print("=" * 60)
    from src.baselines.federated_mlp import run_federated_mlp as _run
    parallel_cfg = fl_cfg.get("parallel", {})
    model_cfg = load_config("model_config")
    baseline_cfg = model_cfg.get("baselines", {})
    return _run(
        train_queries=train_queries,
        val_queries=val_queries,
        num_aps=num_aps,
        hidden_dim=int(baseline_cfg.get("mlp_hidden_dim", model_cfg["gnn"]["hidden_dim"])),
        num_layers=int(baseline_cfg.get("mlp_num_layers", model_cfg["gnn"]["num_layers"])),
        dropout=float(baseline_cfg.get("mlp_dropout", model_cfg["encoder"].get("dropout", 0.2))),
        rounds=fl_cfg["federated"]["rounds"],
        local_epochs=fl_cfg["federated"]["local_epochs"],  # same as FedGNN
        lr=float(train_cfg["training"]["learning_rate"]),
        batch_size=int(train_cfg["training"].get("batch_size", 128)),
        num_clients_per_round=fl_cfg["federated"].get("num_clients_per_round"),
        sampling_strategy=fl_cfg["federated"].get("sampling_strategy", "random"),
        seed=fl_cfg["federated"].get("seed", 42),
        eval_every=int(train_cfg["logging"].get("eval_every", 5)),
        device=device,
        parallel_clients=bool(parallel_cfg.get("enabled", False)),
        parallel_backend=parallel_cfg.get("backend", "thread"),
        max_workers=parallel_cfg.get("max_workers", None),
    )


def run_centralized_gnn(model_cfg, train_cfg, device, allowed_domains, fl_cfg):
    """Run Centralized GNN baseline."""
    print("\n" + "=" * 60)
    print("BASELINE: Centralized GNN")
    print("=" * 60)
    from src.baselines.centralized_gnn import run_centralized_gnn as _run
    return _run(
        num_aps=model_cfg["encoder"]["num_aps"],
        latent_dim=model_cfg["encoder"]["latent_dim"],
        ap_emb_dim=model_cfg["encoder"]["ap_emb_dim"],
        pooling=model_cfg["encoder"]["pooling"],
        arch=model_cfg["gnn"]["arch"],
        hidden_dim=model_cfg["gnn"]["hidden_dim"],
        gnn_layers=model_cfg["gnn"]["num_layers"],
        epochs=fl_cfg["federated"]["rounds"] * fl_cfg["federated"]["local_epochs"],  # total epochs across all rounds
        lr=float(train_cfg["training"]["learning_rate"]),
        eval_every=int(train_cfg["logging"].get("eval_every", 5)),
        device=device,
        allowed_domains=allowed_domains,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config("model_config")
    fl_cfg = load_config("fl_config")
    train_cfg = load_config("train_config")
    device = resolve_device(train_cfg["training"].get("device", "auto"))
    print(f"[device] Using {device}")

    # AP IDs are 1..(num_aps-1), while 0 is reserved padding for embeddings.
    # MLP/kNN vector length must match the observable AP ID range.
    num_aps_vector = int(model_cfg["encoder"]["num_aps"]) - 1
    train_queries, val_queries = load_all_data()

    common_domains = sorted(
        d for d in train_queries.keys()
        if train_queries.get(d) and val_queries.get(d)
    )

    selected_domains = select_federated_client_ids(
        client_ids=common_domains,
        num_clients_per_round=fl_cfg["federated"].get("num_clients_per_round"),
        sampling_strategy=fl_cfg["federated"].get("sampling_strategy", "random"),
        seed=fl_cfg["federated"].get("seed", 42),
    )

    train_queries = {d: train_queries[d] for d in selected_domains}
    val_queries = {d: val_queries[d] for d in selected_domains}

    print(
        f"[fairness] Evaluating all baselines on the same {len(selected_domains)} "
        f"FL-selected train+val domains: {selected_domains}"
    )

    comparison = {}

    # 1. k-NN
    knn_results = run_knn(train_queries, val_queries, num_aps_vector)
    comparison["knn"] = knn_results.get("global", {})
    print(f"  k-NN global: {comparison['knn']}")

    # 2. Centralized MLP
    cmlp_results = run_centralized_mlp(train_queries, val_queries, num_aps_vector, device, train_cfg, fl_cfg)
    comparison["centralized_mlp"] = cmlp_results.get("global", {})
    print(f"  Centralized MLP global: {comparison['centralized_mlp']}")

    # 3. Federated MLP
    fmlp_results = run_federated_mlp(train_queries, val_queries, num_aps_vector, fl_cfg, train_cfg, device)
    comparison["federated_mlp"] = fmlp_results.get("global", {})
    print(f"  Federated MLP global: {comparison['federated_mlp']}")

    # 4. Centralized GNN
    cgnn_results = run_centralized_gnn(model_cfg, train_cfg, device, selected_domains, fl_cfg)
    comparison["centralized_gnn"] = cgnn_results.get("global", {})
    print(f"  Centralized GNN global: {comparison['centralized_gnn']}")

    # Save comparison
    out_path = RESULTS_DIR / "baseline_comparison.json"
    out_payload = {
        "comparison": comparison,
        "fairness": {
            "evaluation_domains": selected_domains,
            "num_domains": len(selected_domains),
            "device": device,
            "federated_protocol": {
                "rounds": fl_cfg["federated"]["rounds"],
                "local_epochs": fl_cfg["federated"]["local_epochs"],
                "num_clients_per_round": fl_cfg["federated"].get("num_clients_per_round"),
                "sampling_strategy": fl_cfg["federated"].get("sampling_strategy", "random"),
                "seed": fl_cfg["federated"].get("seed", 42),
            },
            "domain_selection": {
                "all_common_domains": common_domains,
                "selected_domains": selected_domains,
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2, cls=NumpyEncoder)
    print(f"\nBaseline comparison saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Mean(m)':<10} {'Median(m)':<10} {'P75(m)':<10} {'P90(m)':<10}")
    print("-" * 70)
    for method, metrics in comparison.items():
        if metrics:
            print(f"{method:<20} "
                  f"{metrics.get('mean_error', float('nan')):<10.2f} "
                  f"{metrics.get('median_error', float('nan')):<10.2f} "
                  f"{metrics.get('p75_error', float('nan')):<10.2f} "
                  f"{metrics.get('p90_error', float('nan')):<10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
