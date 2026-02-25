# scripts/run_centralized_gnn_architectures.py
#
# Compare centralized GNN performance across architectures supported by FlexibleGNN:
#   gcn, sage, gat, gin, egnn

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))

from load_config import load_config
from run_fl_experiment import load_queries
from src.baselines.centralized_gnn import run_centralized_gnn
from src.fl.utils import select_federated_client_ids
from src.utils.device import resolve_device

SAMPLES_DIR = Path("data/processed/samples")
SUPPORTED_ARCHS = ["gcn", "sage", "gat", "gin", "egnn"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def select_fair_domains(fl_cfg):
    train_queries = load_queries(SAMPLES_DIR / "train_samples.parquet", "train")
    val_queries = load_queries(SAMPLES_DIR / "val_samples.parquet", "val")

    common_domains = sorted(
        domain_id
        for domain_id in train_queries.keys()
        if train_queries.get(domain_id) and val_queries.get(domain_id)
    )

    selected_domains = select_federated_client_ids(
        client_ids=common_domains,
        num_clients_per_round=fl_cfg["federated"].get("num_clients_per_round"),
        sampling_strategy=fl_cfg["federated"].get("sampling_strategy", "random"),
        seed=fl_cfg["federated"].get("seed", 42),
    )
    return common_domains, selected_domains


def load_experiment_cfg():
    baseline_cfg = load_config("baseline_config")
    exp_cfg = baseline_cfg["centralized_gnn_architecture_experiment"]

    configured_archs = exp_cfg["architectures"]
    architectures = [arch for arch in configured_archs if arch in SUPPORTED_ARCHS]
    if not architectures:
        raise ValueError(
            "baseline_config.centralized_gnn_architecture_experiment.architectures "
            "must include at least one of: " + ", ".join(SUPPORTED_ARCHS)
        )

    results_dir = Path(exp_cfg["results_dir"])
    output_filename = str(exp_cfg["output_filename"])
    return architectures, results_dir, output_filename


def run_architecture_suite(architectures, results_dir: Path, output_filename: str):
    model_cfg = load_config("model_config")
    train_cfg = load_config("train_config")
    fl_cfg = load_config("fl_config")

    device = resolve_device(train_cfg["training"].get("device", "auto"))
    epochs = int(fl_cfg["federated"]["rounds"] * fl_cfg["federated"]["local_epochs"])
    eval_every = int(train_cfg["logging"].get("eval_every", 5))
    lr = float(train_cfg["training"]["learning_rate"])

    common_domains, selected_domains = select_fair_domains(fl_cfg)

    print(f"[device] Using {device}")
    print(
        f"[fairness] Using the same FL-selected train+val domains ({len(selected_domains)}): "
        f"{selected_domains}"
    )

    comparison = {}
    details = {}

    for arch in architectures:
        print("\n" + "=" * 70)
        print(f"CENTRALIZED GNN ARCH: {arch}")
        print("=" * 70)
        try:
            result = run_centralized_gnn(
                num_aps=model_cfg["encoder"]["num_aps"],
                latent_dim=model_cfg["encoder"]["latent_dim"],
                ap_emb_dim=model_cfg["encoder"]["ap_emb_dim"],
                pooling=model_cfg["encoder"]["pooling"],
                arch=arch,
                hidden_dim=model_cfg["gnn"]["hidden_dim"],
                gnn_layers=model_cfg["gnn"]["num_layers"],
                epochs=epochs,
                lr=lr,
                eval_every=eval_every,
                device=device,
                allowed_domains=selected_domains,
                experiment_name=f"centralized_gnn_{arch}",
            )
            comparison[arch] = result.get("global", {})
            details[arch] = {"status": "ok", "error": None}
            print(f"  {arch} global: {comparison[arch]}")
        except Exception as exc:
            comparison[arch] = {}
            details[arch] = {"status": "failed", "error": repr(exc)}
            print(f"  {arch} failed: {exc}")

    payload = {
        "comparison": comparison,
        "details": details,
        "architectures": architectures,
        "fairness": {
            "evaluation_domains": selected_domains,
            "num_domains": len(selected_domains),
            "device": device,
            "training_protocol": {
                "epochs": epochs,
                "learning_rate": lr,
                "eval_every": eval_every,
            },
            "domain_selection": {
                "all_common_domains": common_domains,
                "selected_domains": selected_domains,
                "selection_policy": {
                    "num_clients_per_round": fl_cfg["federated"].get("num_clients_per_round"),
                    "sampling_strategy": fl_cfg["federated"].get("sampling_strategy", "random"),
                    "seed": fl_cfg["federated"].get("seed", 42),
                },
            },
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / output_filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved architecture comparison to {out_path}")
    print("\n" + "=" * 78)
    print(f"{'Architecture':<14} {'Mean(m)':<10} {'Median(m)':<10} {'P75(m)':<10} {'P90(m)':<10}")
    print("-" * 78)
    for arch in architectures:
        metrics = comparison.get(arch, {})
        if metrics:
            print(
                f"{arch:<14} "
                f"{metrics.get('mean_error', float('nan')):<10.2f} "
                f"{metrics.get('median_error', float('nan')):<10.2f} "
                f"{metrics.get('p75_error', float('nan')):<10.2f} "
                f"{metrics.get('p90_error', float('nan')):<10.2f}"
            )
        else:
            print(f"{arch:<14} FAILED")
    print("=" * 78)


def main():
    cfg_architectures, results_dir, output_filename = load_experiment_cfg()

    parser = argparse.ArgumentParser(description="Compare centralized GNN architectures")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=cfg_architectures,
        choices=SUPPORTED_ARCHS,
        help="Architectures to evaluate",
    )
    args = parser.parse_args()

    run_architecture_suite(args.architectures, results_dir, output_filename)


if __name__ == "__main__":
    main()
