# src/baselines/federated_mlp.py
#
# Federated MLP baseline (spec §8).
# Same MLP architecture as centralized, but trained via FedAvg
# with each domain as a separate client. No GNN, no graphs.

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Optional
import multiprocessing as mp
from queue import Empty
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.baselines.centralized_mlp import LocalizationMLP, prepare_data
from src.data.normalization import (
    denormalize_coords,
    fit_domain_normalization_stats,
)
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.tracker import ExperimentTracker
from src.fl.utils import select_federated_client_ids
from src.utils.device import resolve_device


def _client_seed(base_seed, client_id: str):
    if base_seed is None:
        return None
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(client_id)))
    return int(base_seed) + int(offset)


def _mlp_worker_main(
    domain_id,
    X_train,
    y_train,
    model_kwargs,
    lr,
    batch_size,
    in_queue,
    out_queue,
):
    model = LocalizationMLP(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    while True:
        msg = in_queue.get()
        if msg is None:
            break
        if msg.get("type") != "train":
            continue

        global_state = msg.get("state")
        local_epochs = int(msg.get("local_epochs", 1))
        seed = msg.get("seed")

        if seed is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass

        try:
            model.load_state_dict(global_state)
            model.train()

            ds = TensorDataset(X_train, y_train)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            loss_val = 0.0
            for _ in range(local_epochs):
                epoch_loss = 0.0
                n = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n += 1
                loss_val = epoch_loss / max(n, 1)

            out_queue.put(
                {
                    "domain_id": domain_id,
                    "loss": loss_val,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "size": len(X_train),
                }
            )
        except Exception as exc:
            out_queue.put(
                {
                    "domain_id": domain_id,
                    "error": repr(exc),
                }
            )


class _MLPProcessPool:
    def __init__(self, domain_data, model_kwargs, lr, batch_size, ctx):
        self._in_queues = {}
        self._out_queues = {}
        self._procs = {}

        for domain_id, (X_train, y_train) in domain_data.items():
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            proc = ctx.Process(
                target=_mlp_worker_main,
                args=(domain_id, X_train, y_train, model_kwargs, lr, batch_size, in_q, out_q),
                daemon=True,
            )
            proc.start()
            self._in_queues[domain_id] = in_q
            self._out_queues[domain_id] = out_q
            self._procs[domain_id] = proc

    def train_selected(self, selected_ids, global_state, local_epochs, seed=None):
        for domain_id in selected_ids:
            domain_seed = _client_seed(seed, domain_id)
            self._in_queues[domain_id].put(
                {
                    "type": "train",
                    "state": global_state,
                    "local_epochs": local_epochs,
                    "seed": domain_seed,
                }
            )

        results = []
        for domain_id in selected_ids:
            try:
                results.append(self._out_queues[domain_id].get(timeout=1800))
            except Empty:
                proc = self._procs.get(domain_id)
                alive = bool(proc is not None and proc.is_alive())
                raise RuntimeError(
                    f"Timeout waiting for federated_mlp worker {domain_id} (alive={alive})"
                )
        return results

    def shutdown(self):
        for in_q in self._in_queues.values():
            in_q.put(None)
        for proc in self._procs.values():
            proc.join()


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
    parallel_clients: bool = False,
    parallel_backend: str = "thread",
    max_workers: Optional[int] = None,
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
    parallel_backend = (parallel_backend or "thread").lower()

    if parallel_clients and str(device).startswith("cuda"):
        print("  Federated MLP: CUDA detected, disabling parallel clients for stability.")
        parallel_clients = False

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

    use_process_pool = (
        parallel_clients
        and parallel_backend == "process"
        and device == "cpu"
        and len(selected_ids) > 1
    )
    process_pool = None
    if use_process_pool:
        ctx = mp.get_context("spawn")
        model_kwargs = {
            "input_dim": num_aps,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        domain_data_cpu = {
            d: (domain_data[d][0].cpu(), domain_data[d][1].cpu()) for d in selected_ids
        }
        process_pool = _MLPProcessPool(domain_data_cpu, model_kwargs, lr, batch_size, ctx)

    for r in range(rounds):
        round_seed = (seed + r) if seed is not None else None
        client_states = []
        client_sizes = []
        client_losses = {}
        client_update_ids = []

        if use_process_pool:
            process_failed = False
            try:
                results = process_pool.train_selected(
                    selected_ids,
                    global_model.state_dict(),
                    local_epochs,
                    seed=round_seed,
                )
            except Exception as exc:
                print(f"  Federated MLP process backend failed: {exc}. Falling back to serial this round.")
                results = []
                process_failed = True
                use_process_pool = False
            for result in results:
                domain_id = result["domain_id"]
                if "error" in result:
                    print(f"  Client {domain_id} failed in process worker: {result['error']}")
                    continue
                if not math.isfinite(float(result["loss"])):
                    print(f"  Client {domain_id} produced non-finite loss ({result['loss']}); skipping update")
                    continue
                client_losses[domain_id] = result["loss"]
                client_states.append(result["state"])
                client_sizes.append(result["size"])
                client_update_ids.append(domain_id)

            if process_failed:
                for domain_id in selected_ids:
                    if round_seed is not None:
                        torch.manual_seed(_client_seed(round_seed, domain_id))
                    X_train, y_train = domain_data[domain_id]

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

                    if not math.isfinite(float(loss_val)):
                        print(f"  Client {domain_id} produced non-finite loss ({loss_val}); skipping update")
                        continue
                    client_losses[domain_id] = loss_val
                    client_states.append({k: v.clone() for k, v in local_model.state_dict().items()})
                    client_sizes.append(len(X_train))
                    client_update_ids.append(domain_id)
        elif parallel_clients and len(selected_ids) > 1:
            worker_count = max_workers or len(selected_ids)

            def train_one(domain_id):
                if round_seed is not None:
                    torch.manual_seed(_client_seed(round_seed, domain_id))
                X_train, y_train = domain_data[domain_id]
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

                return domain_id, loss_val, {k: v.clone() for k, v in local_model.state_dict().items()}, len(X_train)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(train_one, domain_id) for domain_id in selected_ids]
                for future in as_completed(futures):
                    try:
                        domain_id, loss_val, state, size = future.result()
                    except Exception as exc:
                        print(f"  Client thread failed: {exc}")
                        continue
                    if not math.isfinite(float(loss_val)):
                        print(f"  Client {domain_id} produced non-finite loss ({loss_val}); skipping update")
                        continue
                    client_losses[domain_id] = loss_val
                    client_states.append(state)
                    client_sizes.append(size)
                    client_update_ids.append(domain_id)
        else:
            for domain_id in selected_ids:
                if round_seed is not None:
                    torch.manual_seed(_client_seed(round_seed, domain_id))
                X_train, y_train = domain_data[domain_id]

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
                client_update_ids.append(domain_id)

        if not client_states:
            print(f"  Round {r+1}/{rounds}: no valid client updates; skipping aggregation")
            tracker.log_round(r + 1, client_losses, None)
            continue

        if len(client_update_ids) > 1:
            order = sorted(range(len(client_update_ids)), key=lambda i: client_update_ids[i])
            client_states = [client_states[i] for i in order]
            client_sizes = [client_sizes[i] for i in order]

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

    if process_pool is not None:
        process_pool.shutdown()

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
