# src/fl/run_federated_training.py

import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from src.evaluation.tracker import ExperimentTracker
from src.evaluation.inference import evaluate_fl_model
from src.fl.utils import select_federated_client_ids


def _client_worker_main(client, in_queue, out_queue):
    while True:
        msg = in_queue.get()
        if msg is None:
            break
        if msg.get("type") != "train":
            continue

        params = msg.get("params")
        local_epochs = int(msg.get("local_epochs", 1))
        seed = msg.get("seed")

        if seed is not None:
            try:
                import torch

                torch.manual_seed(seed)
            except Exception:
                pass

        client.set_parameters(params)
        loss_local = None
        for _ in range(local_epochs):
            loss_local = client.train_one_epoch()

        out_queue.put(
            {
                "client_id": client.client_id,
                "loss": loss_local,
                "params": client.get_parameters(),
                "size": len(client.dataset),
            }
        )


class _ProcessClientPool:
    def __init__(self, clients, ctx):
        self._ctx = ctx
        self._in_queues = {}
        self._out_queues = {}
        self._procs = {}

        for client in clients:
            in_q = ctx.Queue()
            out_q = ctx.Queue()
            proc = ctx.Process(
                target=_client_worker_main,
                args=(client, in_q, out_q),
                daemon=True,
            )
            proc.start()
            self._in_queues[client.client_id] = in_q
            self._out_queues[client.client_id] = out_q
            self._procs[client.client_id] = proc

    def train_selected(self, selected_clients, global_params, local_epochs, seed=None):
        for client in selected_clients:
            self._in_queues[client.client_id].put(
                {
                    "type": "train",
                    "params": global_params,
                    "local_epochs": local_epochs,
                    "seed": seed,
                }
            )

        results = []
        for client in selected_clients:
            results.append(self._out_queues[client.client_id].get())

        return results

    def shutdown(self):
        for in_q in self._in_queues.values():
            in_q.put(None)
        for proc in self._procs.values():
            proc.join()


def run_federated_training(
    server,
    clients,
    rounds=10,
    local_epochs=1,
    num_clients_per_round=None,
    sampling_strategy="all",
    seed=None,
    tracker: Optional[ExperimentTracker] = None,
    val_datasets: Optional[Dict] = None,
    eval_every: int = 5,
    device: str = "cpu",
    parallel_clients: bool = False,
    max_workers: Optional[int] = None,
    parallel_backend: str = "thread",
):
    """
    Run federated training with tracking and periodic evaluation.
    
    Args:
        server: FLServer instance
        clients: List of FLClient instances
        rounds: Number of federated rounds
        local_epochs: Number of local epochs per round
        num_clients_per_round: Number of clients to sample (None = use all)
        sampling_strategy: 'random' or 'all'
        seed: Random seed for reproducibility
        tracker: ExperimentTracker for logging metrics
        val_datasets: {domain_id: FLQueryDataset} for periodic evaluation
        eval_every: Evaluate every N rounds (0 = no eval during training)
        device: device string
    """
    all_client_ids = [c.client_id for c in clients]
    selected_ids = select_federated_client_ids(
        client_ids=all_client_ids,
        num_clients_per_round=num_clients_per_round,
        sampling_strategy=sampling_strategy,
        seed=seed,
    )
    client_map = {c.client_id: c for c in clients}
    selected_clients = [client_map[cid] for cid in selected_ids]
    eval_datasets_selected = None
    if val_datasets is not None:
        eval_datasets_selected = {
            domain_id: dataset
            for domain_id, dataset in val_datasets.items()
            if domain_id in selected_ids
        }

    if sampling_strategy == "random" and num_clients_per_round is not None:
        print(
            f"Selected {len(selected_clients)}/{len(clients)} clients for this experiment: "
            f"{selected_ids}"
        )
    else:
        print(f"Training with all {len(selected_clients)} clients")
    
    print()
    
    parallel_backend = (parallel_backend or "thread").lower()
    use_process_pool = parallel_clients and parallel_backend == "process" and len(clients) > 1
    process_pool = None
    if use_process_pool:
        ctx = mp.get_context("spawn")
        process_pool = _ProcessClientPool(clients, ctx)

    for r in range(rounds):
        round_start = time.time()
        print(f"--- Round {r+1}/{rounds} ---")

        global_params = server.get_global_parameters()
        if not use_process_pool:
            for client in selected_clients:
                client.set_parameters(global_params)

        client_params = []
        client_sizes = []
        client_losses = {}

        def train_client_local(client):
            loss_local = None
            for _ in range(local_epochs):
                loss_local = client.train_one_epoch()
            params = client.get_parameters()
            size = len(client.dataset)
            return client.client_id, loss_local, params, size

        if use_process_pool:
            results = process_pool.train_selected(
                selected_clients,
                global_params,
                local_epochs,
                seed=seed,
            )
            for result in results:
                client_id = result["client_id"]
                loss = result["loss"]
                params = result["params"]
                size = result["size"]
                print(f"  Client {client_id} loss: {loss:.4f}")
                client_losses[client_id] = loss
                client_params.append(params)
                client_sizes.append(size)
        elif parallel_clients and len(selected_clients) > 1:
            worker_count = max_workers or len(selected_clients)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(train_client_local, client) for client in selected_clients]
                for future in as_completed(futures):
                    client_id, loss, params, size = future.result()
                    print(f"  Client {client_id} loss: {loss:.4f}")
                    client_losses[client_id] = loss
                    client_params.append(params)
                    client_sizes.append(size)
        else:
            for client in selected_clients:
                client_id, loss, params, size = train_client_local(client)
                print(f"  Client {client_id} loss: {loss:.4f}")
                client_losses[client_id] = loss
                client_params.append(params)
                client_sizes.append(size)

        server.aggregate(client_params, client_sizes)
        
        round_time = time.time() - round_start
        
        # Periodic evaluation
        eval_metrics = None
        if eval_datasets_selected and eval_every > 0 and ((r + 1) % eval_every == 0 or r == 0 or r == rounds - 1):
            print(f"  Evaluating global model...")
            eval_results = evaluate_fl_model(server, eval_datasets_selected, device=device)
            eval_metrics = eval_results.get("global", {})
            if eval_metrics:
                print(f"  Eval: mean_err={eval_metrics['mean_error']:.2f}m  "
                      f"median={eval_metrics['median_error']:.2f}m  "
                      f"p75={eval_metrics['p75_error']:.2f}m  "
                      f"CEP@5m={eval_metrics['cep_5m']:.1%}")
        
        # Log to tracker
        if tracker:
            tracker.log_round(
                round_num=r + 1,
                client_losses=client_losses,
                eval_metrics=eval_metrics,
                round_time=round_time,
            )
        
        avg_loss = sum(client_losses.values()) / len(client_losses)
        print(f"  Avg loss: {avg_loss:.4f} | Round time: {round_time:.1f}s\n")
    
    # Final save
    if tracker:
        tracker.save()

    if process_pool is not None:
        process_pool.shutdown()
    
    return tracker
