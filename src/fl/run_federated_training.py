# src/fl/run_federated_training.py

import random
import time
from typing import Dict, Optional

from src.evaluation.tracker import ExperimentTracker
from src.evaluation.inference import evaluate_fl_model


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
    if seed is not None:
        random.seed(seed)
    
    # Client sampling: done ONCE at the start of the experiment
    if sampling_strategy == "random" and num_clients_per_round is not None:
        selected_clients = random.sample(clients, min(num_clients_per_round, len(clients)))
        print(f"Selected {len(selected_clients)}/{len(clients)} clients for this experiment: "
              f"{[c.client_id for c in selected_clients]}")
    else:
        selected_clients = clients
        print(f"Training with all {len(clients)} clients")
    
    print()
    
    for r in range(rounds):
        round_start = time.time()
        print(f"--- Round {r+1}/{rounds} ---")

        global_params = server.get_global_parameters()
        for client in selected_clients:
            client.set_parameters(global_params)

        client_params = []
        client_sizes = []
        client_losses = {}
        
        for client in selected_clients:
            for _ in range(local_epochs):
                loss = client.train_one_epoch()
            print(f"  Client {client.client_id} loss: {loss:.4f}")
            client_losses[client.client_id] = loss
            client_params.append(client.get_parameters())
            client_sizes.append(len(client.dataset))

        server.aggregate(client_params, client_sizes)
        
        round_time = time.time() - round_start
        
        # Periodic evaluation
        eval_metrics = None
        if val_datasets and eval_every > 0 and ((r + 1) % eval_every == 0 or r == 0 or r == rounds - 1):
            print(f"  Evaluating global model...")
            eval_results = evaluate_fl_model(server, val_datasets, device=device)
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
    
    return tracker
