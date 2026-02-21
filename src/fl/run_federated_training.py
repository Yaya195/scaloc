# src/fl/run_federated_training.py

import random


def run_federated_training(
    server,
    clients,
    rounds=10,
    local_epochs=1,
    num_clients_per_round=None,
    sampling_strategy="all",
    seed=None,
):
    """
    Run federated training with client sampling support.
    
    Args:
        server: FLServer instance
        clients: List of FLClient instances
        rounds: Number of federated rounds
        local_epochs: Number of local epochs per round
        num_clients_per_round: Number of clients to sample per round (None = use all)
        sampling_strategy: 'random' or 'all'
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    # Client sampling: done ONCE at the start of the experiment
    if sampling_strategy == "random" and num_clients_per_round is not None:
        selected_clients = random.sample(clients, min(num_clients_per_round, len(clients)))
        print(f"Selected {len(selected_clients)}/{len(clients)} clients for this experiment: {[c.client_id for c in selected_clients]}")
    else:
        selected_clients = clients
        print(f"Training with all {len(clients)} clients")
    
    print()  # Blank line for readability
    
    for r in range(rounds):
        print(f"--- Round {r+1}/{rounds} ---")

        global_params = server.get_global_parameters()
        for client in selected_clients:
            client.set_parameters(global_params)

        client_params = []
        client_sizes = []
        for client in selected_clients:
            for _ in range(local_epochs):
                loss = client.train_one_epoch()
            print(f"Client {client.client_id} loss: {loss:.4f}")
            client_params.append(client.get_parameters())
            client_sizes.append(len(client.dataset))

        server.aggregate(client_params, client_sizes)
        print("Global model updated.\n")
