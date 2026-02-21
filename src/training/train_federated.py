# src/fl/run_federated_training.py

def run_federated_training(server, clients, rounds=10, local_epochs=1):
    """
    Simple synchronous FedAvg loop.
    """
    for r in range(rounds):
        print(f"--- Round {r+1}/{rounds} ---")

        # 1) Broadcast global params to all clients
        global_params = server.get_global_parameters()
        for client in clients:
            client.set_parameters(global_params)

        # 2) Local training
        client_params = []
        client_sizes = []
        for client in clients:
            for _ in range(local_epochs):
                loss = client.train_one_epoch()
            print(f"Client {client.client_id} loss: {loss:.4f}")

            client_params.append(client.get_parameters())
            client_sizes.append(len(client.dataset))

        # 3) Aggregate
        new_global = server.aggregate(client_params, client_sizes)
        print("Global model updated.\n")