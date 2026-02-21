def select_clients(client_ids: list[str], fraction: float) -> list[str]:
    """Select a prefix subset of clients as a simple baseline."""
    if not client_ids:
        return []
    keep = max(1, int(len(client_ids) * fraction))
    return client_ids[:keep]
