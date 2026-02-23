import random
from typing import List, Optional


def select_federated_client_ids(
    client_ids: List[str],
    num_clients_per_round: Optional[int] = None,
    sampling_strategy: str = "all",
    seed: Optional[int] = None,
) -> List[str]:
    """Select client IDs once per experiment with reproducible sampling."""
    all_ids = sorted(client_ids)
    if not all_ids:
        return []

    strategy = (sampling_strategy or "all").lower()
    if strategy == "random" and num_clients_per_round is not None:
        keep = min(max(1, int(num_clients_per_round)), len(all_ids))
        rng = random.Random(seed)
        return sorted(rng.sample(all_ids, keep))

    return all_ids


def select_clients(client_ids: List[str], fraction: float) -> List[str]:
    """Backward-compatible simple prefix selector."""
    if not client_ids:
        return []
    keep = max(1, int(len(client_ids) * fraction))
    return sorted(client_ids)[:keep]
