# src/evaluation/inference.py
#
# Inference pipeline for the Federated GNN Indoor Localization system.
#
# Pipeline (spec §7):
#   1. Domain selection (strongest AP or classifier)
#   2. Encode query fingerprint via AP-wise encoder
#   3. Inject query into domain graph
#   4. Run GNN
#   5. Output continuous position

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.evaluation.metrics import compute_all_metrics


def _pack_rp_fingerprints(fingerprints, device: str):
    num_rps = len(fingerprints)
    max_len = max((len(fp["ap_ids"]) for fp in fingerprints), default=0)

    if num_rps == 0 or max_len == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0, 0, 1), dtype=torch.float, device=device),
            torch.empty((0, 0), dtype=torch.bool, device=device),
        )

    ap_ids = torch.zeros((num_rps, max_len), dtype=torch.long)
    rssi = torch.zeros((num_rps, max_len, 1), dtype=torch.float)
    mask = torch.zeros((num_rps, max_len), dtype=torch.bool)

    for row, fp in enumerate(fingerprints):
        length = len(fp["ap_ids"])
        if length == 0:
            continue
        ap_ids[row, :length] = torch.tensor(fp["ap_ids"], dtype=torch.long)
        rssi[row, :length, 0] = torch.tensor(fp["rssi"], dtype=torch.float)
        mask[row, :length] = True

    return (
        ap_ids.to(device, non_blocking=True),
        rssi.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
    )


def select_domain_by_strongest_ap(
    query_aps: List[dict],
    domain_ap_sets: Dict[str, set],
) -> str:
    """
    Select domain by finding which domain's AP set has the most overlap
    with the query's visible APs, weighted by RSSI strength.
    
    Args:
        query_aps: list of {"ap_id": int, "rssi": float}
        domain_ap_sets: {domain_id: set of AP IDs in that domain}
    
    Returns:
        Best matching domain_id
    """
    best_domain = None
    best_score = -float("inf")
    
    for domain_id, ap_set in domain_ap_sets.items():
        score = 0.0
        for ap in query_aps:
            if ap["ap_id"] in ap_set:
                # RSSI is negative; stronger = closer to 0 = higher score
                score += -ap["rssi"]  # invert so stronger signal = higher score
        
        if score > best_score:
            best_score = score
            best_domain = domain_id
    
    return best_domain


@torch.no_grad()
def predict_position(
    encoder,
    model,
    graph,
    ap_ids: torch.Tensor,
    rssi: torch.Tensor,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference for a single query.
    
    Args:
        encoder: APWiseEncoder
        model: FLIndoorModel
        graph: PyG Data for the domain
        ap_ids: (N,) AP index tensor
        rssi: (N, 1) RSSI tensor
        device: device string
    
    Returns:
        p_hat_normalized: (2,) predicted position in normalized space
        weights: (num_nodes,) attention weights over RPs
    """
    encoder = encoder.to(device)
    model = model.to(device)
    encoder.eval()
    model.eval()
    
    graph = graph.to(device)
    ap_ids = ap_ids.to(device)
    rssi = rssi.to(device)
    
    # Re-encode RP features with current encoder
    rp_ap_ids, rp_rssi, rp_mask = _pack_rp_fingerprints(graph.rp_fingerprints, device)
    rp_feats = encoder.encode_packed(rp_ap_ids, rp_rssi, rp_mask)
    
    graph.x = rp_feats
    
    # Encode query
    z_q = encoder(ap_ids, rssi)
    
    # Run GNN
    p_hat, w = model(graph, z_q)
    
    return p_hat.cpu().numpy(), w.cpu().numpy()


def denormalize_position(
    p_normalized: np.ndarray,
    coord_min: np.ndarray,
    coord_max: np.ndarray,
) -> np.ndarray:
    """Convert normalized [0,1] position back to metric coordinates."""
    coord_range = coord_max - coord_min
    return p_normalized * coord_range + coord_min


@torch.no_grad()
def evaluate_model_on_domain(
    encoder,
    model,
    dataset,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on all queries in a domain dataset.
    
    Returns:
        preds: (N, 2) predicted positions in METRIC space (denormalized)
        truths: (N, 2) true positions in METRIC space (denormalized)
    """
    encoder = encoder.to(device)
    model = model.to(device)
    encoder.eval()
    model.eval()
    
    graph = dataset.graph.to(device)
    coord_min = dataset.coord_min
    coord_max = dataset.coord_max
    
    # Re-encode graph with current encoder
    rp_ap_ids, rp_rssi, rp_mask = _pack_rp_fingerprints(graph.rp_fingerprints, device)
    rp_feats = encoder.encode_packed(rp_ap_ids, rp_rssi, rp_mask)
    
    graph.x = rp_feats
    
    preds = []
    truths = []
    
    for i in range(len(dataset)):
        ap_ids, rssi, pos_norm = dataset[i]
        ap_ids = ap_ids.to(device)
        rssi = rssi.to(device)
        
        z_q = encoder(ap_ids, rssi)
        p_hat, _ = model(graph, z_q)
        
        # Denormalize
        p_hat_metric = denormalize_position(p_hat.cpu().numpy(), coord_min, coord_max)
        p_true_metric = denormalize_position(pos_norm.numpy(), coord_min, coord_max)
        
        preds.append(p_hat_metric)
        truths.append(p_true_metric)
    
    return np.stack(preds), np.stack(truths)


def evaluate_fl_model(
    server,
    val_datasets: Dict[str, "FLQueryDataset"],
    device: str = "cpu",
) -> Dict[str, dict]:
    """
    Evaluate the global FL model on validation data across all domains.
    
    Args:
        server: FLServer with global model/encoder
        val_datasets: {domain_id: FLQueryDataset} for validation
        device: device string
    
    Returns:
        {domain_id: metrics_dict} + {"global": aggregated metrics}
    """
    encoder = server.global_encoder
    model = server.global_model
    
    all_preds = []
    all_truths = []
    results = {}
    
    for domain_id, dataset in val_datasets.items():
        if len(dataset) == 0:
            continue
        
        preds, truths = evaluate_model_on_domain(encoder, model, dataset, device)
        metrics = compute_all_metrics(preds, truths)
        results[domain_id] = metrics
        
        all_preds.append(preds)
        all_truths.append(truths)
    
    # Global metrics
    if all_preds:
        global_preds = np.concatenate(all_preds)
        global_truths = np.concatenate(all_truths)
        results["global"] = compute_all_metrics(global_preds, global_truths)
    
    return results
