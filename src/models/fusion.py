import torch


def fuse_features(*features: torch.Tensor) -> torch.Tensor:
    """Concatenate modality features along the last dimension."""
    if len(features) == 1:
        return features[0]
    return torch.cat(features, dim=-1)
