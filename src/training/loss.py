import torch


def localization_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error for 2D coordinate regression."""
    return torch.mean((pred - target) ** 2)
