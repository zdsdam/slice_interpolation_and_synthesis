"""Reconstruction losses for 3D volume synthesis."""

import torch
from torch.nn import functional as F


def reconstruction_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean L1 error for a reconstructed volume and its HR target.

    ``prediction`` is the final reconstructed volume (input plus network
    residual), and ``target`` is the high-resolution volume. They typically
    have shape ``(B, 1, D, H, W)``. Equal shapes are required; the mean is
    computed over all elements and returns a scalar tensor.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have matching shapes; got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    return F.l1_loss(prediction, target, reduction="mean")
