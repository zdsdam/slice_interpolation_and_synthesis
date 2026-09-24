"""Residual 3D U-Net for single-channel volume reconstruction."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock3D, DecoderBlock3D, EncoderBlock3D


class ResidualUNet3D(nn.Module):
    """Reconstruct a single-channel volume with a three-level residual U-Net.

    Inputs must have shape ``(N, 1, D, H, W)`` with positive spatial sizes
    divisible by eight. The network predicts an unconstrained residual and
    adds it to the input.
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")

        self.encoder1 = EncoderBlock3D(1, base_channels)
        self.encoder2 = EncoderBlock3D(base_channels, 2 * base_channels)
        self.encoder3 = EncoderBlock3D(2 * base_channels, 4 * base_channels)
        self.bottleneck = nn.Sequential(
            ConvBlock3D(4 * base_channels, 8 * base_channels),
            ConvBlock3D(8 * base_channels, 8 * base_channels),
        )
        self.decoder3 = DecoderBlock3D(8 * base_channels, 4 * base_channels, 4 * base_channels)
        self.decoder2 = DecoderBlock3D(4 * base_channels, 2 * base_channels, 2 * base_channels)
        self.decoder1 = DecoderBlock3D(2 * base_channels, base_channels, base_channels)
        self.residual_head = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input volume plus a learned single-channel residual."""
        if x.ndim != 5:
            raise ValueError("input must be a 5D (batch, channels, depth, height, width) tensor")
        if x.shape[1] != 1:
            raise ValueError(f"input has {x.shape[1]} channels; expected 1")
        if any(size <= 0 or size % 8 != 0 for size in x.shape[2:]):
            raise ValueError("input spatial dimensions must be positive and divisible by 8")

        skip1, down1 = self.encoder1(x)
        skip2, down2 = self.encoder2(down1)
        skip3, down3 = self.encoder3(down2)
        features = self.bottleneck(down3)
        features = self.decoder3(features, skip3)
        features = self.decoder2(features, skip2)
        features = self.decoder1(features, skip1)
        residual = self.residual_head(features)
        return x + residual
