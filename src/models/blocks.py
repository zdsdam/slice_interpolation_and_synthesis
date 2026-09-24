"""Small reusable building blocks for a 3D residual U-Net."""

from __future__ import annotations

import torch
from torch import nn


def _require_5d_input(x: torch.Tensor, name: str = "input") -> None:
    """Require a batched (N, C, D, H, W) tensor."""
    if x.ndim != 5:
        raise ValueError(f"{name} must be a 5D (batch, channels, depth, height, width) tensor")


class ConvBlock3D(nn.Module):
    """Apply a padded 3D convolution followed by ReLU, preserving spatial size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_5d_input(x)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"input has {x.shape[1]} channels; expected {self.in_channels}")
        return self.activation(self.conv(x))


class EncoderBlock3D(nn.Module):
    """Create skip features and halve spatial size with 2x2x2 max pooling.

    Each spatial input dimension must be at least 2. Pooling floors odd sizes;
    U-Net patches should normally be divisible by powers of two so decoder
    upsampling matches skip sizes exactly.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_5d_input(x)
        if any(size < 2 for size in x.shape[2:]):
            raise ValueError("encoder spatial dimensions must each be at least 2")
        skip = self.conv2(self.conv1(x))
        return skip, self.pool(skip)


class DecoderBlock3D(nn.Module):
    """Upsample, concatenate an encoder skip, then refine with two convolutions.

    The skip must match the input batch and have exactly twice its spatial
    dimensions. The block does not crop or pad mismatched feature maps.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip_channels = skip_channels
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock3D(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock3D(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        _require_5d_input(x, "decoder input")
        _require_5d_input(skip, "skip input")
        if skip.shape[1] != self.skip_channels:
            raise ValueError(f"skip input has {skip.shape[1]} channels; expected {self.skip_channels}")
        upsampled = self.up(x)
        if upsampled.shape[0] != skip.shape[0] or upsampled.shape[2:] != skip.shape[2:]:
            raise ValueError(
                "upsampled decoder input and skip input must have matching batch and spatial dimensions"
            )
        combined = torch.cat((upsampled, skip), dim=1)
        return self.conv2(self.conv1(combined))
