"""Normalize and center crop or pad 3D NumPy volumes."""

from __future__ import annotations

import numpy as np


def _as_float32_volume(volume: np.ndarray) -> np.ndarray:
    """Validate a volume and return an independent, safe float32 copy."""
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a NumPy ndarray")
    if volume.ndim != 3:
        raise ValueError("volume must be exactly 3D")
    if any(size == 0 for size in volume.shape):
        raise ValueError("volume dimensions must be non-empty")
    if volume.dtype.kind not in "iuf":
        raise TypeError("volume must have a real numeric dtype")
    if not np.isfinite(volume).all():
        raise ValueError("volume must contain only finite values")

    float32_max = np.finfo(np.float32).max
    if np.any(volume > float32_max) or np.any(volume < -float32_max):
        raise ValueError("volume values must fit in float32")
    return volume.astype(np.float32, copy=True)


def _validate_target_shape(target_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return target dimensions as Python integers after validation."""
    if not isinstance(target_shape, tuple) or len(target_shape) != 3:
        raise ValueError("target_shape must be a tuple of three positive integers")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
        for size in target_shape
    ):
        raise ValueError("target_shape must be a tuple of three positive integers")
    return tuple(int(size) for size in target_shape)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Scale a finite 3D real-valued volume to [0, 1] using whole-volume min/max.

    Values are converted to float32 before statistics are computed in float64.
    Constant volumes become zero arrays. The result is an independent float32
    array; input values are not modified.
    """
    source = _as_float32_volume(volume)
    values = source.astype(np.float64)
    minimum = values.min()
    span = values.max() - minimum
    if span == 0:
        return np.zeros(source.shape, dtype=np.float32)
    return ((values - minimum) / span).astype(np.float32)


def center_crop_or_pad(
    volume: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    """Center crop or zero-pad each axis to ``target_shape`` without normalization.

    For odd size differences, the extra removed or added voxel is at the
    high-index end of that axis. The independent float32 output has no spatial
    metadata; callers that export geometry must track crop or pad offsets.
    """
    source = _as_float32_volume(volume)
    target = _validate_target_shape(target_shape)

    source_slices = []
    target_slices = []
    for source_size, target_size in zip(source.shape, target):
        overlap = min(source_size, target_size)
        source_start = (source_size - overlap) // 2
        target_start = (target_size - overlap) // 2
        source_slices.append(slice(source_start, source_start + overlap))
        target_slices.append(slice(target_start, target_start + overlap))
    result = np.zeros(target, dtype=np.float32)
    result[tuple(target_slices)] = source[tuple(source_slices)]
    return result


def preprocess_volume(
    volume: np.ndarray, target_shape: tuple[int, int, int] | None = None
) -> np.ndarray:
    """Normalize a volume, then optionally center crop or zero-pad it.

    With no target shape, the normalized volume keeps its input shape. Crop or
    pad changes voxel indices; this array API does not update affine metadata.
    """
    normalized = normalize_volume(volume)
    if target_shape is None:
        return normalized
    return center_crop_or_pad(normalized, target_shape)
