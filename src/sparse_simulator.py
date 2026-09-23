"""Simulate sparse through-plane MRI acquisition from a dense volume."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class SparseAcquisition:
    """Sparse samples and the geometry needed to locate them in the source."""

    samples: np.ndarray
    slice_indices: np.ndarray
    original_shape: tuple[int, int, int]
    axis: int
    factor: int


def simulate_sparse_acquisition(
    volume: np.ndarray,
    factor: int,
    *,
    axis: int = 2,
    sigma: float = 1.0,
) -> SparseAcquisition:
    """Blur one volume along ``axis`` and retain every ``factor``-th slice.

    ``axis`` follows NumPy's array-axis convention (default: the last axis),
    and may be specified from -3 through 2. ``factor`` must be at least two.
    ``sigma`` is measured in input voxels along that axis; its default of 1.0
    gives modest voxel-scale smoothing, while zero disables blurring. The
    Gaussian uses reflected boundary values. Sampling starts at slice zero, so
    ``samples`` has a reduced selected dimension and contains no placeholders
    for missing slices. Samples use float32 and the input is unchanged.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a NumPy ndarray")
    if volume.ndim != 3:
        raise ValueError(f"volume must be exactly 3D; got {volume.ndim}D")
    if any(size == 0 for size in volume.shape):
        raise ValueError("volume dimensions must all be non-empty")
    if volume.dtype.kind not in "iuf":
        raise TypeError("volume must have a real numeric dtype (integer or float)")

    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)):
        raise TypeError("factor must be an integer greater than or equal to 2")
    if factor < 2:
        raise ValueError("factor must be greater than or equal to 2")
    factor = int(factor)

    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer from -3 through 2")
    if axis < -3 or axis > 2:
        raise ValueError("axis must be between -3 and 2 for a 3D volume")
    axis = int(axis) % volume.ndim

    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real):
        raise TypeError("sigma must be a finite non-negative real number")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite non-negative real number")
    sigma = float(sigma)

    original_shape = tuple(int(size) for size in volume.shape)
    processed = volume.astype(np.float32, copy=True)
    if sigma > 0:
        processed = gaussian_filter1d(
            processed,
            sigma=sigma,
            axis=axis,
            mode="reflect",
            output=np.float32,
        )

    slice_indices = np.arange(0, original_shape[axis], factor, dtype=np.intp)
    samples = np.take(processed, slice_indices, axis=axis)
    return SparseAcquisition(
        samples=samples,
        slice_indices=slice_indices,
        original_shape=original_shape,
        axis=axis,
        factor=factor,
    )
