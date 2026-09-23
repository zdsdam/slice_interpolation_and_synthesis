"""Reconstruct sparse MRI samples on their original dense grid."""

import numpy as np

from src.sparse_simulator import SparseAcquisition


def _validate_acquisition(acquisition: SparseAcquisition) -> None:
    """Check that sparse samples and their simulator metadata agree."""
    if not isinstance(acquisition, SparseAcquisition):
        raise TypeError("acquisition must be a SparseAcquisition")

    shape = acquisition.original_shape
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < 1
            for size in shape
        )
    ):
        raise ValueError("original_shape must contain three positive integers")
    shape = tuple(int(size) for size in shape)

    axis = acquisition.axis
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer from 0 through 2")
    if axis < 0 or axis > 2:
        raise ValueError("axis must be from 0 through 2")
    axis = int(axis)

    factor = acquisition.factor
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)):
        raise TypeError("factor must be an integer greater than or equal to 2")
    if factor < 2:
        raise ValueError("factor must be greater than or equal to 2")
    factor = int(factor)

    samples = acquisition.samples
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy ndarray")
    if samples.ndim != 3:
        raise ValueError("samples must be exactly 3D")
    if samples.dtype.kind not in "iuf":
        raise TypeError("samples must have a real numeric dtype")
    expected_indices = np.arange(0, shape[axis], factor, dtype=np.intp)
    expected_sample_shape = list(shape)
    expected_sample_shape[axis] = len(expected_indices)
    if samples.shape != tuple(expected_sample_shape):
        raise ValueError("samples shape is inconsistent with original_shape and axis")

    indices = acquisition.slice_indices
    if not isinstance(indices, np.ndarray):
        raise TypeError("slice_indices must be a NumPy ndarray")
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("slice_indices must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= shape[axis]):
        raise ValueError("slice_indices must be within the original slice axis")
    if len(indices) == 0 or np.any(np.diff(indices.astype(np.int64)) <= 0):
        raise ValueError("slice_indices must be non-empty and strictly increasing")
    if not np.array_equal(indices, expected_indices):
        raise ValueError("slice_indices must match the simulator's regular sampling")


def interpolate_sparse_acquisition(
    acquisition: SparseAcquisition, method: str = "linear"
) -> np.ndarray:
    """Fill missing through-plane slices using linear or nearest interpolation.

    Linear interpolation uses each retained slice's actual source index. Values
    outside the sampled range are extended from the nearest endpoint; canonical
    simulator acquisitions start at zero, so this only applies to a trailing
    remainder. Nearest ties select the lower-index sampled slice. The returned
    array has ``original_shape``, is float32, and does not share memory with the
    input samples.
    """
    _validate_acquisition(acquisition)
    if not isinstance(method, str) or method not in ("linear", "nearest"):
        raise ValueError("method must be 'linear' or 'nearest'")

    axis = int(acquisition.axis)
    size = acquisition.original_shape[axis]
    indices = acquisition.slice_indices.astype(np.intp, copy=False)
    source = np.moveaxis(acquisition.samples, axis, 0).astype(np.float64, copy=False)
    result = np.empty((size, *source.shape[1:]), dtype=np.float32)

    if method == "linear":
        for position in range(size):
            right = int(np.searchsorted(indices, position, side="left"))
            if right == 0:
                plane = source[0]
            elif right == len(indices):
                plane = source[-1]
            elif indices[right] == position:
                plane = source[right]
            else:
                left = right - 1
                weight = (position - indices[left]) / (indices[right] - indices[left])
                plane = source[left] * (1.0 - weight) + source[right] * weight
            result[position] = plane
    else:
        right = np.searchsorted(indices, np.arange(size), side="left")
        right = np.clip(right, 0, len(indices) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.arange(size) - indices[left] <= indices[right] - np.arange(size)
        nearest = np.where(choose_left, left, right)
        result[...] = source[nearest]

    # Assign measured locations last so interpolation never perturbs samples.
    result[indices] = source
    return np.moveaxis(result, 0, axis)
