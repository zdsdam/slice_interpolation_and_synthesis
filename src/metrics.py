"""Scalar quality metrics for three-dimensional MRI volumes.

PSNR and SSIM require an explicit ``data_range`` equal to the expected
maximum-minus-minimum intensity range (use 1.0 for normalized [0, 1] MRI).
The range is never inferred from the current volumes; inputs are not clipped
or renormalized, so prediction overshoots contribute to the error.
"""

from __future__ import annotations

from numbers import Real

import numpy as np


def _validated_volumes(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate matching real-valued volumes and return float64 views or copies."""
    arrays = (prediction, target)
    for name, array in zip(("prediction", "target"), arrays):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy ndarray")
        if array.ndim != 3:
            raise ValueError(f"{name} must be exactly three-dimensional")
        if any(length == 0 for length in array.shape):
            raise ValueError(f"{name} must be nonempty")
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must have a real numeric dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    return prediction.astype(np.float64, copy=False), target.astype(
        np.float64, copy=False
    )


def _validated_data_range(data_range: Real) -> float:
    """Return a finite positive real data range."""
    if isinstance(data_range, (bool, np.bool_)) or not isinstance(data_range, Real):
        raise TypeError("data_range must be a real number")
    value = float(data_range)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("data_range must be finite and positive")
    return value


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return mean absolute voxel error for matching 3D volumes."""
    prediction, target = _validated_volumes(prediction, target)
    return float(np.mean(np.abs(prediction - target), dtype=np.float64))


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return mean squared voxel error for matching 3D volumes."""
    prediction, target = _validated_volumes(prediction, target)
    difference = prediction - target
    return float(np.mean(difference * difference, dtype=np.float64))


def psnr(
    prediction: np.ndarray, target: np.ndarray, *, data_range: float
) -> float:
    """Return PSNR in dB; exact matches have positive infinite PSNR."""
    data_range = _validated_data_range(data_range)
    mean_squared_error = mse(prediction, target)
    if mean_squared_error == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mean_squared_error))


def ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float,
    win_size: int = 7,
) -> float:
    """Return true volumetric SSIM over valid centers of uniform 3D windows.

    Uses scikit-image's default constants (K1=0.01, K2=0.03), uniform windows,
    and sample covariance. Border centers without a full ``win_size`` cube are
    excluded from the scalar mean. ``data_range`` is the expected intensity
    maximum-minus-minimum (use 1.0 for normalized [0, 1] MRI); it is not inferred
    from the inputs, which are neither clipped nor renormalized.
    """
    data_range = _validated_data_range(data_range)
    prediction, target = _validated_volumes(prediction, target)
    if isinstance(win_size, (bool, np.bool_)) or not isinstance(
        win_size, (int, np.integer)
    ):
        raise TypeError("win_size must be an integer")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be an odd integer of at least 3")
    if any(length < win_size for length in prediction.shape):
        raise ValueError("win_size must fit along every volume axis")

    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            prediction,
            target,
            data_range=data_range,
            channel_axis=None,
            gaussian_weights=False,
            win_size=int(win_size),
            use_sample_covariance=True,
        )
    )
