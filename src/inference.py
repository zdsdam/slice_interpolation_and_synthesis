"""Reconstruct dense MRI volumes from sparse slice acquisitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.interpolation import _validate_acquisition, interpolate_sparse_acquisition
from src.models.unet3d import ResidualUNet3D
from src.preprocessing import preprocess_volume
from src.sparse_simulator import SparseAcquisition


def _select_device(name: str) -> torch.device:
    """Select CPU or CUDA, with a clear error for unavailable CUDA."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested, but CUDA is unavailable")
    return torch.device(name)


def _checkpoint_file(path: str | Path) -> Path:
    """Resolve a checkpoint file, preferring best.pt in a directory."""
    checkpoint = Path(path)
    if checkpoint.is_dir():
        for name in ("best.pt", "latest.pt"):
            candidate = checkpoint / name
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"no best.pt or latest.pt checkpoint in {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint}")
    return checkpoint.resolve()


def read_checkpoint(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a checkpoint mapping safely and return it with its resolved path."""
    checkpoint_path = _checkpoint_file(path)
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint must contain a mapping")
    return loaded, checkpoint_path


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> ResidualUNet3D:
    """Build and strictly load a deployment model from minimal model fields."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a mapping")
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        missing = [key for key in ("model_config", "model_state_dict") if key not in checkpoint]
        raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")
    config = checkpoint["model_config"]
    if not isinstance(config, dict):
        raise ValueError("checkpoint model_config must be a mapping")
    base_channels = config.get("base_channels")
    if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
        raise ValueError("checkpoint model_config must include a positive base_channels value")
    try:
        model = ResidualUNet3D(**config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint model weights do not match model_config: {exc}") from exc
    model.to(device)
    model.eval()
    return model


def _tile_starts(length: int, tile: int) -> list[int]:
    """Return half-overlap starts, including a final tile anchored at the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile // 2))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def _validate_volume(volume: np.ndarray) -> np.ndarray:
    if not isinstance(volume, np.ndarray):
        raise TypeError("baseline must be a NumPy ndarray")
    if volume.ndim != 3 or any(size == 0 for size in volume.shape):
        raise ValueError("baseline must be a non-empty 3D volume")
    if volume.dtype.kind not in "iuf":
        raise TypeError("baseline must have a real numeric dtype")
    if not np.isfinite(volume).all():
        raise ValueError("baseline must contain only finite values")
    limit = np.finfo(np.float32).max
    if np.any(volume > limit) or np.any(volume < -limit):
        raise ValueError("baseline values must fit in float32")
    return volume.astype(np.float32, copy=False)


def _validate_tile_size(tile_size: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(tile_size, tuple) or len(tile_size) != 3 or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 8
        for size in tile_size
    ):
        raise ValueError("tile_size must contain three positive multiples of 8")
    return tile_size


def reconstruct_volume(
    model: nn.Module,
    baseline: np.ndarray,
    device: torch.device,
    tile_size: tuple[int, int, int] | None = (64, 64, 64),
) -> np.ndarray:
    """Predict a volume using overlap-averaged tiles or one padded full-volume pass.

    ``tile_size=None`` explicitly opts into a single forward pass over the whole
    volume. Otherwise tiles use nominal 50% overlap and high-edge anchoring.
    Smaller dimensions are edge-padded and every result is cropped to the input
    shape. Predictions are returned as float32 without clipping.
    """
    source = _validate_volume(baseline)
    original_shape = source.shape
    model.to(device)
    model.eval()

    if tile_size is None:
        padded_shape = tuple((size + 7) // 8 * 8 for size in original_shape)
        padding = tuple((0, target - size) for size, target in zip(original_shape, padded_shape))
        padded = np.pad(source, padding, mode="edge") if any(high for _, high in padding) else source
        with torch.inference_mode():
            prediction = model(torch.from_numpy(np.ascontiguousarray(padded))[None, None].to(device))
        expected_shape = (1, 1, *padded_shape)
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected_shape:
            raise ValueError(f"model output must have shape {expected_shape}")
        output = prediction[0, 0].cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError("model output must contain only finite values")
        return output[tuple(slice(0, size) for size in original_shape)].astype(np.float32, copy=True)

    tile_size = _validate_tile_size(tile_size)
    padded_shape = tuple(max(length, tile) for length, tile in zip(original_shape, tile_size))
    pad_width = tuple((0, padded - length) for length, padded in zip(original_shape, padded_shape))
    padded = np.pad(source, pad_width, mode="edge") if any(high for _, high in pad_width) else source
    total = np.zeros(padded_shape, dtype=np.float64)
    counts = np.zeros(padded_shape, dtype=np.float32)
    starts = [_tile_starts(length, tile) for length, tile in zip(padded_shape, tile_size)]
    expected_shape = (1, 1, *tile_size)
    with torch.inference_mode():
        for d in starts[0]:
            for h in starts[1]:
                for w in starts[2]:
                    slices = (slice(d, d + tile_size[0]), slice(h, h + tile_size[1]), slice(w, w + tile_size[2]))
                    tile = np.ascontiguousarray(padded[slices])
                    prediction = model(torch.from_numpy(tile)[None, None].to(device))
                    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected_shape:
                        raise ValueError(f"model output must have shape {expected_shape}")
                    values = prediction[0, 0].cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("model output must contain only finite values")
                    total[slices] += values
                    counts[slices] += 1
    output = (total / counts)[tuple(slice(0, length) for length in original_shape)]
    return output.astype(np.float32)


def _validate_samples(samples: np.ndarray) -> None:
    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy ndarray")
    if samples.dtype.kind not in "iuf":
        raise TypeError("samples must have a real numeric dtype")
    if not np.isfinite(samples).all():
        raise ValueError("samples must contain only finite values")
    limit = np.finfo(np.float32).max
    if np.any(samples > limit) or np.any(samples < -limit):
        raise ValueError("sample values must fit in float32")


def acquisition_from_samples(
    samples: np.ndarray,
    *,
    original_shape: tuple[int, int, int],
    axis: int,
    factor: int,
) -> SparseAcquisition:
    """Describe regular sparse samples with their original dense array shape.

    Sampling starts at index zero, matching :func:`simulate_sparse_acquisition`.
    This adapter handles array geometry only: callers retain NIfTI affine and
    header separately. A sparse image affine may encode the wider slice spacing
    and is not necessarily the dense-grid affine.
    """
    _validate_samples(samples)
    shape = tuple(original_shape) if isinstance(original_shape, (tuple, list)) else original_shape
    if not isinstance(shape, tuple) or len(shape) != 3 or any(
        isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1
        for size in shape
    ):
        raise ValueError("original_shape must contain three positive integers")
    shape = tuple(int(size) for size in shape)
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer from 0 through 2")
    if axis < 0 or axis > 2:
        raise ValueError("axis must be from 0 through 2")
    if isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)):
        raise TypeError("factor must be an integer greater than or equal to 2")
    if factor < 2:
        raise ValueError("factor must be greater than or equal to 2")
    axis, factor = int(axis), int(factor)
    acquisition = SparseAcquisition(
        samples=samples,
        slice_indices=np.arange(0, shape[axis], factor, dtype=np.intp),
        original_shape=shape,
        axis=axis,
        factor=factor,
    )
    _validate_acquisition(acquisition)
    return acquisition


def reconstruct_sparse(
    acquisition: SparseAcquisition,
    model_or_checkpoint: nn.Module | str | Path,
    *,
    device: str = "auto",
    normalize: bool = False,
    tile_size: tuple[int, int, int] | None = (64, 64, 64),
) -> np.ndarray:
    """Interpolate and reconstruct one sparse acquisition on its dense grid.

    With ``normalize=False`` (the default), samples are assumed to be on the
    same calibrated or normalized scale used for training and are not changed.
    With ``normalize=True``, sparse samples are globally min-max normalized
    using their observed extrema before interpolation; these can differ from
    the unavailable training HR extrema, and the output remains on this
    normalized scale. No inverse intensity scaling or clipping is applied.
    This array API neither accepts nor changes spatial metadata; callers keep
    their NIfTI affine and header separately.
    """
    _validate_acquisition(acquisition)
    _validate_samples(acquisition.samples)
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool")
    if normalize:
        # preprocess_volume owns the established normalization behavior.
        normalized = preprocess_volume(acquisition.samples)
        acquisition = SparseAcquisition(
            samples=normalized,
            slice_indices=acquisition.slice_indices.copy(),
            original_shape=acquisition.original_shape,
            axis=acquisition.axis,
            factor=acquisition.factor,
        )
    baseline = interpolate_sparse_acquisition(acquisition, method="linear")
    selected_device = _select_device(device)
    if isinstance(model_or_checkpoint, nn.Module):
        model = model_or_checkpoint.to(selected_device)
        model.eval()
    elif isinstance(model_or_checkpoint, (str, Path)):
        checkpoint, _ = read_checkpoint(model_or_checkpoint)
        model = model_from_checkpoint(checkpoint, selected_device)
    else:
        raise TypeError("model_or_checkpoint must be a torch.nn.Module, checkpoint path, or pathlib.Path")
    return reconstruct_volume(model, baseline, selected_device, tile_size)
