"""Compare generated slices on the original array grid (no spatial resampling).

Inputs must already share spatial alignment and intensity scaling. Array shapes
cannot establish anatomical alignment; axis labels refer to NumPy, not anatomy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.interpolation import _validate_acquisition
from src.metrics import _validated_data_range, _validated_volumes
from src.sparse_simulator import SparseAcquisition


@dataclass(frozen=True)
class SliceComparison:
    """Missing-slice scores and pooled missing/all-voxel summaries."""

    axis: int
    retained_indices: np.ndarray
    missing_indices: np.ndarray
    baseline_per_slice: list[dict[str, float | int]]
    reconstruction_per_slice: list[dict[str, float | int]]
    baseline_missing_metrics: dict[str, float]
    reconstruction_missing_metrics: dict[str, float]
    baseline_all_metrics: dict[str, float]
    reconstruction_all_metrics: dict[str, float]


def missing_slice_indices(acquisition: SparseAcquisition) -> np.ndarray:
    """Return sorted original-grid positions absent from a valid acquisition."""
    _validate_acquisition(acquisition)
    missing = np.ones(acquisition.original_shape[acquisition.axis], dtype=bool)
    missing[acquisition.slice_indices] = False
    return np.flatnonzero(missing)


def _scores(prediction: np.ndarray, target: np.ndarray, data_range: float) -> dict[str, float]:
    """Score validated float64 arrays, pooling voxels before computing PSNR."""
    difference = prediction - target
    mse = float(np.mean(difference * difference))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "mse": mse,
        "psnr": float("inf") if mse == 0 else float(
            20 * np.log10(data_range) - 10 * np.log10(mse)
        ),
    }


def _inputs(
    target: np.ndarray,
    baseline: np.ndarray,
    reconstruction: np.ndarray,
    acquisition: SparseAcquisition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = missing_slice_indices(acquisition)
    baseline, target = _validated_volumes(baseline, target)
    reconstruction, _ = _validated_volumes(reconstruction, target)
    if target.shape != acquisition.original_shape:
        raise ValueError("volume shape must match acquisition.original_shape")
    if missing.size == 0:
        raise ValueError("acquisition has no missing slices to compare")
    return target, baseline, reconstruction, missing


def compare_slices(
    target: np.ndarray,
    baseline: np.ndarray,
    reconstruction: np.ndarray,
    acquisition: SparseAcquisition,
    *,
    data_range: float = 1.0,
) -> SliceComparison:
    """Score missing slices against the target, without clipping or normalization.

    Per-slice lists contain only missing positions, including the trailing tail.
    Summary PSNR is computed from pooled MSE, not averaged slice PSNR. Exact
    matches yield positive infinity. A valid singleton acquisition has no missing
    slices and is rejected here (the index helper returns an empty array).
    """
    data_range = _validated_data_range(data_range)
    target, baseline, reconstruction, missing = _inputs(
        target, baseline, reconstruction, acquisition
    )
    axis = int(acquisition.axis)
    missing_target = np.take(target, missing, axis=axis)

    def summarize(prediction: np.ndarray) -> tuple[list[dict[str, float | int]], dict[str, float], dict[str, float]]:
        per_slice = [
            {"slice_index": int(index), **_scores(
                np.take(prediction, index, axis=axis),
                np.take(target, index, axis=axis), data_range,
            )}
            for index in missing
        ]
        return (
            per_slice,
            _scores(np.take(prediction, missing, axis=axis), missing_target, data_range),
            _scores(prediction, target, data_range),
        )

    baseline_slices, baseline_missing, baseline_all = summarize(baseline)
    reconstruction_slices, reconstruction_missing, reconstruction_all = summarize(reconstruction)
    return SliceComparison(
        axis, acquisition.slice_indices.astype(np.intp, copy=True), missing,
        baseline_slices, reconstruction_slices, baseline_missing,
        reconstruction_missing, baseline_all, reconstruction_all,
    )


def save_slice_comparison(comparison: SliceComparison, output_path: str | Path) -> Path:
    """Write slice and summary scores to CSV; perfect-match PSNR is ``inf``."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "scope", "method", "axis", "slice_index", "mae", "mse", "psnr",
        ))
        writer.writeheader()
        for method in ("baseline", "reconstruction"):
            common = {"method": method, "axis": comparison.axis}
            for row in getattr(comparison, f"{method}_per_slice"):
                writer.writerow({**common, "scope": "missing_slice", **row})
            for scope in ("missing", "all"):
                writer.writerow({
                    **common, "scope": scope,
                    **getattr(comparison, f"{method}_{scope}_metrics"),
                })
    return destination


def render_slice_panels(
    target: np.ndarray,
    baseline: np.ndarray,
    reconstruction: np.ndarray,
    acquisition: SparseAcquisition,
    output_dir: str | Path,
    *,
    slice_indices: np.ndarray | None = None,
    data_range: float = 1.0,
) -> list[Path]:
    """Save one PNG per selected missing slice (default: up to five evenly spaced).

    Columns show target, baseline, reconstruction, and their two absolute errors.
    Intensities share [0, data_range]; errors share [0, largest selected error].
    Display clipping never changes scores. Planes are transposed with lower origin
    and labeled by array axes; these panels do not claim anatomical orientation.
    Matplotlib is loaded only when rendering and uses a headless canvas.
    """
    data_range = _validated_data_range(data_range)
    target, baseline, reconstruction, missing = _inputs(
        target, baseline, reconstruction, acquisition
    )
    if slice_indices is None:
        selected = missing[np.linspace(0, len(missing) - 1, min(5, len(missing)), dtype=int)]
    else:
        if not isinstance(slice_indices, np.ndarray):
            raise TypeError("slice_indices must be a NumPy ndarray")
        if slice_indices.ndim != 1 or slice_indices.dtype.kind not in "iu":
            raise ValueError("slice_indices must be a one-dimensional integer array")
        if not slice_indices.size or not np.isin(slice_indices, missing).all():
            raise ValueError("select at least one slice, all from missing positions")
        selected = np.unique(slice_indices)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    axis = int(acquisition.axis)
    planes = [np.moveaxis(volume, axis, 0) for volume in (target, baseline, reconstruction)]
    error_max = max(float(np.max(np.abs(p[selected] - planes[0][selected]))) for p in planes[1:])
    error_max = error_max or data_range
    in_plane = [index for index in range(3) if index != axis]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in selected:
        truth, base, recon = (p[index] for p in planes)
        figure = Figure(figsize=(16, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 5)
        images = (truth, base, recon, np.abs(base - truth), np.abs(recon - truth))
        titles = ("Target", "Baseline", "Reconstruction", "Baseline\nabsolute error", "Reconstruction\nabsolute error")
        for position, (panel, values, title) in enumerate(zip(axes, images, titles)):
            picture = panel.imshow(
                values.T, origin="lower", interpolation="nearest",
                cmap="gray" if position < 3 else "magma", vmin=0,
                vmax=data_range if position < 3 else error_max,
            )
            panel.set_title(title)
            panel.set_xlabel(f"Array axis {in_plane[0]}")
            panel.set_ylabel(f"Array axis {in_plane[1]}")
            figure.colorbar(picture, ax=panel, shrink=0.65)
        figure.suptitle(f"Missing slice {int(index)} · array axis {axis}")
        path = destination / f"axis_{axis}_slice_{int(index):04d}.png"
        figure.savefig(path, dpi=120)
        figure.clear()
        paths.append(path)
    return paths
