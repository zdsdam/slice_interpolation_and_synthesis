"""Evaluate interpolation and a trained 3D U-Net on reserved test subjects.

The checked-in test manifest is reserved from the original 24 subjects with
seed 42 and fraction 0.2. Train using ``python -m src.train --manifest
data/development_subjects.csv``; training then makes its own train/validation
split within those 19 development subjects. Evaluation defaults to
``data/test_subjects.csv`` and rejects checkpoints that used any test subject
or test file during training or validation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data_loader import load_nifti
from src.interpolation import interpolate_sparse_acquisition
from src.metrics import mae, mse, psnr, ssim
from src.inference import (
    _checkpoint_file,
    _select_device,
    model_from_checkpoint,
    read_checkpoint,
    reconstruct_volume,
)
from src.preprocessing import preprocess_volume
from src.sparse_simulator import simulate_sparse_acquisition
from src.train import read_manifest


_METRIC_NAMES = ("mae", "mse", "psnr", "ssim")
_ROW_FIELDS = (
    "scope", "subject_id", "path", "method", *_METRIC_NAMES,
    "checkpoint_path", "checkpoint_epoch", "sparsity_factor", "slice_axis",
    "sigma", "data_range", "tile_size", "tiling_strategy", "ssim_definition",
)


def _required_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Validate checkpoint structure and return its evaluation provenance."""
    required = (
        "model_state_dict", "model_config", "training_config", "train_subjects",
        "validation_subjects", "manifest_paths",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint {checkpoint_path} is missing required fields: {', '.join(missing)}")
    model_config, settings = checkpoint["model_config"], checkpoint["training_config"]
    if not isinstance(model_config, dict) or not isinstance(settings, dict):
        raise ValueError("checkpoint model_config and training_config must be mappings")
    base_channels = model_config.get("base_channels")
    if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 1:
        raise ValueError("checkpoint model_config must include a positive base_channels value")
    for key in ("sparsity_factor", "slice_axis", "sigma", "patch_size"):
        if key not in settings:
            raise ValueError(f"checkpoint training_config is missing required setting {key!r}")
    for key in ("train_subjects", "validation_subjects"):
        ids = checkpoint[key]
        if not isinstance(ids, (list, tuple)) or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError(f"checkpoint {key} must be a list of subject IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"checkpoint {key} contains duplicate subject IDs")
    train_ids, validation_ids = set(checkpoint["train_subjects"]), set(checkpoint["validation_subjects"])
    if not train_ids:
        raise ValueError("checkpoint train_subjects must contain at least one subject")
    if train_ids & validation_ids:
        raise ValueError("checkpoint training and validation subjects overlap")
    paths = checkpoint["manifest_paths"]
    if not isinstance(paths, dict):
        raise ValueError("checkpoint manifest_paths must map subject IDs to file paths")
    known_ids = train_ids | validation_ids
    if set(paths) != known_ids:
        raise ValueError("checkpoint manifest_paths IDs must exactly match train_subjects and validation_subjects")
    for subject_id in known_ids:
        if not isinstance(paths[subject_id], (list, tuple)) or not paths[subject_id]:
            raise ValueError(f"checkpoint manifest_paths has no paths for subject {subject_id!r}")
    patch_size = settings["patch_size"]
    if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
        raise ValueError("checkpoint patch_size must contain three dimensions")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 8 for size in patch_size):
        raise ValueError("checkpoint patch_size dimensions must be positive multiples of 8")
    factor, axis, sigma = settings["sparsity_factor"], settings["slice_axis"], settings["sigma"]
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
        raise ValueError("checkpoint sparsity_factor must be an integer of at least 2")
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
        raise ValueError("checkpoint slice_axis must be 0, 1, or 2")
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("checkpoint sigma must be finite and non-negative")
    return checkpoint


def _resolved_source_path(raw_path: str, checkpoint_path: Path) -> Path:
    """Resolve saved paths relative to cwd, then checkpoint location as fallback."""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (checkpoint_path.parent / candidate).resolve()


def _validate_test_split(
    subjects: dict[str, list[Path]], checkpoint: dict[str, Any], checkpoint_path: Path
) -> None:
    """Reject subject and source-file overlap with checkpoint train or validation data."""
    test_ids = set(subjects)
    train_ids = set(checkpoint["train_subjects"])
    validation_ids = set(checkpoint["validation_subjects"])
    leaked_ids = sorted(test_ids & (train_ids | validation_ids))
    if leaked_ids:
        raise ValueError(
            "test subjects overlap checkpoint training or validation subjects: "
            f"{', '.join(leaked_ids)}. Train with data/development_subjects.csv; "
            "the default training manifest includes the reserved test subjects."
        )
    source_paths: set[Path] = set()
    for subject_id in train_ids | validation_ids:
        for raw_path in checkpoint["manifest_paths"][subject_id]:
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"checkpoint has an invalid source path for {subject_id!r}")
            source_paths.add(_resolved_source_path(raw_path, checkpoint_path))
    shared_files = sorted(
        (path for scan_paths in subjects.values() for path in scan_paths if path in source_paths),
        key=str,
    )
    if shared_files:
        raise ValueError(
            "test file paths overlap checkpoint training or validation files: "
            f"{', '.join(map(str, shared_files))}. Train with data/development_subjects.csv."
        )


def _scores(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute the fixed normalized-volume metrics used by the CSV output."""
    if any(length < 7 for length in target.shape):
        raise ValueError(f"SSIM requires every native volume axis to be at least 7 voxels; got {target.shape}")
    return {
        "mae": mae(prediction, target),
        "mse": mse(prediction, target),
        "psnr": psnr(prediction, target, data_range=1.0),
        "ssim": ssim(prediction, target, data_range=1.0, win_size=7),
    }


def _new_row(scope: str, subject_id: str, path: str, method: str, scores: dict[str, float], provenance: dict[str, Any]) -> dict[str, Any]:
    return {"scope": scope, "subject_id": subject_id, "path": path, "method": method, **scores, **provenance}


def evaluate(
    manifest_path: str | Path = "data/test_subjects.csv",
    checkpoint_path: str | Path = "checkpoints",
    output_path: str | Path = "outputs/evaluation.csv",
    device: str = "auto",
    tile_size: tuple[int, int, int] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate interpolation and the checkpoint on a held-out subject manifest."""
    loaded, checkpoint_file = read_checkpoint(checkpoint_path)
    checkpoint = _required_checkpoint(loaded, checkpoint_file)
    subjects = read_manifest(manifest_path)
    _validate_test_split(subjects, checkpoint, checkpoint_file)

    settings = checkpoint["training_config"]
    chosen_tile = tuple(tile_size) if tile_size is not None else tuple(settings["patch_size"])
    if len(chosen_tile) != 3 or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 8
        for size in chosen_tile
    ):
        raise ValueError("tile_size must contain three positive multiples of 8")
    selected_device = _select_device(device)
    if "seed" in settings and isinstance(settings["seed"], int) and not isinstance(settings["seed"], bool):
        torch.manual_seed(settings["seed"])
    model = model_from_checkpoint(checkpoint, selected_device)

    provenance = {
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "sparsity_factor": settings["sparsity_factor"],
        "slice_axis": settings["slice_axis"],
        "sigma": settings["sigma"],
        "data_range": 1.0,
        "tile_size": "x".join(map(str, chosen_tile)),
        "tiling_strategy": "uniform averaging with nominal 50% overlap and final tiles anchored at high edge; edge pad only if smaller than tile",
        "ssim_definition": "uniform 7x7x7 windows, sample covariance, valid centers, data_range=1",
    }
    rows: list[dict[str, Any]] = []
    per_subject: dict[str, dict[str, list[dict[str, float]]]] = {}
    for subject_id, scans in subjects.items():
        per_subject[subject_id] = {"interpolation": [], "unet": []}
        for scan_path in scans:
            source, _, _ = load_nifti(scan_path)
            hr = preprocess_volume(source)
            del source
            if any(length < 7 for length in hr.shape):
                raise ValueError(
                    f"SSIM requires every native volume axis to be at least 7 voxels; got {hr.shape} in {scan_path}"
                )
            acquisition = simulate_sparse_acquisition(
                hr,
                factor=settings["sparsity_factor"],
                axis=settings["slice_axis"],
                sigma=settings["sigma"],
            )
            baseline = interpolate_sparse_acquisition(acquisition, method="linear")
            del acquisition
            reconstructed = reconstruct_volume(model, baseline, selected_device, chosen_tile)
            for method, prediction in (("interpolation", baseline), ("unet", reconstructed)):
                scores = _scores(prediction, hr)
                per_subject[subject_id][method].append(scores)
                rows.append(_new_row("volume", subject_id, str(scan_path), method, scores, provenance))
    subject_values: dict[str, dict[str, dict[str, float]]] = {}
    for subject_id, methods in per_subject.items():
        subject_values[subject_id] = {}
        for method, scan_scores in methods.items():
            mean_scores = {name: float(np.mean([score[name] for score in scan_scores])) for name in _METRIC_NAMES}
            subject_values[subject_id][method] = mean_scores
            rows.append(_new_row("subject", subject_id, "", method, mean_scores, provenance))
    for method in ("interpolation", "unet"):
        aggregate = {
            name: float(np.mean([subject_values[subject_id][method][name] for subject_id in subjects]))
            for name in _METRIC_NAMES
        }
        rows.append(_new_row("aggregate", "", "", method, aggregate, provenance))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for scope_row in rows:
        if scope_row["scope"] in ("subject", "aggregate"):
            print(
                f"{scope_row['scope']} {scope_row['subject_id'] or 'all'} {scope_row['method']}: "
                + " ".join(f"{name}={scope_row[name]:.6g}" for name in _METRIC_NAMES)
            )
    return rows


def main() -> None:
    """Run held-out evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/test_subjects.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, nargs=3, metavar=("D", "H", "W"))
    args = parser.parse_args()
    evaluate(args.manifest, args.checkpoint, args.output, args.device, tuple(args.tile_size) if args.tile_size else None)


if __name__ == "__main__":
    main()
