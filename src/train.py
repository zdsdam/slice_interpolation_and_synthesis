"""Train the reconstruction model from sparse-slice simulations of HR MRI."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from src.data_loader import load_nifti
from src.interpolation import interpolate_sparse_acquisition
from src.losses import reconstruction_l1_loss
from src.models.unet3d import ResidualUNet3D
from src.preprocessing import preprocess_volume
from src.sparse_simulator import simulate_sparse_acquisition


@dataclass(frozen=True)
class TrainingConfig:
    """Small set of settings for patch-based 3D reconstruction training."""

    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    patch_size: tuple[int, int, int] = (64, 64, 64)
    sparsity_factor: int = 4
    patches_per_volume: int = 8
    validation_fraction: float = 0.2
    seed: int = 42
    base_channels: int = 16
    slice_axis: int = 2
    sigma: float = 1.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patches_per_volume < 1:
            raise ValueError("epochs, batch_size, and patches_per_volume must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if len(self.patch_size) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 8
            for size in self.patch_size
        ):
            raise ValueError("patch_size must contain three positive multiples of 8")
        if self.sparsity_factor < 2:
            raise ValueError("sparsity_factor must be at least 2")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.base_channels < 1:
            raise ValueError("base_channels must be positive")
        if self.slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be finite and non-negative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


def read_manifest(path: str | Path) -> dict[str, list[Path]]:
    """Read subject-to-NIfTI paths; relative paths use the manifest directory."""
    manifest = Path(path)
    subjects: dict[str, list[Path]] = {}
    resolved_paths: set[Path] = set()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"subject_id", "path"}.issubset(reader.fieldnames):
            raise ValueError("manifest must have subject_id and path columns")
        for row_number, row in enumerate(reader, start=2):
            subject_id, raw_path = (row.get("subject_id") or "").strip(), (row.get("path") or "").strip()
            if not subject_id or not raw_path:
                raise ValueError(f"manifest row {row_number} has an empty subject_id or path")
            file_path = Path(raw_path)
            if not file_path.is_absolute():
                file_path = manifest.parent / file_path
            file_path = file_path.resolve()
            if file_path in resolved_paths:
                raise ValueError(f"duplicate NIfTI path in manifest: {file_path}")
            if not file_path.name.endswith((".nii", ".nii.gz")) or not file_path.is_file():
                raise ValueError(f"manifest path is not an existing NIfTI file: {file_path}")
            resolved_paths.add(file_path)
            subjects.setdefault(subject_id, []).append(file_path)
    if not subjects:
        raise ValueError("manifest contains no subjects")
    return subjects


def split_subjects(
    subject_ids: Iterable[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Make a deterministic subject-level split, keeping all scans together."""
    ordered = sorted(set(subject_ids))
    if not ordered:
        raise ValueError("at least one subject is required")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if len(ordered) < 2 or validation_fraction == 0:
        return ordered, []
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    shuffled = np.random.default_rng(seed).permutation(ordered).tolist()
    validation = sorted(shuffled[:count])
    training = sorted(shuffled[count:])
    return training, validation


class SparsePatchDataset(IterableDataset):
    """Yield paired interpolated/HR patches while loading one volume at a time."""

    def __init__(
        self,
        subjects: dict[str, list[Path]],
        config: TrainingConfig,
        *,
        patches_per_volume: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        super().__init__()
        self.subjects = subjects
        self.config = config
        self.patches_per_volume = patches_per_volume
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (1, D, H, W) tensors in native array-axis order.

        No orientation or spacing harmonization is applied. Sigma is measured
        in voxels along the selected native array axis.
        """
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("SparsePatchDataset requires DataLoader num_workers=0")
        rng = np.random.default_rng(self.seed + (self.epoch if self.shuffle else 0))
        subject_ids = sorted(self.subjects)
        if self.shuffle:
            rng.shuffle(subject_ids)
        for subject_id in subject_ids:
            scans = list(self.subjects[subject_id])
            if self.shuffle:
                rng.shuffle(scans)
            for scan in scans:
                volume, _, _ = load_nifti(scan)
                volume = preprocess_volume(volume)
                if any(size < patch for size, patch in zip(volume.shape, self.config.patch_size)):
                    raise ValueError(
                        f"volume {scan} has shape {volume.shape}, smaller than patch_size "
                        f"{self.config.patch_size}"
                    )
                for _ in range(self.patches_per_volume):
                    starts = [
                        int(rng.integers(0, size - patch + 1))
                        for size, patch in zip(volume.shape, self.config.patch_size)
                    ]
                    slices = tuple(
                        slice(start, start + patch)
                        for start, patch in zip(starts, self.config.patch_size)
                    )
                    target = volume[slices]
                    acquisition = simulate_sparse_acquisition(
                        target,
                        factor=self.config.sparsity_factor,
                        axis=self.config.slice_axis,
                        sigma=self.config.sigma,
                    )
                    interpolated = interpolate_sparse_acquisition(acquisition)
                    yield (
                        torch.from_numpy(interpolated).unsqueeze(0),
                        torch.from_numpy(target).unsqueeze(0),
                    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation pass and return mean patch L1 loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = reconstruction_l1_loss(model(inputs), targets)
            if training:
                loss.backward()
                optimizer.step()
        count = int(inputs.shape[0])
        total_loss += float(loss.detach().item()) * count
        sample_count += count
    if sample_count == 0:
        raise ValueError("data loader produced no patches")
    return total_loss / sample_count


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested, but CUDA is unavailable")
    return torch.device(name)


def run_training(
    manifest_path: str | Path,
    config: TrainingConfig = TrainingConfig(),
    checkpoint_dir: str | Path = "checkpoints",
) -> list[dict[str, int | float | None]]:
    """Train for the configured epochs and save latest/best state dictionaries."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    all_subjects = read_manifest(manifest_path)
    train_ids, validation_ids = split_subjects(
        all_subjects, config.validation_fraction, config.seed
    )
    train_subjects = {key: all_subjects[key] for key in train_ids}
    validation_subjects = {key: all_subjects[key] for key in validation_ids}
    train_dataset = SparsePatchDataset(
        train_subjects,
        config,
        patches_per_volume=config.patches_per_volume,
        seed=config.seed,
        shuffle=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=0)
    validation_loader = None
    if validation_subjects:
        validation_dataset = SparsePatchDataset(
            validation_subjects,
            config,
            patches_per_volume=config.patches_per_volume,
            seed=config.seed + 1_000_003,
            shuffle=False,
        )
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=0)

    device = _select_device(config.device)
    model = ResidualUNet3D(base_channels=config.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, int | float | None]] = []
    best_loss = float("inf")

    for epoch in range(config.epochs):
        train_dataset.epoch = epoch
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = None
        if validation_loader is not None:
            val_loss = run_epoch(model, validation_loader, device)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": {"base_channels": config.base_channels},
            "training_config": asdict(config),
            "train_subjects": train_ids,
            "validation_subjects": validation_ids,
            "manifest_paths": {key: [str(path) for path in paths] for key, paths in all_subjects.items()},
        }
        torch.save(checkpoint, directory / "latest.pt")
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, directory / "best.pt")
        if val_loss is None:
            print(f"epoch {epoch + 1}/{config.epochs} train_l1={train_loss:.6f}")
        else:
            print(
                f"epoch {epoch + 1}/{config.epochs} "
                f"train_l1={train_loss:.6f} val_l1={val_loss:.6f}"
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/training_subjects.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--sparsity-factor", type=int, default=4)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--slice-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patch_size=tuple(args.patch_size),
        sparsity_factor=args.sparsity_factor,
        patches_per_volume=args.patches_per_volume,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        base_channels=args.base_channels,
        slice_axis=args.slice_axis,
        sigma=args.sigma,
        device=args.device,
    )
    run_training(args.manifest, config, args.checkpoint_dir)


if __name__ == "__main__":
    main()
