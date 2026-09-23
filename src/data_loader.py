"""Load 3D MRI volumes and spatial metadata from NIfTI files."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np


NiftiHeader = Union[nib.Nifti1Header, nib.Nifti2Header]


def load_nifti(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, NiftiHeader]:
    """Load a scalar 3D NIfTI volume, affine, and copied header.

    Unscaled voxel values retain their on-disk dtype; nibabel scaling may
    promote the array dtype. Spatial metadata is returned as stored by nibabel,
    without reorientation or resampling.
    """
    file_path = Path(path)
    if not file_path.name.endswith((".nii", ".nii.gz")):
        raise ValueError(f"Unsupported NIfTI extension: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI file does not exist: {file_path}")
    if not file_path.is_file():
        raise OSError(f"NIfTI path is not a file: {file_path}")

    try:
        image = nib.load(str(file_path))
    except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError) as exc:
        raise OSError(f"Could not read NIfTI file {file_path}: {exc}") from exc

    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(f"File is not a NIfTI image: {file_path}")

    shape = image.shape
    if len(shape) != 3:
        raise ValueError(
            f"NIfTI image must be exactly 3D; got shape {shape} in {file_path}"
        )
    if any(size == 0 for size in shape):
        raise ValueError(f"NIfTI image dimensions must be non-empty: {file_path}")

    try:
        data = np.asanyarray(image.dataobj)
    except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError) as exc:
        raise OSError(
            f"Could not read voxel data from NIfTI file {file_path}: {exc}"
        ) from exc

    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
        data.dtype, np.complexfloating
    ):
        raise TypeError(
            f"NIfTI image must contain real numeric scalar voxels; "
            f"got dtype {data.dtype} in {file_path}"
        )

    return data, image.affine.copy(), image.header.copy()
