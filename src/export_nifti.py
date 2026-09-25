"""Write dense scalar volumes to NIfTI while preserving spatial metadata."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def export_nifti(
    volume: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header | nib.Nifti2Header,
    output_path: str | Path,
) -> Path:
    """Save a finite 3D volume as float32 NIfTI on the supplied dense grid.

    ``affine`` must describe the output volume's dense grid, including any
    crop or resampling offset. The source header is copied; its intensity
    scaling and scalar intent are reset for the float32 output. Compatible
    metadata such as units and description is retained. A shear is stored
    exactly in the sform; because qforms cannot represent shear, that qform is
    disabled. The output parent directory must exist; an existing file is
    replaced. NIfTI-1 stores affine fields at float32 precision.
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a NumPy ndarray")
    if volume.ndim != 3 or any(size == 0 for size in volume.shape):
        raise ValueError("volume must be a non-empty 3D array")
    if volume.dtype.kind not in "iuf":
        raise TypeError("volume must have a real numeric dtype")
    if not np.isfinite(volume).all():
        raise ValueError("volume must contain only finite values")
    limit = np.finfo(np.float32).max
    if np.any(volume > limit) or np.any(volume < -limit):
        raise ValueError("volume values must fit in float32")

    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 NumPy array")
    if affine.dtype.kind not in "iuf" or not np.isfinite(affine).all():
        raise ValueError("affine must contain finite real numeric values")
    affine_copy = affine.astype(np.float64, copy=True)
    if not np.allclose(affine_copy[3], (0.0, 0.0, 0.0, 1.0), rtol=0, atol=1e-7):
        raise ValueError("affine must have homogeneous bottom row [0, 0, 0, 1]")
    if np.linalg.slogdet(affine_copy[:3, :3])[0] == 0:
        raise ValueError("affine spatial transform must be nonsingular")

    if not isinstance(header, (nib.Nifti1Header, nib.Nifti2Header)):
        raise TypeError("header must be a NIfTI-1 or NIfTI-2 header")
    path = Path(output_path)
    if not path.name.endswith((".nii", ".nii.gz")):
        raise ValueError(f"Unsupported NIfTI extension: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"Output parent directory does not exist: {path.parent}")
    if not path.parent.is_dir():
        raise NotADirectoryError(f"Output parent is not a directory: {path.parent}")
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {path}")

    output_header = header.copy()
    image_type = nib.Nifti2Image if isinstance(header, nib.Nifti2Header) else nib.Nifti1Image
    data = volume.astype(np.float32, copy=True)
    image = image_type(data, affine_copy, header=output_header)
    image.set_data_dtype(np.float32)
    image.header.set_intent("none")
    image.header.set_slope_inter(1.0, 0.0)
    image.header["cal_min"] = float(data.min())
    image.header["cal_max"] = float(data.max())
    sform_code = int(header["sform_code"])
    qform_code = int(header["qform_code"])
    if sform_code <= 0:
        sform_code = qform_code if qform_code > 0 else 2
    image.set_sform(affine_copy, code=sform_code)
    try:
        image.set_qform(
            affine_copy,
            code=qform_code if qform_code > 0 else sform_code,
            strip_shears=False,
        )
    except nib.spatialimages.HeaderDataError:
        image.set_qform(None, code=0)

    nib.save(image, str(path))
    return path
