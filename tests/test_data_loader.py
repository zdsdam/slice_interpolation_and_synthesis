"""Focused tests for NIfTI volume loading."""

from pathlib import Path
import re
import tempfile
import unittest

import nibabel as nib
import numpy as np

from src.data_loader import load_nifti


class NiftiLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)

    def save_image(self, name, data, affine=None):
        if affine is None:
            affine = np.array(
                [
                    [0.0, -1.5, 0.0, 12.0],
                    [2.0, 0.0, 0.0, -4.0],
                    [0.0, 0.0, 3.0, 8.0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
        image = nib.Nifti1Image(data, affine)
        image.set_qform(affine, code=2)
        image.set_sform(affine, code=3)
        image.header.set_xyzt_units("mm", "sec")
        path = self.directory / name
        nib.save(image, str(path))
        return path

    def test_loads_nii_and_nii_gz_values_and_spatial_metadata(self):
        data = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
        for suffix in (".nii", ".nii.gz"):
            with self.subTest(suffix=suffix):
                path = self.save_image("volume" + suffix, data)
                expected_image = nib.load(str(path))

                loaded, affine, header = load_nifti(path)

                self.assertIsInstance(loaded, np.ndarray)
                self.assertEqual(loaded.shape, data.shape)
                self.assertIsInstance(affine, np.ndarray)
                self.assertEqual(affine.shape, (4, 4))
                self.assertIsInstance(header, nib.Nifti1Header)
                np.testing.assert_array_equal(loaded, data)
                self.assertEqual(loaded.dtype, data.dtype)
                np.testing.assert_array_equal(affine, expected_image.affine)
                self.assertEqual(header.get_qform(coded=True)[1], 2)
                self.assertEqual(header.get_sform(coded=True)[1], 3)
                self.assertEqual(header.get_xyzt_units(), ("mm", "sec"))
                np.testing.assert_array_equal(
                    header.get_qform(), expected_image.header.get_qform()
                )
                np.testing.assert_array_equal(
                    header.get_sform(), expected_image.header.get_sform()
                )

    def test_applies_nibabel_voxel_scaling(self):
        raw = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        image = nib.Nifti1Image(raw, np.eye(4))
        image.header.set_slope_inter(2.5, -7.0)
        path = self.directory / "scaled.nii"
        nib.save(image, str(path))

        loaded, _, _ = load_nifti(path)

        np.testing.assert_allclose(loaded, raw * 2.5 - 7.0)
        self.assertTrue(np.issubdtype(loaded.dtype, np.floating))

    def test_returns_independent_affine_and_header_copies(self):
        path = self.save_image("copy.nii", np.zeros((2, 3, 4), dtype=np.float32))
        file_before = path.read_bytes()
        _, affine, header = load_nifti(path)
        original = nib.load(str(path))

        affine[0, 0] = 999
        header["descrip"] = b"changed"

        self.assertNotEqual(original.affine[0, 0], 999)
        self.assertNotEqual(original.header["descrip"], b"changed")
        self.assertEqual(path.read_bytes(), file_before)

    def test_loads_nifti2_image_and_returns_its_header(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.diag([1.25, 2.5, 3.75, 1.0])
        image = nib.Nifti2Image(data, affine)
        path = self.directory / "volume_nifti2.nii"
        nib.save(image, str(path))

        loaded, actual_affine, header = load_nifti(path)

        self.assertEqual(loaded.shape, data.shape)
        self.assertIsInstance(header, nib.Nifti2Header)
        np.testing.assert_array_equal(loaded, data)
        np.testing.assert_array_equal(actual_affine, nib.load(str(path)).affine)

    def test_accepts_string_path(self):
        path = self.save_image("string.nii", np.ones((2, 2, 2), dtype=np.uint8))
        loaded, _, _ = load_nifti(str(path))
        self.assertEqual(loaded.shape, (2, 2, 2))

    def test_missing_path_reports_path(self):
        path = self.directory / "missing.nii.gz"
        with self.assertRaisesRegex(FileNotFoundError, str(path)):
            load_nifti(path)

    def test_rejects_unsupported_extension(self):
        path = self.directory / "volume.mgz"
        path.touch()
        with self.assertRaisesRegex(ValueError, str(path)):
            load_nifti(path)

    def test_rejects_directory_with_nifti_suffix(self):
        path = self.directory / "folder.nii"
        path.mkdir()
        with self.assertRaisesRegex(OSError, str(path)):
            load_nifti(path)

    def test_reports_corrupt_nifti_as_oserror_with_cause(self):
        path = self.directory / "corrupt.nii"
        path.write_bytes(b"not a NIfTI file")
        with self.assertRaisesRegex(OSError, str(path)) as raised:
            load_nifti(path)
        self.assertIsNotNone(raised.exception.__cause__)

    def test_reports_truncated_voxel_payload_as_oserror_with_cause(self):
        path = self.save_image("truncated.nii", np.arange(24, dtype=np.int16).reshape(2, 3, 4))
        path.write_bytes(path.read_bytes()[:352])

        with self.assertRaisesRegex(OSError, str(path)) as raised:
            load_nifti(path)
        self.assertIsNotNone(raised.exception.__cause__)

    def test_rejects_non_3d_images(self):
        for shape in ((3, 4), (2, 3, 4, 5)):
            with self.subTest(shape=shape):
                path = self.save_image(
                    f"shape_{len(shape)}.nii", np.zeros(shape, dtype=np.float32)
                )
                message = rf"exactly 3D; got shape {re.escape(str(shape))}"
                with self.assertRaisesRegex(ValueError, message):
                    load_nifti(path)

    def test_rejects_empty_dimensions_and_non_real_voxels(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            load_nifti(self.save_image("empty.nii", np.empty((0, 2, 3), np.float32)))
        with self.assertRaisesRegex(TypeError, "real numeric"):
            load_nifti(
                self.save_image("complex.nii", np.ones((2, 2, 2), np.complex64))
            )


if __name__ == "__main__":
    unittest.main()
