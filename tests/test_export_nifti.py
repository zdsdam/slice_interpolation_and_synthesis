"""Focused tests for NIfTI volume export and dense-grid metadata."""

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from src.data_loader import load_nifti
from src.export_nifti import export_nifti
from src.inference import reconstruct_sparse
from src.models.unet3d import ResidualUNet3D
from src.preprocessing import preprocess_volume
from src.sparse_simulator import simulate_sparse_acquisition


def affine_matrix():
    return np.array(
        [[0.0, -1.5, 0.0, 12.0], [2.0, 0.0, 0.0, -4.0],
         [0.0, 0.0, -3.0, 8.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
    )


class NiftiExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = np.arange(60, dtype=np.float64).reshape(3, 4, 5) / 7
        self.affine = affine_matrix()
        self.header = nib.Nifti1Header()
        self.header.set_xyzt_units("mm", "sec")
        self.header["descrip"] = b"dense MRI"
        self.header.set_qform(np.eye(4), code=1)
        self.header.set_sform(np.eye(4), code=3)

    def test_exports_nifti1_both_extensions_and_preserves_metadata(self):
        for suffix in (".nii", ".nii.gz"):
            with self.subTest(suffix=suffix):
                path = export_nifti(self.data, self.affine, self.header, self.directory / ("out" + suffix))
                image = nib.load(str(path))
                self.assertIsInstance(image, nib.Nifti1Image)
                self.assertEqual(image.shape, self.data.shape)
                self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
                np.testing.assert_allclose(np.asanyarray(image.dataobj), self.data.astype(np.float32))
                np.testing.assert_allclose(image.get_sform(), self.affine, atol=1e-6)
                np.testing.assert_allclose(image.get_qform(), self.affine, atol=1e-5)
                self.assertEqual(image.header.get_sform(coded=True)[1], 3)
                self.assertEqual(image.header.get_qform(coded=True)[1], 1)
                self.assertEqual(image.header.get_xyzt_units(), ("mm", "sec"))
                self.assertEqual(bytes(image.header["descrip"]).rstrip(b"\x00"), b"dense MRI")
                self.assertEqual(float(image.header["cal_min"]), float(self.data.astype(np.float32).min()))
                self.assertEqual(float(image.header["cal_max"]), float(self.data.astype(np.float32).max()))

    def test_preserves_inputs_and_resets_stale_shape_dtype_and_scaling(self):
        hdr = self.header.copy()
        hdr.set_data_shape((10, 10, 10, 2))
        hdr.set_data_dtype(np.int16)
        hdr.set_slope_inter(2.5, -7.0)
        hdr["cal_min"], hdr["cal_max"] = -100, 100
        volume = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
        before_header, before_affine, before_volume = hdr.binaryblock, self.affine.copy(), volume.copy()
        path = export_nifti(volume, self.affine, hdr, self.directory / "scaled.nii")
        image = nib.load(str(path))
        loaded, _, _ = load_nifti(path)
        np.testing.assert_array_equal(loaded, volume.astype(np.float32))
        self.assertEqual(image.dataobj.slope, 1.0)
        self.assertEqual(image.dataobj.inter, 0.0)
        self.assertEqual(float(image.header["cal_min"]), float(volume.min()))
        self.assertEqual(float(image.header["cal_max"]), float(volume.max()))
        self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
        self.assertEqual(hdr.binaryblock, before_header)
        np.testing.assert_array_equal(self.affine, before_affine)
        np.testing.assert_array_equal(volume, before_volume)
        fractional = volume.astype(np.float32) / 3
        fractional_path = export_nifti(
            fractional, self.affine, hdr, self.directory / "fractional.nii"
        )
        fractional_image = nib.load(str(fractional_path))
        self.assertEqual(fractional_image.get_data_dtype(), np.dtype("float32"))
        np.testing.assert_allclose(np.asanyarray(fractional_image.dataobj), fractional)

    def test_affine_overrides_header_forms_and_zero_codes_get_fallback(self):
        self.header.set_qform(np.eye(4), code=0)
        self.header.set_sform(np.eye(4), code=0)
        path = export_nifti(self.data, self.affine, self.header, self.directory / "forms.nii")
        image = nib.load(str(path))
        np.testing.assert_allclose(image.affine, self.affine, atol=1e-5)
        np.testing.assert_allclose(image.get_sform(), self.affine, atol=1e-6)
        np.testing.assert_allclose(image.get_qform(), self.affine, atol=1e-5)
        self.assertEqual(image.header["sform_code"], 2)
        self.assertEqual(image.header["qform_code"], 2)

    def test_shear_keeps_exact_sform_and_disables_qform(self):
        shear = self.affine.copy()
        shear[0, 0] += 0.25
        path = export_nifti(self.data, shear, self.header, self.directory / "shear.nii")
        image = nib.load(str(path))
        np.testing.assert_allclose(image.get_sform(), shear, atol=1e-6)
        self.assertEqual(image.header["qform_code"], 0)

    def test_nifti2_header_creates_nifti2_image(self):
        header = nib.Nifti2Header()
        path = export_nifti(self.data, self.affine, header, self.directory / "out.nii.gz")
        image = nib.load(str(path))
        self.assertIsInstance(image, nib.Nifti2Image)
        np.testing.assert_allclose(image.affine, self.affine, atol=1e-8)

    def test_validates_volume_affine_header_and_path_before_saving(self):
        path = self.directory / "bad.nii"
        for volume in (np.ones((2, 3)), np.ones((2, 2, 2, 2)), np.empty((0, 2, 3)), np.ones((2, 2, 2), bool),
                       np.ones((2, 2, 2), complex), np.full((2, 2, 2), np.nan),
                       np.full((2, 2, 2), np.finfo(np.float64).max)):
            with self.subTest(volume=getattr(volume, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    export_nifti(volume, self.affine, self.header, path)
        for affine in (np.eye(3), np.full((4, 4), np.nan),
                       np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]]),
                       np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1.]])):
            with self.subTest(affine=affine.shape):
                with self.assertRaises(ValueError):
                    export_nifti(self.data, affine, self.header, path)
        with self.assertRaises(TypeError):
            export_nifti(self.data, self.affine, object(), self.directory / "bad.hdr")
        with self.assertRaises(ValueError):
            export_nifti(self.data, self.affine, self.header, self.directory / "bad.nii.gz.extra")
        with self.assertRaises(FileNotFoundError):
            export_nifti(self.data, self.affine, self.header, self.directory / "missing" / "x.nii")
        directory_output = self.directory / "folder.nii"
        directory_output.mkdir()
        with self.assertRaises(IsADirectoryError):
            export_nifti(self.data, self.affine, self.header, directory_output)

    def test_synthetic_input_to_reconstruction_to_export_preserves_grid(self):
        original = np.arange(9 * 10 * 11, dtype=np.float32).reshape(9, 10, 11)
        source_header = self.header.copy()
        source = nib.Nifti1Image(original, self.affine, header=source_header)
        source.set_qform(self.affine, code=1)
        source.set_sform(self.affine, code=3)
        source_path = self.directory / "source.nii.gz"
        nib.save(source, str(source_path))
        loaded, loaded_affine, loaded_header = load_nifti(source_path)
        dense = preprocess_volume(loaded)
        acquisition = simulate_sparse_acquisition(dense, factor=3, axis=2, sigma=0)
        model = ResidualUNet3D(base_channels=1)
        model.eval()
        import torch
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        reconstructed = reconstruct_sparse(
            acquisition, model, device="cpu", tile_size=None
        )
        output_path = export_nifti(
            reconstructed, loaded_affine, loaded_header, self.directory / "reconstructed.nii.gz"
        )
        exported, exported_affine, exported_header = load_nifti(output_path)
        self.assertEqual(exported.shape, original.shape)
        self.assertTrue(np.isfinite(exported).all())
        np.testing.assert_allclose(exported, reconstructed, rtol=0, atol=1e-6)
        np.testing.assert_allclose(exported_affine, loaded_affine, atol=1e-5)
        self.assertEqual(exported_header.get_sform(coded=True)[1], 3)


if __name__ == "__main__":
    unittest.main()
