"""Focused tests for sparse-volume reconstruction and checkpoint loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.inference import (
    acquisition_from_samples,
    model_from_checkpoint,
    read_checkpoint,
    reconstruct_sparse,
    reconstruct_volume,
)
from src.interpolation import interpolate_sparse_acquisition
from src.models.unet3d import ResidualUNet3D
from src.preprocessing import preprocess_volume
from src.sparse_simulator import SparseAcquisition


def zero_residual_model() -> ResidualUNet3D:
    model = ResidualUNet3D(base_channels=1)
    torch.nn.init.zeros_(model.residual_head.weight)
    torch.nn.init.zeros_(model.residual_head.bias)
    return model


class InferenceTests(unittest.TestCase):
    def test_sparse_reconstruction_scale_and_input_immutability(self):
        values = np.arange(5 * 6 * 9, dtype=np.float32).reshape(5, 6, 9) + 20
        samples = values[:, :, ::3].copy()
        samples_before = samples.copy()
        indices = np.array([0, 3, 6], dtype=np.intp)
        indices_before = indices.copy()
        acquisition = SparseAcquisition(samples, indices, values.shape, axis=2, factor=3)
        metadata = {"affine": np.eye(4), "header": {"sentinel": "unchanged"}}
        expected = interpolate_sparse_acquisition(acquisition)

        actual = reconstruct_sparse(acquisition, zero_residual_model(), device="cpu", tile_size=(8, 8, 16))
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        normalized = reconstruct_sparse(
            acquisition, zero_residual_model(), device="cpu", normalize=True, tile_size=(8, 8, 16)
        )
        np.testing.assert_allclose(normalized, interpolate_sparse_acquisition(
            SparseAcquisition(preprocess_volume(samples), indices, values.shape, axis=2, factor=3)
        ), atol=1e-6)
        np.testing.assert_array_equal(samples, samples_before)
        np.testing.assert_array_equal(indices, indices_before)
        np.testing.assert_array_equal(metadata["affine"], np.eye(4))
        self.assertEqual(metadata["header"]["sentinel"], "unchanged")
        model = zero_residual_model()
        reconstruct_sparse(acquisition, model, device="cpu", tile_size=None)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_adapter_builds_regular_geometry_and_rejects_invalid_inputs(self):
        samples = np.zeros((4, 3, 3), dtype=np.float32)
        acquisition = acquisition_from_samples(samples, original_shape=(4, 3, 7), axis=2, factor=3)
        self.assertEqual(acquisition.original_shape, (4, 3, 7))
        np.testing.assert_array_equal(acquisition.slice_indices, [0, 3, 6])
        with self.assertRaisesRegex(ValueError, "factor"):
            acquisition_from_samples(samples, original_shape=(4, 3, 7), axis=2, factor=0)
        with self.assertRaisesRegex(ValueError, "original_shape"):
            acquisition_from_samples(samples, original_shape=(4, 0, 7), axis=2, factor=3)
        with self.assertRaisesRegex(ValueError, "axis"):
            acquisition_from_samples(samples, original_shape=(4, 3, 7), axis=3, factor=3)
        with self.assertRaisesRegex(ValueError, "shape"):
            acquisition_from_samples(np.zeros((4, 3, 2), dtype=np.float32), original_shape=(4, 3, 8), axis=2, factor=3)
        with self.assertRaisesRegex(ValueError, "finite"):
            acquisition_from_samples(np.full((4, 3, 3), np.nan), original_shape=(4, 3, 7), axis=2, factor=3)

    def test_minimal_checkpoint_loader_and_best_preference(self):
        model = zero_residual_model()
        checkpoint = {"model_config": {"base_channels": 1}, "model_state_dict": model.state_dict()}
        loaded = model_from_checkpoint(checkpoint, torch.device("cpu"))
        self.assertFalse(loaded.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            torch.save(checkpoint, directory / "latest.pt")
            torch.save(checkpoint, directory / "best.pt")
            chosen, path = read_checkpoint(directory)
            self.assertEqual(path.name, "best.pt")
            self.assertEqual(set(chosen), set(checkpoint))
            dense = np.arange(9 * 10 * 11, dtype=np.float32).reshape(9, 10, 11)
            acquisition = SparseAcquisition(
                dense[:, :, ::3].copy(), np.array([0, 3, 6, 9]), dense.shape, axis=2, factor=3
            )
            expected = interpolate_sparse_acquisition(acquisition)
            actual = reconstruct_sparse(acquisition, directory, device="cpu", tile_size=None)
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            self.assertEqual(actual.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "base_channels"):
            model_from_checkpoint({"model_config": {}, "model_state_dict": {}}, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "missing required"):
            model_from_checkpoint({}, torch.device("cpu"))

    def test_full_volume_mode_pads_crops_and_is_deterministic_without_gradients(self):
        class RecordingIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.flags = []

            def forward(self, values):
                self.flags.append((self.training, torch.is_grad_enabled(), tuple(values.shape)))
                return values

        source = np.random.default_rng(2).normal(size=(9, 11, 17)).astype(np.float32)
        model = RecordingIdentity()
        first = reconstruct_volume(model, source, torch.device("cpu"), tile_size=None)
        second = reconstruct_volume(model, source, torch.device("cpu"), tile_size=None)
        np.testing.assert_allclose(first, source)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(model.flags, [(False, False, (1, 1, 16, 16, 24))] * 2)

    def test_tiling_averages_overlap_and_supports_small_and_large_volumes(self):
        class TileMean(torch.nn.Module):
            def forward(self, values):
                return values.mean(dim=(2, 3, 4), keepdim=True).expand_as(values)

        source = np.broadcast_to(np.arange(24, dtype=np.float32)[:, None, None], (24, 8, 8)).copy()
        actual = reconstruct_volume(TileMean(), source, torch.device("cpu"), tile_size=(16, 8, 8))
        # Starts are 0 and 8. Their tile means are 7.5 and 15.5; the overlap
        # averages them, and the final tile is anchored at 8.
        expected_line = np.r_[np.full(8, 7.5), np.full(8, 11.5), np.full(8, 15.5)]
        np.testing.assert_allclose(actual[:, 0, 0], expected_line)
        small = source[:5, :4, :3]
        padded = reconstruct_volume(torch.nn.Identity(), small, torch.device("cpu"), tile_size=(8, 8, 8))
        np.testing.assert_array_equal(padded, small)

    def test_invalid_device_and_bad_model_outputs_fail_clearly(self):
        with patch("src.inference.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                reconstruct_sparse(
                    SparseAcquisition(np.ones((1, 1, 1), dtype=np.float32), np.array([0]), (1, 1, 1), 2, 2),
                    torch.nn.Identity(), device="cuda", tile_size=(8, 8, 8),
                )

        class BadShape(torch.nn.Module):
            def forward(self, values):
                return values[:, :, :-1]

        with self.assertRaisesRegex(ValueError, "model output must have shape"):
            reconstruct_volume(BadShape(), np.zeros((8, 8, 8), dtype=np.float32), torch.device("cpu"), (8, 8, 8))

        class NonFinite(torch.nn.Module):
            def forward(self, values):
                return values * float("nan")

        with self.assertRaisesRegex(ValueError, "finite"):
            reconstruct_volume(NonFinite(), np.zeros((8, 8, 8), dtype=np.float32), torch.device("cpu"), (8, 8, 8))


if __name__ == "__main__":
    unittest.main()
