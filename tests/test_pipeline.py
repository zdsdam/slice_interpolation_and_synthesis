"""Synthetic end-to-end smoke test through reconstruction loss backward."""

import unittest

import numpy as np
import torch

from src.interpolation import interpolate_sparse_acquisition
from src.losses import reconstruction_l1_loss
from src.models.unet3d import ResidualUNet3D
from src.preprocessing import preprocess_volume
from src.sparse_simulator import simulate_sparse_acquisition


class SyntheticPipelineTests(unittest.TestCase):
    def test_synthetic_hr_to_reconstruction_backward(self):
        original = np.random.default_rng(7).uniform(20, 120, (16, 24, 32))
        hr = preprocess_volume(original)
        self.assertEqual(hr.shape, original.shape)
        self.assertEqual(hr.dtype, np.float32)
        self.assertEqual(float(hr.min()), 0.0)
        self.assertEqual(float(hr.max()), 1.0)
        hr_before = hr.copy()

        acquisition = simulate_sparse_acquisition(hr, factor=4, axis=2, sigma=1.0)
        self.assertEqual(acquisition.samples.shape, (16, 24, 8))
        np.testing.assert_array_equal(acquisition.slice_indices, np.arange(0, 32, 4))

        interpolated = interpolate_sparse_acquisition(acquisition, method="linear")
        self.assertEqual(interpolated.shape, hr.shape)
        self.assertEqual(interpolated.dtype, np.float32)
        self.assertTrue(np.isfinite(interpolated).all())
        self.assertFalse(np.array_equal(interpolated, hr))
        np.testing.assert_array_equal(interpolated[:, :, ::4], acquisition.samples)

        # Preserve the array-axis order and add batch and channel dimensions.
        model_input = torch.from_numpy(interpolated).unsqueeze(0).unsqueeze(0)
        target = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)
        self.assertEqual(model_input.shape, (1, 1, 16, 24, 32))
        self.assertEqual(model_input.dtype, torch.float32)
        self.assertEqual(target.shape, model_input.shape)
        input_before = model_input.clone()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = ResidualUNet3D()

        residuals = []
        hook = model.residual_head.register_forward_hook(
            lambda module, inputs, output: residuals.append(output)
        )
        try:
            reconstruction = model(model_input)
        finally:
            hook.remove()

        self.assertEqual(reconstruction.shape, target.shape)
        torch.testing.assert_close(reconstruction, model_input + residuals[0], rtol=0, atol=0)
        torch.testing.assert_close(model_input, input_before, rtol=0, atol=0)
        np.testing.assert_array_equal(hr, hr_before)

        loss = reconstruction_l1_loss(reconstruction, target)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
