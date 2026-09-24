"""Focused tests for the residual 3D U-Net."""

import unittest

import torch

from src.models.unet3d import ResidualUNet3D


class ResidualUNet3DTests(unittest.TestCase):
    def test_default_model_preserves_64_cube_shape_and_single_channel(self):
        model = ResidualUNet3D()
        with torch.no_grad():
            result = model(torch.randn(1, 1, 64, 64, 64))

        self.assertEqual(result.shape, (1, 1, 64, 64, 64))

    def test_preserves_batch_and_valid_non_cubic_shape(self):
        model = ResidualUNet3D(base_channels=2)
        with torch.no_grad():
            batch_result = model(torch.randn(2, 1, 32, 32, 32))
            non_cubic_result = model(torch.randn(1, 1, 8, 16, 24))

        self.assertEqual(batch_result.shape, (2, 1, 32, 32, 32))
        self.assertEqual(non_cubic_result.shape, (1, 1, 8, 16, 24))

    def test_gradients_reach_input_and_every_parameter(self):
        model = ResidualUNet3D(base_channels=2)
        x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)

        model(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        parameters = list(model.parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in parameters))

    def test_zero_residual_head_is_identity(self):
        model = ResidualUNet3D(base_channels=2)
        with torch.no_grad():
            model.residual_head.weight.zero_()
            model.residual_head.bias.zero_()
        x = torch.randn(1, 1, 8, 8, 8)

        torch.testing.assert_close(model(x), x, rtol=0, atol=0)

    def test_negative_head_bias_produces_signed_residual(self):
        model = ResidualUNet3D(base_channels=2)
        with torch.no_grad():
            model.residual_head.weight.zero_()
            model.residual_head.bias.fill_(-0.25)
        x = torch.full((1, 1, 8, 8, 8), 1.0)

        result = model(x)

        torch.testing.assert_close(result, torch.full_like(x, 0.75), rtol=0, atol=0)

    def test_rejects_invalid_rank_channel_and_spatial_sizes(self):
        model = ResidualUNet3D(base_channels=2)
        with self.assertRaisesRegex(ValueError, "5D"):
            model(torch.randn(1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "channels"):
            model(torch.randn(1, 2, 8, 8, 8))
        for shape in ((1, 1, 0, 8, 8), (1, 1, 7, 8, 8), (1, 1, 8, 10, 8)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "spatial dimensions"):
                model(torch.randn(shape))

    def test_base_channels_must_be_positive_non_boolean_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                ResidualUNet3D(base_channels=value)


if __name__ == "__main__":
    unittest.main()
