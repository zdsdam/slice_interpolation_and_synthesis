"""Focused tests for reusable 3D model blocks."""

import unittest

import torch

from src.models.blocks import ConvBlock3D, DecoderBlock3D, EncoderBlock3D


class ConvBlock3DTests(unittest.TestCase):
    def test_preserves_spatial_shape_and_changes_channels(self):
        block = ConvBlock3D(2, 5)
        x = torch.randn(2, 2, 5, 6, 7)

        result = block(x)

        self.assertEqual(result.shape, (2, 5, 5, 6, 7))

    def test_gradients_reach_input_and_parameters(self):
        block = ConvBlock3D(2, 3)
        x = torch.randn(1, 2, 4, 4, 4, requires_grad=True)

        block(x).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in block.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in block.parameters()))

    def test_rejects_invalid_rank_and_channel_count(self):
        block = ConvBlock3D(2, 3)
        with self.assertRaisesRegex(ValueError, "5D"):
            block(torch.randn(2, 2, 4, 4))
        with self.assertRaisesRegex(ValueError, "channels"):
            block(torch.randn(1, 4, 4, 4, 4))


class EncoderBlock3DTests(unittest.TestCase):
    def test_returns_skip_and_halved_features_with_batch_preserved(self):
        block = EncoderBlock3D(2, 4)
        x = torch.randn(3, 2, 8, 10, 12)

        skip, downsampled = block(x)

        self.assertEqual(skip.shape, (3, 4, 8, 10, 12))
        self.assertEqual(downsampled.shape, (3, 4, 4, 5, 6))

    def test_odd_sizes_are_floored_by_pooling(self):
        block = EncoderBlock3D(1, 2)
        skip, downsampled = block(torch.randn(1, 1, 5, 6, 7))

        self.assertEqual(skip.shape, (1, 2, 5, 6, 7))
        self.assertEqual(downsampled.shape, (1, 2, 2, 3, 3))

    def test_gradients_reach_input_and_parameters(self):
        block = EncoderBlock3D(1, 2)
        x = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
        skip, downsampled = block(x)

        (skip.sum() + downsampled.sum()).backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in block.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in block.parameters()))

    def test_rejects_invalid_input(self):
        block = EncoderBlock3D(1, 2)
        with self.assertRaisesRegex(ValueError, "5D"):
            block(torch.randn(1, 1, 4, 4))
        with self.assertRaisesRegex(ValueError, "channels"):
            block(torch.randn(1, 3, 4, 4, 4))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            block(torch.randn(1, 1, 1, 4, 4))


class DecoderBlock3DTests(unittest.TestCase):
    def test_upsamples_and_combines_skip_with_batch_preserved(self):
        block = DecoderBlock3D(in_channels=6, skip_channels=4, out_channels=3)
        x = torch.randn(2, 6, 3, 4, 5)
        skip = torch.randn(2, 4, 6, 8, 10)

        result = block(x, skip)

        self.assertEqual(result.shape, (2, 3, 6, 8, 10))

    def test_gradients_reach_input_skip_and_parameters(self):
        block = DecoderBlock3D(in_channels=4, skip_channels=2, out_channels=3)
        x = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
        skip = torch.randn(1, 2, 4, 4, 4, requires_grad=True)

        block(x, skip).sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(skip.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(skip.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in block.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in block.parameters()))

    def test_rejects_invalid_rank_channels_and_mismatched_shapes(self):
        block = DecoderBlock3D(in_channels=4, skip_channels=2, out_channels=3)
        x = torch.randn(1, 4, 2, 2, 2)
        valid_skip = torch.randn(1, 2, 4, 4, 4)
        with self.assertRaisesRegex(ValueError, "5D"):
            block(torch.randn(1, 4, 2, 2), valid_skip)
        with self.assertRaisesRegex(ValueError, "5D"):
            block(x, torch.randn(1, 2, 4, 4))
        with self.assertRaisesRegex(RuntimeError, "channels"):
            block(torch.randn(1, 5, 2, 2, 2), valid_skip)
        with self.assertRaisesRegex(ValueError, "channels"):
            block(x, torch.randn(1, 5, 4, 4, 4))
        with self.assertRaisesRegex(ValueError, "matching batch and spatial"):
            block(x, torch.randn(2, 2, 4, 4, 4))
        with self.assertRaisesRegex(ValueError, "matching batch and spatial"):
            block(x, torch.randn(1, 2, 4, 4, 5))


if __name__ == "__main__":
    unittest.main()
