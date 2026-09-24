"""Focused tests for volume reconstruction losses."""

import unittest

import torch

from src.losses import reconstruction_l1_loss
from src.models.unet3d import ResidualUNet3D


class ReconstructionL1LossTests(unittest.TestCase):
    def test_identical_volumes_have_zero_scalar_loss(self):
        volume = torch.randn(1, 1, 2, 3, 4)

        loss = reconstruction_l1_loss(volume, volume.clone())

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.item(), 0.0)

    def test_returns_mean_absolute_error_for_signed_values(self):
        prediction = torch.tensor([-2.0, 1.0, 4.0])
        target = torch.tensor([1.0, -1.0, 2.0])

        loss = reconstruction_l1_loss(prediction, target)

        self.assertAlmostEqual(loss.item(), 7.0 / 3.0, places=6)

    def test_gradient_is_signed_and_averaged_over_elements(self):
        prediction = torch.tensor([-2.0, 1.0, 4.0], requires_grad=True)
        target = torch.tensor([1.0, -1.0, 2.0])

        reconstruction_l1_loss(prediction, target).backward()

        torch.testing.assert_close(prediction.grad, torch.tensor([-1 / 3, 1 / 3, 1 / 3]))

    def test_5d_batch_loss_averages_across_samples_and_voxels(self):
        prediction = torch.zeros((2, 1, 2, 2, 2))
        target = torch.stack((torch.ones((1, 2, 2, 2)), torch.full((1, 2, 2, 2), 3.0)))

        loss = reconstruction_l1_loss(prediction, target)

        self.assertEqual(loss.item(), 2.0)

    def test_rejects_broadcastable_mismatched_shapes_and_reports_both(self):
        prediction = torch.zeros((2, 1, 3, 4, 5))
        target = torch.zeros((1, 1, 3, 4, 5))

        with self.assertRaisesRegex(ValueError, r"\(2, 1, 3, 4, 5\).+\(1, 1, 3, 4, 5\)"):
            reconstruction_l1_loss(prediction, target)

    def test_rejects_ordinary_and_rank_mismatches(self):
        with self.assertRaisesRegex(ValueError, r"\(2, 3\).+\(2, 4\)"):
            reconstruction_l1_loss(torch.zeros(2, 3), torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, r"\(2, 3\).+\(6,\)"):
            reconstruction_l1_loss(torch.zeros(2, 3), torch.zeros(6))

    def test_loss_backpropagates_through_residual_unet(self):
        model = ResidualUNet3D(base_channels=2)
        sparse_input = torch.randn(1, 1, 8, 8, 8)
        hr_target = torch.randn_like(sparse_input)

        loss = reconstruction_l1_loss(model(sparse_input), hr_target)
        loss.backward()

        parameters = list(model.parameters())
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in parameters))


if __name__ == "__main__":
    unittest.main()
