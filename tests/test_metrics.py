"""Focused tests for scalar 3D reconstruction metrics."""

import unittest

import numpy as np

from src.metrics import mae, mse, psnr, ssim


class MetricsTests(unittest.TestCase):
    def test_identical_volumes_have_zero_errors_infinite_psnr_and_unit_ssim(self):
        volume = np.linspace(0.0, 1.0, 7**3).reshape((7, 7, 7))
        self.assertEqual(mae(volume, volume), 0.0)
        self.assertEqual(mse(volume, volume), 0.0)
        self.assertEqual(psnr(volume, volume, data_range=1.0), float("inf"))
        self.assertAlmostEqual(ssim(volume, volume, data_range=1.0), 1.0)

    def test_mae_and_mse_match_known_signed_integer_differences(self):
        prediction = np.array([[[-2, 4, 10]]], dtype=np.int16)
        target = np.array([[[1, 2, 6]]], dtype=np.int16)
        self.assertEqual(mae(prediction, target), 3.0)
        self.assertAlmostEqual(mse(prediction, target), 29.0 / 3.0)

    def test_psnr_uses_explicit_range_and_preserves_unsigned_arithmetic(self):
        zeros = np.zeros((2, 2, 2), dtype=np.float32)
        half = np.full((2, 2, 2), 0.5, dtype=np.float32)
        self.assertAlmostEqual(psnr(half, zeros, data_range=1.0), 6.0205999, places=6)
        self.assertAlmostEqual(psnr(half, zeros, data_range=2.0), 12.0411998, places=6)

        black = np.zeros((2, 2, 2), dtype=np.uint8)
        white = np.full((2, 2, 2), 255, dtype=np.uint8)
        self.assertEqual(mse(black, white), 65025.0)

    def test_metrics_do_not_clip_neural_prediction_overshoots(self):
        target = np.zeros((2, 2, 2), dtype=np.float32)
        overshoot = np.full((2, 2, 2), 1.5, dtype=np.float32)
        self.assertEqual(mae(overshoot, target), 1.5)
        self.assertEqual(mse(overshoot, target), 2.25)

    def test_ssim_constant_volumes_and_perturbation(self):
        zero = np.zeros((7, 7, 7), dtype=np.float32)
        four = np.full((7, 7, 7), 4.0, dtype=np.float32)
        distinct = np.full((7, 7, 7), 5.0, dtype=np.float32)
        self.assertAlmostEqual(ssim(zero, zero, data_range=1.0), 1.0)
        self.assertAlmostEqual(ssim(four, four, data_range=5.0), 1.0)
        different_score = ssim(four, distinct, data_range=5.0)
        self.assertTrue(np.isfinite(different_score))
        self.assertLess(different_score, 1.0)

        perturbed = zero.copy()
        perturbed[3, 3, 3] = 0.25
        self.assertLess(ssim(perturbed, zero, data_range=1.0), 1.0)

    def test_data_range_is_required_finite_positive_real_and_not_boolean(self):
        volume = np.zeros((7, 7, 7), dtype=np.float32)
        for metric in (psnr, ssim):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(TypeError):
                    metric(volume, volume)
                for value in (0, -1, np.nan, np.inf):
                    with self.subTest(value=value):
                        with self.assertRaises(ValueError):
                            metric(volume, volume, data_range=value)
                for value in (True, "1"):
                    with self.subTest(value=value):
                        with self.assertRaises(TypeError):
                            metric(volume, volume, data_range=value)

    def test_all_metrics_reject_mismatched_and_broadcastable_shapes(self):
        first = np.zeros((2, 3, 4), dtype=np.float32)
        second = np.zeros((2, 3, 1), dtype=np.float32)
        for metric in (mae, mse, psnr, ssim):
            kwargs = {"data_range": 1.0} if metric in (psnr, ssim) else {}
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(ValueError):
                    metric(first, second, **kwargs)

    def test_all_metrics_reject_wrong_rank_empty_nonfinite_and_nonreal_inputs(self):
        good = np.zeros((3, 3, 3), dtype=np.float32)
        invalid = (
            (np.zeros((3, 3), dtype=np.float32), ValueError),
            (np.zeros((0, 3, 3), dtype=np.float32), ValueError),
            (np.full((3, 3, 3), np.nan), ValueError),
            (np.ones((3, 3, 3), dtype=bool), TypeError),
            (np.ones((3, 3, 3), dtype=complex), TypeError),
            (np.full((3, 3, 3), "1", dtype=object), TypeError),
        )
        for metric in (mae, mse, psnr, ssim):
            kwargs = {"data_range": 1.0} if metric in (psnr, ssim) else {}
            for bad, error in invalid:
                with self.subTest(metric=metric.__name__, dtype=bad.dtype, shape=bad.shape):
                    with self.assertRaises(error):
                        metric(bad, good, **kwargs)

    def test_metrics_reject_non_array_inputs_and_do_not_mutate_arrays(self):
        volume = np.arange(7**3, dtype=np.int16).reshape((7, 7, 7))
        before = volume.copy()
        kwargs = {"data_range": 7**3 - 1}
        for metric in (mae, mse, psnr, ssim):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(TypeError):
                    metric(volume.tolist(), volume, **(kwargs if metric in (psnr, ssim) else {}))
                metric(volume, volume, **(kwargs if metric in (psnr, ssim) else {}))
        np.testing.assert_array_equal(volume, before)

    def test_ssim_window_validation_and_small_volume_window(self):
        small = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
        self.assertAlmostEqual(ssim(small, small, data_range=26, win_size=3), 1.0)
        for window, error in ((7, ValueError), (4, ValueError), (1, ValueError), (2.5, TypeError), (True, TypeError)):
            with self.subTest(window=window):
                with self.assertRaises(error):
                    ssim(small, small, data_range=26, win_size=window)


if __name__ == "__main__":
    unittest.main()
