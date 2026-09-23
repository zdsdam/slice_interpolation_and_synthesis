"""Focused tests for sparse through-plane acquisition simulation."""

import unittest

import numpy as np

from src.sparse_simulator import simulate_sparse_acquisition


class SparseSimulatorTests(unittest.TestCase):
    def test_valid_3d_volume_returns_sparse_float32_samples_and_metadata(self):
        volume = np.arange(3 * 4 * 7, dtype=np.int16).reshape(3, 4, 7)

        result = simulate_sparse_acquisition(volume, factor=2, sigma=0)

        self.assertEqual(result.samples.dtype, np.float32)
        self.assertEqual(result.samples.shape, (3, 4, 4))
        self.assertEqual(result.original_shape, (3, 4, 7))
        self.assertEqual(result.axis, 2)
        self.assertEqual(result.factor, 2)
        np.testing.assert_array_equal(result.slice_indices, [0, 2, 4, 6])
        np.testing.assert_array_equal(result.samples, volume[..., [0, 2, 4, 6]])

    def test_rejects_non_3d_volumes(self):
        with self.assertRaisesRegex(ValueError, "exactly 3D"):
            simulate_sparse_acquisition(np.zeros((3, 4)), factor=2)
        with self.assertRaisesRegex(ValueError, "exactly 3D"):
            simulate_sparse_acquisition(np.zeros((2, 3, 4, 5)), factor=2)

    def test_rejects_empty_or_non_real_numeric_volumes(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            simulate_sparse_acquisition(np.zeros((2, 0, 3)), factor=2)
        with self.assertRaisesRegex(TypeError, "real numeric"):
            simulate_sparse_acquisition(np.zeros((2, 3, 4), dtype=complex), factor=2)
        with self.assertRaisesRegex(TypeError, "real numeric"):
            simulate_sparse_acquisition(np.zeros((2, 3, 4), dtype=object), factor=2)
        with self.assertRaisesRegex(TypeError, "NumPy ndarray"):
            simulate_sparse_acquisition([[[1]]], factor=2)

    def test_factors_two_three_and_four_keep_every_nth_slice_from_zero(self):
        volume = np.broadcast_to(np.arange(11), (2, 3, 11)).copy()

        for factor in (2, 3, 4):
            with self.subTest(factor=factor):
                result = simulate_sparse_acquisition(volume, factor=factor, sigma=0)
                expected_indices = np.arange(0, 11, factor)
                np.testing.assert_array_equal(result.slice_indices, expected_indices)
                np.testing.assert_array_equal(
                    result.samples,
                    np.take(volume.astype(np.float32), expected_indices, axis=2),
                )

    def test_supports_configurable_and_negative_slice_axis(self):
        volume = np.arange(5 * 3 * 2).reshape(5, 3, 2)

        first_axis = simulate_sparse_acquisition(volume, factor=2, axis=0, sigma=0)
        np.testing.assert_array_equal(first_axis.slice_indices, [0, 2, 4])
        np.testing.assert_array_equal(first_axis.samples, volume[[0, 2, 4]].astype(np.float32))
        self.assertEqual(first_axis.axis, 0)

        negative_axis = simulate_sparse_acquisition(volume, factor=2, axis=-1, sigma=0)
        np.testing.assert_array_equal(negative_axis.slice_indices, [0])
        self.assertEqual(negative_axis.axis, 2)

    def test_rejects_invalid_axes(self):
        volume = np.zeros((2, 3, 4))
        for axis in (-4, 3):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "axis"):
                simulate_sparse_acquisition(volume, factor=2, axis=axis)
        for axis in (1.0, True):
            with self.subTest(axis=axis), self.assertRaisesRegex(TypeError, "axis"):
                simulate_sparse_acquisition(volume, factor=2, axis=axis)

    def test_rejects_invalid_factors(self):
        volume = np.zeros((2, 3, 4))
        for factor in (0, 1, -2):
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, "factor"):
                simulate_sparse_acquisition(volume, factor=factor)
        for factor in (2.0, 2.5, True):
            with self.subTest(factor=factor), self.assertRaisesRegex(TypeError, "factor"):
                simulate_sparse_acquisition(volume, factor=factor)

    def test_accepts_numpy_integer_factor(self):
        result = simulate_sparse_acquisition(np.zeros((2, 3, 5)), factor=np.int64(2), sigma=0)
        self.assertEqual(result.factor, 2)

    def test_does_not_mutate_input_and_is_deterministic(self):
        volume = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
        before = volume.copy()

        first = simulate_sparse_acquisition(volume, factor=2)
        second = simulate_sparse_acquisition(volume, factor=2)

        np.testing.assert_array_equal(volume, before)
        np.testing.assert_array_equal(first.samples, second.samples)

    def test_blur_only_affects_selected_axis_and_precedes_subsampling(self):
        expected_neighbor = np.exp(-0.5) / sum(
            np.exp(-0.5 * offset**2) for offset in range(-4, 5)
        )

        for axis in range(3):
            with self.subTest(axis=axis):
                volume = np.zeros((11, 11, 11), dtype=np.float32)
                volume[5, 5, 5] = 1.0  # Slice 5 is omitted by factor-2 sampling.
                result = simulate_sparse_acquisition(
                    volume, factor=2, axis=axis, sigma=1.0
                )

                sample_at_slice_4 = [5, 5, 5]
                sample_at_slice_4[axis] = 2  # Retained source slice 4.
                self.assertAlmostEqual(
                    result.samples[tuple(sample_at_slice_4)], expected_neighbor, places=6
                )
                # The omitted impulse contributes to retained slice 4, proving
                # that blur happens before subsampling.
                self.assertGreater(result.samples[tuple(sample_at_slice_4)], 0.0)

                in_plane_neighbor = sample_at_slice_4.copy()
                in_plane_neighbor[(axis + 1) % 3] = 4
                self.assertEqual(result.samples[tuple(in_plane_neighbor)], 0.0)

    def test_zero_sigma_disables_blur(self):
        volume = np.zeros((3, 3, 5), dtype=np.float32)
        volume[1, 1, 1] = 1.0

        result = simulate_sparse_acquisition(volume, factor=2, sigma=0)

        np.testing.assert_array_equal(result.samples, volume[..., [0, 2, 4]])

    def test_rejects_invalid_sigma(self):
        volume = np.zeros((2, 3, 4))
        for sigma in (-1, float("inf"), float("nan")):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, "sigma"):
                simulate_sparse_acquisition(volume, factor=2, sigma=sigma)
        for sigma in (True, "1.0"):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(TypeError, "sigma"):
                simulate_sparse_acquisition(volume, factor=2, sigma=sigma)


if __name__ == "__main__":
    unittest.main()
