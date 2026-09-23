"""Focused tests for simple 3D volume preprocessing."""

import unittest

import numpy as np

from src.preprocessing import center_crop_or_pad, normalize_volume, preprocess_volume


class PreprocessingTests(unittest.TestCase):
    def test_normalizes_known_values_to_whole_volume_range(self):
        volume = np.array([[[-2, 0], [2, 6]]], dtype=np.int16)
        before = volume.copy()

        result = normalize_volume(volume)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, [[[0, 0.25], [0.5, 1]]])
        np.testing.assert_array_equal(volume, before)
        self.assertFalse(np.shares_memory(result, volume))

    def test_constant_volume_normalizes_to_independent_zeros(self):
        volume = np.full((2, 1, 3), 4.5, dtype=np.float64)
        before = volume.copy()

        result = normalize_volume(volume)

        self.assertEqual(result.shape, volume.shape)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.zeros(volume.shape, dtype=np.float32))
        np.testing.assert_array_equal(volume, before)
        self.assertFalse(np.shares_memory(result, volume))

    def test_float32_endpoints_normalize_finitely(self):
        maximum = np.finfo(np.float32).max
        volume = np.array([[[-maximum, maximum]]], dtype=np.float32)

        result = normalize_volume(volume)

        np.testing.assert_array_equal(result, [[[0.0, 1.0]]])
        self.assertTrue(np.isfinite(result).all())

    def test_normalization_returns_float32_independent_array_for_float32_input(self):
        volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        before = volume.copy()

        result = normalize_volume(volume)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(volume, before)
        self.assertFalse(np.shares_memory(result, volume))

    def test_crop_is_centered_on_each_axis(self):
        volume = np.arange(5 * 6 * 7, dtype=np.int16).reshape(5, 6, 7)

        result = center_crop_or_pad(volume, (3, 4, 3))

        self.assertEqual(result.shape, (3, 4, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, volume[1:4, 1:5, 2:5])

    def test_padding_is_centered_with_zero_edges(self):
        volume = np.array([[[1, 2], [3, 4]]], dtype=np.float64)
        before = volume.copy()

        result = center_crop_or_pad(volume, (3, 4, 5))

        self.assertEqual(result.shape, (3, 4, 5))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[1, 1:3, 1:3], volume[0])
        self.assertEqual(np.count_nonzero(result), volume.size)
        np.testing.assert_array_equal(volume, before)
        self.assertFalse(np.shares_memory(result, volume))

    def test_odd_crop_and_pad_put_extra_voxel_on_high_index_end(self):
        crop_source = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
        padded_source = np.array([[[7, 8, 9]]], dtype=np.float32)

        cropped = center_crop_or_pad(crop_source, (1, 1, 3))
        padded = center_crop_or_pad(padded_source, (1, 1, 6))

        np.testing.assert_array_equal(cropped, [[[1, 2, 3]]])
        np.testing.assert_array_equal(padded, [[[0, 7, 8, 9, 0, 0]]])

    def test_mixed_crop_and_pad_and_singleton_axes(self):
        volume = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)

        result = center_crop_or_pad(volume, (3, 2, 4))

        self.assertEqual(result.shape, (3, 2, 4))
        np.testing.assert_array_equal(result[1, :, 1:3], volume[:, 1:3, :].reshape(2, 2))
        self.assertEqual(np.count_nonzero(result), 4)

    def test_equal_target_shape_returns_independent_copy(self):
        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

        result = center_crop_or_pad(volume, volume.shape)

        np.testing.assert_array_equal(result, volume)
        self.assertFalse(np.shares_memory(result, volume))

    def test_pipeline_keeps_shape_without_target_and_normalizes_first(self):
        volume = np.array([[[0, 10, 20, 30]]], dtype=np.float32)
        before = volume.copy()

        unchanged_shape = preprocess_volume(volume)
        cropped = preprocess_volume(volume, (1, 1, 2))
        padded = preprocess_volume(volume, (1, 1, 6))

        self.assertEqual(unchanged_shape.shape, volume.shape)
        np.testing.assert_allclose(unchanged_shape, [[[0, 1 / 3, 2 / 3, 1]]])
        np.testing.assert_allclose(cropped, [[[1 / 3, 2 / 3]]])
        np.testing.assert_allclose(padded, [[[0, 0, 1 / 3, 2 / 3, 1, 0]]])
        np.testing.assert_array_equal(volume, before)
        for result in (unchanged_shape, cropped, padded):
            self.assertEqual(result.dtype, np.float32)
            self.assertFalse(np.shares_memory(result, volume))

    def test_rejects_invalid_volume_inputs(self):
        invalid_cases = (
            (np.zeros((2, 3)), ValueError),
            (np.zeros((2, 3, 4, 1)), ValueError),
            (np.empty((0, 2, 3)), ValueError),
            (np.zeros((2, 0, 3)), ValueError),
            (np.zeros((2, 3, 0)), ValueError),
            (np.zeros((2, 3, 4), dtype=bool), TypeError),
            (np.zeros((2, 3, 4), dtype=complex), TypeError),
            (np.full((2, 3, 4), "1"), TypeError),
            (np.array([[[np.nan]]]), ValueError),
            (np.array([[[np.inf]]]), ValueError),
            (np.array([[[np.finfo(np.float64).max]]]), ValueError),
            ([[[1]]], TypeError),
        )
        for volume, error in invalid_cases:
            with self.subTest(volume=volume, error=error):
                for function, arguments in (
                    (normalize_volume, (volume,)),
                    (center_crop_or_pad, (volume, (1, 1, 1))),
                    (preprocess_volume, (volume,)),
                    (preprocess_volume, (volume, (1, 1, 1))),
                ):
                    with self.subTest(function=function.__name__, arguments=arguments):
                        with self.assertRaises(error):
                            function(*arguments)

    def test_rejects_invalid_target_shapes(self):
        volume = np.ones((2, 3, 4), dtype=np.float32)
        invalid_targets = (
            (),
            (2, 3),
            (2, 3, 4, 5),
            (2, 3, 0),
            (2, -1, 4),
            (2, True, 4),
            (2, 3.0, 4),
            [2, 3, 4],
            None,
        )
        for target in invalid_targets:
            with self.subTest(target=target), self.assertRaises(ValueError):
                center_crop_or_pad(volume, target)


if __name__ == "__main__":
    unittest.main()
