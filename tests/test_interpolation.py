"""Focused tests for sparse-to-dense through-plane interpolation."""

from dataclasses import replace
import unittest

import numpy as np

from src.interpolation import interpolate_sparse_acquisition
from src.sparse_simulator import SparseAcquisition, simulate_sparse_acquisition


def coordinate_volume(shape: tuple[int, int, int], axis: int) -> np.ndarray:
    """Make a non-cubic volume whose values are a linear ramp on ``axis``."""
    coordinates = np.arange(shape[axis], dtype=np.float32)
    reshape = [1, 1, 1]
    reshape[axis] = shape[axis]
    return np.broadcast_to(coordinates.reshape(reshape), shape).copy()


def coordinate_coded_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """Encode all coordinates so axis and in-plane geometry errors are visible."""
    x, y, z = np.indices(shape)
    return (100 * x + 10 * y + z).astype(np.float32)


class InterpolationTests(unittest.TestCase):
    def test_linear_reconstructs_known_ramps_for_factors_and_each_axis(self):
        shape = (7, 8, 11)
        for axis in range(3):
            for factor in (2, 3, 4):
                with self.subTest(axis=axis, factor=factor):
                    volume = coordinate_volume(shape, axis)
                    sparse = simulate_sparse_acquisition(
                        volume, factor=factor, axis=axis, sigma=0
                    )
                    dense = interpolate_sparse_acquisition(sparse)
                    self.assertEqual(dense.shape, shape)
                    self.assertEqual(dense.dtype, np.float32)
                    expected_ramp = np.interp(
                        np.arange(shape[axis]),
                        sparse.slice_indices,
                        sparse.slice_indices,
                    ).astype(np.float32)
                    reshape = [1, 1, 1]
                    reshape[axis] = shape[axis]
                    expected = np.broadcast_to(expected_ramp.reshape(reshape), shape)
                    np.testing.assert_allclose(dense, expected, rtol=0, atol=0)
                    observed = np.take(dense, sparse.slice_indices, axis=axis)
                    np.testing.assert_array_equal(observed, sparse.samples)

    def test_coordinate_coded_volumes_keep_in_plane_geometry_for_every_axis(self):
        shape = (7, 8, 11)
        volume = coordinate_coded_volume(shape)
        for axis in range(3):
            for factor in (2, 3, 4):
                with self.subTest(axis=axis, factor=factor):
                    sparse = simulate_sparse_acquisition(
                        volume, factor=factor, axis=axis, sigma=0
                    )
                    dense = interpolate_sparse_acquisition(sparse)
                    expected = volume.copy()
                    last = int(sparse.slice_indices[-1])
                    tail = [slice(None)] * 3
                    tail[axis] = slice(last + 1, None)
                    if last + 1 < shape[axis]:
                        endpoint = np.take(sparse.samples, [-1], axis=axis)
                        expected[tuple(tail)] = np.broadcast_to(
                            endpoint, expected[tuple(tail)].shape
                        )
                    np.testing.assert_array_equal(dense, expected)

    def test_nearest_selects_closest_slice_and_breaks_ties_toward_lower(self):
        volume = coordinate_volume((2, 3, 9), axis=2)
        sparse = simulate_sparse_acquisition(volume, factor=4, sigma=0)

        dense = interpolate_sparse_acquisition(sparse, method="nearest")

        np.testing.assert_array_equal(dense[0, 0], [0, 0, 0, 4, 4, 4, 4, 8, 8])

    def test_trailing_remainder_uses_endpoint_clamping(self):
        volume = coordinate_volume((2, 3, 8), axis=2)
        sparse = simulate_sparse_acquisition(volume, factor=3, sigma=0)

        dense = interpolate_sparse_acquisition(sparse)

        np.testing.assert_array_equal(dense[0, 0], [0, 1, 2, 3, 4, 5, 6, 6])

    def test_singleton_sample_broadcasts_and_singleton_dense_axis_works(self):
        samples = np.full((2, 3, 1), 4.25, dtype=np.float32)
        sparse = SparseAcquisition(samples, np.array([0]), (2, 3, 3), 2, 4)
        for method in ("linear", "nearest"):
            with self.subTest(method=method):
                dense = interpolate_sparse_acquisition(sparse, method=method)
                self.assertEqual(dense.shape, (2, 3, 3))
                np.testing.assert_array_equal(dense, np.full((2, 3, 3), 4.25))

        singleton = SparseAcquisition(samples, np.array([0]), (2, 3, 1), 2, 4)
        for method in ("linear", "nearest"):
            with self.subTest(method=method, dense_axis_length=1):
                dense = interpolate_sparse_acquisition(singleton, method=method)
                self.assertEqual(dense.shape, (2, 3, 1))
                np.testing.assert_array_equal(dense, samples)

    def test_float32_output_is_independent_and_input_is_not_mutated(self):
        samples = np.array([[[np.finfo(np.float32).max, -np.finfo(np.float32).max]]])
        before = samples.copy()
        sparse = SparseAcquisition(samples, np.array([0, 2]), (1, 1, 3), 2, 2)

        dense = interpolate_sparse_acquisition(sparse)

        self.assertEqual(dense.dtype, np.float32)
        self.assertTrue(np.isfinite(dense).all())
        self.assertEqual(dense[0, 0, 1], 0.0)
        np.testing.assert_array_equal(samples, before)
        self.assertFalse(np.shares_memory(dense, samples))
        np.testing.assert_array_equal(dense[..., [0, 2]], samples)

    def test_integer_and_float64_inputs_return_float32_without_mutation(self):
        for dtype in (np.int16, np.float64):
            for method in ("linear", "nearest"):
                with self.subTest(dtype=dtype, method=method):
                    volume = np.arange(2 * 3 * 8, dtype=dtype).reshape(2, 3, 8)
                    sparse = simulate_sparse_acquisition(
                        volume, factor=3, sigma=0
                    )
                    sparse = replace(sparse, samples=sparse.samples.astype(dtype))
                    before = sparse.samples.copy()
                    dense = interpolate_sparse_acquisition(sparse, method=method)
                    self.assertEqual(dense.dtype, np.float32)
                    self.assertFalse(np.shares_memory(dense, sparse.samples))
                    np.testing.assert_array_equal(sparse.samples, before)

    def test_rejects_invalid_method(self):
        sparse = simulate_sparse_acquisition(np.zeros((2, 3, 4)), factor=2, sigma=0)
        for method in ("cubic", "", None, np.array(["linear"])):
            with self.subTest(method=method), self.assertRaises(ValueError):
                interpolate_sparse_acquisition(sparse, method=method)

    def test_rejects_malformed_samples_and_metadata(self):
        valid = simulate_sparse_acquisition(np.zeros((3, 4, 7)), factor=2, sigma=0)
        cases = (
            (replace(valid, samples=np.zeros((3, 4, 3))), ValueError),
            (replace(valid, samples=np.zeros((3, 3, 4))), ValueError),
            (replace(valid, samples=np.zeros((3, 4, 4), dtype=complex)), TypeError),
            (replace(valid, samples=[[[]]]), TypeError),
            (replace(valid, original_shape=(3, 4)), ValueError),
            (replace(valid, axis=3), ValueError),
            (replace(valid, axis=1.0), TypeError),
            (replace(valid, factor=1), ValueError),
            (replace(valid, factor=2.5), TypeError),
            (replace(valid, slice_indices=np.array([0, 3, 4, 6])), ValueError),
            (replace(valid, slice_indices=np.array([0, 2, 2, 6])), ValueError),
            (replace(valid, slice_indices=np.array([0, 4, 2, 6])), ValueError),
            (replace(valid, slice_indices=np.array([0, 2, 4])), ValueError),
            (replace(valid, slice_indices=np.array([-1, 2, 4, 6])), ValueError),
            (replace(valid, slice_indices=np.array([0, 2, 4, 7])), ValueError),
            (replace(valid, slice_indices=np.array([0.0, 2.0, 4.0, 6.0])), ValueError),
            (replace(valid, slice_indices=[0, 2, 4, 6]), TypeError),
        )
        for malformed, error in cases:
            with self.subTest(malformed=malformed), self.assertRaises(error):
                interpolate_sparse_acquisition(malformed)

    def test_rejects_non_acquisition_input(self):
        with self.assertRaises(TypeError):
            interpolate_sparse_acquisition(np.zeros((2, 3, 4)))

    def test_simulator_integration_preserves_sampled_positions_and_shape(self):
        volume = np.arange(5 * 6 * 13, dtype=np.float32).reshape(5, 6, 13)
        sparse = simulate_sparse_acquisition(volume, factor=4, axis=1, sigma=0)

        dense = interpolate_sparse_acquisition(sparse)

        self.assertEqual(dense.shape, volume.shape)
        np.testing.assert_array_equal(
            np.take(dense, sparse.slice_indices, axis=1), sparse.samples
        )
        np.testing.assert_array_equal(dense[:, 0, :], volume[:, 0, :])
        np.testing.assert_array_equal(dense[:, 4, :], volume[:, 4, :])

    def test_simulator_blur_samples_are_preserved_exactly(self):
        volume = coordinate_coded_volume((6, 7, 8))
        sparse = simulate_sparse_acquisition(volume, factor=3, axis=0, sigma=1.0)

        dense = interpolate_sparse_acquisition(sparse)

        np.testing.assert_array_equal(
            np.take(dense, sparse.slice_indices, axis=0), sparse.samples
        )


if __name__ == "__main__":
    unittest.main()
