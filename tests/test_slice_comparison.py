"""Focused tests for missing-slice scoring, export, and review panels."""

import csv
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.slice_comparison import (
    compare_slices, missing_slice_indices, render_slice_panels, save_slice_comparison,
)
from src.sparse_simulator import simulate_sparse_acquisition


class SliceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((3, 4, 6), dtype=np.float32)
        self.acquisition = simulate_sparse_acquisition(self.target, 4, sigma=0)

    def test_missing_indices_for_all_axes_factors_and_trailing_slices(self):
        for axis in range(3):
            for factor in (2, 4, 20):
                acquisition = simulate_sparse_acquisition(self.target, factor, axis=axis)
                actual = missing_slice_indices(acquisition)
                expected = [i for i in range(self.target.shape[axis]) if i % factor]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.ndim, 1)
                self.assertIn(actual.dtype.kind, "iu")

    def test_scores_exclude_retained_errors_and_pool_before_psnr(self):
        for axis in range(3):
            target = np.zeros((4, 4, 4), dtype=np.uint8)
            acquisition = simulate_sparse_acquisition(target, 2, axis=axis)
            baseline = target.copy()
            planes = np.moveaxis(baseline, axis, 0)
            planes[0] = 100
            planes[1] = 1
            planes[3] = 3
            comparison = compare_slices(target, baseline, target, acquisition, data_range=2)
            self.assertEqual([r['slice_index'] for r in comparison.baseline_per_slice], [1, 3])
            self.assertEqual(comparison.baseline_missing_metrics['mae'], 2)
            self.assertEqual(comparison.baseline_missing_metrics['mse'], 5)
            self.assertAlmostEqual(comparison.baseline_missing_metrics['psnr'], 10 * np.log10(4 / 5))
            self.assertEqual(comparison.baseline_all_metrics['mse'], 2502.5)
            self.assertEqual(comparison.reconstruction_missing_metrics['psnr'], float('inf'))
            self.assertEqual(comparison.reconstruction_all_metrics['mae'], 0)
            self.assertEqual(comparison.axis, axis)
            np.testing.assert_array_equal(target, 0)
            np.testing.assert_array_equal(planes[0], 100)
            self.assertFalse(np.shares_memory(comparison.retained_indices, acquisition.slice_indices))
            with self.assertRaises(FrozenInstanceError):
                comparison.axis = 0

    def test_singleton_has_empty_indices_but_cannot_be_compared(self):
        target = np.zeros((2, 3, 1))
        acquisition = simulate_sparse_acquisition(target, 2)
        self.assertEqual(missing_slice_indices(acquisition).size, 0)
        with self.assertRaisesRegex(ValueError, 'no missing'):
            compare_slices(target, target, target, acquisition)

    def test_acquisition_validation_is_shared(self):
        for acquisition in (
            replace(self.acquisition, axis=-1),
            replace(self.acquisition, factor=1),
            replace(self.acquisition, slice_indices=np.array([0, 3])),
            replace(self.acquisition, samples=np.zeros((3, 4, 1))),
        ):
            with self.subTest(acquisition=acquisition), self.assertRaises(ValueError):
                missing_slice_indices(acquisition)
        with self.assertRaises(TypeError):
            missing_slice_indices(None)

    def test_invalid_volumes_and_ranges(self):
        invalid = [
            (self.target.tolist(), TypeError),
            (np.zeros((3, 4)), ValueError),
            (np.zeros((0, 4, 6)), ValueError),
            (np.zeros((3, 4, 1)), ValueError),
            (np.full(self.target.shape, np.nan), ValueError),
            (np.full(self.target.shape, np.inf), ValueError),
            (self.target.astype(bool), TypeError),
            (self.target.astype(complex), TypeError),
            (self.target.astype(object), TypeError),
        ]
        for bad, error in invalid:
            for position in range(3):
                volumes = [self.target] * 3
                volumes[position] = bad
                with self.subTest(position=position), self.assertRaises(error):
                    compare_slices(*volumes, self.acquisition)
        for value, error in ((True, TypeError), ('1', TypeError), (0, ValueError), (-1, ValueError), (np.inf, ValueError), (np.nan, ValueError)):
            with self.assertRaises(error):
                compare_slices(self.target, self.target, self.target, self.acquisition, data_range=value)
        wrong_shape = np.zeros((3, 4, 7))
        with self.assertRaisesRegex(ValueError, 'original_shape'):
            compare_slices(wrong_shape, wrong_shape, wrong_shape, self.acquisition)

    def test_csv_and_panels(self):
        baseline = self.target + 0.5
        comparison = compare_slices(self.target, baseline, self.target, self.acquisition)
        with tempfile.TemporaryDirectory() as directory:
            path = save_slice_comparison(comparison, Path(directory) / 'scores' / 'scores.csv')
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]['slice_index'], '1')
            self.assertEqual(float(rows[0]['mse']), 0.25)
            self.assertEqual(rows[-1]['scope'], 'all')
            self.assertEqual(float(rows[-1]['psnr']), float('inf'))
            paths = render_slice_panels(
                self.target, baseline, self.target, self.acquisition, directory,
                slice_indices=np.array([5, 1]),
            )
            self.assertEqual([p.name for p in paths], ['axis_2_slice_0001.png', 'axis_2_slice_0005.png'])
            for panel in paths:
                self.assertEqual(panel.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
                self.assertGreater(panel.stat().st_size, 1000)
            for indices in (np.array([0]), np.array([6]), np.array([], dtype=int), np.array([1.0])):
                with self.assertRaises(ValueError):
                    render_slice_panels(self.target, baseline, self.target, self.acquisition, directory, slice_indices=indices)


if __name__ == '__main__':
    unittest.main()
