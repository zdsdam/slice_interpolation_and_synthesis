"""Focused tests for held-out checkpoint evaluation and tiled inference."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from src.evaluate import _checkpoint_file, evaluate, reconstruct_volume
from src.models.unet3d import ResidualUNet3D
from src.sparse_simulator import simulate_sparse_acquisition as real_simulate_sparse_acquisition
from src.train import split_subjects


def write_volume(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), str(path))


def write_manifest(path: Path, entries: list[tuple[str, Path]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "path"])
        writer.writerows((subject, str(scan)) for subject, scan in entries)
    return path


def make_checkpoint(path: Path, train_paths: list[Path] | None = None, validation_paths: list[Path] | None = None) -> Path:
    model = ResidualUNet3D(base_channels=1)
    torch.nn.init.zeros_(model.residual_head.weight)
    torch.nn.init.zeros_(model.residual_head.bias)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"base_channels": 1},
        "training_config": {
            "sparsity_factor": 2,
            "slice_axis": 2,
            "sigma": 0.0,
            "patch_size": (8, 8, 8),
            "seed": 13,
        },
        "train_subjects": ["train"],
        "validation_subjects": ["validation"],
        "manifest_paths": {
            "train": [str(item) for item in (train_paths or [])] or [str(path.parent / "train.nii.gz")],
            "validation": [str(item) for item in (validation_paths or [])] or [str(path.parent / "validation.nii.gz")],
        },
        "epoch": 3,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_tiling_covers_edges_pads_small_volumes_and_disables_gradients(self):
        class RecordingIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_enabled = []
                self.training_flags = []

            def forward(self, values):
                self.grad_enabled.append(torch.is_grad_enabled())
                self.training_flags.append(self.training)
                return values

        model = RecordingIdentity()
        source = np.arange(9 * 13 * 17, dtype=np.float32).reshape(9, 13, 17) / 100
        actual = reconstruct_volume(model, source, torch.device("cpu"), (8, 8, 8))
        self.assertEqual(actual.shape, source.shape)
        np.testing.assert_allclose(actual, source, atol=1e-7)
        self.assertGreater(len(model.grad_enabled), 1)
        self.assertTrue(all(flag is False for flag in model.grad_enabled))
        self.assertTrue(all(flag is False for flag in model.training_flags))

        small = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
        padded = reconstruct_volume(model, small, torch.device("cpu"), (8, 8, 8))
        self.assertEqual(padded.shape, small.shape)
        np.testing.assert_allclose(padded, small, atol=1e-7)

    def test_end_to_end_scores_both_methods_writes_csv_and_uses_one_sparse_simulation(self):
        rng = np.random.default_rng(4)
        scans = []
        for name, subject, offset in (("a", "test-a", 0), ("b", "test-a", 4), ("c", "test-b", 9)):
            path = self.root / f"{name}.nii.gz"
            write_volume(path, rng.normal(size=(8, 8, 8)) + offset)
            scans.append((subject, path))
        manifest = write_manifest(self.root / "test.csv", scans)
        checkpoint = make_checkpoint(self.root / "checkpoints" / "best.pt")
        output = self.root / "results" / "evaluation.csv"
        with patch("src.evaluate.simulate_sparse_acquisition", wraps=real_simulate_sparse_acquisition) as simulate:
            rows = evaluate(manifest, checkpoint.parent, output, "cpu")
        self.assertEqual(simulate.call_count, 3)
        self.assertEqual(len([row for row in rows if row["scope"] == "volume"]), 6)
        self.assertEqual(len([row for row in rows if row["scope"] == "subject"]), 4)
        aggregates = {row["method"]: row for row in rows if row["scope"] == "aggregate"}
        self.assertEqual(set(aggregates), {"interpolation", "unet"})
        volume_rows = [row for row in rows if row["scope"] == "volume"]
        baseline_scores = {
            (row["subject_id"], row["path"]): {name: row[name] for name in ("mae", "mse", "psnr", "ssim")}
            for row in volume_rows if row["method"] == "interpolation"
        }
        unet_scores = {
            (row["subject_id"], row["path"]): {name: row[name] for name in ("mae", "mse", "psnr", "ssim")}
            for row in volume_rows if row["method"] == "unet"
        }
        self.assertEqual(set(baseline_scores), set(unet_scores))
        for key in baseline_scores:
            for metric in ("mae", "mse", "psnr", "ssim"):
                self.assertAlmostEqual(baseline_scores[key][metric], unet_scores[key][metric], places=6)
        subject_rows = [row for row in rows if row["scope"] == "subject" and row["method"] == "unet"]
        self.assertEqual(len(subject_rows), 2)
        self.assertAlmostEqual(aggregates["unet"]["mae"], np.mean([row["mae"] for row in subject_rows]))
        self.assertAlmostEqual(aggregates["unet"]["psnr"], np.mean([row["psnr"] for row in subject_rows]))
        self.assertEqual(aggregates["unet"]["checkpoint_epoch"], 3)
        self.assertTrue(output.is_file())
        with output.open(newline="", encoding="utf-8") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), len(rows))
        self.assertEqual(saved[0]["tile_size"], "8x8x8")
        self.assertEqual(saved[0]["data_range"], "1.0")
        repeated = evaluate(manifest, checkpoint, self.root / "results" / "repeated.csv", "cpu")
        for first, second in zip(rows, repeated):
            for metric in ("mae", "mse", "psnr", "ssim"):
                self.assertEqual(first[metric], second[metric])

    def test_loaded_nonzero_residual_weights_change_only_unet_scores(self):
        scan = self.root / "test.nii.gz"
        rng = np.random.default_rng(17)
        write_volume(scan, rng.normal(size=(8, 8, 8)))
        manifest = write_manifest(self.root / "test.csv", [("test", scan)])
        checkpoint = make_checkpoint(self.root / "model.pt")
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        raw["model_state_dict"]["residual_head.bias"].fill_(0.15)
        torch.save(raw, checkpoint)
        rows = evaluate(manifest, checkpoint, self.root / "out.csv", "cpu")
        by_method = {row["method"]: row for row in rows if row["scope"] == "volume"}
        self.assertNotEqual(by_method["unet"]["mse"], by_method["interpolation"]["mse"])
        self.assertNotEqual(by_method["unet"]["mae"], by_method["interpolation"]["mae"])

    def test_rejects_test_subject_and_renamed_test_file_leakage(self):
        scan = self.root / "test.nii.gz"
        write_volume(scan, np.zeros((8, 8, 8)))
        manifest = write_manifest(self.root / "test.csv", [("test", scan)])
        checkpoint = make_checkpoint(self.root / "model.pt")
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        raw["train_subjects"] = ["test"]
        raw["manifest_paths"] = {"test": [str(scan)]}
        raw["validation_subjects"] = []
        torch.save(raw, checkpoint)
        with self.assertRaisesRegex(ValueError, "development_subjects.csv"):
            evaluate(manifest, checkpoint, self.root / "out.csv", "cpu")

        raw["train_subjects"] = ["train"]
        raw["validation_subjects"] = ["test"]
        raw["manifest_paths"] = {"train": [str(self.root / "train.nii.gz")], "test": [str(scan)]}
        torch.save(raw, checkpoint)
        with self.assertRaisesRegex(ValueError, "development_subjects.csv"):
            evaluate(manifest, checkpoint, self.root / "out.csv", "cpu")

        raw["train_subjects"] = ["different-id"]
        raw["validation_subjects"] = []
        raw["manifest_paths"] = {"different-id": [str(scan)]}
        torch.save(raw, checkpoint)
        with self.assertRaisesRegex(ValueError, "file paths overlap"):
            evaluate(manifest, checkpoint, self.root / "out.csv", "cpu")

    def test_requires_complete_provenance_and_best_checkpoint_preference(self):
        path = self.root / "checkpoints" / "latest.pt"
        make_checkpoint(path)
        self.assertEqual(_checkpoint_file(path.parent).name, "latest.pt")
        best = make_checkpoint(path.parent / "best.pt")
        self.assertEqual(_checkpoint_file(path.parent), best.resolve())

        raw = torch.load(best, map_location="cpu", weights_only=True)
        raw.pop("training_config")
        torch.save(raw, best)
        scan = self.root / "test.nii.gz"
        write_volume(scan, np.zeros((8, 8, 8)))
        manifest = write_manifest(self.root / "test.csv", [("test", scan)])
        with self.assertRaisesRegex(ValueError, "training_config"):
            evaluate(manifest, best, self.root / "out.csv", "cpu")

    def test_rejects_inconsistent_split_provenance_and_missing_model_config(self):
        checkpoint = make_checkpoint(self.root / "model.pt")
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        raw["manifest_paths"].pop("validation")
        torch.save(raw, checkpoint)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            from src.evaluate import _required_checkpoint

            _required_checkpoint(raw, checkpoint)
        raw["manifest_paths"] = {"train": ["train.nii.gz"], "validation": ["validation.nii.gz"]}
        raw["model_config"] = {}
        torch.save(raw, checkpoint)
        scan = self.root / "test.nii.gz"
        write_volume(scan, np.zeros((8, 8, 8)))
        manifest = write_manifest(self.root / "test.csv", [("test", scan)])
        with self.assertRaisesRegex(ValueError, "base_channels"):
            evaluate(manifest, checkpoint, self.root / "out.csv", "cpu")

    def test_checked_in_split_manifests_are_reproducible_and_disjoint(self):
        def manifest_ids(filename):
            with Path(filename).open(newline="", encoding="utf-8") as stream:
                return [(row["subject_id"], row["path"]) for row in csv.DictReader(stream)]

        all_entries = manifest_ids("data/training_subjects.csv")
        development_entries = manifest_ids("data/development_subjects.csv")
        test_entries = manifest_ids("data/test_subjects.csv")
        all_ids = sorted({subject for subject, _ in all_entries})
        expected_development, expected_test = split_subjects(all_ids, 0.2, 42)
        self.assertEqual(sorted({subject for subject, _ in development_entries}), expected_development)
        self.assertEqual(sorted({subject for subject, _ in test_entries}), expected_test)
        self.assertEqual(set(development_entries) | set(test_entries), set(all_entries))
        self.assertTrue(set(development_entries).isdisjoint(test_entries))


if __name__ == "__main__":
    unittest.main()
