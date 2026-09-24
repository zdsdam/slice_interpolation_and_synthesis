"""Focused tests for manifest parsing, patch generation, and training."""

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.models.unet3d import ResidualUNet3D
from src.interpolation import interpolate_sparse_acquisition
from src.preprocessing import preprocess_volume
from src.sparse_simulator import simulate_sparse_acquisition
from src.train import (
    SparsePatchDataset,
    TrainingConfig,
    read_manifest,
    run_epoch,
    run_training,
    split_subjects,
)


def write_volume(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
    nib.save(image, str(path))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_manifest(self, entries):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject_id", "path"])
            writer.writerows(entries)
        return manifest

    def test_subject_split_keeps_subject_scans_grouped_and_is_deterministic(self):
        first = split_subjects(["b", "a", "c", "d"], 0.25, 17)
        self.assertEqual(first, split_subjects(["d", "c", "b", "a"], 0.25, 17))
        self.assertEqual(set(first[0]).isdisjoint(first[1]), True)
        self.assertEqual(set(first[0]) | set(first[1]), {"a", "b", "c", "d"})
        self.assertEqual(split_subjects(["a"], 0.2, 2), (["a"], []))

    def test_manifest_resolves_relative_paths_and_rejects_duplicate_files(self):
        path = self.root / "scan.nii.gz"
        write_volume(path, np.ones((16, 16, 16)))
        manifest = self.make_manifest([("subject", "scan.nii.gz")])
        self.assertEqual(read_manifest(manifest), {"subject": [path.resolve()]})

        second_path = self.root / "scan2.nii.gz"
        write_volume(second_path, np.full((16, 16, 16), 2))
        grouped = self.make_manifest(
            [("subject", "scan.nii.gz"), ("subject", "scan2.nii.gz")]
        )
        grouped_subjects = read_manifest(grouped)
        self.assertEqual(len(grouped_subjects["subject"]), 2)
        self.assertEqual(split_subjects(grouped_subjects, 0.2, 1), (["subject"], []))

        duplicate = self.make_manifest(
            [("one", "scan.nii.gz"), ("two", str(path.resolve()))]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_manifest(duplicate)

    def test_patches_are_axis_preserving_pairs_and_validation_is_repeatable(self):
        path = self.root / "oriented.nii.gz"
        grid = np.indices((24, 16, 16))[0].astype(np.float32)
        write_volume(path, grid)
        second_path = self.root / "oriented_second.nii.gz"
        write_volume(second_path, grid + 2)
        subjects = {"s": [path, second_path]}
        config = TrainingConfig(patch_size=(16, 16, 16), base_channels=1, slice_axis=0)
        dataset = SparsePatchDataset(
            subjects, config, patches_per_volume=2, seed=9, shuffle=True
        )
        first = list(dataset)
        dataset.epoch = 1
        second = list(dataset)
        self.assertEqual(first[0][0].shape, (1, 16, 16, 16))
        self.assertEqual(first[0][0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(first[0][0]).all())
        hr = preprocess_volume(grid)
        target = first[0][1][0].numpy()
        origin = int(round(float(target[0, 0, 0]) * 23))
        np.testing.assert_array_equal(target, hr[origin : origin + 16, :, :])
        np.testing.assert_allclose(
            first[0][0][0].numpy(),
            interpolate_sparse_acquisition(
                simulate_sparse_acquisition(
                    target, factor=4, axis=0, sigma=config.sigma
                )
            ),
        )
        self.assertFalse(torch.equal(first[0][0], first[0][1]))
        # Fixed seeds reproduce validation patches even if the epoch counter advances.
        validation = SparsePatchDataset(
            subjects, config, patches_per_volume=2, seed=99, shuffle=False
        )
        val_first = list(validation)
        validation.epoch = 4
        val_second = list(validation)
        for (x1, y1), (x2, y2) in zip(val_first, val_second):
            torch.testing.assert_close(x1, x2)
            torch.testing.assert_close(y1, y2)
        self.assertEqual(len(second), 4)
        self.assertTrue(any(not torch.equal(a[1], b[1]) for a, b in zip(first, second)))

    def test_small_volumes_and_invalid_patch_configs_fail_clearly(self):
        path = self.root / "small.nii.gz"
        write_volume(path, np.ones((16, 16, 8)))
        config = TrainingConfig(patch_size=(16, 16, 16), base_channels=1)
        dataset = SparsePatchDataset({"s": [path]}, config, patches_per_volume=1, seed=0, shuffle=False)
        with self.assertRaisesRegex(ValueError, "smaller than patch_size"):
            list(dataset)
        with self.assertRaisesRegex(ValueError, "multiples of 8"):
            TrainingConfig(patch_size=(15, 16, 16))

    def test_one_epoch_checkpoints_reload_and_optimizer_updates(self):
        manifest_entries = []
        for index in range(3):
            scan = self.root / f"subject{index}.nii.gz"
            volume = np.random.default_rng(index).normal(size=(16, 16, 16))
            write_volume(scan, volume)
            manifest_entries.append((f"sub-{index}", scan.name))
        manifest = self.make_manifest(manifest_entries)
        config = TrainingConfig(
            epochs=1,
            batch_size=1,
            patch_size=(16, 16, 16),
            patches_per_volume=1,
            validation_fraction=0.34,
            seed=4,
            base_channels=1,
            device="cpu",
        )
        torch.manual_seed(config.seed)
        initial_model = ResidualUNet3D(base_channels=config.base_channels)
        initial_parameters = [parameter.detach().clone() for parameter in initial_model.parameters()]
        history = run_training(manifest, config, self.root / "checkpoints")
        self.assertEqual(len(history), 1)
        self.assertGreater(history[0]["train_loss"], 0)
        self.assertGreater(history[0]["val_loss"], 0)

        checkpoint = torch.load(self.root / "checkpoints" / "latest.pt", map_location="cpu", weights_only=True)
        model = ResidualUNet3D(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(initial_parameters, model.parameters()))
        )
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(len(checkpoint["train_subjects"]) + len(checkpoint["validation_subjects"]), 3)
        self.assertEqual(len(checkpoint["manifest_paths"]), 3)
        self.assertTrue((self.root / "checkpoints" / "best.pt").is_file())

        no_validation = TrainingConfig(
            epochs=1,
            patch_size=(16, 16, 16),
            patches_per_volume=1,
            validation_fraction=0,
            base_channels=1,
            device="cpu",
        )
        run_training(manifest, no_validation, self.root / "train-only")
        self.assertFalse((self.root / "train-only" / "best.pt").exists())

    def test_run_epoch_updates_training_weights_and_validation_is_read_only(self):
        torch.manual_seed(3)
        model = ResidualUNet3D(base_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        z, y, x = np.mgrid[0:16, 0:16, 0:16].astype(np.float32)
        smooth_signal = (
            0.15 * z / 15
            + 0.10 * y / 15
            + 0.10 * x / 15
            + 0.65 * np.exp(-((z - 8) ** 2 + (y - 7) ** 2 + (x - 9) ** 2) / (2 * 3.5**2))
        )
        hr = preprocess_volume(smooth_signal)
        inputs_np = interpolate_sparse_acquisition(
            simulate_sparse_acquisition(hr, factor=4, axis=2, sigma=1.0)
        )
        inputs = torch.from_numpy(inputs_np).unsqueeze(0).unsqueeze(0)
        targets = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)
        loader = [(inputs, targets)]
        initial = run_epoch(model, loader, torch.device("cpu"))
        original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        run_epoch(model, loader, torch.device("cpu"), optimizer)
        self.assertTrue(
            any(not torch.equal(before, after) for before, after in zip(original_parameters, model.parameters()))
        )
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for _ in range(29):
            run_epoch(model, loader, torch.device("cpu"), optimizer)
        trained_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        previous_grads = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
        final = run_epoch(model, loader, torch.device("cpu"))
        self.assertTrue(
            all(torch.equal(before, after) for before, after in zip(trained_parameters, model.parameters()))
        )
        self.assertTrue(
            all(
                (before is None and parameter.grad is None)
                or (before is not None and torch.equal(before, parameter.grad))
                for before, parameter in zip(previous_grads, model.parameters())
            )
        )
        self.assertLess(final, initial * 0.5)

    def test_run_epoch_weights_batch_losses_by_sample_and_disables_validation_gradients(self):
        class RecordingIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_modes = []

            def forward(self, inputs):
                self.grad_modes.append(torch.is_grad_enabled())
                return inputs

        model = RecordingIdentity()
        loader = [
            (torch.ones((2, 1, 8, 8, 8)), torch.zeros((2, 1, 8, 8, 8))),
            (torch.full((1, 1, 8, 8, 8), 2.0), torch.zeros((1, 1, 8, 8, 8))),
        ]
        loss = run_epoch(model, loader, torch.device("cpu"))
        self.assertAlmostEqual(loss, 4 / 3)
        self.assertEqual(model.grad_modes, [False, False])


if __name__ == "__main__":
    unittest.main()
