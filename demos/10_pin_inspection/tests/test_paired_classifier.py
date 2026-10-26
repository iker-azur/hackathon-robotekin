"""Pair integrity, grouped splits and a real CPU training/checkpoint round trip."""

import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_data import CLASSES, load_pairs, split_pairs


def make_dataset(root, sessions=5):
    rows = []
    for session in range(sessions):
        repo = f"MikelRubi/session_{session}"
        for number, label in enumerate(CLASSES):
            for camera_index, camera in enumerate(("front", "wrist")):
                name = f"pin_snapshots/{number}_{camera}.png"
                image = root / "raw" / repo / "revision" / name
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (40, 24), (session * 31 + 4, number * 130 + 7, camera_index * 120 + 9)).save(image)
                rows.append({"repo_id": repo, "revision": "revision", "pair_id": f"pin_snapshots/index.json:{number}",
                             "image": name, "local_image": str(image), "camera": camera,
                             "label": label, "location": number})
    (root / "inventory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = make_dataset(self.root)

    def save_rows(self):
        (self.root / "inventory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def test_groups_keep_both_views_and_sessions_disjoint(self):
        report = load_pairs(self.root)
        self.assertEqual(report["counts"], {"sin_perno": 5, "con_perno": 5})
        split = split_pairs(report["pairs"])
        self.assertEqual(split, split_pairs(report["pairs"]))
        self.assertEqual(len(split["train"]), 8)
        for partition in ("train", "validation"):
            self.assertEqual({pair["label"] for pair in split[partition]}, set(CLASSES))
        train_sessions = {pair["repo_id"] for pair in split["train"]}
        val_sessions = {pair["repo_id"] for pair in split["validation"]}
        self.assertFalse(train_sessions & val_sessions)
        hashes = lambda part: {pair[camera + "_sha256"] for pair in split[part] for camera in ("front", "wrist")}
        self.assertFalse(hashes("train") & hashes("validation"))

    def test_missing_view_and_corrupt_image_are_excluded(self):
        self.rows.pop(0)
        Path(self.rows[2]["local_image"]).write_bytes(b"broken")
        self.save_rows()
        report = load_pairs(self.root)
        self.assertEqual(len(report["pairs"]), 8)
        self.assertEqual(len(report["rejected"]), 2)

    def test_duplicates_keep_all_sources_and_sessions_together(self):
        copies = []
        for row in self.rows[:2]:
            copied = {**row, "repo_id": "MikelRubi/copied_session", "revision": "other"}
            target = self.root / "raw" / copied["repo_id"] / copied["revision"] / copied["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["local_image"], target)
            copied["local_image"] = str(target)
            copies.append(copied)
        self.rows.extend(copies)
        self.save_rows()
        report = load_pairs(self.root)
        self.assertEqual(report["duplicates_removed"], 1)
        duplicate = next(pair for pair in report["pairs"] if len(pair["sources"]) == 2)
        session_pair = next(pair for pair in report["pairs"] if pair["repo_id"] == "MikelRubi/session_0")
        self.assertEqual(duplicate["group"], session_pair["group"])
        copies[0]["label"] = copies[1]["label"] = "con_perno"
        # rows contains the same dict objects; changing labels creates a contradictory duplicate pair.
        self.save_rows()
        self.assertTrue(any(item["reason"] == "identical_pair_with_conflicting_labels"
                            for item in load_pairs(self.root)["rejected"]))

    def test_one_shared_image_connects_sessions_even_for_different_pairs(self):
        shutil.copy2(self.rows[0]["local_image"], self.rows[4]["local_image"])
        pairs = load_pairs(self.root)["pairs"]
        first = next(pair for pair in pairs if pair["repo_id"] == "MikelRubi/session_0")
        second = next(pair for pair in pairs if pair["repo_id"] == "MikelRubi/session_1")
        self.assertEqual(first["group"], second["group"])

    def test_relocated_raw_files_and_selected_sessions(self):
        relocated = self.root / "relocated"
        relocated.mkdir()
        shutil.copytree(self.root / "raw", relocated / "raw")
        shutil.copy2(self.root / "inventory.jsonl", relocated / "inventory.jsonl")
        report = load_pairs(relocated, ["MikelRubi/session_1", "MikelRubi/session_2"])
        self.assertEqual(len(report["pairs"]), 4)
        self.assertTrue(all(Path(pair["front"]).is_relative_to(relocated.resolve()) for pair in report["pairs"]))

    def test_not_enough_sessions_or_classes_never_falls_back_to_image_split(self):
        report = load_pairs(self.root, ["MikelRubi/session_0"])
        with self.assertRaisesRegex(ValueError, "sesiones"):
            split_pairs(report["pairs"])
        with self.assertRaisesRegex(ValueError, "ambas clases"):
            split_pairs([pair for pair in load_pairs(self.root)["pairs"] if pair["label"] == "con_perno"])

    def test_unpaired_photos_and_conflicting_view_labels_are_rejected(self):
        self.rows[0].pop("pair_id")
        self.rows[2]["label"] = "sin_perno"
        self.save_rows()
        report = load_pairs(self.root)
        self.assertEqual(Counter(item["reason"] for item in report["rejected"]),
                         {"unpaired_image": 1, "incomplete_or_duplicate_camera": 1, "conflicting_or_unknown_labels": 1})


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_dataset(self.root)

    def settings(self, name="run"):
        return {"data_root": str(self.root), "output": str(self.root / name), "epochs": 2,
                "batch_size": 2, "device": "cpu", "pretrained": False}

    def test_cpu_finetuning_checkpoint_and_pair_inference_roundtrip(self):
        from model import PairedClassifier, load_classifier, predict, preprocess
        from training import train
        torch = self.torch
        torch.manual_seed(42)
        initial_weight = PairedClassifier(pretrained=False).backbone.conv1.weight.detach().clone()
        events = []
        result = train(self.settings(), events.append)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([event["epoch"] for event in events if event["event"] == "epoch"], [1, 2])
        folder = self.root / "run"
        loaded = load_classifier(folder / "last.pt")
        self.assertFalse(torch.equal(initial_weight, loaded.backbone.conv1.weight))
        pair = load_pairs(self.root)["pairs"][0]
        probabilities = predict(folder / "best.pt", {"front": pair["front"], "wrist": pair["wrist"]})
        self.assertAlmostEqual(sum(probabilities["probabilities"].values()), 1, places=5)
        self.assertIn(probabilities["label"], CLASSES)
        self.assertEqual(probabilities["cameras"], ["front", "wrist"])
        with self.assertRaisesRegex(ValueError, "wrist"):
            predict(folder / "best.pt", {"front": pair["front"]})
        self.assertEqual(preprocess(pair["front"]).shape, (3, 224, 224))
        before = loaded(preprocess(pair["front"]).unsqueeze(0), preprocess(pair["wrist"]).unsqueeze(0))
        reloaded = load_classifier(folder / "last.pt")
        after = reloaded(preprocess(pair["front"]).unsqueeze(0), preprocess(pair["wrist"]).unsqueeze(0))
        torch.testing.assert_close(before, after)
        saved = torch.load(folder / "last.pt", weights_only=True)
        saved["classes"].reverse()
        torch.save(saved, folder / "wrong.pt")
        with self.assertRaisesRegex(ValueError, "clases"):
            load_classifier(folder / "wrong.pt")

    def test_single_camera_training_and_inference_use_only_that_view(self):
        from model import load_classifier, predict
        from training import train
        for camera in ("front", "wrist"):
            with self.subTest(camera=camera):
                result = train({**self.settings(camera), "cameras": [camera]})
                self.assertEqual(result["status"], "completed")
                folder = self.root / camera
                self.assertEqual(load_classifier(folder / "best.pt").cameras, (camera,))
                self.assertEqual(json.loads((folder / "run_config.json").read_text())["cameras"], [camera])
                pair = load_pairs(self.root)["pairs"][0]
                prediction = predict(folder / "best.pt", {camera: pair[camera]})
                self.assertEqual(prediction["cameras"], [camera])
                self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1, places=5)

    def test_augmentation_is_redrawn_per_load_and_reproducible_with_seed(self):
        from model import preprocess
        from training import DEFAULT_AUGMENTATION, PairedImages, build_augmentation
        torch = self.torch
        pairs = load_pairs(self.root)["pairs"]
        plain = PairedImages(pairs, ["front", "wrist"])
        torch.testing.assert_close(plain[0][0], preprocess(pairs[0]["front"]))
        settings = {**DEFAULT_AUGMENTATION, "horizontal_flip": 1.0, "rotation": 1.0, "translation": 1.0,
                    "perspective": 1.0, "rotation_degrees": 90.0}
        augmented = PairedImages(pairs, ["front", "wrist"], build_augmentation(settings))
        torch.manual_seed(0)
        first = augmented[0]
        second = augmented[0]
        self.assertFalse(torch.equal(first[0], second[0]), "Each epoch must draw new transforms")
        self.assertFalse(torch.equal(first[0], plain[0][0]))
        self.assertEqual(first[0].shape, (3, 224, 224))
        torch.manual_seed(0)
        torch.testing.assert_close(augmented[0][0], first[0])
        flip_only = PairedImages(pairs, ["front"], build_augmentation({**DEFAULT_AUGMENTATION, "horizontal_flip": 1.0}))
        torch.testing.assert_close(flip_only[0][0], plain[0][0].flip(-1))

    def test_training_with_augmentation_records_settings(self):
        from training import train
        events = []
        augmentation = {"vertical_flip": 0.5, "perspective": 0.3, "perspective_scale": 0.4}
        result = train({**self.settings(), "augmentation": augmentation}, events.append)
        self.assertEqual(result["status"], "completed")
        saved = json.loads((self.root / "run/run_config.json").read_text())["augmentation"]
        self.assertEqual(saved["vertical_flip"], 0.5)
        self.assertEqual(saved["horizontal_flip"], 0.0)
        self.assertTrue(any("vertical_flip 50%" in event.get("text", "") for event in events))

    def test_best_checkpoint_prefers_lower_loss_then_accuracy(self):
        from training import selection_key, train
        self.assertLess(selection_key({"accuracy": 0.7, "loss": 0.1}), selection_key({"accuracy": 0.9, "loss": 0.2}))
        self.assertLess(selection_key({"accuracy": 0.8, "loss": 0.3}), selection_key({"accuracy": 0.7, "loss": 0.3}))
        result = train({**self.settings(), "epochs": 4})
        history = json.loads((self.root / "run/history.json").read_text())
        expected = min(history, key=lambda row: (selection_key(row["validation"]), row["epoch"]))["epoch"]
        self.assertEqual(result["best_epoch"], expected)
        self.assertEqual(self.torch.load(self.root / "run/best.pt", weights_only=True)["epoch"], expected)
        metrics = json.loads((self.root / "run/metrics.json").read_text())
        self.assertEqual(metrics["selection"], "minimum_validation_loss_then_maximum_accuracy")

    def test_cooperative_stop_preserves_complete_checkpoint(self):
        from training import train
        settings = self.settings()
        stop = self.root / "stop"
        settings["stop_file"] = str(stop)
        def interrupt(event):
            if event["event"] == "epoch":
                stop.write_text("stop")
        result = train(settings, interrupt)
        self.assertEqual(result["status"], "interrupted")
        self.assertTrue((self.root / "run/best.pt").is_file())
        self.assertEqual(json.loads((self.root / "run/status.json").read_text())["completed_epochs"], 1)

    def test_metrics_and_invalid_settings(self):
        from training import TrainConfig, classification_metrics
        metrics = classification_metrics([[3, 1], [2, 4]])
        self.assertAlmostEqual(metrics["accuracy"], 0.7)
        self.assertAlmostEqual(metrics["per_class"]["con_perno"]["precision"], 0.8)
        self.assertAlmostEqual(metrics["per_class"]["con_perno"]["recall"], 4 / 6)
        with self.assertRaises(ValueError):
            TrainConfig(**{**self.settings(), "learning_rate": float("nan")}).validate()
        for cameras in ([], ["wrist", "front"], ["top"], "front"):
            with self.assertRaisesRegex(ValueError, "cameras"):
                TrainConfig(**{**self.settings(), "cameras": cameras}).validate()
        for augmentation in ({"rotation": 1.5}, {"blur": 0.5}, {"translation_fraction": 0.9},
                             {"perspective": float("nan")}, [0.5]):
            with self.assertRaisesRegex(ValueError, "augmentation"):
                TrainConfig(**{**self.settings(), "augmentation": augmentation}).validate()

    def test_pretrained_download_failure_is_not_replaced_by_random_weights(self):
        from unittest.mock import patch

        from training import train
        with patch("training.PairedClassifier", side_effect=ConnectionError("weights unavailable")), self.assertRaises(ConnectionError):
            train({**self.settings(), "pretrained": True})
        state = json.loads((self.root / "run/status.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertFalse((self.root / "run/best.pt").exists())


if __name__ == "__main__":
    unittest.main()
