"""Run with python -m unittest discover -s demos/10_pin_inspection/tests."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout

from PIL import Image

SPEC = importlib.util.spec_from_file_location("download_pins", Path(__file__).resolve().parents[1] / "download_datasets.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.aliases = MODULE.label_map()

    def write(self, name, content=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def coco(self):
        return {"categories": [{"id": 7, "name": "Pieza sin perno"}, {"id": 19, "name": "Pieza_con_perno"},
                               {"id": 3, "name": "other"}],
                "images": [{"id": 1, "file_name": "mixed.jpg"}],
                "annotations": [{"id": 8, "image_id": 1, "category_id": 7},
                                {"id": 9, "image_id": 1, "category_id": 19}]}

    def test_coco_keeps_both_instances_in_mixed_scene_and_original_ids(self):
        self.write("train/images/mixed.jpg")
        self.write("train/annotations.json", json.dumps(self.coco()))
        rows = MODULE.inventory(self.root, self.aliases)
        self.assertEqual([r["label"] for r in rows], ["sin_perno", "con_perno"])
        self.assertEqual([r["class_id"] for r in rows], [7, 19])
        self.assertEqual(len({r["image"] for r in rows}), 1)

    def test_yolo_reads_class_order_and_keeps_label_line(self):
        self.write("data.yaml", "names: [Pieza_con_perno, Pieza_sin_perno, background]\n")
        self.write("valid/images/a.jpg")
        self.write("valid/labels/a.txt", "1 0.5 0.5 0.2 0.3\n0 0 0 1 0 1 1\n2 0 0 1 1\n")
        self.write("valid/images/unlabelled.jpg")
        rows = MODULE.inventory(self.root, self.aliases)
        self.assertEqual([r["label"] for r in rows], ["sin_perno", "con_perno"])
        self.assertEqual([r["line"] for r in rows], [1, 2])
        self.assertTrue(all(r["image"].endswith("a.jpg") for r in rows))

    def test_folder_labels_are_explicit_and_unknown_images_stay_unlabelled(self):
        self.write("train/Pieza_con_perno/a.png")
        self.write("train/Pieza_sin_perno/b.png")
        self.write("unlabelled/c.png")
        rows = MODULE.inventory(self.root, self.aliases)
        self.assertEqual({r["label"] for r in rows}, {"con_perno", "sin_perno"})
        self.assertEqual(len(rows), 2)

    def test_coco_rejects_missing_or_ambiguous_images(self):
        self.write("annotations.json", json.dumps(self.coco()))
        self.write("train/images/mixed.jpg")
        self.write("test/images/mixed.jpg")
        with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
            MODULE.inventory(self.root, self.aliases)

    def test_conflicting_yolo_definitions_fail(self):
        self.write("data.yaml", "names: [Pieza_con_perno, Pieza_sin_perno]")
        self.write("classes.txt", "Pieza_sin_perno\nPieza_con_perno")
        self.write("train/images/a.png")
        self.write("train/labels/a.txt", "0 0 0 1 1")
        with self.assertRaisesRegex(ValueError, "Conflicting YOLO"):
            MODULE.inventory(self.root, self.aliases)

    def test_download_pins_revision_and_skips_unrelated_dataset(self):
        repos = ["MikelRubi/parts", "MikelRubi/robot"]
        api = SimpleNamespace(list_datasets=lambda **kw: [SimpleNamespace(id=r) for r in repos],
                              dataset_info=lambda repo: SimpleNamespace(sha="abc123"),
                              list_repo_files=lambda repo, **kw: ["train/annotations.json", "train/images/mixed.jpg"]
                              if repo == repos[0] else ["meta/info.json"])
        calls = []

        def fetch(**kwargs):
            calls.append(kwargs)
            root = Path(kwargs["local_dir"])
            root.mkdir(parents=True, exist_ok=True)
            if kwargs["repo_id"] == repos[0]:
                (root / "train").mkdir(exist_ok=True)
                coco = self.coco()
                coco["images"].append({"id": 2, "file_name": "second.jpg"})
                coco["annotations"][1]["image_id"] = 2
                (root / "train/annotations.json").write_text(json.dumps(coco))
                if "allow_patterns" not in kwargs:
                    (root / "train/images").mkdir(exist_ok=True)
                    Image.new("RGB", (8, 8), "red").save(root / "train/images/mixed.jpg")
                    Image.new("RGB", (8, 8), "blue").save(root / "train/images/second.jpg")

        with redirect_stdout(io.StringIO()):
            result = MODULE.download("MikelRubi", self.root, self.aliases, api, fetch)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c["revision"] == "abc123" for c in calls))
        rows = [json.loads(line) for line in (self.root / "inventory.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        report = json.loads((self.root / "classification_report.json").read_text())
        self.assertEqual(report["counts"], {"ConPerno": 1, "SinPerno": 1})
        for item in report["files"]:
            self.assertEqual(Path(item["destination"]).read_bytes(), Path(item["sources"][0]).read_bytes())

    def test_network_failure_writes_incomplete_report(self):
        def fail(**kwargs):
            raise ConnectionError("No network")
        with redirect_stdout(io.StringIO()):
            result = MODULE.download("MikelRubi", self.root, self.aliases,
                                     SimpleNamespace(list_datasets=fail), None)
        self.assertEqual(result, 1)
        report = json.loads((self.root / "download_report.json").read_text())
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("No network", report["errors"][0]["error"])

    def test_custom_labels_cannot_reverse_default_mapping(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            MODULE.label_map(with_pin=["Pieza_sin_perno"])

    def test_snapshot_index_preserves_camera_pair_and_location(self):
        self.write("pin_snapshots/index.json", json.dumps({"images": [
            {"label": "pin", "location": 3, "front": "pin_front.jpg", "wrist": "pin_wrist.jpg"},
            {"label": "nopin", "location": 4, "front": "nopin_front.jpg", "wrist": "nopin_wrist.jpg"}]}))
        for name in ("pin_front.jpg", "pin_wrist.jpg", "nopin_front.jpg", "nopin_wrist.jpg"):
            self.photo("pin_snapshots/" + name, "red")
        evidence = MODULE.matching_classes(self.root, [], self.aliases)
        self.assertEqual({e["label"] for e in evidence}, {"con_perno", "sin_perno"})
        rows = MODULE.inventory(self.root, self.aliases)
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["label"] for r in rows], ["con_perno"] * 2 + ["sin_perno"] * 2)
        self.assertEqual(rows[0]["pair_id"], rows[1]["pair_id"])
        self.assertNotEqual(rows[0]["pair_id"], rows[2]["pair_id"])
        self.assertEqual([r["camera"] for r in rows[:2]], ["front", "wrist"])
        self.assertEqual(rows[0]["location"], 3)

    def test_snapshot_download_excludes_robot_videos_and_skips_unlabelled_lerobot(self):
        repos = ["MikelRubi/snapshots", "MikelRubi/robot"]
        api = SimpleNamespace(list_datasets=lambda **kw: [SimpleNamespace(id=r) for r in repos],
                              dataset_info=lambda repo: SimpleNamespace(sha="revision"),
                              list_repo_files=lambda *args, **kw: ["meta/info.json", "meta/tasks.parquet", "video.mp4"])
        calls = []
        def fetch(**kwargs):
            calls.append(kwargs)
            root = Path(kwargs["local_dir"])
            (root / "meta").mkdir(parents=True, exist_ok=True)
            (root / "meta/info.json").write_text(json.dumps({"robot_type": "so101", "features": {"action": {}}}))
            if kwargs["repo_id"] != repos[0]:
                return
            folder = root / "pin_snapshots"
            folder.mkdir(exist_ok=True)
            (folder / "index.json").write_text(json.dumps({"images": [
                {"label": "pin", "front": "p_front.jpg", "wrist": "p_wrist.jpg"},
                {"label": "nopin", "front": "n_front.jpg", "wrist": "n_wrist.jpg"}]}))
            if kwargs.get("allow_patterns") == ["pin_snapshots/*"]:
                for name, color in (("p_front", "red"), ("p_wrist", "blue"), ("n_front", "green"), ("n_wrist", "white")):
                    Image.new("RGB", (8, 8), color).save(folder / f"{name}.jpg")
        with redirect_stdout(io.StringIO()):
            result = MODULE.download("MikelRubi", self.root, self.aliases, api, fetch)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1]["allow_patterns"], ["pin_snapshots/*"])
        report = json.loads((self.root / "classification_report.json").read_text())
        self.assertEqual(report["counts"], {"ConPerno": 2, "SinPerno": 2})

    def test_snapshot_missing_camera_fails_instead_of_silently_skipping(self):
        self.write("pin_snapshots/index.json", json.dumps({"images": [
            {"label": "pin", "front": "missing.jpg", "wrist": "also_missing.jpg"}]}))
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            MODULE.inventory(self.root, self.aliases)

    def photo(self, name, color):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(path)
        return path

    def test_classification_preserves_images_and_avoids_filename_collisions(self):
        first = self.photo("one/photo.png", "red")
        second = self.photo("two/photo.png", "blue")
        samples = [{"local_image": str(first), "label": "con_perno"},
                   {"local_image": str(second), "label": "con_perno"}]
        report = MODULE.classify_images(samples, self.root)
        self.assertEqual(report["counts"]["ConPerno"], 2)
        self.assertEqual(len(list((self.root / "ConPerno").iterdir())), 2)
        again = MODULE.classify_images(samples, self.root)
        self.assertEqual(report["files"], again["files"])
        self.assertEqual(report["visual_review"], "not_performed")

    def test_classification_excludes_mixed_scenes_and_damaged_images(self):
        mixed = self.photo("mixed.png", "green")
        damaged = self.write("damaged.jpg", "not an image")
        samples = [{"local_image": str(mixed), "label": "con_perno"},
                   {"local_image": str(mixed), "label": "sin_perno"},
                   {"local_image": str(damaged), "label": "sin_perno"}]
        report = MODULE.classify_images(samples, self.root)
        self.assertEqual(report["counts"], {"ConPerno": 0, "SinPerno": 0})
        self.assertEqual(len(report["pending"]), 2)

    def test_identical_bytes_with_conflicting_labels_are_excluded(self):
        first = self.photo("first.png", "red")
        second = self.photo("second.png", "red")
        report = MODULE.classify_images([{"local_image": str(first), "label": "con_perno"},
                                         {"local_image": str(second), "label": "sin_perno"}], self.root)
        self.assertEqual(report["files"], [])
        self.assertEqual(report["pending"][0]["reason"], "mixed_or_conflicting_labels")

    def test_previous_opposite_class_copy_is_reported(self):
        photo = self.photo("photo.png", "red")
        MODULE.classify_images([{"local_image": str(photo), "label": "con_perno"}], self.root)
        report = MODULE.classify_images([{"local_image": str(photo), "label": "sin_perno"}], self.root)
        self.assertEqual(report["files"], [])
        self.assertEqual(report["pending"][0]["reason"], "previous_copy_in_opposite_class")


if __name__ == "__main__":
    unittest.main()
