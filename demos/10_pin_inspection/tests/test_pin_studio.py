"""Studio wiring and real QProcess scan/train/predict, without downloading weights."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "10_training_studio"))

import pin_inspection_page as PAGE
from app import TrainingWindow
from PySide6.QtCore import QProcess, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSpinBox
from test_paired_classifier import make_dataset


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / "data"
        self.data.mkdir()
        make_dataset(self.data)
        self.output_patch = patch.object(PAGE, "OUTPUTS", self.root / "outputs")
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.window = TrainingWindow()
        self.addCleanup(self.window.close)
        self.page = self.window.pin_page
        self.page.data_root.setText(str(self.data))
        self.page.python_path.setText(sys.executable)

    def wait(self, timeout=120):
        deadline = time.monotonic() + timeout
        while self.page.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertFalse(self.page.busy, self.page.log.toPlainText())
        self.assertFalse(self.page._error, self.page.log.toPlainText())

    def test_navigation_and_act_settings_are_independent(self):
        original = self.window.steps.value()
        self.page.epochs.setValue(7)
        self.assertEqual(self.window.steps.value(), original)
        button = next(button for button, _, _ in self.window.nav_buttons if button.text() == "Inspección de pernos")
        button.click()
        self.assertIs(self.window.pages.currentWidget(), self.page)
        self.assertEqual(self.page.tabs.count(), 3)

    def test_real_scan_and_partition_worker(self):
        self.page.scan()
        self.wait()
        self.assertEqual(self.page.sessions.count(), 5)
        self.assertEqual(self.page.pair_list.count(), 10)
        self.page.prepare()
        self.wait()
        self.assertIn("Partición válida", self.page.status.text())
        self.assertTrue(self.page.tabs.isEnabled())

    def test_numeric_controls_support_mouse_buttons_and_direct_typing(self):
        self.window._show_page(self.window.pages.indexOf(self.page))
        self.page.tabs.setCurrentIndex(1)
        self.window.show()
        self.app.processEvents()
        for name, expected in (("epochs", 75), ("batch_size", 32), ("seed", 123)):
            with self.subTest(control=name):
                spin = getattr(self.page, name)
                self.assertGreater(spin.lineEdit().width(), 100, "Editor must not collapse to one pixel")
                option = QStyleOptionSpinBox()
                spin.initStyleOption(option)
                up = spin.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin)
                self.assertFalse(up.isEmpty(), "Numeric controls need visible, clickable step buttons")
                previous = spin.value()
                QTest.mouseClick(spin, Qt.LeftButton, pos=up.center())
                self.assertEqual(spin.value(), previous + 1)
                QTest.mouseClick(spin.lineEdit(), Qt.LeftButton)
                QTest.keyClick(spin.lineEdit(), Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(spin.lineEdit(), str(expected))
                QTest.keyClick(spin.lineEdit(), Qt.Key_Tab)
                self.assertEqual(spin.value(), expected)
                self.assertEqual(spin.text(), str(expected))
        self.page._set_busy(True)
        self.assertFalse(self.page.epochs.isEnabled())
        self.assertFalse(self.page.batch_size.isEnabled())
        self.page._set_busy(False)
        self.assertTrue(self.page.epochs.isEnabled())
        self.assertTrue(self.page.batch_size.isEnabled())

    def test_training_receives_epochs_and_batch_edited_with_keyboard(self):
        from paired_data import load_pairs
        pairs = load_pairs(self.data)["pairs"]
        self.page._action = "scan"
        sessions = {source["repo_id"]: {} for pair in pairs for source in pair["sources"]}
        self.page._handle_event({"event": "data", "pairs": pairs, "sessions": sessions,
                                 "rejected": [], "duplicates_removed": 0})
        for spin, value in ((self.page.epochs, "50"), (self.page.batch_size, "8")):
            spin.lineEdit().setFocus()
            spin.lineEdit().selectAll()
            QTest.keyClicks(spin.lineEdit(), value)
            QTest.keyClick(spin.lineEdit(), Qt.Key_Enter)
        with patch.object(self.page, "_launch") as launch:
            self.page.start_training()
        self.assertEqual(launch.call_args.args[1]["epochs"], 50)
        self.assertEqual(launch.call_args.args[1]["batch_size"], 8)
        self.assertTrue(all(value == 0 for key, value in launch.call_args.args[1]["augmentation"].items()
                            if key in ("horizontal_flip", "vertical_flip", "rotation", "translation", "perspective")))

    def test_augmentation_controls_send_selected_probabilities(self):
        self.window._show_page(self.window.pages.indexOf(self.page))
        self.page.tabs.setCurrentIndex(1)
        self.window.show()
        self.app.processEvents()
        check, chance, _, _ = self.page.augmentations["rotation"]
        self.assertFalse(check.isEnabled(), "Transforms stay off until augmentation is enabled")
        self.page.augmentation_box.setChecked(True)
        self.assertTrue(check.isEnabled())
        check.setChecked(True)
        QTest.mouseClick(chance.lineEdit(), Qt.LeftButton)
        QTest.keyClick(chance.lineEdit(), Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(chance.lineEdit(), "35")
        QTest.keyClick(chance.lineEdit(), Qt.Key_Tab)
        rotation = self.page.augmentations["rotation"][3]
        rotation.setValue(25)
        self.page.augmentations["perspective"][0].setChecked(True)
        self.page.augmentations["translation"][3].setValue(20)
        settings = self.page.augmentation_settings()
        self.assertEqual(settings["rotation"], 0.35)
        self.assertEqual(settings["rotation_degrees"], 25)
        self.assertEqual(settings["perspective"], 0.3)
        self.assertEqual(settings["translation"], 0.0)
        self.assertEqual(settings["translation_fraction"], 0.2)
        self.assertGreater(chance.lineEdit().width(), 20)
        self.page._set_busy(True)
        self.assertFalse(check.isEnabled())
        self.page._set_busy(False)
        self.assertTrue(check.isEnabled())
        from training import TrainConfig
        TrainConfig(data_root="d", output="o", augmentation=settings).validate()

    def test_subprocess_training_and_inference_roundtrip(self):
        self.page.scan()
        self.wait()
        self.page.epochs.setValue(1)
        self.page.batch_size.setValue(2)
        self.page.device.setCurrentText("cpu")
        self.page.run_name.setText("smoke")
        launch = self.page._launch

        def offline_launch(action, settings, **kwargs):
            if action == "train":
                settings["pretrained"] = False
            return launch(action, settings, **kwargs)

        with patch.object(self.page, "_launch", offline_launch):
            self.page.start_training()
        self.wait()
        output = self.root / "outputs/smoke"
        self.assertTrue((output / "best.pt").is_file(), self.page.log.toPlainText())
        self.assertIn("accuracy", self.page.metrics.text())
        self.assertEqual(self.page.runs.count(), 1)
        pair = self.page.pairs[0]
        for camera in ("front", "wrist"):
            self.page.inference_paths[camera].setText(pair[camera])
        self.page.predict()
        self.wait()
        self.assertIn("%", self.page.prediction.text())
        self.assertIn("perno", self.page.prediction.text())

    def test_single_camera_run_only_needs_that_image(self):
        self.page.scan()
        self.wait()
        self.page.epochs.setValue(1)
        self.page.batch_size.setValue(2)
        self.page.device.setCurrentText("cpu")
        self.page.cameras.setCurrentText("Solo wrist")
        self.page.run_name.setText("wrist_only")
        launch = self.page._launch

        def offline_launch(action, settings, **kwargs):
            if action == "train":
                self.assertEqual(settings["cameras"], ["wrist"])
                settings["pretrained"] = False
            return launch(action, settings, **kwargs)

        with patch.object(self.page, "_launch", offline_launch):
            self.page.start_training()
        self.wait()
        self.assertIn("wrist", self.page.result_metrics.toPlainText())
        self.page.inference_paths["front"].clear()
        self.page.inference_paths["wrist"].setText(self.page.pairs[0]["wrist"])
        self.page.predict()
        self.wait()
        self.assertIn("%", self.page.prediction.text())

    def test_heavy_job_guard_and_invalid_interpreter(self):
        with patch.object(self.page.process, "state", return_value=QProcess.Running):
            self.page._action = "train"
            with patch.object(self.window, "_show_error") as error:
                self.window._launch_process(Path(sys.executable), [], "training", "act")
                error.assert_called_once()
        with patch.object(self.window.process, "state", return_value=QProcess.Running):
            self.page.start_training()
            self.assertIn("ACT", self.page.status.text())
        self.page.python_path.setText(str(self.root / "missing-python"))
        self.page.scan()
        self.assertIn("intérprete", self.page.status.text())
        self.assertFalse(self.page.busy)

    def test_protocol_errors_and_forced_stop_mark_run_interrupted(self):
        self.page._consume_line("@@PIN@@not-json")
        self.assertTrue(self.page._error)
        self.page._run_output = self.root / "interrupted"
        self.page._run_output.mkdir()
        path = self.page._run_output / "status.json"
        path.write_text('{"status":"running","completed_epochs":1}')
        self.page._stop_requested = True
        self.page._mark_run_stopped()
        self.assertEqual(json.loads(path.read_text())["status"], "interrupted")
        self.assertEqual(json.loads(path.read_text())["completed_epochs"], 1)


if __name__ == "__main__":
    unittest.main()
