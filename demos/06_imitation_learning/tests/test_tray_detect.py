"""The tray detector on synthetic frames: orange floor, grey metal piece, and the debounce."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def load():
    path = Path(__file__).resolve().parent.parent / "tray_detect.py"
    spec = importlib.util.spec_from_file_location("tray_detect_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


tray = load()
ORANGE, SILVER = (205, 95, 55), (170, 172, 176)


def frame(piece=None, size=(480, 640)):
    image = np.empty((*size, 3), dtype=np.uint8)
    image[:] = ORANGE
    rng = np.random.default_rng(0)
    image = np.clip(image.astype(int) + rng.integers(-12, 12, image.shape), 0, 255).astype(np.uint8)  # floor grain
    if piece:
        x0, y0, x1, y1 = piece
        image[y0:y1, x0:x1] = SILVER
    return image


class DetectPieceTests(unittest.TestCase):
    def test_empty_floor_is_not_a_piece(self):
        reading = tray.detect_piece(frame())
        self.assertFalse(reading.present)
        self.assertEqual(reading.area, 0)

    def test_grey_piece_inside_the_roi_is_found(self):
        reading = tray.detect_piece(frame(piece=(250, 250, 300, 300)))
        self.assertTrue(reading.present)
        self.assertGreater(reading.area, 2000)

    def test_piece_outside_the_roi_is_ignored(self):
        self.assertFalse(tray.detect_piece(frame(piece=(560, 60, 620, 120))).present)  # top right: the arm's side

    def test_speckle_below_the_blob_threshold_is_ignored(self):
        self.assertFalse(tray.detect_piece(frame(piece=(250, 250, 260, 260))).present)  # 100 px

    def test_threshold_scales_with_resolution(self):
        small = np.ascontiguousarray(frame(piece=(250, 250, 300, 300))[::2, ::2])  # 320x240, the piece is 25x25
        self.assertTrue(tray.detect_piece(small).present)
        self.assertFalse(tray.detect_piece(np.ascontiguousarray(frame(piece=(250, 250, 260, 260))[::2, ::2])).present)

    def test_roi_parsing(self):
        self.assertEqual(tray.parse_roi("0.1,0.2,0.7,0.9"), (0.1, 0.2, 0.7, 0.9))
        for bad in ("0.1,0.2,0.7", "0.7,0.2,0.1,0.9", "0,0,1,2", "a,b,c,d"):
            with self.assertRaises(ValueError):
                tray.parse_roi(bad)

    def test_overlay_copies_the_frame_and_marks_the_rectangle(self):
        original = frame()
        shown = tray.draw_overlay(original, tray.detect_piece(original))
        self.assertIsNot(shown, original)
        self.assertFalse(np.array_equal(shown, original))


class PieceWatcherTests(unittest.TestCase):
    def test_needs_continuous_presence(self):
        watcher = tray.PieceWatcher(1.0)
        self.assertFalse(watcher.update(True, 0.0))
        self.assertFalse(watcher.update(True, 0.9))
        self.assertTrue(watcher.update(True, 1.0))

    def test_a_gap_restarts_the_count(self):
        watcher = tray.PieceWatcher(1.0)
        watcher.update(True, 0.0)
        watcher.update(False, 0.5)
        self.assertFalse(watcher.update(True, 1.0))
        self.assertFalse(watcher.update(True, 1.9))
        self.assertTrue(watcher.update(True, 2.0))

    def test_reset_forgets_the_piece(self):
        watcher = tray.PieceWatcher(0.5)
        watcher.update(True, 0.0)
        watcher.reset()
        self.assertFalse(watcher.update(True, 1.0))


if __name__ == "__main__":
    unittest.main()
