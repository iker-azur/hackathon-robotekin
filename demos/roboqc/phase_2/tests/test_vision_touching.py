"""Touching objects of different colors are separate detections, not one merged blob."""

import numpy as np
import pytest

from roboqc.vision.config import load_config
from roboqc.vision.detector import ObjectDetector

cv2 = pytest.importorskip("cv2")


def test_touching_objects_of_different_colors_are_separate():
    img = np.full((240, 400, 3), (115, 115, 115), dtype=np.uint8)  # neutral grey table
    cv2.rectangle(img, (290, 60), (349, 119), (0, 255, 255), -1)  # yellow cube
    cv2.rectangle(img, (280, 120), (359, 159), (0, 200, 0), -1)  # green box, touching the cube
    cv2.circle(img, (320, 190), 30, (255, 0, 0), -1)  # blue cylinder, touching the box

    detector = ObjectDetector(load_config("__no_such_file__.json"))
    colors = sorted(d.color for d in detector.detect(img))
    assert colors == ["amarillo", "azul", "verde"]


def test_overlapping_objects_of_the_same_color_are_split():
    img = np.full((200, 400, 3), (115, 115, 115), dtype=np.uint8)
    cv2.circle(img, (300, 100), 30, (0, 0, 255), -1)  # red disc
    cv2.circle(img, (350, 100), 30, (0, 0, 255), -1)  # second red disc, overlapping the first

    detector = ObjectDetector(load_config("__no_such_file__.json"))
    colors = sorted(d.color for d in detector.detect(img))
    assert colors == ["rojo", "rojo"]
