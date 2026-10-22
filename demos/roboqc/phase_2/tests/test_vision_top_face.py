"""A shaded cube is classified by its lit top face, not by its whole silhouette."""

import numpy as np
import pytest

from roboqc.vision.config import load_config
from roboqc.vision.detector import ObjectDetector

cv2 = pytest.importorskip("cv2")


def test_shaded_cube_is_square_from_its_top_face():
    img = np.full((240, 320, 3), (115, 115, 115), dtype=np.uint8)  # neutral grey table
    cv2.rectangle(img, (100, 80), (159, 139), (255, 0, 0), -1)  # lit top face, 60 x 60 px
    cv2.rectangle(img, (100, 140), (159, 169), (100, 0, 0), -1)  # dark side face below it

    detector = ObjectDetector(load_config("__no_such_file__.json"))
    (det,) = detector.detect(img)
    assert det.color == "azul"
    assert det.shape == "cuadrado"
    assert det.object_type == "cubo"


def test_spotlight_gradient_on_the_top_face_keeps_the_square():
    img = np.full((240, 320, 3), (115, 115, 115), dtype=np.uint8)
    for x in range(100, 160):  # top face brightness rises from left to right
        value = 150 + (x - 100) * 2  # 150 .. 248
        cv2.line(img, (x, 80), (x, 139), (value, 0, 0), 1)
    cv2.rectangle(img, (100, 140), (159, 169), (100, 0, 0), -1)  # dark side face

    detector = ObjectDetector(load_config("__no_such_file__.json"))
    (det,) = detector.detect(img)
    assert det.shape == "cuadrado"
