"""Color classification, color normalization and shape identification."""

import numpy as np
import pytest

from roboqc.commands.entities import normalize_color
from roboqc.vision.color import classify_color, normalize_ranges
from roboqc.vision.config import DEFAULTS
from roboqc.vision.shape import classify_shape, compute_features

cv2 = pytest.importorskip("cv2")


def test_color_normalization_accepts_spanish_forms():
    assert normalize_color("rojas") == "rojo"
    assert normalize_color("Amarillas") == "amarillo"
    assert normalize_color("marrón") == "marron"
    assert normalize_color("violeta") is None  # unsupported color is rejected, not guessed


def test_normalize_ranges_canonicalizes_names_and_validates_values():
    parsed = normalize_ranges({"rojas": [[0, 8, 90, 255, 60, 255]]})
    assert "rojo" in parsed and parsed["rojo"] == [(0, 8, 90, 255, 60, 255)]
    with pytest.raises(ValueError):
        normalize_ranges({"azul": [[0, 200, 0, 255, 0, 255]]})  # hue above 179
    with pytest.raises(ValueError):
        normalize_ranges({"azul": [[0, 10, 0, 255]]})  # wrong length


def test_classify_color_picks_dominant_and_rejects_mixed():
    assert classify_color({"rojo": 0.9, "naranja": 0.1}, 0.6) == ("rojo", 0.9)
    assert classify_color({"rojo": 0.4, "naranja": 0.35}, 0.6) == (None, 0.4)
    assert classify_color({}, 0.6) == (None, 0.0)


def _square(side=50, origin=(0, 0)):
    x, y = origin
    return np.array([[[x, y]], [[x + side, y]], [[x + side, y + side]], [[x, y + side]]], dtype=np.int32)


def _rect(w, h):
    return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)


def _circle_contour(radius=60):  # large enough that pixel staircasing keeps circularity above 0.85
    mask = np.zeros((2 * radius + 10, 2 * radius + 10), dtype=np.uint8)
    cv2.circle(mask, (radius + 5, radius + 5), radius, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


def _classify(contour):
    shape_cfg = DEFAULTS["shape"]
    feats = compute_features(contour, shape_cfg["approx_epsilon_ratio"])
    return classify_shape(feats, shape_cfg["circularity_min"], shape_cfg["square_aspect_tolerance"])


def test_square_is_cube():
    shape, obj_type, _ = _classify(_square())
    assert (shape, obj_type) == ("cuadrado", "cubo")


def test_rectangle_is_prism():
    shape, obj_type, _ = _classify(_rect(120, 40))
    assert (shape, obj_type) == ("rectangulo", "prisma")


def test_circle_is_reported_as_circular_not_guessed():
    shape, obj_type, _ = _classify(_circle_contour())
    assert shape == "circulo"
    assert obj_type == "circular"  # cylinder vs sphere is not decidable from above


def test_orientation_is_reported_for_rotated_square():
    rotated = np.array([[[50, 0]], [[100, 50]], [[50, 100]], [[0, 50]]], dtype=np.int32)  # 45 degrees
    feats = compute_features(rotated, DEFAULTS["shape"]["approx_epsilon_ratio"])
    assert 40.0 <= feats.angle_deg <= 50.0


def test_irregular_outline_is_unknown():
    star = np.array([[[50, 0]], [[60, 40]], [[100, 50]], [[60, 60]], [[50, 100]],
                     [[40, 60]], [[0, 50]], [[40, 40]]], dtype=np.int32)
    shape, obj_type, conf = _classify(star)
    assert obj_type == "desconocido"
    assert conf == 0.0
