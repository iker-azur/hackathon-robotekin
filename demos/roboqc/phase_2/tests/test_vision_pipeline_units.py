"""Detection on synthetic frames, tracking across frames, and quality rules."""

import copy

import numpy as np
import pytest

from roboqc.vision.config import DEFAULTS, load_config
from roboqc.vision.detector import Detection, ObjectDetector
from roboqc.vision.quality import DEFECTIVE, GOOD, UNCERTAIN, QualityRule, QualityRuleSet
from roboqc.vision.tracker import Tracker

cv2 = pytest.importorskip("cv2")


def _cfg(**overrides):
    cfg = load_config("__no_such_file__.json")  # missing file -> built-in defaults
    for key, value in overrides.items():
        if isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def _scene():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)  # neutral dark-gray table
    cv2.rectangle(frame, (100, 100), (160, 160), (0, 0, 255), -1)  # red square
    cv2.circle(frame, (320, 240), 40, (255, 0, 0), -1)  # blue disc
    cv2.rectangle(frame, (450, 300), (550, 350), (0, 255, 0), -1)  # green rectangle
    return frame


def test_detects_colors_and_shapes_on_synthetic_scene():
    dets = ObjectDetector(_cfg()).detect(_scene())
    by_color = {d.color: d for d in dets}
    assert set(by_color) == {"rojo", "azul", "verde"}
    assert by_color["rojo"].object_type == "cubo"
    assert by_color["azul"].object_type == "circular"
    assert by_color["verde"].object_type == "prisma"
    assert all(not d.touches_border for d in dets)


def test_two_same_colored_objects_are_two_detections():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (110, 110), (0, 0, 255), -1)
    cv2.rectangle(frame, (300, 50), (360, 110), (0, 0, 255), -1)
    dets = ObjectDetector(_cfg()).detect(frame)
    assert [d.color for d in dets] == ["rojo", "rojo"]


def test_object_cut_by_frame_edge_is_partial():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (0, 200), (40, 240), (0, 0, 255), -1)
    (det,) = ObjectDetector(_cfg()).detect(frame)
    assert det.touches_border is True


def test_roi_filters_objects_and_keeps_absolute_coordinates():
    cfg = _cfg(roi=[50, 50, 200, 200])
    dets = ObjectDetector(cfg).detect(_scene())
    assert [d.color for d in dets] == ["rojo"]
    assert dets[0].bbox[0] == 100  # red square starts at x=100 in full-frame pixels


def test_tiny_specks_are_filtered_by_min_area():
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (200, 200), 3, (0, 0, 255), -1)
    assert ObjectDetector(_cfg()).detect(frame) == []


def test_black_objects_need_explicit_dark_detection():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # light background
    cv2.rectangle(frame, (100, 100), (160, 160), (0, 0, 0), -1)
    assert ObjectDetector(_cfg()).detect(frame) == []
    dets = ObjectDetector(_cfg(foreground={"detect_dark": True})).detect(frame)
    assert [d.color for d in dets] == ["negro"]


def test_empty_frame_returns_no_detections():
    assert ObjectDetector(_cfg()).detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


# ---- tracking ---------------------------------------------------------------

def _det(color, x, y, obj_type="cubo", touches=False):
    return Detection(
        color=color, color_confidence=0.9, shape="cuadrado" if obj_type == "cubo" else "rectangulo",
        object_type=obj_type, shape_confidence=0.9, bbox=(int(x) - 25, int(y) - 25, 50, 50),
        centroid=(float(x), float(y)), area_px=2500.0, orientation_deg=0.0, solidity=0.95,
        vertices=4, touches_border=touches,
    )


def test_same_colored_objects_keep_their_ids_across_frames():
    tracker = Tracker(DEFAULTS["tracking"])
    tracker.update([_det("rojo", 100, 100), _det("rojo", 300, 100)], 0.0)
    events = tracker.update([_det("rojo", 106, 100), _det("rojo", 294, 100)], 0.1)
    positions = {t.object_id: t.detection.centroid[0] for t in tracker.tracks}
    assert positions == {"OBJ_001": 106.0, "OBJ_002": 294.0}
    assert sorted(events.entered) == ["OBJ_001", "OBJ_002"]


def test_single_frame_noise_is_not_reported():
    tracker = Tracker(DEFAULTS["tracking"])
    tracker.update([_det("rojo", 100, 100)], 0.0)
    assert tracker.tracks[0].confirmed is False


def test_missing_object_is_occluded_then_disappears():
    cfg = copy.deepcopy(DEFAULTS["tracking"])
    cfg["max_missed_frames"] = 2
    tracker = Tracker(cfg)
    tracker.update([_det("rojo", 100, 100), _det("azul", 300, 100)], 0.0)
    tracker.update([_det("rojo", 100, 100), _det("azul", 300, 100)], 0.1)
    tracker.update([_det("rojo", 100, 100)], 0.2)
    blue = next(t for t in tracker.tracks if t.stable_color == "azul")
    assert blue.status == "OCCLUDED"
    tracker.update([_det("rojo", 100, 100)], 0.3)
    events = tracker.update([_det("rojo", 100, 100)], 0.4)
    assert events.disappeared == ["OBJ_002"]
    assert [t.object_id for t in tracker.tracks] == ["OBJ_001"]


def test_color_is_stabilized_by_majority_vote():
    tracker = Tracker(DEFAULTS["tracking"])
    tracker.update([_det("rojo", 100, 100)], 0.0)
    tracker.update([_det("naranja", 101, 100)], 0.1)
    tracker.update([_det("rojo", 102, 100)], 0.2)
    assert tracker.tracks[0].stable_color == "rojo"


# ---- quality ----------------------------------------------------------------

def _rules(**quality):
    cfg = copy.deepcopy(DEFAULTS["quality"])
    cfg.update(quality)
    return QualityRuleSet(cfg)


def test_color_rule_marks_matching_objects_defective_only():
    rules = _rules()
    rules.add_rule(QualityRule(name="piezas azules", verdict=DEFECTIVE, color="azul"))
    blue = rules.evaluate("azul", "cubo", 0.95, 2500, 0.9, partial=False)
    red = rules.evaluate("rojo", "cubo", 0.95, 2500, 0.9, partial=False)
    assert blue.quality == DEFECTIVE
    assert blue.reasons == ["regla activa: piezas azules defectuosas"]
    assert red.quality == GOOD


def test_newer_rule_for_same_target_replaces_older():
    rules = _rules()
    rules.add_rule(QualityRule(name="piezas azules", verdict=DEFECTIVE, color="azul"))
    rules.add_rule(QualityRule(name="piezas azules", verdict=GOOD, color="azul"))
    assert len(rules.rules) == 1
    assert rules.evaluate("azul", "cubo", 0.95, 2500, 0.9, partial=False).quality == GOOD


def test_missing_corners_are_defective():
    result = _rules().evaluate("rojo", "cubo", 0.70, 2500, 0.9, partial=False)
    assert result.quality == DEFECTIVE
    assert "bordes o esquinas anomalos" in result.reasons[0]


def test_partially_visible_object_is_uncertain_never_good():
    result = _rules().evaluate("rojo", "cubo", 0.95, 2500, 0.9, partial=True)
    assert result.quality == UNCERTAIN


def test_inconclusive_color_is_uncertain():
    assert _rules().evaluate(None, "cubo", 0.95, 2500, 0.2, partial=False).quality == UNCERTAIN


def test_unexpected_type_is_defective_when_configured():
    rules = _rules(expected_types=["cubo"])
    assert rules.evaluate("rojo", "prisma", 0.95, 2500, 0.9, partial=False).quality == DEFECTIVE
    assert rules.evaluate("rojo", "cubo", 0.95, 2500, 0.9, partial=False).quality == GOOD


def test_area_outside_expected_range_is_defective():
    rules = _rules(expected_area_px=[2000, 3000], expected_area_tolerance=0.1)
    assert rules.evaluate("rojo", "cubo", 0.95, 5000, 0.9, partial=False).quality == DEFECTIVE
    assert rules.evaluate("rojo", "cubo", 0.95, 2500, 0.9, partial=False).quality == GOOD
