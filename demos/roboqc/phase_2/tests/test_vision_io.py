"""Camera errors, calibration and configuration round-trips."""

import json

import pytest

from roboqc.vision.calibration import load_calibration
from roboqc.vision.camera import CameraError, FpsMeter, open_source
from roboqc.vision.config import DEFAULTS, load_config, save_roi

cv2 = pytest.importorskip("cv2")


def test_missing_file_is_a_camera_error(tmp_path):
    with pytest.raises(CameraError, match="No existe el archivo"):
        open_source(str(tmp_path / "no_existe.png"))


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "notas.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(CameraError, match="Formato no soportado"):
        open_source(str(path))


def test_corrupt_image_is_a_camera_error(tmp_path):
    path = tmp_path / "roto.png"
    path.write_bytes(b"esto no es una imagen")
    with pytest.raises(CameraError, match="No se puede leer la imagen"):
        open_source(str(path))


def test_missing_webcam_index_gives_actionable_message():
    with pytest.raises(CameraError, match="05_test_camera.py --list"):
        open_source(99)


def test_still_image_source_returns_frames(tmp_path):
    import numpy as np
    path = tmp_path / "pieza.png"
    cv2.imwrite(str(path), np.zeros((20, 30, 3), dtype=np.uint8))
    with open_source(str(path)) as src:
        frame = src.read()
        assert frame.shape == (20, 30, 3)


def test_fps_meter_is_zero_without_samples():
    meter = FpsMeter()
    assert meter.fps == 0.0
    assert meter.latency_ms == 0.0


def test_calibration_maps_pixels_to_millimetres(tmp_path):
    from roboqc.vision.calibration import WorkspaceCalibration
    pixels = [[0, 0], [100, 0], [100, 50], [0, 50]]
    world = [[0, 0], [300, 0], [300, 150], [0, 150]]
    cal = WorkspaceCalibration.from_points(pixels, world, frame="table", units="mm")
    x, y = cal.pixel_to_world(50, 25)
    assert abs(x - 150) < 1e-3 and abs(y - 75) < 1e-3
    assert cal.describe(50, 25)["units"] == "mm"

    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"pixel_points": pixels, "world_points": world}), encoding="utf-8")
    assert load_calibration(str(path)) is not None


def test_no_calibration_file_means_no_world_coordinates():
    assert load_calibration(None) is None
    assert load_calibration("no_such_calibration.json") is None


def test_calibration_needs_four_points():
    from roboqc.vision.calibration import WorkspaceCalibration
    with pytest.raises(ValueError):
        WorkspaceCalibration.from_points([[0, 0], [1, 0]], [[0, 0], [1, 0]])


def test_missing_config_uses_defaults():
    cfg = load_config("no_such_config.json")
    assert cfg["tracking"] == DEFAULTS["tracking"]


def test_roi_round_trip_keeps_other_keys(tmp_path):
    path = tmp_path / "vision.json"
    path.write_text(json.dumps({"camera": {"index": 2}}), encoding="utf-8")
    save_roi(str(path), [10, 20, 300, 200])
    cfg = load_config(str(path))
    assert cfg["roi"] == [10, 20, 300, 200]
    assert cfg["camera"]["index"] == 2
    assert cfg["tracking"] == DEFAULTS["tracking"]
