"""The MuJoCo 3D workspace: physics, rendering, ground truth, and the detector on its frames."""

import json

import pytest

from roboqc.vision.camera import CameraError, open_source
from roboqc.vision.config import load_config
from roboqc.vision.pipeline import VisionPipeline
from roboqc.vision.sim3d import Sim3DSource, load_scene3d

pytest.importorskip("cv2")
pytest.importorskip("mujoco")


def test_open_source_returns_3d_source():
    with open_source("sim3d:demo3d") as src:
        assert isinstance(src, Sim3DSource)
        frame = src.read()
        assert frame.shape == (480, 640, 3)
        assert frame.dtype.name == "uint8"


def test_unknown_3d_scene_is_a_camera_error():
    with pytest.raises(CameraError, match="Escena 3D desconocida"):
        open_source("sim3d:no_existe")


def test_scene_with_late_moving_object_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    scene = load_scene3d("demo3d")
    scene["objects"][0]["enter_frame"] = 10
    scene["objects"][0]["velocity"] = [1.0, 0.0]
    path.write_text(json.dumps(scene), encoding="utf-8")
    with pytest.raises(CameraError, match="debe estar desde el frame 0"):
        load_scene3d(str(path))


def test_truth_at_first_frame_matches_scene():
    src = Sim3DSource("demo3d")
    src.read()
    shapes = {t["id"]: t["shape"] for t in src.truth()}
    assert shapes == {"cubo_rojo_1": "square", "cubo_rojo_2": "square",
                      "cilindro_azul": "circle", "prisma_verde": "rect"}


def test_truth_centers_are_inside_the_frame():
    src = Sim3DSource("demo3d")
    src.read()
    for t in src.truth():
        u, v = t["center"]
        assert 0 <= u < 640 and 0 <= v < 480


def test_resting_objects_stay_on_the_table():
    src = Sim3DSource("demo3d")
    src.read()
    before = {t["id"]: t["center_mm"] for t in src.truth()}
    for _ in range(30):  # one simulated second
        src.read()
    after = {t["id"]: t["center_mm"] for t in src.truth()}
    assert before["cubo_rojo_1"] == pytest.approx(after["cubo_rojo_1"], abs=1.0)
    assert before["cilindro_azul"] == pytest.approx(after["cilindro_azul"], abs=1.0)


def test_enter_frame_is_respected():
    src = Sim3DSource("demo3d")
    for _ in range(45):  # frames 0..44
        src.read()
    assert len(src.truth()) == 4
    src.read()  # frame 45: the yellow cube enters
    assert len(src.truth()) == 5


def test_detector_finds_every_object_on_first_frame():
    cfg = load_config("__no_such_file__.json")
    pipeline = VisionPipeline(cfg)
    src = Sim3DSource("demo3d")
    frame = src.read()
    result = pipeline.process(frame, timestamp=0.0)
    found = sorted(d.color for d in result.detections)
    expected = sorted(t["color"] for t in src.truth())
    assert found == expected
