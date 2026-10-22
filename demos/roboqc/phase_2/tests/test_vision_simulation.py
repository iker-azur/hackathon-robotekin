"""The simulated workspace: rendering, ground truth, and the full pipeline on simulated frames."""

import numpy as np
import pytest

from roboqc.vision.camera import CameraError, open_source
from roboqc.vision.config import load_config
from roboqc.vision.pipeline import VisionPipeline
from roboqc.vision.simulation import SimulatedSource, ground_truth, load_scene, render_scene

cv2 = pytest.importorskip("cv2")


def test_open_source_returns_simulated_source():
    with open_source("sim:static") as src:
        assert isinstance(src, SimulatedSource)
        frame = src.read()
        assert frame.shape == (480, 640, 3)


def test_unknown_scene_is_a_camera_error():
    with pytest.raises(CameraError, match="Escena desconocida"):
        open_source("sim:no_existe")


def test_scene_with_unknown_color_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"width": 64, "height": 48, "background": [40,40,40],'
        ' "objects": [{"id": "x", "shape": "square", "color": "violeta", "center": [10,10], "size": [8,8]}]}',
        encoding="utf-8",
    )
    with pytest.raises(CameraError, match="color no soportado"):
        load_scene(str(path))


def test_ground_truth_respects_enter_frame():
    scene = load_scene("demo")
    assert len(ground_truth(scene, 0)) == 4  # cubo_amarillo enters at frame 45
    assert len(ground_truth(scene, 50)) == 5


def test_static_scene_renders_its_colors_without_noise():
    scene = load_scene("static")
    frame = render_scene(scene, 0)
    center_red = frame[140, 150]  # BGR of the red cube
    assert tuple(int(v) for v in center_red) == (0, 0, 255)
    assert tuple(int(v) for v in frame[0, 0]) == (40, 40, 40)


def test_detector_finds_every_known_object_on_static_scene():
    cfg = load_config("__no_such_file__.json")
    pipeline = VisionPipeline(cfg)
    scene = load_scene("static")
    result = pipeline.process(render_scene(scene, 0), timestamp=0.0)
    found = sorted(d.color for d in result.detections)
    expected = sorted(t["color"] for t in ground_truth(scene, 0))
    assert found == expected


def test_pipeline_tracks_simulated_objects_across_frames():
    cfg = load_config("__no_such_file__.json")
    pipeline = VisionPipeline(cfg)
    scene = load_scene("demo")
    for i in range(50):
        result = pipeline.process(render_scene(scene, i), timestamp=i * 0.066)
    # At frame 49 all five simulated objects are visible (the yellow cube entered at 45).
    visible = result.snapshot.visible()
    assert len(visible) == len(ground_truth(scene, 49)) == 5
    assert len({o.object_id for o in visible}) == 5
    assert all(o.quality == "GOOD" for o in visible)


def test_source_loops_after_loop_frames():
    src = SimulatedSource("static")
    src._loop = 3
    for _ in range(4):
        src.read()
    assert src.frame_index == 1
