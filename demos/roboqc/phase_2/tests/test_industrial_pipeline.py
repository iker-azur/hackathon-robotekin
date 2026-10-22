import dataclasses
import json
import time

import numpy as np
import pytest

from roboqc.industrial.voice import understand
from roboqc.industrial.vision_inference import mask_geometry, classify_mask_color
from roboqc.vision.calibration import WorkspaceCalibration
from roboqc.vision.config import load_config
from roboqc.vision.detector import Detection
from roboqc.vision.pipeline import VisionPipeline
from roboqc.vision.sorting import SortingCoordinator
from roboqc.vision.world_state import ObjectRecord, WorldSnapshot


def _object(**changes):
    values = dict(object_id="OBJ_001",color="red",object_type="bolt",shape="industrial",
        position_px=(20,20),bounding_box=(10,10,20,20),area_px=400,orientation_deg=0,
        quality="UNCERTAIN",quality_reasons=(),detection_status="VISIBLE",motion="STATIONARY",
        confidence=.9,first_seen=0,last_seen=1,world_position=None,selected=False,
        industrial_quality="unknown",grasp_candidates=({"position_px":[20,20],"orientation_deg":0,"validated":False},))
    values.update(changes)
    return ObjectRecord(**values)


def test_hole_is_not_a_grasp_and_symmetric_orientation_is_unknown():
    mask = np.zeros((80,80),np.uint8)
    mask[10:70,10:70] = 1
    mask[25:55,25:55] = 0
    geometry = mask_geometry(mask,symmetric=True)
    assert geometry["orientation_deg"] is None
    for grasp in geometry["grasp_candidates"]:
        x,y = grasp["position_px"]
        assert mask[y,x] == 1 and not grasp["validated"]


def test_color_uses_mask_interior_not_background():
    frame = np.zeros((60,60,3),np.uint8)
    frame[:] = [255,0,0]
    frame[10:40,10:40] = [0,0,255]
    mask = np.zeros((60,60),np.uint8); mask[10:40,10:40] = 1
    assert classify_mask_color(frame,mask)[0] == "red"
    assert classify_mask_color(np.full_like(frame,120),mask)[0] is None


def test_calibration_rejects_outside_degenerate_and_nonfinite():
    pixels = [[0,0],[100,0],[100,100],[0,100]]
    cal = WorkspaceCalibration.from_points(pixels,pixels,robot_transform=[[1,0,10],[0,1,20],[0,0,1]])
    assert cal.describe(10,10) == {"frame":"robot","units":"mm","xy":[20.,30.]}
    assert cal.describe(101,10) is None
    assert cal.orientation_to_world(10,10,0) == pytest.approx(0)
    with pytest.raises(ValueError):
        WorkspaceCalibration(np.zeros((3,3)),"robot","mm")
    with pytest.raises(ValueError):
        cal.pixel_to_world(float("nan"),0)


def test_unknown_quality_does_not_use_geometric_good_heuristic():
    cfg = load_config("__missing__")
    cfg["tracking"]["min_hits"] = 1
    pipeline = VisionPipeline(cfg)
    det = Detection("red",.99,"industrial","nut",.99,(10,10,20,20),(20,20),400,None,.99,8,False,industrial_quality="unknown")
    pipeline.world.update([det],1)
    record = pipeline.world.snapshot().objects[0]
    assert record.quality == "UNCERTAIN"
    assert record.industrial_quality == "unknown"


def test_plan_pause_revalidation_unknown_quality_and_simulated_events():
    pixels = [[0,0],[100,0],[100,100],[0,100]]
    cal = WorkspaceCalibration.from_points(pixels,pixels,frame="robot")
    coordinator = SortingCoordinator({"source_region":[0,0,50,50],"destinations":{"blue":[60,60,10,10]}},cal)
    coordinator.accept(understand("Pon las rojas en la azul"))
    snapshot = WorldSnapshot(time.monotonic(),(_object(),),())
    assert coordinator.build(snapshot)[0]["status"] == "ready_for_simulation"
    coordinator.accept(understand("Deten la clasificacion"))
    assert coordinator.simulate_one(snapshot)["status"] == "blocked"
    coordinator.accept(understand("Continua clasificando"))
    response = coordinator.simulate_one(snapshot)
    assert response["status"] == "simulated"
    assert [e["event"] for e in response["events"]] == ["action_requested","simulated_result","simulated_verification"]
    assert all(e["physical_success"] is None for e in response["events"])
    assert coordinator.simulate_one(snapshot)["status"] == "blocked"
    coordinator.accept(understand("Pon las buenas en la azul"))
    assert coordinator.build(snapshot)[0]["reason"] == "unknown_category_or_no_rule"


def test_no_coordinates_without_calibration_and_mask_json_roundtrip():
    coordinator = SortingCoordinator({"source_region":[0,0,50,50],"destinations":{"blue":[60,60,10,10]}})
    coordinator.accept(understand("Pon las rojas en la azul"))
    obj = _object(mask=np.array([[0,1],[1,0]],np.uint8))
    assert coordinator.build(WorldSnapshot(time.monotonic(),(obj,),()))[0]["reason"] == "robot_frame_calibration_required"
    value = json.loads(json.dumps(obj.to_dict()))
    assert value["mask_rle"] == {"size":[2,2],"order":"C","counts":[1,2,1]}
