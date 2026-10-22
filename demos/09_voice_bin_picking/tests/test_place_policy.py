"""Colocación en cajas y salvaguarda de la política (sin simulador)."""

import json
import math

import numpy as np
import pytest

from binpick import config as C
from binpick import observation as O
from binpick import place as PL
from binpick import policy as POL
from recoverykit.workcell import base_transform

CFG = C.BinPickCfg()
SCENE = C.build_scene(CFG)


# ------------------------------------------------------------------ colocación
def always(_p):
    return True


def test_candidate_points_inside_box_with_margin():
    for dest in C.DESTINATIONS:
        x0, x1, y0, y1 = C.inner_margin_box(SCENE, C.DEST_IDS[dest], PL.EDGE_MARGIN_M - 1e-6)
        pts = PL.candidate_points(SCENE, dest)
        assert pts and all(x0 <= x <= x1 + 1e-9 and y0 <= y <= y1 + 1e-9 for x, y in pts)


def test_choose_slot_keeps_distance_and_prefers_free_space():
    first = PL.choose_slot(SCENE, "izquierda", [], always)
    assert first is not None
    second = PL.choose_slot(SCENE, "izquierda", [first], always)
    assert second is not None and math.hypot(second[0] - first[0], second[1] - first[1]) >= PL.MIN_GAP_M


def test_box_fills_up_and_then_reports_full():
    occupied = []
    while True:
        p = PL.choose_slot(SCENE, "derecha", occupied, always)
        if p is None:
            break
        occupied.append(p)
        assert len(occupied) < 12
    assert 3 <= len(occupied) <= 6
    assert PL.capacity(SCENE, "derecha", always) == len(occupied)


def test_fingers_near_a_wall_only_allow_orientations_that_point_inboard():
    c = C.container(SCENE, C.DEST_IDS["izquierda"])
    x0, x1, y0, y1 = C.inner_margin_box(SCENE, C.DEST_IDS["izquierda"], 0.0)
    near_wall = (x0 + 0.016, c.center_xy_m[1])               # a 3,5 mm de la pared -x
    middle = tuple(c.center_xy_m)
    assert len(PL.yaws_free(SCENE, "izquierda", middle, [])) == 4
    ok = PL.yaws_free(SCENE, "izquierda", near_wall, [])
    assert 0 < len(ok) < 4
    # un vecino pegado a un lado quita orientaciones
    assert len(PL.yaws_free(SCENE, "izquierda", middle, [(middle[0] + 0.03, middle[1])])) < 4


def test_unreachable_points_are_never_chosen():
    far = lambda p: p[0] > SCENE.containers[1].center_xy_m[0]       # sólo la mitad lejana de la caja
    for _ in range(3):
        p = PL.choose_slot(SCENE, "izquierda", [], far)
        assert p is None or far(p)
    assert PL.capacity(SCENE, "izquierda", lambda p: False) == 0


def test_real_kinematics_capacity_of_both_boxes():
    placer = PL.Placer(SCENE, base_transform(SCENE), np.array(SCENE.robot.start_joints[:5]))
    caps = {d: placer.capacity(d) for d in C.DESTINATIONS}
    assert all(c >= 2 for c in caps.values()), caps               # lo que cabe de verdad con la cinemática del robot
    q0 = np.array(SCENE.robot.start_joints[:5])
    occupied = []
    for dest in C.DESTINATIONS:
        occupied = []
        for _ in range(caps[dest]):
            p = placer.choose(dest, occupied)
            assert p is not None
            plan = placer.plan(dest, p, q0, occupied)
            assert not isinstance(plan, str), (dest, p, plan)      # cada hueco elegido tiene un plan completo
            occupied.append(p)
        assert placer.choose(dest, occupied) is None


# --------------------------------------------------------------- salvaguarda
def test_limiter_clips_joint_limits_and_speed():
    lim = POL.ActionLimiter()
    q0 = np.zeros(6)
    lim.reset(q0)
    out = lim(np.array([3.0, 0, 0, 0, 0, 0]))               # pide un salto enorme
    assert 0 < out[0] <= lim.max_arm_step + 1e-9
    assert lim.clipped_speed == 1 and lim.clipped_limits == 1
    for _ in range(100):
        out = lim(np.array([3.0, 0, 0, 0, 0, 0]))
    assert out[0] <= lim.hi[0] + 1e-9                         # nunca pasa del límite de la articulación


def test_limiter_gripper_step_and_requires_reset():
    lim = POL.ActionLimiter()
    with pytest.raises(RuntimeError):
        lim(np.zeros(6))
    lim.reset(np.zeros(6))
    out = lim(np.array([0, 0, 0, 0, 0, 1.5]))
    assert out[5] == pytest.approx(lim.max_gripper_step)


def test_limiter_rejects_motion_into_the_table_and_holds():
    lim = POL.ActionLimiter(clearance_fn=lambda q: -0.01 if q[1] > 0.05 else 0.05)
    lim.reset(np.zeros(6))
    out = lim(np.array([0, 0.5, 0, 0, 0, 0]))
    assert np.allclose(out, 0) and lim.rejected_table == 1
    assert lim.stats()["rechazos_por_mesa"] == 1


def test_info_roundtrip_and_compat(tmp_path):
    spec = O.ObsSpec((10, 20, 200, 400), (288, 160))
    folder = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    folder.mkdir(parents=True)
    (tmp_path / "run" / POL.INFO_FILE).write_text(json.dumps(
        {"obs_mode": "rgb_mask", "roi": list(spec.roi), "out_hw": list(spec.out_hw), "fps": 30, "segmenter": "hsv",
         "joint_units": "range_m100_100"}), encoding="utf-8")
    info = POL.load_info(folder)                              # lo encuentra subiendo carpetas
    assert info["obs_mode"] == "rgb_mask"
    assert POL.check_compat(info, spec, "hsv") == ([], [])
    errors, warnings = POL.check_compat(info, O.ObsSpec((0, 0, 1, 1), (288, 160)), "yolo:x.pt")
    assert errors and "recorte" in errors[0] and warnings and "hsv" in warnings[0]
    errors, _ = POL.check_compat({**info, "fps": 60}, spec, "hsv")
    assert any("Frecuencia" in e for e in errors)
    assert POL.check_compat({}, spec, "hsv")[1]               # sin info: sólo aviso


def test_scripted_policy_is_labelled_not_learned():
    p, = [POL.load("guion")]
    assert p.kind == "guion" and "no aprendido" in p.name
    with pytest.raises(NotImplementedError):
        p.act(None)


def test_callable_policy_for_runner_tests():
    p = POL.CallablePolicy(lambda obs: np.arange(6) * 0.1, "demo")
    assert p.kind == "act" and np.allclose(p.act(None), np.arange(6) * 0.1)


def test_act_policy_refuses_a_missing_checkpoint(tmp_path):
    with pytest.raises(Exception):
        POL.ACTPickPolicy(tmp_path / "no_existe")
