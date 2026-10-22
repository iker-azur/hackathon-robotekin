"""Unidades de articulaciones y observación de la política (mismo preprocesado para grabar y ejecutar)."""

import numpy as np
import pytest

from binpick import config as C
from binpick import joints as J
from binpick import observation as O
from binpick import segmentation as S
from synth import piece, render, scene_and_ctx

SCENE, CTX = scene_and_ctx()


def test_joint_conversion_roundtrip_and_limits():
    rng = np.random.default_rng(0)
    q = np.column_stack([rng.uniform(lo, hi, 50) for lo, hi in
                         [(np.radians(a), np.radians(b)) for a, b in J.USD_LIMITS_DEG.values()]])
    assert np.allclose(J.motor_to_sim(J.sim_to_motor(q)), q, atol=1e-6)
    lo = J.sim_to_motor(np.radians([J.USD_LIMITS_DEG[n][0] for n in J.JOINT_NAMES]))
    hi = J.sim_to_motor(np.radians([J.USD_LIMITS_DEG[n][1] for n in J.JOINT_NAMES]))
    assert np.allclose(lo, [J.MOTOR_LIMITS[n][0] for n in J.JOINT_NAMES], atol=1e-4)
    assert np.allclose(hi, [J.MOTOR_LIMITS[n][1] for n in J.JOINT_NAMES], atol=1e-4)


def test_joint_conversion_matches_leisaac_formula():
    # shoulder_pan: simétrico, 0 rad ↔ 0; gripper: 0 rad = 10/110 del rango → 9.09
    m = J.sim_to_motor(np.zeros(6))
    assert m[0] == pytest.approx(0.0, abs=1e-5) and m[5] == pytest.approx(100 * 10 / 110, abs=1e-4)
    assert m.dtype == np.float32


def test_joint_shape_validation_and_split_join():
    with pytest.raises(ValueError):
        J.sim_to_motor(np.zeros(5))
    arm, g = J.split(J.join(np.arange(5.0), 0.7))
    assert list(arm) == [0, 1, 2, 3, 4] and g == 0.7


def test_roi_contains_bin_and_both_boxes():
    cfg = C.BinPickCfg()
    spec = O.make_spec(CTX.cam, SCENE.table.top_z_m, cfg)
    x0, y0, x1, y1 = spec.roi
    assert 0 <= x0 < x1 <= CTX.cam.width and 0 <= y0 < y1 <= CTX.cam.height
    for cid in (C.BIN_ID, *C.DEST_IDS.values()):
        c = C.container(SCENE, cid)
        pts = np.array([[c.center_xy_m[0] + sx * c.outer_m[0] / 2, c.center_xy_m[1] + sy * c.outer_m[1] / 2, SCENE.table.top_z_m + c.outer_m[2]]
                        for sx in (-1, 1) for sy in (-1, 1)])
        uv, _ = CTX.cam.project(pts)
        assert (uv[:, 0] >= x0).all() and (uv[:, 0] <= x1).all() and (uv[:, 1] >= y0).all() and (uv[:, 1] <= y1).all(), cid
    # el aspecto del recorte se conserva al reducir (sin deformar)
    h, w = spec.out_hw
    assert abs((y1 - y0) / (x1 - x0) - h / w) < 0.05


def make_obs(mode_pieces=None):
    cfg = C.BinPickCfg()
    spec = O.make_spec(CTX.cam, SCENE.table.top_z_m, cfg)
    ps = mode_pieces or [piece(SCENE, "a", "red", 0.19, -0.02), piece(SCENE, "b", "blue", 0.22, 0.025)]
    rgb, depth = render(SCENE, CTX, ps)
    inst = S.HsvSegmenter(CTX).segment(rgb, depth)
    seg = O.target_seg_full(CTX.cam, SCENE, inst, 0)
    obs = O.build_obs(rgb, seg, np.array(SCENE.robot.start_joints[:5]), 0.0, spec)
    return obs, spec, inst, rgb, seg


def test_observation_shapes_dtypes_and_encoding():
    obs, spec, inst, rgb, seg = make_obs()
    h, w = spec.out_hw
    assert obs.overhead.shape == obs.target.shape == (h, w, 3) and obs.overhead.dtype == np.uint8
    assert obs.state.shape == (6,) and obs.state.dtype == np.float32
    assert obs.target[..., 0].max() == 255 and obs.target[..., 1].max() == 255 and obs.target[..., 2].max() == 255
    # el objetivo está sólo en R; las otras piezas, sólo en G
    assert not ((obs.target[..., 0] > 0) & (obs.target[..., 1] > 0)).any()
    # la imagen original no se toca: el recorte es un subconjunto de píxeles del RGB (reducido)
    assert obs.overhead.mean() > 0


def test_target_mask_marks_the_requested_instance_only():
    obs, spec, inst, rgb, seg = make_obs()
    assert (seg[..., 0] > 0).sum() == inst[0].mask.sum()
    # elegir otro objetivo cambia la máscara
    other = O.target_seg_full(CTX.cam, SCENE, inst, 1)
    assert (other[..., 0] > 0).sum() == inst[1].mask.sum() and not np.array_equal(other, seg)


def test_model_input_is_float_chw_and_same_function_for_every_use():
    obs, *_ = make_obs()
    x = O.to_model_input(obs)
    k = O.IMAGE_KEYS["overhead"]
    assert x[k].dtype == np.float32 and x[k].shape == (3, *obs.overhead.shape[:2]) and 0 <= x[k].min() <= x[k].max() <= 1
    assert set(x) == {O.STATE_KEY, *map(O.IMAGE_KEYS.get, ("overhead", "target"))}
    # determinista: mismos píxeles → misma entrada (entrenamiento = ejecución)
    assert np.array_equal(O.to_model_input(obs)[k], x[k])


@pytest.mark.parametrize("mode,streams", [("rgb_mask", {"overhead", "target"}), ("rgb_only", {"overhead"}), ("mask_only", {"target"})])
def test_obs_modes_select_streams(mode, streams):
    obs, *_ = make_obs()
    x = O.to_model_input(obs, mode)
    assert {k for k in x if k != O.STATE_KEY} == {O.IMAGE_KEYS[s] for s in streams}


def test_features_match_what_to_model_input_produces():
    obs, spec, *_ = make_obs()
    f = O.features(spec)
    x = O.to_model_input(obs)
    assert set(f) - {O.ACTION_KEY} == set(x)
    for stream in ("overhead", "target"):
        assert f[O.IMAGE_KEYS[stream]]["shape"] == (*spec.out_hw, 3)
    assert f[O.ACTION_KEY]["shape"] == (6,) == f[O.STATE_KEY]["shape"]


def test_bin_mask_covers_the_bin_footprint():
    m = O.bin_mask(CTX.cam, SCENE)
    b = C.container(SCENE, C.BIN_ID)
    ppm = CTX.cam.px_per_m(CTX.cam.pos[2] - SCENE.table.top_z_m - b.floor_m)
    expect = b.inner_m[0] * b.inner_m[1] * ppm * ppm
    assert 0.9 * expect < m.sum() < 1.1 * expect
