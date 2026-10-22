"""Escenas aleatorias: reproducibles, variadas, y su configuración final se mide."""

import math

import numpy as np
import pytest

from binpick import config as C
from binpick import scene_gen as G
from synth import quat_yaw

CFG = C.BinPickCfg()
SCENE = C.build_scene(CFG)
S = C.PIECE_SIZE_M


def test_same_seed_same_scene_different_seed_different_scene():
    a, b, c = (G.sample_scene(CFG, s).as_dict() for s in (7, 7, 8))
    assert a == b and a != c


def test_piece_count_and_pool_limits():
    for seed in range(60):
        spec = G.sample_scene(CFG, seed)
        ids = [d.piece_id for d in spec.drops]
        assert 4 <= len(ids) <= C.MAX_PIECES and len(set(ids)) == len(ids)
        assert all(n <= CFG.cubes_per_color + CFG.cylinders_per_color for n in spec.colors().values())


def test_color_distribution_varies_across_seeds():
    seen = {tuple(sorted(G.sample_scene(CFG, s).colors().items())) for s in range(40)}
    assert len(seen) > 15
    # alguna escena no tiene los tres colores: el sistema debe poder decir «no veo piezas azules»
    assert any(len(G.sample_scene(CFG, s, n_pieces=4).colors()) < 3 for s in range(80))


def test_modes_produce_their_drop_kinds():
    sep = G.sample_scene(CFG, 1, n_pieces=6, mode="separadas")
    assert {d.mode for d in sep.drops} == {"free"}
    near = G.sample_scene(CFG, 1, n_pieces=6, mode="juntas")
    assert [d.mode for d in near.drops].count("touch") == 5
    stacks = sum(sum(d.mode == "stack" for d in G.sample_scene(CFG, s, n_pieces=6, mode="apiladas").drops) for s in range(10))
    assert stacks > 10
    mixed = {d.mode for s in range(30) for d in G.sample_scene(CFG, s, n_pieces=8, mode="mixto").drops}
    assert mixed == {"free", "touch", "stack"}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        G.sample_scene(CFG, 0, mode="caos")


def test_free_drops_are_inside_the_bin_away_from_walls():
    x0, x1, y0, y1 = C.inner_margin_box(SCENE, C.BIN_ID, 0.0)
    for seed in range(30):
        for d in G.sample_scene(CFG, seed).drops:
            if d.mode == "free":
                assert x0 + 0.015 < d.xy[0] < x1 - 0.015 and y0 + 0.015 < d.xy[1] < y1 - 0.015


def test_resolve_touch_and_stack_geometry():
    bx, by = C.container(SCENE, C.BIN_ID).center_xy_m
    ref = {"a": np.array([bx, by, 0.76 + 0.004 + S / 2])}
    touch = G.DropSpec("b", "red", "cube", mode="touch", ref="a", angle=math.pi / 2)
    p, q = G.resolve_drop(touch, ref, SCENE)
    assert abs(np.linalg.norm(p[:2] - ref["a"][:2]) - (S + 0.0008)) < 1e-6
    stack = G.DropSpec("c", "red", "cube", mode="stack", ref="a", angle=math.pi / 2, offset_m=0.008)
    p, _ = G.resolve_drop(stack, ref, SCENE)
    assert p[2] > ref["a"][2] + S and abs(p[1] - by - 0.008) < 1e-6
    assert abs(np.linalg.norm(q) - 1) < 1e-9


def test_resolve_clamps_inside_bin_and_falls_back_to_free_without_ref():
    x0, x1, y0, y1 = C.inner_margin_box(SCENE, C.BIN_ID, 0.0)
    ref = {"a": np.array([x1 - 0.002, y1 - 0.002, 0.78])}
    p, _ = G.resolve_drop(G.DropSpec("b", "red", "cube", mode="touch", ref="a", angle=0.7), ref, SCENE)
    assert x0 < p[0] < x1 and y0 < p[1] < y1
    p, _ = G.resolve_drop(G.DropSpec("b", "red", "cube", mode="stack", ref="ausente"), {}, SCENE)
    assert x0 < p[0] < x1


def test_polygon_gap():
    a = G._square(np.array([0.0, 0.0]), 0.0, 0.0125)
    assert abs(G.polygon_gap(a, G._square(np.array([0.035, 0.0]), 0.0, 0.0125)) - 0.01) < 1e-9
    assert G.polygon_gap(a, G._square(np.array([0.02, 0.0]), 0.0, 0.0125)) < 0
    # esquina contra esquina, giradas 45°
    assert G.polygon_gap(a, G._square(np.array([0.05, 0.05]), math.pi / 4, 0.0125)) > 0.02


def pose(x, y, z, yaw=0.0):
    return np.array([x, y, z]), quat_yaw(yaw)


Z0 = 0.76 + 0.004 + S / 2


def test_analyze_separate_touching_stacked():
    sep = G.analyze_layout({"a": pose(0.18, -0.03, Z0), "b": pose(0.23, 0.03, Z0)})
    assert sep.touching_pairs == 0 and sep.stacked == 0 and sep.isolated == 2 and sep.categories == ["separadas"]
    touch = G.analyze_layout({"a": pose(0.2, 0.0, Z0), "b": pose(0.2 + S + 0.0005, 0.0, Z0)})
    assert touch.touching_pairs == 1 and "tocándose" in touch.categories
    stack = G.analyze_layout({"a": pose(0.2, 0.0, Z0), "b": pose(0.208, 0.004, Z0 + S)})
    assert stack.stacked == 1 and "apiladas" in stack.categories


def test_validate_settled_flags_problems():
    ok = {"a": pose(0.205, 0.0, Z0)}
    assert G.validate_settled(ok, SCENE) == []
    assert any("fuera" in m for m in G.validate_settled({"a": pose(0.4, 0.0, Z0)}, SCENE))
    assert any("alta" in m for m in G.validate_settled({"a": pose(0.205, 0.0, Z0 + 3 * S)}, SCENE))
    assert any("suelo" in m for m in G.validate_settled({"a": pose(0.205, 0.0, 0.76)}, SCENE))
    assert any("asentarse" in m for m in G.validate_settled(ok, SCENE, speeds={"a": 0.5}))


def test_quat_from_rpy_is_unit_and_yaw_roundtrips():
    q = G.quat_from_rpy(0.1, -0.2, 0.7)
    assert abs(np.linalg.norm(q) - 1) < 1e-12
    assert abs(G.yaw_of(G.quat_from_rpy(0.0, 0.0, 0.7)) - 0.7) < 1e-9
    assert G.tilt_of(G.quat_from_rpy(0.3, 0.0, 0.0)) == pytest.approx(0.3, abs=1e-9)
