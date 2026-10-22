"""Segmentación: instancias, color, posición, separación de piezas pegadas y apiladas, YOLO simulado."""

import math

import numpy as np
import pytest

from binpick import segmentation as S
from binpick.config import COLORS
from synth import piece, render, scene_and_ctx


@pytest.fixture(scope="module")
def world():
    return scene_and_ctx()


def segment(world, pieces, **kw):
    scene, ctx = world
    rgb, depth = render(scene, ctx, pieces, **kw)
    return S.HsvSegmenter(ctx).segment(rgb, depth), rgb, depth


def test_single_cube_position_color_shape(world):
    scene, ctx = world
    t = piece(scene, "r1", "red", 0.205, 0.01, yaw=math.radians(20))
    inst, *_ = segment(world, [t])
    assert len(inst) == 1
    i = inst[0]
    assert i.color == "red" and i.shape == "cube" and i.region == "origen"
    assert np.linalg.norm(i.pos[:2] - t.pos[:2]) < 0.004
    assert abs(i.pos[2] - t.pos[2]) < 0.004
    assert i.visibility > 0.9
    assert abs(math.degrees(i.yaw) - 20) < 6


def test_cylinder_is_not_a_cube(world):
    scene, _ = world
    inst, *_ = segment(world, [piece(scene, "g1", "green", 0.2, -0.015, shape="cylinder")])
    assert [i.shape for i in inst] == ["cylinder"] and inst[0].color == "green"


def test_three_colors_separated(world):
    scene, _ = world
    ps = [piece(scene, "r", "red", 0.18, -0.03), piece(scene, "g", "green", 0.205, 0.0),
          piece(scene, "b", "blue", 0.23, 0.03)]
    inst, *_ = segment(world, ps)
    assert sorted(i.color for i in inst) == ["blue", "green", "red"]


def test_touching_same_color_cubes_are_split(world):
    scene, _ = world
    s = 0.025
    ps = [piece(scene, "a", "blue", 0.195, 0.0), piece(scene, "b", "blue", 0.195 + s + 0.0004, 0.0)]
    inst, *_ = segment(world, ps)
    assert len(inst) == 2
    xs = sorted(i.pos[0] for i in inst)
    assert abs((xs[1] - xs[0]) - s) < 0.006


@pytest.mark.parametrize("angle,stagger", [(15, 6), (20, 4), (5, 5)])
def test_staggered_touching_squares_without_a_seam_are_split_in_two(angle, stagger):
    """Caso medido en renders de Isaac: dos cubos pegados y escalonados no tienen costura de color ni dos picos de
    distancia, y ni el reparto alineado ni el watershed los partían (quedaba una mancha de 2 piezas)."""
    import cv2

    a = math.radians(angle)
    m = np.zeros((120, 120), np.uint8)
    for cx, cy in ((40, 40), (40 + 26 * math.cos(a) + stagger * math.sin(a), 40 + 26 * math.sin(a) - stagger * math.cos(a))):
        cv2.fillPoly(m, [cv2.boxPoints(((float(cx), float(cy)), (26.0, 26.0), angle)).astype(np.int32)], 1)
    parts = S.split_blob(m.astype(bool), np.full((120, 120), 1.0, np.float32), 26 * 26)
    assert len(parts) == 2
    assert all(0.8 * 676 < p.sum() < 1.2 * 676 for p in parts)


def test_stacked_cube_is_a_second_instance_at_greater_height(world):
    scene, _ = world
    ps = [piece(scene, "low", "red", 0.2, 0.0), piece(scene, "up", "red", 0.2 + 0.012, 0.004, layer=1)]
    inst, *_ = segment(world, ps)
    assert len(inst) == 2
    tops = sorted(i.top_z for i in inst)
    assert tops[1] - tops[0] > 0.015


def test_partially_covered_piece_has_lower_visibility(world):
    scene, _ = world
    ps = [piece(scene, "low", "red", 0.2, 0.0), piece(scene, "up", "green", 0.2 + 0.008, 0.004, layer=1)]
    inst, *_ = segment(world, ps)
    red = [i for i in inst if i.color == "red"][0]
    green = [i for i in inst if i.color == "green"][0]
    assert red.visibility < 0.6 < green.visibility        # tapa ~57 %: visibilidad ~0.43


def test_gray_table_and_brown_bin_are_not_pieces(world):
    inst, *_ = segment(world, [])
    assert inst == []


def test_color_survives_sensor_noise(world):
    scene, _ = world
    ps = [piece(scene, "r", "red", 0.18, -0.03), piece(scene, "b", "blue", 0.23, 0.03)]
    inst, *_ = segment(world, ps, noise=6.0, seed=3)
    assert sorted(i.color for i in inst) == ["blue", "red"]


def test_classify_color_rejects_unsaturated(world):
    rgb = np.full((40, 40, 3), (140, 107, 71), np.uint8)
    color, conf = S.A.classify_color(rgb, np.ones((40, 40), bool))
    assert color is None


def test_ground_truth_painter_order_and_silhouette(world):
    scene, ctx = world
    lo = piece(scene, "lo", "red", 0.2, 0.0)
    hi = piece(scene, "hi", "blue", 0.2 + 0.01, 0.0, layer=1)
    labels, sils = S.ground_truth(ctx.cam, [lo, hi])
    assert sils["lo"].sum() > 0 and sils["hi"].sum() > 0
    assert (labels == 2).sum() == sils["hi"].sum()          # la alta se ve entera
    assert (labels == 1).sum() < sils["lo"].sum()           # la baja queda recortada


def test_gt_instances_match_hsv_instances(world):
    scene, ctx = world
    ps = [piece(scene, "r", "red", 0.18, -0.03), piece(scene, "b", "blue", 0.23, 0.03)]
    rgb, depth = render(scene, ctx, ps)
    gt = S.gt_instances(rgb, depth, ctx, ps)
    hsv = S.HsvSegmenter(ctx).segment(rgb, depth)
    assert len(gt) == len(hsv) == 2
    for g in gt:
        best = max(S.mask_iou(g.mask, h.mask) for h in hsv if h.color == g.color)
        assert best > 0.8


def test_yolo_wrapper_with_injected_predictor_resolves_overlaps(world):
    scene, ctx = world
    ps = [piece(scene, "r", "red", 0.18, -0.03), piece(scene, "b", "blue", 0.23, 0.03)]
    rgb, depth = render(scene, ctx, ps)
    ref = S.gt_instances(rgb, depth, ctx, ps)
    big = np.zeros_like(ref[0].mask)
    big[:] = ref[0].mask | ref[1].mask                      # una máscara que cubre las dos piezas
    fake = lambda img: [(ref[0].mask, 0.9), (big, 0.5), (ref[1].mask, 0.8)]
    inst = S.YoloSegmenter(ctx, predict_fn=fake).segment(rgb, depth)
    assert sorted(i.color for i in inst) == ["blue", "red"]  # el color lo mide la máscara, no la red
    assert all(i.source == "yolo" for i in inst)


def test_make_segmenter_rejects_unknown(world):
    with pytest.raises(ValueError):
        S.make_segmenter("otra", world[1])
    assert isinstance(S.make_segmenter("hsv", world[1]), S.HsvSegmenter)


def test_pool_colors_exist():
    assert set(COLORS) == {"red", "green", "blue"}
