"""Percepción sobre imágenes sintéticas con geometría conocida (sin Isaac)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recoverykit import perception as P  # noqa: E402

TABLE_Z = 0.76
CAM_Z = 1.30
W, H = 960, 720
F = 824.6

# cámara mirando hacia abajo; marco óptico: x der. = +x mundo, y abajo = −y mundo, z delante = −z mundo
ROT = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
CAM = P.Camera(np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]), ROT, np.array([0.15, 0.0, CAM_Z]), W, H)
EXP = P.Expectation(table_z=TABLE_Z)
RGB = {"red": (205, 15, 12), "blue": (12, 50, 215), "green": (12, 150, 30), "yellow": (230, 190, 12)}
BOX_RGB = (140, 107, 71)


def scene(pieces, box_center=None):
    """pieces: (color, shape, x, y, yaw, size, z_base)."""
    rgb = np.full((H, W, 3), 95, np.uint8)
    depth = np.full((H, W), CAM_Z - TABLE_Z, np.float32)
    if box_center is not None:
        cx, cy = box_center
        corners = np.array([[cx + sx * 0.055, cy + sy * 0.055, TABLE_Z + 0.045] for sx, sy in
                            ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        uv, _ = CAM.project(corners)
        cv2.fillPoly(rgb, [uv.astype(np.int32)], BOX_RGB)
        inner = np.array([[cx + sx * 0.051, cy + sy * 0.051, TABLE_Z + 0.004] for sx, sy in
                          ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        uvi, _ = CAM.project(inner)
        cv2.fillPoly(depth, [uvi.astype(np.int32)], float(CAM_Z - TABLE_Z - 0.004))
        cv2.fillPoly(rgb, [uvi.astype(np.int32)], (120, 92, 60))
    for color, shape, x, y, yaw, size, z_base in pieces:
        z_top = z_base + size
        if shape == "cube":
            c, s = math.cos(yaw), math.sin(yaw)
            pts = np.array([[x + c * dx - s * dy, y + s * dx + c * dy, z_top] for dx, dy in
                            ((-size / 2, -size / 2), (size / 2, -size / 2), (size / 2, size / 2), (-size / 2, size / 2))])
            uv, _ = CAM.project(pts)
            cv2.fillPoly(rgb, [uv.astype(np.int32)], RGB[color])
            cv2.fillPoly(depth, [uv.astype(np.int32)], float(CAM_Z - z_top))
        else:
            uv, _ = CAM.project(np.array([[x, y, z_top]]))
            r = int(round(size / 2 * F / (CAM_Z - z_top)))
            cv2.circle(rgb, tuple(uv[0].astype(int)), r, RGB[color], -1)
            cv2.circle(depth, tuple(uv[0].astype(int)), r, float(CAM_Z - z_top), -1)
    return rgb, depth


def test_backproject_roundtrip():
    p = np.array([0.2, -0.05, TABLE_Z + 0.025])
    uv, z = CAM.project(p[None])
    q = CAM.backproject(uv[0, 0], uv[0, 1], z[0])
    assert np.allclose(p, q, atol=1e-9)


def test_detects_positions_shapes_and_yaw():
    pieces = [("red", "cube", 0.17, -0.07, math.radians(20), 0.025, TABLE_Z),
              ("blue", "cube", 0.20, -0.14, math.radians(-30), 0.025, TABLE_Z),
              ("green", "cylinder", 0.16, 0.0, 0.0, 0.025, TABLE_Z),
              ("yellow", "cube", 0.12, -0.13, 0.0, 0.025, TABLE_Z)]
    rgb, depth = scene(pieces)
    obs = P.detect(rgb, depth, CAM, EXP)
    assert len(obs.pieces) == 4
    for color, shape, x, y, yaw, size, _ in pieces:
        d = next(d for d in obs.pieces if d.color == color)
        assert d.shape == shape, (color, d.shape, d.fill)
        assert math.hypot(d.pos[0] - x, d.pos[1] - y) < 0.002
        assert abs(d.pos[2] - (TABLE_Z + size / 2)) < 0.002
        assert abs(d.height_m - size) < 0.002
        if shape == "cube":
            err = (d.yaw - yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
            assert abs(err) < math.radians(3), (math.degrees(d.yaw), math.degrees(yaw))
        assert not d.in_box


def test_box_and_contents():
    box = (0.141, 0.127)
    pieces = [("red", "cube", 0.12, 0.11, 0.0, 0.025, TABLE_Z + 0.004),
              ("blue", "cube", 0.18, -0.05, 0.0, 0.025, TABLE_Z)]
    rgb, depth = scene(pieces, box_center=box)
    obs = P.detect(rgb, depth, CAM, EXP)
    assert len(obs.boxes) == 1
    assert math.hypot(obs.boxes[0].center[0] - box[0], obs.boxes[0].center[1] - box[1]) < 0.004
    red = next(d for d in obs.pieces if d.color == "red")
    blue = next(d for d in obs.pieces if d.color == "blue")
    assert red.in_box and red.box_id == "caja"
    assert not blue.in_box
    assert abs(red.height_m - 0.025) < 0.003


def test_robot_mask_hides_pieces():
    rgb, depth = scene([("red", "cube", 0.17, -0.07, 0.0, 0.025, TABLE_Z)])
    pts = np.array([[0.17 + dx, -0.07 + dy, TABLE_Z + 0.1] for dx in np.linspace(-0.03, 0.03, 30)
                    for dy in np.linspace(-0.03, 0.03, 30)])
    mask = P.robot_pixels(CAM, pts)
    obs = P.detect(rgb, depth, CAM, EXP, robot_px=mask)
    assert obs.pieces == []
    assert obs.occluded_px > 0


def test_touching_pieces_flagged_as_merged():
    rgb, depth = scene([("blue", "cube", 0.17, -0.07, 0.0, 0.025, TABLE_Z),
                        ("blue", "cube", 0.195, -0.07, 0.0, 0.025, TABLE_Z)])
    obs = P.detect(rgb, depth, CAM, EXP)
    assert len(obs.pieces) == 1
    assert obs.pieces[0].merged == 2
