"""Escenas sintéticas para probar percepción y selección sin simulador.

Un render mínimo con OpenCV de la cámara cenital del demo: mesa gris, contenedor marrón y
piezas de colores saturados con su cara superior y sus laterales más oscuros, más el mapa de
profundidad. No sustituye a Isaac (no hay sombras, ni reflejos, ni ruido de sensor): sirve para
comprobar la lógica de separación, color, posición y visibilidad con la verdad conocida.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from binpick import config as C
from binpick.segmentation import Context, PieceTruth, overhead_camera

RGB = {k: tuple(int(255 * v) for v in rgb) for k, rgb in
       {"red": (0.80, 0.06, 0.05), "green": (0.05, 0.60, 0.12), "blue": (0.05, 0.20, 0.85)}.items()}
TABLE_RGB = (92, 94, 99)
BOX_RGB = (140, 107, 71)


def scene_and_ctx(cfg: C.BinPickCfg | None = None):
    scene = C.build_scene(cfg)
    cam = overhead_camera(scene.cameras[0])
    ctx = Context(cam, scene.table.top_z_m, lambda x, y, z: C.region_of(scene, x, y, z))
    return scene, ctx


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


LEGACY_BIN_CENTER = (0.205, 0.0)       # donde estaba el contenedor cuando se escribieron las coordenadas de las pruebas


def piece(scene, pid: str, color: str, x: float, y: float, yaw: float = 0.0, layer: int = 0,
          shape: str = "cube", dz: float = 0.0, absolute: bool = False) -> PieceTruth:
    """Una pieza de 25 mm en el contenedor, ``layer`` capas por encima del suelo.

    ``x, y`` se leen relativos al centro del contenedor (con el origen histórico (0.205, 0)), de modo que
    mover el contenedor no obliga a reescribir las pruebas; ``absolute=True`` los toma como coordenadas del mundo.
    """
    s = C.PIECE_SIZE_M
    b = C.container(scene, C.BIN_ID)
    if not absolute:
        x, y = x - LEGACY_BIN_CENTER[0] + b.center_xy_m[0], y - LEGACY_BIN_CENTER[1] + b.center_xy_m[1]
    z = scene.table.top_z_m + b.floor_m + s / 2 + layer * s + dz
    return PieceTruth(pid, color, shape, s, s, np.array([x, y, z]), quat_yaw(yaw))


def _rot_yaw(q) -> np.ndarray:
    yaw = 2 * math.atan2(q[3], q[0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def render(scene, ctx: Context, pieces: list[PieceTruth], noise: float = 0.0, seed: int = 0):
    cam = ctx.cam
    h, w = cam.height, cam.width
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = TABLE_RGB
    depth = np.full((h, w), cam.pos[2] - ctx.table_z, np.float32)
    # contenedor de origen (suelo marrón un poco más cerca de la cámara)
    b = C.container(scene, C.BIN_ID)
    x0, x1 = b.center_xy_m[0] - b.inner_m[0] / 2, b.center_xy_m[0] + b.inner_m[0] / 2
    y0, y1 = b.center_xy_m[1] - b.inner_m[1] / 2, b.center_xy_m[1] + b.inner_m[1] / 2
    corners = np.array([[x0, y0, ctx.table_z + b.floor_m], [x1, y0, ctx.table_z + b.floor_m],
                        [x1, y1, ctx.table_z + b.floor_m], [x0, y1, ctx.table_z + b.floor_m]])
    uv, _ = cam.project(corners)
    cv2.fillConvexPoly(rgb, np.round(uv).astype(np.int32), BOX_RGB)
    cv2.fillConvexPoly(depth, np.round(uv).astype(np.int32), float(cam.pos[2] - ctx.table_z - b.floor_m))
    for p in sorted(pieces, key=lambda t: t.pos[2]):
        r = _rot_yaw(p.quat)
        s, hgt = p.size, p.height
        if p.shape == "cylinder":
            th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            loc_all = np.array([[s / 2 * math.cos(a), s / 2 * math.sin(a), sz * hgt / 2] for a in th for sz in (-1, 1)])
            loc_top = np.array([[s / 2 * math.cos(a), s / 2 * math.sin(a), hgt / 2] for a in th])
        else:
            loc_all = np.array([[sx * s / 2, sy * s / 2, sz * hgt / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            loc_top = np.array([[-s / 2, -s / 2, hgt / 2], [s / 2, -s / 2, hgt / 2], [s / 2, s / 2, hgt / 2], [-s / 2, s / 2, hgt / 2]])
        d_top = float(cam.pos[2] - (p.pos[2] + hgt / 2))
        side = tuple(int(0.6 * c) for c in RGB[p.color])
        uv_all, _ = cam.project(loc_all @ r.T + p.pos)
        hull = cv2.convexHull(np.round(uv_all).astype(np.int32))
        cv2.fillConvexPoly(rgb, hull, side)
        cv2.fillConvexPoly(depth, hull, d_top + hgt / 2)
        uv_top, _ = cam.project(loc_top @ r.T + p.pos)
        poly = np.round(uv_top).astype(np.int32)
        cv2.fillConvexPoly(rgb, poly, RGB[p.color])
        cv2.fillConvexPoly(depth, poly, d_top)
    if noise:
        rng = np.random.default_rng(seed)
        rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, noise, rgb.shape), 0, 255).astype(np.uint8)
        depth = depth + rng.normal(0, 0.0004, depth.shape).astype(np.float32)
    return rgb, depth
