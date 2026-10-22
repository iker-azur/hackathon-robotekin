"""Colocación reproducible de las piezas sobre la mesa (sin Isaac).

Muestreo por rechazo dentro de la zona de trabajo: cada pieza queda a una
distancia mínima de las demás y fuera de la huella de las cajas. Las piezas con
posición fija en la configuración se respetan tal cual.
"""

from __future__ import annotations

import math
import random

from .config import SceneCfg

BOX_CLEARANCE_M = 0.03


def _in_box_zone(cfg: SceneCfg, x: float, y: float) -> bool:
    for c in cfg.containers:
        hx = c.outer_m[0] / 2 + BOX_CLEARANCE_M
        hy = c.outer_m[1] / 2 + BOX_CLEARANCE_M
        if abs(x - c.center_xy_m[0]) < hx and abs(y - c.center_xy_m[1]) < hy:
            return True
    return False


def sample(cfg: SceneCfg, seed: int | None = None, attempts: int = 300,
           restarts: int = 200) -> dict[str, tuple[float, float, float]]:
    """Devuelve ``{id: (x, y, yaw_rad)}`` para todas las piezas.

    La colocación es voraz; si una pieza no cabe por cómo han quedado las
    anteriores, se vuelve a empezar (mismo generador, así que es reproducible).
    """
    rng = random.Random(cfg.seed if seed is None else seed)
    for _ in range(restarts):
        placed = _attempt(cfg, rng, attempts)
        if placed is not None:
            return placed
    raise RuntimeError("no caben todas las piezas en la zona de trabajo")


def _yaw(p, rng: random.Random) -> float:
    return math.radians(p.yaw_deg) if p.yaw_deg is not None else rng.uniform(-math.pi / 4, math.pi / 4)


def _attempt(cfg: SceneCfg, rng: random.Random, attempts: int) -> dict[str, tuple[float, float, float]] | None:
    area = cfg.work_area
    placed: dict[str, tuple[float, float, float]] = {}
    for p in cfg.pieces:
        if p.pos_xy_m is not None:
            placed[p.id] = (p.pos_xy_m[0], p.pos_xy_m[1], _yaw(p, rng))
    bx, by = cfg.robot.base_xy_m
    for p in cfg.pieces:
        if p.id in placed:
            continue
        for _ in range(attempts):
            # uniforme en área dentro del sector de corona
            r = math.sqrt(rng.uniform(area.r_m[0] ** 2, area.r_m[1] ** 2))
            a = math.radians(rng.uniform(*area.angle_deg))
            x, y = bx + r * math.cos(a), by + r * math.sin(a)
            if _in_box_zone(cfg, x, y):
                continue
            if all(math.hypot(x - q[0], y - q[1]) >= area.min_gap_m for q in placed.values()):
                placed[p.id] = (x, y, _yaw(p, rng))
                break
        else:
            return None
    return placed
