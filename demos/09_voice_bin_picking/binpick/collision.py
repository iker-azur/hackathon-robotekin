"""¿Choca el brazo con las paredes del contenedor o de las cajas en un trayecto? (numpy, sin Isaac)

``skills.min_clearance`` sólo mira la mesa. Aquí se proyectan los colisionadores del brazo (cinemática
directa del demo 08) sobre las cajas macizas de las paredes, a lo largo del trayecto en el espacio
articular que recorre el robot de verdad (interpolación lineal de ``Controller.move_joints``) y de las
rectas de bajada y subida. Es una comprobación de vértices con margen, no un motor de colisiones:
atrapa un brazo que cruza una pared, no un roce de milímetros.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from recoverykit import kinematics as K  # noqa: E402

LINKS = ("upper_arm", "lower_arm", "wrist", "gripper", "jaw")
MARGIN_M = 0.002


def wall_boxes(scene, margin: float = MARGIN_M) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Paredes (AABB) de todos los contenedores, con la misma geometría con la que las construye ``workcell.build``."""
    top = scene.table.top_z_m
    out = []
    for c in scene.containers:
        cx, cy = c.center_xy_m
        ox, oy, oz = c.outer_m
        w, f = c.wall_m, c.floor_m
        h = oz - f
        z0, z1 = top + f, top + f + h
        walls = {
            "px": ((cx + ox / 2 - w, cy - oy / 2, z0), (cx + ox / 2, cy + oy / 2, z1)),
            "nx": ((cx - ox / 2, cy - oy / 2, z0), (cx - ox / 2 + w, cy + oy / 2, z1)),
            "py": ((cx - ox / 2 + w, cy + oy / 2 - w, z0), (cx + ox / 2 - w, cy + oy / 2, z1)),
            "ny": ((cx - ox / 2 + w, cy - oy / 2, z0), (cx + ox / 2 - w, cy - oy / 2 + w, z1)),
        }
        for k, (lo, hi) in walls.items():
            out.append((f"{c.id}/{k}", np.array(lo) - margin, np.array(hi) + margin))
    return out


def arm_points(base: np.ndarray, q5: np.ndarray, stride: int = 2) -> dict[str, np.ndarray]:
    poses = K.fk(dict(zip(K.ARM, q5)), base)
    pts = {}
    for link in LINKS:
        v = K.colliders()[link][::stride]
        t = poses[link]
        pts[link] = v @ t[:3, :3].T + t[:3, 3]
    return pts


def hits(points: dict[str, np.ndarray], boxes) -> tuple[str, str] | None:
    """(eslabón, pared) del primer vértice dentro de una pared, o ``None``."""
    for link, p in points.items():
        for name, lo, hi in boxes:
            if ((p >= lo) & (p <= hi)).all(axis=1).any():
                return link, name
    return None


def joint_path(q0: np.ndarray, q1: np.ndarray, n: int = 30) -> list[np.ndarray]:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return [q0 + (q1 - q0) * (i / n) for i in range(n + 1)]


def path_hit(base: np.ndarray, scene, qs: list[np.ndarray], boxes=None) -> str | None:
    """Descripción (en español) del primer choque a lo largo de ``qs``, o ``None`` si el trayecto está libre."""
    boxes = boxes if boxes is not None else wall_boxes(scene)
    for i, q in enumerate(qs):
        h = hits(arm_points(base, q), boxes)
        if h:
            return f"el {h[0]} chocaría con la pared {h[1]} (a {100 * i / max(len(qs) - 1, 1):.0f} % del trayecto)"
    return None


def check_pick(base: np.ndarray, scene, q_from: np.ndarray, plan) -> str | None:
    """Trayecto completo de un agarre: de ``q_from`` a la aproximación, la bajada y la subida."""
    boxes = wall_boxes(scene)
    for name, qs in (("la aproximación", joint_path(q_from, plan.q_pre)), ("la bajada", plan.descent),
                     ("la subida", plan.lift)):
        problem = path_hit(base, scene, [np.asarray(q) for q in qs], boxes)
        if problem:
            return f"en {name}: {problem}"
    return None
