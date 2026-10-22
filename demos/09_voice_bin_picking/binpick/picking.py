"""Planes de agarre de una pieza dentro del contenedor (numpy, sin Isaac).

``recoverykit.skills.plan_pick`` asume una pieza apoyada en la mesa y fija la
subida en 6,5 cm para pasar el borde de una caja de 4,5 cm. Dentro del contenedor
hay piezas apiladas, y con la pinza vertical el SO-101 no llega 6,5 cm más arriba
de una pieza de la segunda capa (``tools/check_layout_reach.py``: 0 de 42 puntos
con la subida del demo 08). Aquí la subida depende de la altura a la que está la
pieza: lo justo para que su base pase la pared del contenedor.

Todo lo demás (giro de la pinza, IK vertical, bajada recta, comprobación contra
la mesa) es el código validado de ``recoverykit.skills``.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C  # noqa: F401  (sys.path de recoverykit)
from recoverykit import kinematics as K  # noqa: E402
from recoverykit import skills  # noqa: E402
from recoverykit.control import line_path  # noqa: E402

# holgura de apertura de la pinza sobre la anchura de la pieza. El demo 08 validó 16 mm (30/30); con 12 mm caben
# los dedos entre piezas más juntas. Es 4 mm menos de margen de puntería: NO está medido en simulación todavía.
OPEN_MARGIN_M = 0.012
WALL_CLEARANCE_M = 0.016        # la base de la pieza pasa la pared del contenedor con esta holgura
MIN_LIFT_M = 0.025
MAX_LIFT_M = skills.LIFT_M      # lo que ya validó el demo 08 en la capa del suelo


def lift_height(center_z: float, floor_z: float, piece_h: float, wall_top_z: float) -> float:
    """Cuánto hay que subir el centro de la pieza para que su base pase la pared."""
    bottom = center_z - piece_h / 2
    need = wall_top_z + WALL_CLEARANCE_M - bottom
    return float(min(max(need, MIN_LIFT_M), MAX_LIFT_M))


def plan_pick_ex(base: np.ndarray, table_z: float, target: skills.Target, center_z: float, wall_top_z: float,
                 q_seed: np.ndarray, grasp_lower_m: float = 0.0, yaw_offset: float = 0.0,
                 floor_z: float | None = None, yaw_choices: list[float] | None = None) -> skills.PickPlan | str:
    """Plan de agarre vertical de una pieza cuyo centro está a ``center_z`` (mundo).

    ``table_z`` es la altura de la mesa (para comprobar que el brazo no la toca);
    ``floor_z`` el suelo del contenedor (por defecto, la mesa). Devuelve un texto
    si no es posible, como ``skills.plan_pick``. ``yaw_choices`` limita los giros de la pinza que
    se prueban (los que ``selection`` ha visto con sitio para los dedos); por defecto, las cuatro
    orientaciones de una pieza cuadrada.
    """
    floor_z = table_z if floor_z is None else floor_z
    h = target.height
    # puntas de los dedos 4 mm por encima de la base de la pieza (o de la mesa si ya apoya en ella)
    tips_z = (center_z - h / 2) + skills.TIP_CLEARANCE_M - grasp_lower_m
    local_z = min(K.FINGERTIP_Z + (center_z - tips_z), -0.075)
    local = K.tcp_local(target.width, local_z)
    q_contact = K.q_for_gap(target.width, local_z)
    q_open = K.q_for_gap(target.width + OPEN_MARGIN_M, local_z)
    point = np.array([target.x, target.y, center_z])
    above = point + np.array([0.0, 0.0, h / 2 + skills.APPROACH_CLEARANCE_M])
    base_yaws = yaw_choices if yaw_choices else skills._yaw_candidates(target.yaw, target.shape)
    yaws = [y + yaw_offset for y in base_yaws]
    r, yaw = skills.solve_topdown(base, above, yaws, local, q_seed)
    if r is None:
        return "no hay postura del brazo que llegue desde arriba"
    descent, err_d = line_path(base, r.q, above, point, yaw, local)
    if err_d > 0.002:
        return f"la bajada recta no es alcanzable (error {err_d * 1000:.1f} mm)"
    lift_m = lift_height(center_z, floor_z, h, wall_top_z)
    lift, err_l = line_path(base, descent[-1], point, point + np.array([0, 0, lift_m]), yaw, local)
    if err_l > 0.002:
        return f"la subida recta no es alcanzable (error {err_l * 1000:.1f} mm)"
    clearance = skills.min_clearance(base, descent[-1], table_z)
    if clearance < 0.0:
        return f"el brazo tocaría la mesa ({clearance * 1000:.1f} mm)"
    return skills.PickPlan(target, yaw, local, q_open, q_contact, q_contact - skills.SQUEEZE_RAD, r.q,
                           descent, lift, center_z)


def lift_target_z(plan: skills.PickPlan, lift_m: float) -> float:
    """Altura (mundo) del centro de la pieza al terminar la subida."""
    return plan.grasp_z + lift_m


def tcp_height(base: np.ndarray, q_arm: np.ndarray, local: np.ndarray) -> float:
    """Altura (mundo) del punto de agarre de la pinza para unas articulaciones (cinemática directa)."""
    t = K.fk(dict(zip(K.ARM, q_arm)), base)["gripper"]
    return float((t[:3, :3] @ local + t[:3, 3])[2])


def tcp_xy(base: np.ndarray, q_arm: np.ndarray, local: np.ndarray) -> np.ndarray:
    t = K.fk(dict(zip(K.ARM, q_arm)), base)["gripper"]
    return (t[:3, :3] @ local + t[:3, 3])[:2]


def yaw_candidates_deg(shape: str) -> list[float]:
    return [math.degrees(y) for y in skills._yaw_candidates(0.0, shape)]


def find_plan(base: np.ndarray, scene, target: skills.Target, center_z: float, floor_z: float, q_seed: np.ndarray,
              yaws: list[float], wall_top_z: float | None = None, grasp_lower_m: float = 0.0, yaw_offset: float = 0.0,
              check_collisions: bool = True) -> tuple[skills.PickPlan | None, str]:
    """Primer giro de ``yaws`` (mejor holgura primero) con plan de agarre completo y trayecto sin choques.

    Devuelve ``(plan, "")`` o ``(None, motivo)``. Es el único sitio donde se unen cinemática inversa y
    comprobación de colisión: ``selection`` ya filtró por el hueco de los dedos, y esto añade lo que sólo se
    sabe con la geometría del brazo (el cuerpo de la pinza también choca con las paredes).
    """
    from . import collision

    top = scene.table.top_z_m
    wall_top = top + scene.containers[0].outer_m[2] if wall_top_z is None else wall_top_z
    reason = "ningún giro de la pinza tiene sitio"
    for yaw in yaws:
        plan = plan_pick_ex(base, top, target, center_z, wall_top, q_seed, grasp_lower_m=grasp_lower_m,
                            yaw_offset=yaw_offset, floor_z=floor_z, yaw_choices=[yaw])
        if isinstance(plan, str):
            reason = plan
            continue
        if check_collisions:
            hit = collision.check_pick(base, scene, q_seed, plan)
            if hit:
                reason = hit
                continue
        return plan, ""
    return None, reason
