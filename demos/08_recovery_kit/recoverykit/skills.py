"""Habilidades de manipulación: coger y dejar, con comprobaciones propioceptivas.

Un plan (``plan_pick``/``plan_place``) es numpy puro: decide el giro de la
pinza, la pose de aproximación y las rectas de bajada y subida, y dice si algo
no es alcanzable antes de mover el brazo. La ejecución usa ``control.Controller``.

Comprobación del agarre sin mirar el simulador: al cerrar sobre una pieza la
mordaza se queda bloqueada a la anchura de la pieza; si cierra en vacío llega
hasta la consigna de apriete. Es lo que reporta un STS3215 real (posición y
carga), así que la misma regla sirve en el robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import kinematics as K
from .control import Controller, line_path

# Alturas limitadas por el alcance vertical medido (tools/reach_map.py): con la
# pinza vertical el SO-101 sólo llega a ~9 cm sobre la mesa en la corona útil.
APPROACH_CLEARANCE_M = 0.04    # altura de la aproximación sobre la cara superior de la pieza
OPEN_MARGIN_M = 0.016          # apertura sobre la anchura de la pieza
SQUEEZE_RAD = 0.15             # consigna de cierre más allá del contacto
HELD_TOLERANCE_RAD = 0.07      # la mordaza bloqueada debe estar a menos de esto del contacto
TIP_CLEARANCE_M = 0.004        # puntas de los dedos sobre la mesa al agarrar
LIFT_M = 0.065                 # la base de la pieza pasa 2 cm por encima del borde de la caja
DESCENT_SPEED = 0.06           # m/s
LIFT_SPEED = 0.07
RELEASE_DROP_M = 0.012         # base de la pieza sobre el suelo de la caja al soltar
TRANSIT_Z_M = 0.09             # altura de tránsito del punto de agarre sobre la mesa


@dataclass
class Target:
    """Lo que el agente sabe de una pieza (por la cámara o, en pruebas, por la verdad)."""

    x: float
    y: float
    yaw: float             # orientación de una cara; irrelevante para cilindros
    width: float           # anchura entre caras paralelas (diámetro en cilindros)
    height: float
    shape: str = "cube"


@dataclass
class PickPlan:
    target: Target
    yaw: float
    local: np.ndarray
    q_open: float
    q_contact: float
    q_squeeze: float
    q_pre: np.ndarray
    descent: list[np.ndarray] = field(default_factory=list)
    lift: list[np.ndarray] = field(default_factory=list)
    grasp_z: float = 0.0


@dataclass
class PlacePlan:
    point: np.ndarray
    yaw: float
    q_above: np.ndarray
    lower: list[np.ndarray] = field(default_factory=list)
    retreat: list[np.ndarray] = field(default_factory=list)


@dataclass
class Outcome:
    status: str            # "ok" | "vacío" | "deslizada" | "abortado" | "fallo de movimiento"
    detail: str = ""
    gripper_q: float | None = None
    phase: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# --------------------------------------------------------------------- planes
def _yaw_candidates(yaw: float, shape: str) -> list[float]:
    if shape == "cylinder":
        return [yaw + k * math.pi / 4 for k in range(8)]
    if shape == "prism":
        return [yaw + k * math.pi / 3 for k in range(6)]
    return [yaw + k * math.pi / 2 for k in range(4)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def solve_topdown(base: np.ndarray, point: np.ndarray, yaws: list[float], local: np.ndarray,
                  q_seed: np.ndarray) -> tuple[K.IKResult | None, float]:
    """Mejor solución vertical entre varios giros: la que menos mueve el brazo.

    Además de la configuración actual se prueban semillas con el hombro ya
    girado hacia el punto: la IK local no cruza bien de un lado al otro.
    """
    rel = np.linalg.inv(base) @ np.r_[point, 1.0]
    # en el asset, pan = 0 apunta a −y de la base y pan > 0 gira hacia −x:
    # ángulo en la base ≈ −π/2 − pan (medido con kinematics.fk)
    pan = -(float(np.arctan2(rel[1], rel[0])) + np.pi / 2)
    seeds = [np.asarray(q_seed, float)] + [np.array([pan, s1, s2, s3, 0.0])
                                           for s1, s2, s3 in ((-0.3, 0.6, 1.2), (0.3, 0.0, 1.4), (0.0, 0.9, 0.9))]
    best, best_yaw, best_cost = None, 0.0, float("inf")
    for yaw in yaws:
        for seed in seeds:
            r = K.ik(K.gripper_target(point, _wrap(yaw), local), base, seed)
            if not r.ok:
                continue
            cost = float(np.sum(np.abs(r.q - q_seed)[:4])) + 0.5 * abs(r.q[4] - q_seed[4])
            if cost < best_cost:
                best, best_yaw, best_cost = r, _wrap(yaw), cost
            break
    return best, best_yaw


def plan_pick(base: np.ndarray, table_z: float, t: Target, q_seed: np.ndarray,
              grasp_lower_m: float = 0.0, yaw_offset: float = 0.0) -> PickPlan | str:
    """Plan de agarre vertical. Devuelve un texto si no es posible.

    ``grasp_lower_m`` y ``yaw_offset`` son ajustes de estrategia tras un fallo.
    El punto de agarre (centro de la pieza) queda a la altura a la que las
    puntas de los dedos están a ``TIP_CLEARANCE_M`` de la mesa.
    """
    center_z = table_z + t.height / 2
    tips_z = table_z + TIP_CLEARANCE_M - grasp_lower_m
    local_z = min(K.FINGERTIP_Z + (center_z - tips_z), -0.075)   # dentro de la zona de los dedos
    local = K.tcp_local(t.width, local_z)
    q_contact = K.q_for_gap(t.width, local_z)
    q_open = K.q_for_gap(t.width + OPEN_MARGIN_M, local_z)
    point = np.array([t.x, t.y, center_z])
    above = point + np.array([0.0, 0.0, t.height / 2 + APPROACH_CLEARANCE_M])
    yaws = [y + yaw_offset for y in _yaw_candidates(t.yaw, t.shape)]
    r, yaw = solve_topdown(base, above, yaws, local, q_seed)
    if r is None:
        return "no hay postura del brazo que llegue desde arriba"
    descent, err_d = line_path(base, r.q, above, point, yaw, local)
    if err_d > 0.002:
        return f"la bajada recta no es alcanzable (error {err_d*1000:.1f} mm)"
    lift, err_l = line_path(base, descent[-1], point, point + np.array([0, 0, LIFT_M]), yaw, local)
    if err_l > 0.002:
        return f"la subida recta no es alcanzable (error {err_l*1000:.1f} mm)"
    clearance = min_clearance(base, descent[-1], table_z)
    if clearance < 0.0:
        return f"el brazo tocaría la mesa ({clearance*1000:.1f} mm)"
    return PickPlan(t, yaw, local, q_open, q_contact, q_contact - SQUEEZE_RAD, r.q, descent, lift, center_z)


def plan_place(base: np.ndarray, table_z: float, point_xy: tuple[float, float], floor_z: float,
               piece_h: float, local: np.ndarray, yaws: list[float], q_seed: np.ndarray) -> PlacePlan | str:
    center = np.array([point_xy[0], point_xy[1], floor_z + RELEASE_DROP_M + piece_h / 2])
    above = np.array([point_xy[0], point_xy[1], table_z + TRANSIT_Z_M])
    r, yaw = solve_topdown(base, above, yaws, local, q_seed)
    if r is None:
        return "la caja no es alcanzable desde arriba"
    lower, err = line_path(base, r.q, above, center, yaw, local)
    if err > 0.002:
        return f"la bajada a la caja no es alcanzable (error {err*1000:.1f} mm)"
    retreat, _ = line_path(base, lower[-1], center, above, yaw, local)
    return PlacePlan(center, yaw, r.q, lower, retreat)


def min_clearance(base: np.ndarray, q: np.ndarray, table_z: float) -> float:
    """Distancia mínima de los eslabones del brazo (sin dedos) a la mesa."""
    poses = K.fk(dict(zip(K.ARM, q)), base)
    worst = float("inf")
    for link in ("upper_arm", "lower_arm", "wrist"):
        v = K.colliders()[link]
        t = poses[link]
        z = v @ t[2, :3] + t[2, 3]
        worst = min(worst, float(z.min()) - table_z)
    return worst


# ------------------------------------------------------------------ ejecución
def held(gripper_q: float, plan: PickPlan) -> bool:
    """La mordaza está bloqueada por una pieza (y no cerrada en vacío)."""
    return gripper_q > plan.q_contact - HELD_TOLERANCE_RAD


def _noop(_phase: str) -> None:
    return None


def execute_pick(ctrl: Controller, plan: PickPlan, speed: float = 1.0, lift_speed: float = LIFT_SPEED,
                 on_phase=_noop) -> Outcome:
    """Abrir, aproximar, bajar, cerrar y subir. ``on_phase(nombre)`` se llama al empezar cada fase."""
    on_phase("abrir")
    r = ctrl.gripper(plan.q_open, duration_s=0.4, hold_s=0.05)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="abrir")
    on_phase("aproximación")
    r = ctrl.move_joints(plan.q_pre, speed=speed)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="aproximación")
    if not r.ok:
        return Outcome("fallo de movimiento", r.reason, phase="aproximación")
    on_phase("bajada")
    r = ctrl.move_line(ctrl.q(), plan.descent, DESCENT_SPEED * speed)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="bajada")
    on_phase("cierre")
    r = ctrl.gripper(plan.q_squeeze, duration_s=0.5, hold_s=0.3)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="cierre")
    q_closed = ctrl.cell.proprio()["gripper_q"]
    if not held(q_closed, plan):
        return Outcome("vacío", "la pinza cerró sin encontrar la pieza", q_closed, "cierre")
    on_phase("subida")
    r = ctrl.move_line(ctrl.q(), plan.lift, lift_speed)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="subida")
    q_lift = ctrl.cell.proprio()["gripper_q"]
    if not held(q_lift, plan):
        return Outcome("deslizada", "la pieza se soltó al levantar", q_lift, "subida")
    return Outcome("ok", gripper_q=q_lift, phase="subida")


def execute_place(ctrl: Controller, pick: PickPlan, place: PlacePlan, speed: float = 1.0,
                  on_phase=_noop) -> Outcome:
    """Transportar, bajar dentro de la caja, soltar y retirarse."""
    on_phase("transporte")
    r = ctrl.move_joints(place.q_above, speed=speed)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="transporte")
    q_now = ctrl.cell.proprio()["gripper_q"]
    if not held(q_now, pick):
        return Outcome("deslizada", "la pieza se cayó durante el transporte", q_now, "transporte")
    on_phase("bajada a la caja")
    r = ctrl.move_line(ctrl.q(), place.lower, DESCENT_SPEED * speed)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="bajada a la caja")
    on_phase("soltar")
    r = ctrl.gripper(pick.q_open, duration_s=0.35, hold_s=0.25)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="soltar")
    on_phase("retirada")
    r = ctrl.move_line(ctrl.q(), place.retreat, LIFT_SPEED)
    if r.aborted:
        return Outcome("abortado", r.reason, phase="retirada")
    return Outcome("ok", phase="soltar")


def box_slots(center_xy: tuple[float, float], inner_xy: tuple[float, float], n: int = 4) -> list[tuple[float, float]]:
    """Huecos de la caja: rejilla 2×2 separada para que la mordaza abra sin tocar vecinas."""
    cx, cy = center_xy
    dx, dy = inner_xy[0] / 4, inner_xy[1] / 4
    slots = [(cx - dx, cy - dy), (cx + dx, cy - dy), (cx - dx, cy + dy), (cx + dx, cy + dy)]
    return slots[:n]
