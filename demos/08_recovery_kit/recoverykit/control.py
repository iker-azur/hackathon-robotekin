"""Control de movimiento: trayectorias articulares y cartesianas, y la pinza.

Los servos del SO-101 reciben consignas de posición. Aquí se generan esas
consignas paso a paso con límites de velocidad y aceleración, igual que haría
un controlador real a 50–100 Hz; PhysX resuelve el PD de cada servo con su
límite de par. Nada mueve una pieza: sólo el brazo empuja, aprieta y suelta.

Cada movimiento llama a ``monitor(info)`` cada ``MONITOR_EVERY`` pasos. Si el
monitor devuelve un texto, el movimiento se detiene y devuelve ese motivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import kinematics as K

MONITOR_EVERY = 12            # pasos (20 Hz con dt = 1/240)
SETTLE_TOL_RAD = 0.004
SETTLE_MAX_S = 0.6
INTEGRAL_GAIN = 8.0           # 1/s
MAX_TRIM_RAD = 0.08
LINE_STEP_M = 0.004


@dataclass
class MoveResult:
    ok: bool
    reason: str = ""
    duration_s: float = 0.0
    final_error_rad: float = 0.0
    aborted: bool = False


def trapezoid(q0: np.ndarray, q1: np.ndarray, vmax: float, amax: float) -> tuple[float, Callable[[float], np.ndarray]]:
    """Perfil trapezoidal sincronizado: todas las articulaciones llegan a la vez."""
    q0 = np.asarray(q0, float)
    d = np.asarray(q1, float) - q0
    dist = float(np.max(np.abs(d))) if len(d) else 0.0
    if dist < 1e-9:
        return 0.0, lambda t: q0 + d
    t_acc = vmax / amax
    if amax * t_acc ** 2 >= dist:          # triangular: no llega a vmax
        t_acc = float(np.sqrt(dist / amax))
        v = amax * t_acc
        total = 2 * t_acc
    else:
        v = vmax
        total = 2 * t_acc + (dist - amax * t_acc ** 2) / vmax

    def at(t: float) -> np.ndarray:
        t = min(max(t, 0.0), total)
        if t < t_acc:
            x = 0.5 * amax * t * t
        elif t > total - t_acc:
            r = total - t
            x = dist - 0.5 * amax * r * r
        else:
            x = 0.5 * amax * t_acc ** 2 + v * (t - t_acc)
        return q0 + d * (x / dist)

    return total, at


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Controller:
    """Mueve el brazo de una ``Workcell`` respetando los límites del servo."""

    def __init__(self, cell, monitor: Callable[[dict], str | None] | None = None,
                 on_step: Callable[[], None] | None = None):
        from .workcell import base_transform

        self.cell = cell
        self.cfg = cell.cfg
        self.base = base_transform(cell.cfg)
        self.monitor = monitor
        self.on_step = on_step
        self.dt = cell.cfg.physics.dt
        self.gripper_cmd = float(cell.cfg.robot.start_joints[5])

    # ---------------------------------------------------------------- básicos
    def q(self) -> np.ndarray:
        return self.cell.proprio()["arm_q"]

    def _tick(self, n: int) -> str | None:
        self.cell.step()
        if self.on_step:
            self.on_step()
        if self.monitor and n % MONITOR_EVERY == 0:
            return self.monitor({"step": n, "proprio": self.cell.proprio()})
        return None

    def _follow(self, total: float, at: Callable[[float], np.ndarray], goal: np.ndarray) -> MoveResult:
        steps = int(np.ceil(total / self.dt))
        for n in range(steps + 1):
            self.cell.set_arm_targets(at(n * self.dt))
            reason = self._tick(n)
            if reason:
                return MoveResult(False, reason, n * self.dt, aborted=True)
        # asentamiento con corrección integral: un servo de posición con carga
        # (gravedad) se queda corto; se lee la posición real y se corrige la
        # consigna, como se haría con Present_Position en el robot real
        n_settle = 0
        trim = np.zeros_like(goal)
        while n_settle * self.dt < SETTLE_MAX_S:
            err_v = goal - self.q()
            if float(np.max(np.abs(err_v))) < SETTLE_TOL_RAD:
                break
            trim = np.clip(trim + INTEGRAL_GAIN * self.dt * err_v, -MAX_TRIM_RAD, MAX_TRIM_RAD)
            self.cell.set_arm_targets(goal + trim)
            reason = self._tick(steps + n_settle)
            if reason:
                return MoveResult(False, reason, (steps + n_settle) * self.dt, aborted=True)
            n_settle += 1
        self.trim = trim
        err = float(np.max(np.abs(self.q() - goal)))
        ok = err < 0.02
        return MoveResult(ok, "" if ok else "no alcanza la consigna", (steps + n_settle) * self.dt, err)

    # ------------------------------------------------------------- movimientos
    def move_joints(self, goal: np.ndarray, speed: float = 1.0) -> MoveResult:
        r = self.cfg.robot
        total, at = trapezoid(self.q(), goal, r.max_joint_speed * speed, r.max_joint_accel * speed)
        return self._follow(total, at, np.asarray(goal, float))

    def move_line(self, start_q: np.ndarray, path: list[np.ndarray], speed_m_s: float) -> MoveResult:
        """Recorre configuraciones de IK sobre una recta a velocidad cartesiana ~constante.

        Cada tramo de ``LINE_STEP_M`` dura lo que marca ``speed_m_s`` (o más si
        una articulación llegaría a su velocidad máxima). Una rampa smoothstep
        sobre el parámetro del camino arranca y frena sin saltos; alargar la
        duración ×1.5 mantiene la velocidad de pico en la nominal.
        """
        if not path:
            return MoveResult(True)
        vmax = self.cfg.robot.max_joint_speed
        knots = [np.asarray(start_q, float)] + [np.asarray(p, float) for p in path]
        times = [0.0]
        for a, b in zip(knots[:-1], knots[1:]):
            times.append(times[-1] + max(LINE_STEP_M / speed_m_s, float(np.max(np.abs(b - a))) / vmax))
        nominal = times[-1]
        duration = 1.5 * nominal

        def at(t: float) -> np.ndarray:
            tt = nominal * smoothstep(t / duration)
            i = min(int(np.searchsorted(times, tt, side="right") - 1), len(knots) - 2)
            seg = times[i + 1] - times[i]
            a = 0.0 if seg <= 0 else (tt - times[i]) / seg
            return knots[i] * (1 - a) + knots[i + 1] * a

        return self._follow(duration, at, knots[-1])

    # ------------------------------------------------------------------ pinza
    def gripper(self, q_target: float, duration_s: float = 0.6, hold_s: float = 0.25) -> MoveResult:
        """Lleva la consigna de la pinza a ``q_target`` en rampa y espera ``hold_s``.

        Al cerrar sobre una pieza la consigna queda más allá del contacto: el
        servo aprieta con su par limitado, como el STS3215 real.
        """
        q0 = self.gripper_cmd
        steps = max(1, int(duration_s / self.dt))
        hold = int(hold_s / self.dt)
        for n in range(steps + hold):
            self.gripper_cmd = q0 + (q_target - q0) * smoothstep(n / steps)
            self.cell.set_gripper_target(self.gripper_cmd)
            reason = self._tick(n)
            if reason:
                return MoveResult(False, reason, n * self.dt, aborted=True)
        return MoveResult(True, duration_s=(steps + hold) * self.dt)

    def wait(self, seconds: float) -> MoveResult:
        for n in range(int(seconds / self.dt)):
            reason = self._tick(n)
            if reason:
                return MoveResult(False, reason, n * self.dt, aborted=True)
        return MoveResult(True, duration_s=seconds)


def line_path(base: np.ndarray, q_start: np.ndarray, p0: np.ndarray, p1: np.ndarray, yaw: float,
              local: np.ndarray) -> tuple[list[np.ndarray], float]:
    """IK encadenada a lo largo de la recta p0→p1 (posición del punto ``local`` de la pinza).

    Devuelve las configuraciones y el peor error de posición a lo largo de la recta.
    """
    dist = float(np.linalg.norm(p1 - p0))
    n = max(1, int(np.ceil(dist / LINE_STEP_M)))
    q = np.asarray(q_start, float)
    path, worst = [], 0.0
    for i in range(1, n + 1):
        p = p0 + (p1 - p0) * (i / n)
        r = K.ik(K.gripper_target(p, yaw, local), base, q, iters=60)
        worst = max(worst, r.pos_err_m)
        q = r.q
        path.append(q.copy())
    return path, worst
