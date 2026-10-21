"""Cinemática del SO-101 sin Isaac: directa, inversa y geometría de la pinza.

La cadena se lee de ``assets/so101/kinematics.json``, extraída del mismo USD
que simula Isaac (``tools/extract_kinematics.py``). Convención de PhysX para
una articulación de revolución entre ``parent`` y ``child``::

    T_child = T_parent · T(pos0, rot0) · R(eje, q) · T(pos1, rot1)⁻¹

Todo es numpy y no depende del simulador: el planificador lo usa para decidir
si una pieza es alcanzable antes de mover nada, y sirve igual en el robot real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "so101"
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


# ------------------------------------------------------------------ utilidades
def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat(m) -> np.ndarray:
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def transform(pos, rot_wxyz) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_to_mat(rot_wxyz)
    t[:3, 3] = pos
    return t


def axis_rot(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    t = np.eye(4)
    if axis == "X":
        t[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "Y":
        t[0, 0], t[0, 2], t[2, 0], t[2, 2] = c, s, -s, c
    else:
        t[0:2, 0:2] = [[c, -s], [s, c]]
    return t


def yaw_transform(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    t = axis_rot("Z", yaw)
    t[:3, 3] = (x, y, z)
    return t


# ------------------------------------------------------------------- cadena
@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    t0: np.ndarray
    t1_inv: np.ndarray
    axis: str
    lower: float
    upper: float


@lru_cache(maxsize=1)
def chain() -> tuple[Joint, ...]:
    data = json.loads((ASSETS / "kinematics.json").read_text(encoding="utf-8"))
    joints = [Joint(j["name"], j["parent"], j["child"], transform(j["pos0"], j["rot0"]),
                    np.linalg.inv(transform(j["pos1"], j["rot1"])), j["axis"], j["lower"], j["upper"])
              for j in data["joints"]]
    ordered, known = [], {data["root"]}
    while len(ordered) < len(joints):
        for j in joints:
            if j not in ordered and j.parent in known:
                ordered.append(j)
                known.add(j.child)
    return tuple(ordered)


def limits() -> dict[str, tuple[float, float]]:
    return {j.name: (j.lower, j.upper) for j in chain()}


def fk(q: dict[str, float], base: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Poses 4×4 de todos los eslabones. ``base`` es la pose del eslabón raíz."""
    poses = {chain()[0].parent: np.eye(4) if base is None else base}
    for j in chain():
        poses[j.child] = poses[j.parent] @ j.t0 @ axis_rot(j.axis, q.get(j.name, 0.0)) @ j.t1_inv
    return poses


@lru_cache(maxsize=1)
def colliders() -> dict[str, np.ndarray]:
    d = np.load(ASSETS / "colliders.npz")
    return {k: d[k].astype(np.float64) for k in d.files}


# ------------------------------------------------------------ pinza y TCP
FINGERTIP_Z = -0.104      # punta de los dedos en el marco "gripper" (medido de los colisionadores)
GRIP_SLAB_M = 0.006       # media altura de la franja de contacto que se mira


def jaw_points(q_grip: float) -> np.ndarray:
    """Vértices de la mordaza móvil en el marco de la pinza para un ángulo dado."""
    j = next(x for x in chain() if x.name == "gripper")
    t = j.t0 @ axis_rot(j.axis, q_grip) @ j.t1_inv
    v = colliders()["jaw"]
    return v @ t[:3, :3].T + t[:3, 3]


def _slab(points: np.ndarray, z: float) -> np.ndarray:
    return points[np.abs(points[:, 2] - z) <= GRIP_SLAB_M]


def fixed_face_x(z: float) -> float:
    """Cara interior del dedo fijo (x máximo) a la altura ``z`` del marco de la pinza."""
    s = _slab(colliders()["gripper"], z)
    s = s[np.abs(s[:, 1]) < 0.01]
    return float(s[:, 0].max())


def gap(q_grip: float, z: float) -> float:
    """Separación entre caras interiores de los dedos a la altura ``z`` (m)."""
    s = _slab(jaw_points(q_grip), z)
    s = s[np.abs(s[:, 1]) < 0.01]
    if len(s) == 0:
        return float("inf")
    return float(s[:, 0].min()) - fixed_face_x(z)


@lru_cache(maxsize=64)
def q_for_gap(width: float, z: float = -0.0955) -> float:
    """Ángulo de la pinza al que la separación vale ``width`` (búsqueda en rejilla fina)."""
    qs = np.linspace(limits()["gripper"][0], limits()["gripper"][1], 800)
    gaps = np.array([gap(q, z) for q in qs])
    ok = np.where(gaps >= width)[0]
    return float(qs[ok[0]]) if len(ok) else float(qs[-1])


def tcp_local(width: float, z: float = -0.0955) -> np.ndarray:
    """Centro de una pieza de anchura ``width`` sujeta contra el dedo fijo (marco pinza)."""
    return np.array([fixed_face_x(z) + width / 2, 0.0, z])


def topdown_rotation(yaw: float) -> np.ndarray:
    """Pinza con los dedos hacia abajo y el eje de cierre (x de la pinza) a ``yaw``."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, 1.0])      # el eje z de la pinza apunta arriba: los dedos (−z) miran a la mesa
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


# ------------------------------------------------------------- cinemática inversa
def _rot_error(r_now: np.ndarray, r_goal: np.ndarray) -> np.ndarray:
    e = r_goal @ r_now.T
    angle = np.arccos(np.clip((np.trace(e) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]])
    return w * angle / (2 * np.sin(angle)) if angle < np.pi - 1e-6 else w


@dataclass
class IKResult:
    q: np.ndarray            # 5 articulaciones del brazo
    pos_err_m: float
    rot_err_rad: float
    ok: bool


def ik(target: np.ndarray, base: np.ndarray, q0: np.ndarray, link: str = "gripper",
       rot_weight: float = 0.3, iters: int = 200, tol_pos: float = 5e-4, tol_rot: float = 0.01) -> IKResult:
    """Mínimos cuadrados amortiguados sobre la pose 4×4 ``target`` del eslabón ``link``."""
    lim = limits()
    lo = np.array([lim[n][0] for n in ARM])
    hi = np.array([lim[n][1] for n in ARM])
    q = np.clip(np.asarray(q0, dtype=float).copy(), lo, hi)
    axis_col = {"X": 0, "Y": 1, "Z": 2}
    arm_joints = [j for j in chain() if j.name in ARM]

    def err_jac(qv, with_jac=True):
        poses = fk(dict(zip(ARM, qv)), base)
        t = poses[link]
        e = np.r_[target[:3, 3] - t[:3, 3], rot_weight * _rot_error(t[:3, :3], target[:3, :3])]
        if not with_jac:
            return e, None
        # jacobiano geométrico de una cadena de revolución: [zᵢ × (p − pᵢ); zᵢ]
        jac = np.zeros((6, 5))
        for k, j in enumerate(arm_joints):
            frame = poses[j.parent] @ j.t0
            z = frame[:3, axis_col[j.axis]]
            jac[:3, k] = np.cross(z, t[:3, 3] - frame[:3, 3])
            jac[3:, k] = rot_weight * z
        return e, jac

    lam = 0.01
    e, jac = err_jac(q)
    for _ in range(iters):
        step = np.linalg.solve(jac.T @ jac + lam * np.eye(5), jac.T @ e)
        q_new = np.clip(q + step, lo, hi)
        e_new, _ = err_jac(q_new, with_jac=False)
        if np.linalg.norm(e_new) < np.linalg.norm(e):
            q, lam = q_new, max(lam / 3, 1e-6)
            e, jac = err_jac(q)
        else:
            lam *= 4
            if lam > 1e3:
                break
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) / rot_weight < tol_rot:
            break
    pe, re = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]) / rot_weight)
    return IKResult(q, pe, re, pe < 1e-3 and re < 0.02)


def gripper_target(point: np.ndarray, yaw: float, local: np.ndarray) -> np.ndarray:
    """Pose del eslabón "gripper" que pone el punto ``local`` de la pinza en ``point``."""
    r = topdown_rotation(yaw)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = point - r @ local
    return t
