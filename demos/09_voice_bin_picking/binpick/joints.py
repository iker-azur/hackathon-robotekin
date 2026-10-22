"""Unidades de las articulaciones: radianes del simulador ⇄ unidades de LeRobot.

El dataset y la política hablan el mismo idioma que un SO-101 real controlado
con LeRobot en modo ``range_m100_100`` (cuerpo en −100…100, pinza en 0…100), que
es también lo que exporta LeIsaac. Así una política entrenada aquí tiene el
mismo espacio de acciones que el robot físico (la calibración fina del robot
real sigue siendo cosa de ``lerobot-calibrate``).

La conversión es la de ``leisaac.utils.robot_utils`` (mapa lineal entre los
límites de las articulaciones del USD y los límites del motor), reescrita en
numpy para poder usarla y probarla sin Isaac.
"""

from __future__ import annotations

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# límites de las articulaciones en el USD de LeIsaac (grados)
USD_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
# límites del motor normalizado en LeRobot
MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

_USD = np.array([USD_LIMITS_DEG[n] for n in JOINT_NAMES], dtype=np.float64)
_MOT = np.array([MOTOR_LIMITS[n] for n in JOINT_NAMES], dtype=np.float64)


def _check(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"se esperaban {len(JOINT_NAMES)} articulaciones, llegaron {q.shape[-1]}")
    return q


def sim_to_motor(q_rad: np.ndarray) -> np.ndarray:
    """Radianes del simulador (…, 6) → unidades de LeRobot (…, 6), float32."""
    deg = np.degrees(_check(q_rad))
    frac = (deg - _USD[:, 0]) / (_USD[:, 1] - _USD[:, 0])
    return (frac * (_MOT[:, 1] - _MOT[:, 0]) + _MOT[:, 0]).astype(np.float32)


def motor_to_sim(m: np.ndarray) -> np.ndarray:
    """Unidades de LeRobot (…, 6) → radianes del simulador (…, 6), float64."""
    frac = (_check(m) - _MOT[:, 0]) / (_MOT[:, 1] - _MOT[:, 0])
    return np.radians(frac * (_USD[:, 1] - _USD[:, 0]) + _USD[:, 0])


def split(q6: np.ndarray) -> tuple[np.ndarray, float]:
    """(6,) → (brazo (5,), pinza)."""
    q6 = np.asarray(q6, dtype=np.float64)
    return q6[:5].copy(), float(q6[5])


def join(arm_q: np.ndarray, gripper_q: float) -> np.ndarray:
    return np.r_[np.asarray(arm_q, dtype=np.float64), float(gripper_q)]
