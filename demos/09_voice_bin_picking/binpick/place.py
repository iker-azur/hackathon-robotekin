"""Dónde dejar una pieza dentro de una caja: el hueco libre más holgado al que llega el brazo.

La colocación es la parte **programada** del sistema híbrido (la política aprendida sólo coge): el
destino es una caja fija y conocida, y lo que hay que decidir es un punto libre. Se elige entre una
rejilla de puntos del interior de la caja:

* alcanzable con la pinza vertical (cinemática inversa, ``recoverykit.skills``);
* a ≥ ``MIN_GAP_M`` de las piezas que la cámara ve ya dentro (los dedos abiertos necesitan sitio);
* el que más lejos queda de las demás, con preferencia por el centro.

La capacidad de una caja es lo que cabe con esa separación entre puntos alcanzables, no un número
fijo; el orquestador avisa antes de empezar si una orden no cabe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config as C
from recoverykit import kinematics as K  # noqa: E402
from recoverykit import skills  # noqa: E402

MIN_GAP_M = 0.032            # entre centros de piezas dejadas (7 mm entre caras); que quepan los dedos lo decide yaws_free
GRID_STEP_M = 0.012
EDGE_MARGIN_M = 0.016        # el centro queda a 16 mm de la pared (3,5 mm de pieza a pared); los dedos los comprueba yaws_free


def candidate_points(scene, dest: str, step: float = GRID_STEP_M, margin: float = EDGE_MARGIN_M) -> list[tuple[float, float]]:
    x0, x1, y0, y1 = C.inner_margin_box(scene, C.DEST_IDS[dest], margin)
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    return [(float(x), float(y)) for x in xs for y in ys]


def yaws_free(scene, dest: str, xy: tuple[float, float], occupied: list[tuple[float, float]],
              size: float = C.PIECE_SIZE_M) -> list[float]:
    """Giros de la pinza (rad) cuyos dedos caben en ``xy`` sin chocar con la pared de la caja ni con las piezas ya dejadas.

    Los dedos son asimétricos (el fijo pegado a la pieza, el móvil con holgura al otro lado), así que cerca
    de una pared sólo valen los giros que dejan el lado largo hacia dentro.
    """
    from . import selection as SEL
    from .scene_gen import _square, polygon_gap

    rect = C.inner_margin_box(scene, C.DEST_IDS[dest], 0.0)
    neigh = [_square(np.array(q), 0.0, size / 2 + 0.001) for q in occupied if math.hypot(q[0] - xy[0], q[1] - xy[1]) < 0.09]
    ok = []
    for yaw in skills._yaw_candidates(0.0, "cube"):
        worst = np.inf
        for slot in SEL.jaw_slots(np.array(xy), yaw, size):
            worst = min(worst, SEL._wall_clearance(slot, rect))
            for sq in neigh:
                worst = min(worst, polygon_gap(slot, sq))
        if worst >= SEL.CLEAR_OK_M:
            ok.append((worst, yaw))
    return [y for _w, y in sorted(ok, reverse=True)]


def choose_slot(scene, dest: str, occupied: list[tuple[float, float]], reach: Callable[[tuple[float, float]], bool],
                min_gap: float = MIN_GAP_M, yaws: Callable[[tuple[float, float]], list[float]] | None = None
                ) -> tuple[float, float] | None:
    """El mejor punto libre y alcanzable, o ``None`` si la caja está llena.

    Con la caja vacía se empieza por una esquina: así quedan sitios para las siguientes. ``yaws(p)`` (por defecto
    ``yaws_free``) debe dar al menos un giro de la pinza con sitio para los dedos.
    """
    c = C.container(scene, C.DEST_IDS[dest])
    cx, cy = c.center_xy_m
    yaws = yaws or (lambda p: yaws_free(scene, dest, p, occupied))
    best, best_score = None, -1e9
    for p in candidate_points(scene, dest):
        d = min((math.hypot(p[0] - q[0], p[1] - q[1]) for q in occupied), default=None)
        if d is not None and d < min_gap:
            continue
        if not reach(p) or not yaws(p):
            continue
        # vacía: la esquina más lejana al centro (deja sitio); con piezas: lo más lejos posible de ellas
        score = (math.hypot(p[0] - cx, p[1] - cy) if d is None else min(d, 0.09))
        if score > best_score:
            best, best_score = p, score
    return best


def capacity(scene, dest: str, reach: Callable[[tuple[float, float]], bool], min_gap: float = MIN_GAP_M) -> int:
    """Piezas que caben: se van dejando con ``choose_slot`` hasta que no queda hueco."""
    occupied: list[tuple[float, float]] = []
    while len(occupied) < 12:
        p = choose_slot(scene, dest, occupied, reach, min_gap)
        if p is None:
            break
        occupied.append(p)
    return len(occupied)


@dataclass
class Placer:
    """Cinemática de la colocación para un robot concreto (base, mesa y postura de partida)."""

    scene: object
    base: np.ndarray
    q_seed: np.ndarray
    _reach: dict = field(default_factory=dict)
    _cap: dict = field(default_factory=dict)

    def __post_init__(self):
        self.local = K.tcp_local(C.PIECE_SIZE_M)
        self.top = self.scene.table.top_z_m

    def reach_yaw(self, xy: tuple[float, float], yaw: float) -> bool:
        """¿Llega el brazo a ``xy`` (altura de tránsito) con la pinza girada ``yaw``?"""
        key = (round(xy[0], 3), round(xy[1], 3), round(yaw, 3))
        if key not in self._reach:
            above = np.array([xy[0], xy[1], self.top + skills.TRANSIT_Z_M])
            r, _ = skills.solve_topdown(self.base, above, [yaw], self.local, self.q_seed)
            self._reach[key] = r is not None
        return self._reach[key]

    def usable_yaws(self, dest: str, xy: tuple[float, float], occupied: list[tuple[float, float]]) -> list[float]:
        """Giros que tienen sitio para los dedos Y a los que llega el brazo (mejor holgura primero)."""
        return [y for y in yaws_free(self.scene, dest, xy, occupied) if self.reach_yaw(xy, y)]

    def reach_xy(self, xy: tuple[float, float]) -> bool:
        return any(self.reach_yaw(xy, y) for y in skills._yaw_candidates(0.0, "cube"))

    def choose(self, dest: str, occupied: list[tuple[float, float]]) -> tuple[float, float] | None:
        return choose_slot(self.scene, dest, occupied, lambda p: True,
                           yaws=lambda p: self.usable_yaws(dest, p, occupied))

    def capacity(self, dest: str) -> int:
        if dest not in self._cap:
            occupied: list[tuple[float, float]] = []
            while len(occupied) < 12:
                p = self.choose(dest, occupied)
                if p is None:
                    break
                occupied.append(p)
            self._cap[dest] = len(occupied)
        return self._cap[dest]

    def plan(self, dest: str, xy: tuple[float, float], q_seed: np.ndarray, occupied: list[tuple[float, float]] | None = None,
             piece_h: float = C.PIECE_SIZE_M):
        c = C.container(self.scene, C.DEST_IDS[dest])
        yaws = self.usable_yaws(dest, xy, occupied or []) or skills._yaw_candidates(0.0, "cube")
        return skills.plan_place(self.base, self.top, xy, self.top + c.floor_m, piece_h, self.local, yaws, q_seed)
