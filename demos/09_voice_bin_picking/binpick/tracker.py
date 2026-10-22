"""Seguimiento de piezas entre observaciones: la misma pieza conserva su identificador.

Se asocia cada instancia nueva con las pistas conocidas (húngaro sobre la distancia en el
plano de la mesa; el color tiene que coincidir). Una pieza que se mueve (empujón, cambio de
caja) se vuelve a asociar en una segunda pasada con un radio mayor. De la comparación salen
los eventos que la máquina de estados necesita para verificar: «movida», «cambió de caja»,
«perdida». Una pieza que no se ve porque el brazo la tapa no se da por perdida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .instances import Instance

GATE_M = 0.025            # misma pieza sin moverse
REASSOC_M = 0.45          # misma pieza movida: cualquier sitio de la mesa
MOVED_M = 0.010
LOST_AFTER = 3            # observaciones seguidas a la vista sin encontrarla


@dataclass
class Track:
    id: str
    color: str | None
    shape: str | None
    pos: np.ndarray
    region: str
    top_z: float
    seen: int = 1
    missing: int = 0
    last_t: float = 0.0
    lost: bool = False

    def as_dict(self) -> dict:
        return {"id": self.id, "color": self.color, "shape": self.shape, "region": self.region,
                "pos": [round(float(v), 4) for v in self.pos], "lost": self.lost}


@dataclass
class Event:
    kind: str            # nueva | movida | cambio_de_caja | perdida | reaparece
    track: str
    detail: str = ""


def _compatible(t: Track, i: Instance) -> bool:
    if t.color is not None and i.color is not None and t.color != i.color:
        return False
    return True


@dataclass
class Tracker:
    tracks: dict[str, Track] = field(default_factory=dict)
    _n: int = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._n = 0

    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if not t.lost]

    def update(self, instances: list[Instance], t: float = 0.0,
               visible: Callable[[np.ndarray], bool] | None = None) -> list[Event]:
        """Asocia, pone ``Instance.track`` y devuelve los eventos. ``visible(pos)`` dice si el punto se ve (no lo tapa el brazo)."""
        events: list[Event] = []
        tracks = self.active()
        # una pieza en el aire (en la pinza) se sigue por color y forma, con el radio grande
        assigned: dict[int, str] = {}
        for radius in (GATE_M, REASSOC_M):
            free_t = [tr for tr in tracks if tr.id not in assigned.values()]
            free_i = [k for k in range(len(instances)) if k not in assigned]
            if not free_t or not free_i:
                break
            cost = np.full((len(free_t), len(free_i)), 1e6)
            for a, tr in enumerate(free_t):
                for b, k in enumerate(free_i):
                    inst = instances[k]
                    if inst.pos is None or not _compatible(tr, inst):
                        continue
                    d = float(np.linalg.norm(tr.pos[:2] - inst.pos[:2]))
                    if d <= radius:
                        penalty = 0.0 if (tr.shape in (None, "?") or inst.shape in (None, "?") or tr.shape == inst.shape) else 0.05
                        cost[a, b] = d + penalty
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < 1e5:
                    assigned[free_i[b]] = free_t[a].id

        for k, inst in enumerate(instances):
            if k in assigned:
                tr = self.tracks[assigned[k]]
                moved = float(np.linalg.norm(tr.pos[:2] - inst.pos[:2]))
                if tr.region != inst.region:
                    events.append(Event("cambio_de_caja", tr.id, f"{tr.region} → {inst.region}"))
                elif moved > MOVED_M:
                    events.append(Event("movida", tr.id, f"se ha movido {moved * 1000:.0f} mm"))
                tr.pos, tr.region, tr.top_z = inst.pos.copy(), inst.region, float(inst.top_z or 0.0)
                tr.seen += 1
                tr.missing = 0
                tr.last_t = t
                if inst.color and tr.color is None:
                    tr.color = inst.color
                if inst.shape not in (None, "?"):
                    tr.shape = inst.shape
                inst.track = tr.id
            else:
                if inst.pos is None:
                    continue
                self._n += 1
                tr = Track(f"P{self._n}", inst.color, inst.shape, inst.pos.copy(), inst.region,
                           float(inst.top_z or 0.0), last_t=t)
                self.tracks[tr.id] = tr
                inst.track = tr.id
                events.append(Event("nueva", tr.id, f"{inst.label()} en {inst.region}"))

        matched = set(assigned.values()) | {i.track for i in instances if i.track}
        for tr in tracks:
            if tr.id in matched:
                continue
            if visible is not None and not visible(tr.pos):
                continue                     # tapada por el brazo: no se sabe
            tr.missing += 1
            if tr.missing >= LOST_AFTER:
                tr.lost = True
                events.append(Event("perdida", tr.id, "ya no la veo"))
        return events

    def counts(self, region: str, color: str | None = None) -> int:
        return sum(1 for tr in self.active() if tr.region == region and (color is None or tr.color == color))
