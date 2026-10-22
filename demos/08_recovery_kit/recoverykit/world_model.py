"""Modelo del mundo: piezas seguidas entre observaciones de la cámara.

Cada observación se asocia a las piezas conocidas (mismo color y forma, la más
cercana). De la comparación salen los eventos que el agente necesita para
decidir: una pieza se ha movido, ha entrado en la caja, se ha caído de ella o
ha desaparecido. Las piezas tapadas por el brazo no se dan por perdidas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .perception import Detection, Observation

MATCH_GATE_M = 0.03       # misma pieza sin moverse (ruido + pequeños empujones)
REASSOC_MAX_M = 0.45      # misma pieza movida: cualquier sitio de la mesa
MOVED_M = 0.012           # desplazamiento que cuenta como "movida"
LOST_AFTER = 3            # observaciones seguidas sin verla, estando a la vista
LIFTED_EXTRA_M = 0.015    # más alta que su tamaño: está en el aire (en la pinza)

COLOR_M = {"red": "rojo", "blue": "azul", "green": "verde", "yellow": "amarillo"}
COLOR_F = {"red": "roja", "blue": "azul", "green": "verde", "yellow": "amarilla"}
SHAPE_ES = {"cube": ("cubo", "m"), "cylinder": ("cilindro", "m"), "prism": ("prisma", "m"), "?": ("pieza", "f")}


@dataclass
class Track:
    id: str
    color: str
    shape: str
    pos: np.ndarray
    yaw: float
    width: float
    height: float
    state: str = "mesa"         # "mesa" | "caja" | "pinza" | "perdida"
    box_id: str | None = None
    seen: int = 1
    missing: int = 0
    last_t: float = 0.0

    def _noun(self) -> tuple[str, str]:
        name, gender = SHAPE_ES.get(self.shape, ("pieza", "f"))
        adj = (COLOR_F if gender == "f" else COLOR_M).get(self.color, self.color)
        return f"{name} {adj} ({self.id})", gender

    def label(self) -> str:
        """'cubo rojo (rojo_1)'."""
        return self._noun()[0]

    def the(self) -> str:
        """'el cubo rojo (rojo_1)' / 'la pieza roja (rojo_1)'."""
        noun, g = self._noun()
        return ("la " if g == "f" else "el ") + noun

    def a(self) -> str:
        noun, g = self._noun()
        return ("una " if g == "f" else "un ") + noun

    def as_dict(self) -> dict:
        return {"id": self.id, "color": self.color, "shape": self.shape, "state": self.state,
                "pos": [round(float(v), 4) for v in self.pos], "yaw": round(float(self.yaw), 3),
                "box": self.box_id, "label": self.label()}


@dataclass
class Event:
    kind: str                   # "nueva" | "movida" | "en_caja" | "fuera_de_caja" | "perdida" | "reaparece"
    track: str
    detail: str
    displacement_m: float = 0.0


def _compatible(t: Track, d: Detection) -> bool:
    return t.color == d.color and (t.shape == d.shape or "?" in (t.shape, d.shape))


@dataclass
class WorldModel:
    tracks: dict[str, Track] = field(default_factory=dict)
    held: str | None = None
    boxes: list = field(default_factory=list)
    _counter: dict[str, int] = field(default_factory=dict)

    def _new_id(self, color: str) -> str:
        n = self._counter.get(color, 0) + 1
        self._counter[color] = n
        return f"{COLOR_M.get(color, color)}_{n}"

    def update(self, obs: Observation, t: float,
               visible: Callable[[np.ndarray], bool] | None = None) -> list[Event]:
        """Incorpora una observación. ``visible(p)`` dice si el punto p no lo tapa el brazo."""
        if obs.boxes:
            self.boxes = obs.boxes
        events: list[Event] = []
        # una pieza en el aire (en la pinza) o el propio brazo no se asocian
        dets = obs.resting()
        cands = [k for k in self.tracks if k != self.held]
        used_t, used_d = set(), set()

        # dos pasadas: primero la misma pieza quieta, después la misma pieza movida
        for gate in (MATCH_GATE_M, REASSOC_MAX_M):
            tl = [k for k in cands if k not in used_t]
            dl = [i for i in range(len(dets)) if i not in used_d]
            if not tl or not dl:
                break
            cost = np.full((len(tl), len(dl)), 1e6)
            for a, k in enumerate(tl):
                for b, i in enumerate(dl):
                    if _compatible(self.tracks[k], dets[i]):
                        cost[a, b] = float(np.linalg.norm(self.tracks[k].pos[:2] - dets[i].pos[:2]))
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] <= gate:
                    used_t.add(tl[a])
                    used_d.add(dl[b])
                    events += self._apply(self.tracks[tl[a]], dets[dl[b]], t)

        for i, d in enumerate(dets):
            if i in used_d:
                continue
            tr = Track(self._new_id(d.color), d.color, d.shape, d.pos.copy(), d.yaw, d.width_m, d.height_m,
                       state="caja" if d.in_box else "mesa", box_id=d.box_id, last_t=t)
            self.tracks[tr.id] = tr
            events.append(Event("nueva", tr.id, f"veo {tr.a()}" + (" dentro de la caja" if d.in_box else "")))

        for k in cands:
            if k in used_t:
                continue
            tr = self.tracks[k]
            if visible is not None and not visible(tr.pos):
                continue
            tr.missing += 1
            if tr.missing >= LOST_AFTER and tr.state != "perdida":
                tr.state = "perdida"
                events.append(Event("perdida", k, f"ya no veo {tr.the()}"))
        return events

    def _apply(self, tr: Track, d: Detection, t: float) -> list[Event]:
        events = []
        moved = float(np.linalg.norm(tr.pos[:2] - d.pos[:2]))
        was = tr.state
        now = "caja" if d.in_box else "mesa"
        if was == "perdida":
            events.append(Event("reaparece", tr.id, f"vuelvo a ver {tr.the()}", moved))
        if was in ("mesa", "perdida") and now == "caja":
            events.append(Event("en_caja", tr.id, f"{tr.the()} está dentro de la caja", moved))
        elif was == "caja" and now == "mesa":
            events.append(Event("fuera_de_caja", tr.id, f"{tr.the()} está fuera de la caja", moved))
        elif was == "mesa" and now == "mesa" and moved > MOVED_M:
            events.append(Event("movida", tr.id, f"{tr.the()} se ha movido {moved*1000:.0f} mm", moved))
        tr.pos, tr.yaw = d.pos.copy(), d.yaw
        tr.width, tr.height = d.width_m, d.height_m
        if tr.shape == "?" and d.shape != "?":
            tr.shape = d.shape
        tr.state, tr.box_id = now, d.box_id
        tr.seen += 1
        tr.missing = 0
        tr.last_t = t
        return events

    # ------------------------------------------------------------ la pinza
    def grasped(self, track_id: str) -> None:
        self.held = track_id
        self.tracks[track_id].state = "pinza"

    def released(self) -> str | None:
        """Tras soltar, la pieza vuelve a "mesa" hasta que la cámara diga dónde está."""
        k, self.held = self.held, None
        if k is not None and k in self.tracks:
            self.tracks[k].state = "mesa"
            self.tracks[k].missing = 0
        return k

    def snapshot(self) -> list[dict]:
        return [t.as_dict() for t in self.tracks.values()]
