"""Planificador de tareas: qué falta en la caja y con qué pieza conseguirlo.

El estado de referencia es lo que la cámara ve dentro de la caja, no lo que el
robot cree haber hecho. En cada ciclo:

1. El pedido se expande en unidades ("pieza roja 1/1", "pieza azul 1/2"…).
2. Las piezas vistas dentro de la caja cubren unidades (primero las más
   específicas). Lo que sobra en la caja da tareas de retirar.
3. Cada unidad sin cubrir recibe la mejor pieza de la mesa: compatible, no
   descartada, alcanzable y cerca de la caja.

Así una pieza que se cae fuera de la caja vuelve a ser candidata sin ninguna
regla especial, y una pieza movida se coge donde está ahora.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .kit import COLOR_ES_F, COLOR_ES_M, SHAPE_ES, Goal
from .world_model import Track, WorldModel


@dataclass
class Unit:
    key: tuple[str, str | None]
    index: int
    of: int

    @property
    def id(self) -> str:
        return f"{self.key[0]}:{self.key[1] or 'any'}:{self.index}"

    def label(self) -> str:
        color, shape = self.key
        sing, _, gender = SHAPE_ES[shape]
        adj = (COLOR_ES_F if gender == "f" else COLOR_ES_M)[color][0]
        return f"{sing} {adj} {self.index}/{self.of}"


@dataclass
class Task:
    id: str
    kind: str                  # "colocar" | "retirar"
    label: str
    unit: str | None
    track: str | None = None
    slot: int | None = None
    state: str = "pendiente"   # "pendiente" | "ejecutando" | "completada" | "fallida" | "bloqueada"
    attempts: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "track": self.track, "slot": self.slot,
                "state": self.state, "attempts": self.attempts, "note": self.note}


@dataclass
class KitPlan:
    units: list[Unit]
    covered: dict[str, str]                 # unidad → pieza verificada en la caja
    tasks: list[Task] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.covered) == len(self.units) and not self.extras

    def next_task(self) -> Task | None:
        return next((t for t in self.tasks if t.state == "pendiente" and t.track is not None), None)


def matches(key: tuple[str, str | None], tr: Track) -> bool:
    color, shape = key
    return tr.color == color and (shape is None or tr.shape == shape or tr.shape == "?")


def expand(goal: Goal, world: WorldModel) -> list[Unit]:
    units: list[Unit] = []
    # las unidades con forma primero: así una pieza "cualquiera" no se queda
    # con el único cubo que pedía otra unidad
    for key, n in sorted(goal.items(), key=lambda kv: kv[0][1] is None):
        if n == "all":
            n = sum(1 for t in world.tracks.values() if matches(key, t) and t.state != "perdida")
        for i in range(1, int(n) + 1):
            units.append(Unit(key, i, int(n)))
    return units


def plan(goal: Goal, world: WorldModel, slots: list[tuple[float, float]],
         reach: Callable[[Track], str | None], blacklist: set[str], previous: list[Task] | None = None,
         box_id: str = "caja") -> KitPlan:
    units = expand(goal, world)
    in_box = [t for t in world.tracks.values() if t.state == "caja" and t.box_id in (None, box_id)]
    covered: dict[str, str] = {}
    used: set[str] = set()
    for u in units:
        for t in in_box:
            if t.id not in used and matches(u.key, t):
                covered[u.id] = t.id
                used.add(t.id)
                break
    extras = [t.id for t in in_box if t.id not in used]

    occupied = [t.pos[:2] for t in in_box]
    free_slots = [i for i, s in enumerate(slots)
                  if all(np.hypot(s[0] - p[0], s[1] - p[1]) > 0.025 for p in occupied)]

    prev = {t.unit: t for t in (previous or []) if t.unit}
    kp = KitPlan(units, covered, extras=extras)
    box_c = np.mean(np.array(slots), axis=0) if slots else np.zeros(2)
    taken: set[str] = set(used)
    reach_cache: dict[str, str | None] = {}
    for u in units:
        old = prev.get(u.id)
        task = Task(id=u.id, kind="colocar", label=u.label(), unit=u.id, attempts=old.attempts if old else 0)
        if u.id in covered:
            task.state, task.track, task.note = "completada", covered[u.id], "verificada en la caja"
            kp.tasks.append(task)
            continue
        cands = [t for t in world.tracks.values()
                 if t.state == "mesa" and t.id not in taken and t.id not in blacklist and matches(u.key, t)]
        # se mantiene la pieza elegida antes si sigue siendo válida (plan estable)
        cands.sort(key=lambda t: (0 if old and old.track == t.id else 1, float(np.hypot(*(t.pos[:2] - box_c)))))
        chosen, why = None, []
        for t in cands:
            if t.id not in reach_cache:
                reach_cache[t.id] = reach(t)
            if reach_cache[t.id] is None:
                chosen = t
                break
            why.append(f"{t.label()}: {reach_cache[t.id]}")
        if chosen is None:
            task.state = "bloqueada"
            task.note = "no queda ninguna pieza que encaje" if not cands else "; ".join(why)
            kp.missing.append(u.label())
        else:
            taken.add(chosen.id)
            task.track = chosen.id
            task.slot = free_slots.pop(0) if free_slots else None
            if task.slot is None:
                task.state, task.note = "bloqueada", "no queda sitio libre en la caja"
        kp.tasks.append(task)
    for k in extras:
        kp.tasks.append(Task(id=f"retirar:{k}", kind="retirar", label=f"retirar {world.tracks[k].label()}",
                             unit=None, track=k))
    return kp
