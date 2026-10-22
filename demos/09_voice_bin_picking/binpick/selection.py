"""Qué pieza coger: las que piden las reglas, ordenadas por lo fácil que es agarrarlas (sin Isaac).

Con una sola cámara cenital una pieza tapada o rodeada es difícil de agarrar, y coger la de
abajo de un montón la arruina. Cada candidata se puntúa con lo que *se ve*:

* altura de su cara superior (las de arriba primero);
* visibilidad (fracción de su cara superior que no tapa otra pieza);
* hueco para los dedos: la pinza cierra a lo largo de su eje x, con el dedo fijo pegado a
  una cara y 16 mm de holgura al otro lado (``picking.OPEN_MARGIN_M``); se
  comprueba, para cada una de las orientaciones posibles, que ese hueco no choca con la pared
  del contenedor ni con las vecinas;
* alcance del brazo (cinemática inversa), que se comprueba sólo para las mejor puntuadas.

Todo candidato descartado dice por qué, en español, y el panel lo enseña.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config as C
from . import picking
from .commands import SortOrder
from .instances import Instance
from .scene_gen import _square, polygon_gap
from recoverykit import skills  # noqa: E402

# medido sobre los colisionadores del SO-101 a 4–10 mm de la punta: grosor 8–10 mm, anchura 12–14 mm
FINGER_T_M = 0.010           # grosor de un dedo a lo largo del eje de cierre
FINGER_HALF_W_M = 0.0075     # semianchura del dedo (perpendicular al cierre)
JAW_OPEN_SIDE_M = picking.OPEN_MARGIN_M  # holgura de la mordaza móvil al abrir
COVER_PENETRATION_M = 0.004   # una pieza más alta que se mete así en la huella de otra la tapa de verdad
CLEAR_OK_M = 0.0005          # separación mínima para decir que cabe
CLEAR_FULL_M = 0.010         # a partir de aquí el hueco ya no mejora la puntuación
MIN_VISIBILITY = 0.35
W_HEIGHT, W_VIS, W_CLEAR, W_CENTER = 0.35, 0.25, 0.30, 0.10


@dataclass
class Grasp:
    yaws_ok: list[float]                 # giros de la pinza con sitio para los dedos, mejor primero
    clearance_m: float                   # holgura del mejor giro
    covered_by: str | None = None
    yaws_all: list[float] = field(default_factory=list)   # todos los giros, de más a menos holgura (aunque no quepan)


@dataclass
class Candidate:
    inst: Instance
    rule: int
    dest: str
    score: float
    parts: dict
    grasp: Grasp
    blocked: str | None = None

    @property
    def track(self) -> str:
        return self.inst.track or f"#{self.inst.idx}"

    def why(self) -> str:
        if self.blocked:
            return f"{self.track}: {self.blocked}"
        p = self.parts
        bits = []
        if p["altura"] > 0.8:
            bits.append("está arriba del montón")
        if p["visibilidad"] > 0.9:
            bits.append("se ve entera")
        if p["hueco"] > 0.9:
            bits.append("los dedos caben bien")
        return f"{self.track}: " + (", ".join(bits) if bits else "la mejor disponible")


def jaw_slots(center_xy: np.ndarray, yaw: float, width: float) -> list[np.ndarray]:
    """Rectángulos (4×2) que ocupan los dos dedos para un giro de la pinza (cierre a lo largo de ``yaw``)."""
    d = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-d[1], d[0]])

    def rect(a0: float, a1: float) -> np.ndarray:
        return np.array([center_xy + a * d + b * n for a, b in
                         ((a0, -FINGER_HALF_W_M), (a1, -FINGER_HALF_W_M), (a1, FINGER_HALF_W_M), (a0, FINGER_HALF_W_M))])

    fixed = rect(-(width / 2 + FINGER_T_M), -width / 2)                       # dedo fijo, pegado a la pieza
    moving = rect(width / 2 + JAW_OPEN_SIDE_M, width / 2 + JAW_OPEN_SIDE_M + FINGER_T_M)
    return [fixed, moving]


def _wall_clearance(poly: np.ndarray, rect: tuple[float, float, float, float]) -> float:
    x0, x1, y0, y1 = rect
    return float(min((poly[:, 0] - x0).min(), (x1 - poly[:, 0]).min(), (poly[:, 1] - y0).min(), (y1 - poly[:, 1]).min()))


TABLE_RECT = (-1.0, 1.0, -1.0, 1.0)      # sobre la mesa no hay paredes


def assess_grasp(inst: Instance, others: list[Instance], bin_rect: tuple[float, float, float, float],
                 size: float = C.PIECE_SIZE_M) -> Grasp:
    c = inst.pos[:2]
    bottom = (inst.top_z or 0.0) - size
    covered_by = None
    footprint = _square(c, inst.yaw, size / 2)
    neighbours = []
    for o in others:
        if o is inst or o.pos is None or o.region != inst.region:
            continue
        sq = _square(o.pos[:2], o.yaw, size / 2 + 0.001)
        if (o.top_z or 0.0) > (inst.top_z or 0.0) + 0.01 and polygon_gap(footprint, sq) < -COVER_PENETRATION_M:
            covered_by = o.track or f"#{o.idx}"
        if (o.top_z or 0.0) > bottom + 0.004:              # llega a la altura de las puntas de los dedos
            neighbours.append(sq)
    scored: list[tuple[float, float]] = []
    for yaw in skills._yaw_candidates(inst.yaw if inst.shape == "cube" else 0.0, inst.shape or "cube"):
        worst = np.inf
        for slot in jaw_slots(c, yaw, size):
            worst = min(worst, _wall_clearance(slot, bin_rect))
            for sq in neighbours:
                worst = min(worst, polygon_gap(slot, sq))
        scored.append((float(worst), yaw))
    scored.sort(key=lambda t: -t[0])
    ok = [y for g, y in scored if g >= CLEAR_OK_M]
    return Grasp(ok, max(scored[0][0], 0.0) if scored else 0.0, covered_by, [y for _g, y in scored])


def score(inst: Instance, grasp: Grasp, floor_z: float, bin_rect, size: float = C.PIECE_SIZE_M) -> tuple[float, dict]:
    height = float(np.clip(((inst.top_z or floor_z) - floor_z) / (2 * size), 0.0, 1.0))
    clear = float(np.clip(grasp.clearance_m / CLEAR_FULL_M, 0.0, 1.0))
    x0, x1, y0, y1 = bin_rect
    edge = min(inst.pos[0] - x0, x1 - inst.pos[0], inst.pos[1] - y0, y1 - inst.pos[1])
    center = float(np.clip(edge / 0.03, 0.0, 1.0))
    parts = {"altura": height, "visibilidad": float(inst.visibility), "hueco": clear, "centrado": center}
    return W_HEIGHT * height + W_VIS * inst.visibility + W_CLEAR * clear + W_CENTER * center, parts


@dataclass
class Selection:
    usable: list[Candidate] = field(default_factory=list)
    blocked: list[Candidate] = field(default_factory=list)
    pending: int = 0                         # piezas que aún tocan según las reglas (hayan o no candidatas)
    notes: list[str] = field(default_factory=list)


def rank(instances: list[Instance], order: SortOrder, scene, done: dict[int, int] | None = None,
         blacklist: set[str] | None = None, attempts: dict[str, int] | None = None,
         full: set[str] | None = None) -> Selection:
    """Candidatas que piden las reglas, ordenadas (mejor primero), más las descartadas y por qué.

    ``done[i]``: piezas ya movidas por la regla *i* (para las órdenes con cantidad);
    ``full``: cajas de destino sin sitio.
    """
    done, blacklist, attempts, full = done or {}, blacklist or set(), attempts or {}, full or set()
    b = C.container(scene, C.BIN_ID)
    floor_z = scene.table.top_z_m + b.floor_m
    bin_rect = C.inner_margin_box(scene, C.BIN_ID, 0.0)
    # piezas del contenedor y piezas caídas sobre la mesa (se pueden recuperar); las de las cajas no
    in_bin = [i for i in instances if i.region in ("origen", "mesa") and i.pos is not None]
    sel = Selection()
    taken: set[int] = set()
    for ri, r in enumerate(order.rules):
        s = r.selector
        matching = [i for i in in_bin if i.idx not in taken and (s.rest or s.matches(i.color, i.shape, i.attributes))]
        if s.rest:
            # «las demás»: lo que ninguna otra regla ha reclamado
            claimed = {i.idx for rr in order.rules if not rr.selector.rest for i in in_bin
                       if rr.selector.matches(i.color, i.shape, i.attributes)}
            matching = [i for i in matching if i.idx not in claimed]
        quota = len(matching) if s.count == "all" else max(0, int(s.count) - done.get(ri, 0))
        sel.pending += min(quota, len(matching))
        if r.dest in full:
            sel.notes.append(f"La caja {r.dest} está llena.")
            for i in matching[:quota]:
                taken.add(i.idx)
            continue
        for i in matching:
            taken.add(i.idx)
        cands = []
        for i in matching:
            rect = bin_rect if i.region == "origen" else TABLE_RECT
            g = assess_grasp(i, in_bin, rect)
            sc, parts = score(i, g, floor_z if i.region == "origen" else scene.table.top_z_m, bin_rect)
            sc -= 0.25 * attempts.get(i.track or "", 0)
            c = Candidate(i, ri, r.dest, sc, parts, g)
            if (i.track or "") in blacklist:
                c.blocked = "descartada tras varios fallos"
            elif g.covered_by:
                c.blocked = f"tiene {g.covered_by} encima"
            elif i.visibility < MIN_VISIBILITY:
                c.blocked = f"está demasiado tapada (se ve el {i.visibility * 100:.0f} %)"
            elif not g.yaws_ok:
                c.blocked = "los dedos no caben junto a la pared o las vecinas"
            cands.append(c)
        cands.sort(key=lambda c: -c.score)
        usable = [c for c in cands if not c.blocked]
        sel.usable.extend(usable[:quota] if s.count != "all" else usable)
        sel.blocked.extend(c for c in cands if c.blocked)
    sel.usable.sort(key=lambda c: -c.score)
    return sel


def first_reachable(sel: Selection, reach: Callable[[Candidate], str | None],
                    relaxed: Callable[[Candidate], str | None] | None = None) -> tuple[Candidate | None, list[str]]:
    """Primera candidata a la que llega el brazo, probando por orden; devuelve también los motivos de las que no.

    ``relaxed`` es el último recurso: si ninguna pasa la comprobación estricta, se prueban de nuevo con una más permisiva
    y, tras ellas, las que se descartaron por no caber los dedos (si se ven enteras y no tienen nada encima). Medido en
    Isaac con ``tools/bench_grasp.py``: dejar rozar una pared, 11 de 16 agarradas; sin hueco para los dedos, 7 de 19 (37 %,
    frente a ~75 % de las que el modelo admite). No es fiable, pero es mejor que dar la orden por imposible.
    Una pieza que pasa la comprobación estricta siempre va antes.
    """
    why: list[str] = []
    for c in sel.usable:
        problem = reach(c)
        if problem is None:
            return c, why
        c.blocked = f"inalcanzable: {problem}"
        why.append(f"{c.track}: {c.blocked}")
    if relaxed is not None:
        tight = [c for c in sel.blocked if c.blocked and "dedos" in c.blocked and c.grasp.yaws_all
                 and c.grasp.covered_by is None and c.inst.visibility >= 0.7]
        for c in list(sel.usable) + tight:
            if not c.grasp.yaws_ok:
                c.grasp.yaws_ok = list(c.grasp.yaws_all[:2])
            if relaxed(c) is None:
                c.blocked = None
                return c, []
    return None, why
