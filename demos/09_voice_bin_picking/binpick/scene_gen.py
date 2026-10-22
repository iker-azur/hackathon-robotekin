"""Escenas aleatorias del contenedor: qué piezas, de qué color y cómo se dejan caer (sin Isaac).

Las posiciones finales **no se escriben**: se dejan caer las piezas de una en una y la
física las asienta (gravedad, fricción, choques). Este módulo sólo decide las caídas
(``sample_scene``), calcula dónde empieza cada una dada la posición ya asentada de
las anteriores (``resolve_drop``) y, con las poses finales, mide qué configuración ha
salido (``analyze_layout``) y si es válida (``validate_settled``). Así «separadas,
pegadas, solapadas y parcialmente apiladas» son resultados que se miden, no
promesas del generador.

Modos de una pieza: ``free`` (cae en un punto libre), ``touch`` (cae junto a otra,
casi tocándola), ``stack`` (cae sobre otra, descentrada: acaba apilada o ladeada).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import config as C

MODES = ("separadas", "juntas", "apiladas", "mixto")
WALL_MARGIN_M = 0.022            # el centro de una pieza que cae libre queda a esta distancia de la pared (los dedos miden 10 mm)
PILE_WALL_MARGIN_M = 0.012       # holgura extra (más allá de media pieza) para las que caen junto a otra
SETTLE_SPEED_M_S = 0.01
TOUCH_GAP_M = 0.0015             # distancia entre piezas a la misma altura que cuenta como «tocándose»


@dataclass
class DropSpec:
    piece_id: str
    color: str
    shape: str
    mode: str = "free"                    # free | touch | stack
    ref: str | None = None                # pieza de referencia (touch/stack)
    xy: tuple[float, float] | None = None  # sólo free
    angle: float = 0.0                    # touch: dirección respecto a ref; stack: dirección del descentrado
    offset_m: float = 0.0                 # stack: descentrado
    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    height_m: float = 0.03                # free: altura de caída sobre el suelo


@dataclass
class SceneSpec:
    seed: int
    mode: str
    drops: list[DropSpec] = field(default_factory=list)

    def colors(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.drops:
            out[d.color] = out.get(d.color, 0) + 1
        return out

    def as_dict(self) -> dict:
        return {"seed": self.seed, "mode": self.mode, "n": len(self.drops), "colors": self.colors(),
                "drops": [{"id": d.piece_id, "mode": d.mode, "ref": d.ref} for d in self.drops]}


# --------------------------------------------------------------------- muestreo
def sample_scene(cfg: C.BinPickCfg, seed: int, n_pieces: int | None = None, mode: str | None = None,
                 min_pieces: int = 4, max_pieces: int = 6) -> SceneSpec:
    """Escena reproducible: número de piezas, reparto de colores y modo de caída de cada una."""
    rng = random.Random(seed)
    scene = C.build_scene(cfg)
    bx0, bx1, by0, by1 = C.inner_margin_box(scene, C.BIN_ID, WALL_MARGIN_M)
    # «juntas» (todas pegadas en cadena) es casi un atasco con dedos de 10 mm: sólo si se pide (--layout juntas)
    mode = mode or rng.choice(("separadas", "separadas", "mixto", "mixto", "mixto", "apiladas"))
    if mode not in MODES:
        raise ValueError(f"modo desconocido: {mode} (usa {', '.join(MODES)})")
    n = n_pieces if n_pieces is not None else rng.randint(min_pieces, max_pieces)
    n = max(1, min(n, C.MAX_PIECES))

    pool: dict[str, list[tuple[str, str]]] = {c: [] for c in C.COLORS}
    for pid, color, shape in C.piece_pool(cfg):
        pool[color].append((pid, shape))
    for color in pool:
        rng.shuffle(pool[color])
    weights = [rng.random() + 0.15 for _ in C.COLORS]            # reparto de colores distinto en cada escena
    chosen: list[tuple[str, str, str]] = []
    guard = 0
    while len(chosen) < n and guard < 200:
        guard += 1
        color = rng.choices(C.COLORS, weights=weights)[0]
        if pool[color]:
            pid, shape = pool[color].pop()
            chosen.append((pid, color, shape))
    rng.shuffle(chosen)

    spec = SceneSpec(seed=seed, mode=mode)
    placed: list[tuple[str, tuple[float, float]]] = []
    for i, (pid, color, shape) in enumerate(chosen):
        m = _piece_mode(mode, i, rng)
        d = DropSpec(pid, color, shape, yaw=rng.uniform(-math.pi, math.pi))
        if rng.random() < 0.5:                                   # algunas caen un poco ladeadas
            d.roll, d.pitch = (math.radians(rng.uniform(-12, 12)) for _ in range(2))
        if m != "free" and placed:
            ref_id, _ = rng.choice(placed)
            d.mode, d.ref = m, ref_id
            if m == "touch":
                # junto a otra a lo largo de y (el eje largo del contenedor), donde queda sitio para los dedos de las dos
                d.angle = rng.choice((-1, 1)) * math.pi / 2 + rng.uniform(-0.6, 0.6)
            else:
                d.angle = rng.uniform(-math.pi, math.pi)
                d.offset_m = rng.uniform(0.004, 0.35 * C.PIECE_SIZE_M)
        else:
            d.mode = "free"
            d.xy = _free_xy(placed, (bx0, bx1, by0, by1), rng, 0.045 if mode in ("separadas", "mixto") else 0.03)
            d.height_m = rng.uniform(0.012, 0.05)
        placed.append((pid, d.xy or (0.0, 0.0)))
        spec.drops.append(d)
    return spec


def _piece_mode(mode: str, i: int, rng: random.Random) -> str:
    if i == 0 or mode == "separadas":
        return "free"
    if mode == "juntas":
        return "touch"
    if mode == "apiladas":
        return "stack" if rng.random() < 0.7 else "free"
    return rng.choices(("free", "touch", "stack"), weights=(0.55, 0.25, 0.20))[0]


def _free_xy(placed, box, rng: random.Random, min_gap: float) -> tuple[float, float]:
    x0, x1, y0, y1 = box
    best, best_d = None, -1.0
    for _ in range(60):
        p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
        d = min((math.hypot(p[0] - q[0], p[1] - q[1]) for _id, q in placed if q != (0.0, 0.0)), default=1.0)
        if d >= min_gap:
            return p
        if d > best_d:
            best, best_d = p, d
    return best


# ---------------------------------------------------------------- resolver caídas
def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def resolve_drop(d: DropSpec, settled: dict[str, np.ndarray], scene, cfg: C.BinPickCfg | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Posición y orientación iniciales de una caída, dadas las piezas ya asentadas (``id → xyz``)."""
    s = C.PIECE_SIZE_M
    b = C.container(scene, C.BIN_ID)
    floor_z = scene.table.top_z_m + b.floor_m
    x0, x1, y0, y1 = C.inner_margin_box(scene, C.BIN_ID, s / 2 + PILE_WALL_MARGIN_M)
    ref = settled.get(d.ref) if d.ref else None
    if d.mode == "free" or ref is None:
        x, y = d.xy if d.xy is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
        z = floor_z + s / 2 + d.height_m
    elif d.mode == "touch":
        gap = s + 0.0008
        x, y = ref[0] + gap * math.cos(d.angle), ref[1] + gap * math.sin(d.angle)
        z = floor_z + s / 2 + 0.006
    else:  # stack: sobre la referencia, descentrada
        x, y = ref[0] + d.offset_m * math.cos(d.angle), ref[1] + d.offset_m * math.sin(d.angle)
        z = float(ref[2]) + s + 0.006
    x, y = min(max(x, x0), x1), min(max(y, y0), y1)
    return np.array([x, y, z]), quat_from_rpy(d.roll, d.pitch, d.yaw)


# ------------------------------------------------------------------- geometría
def _square(c: np.ndarray, yaw: float, half: float) -> np.ndarray:
    r = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    pts = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return pts @ r.T + c[:2]


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_gap(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia entre dos polígonos convexos (negativa = solapan, en valor de la penetración mínima)."""
    best_sep = -np.inf
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([e[1], -e[0]])
            n = n / (np.linalg.norm(n) + 1e-12)
            a, b = p @ n, q @ n
            best_sep = max(best_sep, max(a.min() - b.max(), b.min() - a.max()))
    if best_sep <= 0:
        return float(best_sep)
    d = min(_seg_dist(v, poly[i], poly[(i + 1) % len(poly)]) for poly, other in ((p, q), (q, p))
            for v in other for i in range(len(poly)))
    return float(d)


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_of(quat: np.ndarray) -> float:
    """Ángulo (rad) entre el eje z de la pieza y la vertical."""
    w, x, y, z = quat
    return math.acos(float(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


@dataclass
class LayoutStats:
    n: int
    touching_pairs: int
    stacked: int                    # piezas que tienen otra debajo
    isolated: int                   # piezas sin ninguna otra a menos de ~1 cm
    mean_nn_gap_m: float
    max_tilt_deg: float
    categories: list[str]

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def analyze_layout(poses: dict[str, tuple[np.ndarray, np.ndarray]], size: float = C.PIECE_SIZE_M) -> LayoutStats:
    """Qué configuración ha salido: pegadas, apiladas, aisladas."""
    ids = list(poses)
    half = size / 2
    touching = 0
    has_below: set[str] = set()
    near: dict[str, float] = {i: np.inf for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            (pa, qa), (pb, qb) = poses[ids[a]], poses[ids[b]]
            gap = polygon_gap(_square(pa, yaw_of(qa), half), _square(pb, yaw_of(qb), half))
            dz = float(pa[2] - pb[2])
            if abs(dz) > 0.6 * size and gap < 0.004:
                has_below.add(ids[a] if dz > 0 else ids[b])
            elif abs(dz) <= 0.6 * size:
                if gap < TOUCH_GAP_M:
                    touching += 1
                near[ids[a]] = min(near[ids[a]], max(gap, 0.0))
                near[ids[b]] = min(near[ids[b]], max(gap, 0.0))
    isolated = sum(1 for i in ids if near[i] > 0.01 and i not in has_below)
    finite = [v for v in near.values() if np.isfinite(v)]
    tilt = max((math.degrees(tilt_of(q)) for _p, q in poses.values()), default=0.0)
    cats = []
    if isolated:
        cats.append("separadas")
    if touching:
        cats.append("tocándose")
    if has_below:
        cats.append("apiladas")
    return LayoutStats(len(ids), touching, len(has_below), isolated, float(np.mean(finite)) if finite else 0.0, tilt, cats)


def validate_settled(poses: dict[str, tuple[np.ndarray, np.ndarray]], scene, speeds: dict[str, float] | None = None
                     ) -> list[str]:
    """Motivos por los que una escena asentada no vale (lista vacía = válida)."""
    s = C.PIECE_SIZE_M
    b = C.container(scene, C.BIN_ID)
    x0, x1, y0, y1 = C.inner_margin_box(scene, C.BIN_ID, 0.0)
    floor_z = scene.table.top_z_m + b.floor_m
    problems = []
    for pid, (p, q) in poses.items():
        if not (x0 - 0.003 <= p[0] <= x1 + 0.003 and y0 - 0.003 <= p[1] <= y1 + 0.003):
            problems.append(f"{pid} está fuera del contenedor")
        if p[2] > floor_z + 2.6 * s:
            problems.append(f"{pid} está demasiado alta (apoyada en la pared o en un montón de tres)")
        if p[2] < floor_z + 0.3 * s:
            problems.append(f"{pid} atraviesa el suelo")
        if speeds and speeds.get(pid, 0.0) > SETTLE_SPEED_M_S:
            problems.append(f"{pid} no ha terminado de asentarse")
    return problems


def approximate_settle(spec: SceneSpec, scene) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Dónde quedaría cada pieza si cae recta, SIN física: ``id → (posición, orientación, capa)``.

    Sólo una caída ``stack`` sube de capa (sobre la de su referencia). Es para los simuladores de prueba y para
    estimar la escena sin Isaac; las escenas reales las asienta la física (``cell.BinCell.reset_scene``).
    """
    s = C.PIECE_SIZE_M
    b = C.container(scene, C.BIN_ID)
    floor_z = scene.table.top_z_m + b.floor_m
    settled: dict[str, np.ndarray] = {}
    layers: dict[str, int] = {}
    out: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for d in spec.drops:
        pos, q = resolve_drop(d, settled, scene)
        layer = min(layers.get(d.ref, 0) + 1, 2) if d.mode == "stack" and d.ref in settled else 0
        pos[2] = floor_z + s / 2 + layer * s
        settled[d.piece_id] = pos
        layers[d.piece_id] = layer
        yaw = yaw_of(q)
        out[d.piece_id] = (pos.copy(), np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]), layer)
    return out
