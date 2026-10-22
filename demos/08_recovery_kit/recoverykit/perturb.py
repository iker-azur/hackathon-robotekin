"""Perturbaciones físicas: lo que el mundo le hace al robot sin avisar.

Sólo fuerzas e impulsos sobre las piezas y un fallo del actuador de la pinza;
nunca se escribe una pose. El perturbador es parte del "mundo", no del agente:
puede consultar la verdad del simulador para saber qué pieza empujar, pero el
agente no recibe nada de aquí; tiene que darse cuenta con la cámara y la pinza.

Se disparan en eventos de fase del agente (empieza la aproximación, empieza el
cierre, la subida, el transporte), con magnitudes y direcciones aleatorias de
una semilla, no en posiciones fijas.

Tipos:
- ``empujar`` (A): al empezar la aproximación, la pieza objetivo se desliza 3–6 cm.
- ``golpe_cierre`` (B): mientras la pinza cierra, la pieza sale de entre los dedos.
- ``tirón_subida`` (B): al subir, un tirón creciente arranca la pieza de la pinza.
- ``fallo_pinza`` (C): sobre la pared de la caja la pinza se abre sola y la pieza cae fuera.
- ``sacar_caja`` (C): una pieza ya colocada sale despedida de la caja.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

KINDS = {
    "empujar": "aproximación",
    "golpe_cierre": "cierre",
    "tirón_subida": "subida",
    "fallo_pinza": "transporte",
    "sacar_caja": "colocada",
}
LABELS = {
    "empujar": "alguien mueve la pieza antes de que el robot la coja",
    "golpe_cierre": "un golpe saca la pieza de la pinza mientras cierra",
    "tirón_subida": "un tirón arranca la pieza de la pinza al levantarla",
    "fallo_pinza": "la pinza se abre sola encima del borde de la caja",
    "sacar_caja": "alguien saca una pieza de la caja",
}
G = 9.81
SQUEEZE_MEASURED_N = 3.0   # fuerza normal de la pinza medida (results/grip_capacity_*.json)


@dataclass
class Armed:
    kind: str
    fired: bool = False


@dataclass
class Active:
    piece: str
    force: np.ndarray
    steps_left: int
    stop_after_m: float | None = None
    start: np.ndarray | None = None
    relative_to: str | None = None      # eslabón del robot: el desplazamiento se mide respecto a él
    ramp_steps: int = 0                 # la fuerza crece de 0 al valor nominal en estos pasos
    age: int = 0
    wait_above_z: float | None = None   # no empieza hasta que la pieza supera esta altura
    hold_on_slip_m: float | None = None  # al empezar a deslizar, la fuerza deja de crecer
    frozen_scale: float | None = None


@dataclass
class Perturber:
    cell: object
    seed: int = 0
    armed: list[Armed] = field(default_factory=list)
    active: list[Active] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    _rng: random.Random | None = None
    _watch_rim: dict | None = None
    _override_steps: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self.cell.step_hooks.append(self.step)

    # ------------------------------------------------------------------ API
    def arm(self, kind: str) -> None:
        if kind not in KINDS:
            raise ValueError(kind)
        self.armed.append(Armed(kind))

    def pending(self) -> list[str]:
        return [a.kind for a in self.armed if not a.fired]

    def clear(self) -> None:
        self.armed.clear()
        for a in self.active:
            self.cell.clear_force(a.piece)
        self.active.clear()
        self._watch_rim = None
        self._override_steps = 0
        self.cell.gripper_override = None

    def notify(self, event: str, ctx: dict) -> None:
        """El agente avisa de una fase; ``ctx`` lleva lo que el agente cree (posición objetivo, giro…)."""
        for a in self.armed:
            if a.fired or KINDS[a.kind] != event:
                continue
            a.fired = True
            getattr(self, "_" + a.kind)(ctx)

    # ----------------------------------------------------------- utilidades
    def _pos(self, name: str) -> np.ndarray:
        return self.cell.truth(name)["pos"]

    def _nearest_piece(self, xy, max_dist: float = 0.05) -> str | None:
        best, best_d = None, max_dist
        for name in self.cell.pieces:
            p = self._pos(name)
            if p[2] < self.cell.cfg.table.top_z_m - 0.1:
                continue
            d = math.hypot(p[0] - xy[0], p[1] - xy[1])
            if d < best_d:
                best, best_d = name, d
        return best

    def _record(self, kind: str, piece: str | None, **info) -> None:
        self.log.append({"t": round(self.cell.sim_time, 3), "kind": kind, "label": LABELS[kind], "piece": piece,
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}})

    def _mu_table(self) -> float:
        cfg = self.cell.cfg
        return (cfg.physics.piece_material.dynamic_friction + cfg.table.material.dynamic_friction) / 2

    def _slide_force(self, mass: float, distance: float, push_s: float) -> float:
        """Fuerza horizontal constante durante ``push_s`` que desliza ``distance`` (Coulomb).

        Con a = F/m − μg: d = a·T²/2 + (a·T)²/(2μg); se despeja a.
        """
        mu = self._mu_table()
        qa, qb, qc = push_s ** 2 / (2 * mu * G), push_s ** 2 / 2, -distance
        a = (-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
        return mass * (a + mu * G)

    def _reach_score(self, xy) -> float:
        """0 si (x, y) está dentro de la corona alcanzable; si no, cuánto se sale (m)."""
        area = self.cell.cfg.work_area
        r = math.hypot(*xy)
        ang = math.degrees(math.atan2(xy[1], xy[0]))
        out_r = max(0.0, area.r_m[0] - r, r - area.r_m[1])
        out_a = max(0.0, area.angle_deg[0] - ang, ang - (area.angle_deg[1] + 30)) * math.pi / 180 * r
        return out_r + out_a

    def _clearance(self, piece: str, xy) -> float:
        """Distancia libre a otras piezas y a la caja (m)."""
        d = 1.0
        for name in self.cell.pieces:
            if name == piece:
                continue
            p = self._pos(name)
            if p[2] > self.cell.cfg.table.top_z_m - 0.1:
                d = min(d, math.hypot(p[0] - xy[0], p[1] - xy[1]))
        for c in self.cell.cfg.containers:
            dx = max(0.0, abs(xy[0] - c.center_xy_m[0]) - c.outer_m[0] / 2)
            dy = max(0.0, abs(xy[1] - c.center_xy_m[1]) - c.outer_m[1] / 2)
            d = min(d, math.hypot(dx, dy) + 0.0125)
        return d

    def _best_destination(self, piece: str, p, dists: tuple[float, float], tries: int = 160):
        """Destino aleatorio alcanzable y despejado; si no hay uno perfecto, el mejor."""
        best, best_cost = None, float("inf")
        for _ in range(tries):
            dist = self._rng.uniform(*dists)
            ang = self._rng.uniform(-math.pi, math.pi)
            dest = (p[0] + dist * math.cos(ang), p[1] + dist * math.sin(ang))
            cost = 10 * self._reach_score(dest) + max(0.0, 0.045 - self._clearance(piece, dest))
            if cost < best_cost:
                best, best_cost = (dist, ang), cost
            if cost == 0.0:
                break
        return best, best_cost

    # ------------------------------------------------------------- perturbaciones
    def _empujar(self, ctx: dict) -> None:
        piece = self._nearest_piece(ctx["target_xy"])
        if piece is None:
            self._record("empujar", None, note="no había pieza cerca del objetivo")
            return
        p = self._pos(piece)
        (dist, ang), cost = self._best_destination(piece, p, (0.03, 0.06))
        push_s = 0.2
        f = self._slide_force(self.cell.cfg.piece(piece).mass_kg, dist, push_s)
        self.active.append(Active(piece, np.array([f * math.cos(ang), f * math.sin(ang), 0.0]),
                                  int(push_s / self.cell.cfg.physics.dt)))
        self._record("empujar", piece, distance_m=dist, direction_deg=math.degrees(ang), force_n=f,
                     destination_ok=cost == 0.0)

    def _golpe_cierre(self, ctx: dict) -> None:
        piece = self._nearest_piece(ctx["target_xy"])
        if piece is None:
            return
        # perpendicular al eje de cierre: la pieza sale por el hueco entre los dedos
        yaw = ctx.get("yaw", 0.0) + math.pi / 2
        p = self._pos(piece)
        dist = self._rng.uniform(0.04, 0.055)
        ahead = (p[0] + dist * math.cos(yaw), p[1] + dist * math.sin(yaw))
        behind = (p[0] - dist * math.cos(yaw), p[1] - dist * math.sin(yaw))
        sign = 1.0 if self._reach_score(ahead) <= self._reach_score(behind) else -1.0
        f = self._slide_force(self.cell.cfg.piece(piece).mass_kg, dist, 0.12)
        force = np.array([sign * f * math.cos(yaw), sign * f * math.sin(yaw), 0.0])
        self.active.append(Active(piece, force, int(0.12 / self.cell.cfg.physics.dt)))
        self._record("golpe_cierre", piece, distance_m=dist, force_n=f)

    def _tirón_subida(self, ctx: dict) -> None:
        piece = self._nearest_piece(ctx["target_xy"], max_dist=0.04)
        if piece is None:
            return
        cfg = self.cell.cfg
        # tirón creciente (hacia abajo y algo de lado) hasta que la pieza se suelta de la
        # pinza: se mide el deslizamiento respecto a la pinza, no respecto al mundo,
        # porque la pinza está subiendo
        # capacidad medida de la sujeción: F ≈ N·(μ_mordaza + μ_pieza), N ≈ 3.0 N
        # (tools/probe_grip_capacity.py); el tirón llega a 1.6–2.2 veces eso
        capacity = SQUEEZE_MEASURED_N * (cfg.robot.jaw_material.static_friction
                                         + cfg.physics.piece_material.static_friction)
        peak = capacity * self._rng.uniform(1.6, 2.2)
        ang = self._rng.uniform(-math.pi, math.pi)
        force = np.array([0.3 * peak * math.cos(ang), 0.3 * peak * math.sin(ang), -peak])
        steps = int(0.4 / cfg.physics.dt)
        # empieza cuando la pieza ya ha despegado 2 cm (antes la mesa la sostiene) y se
        # para cuando ha salido de entre los dedos (~35 mm de deslizamiento respecto
        # a la pinza); si se parase antes, la fricción la volvería a sujetar más abajo
        z_lift = cfg.table.top_z_m + cfg.piece(piece).height / 2 + 0.02
        # al empezar a deslizar, el tirón deja de crecer (como una mano): con fricción
        # dinámica menor que la estática la pieza sale despacio, no disparada
        self.active.append(Active(piece, force, steps, stop_after_m=0.035, relative_to="gripper",
                                  ramp_steps=steps, wait_above_z=z_lift, hold_on_slip_m=0.003))
        self._record("tirón_subida", piece, peak_force_n=peak)

    def _fallo_pinza(self, ctx: dict) -> None:
        piece = self._nearest_piece(ctx["target_xy"], max_dist=0.08)
        self._watch_rim = {"piece": piece, "box": self.cell.cfg.containers[0], "open_q": ctx.get("q_open", 0.5)}
        self._record("fallo_pinza", piece, note="armado: se abrirá al cruzar la pared de la caja")

    def _sacar_caja(self, ctx: dict) -> None:
        box = self.cell.cfg.containers[0]
        inside = [n for n in self.cell.pieces
                  if abs(self._pos(n)[0] - box.center_xy_m[0]) < box.inner_m[0] / 2
                  and abs(self._pos(n)[1] - box.center_xy_m[1]) < box.inner_m[1] / 2]
        if not inside:
            return
        piece = self._rng.choice(inside)
        p = self._pos(piece)
        # tiro parabólico: sube por encima del borde antes de llegar a la pared y cae
        # dentro de la zona alcanzable; se elige la dirección que mejor lo cumple
        vz = 1.4
        t_air = 2 * vz / G
        rim = box.outer_m[2] - box.floor_m + 0.006
        best, best_cost = None, float("inf")
        for k in range(24):
            ang = k * math.pi / 12 + self._rng.uniform(-0.1, 0.1)
            vh = self._rng.uniform(0.25, 0.35)
            d_wall = self._dist_to_wall(p, ang, box)
            t_wall = d_wall / vh
            z_wall = vz * t_wall - 0.5 * G * t_wall * t_wall
            land = (p[0] + vh * t_air * math.cos(ang), p[1] + vh * t_air * math.sin(ang))
            cost = (0.0 if z_wall > rim else 1.0) + 10 * self._reach_score(land) \
                + max(0.0, 0.04 - self._clearance(piece, land))
            if cost < best_cost:
                best, best_cost = (ang, vh), cost
        ang, vh = best
        mass = self.cell.cfg.piece(piece).mass_kg
        duration = 0.04
        force = np.array([math.cos(ang) * mass * vh / duration, math.sin(ang) * mass * vh / duration,
                          mass * (vz / duration + G)])
        self.active.append(Active(piece, force, int(duration / self.cell.cfg.physics.dt)))
        self._record("sacar_caja", piece, speed_m_s=vh, direction_deg=math.degrees(ang), landing_ok=best_cost == 0.0)

    @staticmethod
    def _dist_to_wall(p, ang: float, box) -> float:
        """Distancia horizontal desde p hasta la cara interior de la pared en la dirección ang."""
        c, s = math.cos(ang), math.sin(ang)
        hx, hy = box.inner_m[0] / 2, box.inner_m[1] / 2
        dx, dy = p[0] - box.center_xy_m[0], p[1] - box.center_xy_m[1]
        ts = []
        if abs(c) > 1e-6:
            ts.append(((hx if c > 0 else -hx) - dx) / c)
        if abs(s) > 1e-6:
            ts.append(((hy if s > 0 else -hy) - dy) / s)
        return max(0.0, min(t for t in ts if t >= 0) - 0.0125) if ts else 0.0

    # ----------------------------------------------------------------- paso
    def step(self) -> None:
        if self._watch_rim is not None and self._watch_rim["piece"] is not None:
            w = self._watch_rim
            p = self._pos(w["piece"])
            b = w["box"]
            dx = abs(p[0] - b.center_xy_m[0]) - b.outer_m[0] / 2
            dy = abs(p[1] - b.center_xy_m[1]) - b.outer_m[1] / 2
            # centro de la pieza sobre la pared (o justo fuera), dentro del otro eje
            if -b.wall_m - 0.004 <= max(dx, dy) <= 0.006 and min(dx, dy) <= 0.0:
                self.cell.gripper_override = w["open_q"]
                self._override_steps = int(0.5 / self.cell.cfg.physics.dt)
                out = np.array([p[0] - b.center_xy_m[0], p[1] - b.center_xy_m[1]])
                out = out / (np.linalg.norm(out) + 1e-9)
                mass = self.cell.cfg.piece(w["piece"]).mass_kg
                self.active.append(Active(w["piece"], np.array([out[0], out[1], 0.0]) * mass * 6.0,
                                          int(0.08 / self.cell.cfg.physics.dt)))
                self._record("fallo_pinza", w["piece"], note="la pinza se ha abierto sobre la pared")
                self._watch_rim = None
        if self._override_steps > 0:
            self._override_steps -= 1
            if self._override_steps == 0:
                self.cell.gripper_override = None
        for a in list(self.active):
            if a.wait_above_z is not None:
                if self._pos(a.piece)[2] < a.wait_above_z:
                    continue
                a.wait_above_z = None
                if a.relative_to:
                    a.start = self._pos(a.piece) - self.cell.body_pose(a.relative_to)[0]
            done = a.steps_left <= 0
            slip = 0.0
            if a.stop_after_m is not None and a.start is not None:
                ref = self.cell.body_pose(a.relative_to)[0] if a.relative_to else 0.0
                slip = float(np.linalg.norm(self._pos(a.piece) - ref - a.start))
                done = done or slip > a.stop_after_m
            if done:
                self.cell.clear_force(a.piece)
                self.active.remove(a)
                continue
            scale = min(1.0, (a.age + 1) / a.ramp_steps) if a.ramp_steps else 1.0
            if a.hold_on_slip_m is not None:
                if a.frozen_scale is None and slip > a.hold_on_slip_m:
                    a.frozen_scale = scale
                if a.frozen_scale is not None:
                    scale = a.frozen_scale
            self.cell.apply_force(a.piece, tuple(float(v) * scale for v in a.force))
            a.steps_left -= 1
            a.age += 1
