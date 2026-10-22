"""Un robot *lógico* para probar la máquina de estados sin Isaac. NO es la simulación física.

No hay física, ni imágenes, ni política: ``FakeRobot`` guarda dónde está cada pieza y decide si un
agarre o una colocación salen bien con una probabilidad o con un guion de resultados. Sirve para
probar lo que hace el orquestador (qué pieza elige, cómo reintenta, qué pasa con una caja llena,
cómo termina), no si el brazo coge de verdad. Cualquier número sacado de aquí es de *lógica* y así
hay que llamarlo (``docs/04_validacion.md``).

Las piezas salen de ``scene_gen``: las mismas escenas aleatorias (separadas, juntas, apiladas) que
usa el simulador, sin asentarlas con física.
"""

from __future__ import annotations

import random

import numpy as np

from . import config as C
from . import scene_gen as G
from .instances import Instance
from .orchestrator import Observation, PickResult, PlaceResult
from .selection import Candidate

KIND = "lógico (sin física)"


class FakeRobot:
    def __init__(self, cfg: C.BinPickCfg | None = None, seed: int = 0, n_pieces: int | None = None,
                 mode: str | None = None, p_grasp: float = 1.0, p_place: float = 1.0,
                 pick_script: list[str] | None = None, place_script: list[str] | None = None, capacity: int = 3):
        self.cfg = cfg or C.BinPickCfg()
        self.scene = C.build_scene(self.cfg)
        self.rng = random.Random(seed)
        self.p_grasp, self.p_place = p_grasp, p_place
        self.pick_script = list(pick_script or [])
        self.place_script = list(place_script or [])
        self._capacity = capacity
        self.log: list[tuple] = []
        self.pieces: dict[str, dict] = {}
        self.held: str | None = None
        spec = G.sample_scene(self.cfg, seed, n_pieces=n_pieces, mode=mode)
        self.spec = spec
        placed = G.approximate_settle(spec, self.scene)
        for k, d in enumerate(spec.drops):
            pos, quat, layer = placed[d.piece_id]
            self.pieces[f"P{k + 1}"] = {"id": d.piece_id, "color": d.color, "shape": d.shape, "region": "origen",
                                       "pos": pos, "yaw": G.yaw_of(quat), "layer": layer}

    # ------------------------------------------------------------- RobotPort
    def home(self) -> None:
        self.log.append(("home",))

    def capacity(self, dest: str) -> int:
        return self._capacity

    def reach(self, cand: Candidate) -> str | None:
        return None

    def observe(self, why: str) -> Observation:
        self.log.append(("observe", why))
        out = []
        for track, p in self.pieces.items():
            if p["region"] in ("aire", "perdida"):
                continue
            top_z = float(p["pos"][2] + C.PIECE_SIZE_M / 2)
            # una pieza queda tapada si tiene otra encima, cerca
            covered = any(o is not p and o["region"] == p["region"] == "origen" and o["layer"] > p["layer"]
                          and np.linalg.norm(o["pos"][:2] - p["pos"][:2]) < 0.014 for o in self.pieces.values())
            inst = Instance(idx=len(out), mask=np.zeros((2, 2), bool), color=p["color"], shape=p["shape"],
                            pos=p["pos"].copy(), top_z=top_z, yaw=p["yaw"], visibility=0.5 if covered else 1.0,
                            region=p["region"], source="fake", track=track, width_m=C.PIECE_SIZE_M)
            out.append(inst)
        return Observation(out)

    def _next(self, script: list[str], p: float) -> str:
        if script:
            return script.pop(0)
        return "ok" if self.rng.random() < p else "fallo"

    def pick(self, cand: Candidate, strategy: dict, policy: str) -> PickResult:
        self.log.append(("pick", cand.track, policy, dict(strategy)))
        res = self._next(self.pick_script, self.p_grasp)
        if res == "ok":
            self.held = cand.track
            self.pieces[cand.track]["region"] = "aire"
            return PickResult(True, "ok", policy=policy, steps=150)
        if res == "movida":
            self.pieces[cand.track]["pos"] = self.pieces[cand.track]["pos"] + np.array([0.02, 0.0, 0.0])
        status = res if res in ("vacío", "deslizada", "movida") else "vacío"
        return PickResult(False, status, f"fallo simulado: {status}", policy=policy, steps=90)

    def place(self, cand: Candidate) -> PlaceResult:
        self.log.append(("place", cand.track, cand.dest))
        track = self.held
        self.held = None
        if track is None:
            return PlaceResult(False, "deslizada", "no llevaba nada")
        p = self.pieces[track]
        res = self._next(self.place_script, self.p_place)
        load = sum(1 for q in self.pieces.values() if q["region"] == cand.dest)
        if res == "ok" and load >= self._capacity:
            res = "fuera"
        if res == "ok":
            c = C.container(self.scene, C.DEST_IDS[cand.dest])
            p["region"] = cand.dest
            p["pos"] = np.array([c.center_xy_m[0], c.center_xy_m[1], self.scene.table.top_z_m + c.floor_m + C.PIECE_SIZE_M / 2])
            return PlaceResult(True)
        if res == "vuelve":
            p["region"], p["layer"] = "origen", 0
            return PlaceResult(False, "deslizada", "la pieza volvió al contenedor")
        if res == "perdida":
            p["region"] = "perdida"
            p["pos"] = np.array([5.0, 5.0, 0.0])
            return PlaceResult(False, "abortado", "la pieza desapareció")
        p["region"] = "mesa"
        p["pos"] = np.array([0.12, 0.0, self.scene.table.top_z_m + C.PIECE_SIZE_M / 2])
        return PlaceResult(False, "deslizada", "cayó fuera de la caja")

    # ----------------------------------------------------------- consultas
    def where(self) -> dict[str, str]:
        return {t: p["region"] for t, p in self.pieces.items()}

    def truth_counts(self, dest: str, color: str | None = None) -> int:
        return sum(1 for p in self.pieces.values() if p["region"] == dest and (color is None or p["color"] == color))
