"""La escena física en Isaac Sim: contenedor, dos cajas, piezas que caen de verdad y se asientan.

Reutiliza ``recoverykit.workcell`` (el taller validado del demo 08: robot, mesa, cajas, cámara, vigilante
de violaciones físicas) y le añade lo propio de este demo:

* **un conjunto fijo de piezas** (3 colores × cubos y cilindros) creadas al arrancar, porque Isaac no añade
  cuerpos con la escena en marcha; en cada reinicio se *activan* sólo las de la escena y el resto se aparca;
* **caída y asentamiento**: las posiciones finales no se escriben. Cada pieza se suelta (pose inicial de
  ``scene_gen``) y se deja caer de una en una hasta que queda quieta; la física decide cómo quedan (juntas,
  apiladas, ladeadas). Se valida el resultado (dentro del contenedor, sin atravesar nada, quieta) y, si no vale,
  se repite con otra caída (semilla derivada);
* la **verdad del simulador** (poses, regiones) sólo para auditar y etiquetar: el agente no la usa.

**No se ha ejecutado todavía en Isaac** (no se lanzan simulaciones desde la revisión). ``tools/check_scene_randomization.py``
la ejercita y escribe las estadísticas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import scene_gen as G
from .segmentation import PieceTruth

MAX_DROP_SETTLE_S = 0.9
MAX_FINAL_SETTLE_S = 2.5
DROP_SPEED_M_S = 0.02


@dataclass
class ResetResult:
    spec: G.SceneSpec
    poses: dict[str, tuple[np.ndarray, np.ndarray]]
    stats: G.LayoutStats
    problems: list[str] = field(default_factory=list)
    attempts: int = 1
    sim_s: float = 0.0

    def as_dict(self) -> dict:
        return {"seed": self.spec.seed, "mode": self.spec.mode, "n": len(self.spec.drops), "colors": self.spec.colors(),
                "layout": self.stats.as_dict(), "problems": self.problems, "attempts": self.attempts,
                "settle_s": round(self.sim_s, 2)}


class BinCell:
    def __init__(self, cell, scene, cfg: C.BinPickCfg):
        self.cell, self.scene, self.cfg = cell, scene, cfg
        self.all_pieces = dict(cell.pieces)
        self.spec: G.SceneSpec | None = None

    @classmethod
    def build(cls, cfg: C.BinPickCfg, sim) -> "BinCell":
        from recoverykit import workcell

        scene = C.build_scene(cfg)
        return cls(workcell.build(scene, sim), scene, cfg)

    # -------------------------------------------------------------- escena
    def reset_scene(self, spec: G.SceneSpec, max_attempts: int = 5) -> ResetResult:
        """Aparca todo, coloca el brazo en su postura de observación y deja caer las piezas de ``spec``."""
        sub, attempt = spec, 0
        while True:
            res = self._drop(sub)
            res.attempts = attempt + 1
            if not res.problems or attempt + 1 >= max_attempts:
                self.spec = res.spec
                return res
            attempt += 1
            sub = G.sample_scene(self.cfg, spec.seed * 1000 + attempt, n_pieces=len(spec.drops), mode=spec.mode)

    def _drop(self, spec: G.SceneSpec) -> ResetResult:
        cell = self.cell
        cell.watchdog.disarm()
        cell.pieces = self.all_pieces
        cell.reset_layout({}, settle_s=0.1)           # aparca todas las piezas y deja el brazo en su postura inicial
        cell.watchdog.disarm()
        active = {d.piece_id: self.all_pieces[d.piece_id] for d in spec.drops}
        cell.pieces = active                          # el resto no se actualiza: menos trabajo por paso de física
        t0 = cell.sim_time
        settled: dict[str, np.ndarray] = {}
        for d in spec.drops:
            pos, quat = G.resolve_drop(d, settled, self.scene)
            self._write_pose(active[d.piece_id], pos, quat)
            self._settle({d.piece_id}, MAX_DROP_SETTLE_S, DROP_SPEED_M_S)
            settled[d.piece_id] = cell.truth(d.piece_id)["pos"]
        self._settle(set(active), MAX_FINAL_SETTLE_S, G.SETTLE_SPEED_M_S)
        poses = self.truth_poses()
        speeds = {pid: float(np.linalg.norm(cell.truth(pid)["lin_vel"])) for pid in active}
        problems = G.validate_settled(poses, self.scene, speeds)
        stats = G.analyze_layout(poses)
        cell.arm_watchdog()
        return ResetResult(spec, poses, stats, problems, sim_s=cell.sim_time - t0)

    def _write_pose(self, obj, pos: np.ndarray, quat: np.ndarray) -> None:
        import torch

        pose = torch.tensor([[*map(float, pos), *map(float, quat)]], dtype=torch.float32, device=obj.device)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=obj.device))

    def _settle(self, ids: set[str], max_s: float, thresh: float, quiet_steps: int = 12) -> None:
        cell = self.cell
        quiet = 0
        for _ in range(int(max_s / cell.cfg.physics.dt)):
            cell.step()
            v = max(float(np.linalg.norm(cell.truth(i)["lin_vel"])) for i in ids) if ids else 0.0
            quiet = quiet + 1 if v < thresh else 0
            if quiet >= quiet_steps:
                return

    # ----------------------------------------------------- verdad (no para el agente)
    def active_ids(self) -> list[str]:
        return list(self.cell.pieces)

    def truth_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for pid in self.cell.pieces:
            t = self.cell.truth(pid)
            out[pid] = (t["pos"], t["quat"])
        return out

    def piece_truths(self) -> list[PieceTruth]:
        out = []
        for pid, (p, q) in self.truth_poses().items():
            pc = self.cell.cfg.piece(pid)
            out.append(PieceTruth(pid, pc.color, pc.shape, pc.size_m, pc.height, p, q))
        return out

    def audit(self) -> dict:
        """Dónde está cada pieza de verdad (para comparar con lo que cree el agente)."""
        regions: dict[str, dict[str, int]] = {}
        where = {}
        for pid, (p, _q) in self.truth_poses().items():
            pc = self.cell.cfg.piece(pid)
            r = C.region_of(self.scene, float(p[0]), float(p[1]), float(p[2]))
            where[pid] = r
            regions.setdefault(r, {})
            regions[r][pc.color] = regions[r].get(pc.color, 0) + 1
        return {"regions": regions, "where": where}
