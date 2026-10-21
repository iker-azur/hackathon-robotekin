"""Ajustes de PhysX y vigilante de violaciones físicas.

El vigilante comprueba en cada paso que ninguna pieza se teletransporta, se
hunde en la mesa o sale disparada. Lo usa el demo y lo usan las pruebas: una
ejecución sólo cuenta como válida con cero violaciones después del reinicio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SceneCfg

# velocidad máxima creíble de una pieza en esta célula (m/s): una caída desde
# 0.3 m llega a ~2.4 m/s; por encima, el salto viene de una escritura de pose
MAX_PIECE_SPEED = 3.0
TELEPORT_MARGIN_M = 0.002
PENETRATION_TOLERANCE_M = 0.0015


def sim_cfg(cfg: SceneCfg):
    import isaaclab.sim as sim_utils

    p = cfg.physics
    return sim_utils.SimulationCfg(
        dt=p.dt,
        render_interval=p.render_interval,
        device=p.device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=p.solver_type,
            enable_ccd=p.enable_ccd,
            bounce_threshold_velocity=p.bounce_threshold_m_s,
            friction_correlation_distance=p.friction_correlation_m,
            friction_offset_threshold=p.friction_offset_threshold_m,
            enable_stabilization=False,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=cfg.table.material.static_friction,
            dynamic_friction=cfg.table.material.dynamic_friction,
            restitution=cfg.table.material.restitution,
        ),
    )


@dataclass
class Violation:
    step: int
    piece: str
    kind: str      # "teleport" | "penetración" | "velocidad"
    value: float

    def text(self) -> str:
        if self.kind == "velocidad":
            return f"paso {self.step}: {self.piece} — velocidad imposible ({self.value:.2f} m/s)"
        what = "salto de posición" if self.kind == "teleport" else "atraviesa la mesa"
        return f"paso {self.step}: {self.piece} — {what} ({self.value*1000:.1f} mm)"


@dataclass
class Watchdog:
    """Compara cada paso con el anterior. ``arm()`` tras el reinicio.

    ``min_half`` es, por pieza, la mitad de su dimensión más pequeña: el centro
    de una pieza apoyada nunca está más bajo que eso sobre su apoyo.
    ``support(x, y)`` da la altura del apoyo: suelo de la caja, tablero o suelo.
    """

    dt: float
    support: object
    min_half: dict[str, float]
    violations: list[Violation] = field(default_factory=list)
    steps: int = 0
    _last: dict[str, np.ndarray] | None = None

    def arm(self, positions: dict[str, np.ndarray]) -> None:
        self._last = {k: v.copy() for k, v in positions.items()}

    def disarm(self) -> None:
        self._last = None

    @property
    def armed(self) -> bool:
        return self._last is not None

    def check(self, positions: dict[str, np.ndarray], speeds: dict[str, float]) -> list[Violation]:
        self.steps += 1
        if self._last is None:
            return []
        new: list[Violation] = []
        limit = MAX_PIECE_SPEED * self.dt + TELEPORT_MARGIN_M
        for name, pos in positions.items():
            jump = float(np.linalg.norm(pos - self._last[name]))
            if jump > limit:
                new.append(Violation(self.steps, name, "teleport", jump))
            low = self.support(float(pos[0]), float(pos[1])) + self.min_half[name] - float(pos[2])
            if low > PENETRATION_TOLERANCE_M:
                new.append(Violation(self.steps, name, "penetración", low))
            if speeds.get(name, 0.0) > MAX_PIECE_SPEED:
                new.append(Violation(self.steps, name, "velocidad", speeds[name]))
        self._last = {k: v.copy() for k, v in positions.items()}
        self.violations.extend(new)
        return new

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for v in self.violations:
            kinds[v.kind] = kinds.get(v.kind, 0) + 1
        return {"steps": self.steps, "violations": len(self.violations), "by_kind": kinds,
                "first": [v.text() for v in self.violations[:10]]}
