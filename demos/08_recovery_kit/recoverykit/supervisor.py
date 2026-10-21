"""Supervisor: qué hacer cuando algo sale mal.

Recibe un fallo observado (por la cámara o por la pinza) y decide la siguiente
acción. No sabe qué perturbación se ha inyectado ni dónde: decide con lo que
el agente ha medido y con el historial de intentos de cada pieza.

Escalera de estrategias para un agarre fallido de la misma pieza:
1. volver a mirar y apuntar al centro medido de nuevo;
2. girar la pinza a la otra pareja de caras;
3. agarrar un poco más abajo y subir más despacio;
4. descartar esa pieza y usar otra igual, si la hay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_GRASP_ATTEMPTS = 4
MAX_PLACE_ATTEMPTS = 3
MAX_RECOVERIES = 14


@dataclass
class Failure:
    kind: str          # "vacío" | "deslizada" | "movida" | "fuera_de_caja" | "inalcanzable" | "perdida" | "movimiento"
    task: str | None
    track: str | None
    detail: str = ""
    shape: str = "cube"


@dataclass
class Decision:
    action: str        # "reintentar" | "replanificar" | "otra_pieza" | "abandonar"
    reason: str
    strategy: dict = field(default_factory=dict)
    blacklist: str | None = None

    def as_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "strategy": self.strategy}


STRATEGY_TEXT = {
    1: "vuelvo a localizarla y apunto a su centro medido",
    2: "pruebo con la otra pareja de caras (pinza girada)",
    3: "la cojo un poco más abajo y subo más despacio",
}


@dataclass
class Supervisor:
    grasp_attempts: dict[str, int] = field(default_factory=dict)
    place_attempts: dict[str, int] = field(default_factory=dict)
    recoveries: int = 0
    history: list[tuple[Failure, Decision]] = field(default_factory=list)

    def strategy_for(self, track: str, shape: str) -> dict:
        """Ajustes de agarre según los intentos fallidos ya hechos con esta pieza."""
        n = self.grasp_attempts.get(track, 0)
        s: dict = {}
        if n >= 2:
            s["yaw_offset"] = math.pi / 2 if shape == "cube" else math.pi / 4
        if n >= 3:
            s["grasp_lower_m"] = 0.002
            s["lift_speed"] = 0.04
        return s

    def decide(self, f: Failure, alternatives: int) -> Decision:
        """``alternatives``: piezas compatibles que quedan en la mesa además de ésta."""
        self.recoveries += 1
        if self.recoveries > MAX_RECOVERIES:
            d = Decision("abandonar", "demasiadas recuperaciones seguidas: paro y pido ayuda")
        elif f.kind == "movida":
            d = Decision("replanificar", f"{f.detail}; vuelvo a mirar y recalculo el agarre")
        elif f.kind in ("vacío", "deslizada"):
            n = self.grasp_attempts.get(f.track, 0) + 1
            self.grasp_attempts[f.track] = n
            what = "la pinza se cerró sin la pieza" if f.kind == "vacío" else "la pieza se soltó"
            if n < MAX_GRASP_ATTEMPTS:
                d = Decision("reintentar", f"{what}; intento {n + 1}: {STRATEGY_TEXT[n]}",
                             strategy=self.strategy_for(f.track, f.shape))
            elif alternatives > 0:
                d = Decision("otra_pieza", f"{what} {n} veces; descarto esta pieza y uso otra igual",
                             blacklist=f.track)
            else:
                d = Decision("abandonar", f"{what} {n} veces y no hay otra pieza igual")
        elif f.kind == "fuera_de_caja":
            n = self.place_attempts.get(f.track, 0) + 1
            self.place_attempts[f.track] = n
            if n < MAX_PLACE_ATTEMPTS:
                d = Decision("replanificar", f"{f.detail}; la vuelvo a coger desde donde ha caído")
            elif alternatives > 0:
                d = Decision("otra_pieza", "esta pieza ha caído fuera varias veces; uso otra", blacklist=f.track)
            else:
                d = Decision("abandonar", "la pieza cae fuera de la caja una y otra vez")
        elif f.kind == "inalcanzable":
            if alternatives > 0:
                d = Decision("otra_pieza", f"{f.detail}; uso otra pieza igual", blacklist=f.track)
            else:
                d = Decision("abandonar", f"{f.detail}; no hay otra pieza alcanzable")
        elif f.kind == "perdida":
            d = Decision("replanificar", f"{f.detail}; vuelvo a mirar la mesa")
        else:
            n = self.grasp_attempts.get(f.track or "", 0) + 1
            self.grasp_attempts[f.track or ""] = n
            d = Decision("reintentar" if n < 2 else "abandonar", f"fallo de movimiento: {f.detail}")
        self.history.append((f, d))
        return d
