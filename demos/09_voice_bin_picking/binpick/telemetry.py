"""Estado compartido entre el hilo de la simulación y el panel web (amplía el de RecoveryKit).

El hilo de Isaac escribe; el servidor lee copias y deja órdenes en una cola. Ningún objeto de Isaac
cruza esa frontera.
"""

from __future__ import annotations

import copy

from . import config as C
from recoverykit.telemetry import STAGES, Command, Stopped, Telemetry  # noqa: E402,F401

STAGE_ES = {"ENTENDER": "Entender", "VER": "Ver", "PLANIFICAR": "Planificar", "ACTUAR": "Actuar",
            "VERIFICAR": "Verificar", "RECUPERAR": "Recuperarse"}
FRAME_NAMES = ("cam", "seg", "vista")


class BinTelemetry(Telemetry):
    def __init__(self):
        super().__init__()
        self._frames: dict[str, tuple[int, bytes | None]] = {n: (0, None) for n in FRAME_NAMES}
        with self._lock:
            self._state.update({
                "status": "iniciando",
                "heard": None,                       # lo que se ha oído (voz) o escrito
                "order": None,                       # la orden interpretada
                "instances": [],                     # piezas detectadas
                "boxes": {"origen": 0, "izquierda": 0, "derecha": 0},
                "capacity": {"izquierda": None, "derecha": None},
                "report": None,
                "policy": {"kind": "guion", "name": "guion del experto (no aprendido)", "mode": "guion"},
                "segmenter": "hsv",
                "scene": None,
                "voice": {"state": "apagada", "detail": ""},
                "has_view": False,
                "ready": False,
            })

    # ----------------------------------------------------------------- imágenes
    def push_named(self, name: str, jpeg: bytes | None) -> None:
        if jpeg is None:
            return
        with self._lock:
            fid, _ = self._frames[name]
            self._frames[name] = (fid + 1, jpeg)

    def named(self, name: str) -> tuple[int, bytes | None]:
        with self._lock:
            return self._frames.get(name, (0, None))

    def frame_ids(self) -> dict[str, int]:
        with self._lock:
            return {n: f[0] for n, f in self._frames.items()}

    # ------------------------------------------------------------------ estado
    def clear_run(self) -> None:
        with self._lock:
            for k in ("events", "recoveries"):
                self._state[k] = []
            self._state.update(order=None, report=None, stage=None, stage_detail="", heard=None)

    def snapshot(self) -> dict:
        s = super().snapshot()
        s["frames"] = self.frame_ids()
        return s

    def set_boxes(self, instances) -> None:
        boxes = {"origen": 0, "izquierda": 0, "derecha": 0}
        for i in instances:
            if i.region in boxes:
                boxes[i.region] += 1
            elif i.region == "mesa":
                boxes["origen"] += 1
        self.set(boxes=boxes)

    def send(self, kind: str, **payload) -> None:
        self.commands.put(Command(kind, payload))


def piece_rows(instances) -> list[dict]:
    """Filas de la tabla de piezas del panel (sin máscaras)."""
    rows = [i.as_dict() for i in instances]
    rows.sort(key=lambda r: (r["region"] != "origen", r["id"]))
    return copy.deepcopy(rows)
