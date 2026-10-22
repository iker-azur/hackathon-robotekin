"""Estado compartido entre el agente (hilo de la simulación) y el servidor web.

El agente escribe; el servidor lee instantáneas y deja órdenes en una cola.
Todo pasa por un candado y las instantáneas son copias: el servidor nunca toca
objetos de Isaac.
"""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass

STAGES = ("VER", "ENTENDER", "PLANIFICAR", "ACTUAR", "VERIFICAR", "RECUPERAR")
STAGE_ES = {"VER": "Ver", "ENTENDER": "Entender", "PLANIFICAR": "Planificar", "ACTUAR": "Actuar",
            "VERIFICAR": "Verificar", "RECUPERAR": "Recuperarse"}
MAX_EVENTS = 300


class Stopped(Exception):
    """El usuario ha pedido parar."""


@dataclass
class Command:
    kind: str           # "kit" | "pause" | "resume" | "stop" | "reset" | "randomize" | "perturb" | "config" | "quit"
    payload: dict


class Telemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {
            "status": "esperando",       # esperando | ejecutando | completado | incompleto | fallido | detenido
            "stage": None, "stage_detail": "",
            "command": None, "kit": None, "kit_errors": [],
            "tasks": [], "world": [], "events": [], "recoveries": [],
            "kit_status": {"required": 0, "verified": 0, "complete": False},
            "perturbations": [], "armed_perturbations": [],
            "metrics": {}, "audit": None, "sim_time": 0.0, "rtf": None, "seed": None, "config": None,
        }
        self._frame: bytes | None = None
        self._frame_id = 0
        self._poses: dict | None = None
        self.commands: queue.Queue[Command] = queue.Queue()
        self._pause = threading.Event()
        self._stop = threading.Event()

    # ------------------------------------------------------------ escritura
    def set(self, **kw) -> None:
        with self._lock:
            self._state.update(copy.deepcopy(kw))

    def stage(self, name: str, detail: str = "") -> None:
        with self._lock:
            self._state["stage"] = name
            self._state["stage_detail"] = detail

    def event(self, kind: str, text: str, **data) -> None:
        with self._lock:
            ev = {"t": round(self._state["sim_time"], 2), "kind": kind, "text": text, **data}
            self._state["events"].append(ev)
            del self._state["events"][:-MAX_EVENTS]

    def recovery(self, failure: dict, decision: dict) -> None:
        with self._lock:
            self._state["recoveries"].append({"t": round(self._state["sim_time"], 2), "failure": failure,
                                              "decision": decision})

    def push_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame = jpeg
            self._frame_id += 1

    def push_poses(self, poses: dict) -> None:
        with self._lock:
            self._poses = poses

    def clear_run(self) -> None:
        with self._lock:
            for k in ("tasks", "events", "recoveries", "perturbations"):
                self._state[k] = []
            self._state.update(command=None, kit=None, kit_errors=[], audit=None, metrics={}, stage=None,
                               stage_detail="", kit_status={"required": 0, "verified": 0, "complete": False})

    # -------------------------------------------------------------- lectura
    def snapshot(self) -> dict:
        with self._lock:
            s = copy.deepcopy(self._state)
        s["paused"] = self._pause.is_set()
        s["frame_id"] = self._frame_id
        return s

    def frame(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._frame_id, self._frame

    def poses(self) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._poses)

    # ------------------------------------------------------------- control
    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def request_stop(self) -> None:
        self._stop.set()
        self._pause.clear()

    def clear_stop(self) -> None:
        self._stop.clear()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()
