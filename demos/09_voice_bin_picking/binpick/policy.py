"""La política de agarre: ACT de LeRobot (aprendida) o el guion del experto (NO aprendido).

``ACTPickPolicy`` carga un checkpoint de LeRobot con sus pre y postprocesadores (los que normalizan con
las estadísticas del entrenamiento) y convierte una ``PolicyObs`` en una orden de articulaciones en
radianes del simulador. La observación se construye con ``observation.build_obs``/``to_model_input``, la
misma función que usó ``recorder.py`` al grabar: así el preprocesado de entrenamiento e inferencia es
idéntico. Un checkpoint lleva al lado ``binpick_info.json`` (modo de observación, recorte, segmentador,
unidades) y ``check_compat`` se niega a ejecutarlo si el recorte o las unidades no coinciden.

``ActionLimiter`` es la salvaguarda entre la red y el simulador (y, mañana, el robot real): recorta a los
límites de las articulaciones y a la velocidad máxima por paso, y rechaza una orden que metería el brazo
en la mesa. Cuenta cuántas veces actúa; si actúa mucho, la política no está bien.

La inferencia va en CPU por defecto: cargar un modelo de torch con CUDA antes de arrancar Isaac deja un
contexto que hace fallar a PhysX (``illegal memory access``, demo 06).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import config as C
from . import joints
from . import observation as O
from recoverykit import kinematics as K  # noqa: E402

INFO_FILE = "binpick_info.json"


class PickPolicy(Protocol):
    kind: str          # "act" (aprendida) | "guion" (experto programado)
    name: str

    def reset(self) -> None: ...
    def act(self, obs: O.PolicyObs) -> np.ndarray: ...


# ----------------------------------------------------------------- salvaguarda
@dataclass
class ActionLimiter:
    """Orden (6, rad) → orden segura. ``clearance_fn(q5)`` devuelve la holgura (m) del brazo a la mesa."""

    max_arm_step: float = 1.5 * 2.0 / C.FPS           # 1,5 × la velocidad máxima de articulación por paso de control
    max_gripper_step: float = 0.30
    clearance_fn: Callable[[np.ndarray], float] | None = None
    prev: np.ndarray | None = None
    clipped_limits: int = 0
    clipped_speed: int = 0
    rejected_table: int = 0
    steps: int = 0

    def __post_init__(self):
        lim = K.limits()
        self.lo = np.array([lim[n][0] for n in K.ARM] + [lim["gripper"][0]])
        self.hi = np.array([lim[n][1] for n in K.ARM] + [lim["gripper"][1]])

    def reset(self, q_now: np.ndarray) -> None:
        self.prev = np.asarray(q_now, dtype=float).copy()

    def __call__(self, target: np.ndarray) -> np.ndarray:
        if self.prev is None:
            raise RuntimeError("llama a reset(q_actual) antes de limitar")
        self.steps += 1
        t = np.asarray(target, dtype=float).copy()
        clipped = np.clip(t, self.lo, self.hi)
        self.clipped_limits += int(np.any(clipped != t))
        step = np.clip(clipped - self.prev, -self._steps(), self._steps())
        self.clipped_speed += int(np.any(np.abs(clipped - self.prev) > self._steps() + 1e-12))
        safe = self.prev + step
        if self.clearance_fn is not None and self.clearance_fn(safe[:5]) < 0.0:
            self.rejected_table += 1
            safe = self.prev.copy()
        self.prev = safe
        return safe

    def _steps(self) -> np.ndarray:
        return np.array([self.max_arm_step] * 5 + [self.max_gripper_step])

    def stats(self) -> dict:
        return {"pasos": self.steps, "recortes_de_limite": self.clipped_limits,
                "recortes_de_velocidad": self.clipped_speed, "rechazos_por_mesa": self.rejected_table}


# ------------------------------------------------------------ información del modelo
def load_info(policy_path: str | Path) -> dict:
    folder = Path(policy_path).resolve()
    if folder.is_file():
        folder = folder.parent
    for cand in [folder, *list(folder.parents)[:4]]:
        f = cand / INFO_FILE
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    return {}


def check_compat(info: dict, spec: O.ObsSpec, segmenter: str, fps: int = C.FPS) -> tuple[list[str], list[str]]:
    """(errores, avisos) al ejecutar un checkpoint con este recorte, segmentador y frecuencia."""
    errors, warnings = [], []
    if not info:
        warnings.append(f"El modelo no trae {INFO_FILE}: no se puede comprobar que el preprocesado coincide.")
        return errors, warnings
    if info.get("roi") and list(info["roi"]) != list(spec.roi):
        errors.append(f"El recorte de la imagen no coincide: el modelo se entrenó con {info['roi']} y ahora es {list(spec.roi)} "
                      "(¿cambió la cámara o la escena?).")
    if info.get("out_hw") and list(info["out_hw"]) != list(spec.out_hw):
        errors.append(f"El tamaño de imagen no coincide: {info['out_hw']} frente a {list(spec.out_hw)}.")
    if info.get("joint_units", "range_m100_100") != "range_m100_100":
        errors.append(f"Unidades de articulación desconocidas: {info.get('joint_units')}.")
    if info.get("fps") and int(info["fps"]) != fps:
        errors.append(f"Frecuencia distinta: entrenado a {info['fps']} Hz, ejecutado a {fps} Hz.")
    if info.get("segmenter") and info["segmenter"] != segmenter:
        warnings.append(f"El modelo se entrenó con máscaras de «{info['segmenter']}» y ahora se usa «{segmenter}»: "
                        "la máscara de objetivo puede verse distinta.")
    return errors, warnings


# --------------------------------------------------------------------- políticas
class ScriptedPolicy:
    """Marcador del experto programado. Lo ejecuta ``recoverykit.skills.execute_pick``, no ``act``: NO es aprendizaje."""

    kind = "guion"
    name = "guion del experto (no aprendido)"

    def reset(self) -> None:
        return None

    def act(self, obs):  # pragma: no cover - no se usa
        raise NotImplementedError("el guion se ejecuta con skills.execute_pick")


class CallablePolicy:
    """Envuelve una función ``obs -> (6,)``: para pruebas y para probar la ejecución sin un modelo."""

    kind = "act"

    def __init__(self, fn: Callable[[O.PolicyObs], np.ndarray], name: str = "función"):
        self.fn, self.name = fn, name

    def reset(self) -> None:
        return None

    def act(self, obs: O.PolicyObs) -> np.ndarray:
        return np.asarray(self.fn(obs), dtype=float)


class ACTPickPolicy:
    kind = "act"

    def __init__(self, path: str | Path, obs_mode: str | None = None, device: str = "cpu", n_action_steps: int | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.path = Path(path)
        self.info = load_info(self.path)
        self.obs_mode = obs_mode or self.info.get("obs_mode") or O.DEFAULT_MODE
        if self.obs_mode not in O.OBS_MODES:
            raise ValueError(f"modo de observación desconocido: {self.obs_mode}")
        saved = PreTrainedConfig.from_pretrained(str(self.path))
        saved.device = device
        if n_action_steps is not None:
            saved.n_action_steps = min(n_action_steps, saved.chunk_size)
        self.policy = ACTPolicy.from_pretrained(str(self.path), config=saved)
        self.policy.to(device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self.name = f"ACT ({self.path.parent.parent.name if self.path.name == 'pretrained_model' else self.path.name})"
        wanted = {k for k in self.policy.config.input_features if k.startswith("observation.images.")}
        expected = {O.IMAGE_KEYS[s] for s in O.OBS_MODES[self.obs_mode]}
        if wanted != expected:
            raise ValueError(f"El modelo espera las imágenes {sorted(wanted)} pero el modo «{self.obs_mode}» da {sorted(expected)}.")
        self._torch = torch

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: O.PolicyObs) -> np.ndarray:
        torch = self._torch
        batch = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k, v in O.to_model_input(obs, self.obs_mode).items()}
        with torch.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch)))
        motor = action.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return joints.motor_to_sim(motor)


def load(spec: str, device: str = "cpu", obs_mode: str | None = None):
    """``guion`` | ``<carpeta pretrained_model>`` → (política, modo). ``guion`` devuelve ``ScriptedPolicy``."""
    if spec in ("guion", "scripted"):
        return ScriptedPolicy()
    return ACTPickPolicy(spec, obs_mode=obs_mode, device=device)
