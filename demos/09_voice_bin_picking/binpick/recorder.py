"""Grabación de demostraciones del experto en el formato de LeRobot (dataset v3), a 30 Hz.

El experto es ``recoverykit.skills.execute_pick`` (validado: agarre por fricción 30/30 en el demo 08). Aquí no
se reimplementa: se **observa** mientras actúa.

Cada fotograma guarda, con el preprocesado de ``observation.py`` (el mismo que usa la política al ejecutar):

* ``observation.images.overhead``  RGB recortado y reducido;
* ``observation.images.target``    segmentación codificada (objetivo / otras piezas / contenedor), fija durante el pick;
* ``observation.state``            articulaciones leídas del robot, unidades de LeRobot;
* ``action``                       la consigna que el experto tenía puesta al leer esa observación (unidades de LeRobot).

La acción es la **consigna**, no la posición que resulta: grabar el resultado enseña «repite lo que acabo de
hacer» y la política se queda quieta (demo 06). Con el control a 30 Hz la consigna adelanta a la posición unos
milímetros; ACT predice bloques de 50 consignas, que es lo que compensa.

**Ruido de aproximación (estilo DART).** En una fracción de los episodios se suma ruido suave y correlacionado a
lo que *se ejecuta* durante la aproximación, pero se **graba la consigna limpia**. La política ve estados algo
desviados con la orden correcta de volver a la trayectoria: así no aprende a copiar el error de su propio brazo
(la «confusión causal» del demo 06) y se recupera de sus propias desviaciones en lazo cerrado.

El episodio se guarda sólo si el criterio físico (``is_success``) lo confirma con la verdad del simulador: la
pieza se levanta y viaja con la pinza. Los fallidos se descartan y quedan en el registro con su causa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from . import joints
from . import observation as O
from . import picking
from recoverykit import skills  # noqa: E402

INFO_FILE = "binpick_info.json"
LOG_FILE = "binpick_episodes.jsonl"
ROBOT_TYPE = "so101_follower"
NOISE_PHASES = ("aproximación",)
NOISE_RHO = 0.995                  # correlación por paso de física (~0,8 s)


@dataclass
class NoiseCfg:
    std_rad: float = 0.02
    fraction: float = 0.5          # fracción de episodios con ruido
    phases: tuple[str, ...] = NOISE_PHASES

    def as_dict(self) -> dict:
        return {"std_rad": self.std_rad, "fraction": self.fraction, "phases": list(self.phases)}


@dataclass
class Frame:
    overhead: np.ndarray
    target: np.ndarray
    state: np.ndarray
    action: np.ndarray


class EpisodeRecorder:
    """Se engancha a una ``Workcell`` (o a la falsa de las pruebas): lee las consignas y muestrea cada 8 pasos."""

    def __init__(self, cell, spec: O.ObsSpec, grab, steps_per_frame: int = C.STEPS_PER_FRAME,
                 noise: NoiseCfg | None = None, rng: np.random.Generator | None = None):
        self.cell, self.spec, self.grab, self.spf = cell, spec, grab, steps_per_frame
        self.noise = noise or NoiseCfg(0.0, 0.0)
        self.rng = rng or np.random.default_rng(0)
        self.frames: list[Frame] = []
        self.active = False
        self.noisy = False
        self._phase = ""
        self._seg: np.ndarray | None = None
        self._n = 0
        self._off = np.zeros(5)
        self._clean_arm = np.array(cell.proprio()["arm_q"], float)
        self._clean_g = float(cell.proprio()["gripper_q"])
        self._orig_arm, self._orig_g = cell.set_arm_targets, cell.set_gripper_target
        cell.set_arm_targets = self._set_arm
        cell.set_gripper_target = self._set_g
        cell.step_hooks.append(self._on_step)

    # ----------------------------------------------------------- consignas
    def _set_arm(self, q) -> None:
        self._clean_arm = np.asarray(q, float).copy()
        self._orig_arm(self._clean_arm + self._off)

    def _set_g(self, g) -> None:
        self._clean_g = float(g)
        self._orig_g(float(g))

    def _update_noise(self) -> None:
        s = self.noise.std_rad
        if self.active and self.noisy and self._phase in self.noise.phases and s > 0:
            self._off = NOISE_RHO * self._off + np.sqrt(1 - NOISE_RHO ** 2) * s * self.rng.standard_normal(5)
        else:
            self._off = self._off * 0.99                      # fuera de la aproximación el desvío se desvanece

    # ---------------------------------------------------------- muestreo
    def _on_step(self) -> None:
        self._update_noise()
        if not self.active:
            return
        if self._n % self.spf == 0:
            rgb, _depth = self.grab()
            pro = self.cell.proprio()
            obs = O.build_obs(rgb, self._seg, pro["arm_q"], pro["gripper_q"], self.spec)
            action = joints.sim_to_motor(joints.join(self._clean_arm, self._clean_g))
            self.frames.append(Frame(obs.overhead, obs.target, obs.state, action))
        self._n += 1

    # -------------------------------------------------------------- API
    def start(self, seg_full: np.ndarray, noisy: bool) -> None:
        self.frames.clear()
        self._seg, self.noisy, self.active, self._n = seg_full, noisy, True, 0
        self._phase = ""

    def on_phase(self, name: str) -> None:
        self._phase = name

    def stop(self) -> list[Frame]:
        self.active = False
        return list(self.frames)

    def detach(self) -> None:
        self.cell.set_arm_targets, self.cell.set_gripper_target = self._orig_arm, self._orig_g
        if self._on_step in self.cell.step_hooks:
            self.cell.step_hooks.remove(self._on_step)


# ------------------------------------------------------------ criterio físico
def is_success(plan: skills.PickPlan, base: np.ndarray, q_arm: np.ndarray, gripper_q: float, piece_pos: np.ndarray) -> tuple[bool, str]:
    """¿Se levantó y viaja con la pinza? Con la verdad del simulador (sólo para filtrar demostraciones)."""
    grasp_h = picking.tcp_height(base, plan.descent[-1], plan.local)
    lift_h = picking.tcp_height(base, plan.lift[-1], plan.local)
    need = plan.grasp_z + 0.7 * (lift_h - grasp_h)
    tcp = picking.tcp_xy(base, q_arm, plan.local)
    if float(piece_pos[2]) < need:
        return False, "no levantó la pieza"
    if float(np.linalg.norm(piece_pos[:2] - tcp)) > 0.02:
        return False, "la pieza no viaja con la pinza"
    if not skills.held(gripper_q, plan):
        return False, "la pinza no sujeta"
    return True, ""


# ------------------------------------------------------------------ dataset
def write_info(root: Path, spec: O.ObsSpec, segmenter: str, cfg: C.BinPickCfg, noise: NoiseCfg, extra: dict | None = None) -> dict:
    info = {"obs_mode": O.DEFAULT_MODE, "roi": list(spec.roi), "out_hw": list(spec.out_hw), "fps": C.FPS,
            "segmenter": segmenter, "joint_units": "range_m100_100", "joint_names": list(joints.JOINT_NAMES),
            "camera": C.CAMERA, "config": cfg.to_dict(), "noise": noise.as_dict(),
            "expert": {"open_margin_m": picking.OPEN_MARGIN_M, "source": "recoverykit.skills.execute_pick"},
            "task": O.TASK_TEXT, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra or {})}
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def open_dataset(root: Path, repo_id: str, spec: O.ObsSpec, resume: bool, threads: int = 4):
    """Crea el dataset o, con ``resume``, abre el existente para añadir episodios."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if resume and (root / "meta" / "info.json").exists():
        ds = LeRobotDataset(repo_id=repo_id, root=str(root))
        if threads > 0 and hasattr(ds, "start_image_writer"):
            ds.start_image_writer(num_processes=0, num_threads=threads)
        return ds
    if (root / "meta" / "info.json").exists():
        raise FileExistsError(f"{root} ya contiene un dataset; usa --resume para añadir o elige otro nombre")
    return LeRobotDataset.create(repo_id=repo_id, fps=C.FPS, features=O.features(spec), root=str(root),
                                 robot_type=ROBOT_TYPE, use_videos=True, image_writer_threads=threads)


def add_episode(dataset, frames: list[Frame], task: str = O.TASK_TEXT) -> None:
    for f in frames:
        dataset.add_frame({O.IMAGE_KEYS["overhead"]: f.overhead, O.IMAGE_KEYS["target"]: f.target,
                           O.STATE_KEY: f.state, O.ACTION_KEY: f.action, "task": task})
    # codificación en serie: la paralela relanza ``__main__`` en procesos hijos, que en Windows (spawn) rompe el pool
    dataset.save_episode(parallel_encoding=False)


def read_log(root: Path) -> list[dict]:
    p = root / "meta" / LOG_FILE
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(root: Path, record: dict) -> None:
    p = root / "meta" / LOG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
