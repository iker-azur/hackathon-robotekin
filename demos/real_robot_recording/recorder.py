"""SO-101 recording/data logic; hardware and terminal UI live in record.py."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)
LEROBOT_VERSION = "0.6.1"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
POSITION_KEYS = tuple(f"{name}.pos" for name in JOINTS)
IMAGE_KEYS = ("observation.images.wrist", "observation.images.overhead")


@dataclass
class Config:
    repo_id: str
    root: str
    task: str
    follower_port: str
    leader_port: str
    wrist_camera: str | int
    overhead_camera: str | int
    follower_id: str = "robotekin_follower"
    leader_id: str = "robotekin_leader"
    follower_calibration_dir: str | None = None
    leader_calibration_dir: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    episode_time_s: float = 60
    max_relative_target: float = 5.0
    streaming_encoding: bool = False
    display_data: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo_id) or "TU_USUARIO" in self.repo_id:
            raise ValueError("repo_id debe ser usuario/dataset real, sin placeholders")
        if not self.task.strip() or not self.root.strip():
            raise ValueError("task y root no pueden estar vacíos")
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 requiere width y height pares")
        for name in ("episode_time_s", "max_relative_target"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} debe ser positivo y finito")
        for name in ("streaming_encoding", "display_data"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} debe ser un booleano JSON")
        if not self.follower_id or not self.leader_id:
            raise ValueError("Ambos brazos necesitan un id de calibración")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        for key in ("root", "follower_calibration_dir", "leader_calibration_dir"):
            if values.get(key):
                values[key] = str((path.parent / Path(values[key]).expanduser()).resolve())
        return cls(**values)

    @property
    def local(self):
        # Local journals and locks must never be included in a Hub upload.
        root = Path(self.root)
        return root.with_name(root.name + ".recorder")

    def contract(self):
        return {
            "recorder_schema": 1, "lerobot_version": LEROBOT_VERSION,
            "repo_id": self.repo_id, "task": self.task,
            "fps": self.fps, "width": self.width, "height": self.height,
            "joint_order": list(JOINTS), "arm_unit": "degrees", "gripper_unit": "0..100",
            "views": ["wrist", "overhead"],
            "camera_devices": {"wrist": self.wrist_camera, "overhead": self.overhead_camera},
            "follower_id": self.follower_id, "leader_id": self.leader_id,
        }


def require_version():
    actual = importlib.metadata.version("lerobot")
    if actual != LEROBOT_VERSION:
        raise RuntimeError(f"Se requiere lerobot=={LEROBOT_VERSION}; instalado: {actual}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


@contextlib.contextmanager
def dataset_lock(cfg):
    """An OS lock releases automatically even when the process dies."""
    cfg.local.mkdir(parents=True, exist_ok=True)
    with (cfg.local / "writer.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Otro proceso está usando este dataset") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def features(cfg):
    result = {
        name: {"dtype": "float32", "shape": (6,), "names": list(JOINTS)}
        for name in ("observation.state", "action", "leader_state")
    }
    result["capture_time_s"] = {"dtype": "float64", "shape": (1,), "names": None}
    for name in IMAGE_KEYS:
        result[name] = {"dtype": "video", "shape": (cfg.height, cfg.width, 3),
                        "names": ["height", "width", "channels"]}
    return result


def positions(values):
    result = np.asarray([values[key] for key in POSITION_KEYS], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Lectura articular no finita")
    return result


def sample(cfg, follower, leader, elapsed):
    """Observe BEFORE applying the next command. Read the leader exactly once."""
    observation = follower.get_observation()
    state = positions(observation)
    images = {}
    for name in ("wrist", "overhead"):
        image = observation[name]
        if image.dtype != np.uint8 or image.shape != (cfg.height, cfg.width, 3):
            raise ValueError(f"{name}: se esperaba RGB uint8 {cfg.height}x{cfg.width}x3")
        images[f"observation.images.{name}"] = image
    command = leader.get_action()
    leader_state = positions(command)
    sent = follower.send_action(command)
    return {**images, "observation.state": state, "action": positions(sent),
            "leader_state": leader_state,
            "capture_time_s": np.asarray([elapsed], dtype=np.float64), "task": cfg.task}


def writer(cfg, resume):
    from lerobot.configs import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    options = {"repo_id": cfg.repo_id, "root": cfg.root, "batch_encoding_size": 1,
               "rgb_encoder": RGBEncoderConfig(vcodec="h264"), "encoder_threads": 2,
               "streaming_encoding": cfg.streaming_encoding, "video_backend": "pyav",
               "image_writer_processes": 0, "image_writer_threads": 8}
    if resume:
        return LeRobotDataset.resume(**options)
    return LeRobotDataset.create(**options, fps=cfg.fps, features=features(cfg), robot_type="so101_follower")


class Session:
    """Journal every writer lifetime; do not silently repair interrupted files."""

    def __init__(self, cfg, resume=False, provenance=None):
        self.cfg = cfg
        self.dataset = None
        self.count = 0
        self.state_path = cfg.local / "state.json"
        self.state = {"phase": "idle", "episodes": 0, "uploaded_episodes": 0}
        contract_path = cfg.local / "contract.json"
        exists = Path(cfg.root).exists()
        if exists and not resume:
            raise FileExistsError("El directorio ya existe. Usa --resume o una ruta nueva")
        if resume:
            if not contract_path.exists() or not self.state_path.exists():
                raise RuntimeError("No es un dataset local creado por este grabador; usa una ruta nueva")
            if json.loads(contract_path.read_text(encoding="utf-8")) != cfg.contract():
                raise ValueError("Configuración incompatible con la sesión guardada")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["phase"] != "idle":
                raise RuntimeError("Escritura interrumpida. Conserva los archivos y ejecuta validate; "
                                   "no se reanuda ni repara automáticamente")
            self.count = self.state["episodes"]
            if self.count and validate(cfg)["episodes"] != self.count:
                raise RuntimeError("El diario y los episodios locales no coinciden")
        else:
            if self.state_path.exists():
                raise FileExistsError("Hay un diario anterior para esta ruta. Usa una ruta nueva")
            atomic_json(contract_path, cfg.contract())
            self._state("idle")
        self.provenance = provenance or {}

    def _state(self, phase):
        self.state.update(phase=phase, episodes=self.count)
        atomic_json(self.state_path, self.state)

    def begin(self):
        self._state("recording")
        self.dataset = writer(self.cfg, resume=(Path(self.cfg.root) / "meta/info.json").exists())
        if self.count == 0:
            atomic_json(Path(self.cfg.root) / "recording.json",
                        {**self.cfg.contract(), "calibrations": self.provenance})

    def add(self, frame):
        self.dataset.add_frame(frame)

    def discard(self):
        if self.dataset is None:
            return
        # finalize drains pending PNG writes and cancels unsaved streaming video.
        # Clear only afterwards, so background writes cannot recreate discarded images.
        self.dataset.finalize()
        self.dataset.clear_episode_buffer()
        self.dataset = None
        self._state("idle")

    def accept(self, report):
        if self.dataset is None or not self.dataset.has_pending_frames():
            raise ValueError("No hay fotogramas que aceptar")
        self._state("saving")
        # Two camera streams do not need extra Python processes (expensive on spawn platforms).
        self.dataset.save_episode(parallel_encoding=False)
        self.dataset.finalize()
        self.count = self.dataset.num_episodes
        atomic_json(self.cfg.local / f"episode-{self.count - 1:06d}.json", report)
        self.dataset = None
        self._state("idle")

    def sync(self, push=None, sleep=time.sleep):
        if self.state["phase"] != "idle" or self.dataset is not None:
            raise RuntimeError("No se puede subir un dataset abierto")
        if self.count == 0:
            return True
        if push is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(self.cfg.repo_id, root=self.cfg.root, video_backend="pyav")
            push = lambda: dataset.push_to_hub(
                private=False, license=None, tags=["lerobot", "so101", "act"],
                allow_patterns=["data/**", "videos/**", "meta/**", "recording.json", "README.md"],
            )
        for attempt in range(3):
            try:
                push()
            except Exception as exc:  # noqa: BLE001 -- every upload failure must preserve offline capture
                # Hub errors may contain request URLs; do not log tokens or request headers.
                LOG.warning("Subida %s/3 fallida (%s). Datos locales conservados.",
                            attempt + 1, type(exc).__name__)
                if attempt < 2:
                    sleep(2 ** (attempt + 1))
            else:
                self.state["uploaded_episodes"] = self.count
                self._state("idle")
                return True
        return False


def validate(cfg, root=None):
    """Read all rows and each episode's first/last camera frames without hardware."""
    import pyarrow.parquet as pq
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root or cfg.root)
    # Preflight locally: LeRobot's reader otherwise downloads missing files from the Hub.
    for relative in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "recording.json"):
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    recording = json.loads((root / "recording.json").read_text(encoding="utf-8"))
    if any(recording.get(key) != value for key, value in cfg.contract().items()):
        raise ValueError("El manifest de grabación es incompatible")
    if not list((root / "meta/episodes").rglob("*.parquet")):
        raise ValueError("No hay metadatos de episodios aceptados")
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=root)
    data_paths = set()
    for ep in range(meta.total_episodes):
        data_paths.add(root / meta.get_data_file_path(ep))
        for name in IMAGE_KEYS:
            path = root / meta.get_video_file_path(ep, name)
            if not path.is_file():
                raise FileNotFoundError(path)
    available = set()
    for path in data_paths:
        available.update(pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist())
    if available != set(range(meta.total_episodes)):
        raise ValueError("Faltan episodios locales; no se descargan ni reparan automáticamente")
    dataset = LeRobotDataset(cfg.repo_id, root=root, video_backend="pyav")
    if dataset.fps != cfg.fps or dataset.num_episodes <= 0:
        raise ValueError("FPS incompatible o dataset sin episodios aceptados")
    for name, expected in features(cfg).items():
        actual = dataset.features.get(name, {})
        if actual.get("dtype") != expected["dtype"] or tuple(actual.get("shape", ())) != expected["shape"]:
            raise ValueError(f"Feature incompatible: {name}")
        if actual.get("names") != expected["names"]:
            raise ValueError(f"Orden/nombres incompatibles: {name}")
    rows = dataset.hf_dataset
    if list(dataset.meta.tasks.index) != [cfg.task]:
        raise ValueError("Tarea incompatible")
    total = 0
    for ep in range(dataset.num_episodes):
        # Iterate contiguous rows, avoiding video decoding for every sample.
        start = total
        times = []
        while total < len(rows) and int(rows[total]["episode_index"]) == ep:
            row = rows[total]
            if int(row["frame_index"]) != total - start or int(row["index"]) != total:
                raise ValueError("Índices de fotogramas discontinuos")
            if abs(float(row["timestamp"]) - (total - start) / cfg.fps) > 1e-4:
                raise ValueError("Timestamp nominal incompatible")
            for name in ("observation.state", "action", "leader_state"):
                values = np.asarray(row[name])
                if values.shape != (6,) or not np.isfinite(values).all():
                    raise ValueError(f"Datos articulares inválidos: {name}")
            times.append(float(row["capture_time_s"]))
            total += 1
        if not times or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f"Episodio {ep}: tiempo real inválido")
        metadata = dataset.meta.episodes[ep]
        if (metadata["dataset_from_index"], metadata["dataset_to_index"], metadata["length"]) != (
            start, total, total - start
        ):
            raise ValueError(f"Episodio {ep}: límites incompatibles")
        for index in {start, total - 1}:
            item = dataset[index]
            for name in IMAGE_KEYS:
                if tuple(item[name].shape) != (3, cfg.height, cfg.width):
                    raise ValueError(f"Vídeo incompatible: {name}")
    if total != len(dataset) or total != len(rows):
        raise ValueError("Recuento de fotogramas incompatible")
    for name in ("observation.state", "action", "leader_state", *IMAGE_KEYS):
        stats = dataset.meta.stats.get(name, {})
        if not stats or any(not np.isfinite(np.asarray(v)).all() for v in stats.values()):
            raise ValueError(f"Estadísticas ausentes/no finitas: {name}")
    return {"episodes": dataset.num_episodes, "frames": total, "fps": dataset.fps}
