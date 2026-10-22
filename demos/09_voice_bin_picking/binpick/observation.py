"""La observación de la política: un único preprocesado para grabar demostraciones y para ejecutar.

Entradas de ACT (claves de LeRobot):

``observation.images.overhead``  RGB de la cámara cenital, recortado a la región de interés fija
                                 (contenedor + cajas) y reducido. Es la imagen original, sin tocar.
``observation.images.target``    máscara de segmentación codificada en color, mismo recorte:
                                 R = pieza objetivo, G = las demás piezas, B = interior del contenedor.
``observation.state``            6 articulaciones en unidades de LeRobot (``joints.sim_to_motor``).

Por qué dos imágenes y no una sola segmentada (decisión razonada en ``docs/02_arquitectura.md``):

* con *sólo* la segmentación se pierden las paredes, el brazo y la profundidad aparente, que
  son lo que guía la bajada: la variante ``mask_only`` existe para medir esa pérdida, no porque
  se espere que gane;
* con *sólo* RGB la política no sabe cuál de las piezas tiene que coger (``rgb_only`` es el
  control: tiene que fallar en escenas con varias piezas);
* ``rgb_mask`` da la imagen completa y, aparte, qué pieza y cuáles son obstáculos. ACT ya sabe
  procesar varias cámaras con el mismo ResNet, así que no hace falta tocar LeRobot.

La máscara se calcula **una vez** al empezar el pick (con el segmentador activo) y se mantiene
fija: dice *qué* pieza coger, no dónde está en cada instante (eso lo ve el RGB). Es lo mismo que
hizo el demo 06 con el centroide del cubo y evita depender de seguir la pieza mientras el brazo
la tapa.

Todo lo que transforma píxeles está en este módulo y lo llaman igual ``recorder.py`` y ``policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C
from . import joints

STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEYS = {"overhead": "observation.images.overhead", "target": "observation.images.target"}
# qué imágenes recibe la política en cada variante del experimento
OBS_MODES = {
    "rgb_mask": ("overhead", "target"),      # recomendada
    "rgb_only": ("overhead",),               # control: sin saber qué pieza
    "mask_only": ("target",),                # segmentación sola
}
DEFAULT_MODE = "rgb_mask"
TASK_TEXT = "Pick the marked industrial piece out of the bin and lift it clear of the walls."


@dataclass(frozen=True)
class ObsSpec:
    roi: tuple[int, int, int, int]          # x0, y0, x1, y1 en la imagen completa
    out_hw: tuple[int, int]                 # alto, ancho final

    def as_dict(self) -> dict:
        return {"roi": list(self.roi), "out_hw": list(self.out_hw)}


def roi_from_world(cam, x_range: tuple[float, float], y_range: tuple[float, float], z: float) -> tuple[int, int, int, int]:
    """Rectángulo de píxeles que cubre una región rectangular del plano z (mundo)."""
    pts = np.array([[x, y, z] for x in x_range for y in y_range])
    uv, _ = cam.project(pts)
    x0, y0 = np.floor(uv.min(0)).astype(int)
    x1, y1 = np.ceil(uv.max(0)).astype(int)
    return (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, cam.width)), int(min(y1, cam.height)))


def make_spec(cam, table_z: float, cfg: C.BinPickCfg) -> ObsSpec:
    return ObsSpec(roi_from_world(cam, cfg.roi_x_m, cfg.roi_y_m, table_z), cfg.policy_image_hw)


def crop_resize(img: np.ndarray, spec: ObsSpec, nearest: bool = False) -> np.ndarray:
    x0, y0, x1, y1 = spec.roi
    h, w = spec.out_hw
    return cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def bin_mask(cam, scene) -> np.ndarray:
    """Interior del contenedor de origen en la imagen (bool HxW), por geometría y calibración."""
    b = C.container(scene, C.BIN_ID)
    z = scene.table.top_z_m + b.floor_m
    cx, cy = b.center_xy_m
    pts = np.array([[cx + sx * b.inner_m[0] / 2, cy + sy * b.inner_m[1] / 2, z]
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    uv, _ = cam.project(pts)
    m = np.zeros((cam.height, cam.width), np.uint8)
    cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    return m.astype(bool)


def encode_seg(shape_hw: tuple[int, int], target: np.ndarray | None, others: list[np.ndarray],
               bin_footprint: np.ndarray) -> np.ndarray:
    """Imagen completa (H, W, 3) uint8: R = objetivo, G = otras piezas, B = interior del contenedor."""
    seg = np.zeros((*shape_hw, 3), np.uint8)
    seg[bin_footprint, 2] = 255
    for m in others:
        seg[m, 1] = 255
    if target is not None:
        seg[target, 0] = 255
        seg[target, 1] = 0
    return seg


@dataclass
class PolicyObs:
    overhead: np.ndarray        # (H, W, 3) uint8
    target: np.ndarray          # (H, W, 3) uint8
    state: np.ndarray           # (6,) float32, unidades de LeRobot


def build_obs(rgb_full: np.ndarray, seg_full: np.ndarray, arm_q: np.ndarray, gripper_q: float, spec: ObsSpec) -> PolicyObs:
    """RGB y segmentación a pantalla completa + articulaciones (rad del simulador) → observación de la política."""
    state = joints.sim_to_motor(joints.join(arm_q, gripper_q))
    return PolicyObs(crop_resize(rgb_full, spec), crop_resize(seg_full, spec, nearest=True), state)


def image_to_input(img_hwc: np.ndarray) -> np.ndarray:
    """(H, W, C) uint8 → (C, H, W) float32 en [0, 1]: la forma que espera un preprocesador de LeRobot."""
    return np.ascontiguousarray(np.asarray(img_hwc).transpose(2, 0, 1)).astype(np.float32) / 255.0


def to_model_input(obs: PolicyObs, mode: str = DEFAULT_MODE) -> dict[str, np.ndarray]:
    """Diccionario con las claves que declara el dataset, sólo las de la variante ``mode``.

    ``observation.state`` va siempre: ACT de LeRobot 0.4.4 lee su ``.device`` aunque no se use
    (demo 06), y el entrenamiento decide si la red lo consume.
    """
    out = {STATE_KEY: obs.state.astype(np.float32)}
    for stream in OBS_MODES[mode]:
        out[IMAGE_KEYS[stream]] = image_to_input(getattr(obs, stream))
    return out


def target_seg_full(cam, scene, instances, target_idx: int | None) -> np.ndarray:
    """Segmentación completa para un objetivo: la máscara de ``instances[target_idx]`` en R, el resto en G."""
    target = instances[target_idx].mask if target_idx is not None else None
    others = [i.mask for k, i in enumerate(instances) if k != target_idx and i.region == "origen"]
    return encode_seg((cam.height, cam.width), target, others, bin_mask(cam, scene))


def features(spec: ObsSpec, fps: int = C.FPS) -> dict:
    """Features para ``LeRobotDataset.create`` (los mismos que lee ``policy.py``)."""
    h, w = spec.out_hw
    video = lambda: {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}  # noqa: E731
    names = list(joints.JOINT_NAMES)
    return {
        IMAGE_KEYS["overhead"]: video(),
        IMAGE_KEYS["target"]: video(),
        STATE_KEY: {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in names]},
        ACTION_KEY: {"dtype": "float32", "shape": (6,), "names": [f"{n}.pos" for n in names]},
    }
