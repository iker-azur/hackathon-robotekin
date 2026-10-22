"""Simulated workspace: runs the whole Phase 2 stack without a camera.

A scene describes a background and objects. Objects can move at a constant
velocity, enter at a given frame and leave at another. Frames are rendered with
OpenCV and optional Gaussian noise, so the detector sees the same kind of input as
from a webcam. The known object positions are available as ground truth.

Use it as a source string: "sim:demo", "sim:static" or "sim:path/to/scene.json".
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .camera import CameraError, FrameSource

SIM_PREFIX = "sim:"
SHAPES = ("square", "rect", "circle", "triangle")
# OpenCV BGR values for the supported colors (same names as the voice commands).
BGR = {
    "rojo": (0, 0, 255), "naranja": (0, 128, 255), "amarillo": (0, 255, 255),
    "verde": (0, 200, 0), "azul": (255, 0, 0), "morado": (180, 40, 160),
    "blanco": (240, 240, 240), "gris": (128, 128, 128), "negro": (0, 0, 0),
}

BUILTIN_SCENES: Dict[str, Dict[str, Any]] = {
    "demo": {
        "width": 640, "height": 480, "background": [40, 40, 40], "noise_sigma": 3.0, "seed": 42,
        "loop_frames": 300,
        "objects": [
            {"id": "cubo_rojo_1", "shape": "square", "color": "rojo", "center": [150, 140], "size": [60, 60]},
            {"id": "cubo_rojo_2", "shape": "square", "color": "rojo", "center": [330, 140], "size": [60, 60]},
            {"id": "cilindro_azul", "shape": "circle", "color": "azul", "center": [500, 140], "radius": 35},
            {"id": "prisma_verde", "shape": "rect", "color": "verde", "center": [150, 320],
             "size": [110, 55], "velocity": [1.2, 0.0]},
            {"id": "cubo_amarillo", "shape": "square", "color": "amarillo", "center": [480, 340],
             "size": [60, 60], "enter_frame": 45},
        ],
    },
    "static": {
        "width": 640, "height": 480, "background": [40, 40, 40], "noise_sigma": 0.0, "seed": 0,
        "loop_frames": 300,
        "objects": [
            {"id": "cubo_rojo", "shape": "square", "color": "rojo", "center": [150, 140], "size": [60, 60]},
            {"id": "cilindro_azul", "shape": "circle", "color": "azul", "center": [320, 240], "radius": 40},
            {"id": "prisma_verde", "shape": "rect", "color": "verde", "center": [480, 340], "size": [110, 55]},
        ],
    },
}


def load_scene(name_or_path: str) -> Dict[str, Any]:
    """Return a validated copy of a built-in scene or of a JSON scene file."""
    if name_or_path in BUILTIN_SCENES:
        scene = copy.deepcopy(BUILTIN_SCENES[name_or_path])
    elif os.path.exists(name_or_path):
        with open(name_or_path, encoding="utf-8") as f:
            scene = json.load(f)
    else:
        raise CameraError(
            f"Escena desconocida: {name_or_path}. Disponibles: {', '.join(BUILTIN_SCENES)} "
            "o una ruta a un JSON de escena."
        )
    _validate(scene, name_or_path)
    return scene


def _validate(scene: Dict[str, Any], label: str) -> None:
    for key in ("width", "height", "background", "objects"):
        if key not in scene:
            raise CameraError(f"Escena {label}: falta la clave '{key}'.")
    scene.setdefault("noise_sigma", 0.0)
    scene.setdefault("seed", 0)
    scene.setdefault("loop_frames", 300)
    for obj in scene["objects"]:
        if obj.get("color") not in BGR:
            raise CameraError(f"Escena {label}: color no soportado '{obj.get('color')}'. Usa: {', '.join(BGR)}.")
        if obj.get("shape") not in SHAPES:
            raise CameraError(f"Escena {label}: forma no soportada '{obj.get('shape')}'. Usa: {', '.join(SHAPES)}.")
        obj.setdefault("velocity", [0.0, 0.0])
        obj.setdefault("enter_frame", 0)
        obj.setdefault("exit_frame", None)


def _position(obj: Dict[str, Any], frame_index: int) -> tuple:
    vx, vy = obj["velocity"]
    return obj["center"][0] + vx * frame_index, obj["center"][1] + vy * frame_index


def is_visible(obj: Dict[str, Any], frame_index: int) -> bool:
    if frame_index < obj["enter_frame"]:
        return False
    return obj["exit_frame"] is None or frame_index < obj["exit_frame"]


def ground_truth(scene: Dict[str, Any], frame_index: int) -> List[Dict[str, Any]]:
    """Known objects for a frame: id, color, shape and center. Only visible objects."""
    out = []
    for obj in scene["objects"]:
        if not is_visible(obj, frame_index):
            continue
        cx, cy = _position(obj, frame_index)
        out.append({"id": obj["id"], "color": obj["color"], "shape": obj["shape"], "center": (cx, cy)})
    return out


def render_scene(scene: Dict[str, Any], frame_index: int) -> np.ndarray:
    if cv2 is None:
        raise CameraError("OpenCV no esta instalado. Ejecuta: pip install opencv-python")
    img = np.full((scene["height"], scene["width"], 3), scene["background"], dtype=np.uint8)
    for obj in scene["objects"]:
        if not is_visible(obj, frame_index):
            continue
        cx, cy = _position(obj, frame_index)
        color = BGR[obj["color"]]
        if obj["shape"] in ("square", "rect"):
            w, h = obj["size"]
            x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
            cv2.rectangle(img, (x0, y0), (x0 + int(w) - 1, y0 + int(h) - 1), color, -1)
        elif obj["shape"] == "circle":
            cv2.circle(img, (int(round(cx)), int(round(cy))), int(obj["radius"]), color, -1)
        else:  # triangle, pointing up, bounding size taken from "radius"
            r = float(obj["radius"])
            pts = np.array([[cx, cy - r], [cx + r, cy + r], [cx - r, cy + r]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
    if scene["noise_sigma"] > 0:
        rng = np.random.default_rng(scene["seed"] + frame_index)
        noisy = img.astype(np.float32) + rng.normal(0.0, scene["noise_sigma"], img.shape)
        img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img


class SimulatedSource(FrameSource):
    """A scene rendered frame by frame. Restarts after loop_frames frames."""

    def __init__(self, name_or_path: str):
        self.scene = load_scene(name_or_path)
        self.description = f"simulacion {name_or_path}"
        self._loop = int(self.scene["loop_frames"])
        self.frame_index = 0

    def read(self) -> Optional[np.ndarray]:
        frame = render_scene(self.scene, self.frame_index)
        self.frame_index = (self.frame_index + 1) % self._loop
        return frame

    def truth(self) -> List[Dict[str, Any]]:
        """Ground truth for the frame returned by the last read() call."""
        last = (self.frame_index - 1) % self._loop
        return ground_truth(self.scene, last)
