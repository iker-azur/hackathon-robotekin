"""Simulated 3D workspace: a physics scene rendered with MuJoCo.

Objects are rigid bodies on a table. They fall under gravity, rest on the table and
can be knocked over by other objects. Objects with a "velocity" are kinematic: they
follow a fixed path, which stands in for a conveyor. The camera renders with shadows
and directional light. Ground truth is read from the physics state, so the reported
pixel and table positions match what was rendered.

Use it as a source string: "sim3d:demo3d" or "sim3d:path/to/scene.json".
Needs: pip install mujoco
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None

from .camera import CameraError, FrameSource
from .simulation import BGR, is_visible

SIM3D_PREFIX = "sim3d:"
KINDS = ("box", "cylinder")
TIMESTEP = 0.002  # s of simulated time per physics step
STEPS_PER_FRAME = 17  # about 30 frames per simulated second
SETTLE_STEPS = 500  # 1 s of simulated time so resting objects are still before frame 0
DROP_HEIGHT = 0.15  # m above the table for objects that enter later
PARK_Z = -0.5  # m: objects not yet in the scene wait below the table, out of view

BUILTIN_SCENES3D: Dict[str, Dict[str, Any]] = {
    "demo3d": {
        "width": 640, "height": 480, "background": [40, 40, 40], "noise_sigma": 2.0, "seed": 7,
        "loop_frames": 300,
        # Positions in mm, table surface at z = 0. The camera looks at the middle of the table.
        "camera": {"eye": [300, -350, 700], "target": [300, 200, 0], "fovy": 44},
        "table": {"size": [600, 400]},
        "objects": [
            {"id": "cubo_rojo_1", "kind": "box", "color": "rojo", "center": [120, 120], "size": [60, 60, 60]},
            {"id": "cubo_rojo_2", "kind": "box", "color": "rojo", "center": [260, 120], "size": [60, 60, 60]},
            {"id": "cilindro_azul", "kind": "cylinder", "color": "azul", "center": [420, 120],
             "radius": 35, "height": 50},
            {"id": "prisma_verde", "kind": "box", "color": "verde", "center": [150, 220],
             "size": [110, 55, 40], "velocity": [1.2, 0.0]},
            {"id": "cubo_amarillo", "kind": "box", "color": "amarillo", "center": [420, 330],
             "size": [60, 60, 60], "enter_frame": 45},
        ],
    },
}


def _require_mujoco() -> None:
    if mujoco is None:
        raise CameraError("MuJoCo no esta instalado. Ejecuta: pip install mujoco")


def _m(mm: float) -> float:
    return mm / 1000.0


def load_scene3d(name_or_path: str) -> Dict[str, Any]:
    """Return a validated copy of a built-in 3D scene or of a JSON scene file."""
    if name_or_path in BUILTIN_SCENES3D:
        scene = copy.deepcopy(BUILTIN_SCENES3D[name_or_path])
    elif os.path.exists(name_or_path):
        with open(name_or_path, encoding="utf-8") as f:
            scene = json.load(f)
    else:
        raise CameraError(
            f"Escena 3D desconocida: {name_or_path}. Disponibles: {', '.join(BUILTIN_SCENES3D)} "
            "o una ruta a un JSON de escena 3D."
        )
    _validate3d(scene, name_or_path)
    return scene


def _validate3d(scene: Dict[str, Any], label: str) -> None:
    for key in ("width", "height", "background", "camera", "table", "objects"):
        if key not in scene:
            raise CameraError(f"Escena 3D {label}: falta la clave '{key}'.")
    scene.setdefault("noise_sigma", 0.0)
    scene.setdefault("seed", 0)
    scene.setdefault("loop_frames", 300)
    scene["camera"].setdefault("fovy", 44)
    for obj in scene["objects"]:
        if obj.get("color") not in BGR:
            raise CameraError(f"Escena 3D {label}: color no soportado '{obj.get('color')}'. Usa: {', '.join(BGR)}.")
        if obj.get("kind") not in KINDS:
            raise CameraError(f"Escena 3D {label}: tipo no soportado '{obj.get('kind')}'. Usa: {', '.join(KINDS)}.")
        if obj["kind"] == "box" and len(obj.get("size", [])) != 3:
            raise CameraError(f"Escena 3D {label}: un 'box' necesita 'size' = [ancho, fondo, alto] en mm.")
        if obj["kind"] == "cylinder" and ("radius" not in obj or "height" not in obj):
            raise CameraError(f"Escena 3D {label}: un 'cylinder' necesita 'radius' y 'height' en mm.")
        obj.setdefault("velocity", [0.0, 0.0])
        obj.setdefault("enter_frame", 0)
        obj.setdefault("exit_frame", None)
        if any(obj["velocity"]) and obj["enter_frame"] != 0:
            raise CameraError(f"Escena 3D {label}: un objeto con 'velocity' debe estar desde el frame 0.")


def _half_height_mm(obj: Dict[str, Any]) -> float:
    return obj["size"][2] / 2 if obj["kind"] == "box" else obj["height"] / 2


def _shape_label(obj: Dict[str, Any]) -> str:
    if obj["kind"] == "cylinder":
        return "circle"
    w, d, _ = obj["size"]
    return "square" if w == d else "rect"


def _geom_xml(obj: Dict[str, Any]) -> str:
    rgba = " ".join(f"{c / 255:.3f}" for c in reversed(BGR[obj["color"]])) + " 1"  # BGR -> RGB
    if obj["kind"] == "box":
        w, d, h = (_m(v) / 2 for v in obj["size"])
        return f'<geom type="box" size="{w:.4f} {d:.4f} {h:.4f}" rgba="{rgba}" mass="0.05"/>'
    return (f'<geom type="cylinder" size="{_m(obj["radius"]):.4f} {_m(obj["height"]) / 2:.4f}" '
            f'rgba="{rgba}" mass="0.05"/>')


def _body_xml(obj: Dict[str, Any]) -> str:
    x, y = _m(obj["center"][0]), _m(obj["center"][1])
    z = _m(_half_height_mm(obj)) + 0.002
    name = obj["id"]
    if any(obj["velocity"]):  # kinematic: its position is set every frame
        return f'<body name="{name}" mocap="true" pos="{x:.4f} {y:.4f} {z:.4f}">{_geom_xml(obj)}</body>'
    return f'<body name="{name}" pos="{x:.4f} {y:.4f} {z:.4f}"><freejoint name="{name}_joint"/>{_geom_xml(obj)}</body>'


def _scene_xml(scene: Dict[str, Any]) -> str:
    cam = scene["camera"]
    eye = np.asarray(cam["eye"], dtype=float) / 1000.0
    forward = np.asarray(cam["target"], dtype=float) / 1000.0 - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)  # MuJoCo cameras look along their negative z axis
    table_w, table_d = (_m(v) for v in scene["table"]["size"])
    sky = " ".join(f"{c / 255:.3f}" for c in reversed(scene["background"]))
    bodies = "".join(_body_xml(obj) for obj in scene["objects"])
    return f"""<mujoco model="roboqc">
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="{scene['width']}" offheight="{scene['height']}"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="{sky}" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="{table_w / 2:.4f} {table_d / 2:.4f} 1.5" dir="0 0 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="0.9 -0.4 1.0" dir="-0.5 0.6 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -1" rgba="0.2 0.2 0.2 1"/>
    <geom name="table" type="box" size="{table_w / 2:.4f} {table_d / 2:.4f} 0.02"
          pos="{table_w / 2:.4f} {table_d / 2:.4f} -0.02" rgba="0.45 0.45 0.47 1"/>
    <camera name="cam" pos="{eye[0]:.4f} {eye[1]:.4f} {eye[2]:.4f}"
            xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} {up[0]:.4f} {up[1]:.4f} {up[2]:.4f}"
            fovy="{cam['fovy']}"/>
    {bodies}
  </worldbody>
</mujoco>"""


class Sim3DSource(FrameSource):
    """A physics scene rendered frame by frame. Restarts after loop_frames frames."""

    def __init__(self, name_or_path: str):
        _require_mujoco()
        self.scene = load_scene3d(name_or_path)
        self.description = f"simulacion 3D {name_or_path}"
        self._loop = int(self.scene["loop_frames"])
        self.frame_index = 0
        self._truth: List[Dict[str, Any]] = []
        self._objects = {obj["id"]: obj for obj in self.scene["objects"]}
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(self.scene))
        self.data = mujoco.MjData(self.model)
        self._cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        self._renderer = mujoco.Renderer(self.model, self.scene["height"], self.scene["width"])
        half_fov = math.radians(float(self.model.cam_fovy[self._cam])) / 2
        self._focal = (self.scene["height"] / 2) / math.tan(half_fov)

    def read(self):
        t = self.frame_index
        if t == 0:
            self._reset()
        else:
            self._apply_events(t)
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(self.model, self.data)
        frame = self._render()
        self._truth = self._ground_truth(t)
        self.frame_index = (t + 1) % self._loop
        return frame

    def truth(self) -> List[Dict[str, Any]]:
        """Ground truth for the frame returned by the last read() call."""
        return self._truth

    def close(self) -> None:
        self._renderer.close()

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for obj in self.scene["objects"]:
            if any(obj["velocity"]):
                continue
            # Objects that start in the scene rest on the table; the others wait below it.
            z = _m(_half_height_mm(obj)) + 0.002 if obj["enter_frame"] == 0 else PARK_Z
            self._place(obj, z)
        self._apply_events(0)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

    def _apply_events(self, t: int) -> None:
        for obj in self.scene["objects"]:
            if any(obj["velocity"]):
                cx, cy = self._nominal_center(obj, t)
                mocap = int(self.model.body(obj["id"]).mocapid[0])
                self.data.mocap_pos[mocap] = [_m(cx), _m(cy), _m(_half_height_mm(obj)) + 0.002]
            elif obj["enter_frame"] == t and t > 0:
                self._place(obj, _m(_half_height_mm(obj)) + DROP_HEIGHT)

    def _place(self, obj: Dict[str, Any], z: float) -> None:
        x, y = _m(obj["center"][0]), _m(obj["center"][1])
        joint = self.data.joint(obj["id"] + "_joint")
        joint.qpos[:] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
        joint.qvel[:] = 0.0

    @staticmethod
    def _nominal_center(obj: Dict[str, Any], t: int):
        vx, vy = obj["velocity"]
        return obj["center"][0] + vx * t, obj["center"][1] + vy * t

    def _render(self) -> np.ndarray:
        for obj in self.scene["objects"]:
            geom = int(self.model.body(obj["id"]).geomadr[0])
            # Parking below the table can still be visible in a tilted camera.
            self.model.geom_rgba[geom,3] = 1.0 if is_visible(obj,self.frame_index) else 0.0
        self._renderer.update_scene(self.data, camera="cam")
        img = np.ascontiguousarray(self._renderer.render()[:, :, ::-1])  # RGB -> BGR for OpenCV
        if self.scene["noise_sigma"] > 0:
            rng = np.random.default_rng(self.scene["seed"] + self.frame_index)
            noisy = img.astype(np.float32) + rng.normal(0.0, self.scene["noise_sigma"], img.shape)
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        return img

    def _project(self, point_m: np.ndarray) -> tuple:
        rot = self.data.cam_xmat[self._cam].reshape(3, 3)
        p = rot.T @ (point_m - self.data.cam_xpos[self._cam])
        depth = -p[2]
        u = self.scene["width"] / 2 + self._focal * p[0] / depth
        v = self.scene["height"] / 2 - self._focal * p[1] / depth
        return float(u), float(v)

    def _ground_truth(self, t: int) -> List[Dict[str, Any]]:
        out = []
        for obj in self.scene["objects"]:
            if not is_visible(obj, t):
                continue
            center = np.array(self.data.body(obj["id"]).xpos)
            top = center + [0.0, 0.0, _m(_half_height_mm(obj))]
            out.append({
                "id": obj["id"], "color": obj["color"], "shape": _shape_label(obj),
                "center": self._project(top),
                "center_mm": (round(center[0] * 1000, 1), round(center[1] * 1000, 1)),
            })
        return out
