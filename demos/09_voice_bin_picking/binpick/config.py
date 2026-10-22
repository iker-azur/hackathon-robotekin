"""Escena y constantes del demo de clasificación (sin Isaac).

Marco del mundo del demo 08: origen en el suelo, z arriba, base del SO-101 en
(0, 0) sobre la mesa mirando a +x. Visto desde arriba con la cámara cenital, la
derecha de la imagen es +x y abajo es −y; el lado **izquierdo del robot es +y**.

Geometría: el SO-101 con la pinza vertical sólo llega a todas las alturas del
ciclo en una corona estrecha (``08_recovery_kit/results/reach_map.json``). El
contenedor de origen va delante del robot y las dos cajas de destino a sus
lados, dentro de esa corona; ``tools/check_layout_reach.py`` lo comprueba con la
cinemática inversa y escribe el resultado en ``results/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import joints  # noqa: F401  (asegura el sys.path de recoverykit)

# ------------------------------------------------------------------ constantes
FPS = 30                       # frecuencia de control de la política
PHYSICS_DT = 1.0 / 240.0
STEPS_PER_FRAME = 8            # 240 / 30
COLORS = ("red", "green", "blue")   # el amarillo se confunde con el brazo (08_recovery_kit)
COLOR_ES = {"red": ("roja", "rojas"), "green": ("verde", "verdes"), "blue": ("azul", "azules")}
COLOR_ES_M = {"red": ("rojo", "rojos"), "green": ("verde", "verdes"), "blue": ("azul", "azules")}
DESTINATIONS = ("izquierda", "derecha")
BIN_ID = "origen"
DEST_IDS = {"izquierda": "caja_izquierda", "derecha": "caja_derecha"}
CAMERA = "cenital"
VIEW_CAMERA = "vista"
PIECE_SIZE_M = 0.025
MAX_PIECES = 8
# pan, lift, elbow, wrist_flex, wrist_roll, gripper (rad)
# Plegado sobre la base sin tapar contenedor ni cajas (solape 0 en la imagen cenital) Y físicamente posible: la primera
# postura que se eligió sólo con cinemática, (0.1, -1.3, 1.3, 1.6), tiene los eslabones tocándose; en Isaac el par se saturaba
# y el brazo no se desplegaba nunca ("no alcanza la consigna"). Comprobado en Isaac: reposo a < 0.025 rad, par < 0.65 N·m
# y se despliega hacia una pose de trabajo con 0.004 rad de error.
OBSERVE_JOINTS = (0.0, -1.1, 1.2, 1.4, 0.0, 0.0)

# Fantasmas del render: con el antialiasing por defecto (DLSS, /rtx/post/aa/op = 3) la imagen acumula fotogramas anteriores y,
# al cambiar de sitio una pieza, quedan 850–1900 píxeles de la escena previa tras 2 renders (medido en Isaac). Con FXAA (2) no hay
# acumulación temporal y bastan ~8 renders tras un cambio brusco. El bucle de 30 Hz de la política sigue con 2 renders por paso.
RTX_AA_OP = 2
OBSERVE_RENDERS = 8

_ID = {"red": "rojo", "green": "verde", "blue": "azul"}


@dataclass
class BinPickCfg:
    seed: int = 18000
    camera_res: tuple[int, int] = (800, 600)         # ancho, alto de la cámara cenital
    view_camera: bool = False                        # cámara lateral para el panel (cuesta un render más)
    # contenedor de origen
    bin_center_xy_m: tuple[float, float] = (0.19, 0.02)
    bin_outer_m: tuple[float, float, float] = (0.088, 0.208, 0.034)   # alargado en y: ahí el alcance sobra
    # cajas de destino
    left_center_xy_m: tuple[float, float] = (0.07, 0.17)
    right_center_xy_m: tuple[float, float] = (0.07, -0.13)
    box_outer_m: tuple[float, float, float] = (0.115, 0.115, 0.045)
    # piezas: por color, cubos y cilindros (el taller del demo 08 los valida: 30/30 cada uno)
    cubes_per_color: int = 3
    cylinders_per_color: int = 1
    # región de interés que ve la política (mundo, m): contiene el contenedor y las dos cajas
    roi_x_m: tuple[float, float] = (0.0, 0.26)
    roi_y_m: tuple[float, float] = (-0.215, 0.26)
    policy_image_hw: tuple[int, int] = (288, 160)    # alto, ancho tras recortar y reducir

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def piece_pool(cfg: BinPickCfg) -> list[tuple[str, str, str]]:
    """``[(id, color, forma)]``: las piezas que existen en la escena (se activan por escena)."""
    pool = []
    for color in COLORS:
        for i in range(cfg.cubes_per_color):
            pool.append((f"{_ID[color]}_c{i + 1}", color, "cube"))
        for i in range(cfg.cylinders_per_color):
            pool.append((f"{_ID[color]}_y{i + 1}", color, "cylinder"))
    return pool


def build_scene(cfg: BinPickCfg | None = None):
    """``SceneCfg`` del demo 08 con el contenedor, las dos cajas, el conjunto de piezas y las cámaras."""
    from recoverykit import config as rk

    cfg = cfg or BinPickCfg()
    scene = rk.SceneCfg(name="clasificacion", seed=cfg.seed)
    scene.pieces = [rk.PieceCfg(pid, color, shape=shape, size_m=PIECE_SIZE_M)
                    for pid, color, shape in piece_pool(cfg)]
    # postura de observación: elegida por búsqueda para que el brazo plegado no tape ni el contenedor ni las
    # cajas en la imagen cenital (la del demo 08 tapaba con el codo la esquina de una caja)
    scene.robot.start_joints = OBSERVE_JOINTS
    wall = 0.004
    scene.containers = [
        rk.ContainerCfg(BIN_ID, name="contenedor", center_xy_m=cfg.bin_center_xy_m, outer_m=cfg.bin_outer_m,
                        wall_m=wall, floor_m=0.004),
        rk.ContainerCfg(DEST_IDS["izquierda"], name="caja izquierda", center_xy_m=cfg.left_center_xy_m,
                        outer_m=cfg.box_outer_m, wall_m=wall, floor_m=0.004),
        rk.ContainerCfg(DEST_IDS["derecha"], name="caja derecha", center_xy_m=cfg.right_center_xy_m,
                        outer_m=cfg.box_outer_m, wall_m=wall, floor_m=0.004),
    ]
    w, h = cfg.camera_res
    cams = [rk.CameraCfg(CAMERA, pos_m=(0.15, 0.0, 1.30), look_at_m=(0.15, 0.0, 0.76), width=w, height=h,
                       focal_length_mm=14.0)]      # 74° de campo: se ven las dos cajas con sus paredes
    if cfg.view_camera:
        # vista 3/4 desde delante-izquierda para que el panel enseñe el robot trabajando
        cams.append(rk.CameraCfg(VIEW_CAMERA, pos_m=(0.62, 0.42, 1.02), look_at_m=(0.10, 0.0, 0.80),
                                 width=640, height=480, focal_length_mm=16.0))
    scene.cameras = cams
    return scene


def container(scene, cid: str):
    for c in scene.containers:
        if c.id == cid:
            return c
    raise KeyError(cid)


def region_of(scene, x: float, y: float, z: float | None = None) -> str:
    """Dónde está un punto del mundo: ``origen`` | ``izquierda`` | ``derecha`` | ``mesa``."""
    top = scene.table.top_z_m
    for name, cid in ((BIN_ID, BIN_ID), ("izquierda", DEST_IDS["izquierda"]), ("derecha", DEST_IDS["derecha"])):
        c = container(scene, cid)
        if abs(x - c.center_xy_m[0]) <= c.inner_m[0] / 2 and abs(y - c.center_xy_m[1]) <= c.inner_m[1] / 2:
            if z is None or z < top + c.outer_m[2] + 0.06:
                return name
    return "mesa"


def inner_margin_box(scene, cid: str, margin: float) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) del interior de un contenedor recortado ``margin`` por cada lado."""
    c = container(scene, cid)
    hx, hy = c.inner_m[0] / 2 - margin, c.inner_m[1] / 2 - margin
    return (c.center_xy_m[0] - hx, c.center_xy_m[0] + hx, c.center_xy_m[1] - hy, c.center_xy_m[1] + hy)


def polar(x: float, y: float) -> tuple[float, float]:
    """(radio m, ángulo grados) respecto a la base del robot."""
    return math.hypot(x, y), math.degrees(math.atan2(y, x))
