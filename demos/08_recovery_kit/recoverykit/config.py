"""Configuración de la célula RecoveryKit.

Todo lo que define la escena está aquí: mesa, robot, piezas, cajas, cámaras y
física. No importa Isaac, así que los tests y el servidor web lo usan sin Kit.

Las unidades son SI (m, kg, s, rad salvo que el nombre diga ``_deg``). El marco
del mundo tiene el origen en el suelo, z hacia arriba; la base del robot está
sobre la mesa y mira hacia +x.

Cada valor físico lleva su procedencia en ``SOURCES``. Los que son una
suposición lo dicen; se calibran con las pruebas de ``tools/validate_physics.py``.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERSION = 1

COLORS = {
    # rgb lineal para el material visual; la percepción no usa estos valores
    "red": (0.80, 0.06, 0.05),
    "blue": (0.05, 0.20, 0.85),
    "green": (0.05, 0.60, 0.12),
    "yellow": (0.90, 0.75, 0.05),
}
SHAPES = ("cube", "cylinder", "prism")


@dataclass
class Material:
    static_friction: float
    dynamic_friction: float
    restitution: float


@dataclass
class TableCfg:
    size_m: tuple[float, float, float] = (0.80, 0.70, 0.04)
    top_z_m: float = 0.76
    center_xy_m: tuple[float, float] = (0.15, 0.0)
    material: Material = field(default_factory=lambda: Material(0.45, 0.38, 0.10))


@dataclass
class ServoCfg:
    """Modelo del Feetech STS3215 en una articulación (implícito en PhysX)."""

    stall_torque_nm: float
    no_load_speed_rad_s: float
    stiffness: float
    damping: float
    armature: float
    friction: float


@dataclass
class RobotCfg:
    usd: str = "LEISAAC:robots/so101_follower.usd"
    base_xy_m: tuple[float, float] = (0.0, 0.0)
    # con pan = 0 el brazo del asset apunta a −y de su base; +90° lo orienta a +x
    base_yaw_deg: float = 90.0
    # el colisionador de la base empieza 30.1 mm por encima de su origen
    # (assets/so101/colliders.npz): así la base apoya en el tablero
    base_z_offset_m: float = -0.0301
    # pose de partida y de observación (rad): pan, lift, elbow, wrist_flex, wrist_roll, gripper.
    # Plegado y girado hacia −106° (ángulo en el mundo ≈ −pan), fuera de la zona de
    # trabajo (−65°…8°) y de la caja (42°): la cámara cenital ve todo lo que importa.
    start_joints: tuple[float, ...] = (1.85, -1.65, 1.50, 0.95, 0.0, 0.0)
    arm_servo: ServoCfg = field(default_factory=lambda: ServoCfg(
        stall_torque_nm=1.91, no_load_speed_rad_s=4.71,
        stiffness=30.0, damping=1.0, armature=0.005, friction=0.02))
    gripper_servo: ServoCfg = field(default_factory=lambda: ServoCfg(
        stall_torque_nm=1.91, no_load_speed_rad_s=4.71,
        stiffness=4.0, damping=0.08, armature=0.002, friction=0.01))
    # fuerza de cierre limitada por el controlador (fracción del par de bloqueo)
    gripper_effort_fraction: float = 0.25
    jaw_material: Material = field(default_factory=lambda: Material(0.45, 0.38, 0.10))
    # velocidades y aceleraciones de trayectoria (rad/s, rad/s²)
    max_joint_speed: float = 2.0
    max_joint_accel: float = 6.0


@dataclass
class PieceCfg:
    id: str
    color: str
    shape: str = "cube"
    size_m: float = 0.025          # arista del cubo, diámetro del cilindro, lado del prisma
    height_m: float | None = None  # None = igual que size_m
    mass_kg: float = 0.012
    pos_xy_m: tuple[float, float] | None = None   # None = colocación aleatoria
    yaw_deg: float | None = None                   # None = aleatorio

    @property
    def height(self) -> float:
        return self.height_m if self.height_m is not None else self.size_m


@dataclass
class ContainerCfg:
    id: str
    name: str = "caja"
    center_xy_m: tuple[float, float] = (0.141, 0.127)   # r = 0.19 m, 42°: dentro de la corona alcanzable
    outer_m: tuple[float, float, float] = (0.11, 0.11, 0.045)
    wall_m: float = 0.004
    floor_m: float = 0.004
    color: tuple[float, float, float] = (0.55, 0.42, 0.28)
    fixed: bool = True   # pegada a la mesa; False = caja suelta con masa
    mass_kg: float = 0.10

    @property
    def inner_m(self) -> tuple[float, float]:
        return (self.outer_m[0] - 2 * self.wall_m, self.outer_m[1] - 2 * self.wall_m)


@dataclass
class CameraCfg:
    name: str
    pos_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]
    width: int = 960
    height: int = 720
    focal_length_mm: float = 18.0
    horizontal_aperture_mm: float = 20.955
    mount: str = "world"   # "world" o "wrist"
    enabled: bool = True


@dataclass
class PhysicsCfg:
    dt: float = 1.0 / 240.0
    render_interval: int = 8
    solver_type: int = 1             # 1 = TGS
    robot_position_iterations: int = 32
    robot_velocity_iterations: int = 4
    piece_position_iterations: int = 16
    piece_velocity_iterations: int = 4
    contact_offset_m: float = 0.002
    rest_offset_m: float = 0.0
    bounce_threshold_m_s: float = 0.2
    friction_correlation_m: float = 0.005
    friction_offset_threshold_m: float = 0.01
    piece_material: Material = field(default_factory=lambda: Material(0.42, 0.34, 0.15))
    piece_linear_damping: float = 0.0
    piece_angular_damping: float = 0.02
    max_depenetration_m_s: float = 0.5
    enable_ccd: bool = False
    device: str = "cpu"


@dataclass
class WorkArea:
    """Sector de corona alrededor de la base donde se reparten las piezas.

    Con la pinza vertical el SO-101 sólo llega a todas las alturas del ciclo
    (agarre, aproximación, subida y tránsito) en una corona de 0.16–0.20 m
    alrededor de la base, para cualquier giro de la pieza; medido con
    ``tools/reach_map.py`` (results/reach_map.json). Ángulos desde +x, en grados.
    """

    r_m: tuple[float, float] = (0.155, 0.205)
    angle_deg: tuple[float, float] = (-65.0, 8.0)
    # 25 mm de pieza + apertura de 16 mm + grosor de dedo: ~52 mm entre centros
    min_gap_m: float = 0.055


@dataclass
class LightingCfg:
    dome: float = 300.0
    key: float = 450.0


@dataclass
class SceneCfg:
    version: int = VERSION
    name: str = "demo"
    seed: int = 17000
    table: TableCfg = field(default_factory=TableCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    pieces: list[PieceCfg] = field(default_factory=list)
    containers: list[ContainerCfg] = field(default_factory=list)
    cameras: list[CameraCfg] = field(default_factory=list)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    work_area: WorkArea = field(default_factory=WorkArea)
    lighting: LightingCfg = field(default_factory=LightingCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def piece(self, piece_id: str) -> PieceCfg:
        for p in self.pieces:
            if p.id == piece_id:
                return p
        raise KeyError(piece_id)


SOURCES = {
    "robot.usd": "LeIsaac so101_follower.usd; raíz 'so101_new_calib' = URDF oficial SO-101 "
                 "(TheRobotStudio/SO-ARM100). Masas e inercias de CAD, ver results/asset_leisaac_so101.json.",
    "servo.stall_torque_nm": "STS3215 versión 7.4 V: 19.5 kg·cm = 1.91 N·m (hoja de datos Feetech). "
                             "La versión 12 V da 30 kg·cm; cambiar si el seguidor usa esa.",
    "servo.no_load_speed_rad_s": "STS3215 7.4 V: 0.222 s/60° = 4.71 rad/s (hoja de datos Feetech).",
    "servo.stiffness/damping": "SUPOSICIÓN: sin trazas de hardware. kp = 30 N·m/rad satura el par de "
                               "bloqueo con ~3.6° de error; d = 1.0 ≈ amortiguamiento crítico para la inercia "
                               "reflejada estimada (0.007–0.009 kg·m²). kp = 12 dejaba 0.07 rad de caída por "
                               "gravedad y 0.09 rad de sobreoscilación (results/physics_validation.json).",
    "servo.armature": "SUPOSICIÓN: inercia del rotor reflejada por la reductora 1:345, orden de magnitud.",
    "piece.mass_kg": "Cubo de 25 mm impreso en PLA con ~50 % de relleno ≈ 12 g. Pesar las piezas reales.",
    "material.piece": "PLA sobre PLA/madera: μs 0.3–0.5 en la literatura; 0.42/0.34 elegido en el centro.",
    "material.table": "Tablero de melamina/madera: μ ~0.4–0.5 contra plástico.",
    "physics.dt": "1/240 s: el agarre por fricción necesita pasos cortos; medido en validate_physics.",
}


def default_cameras() -> list[CameraCfg]:
    return [
        CameraCfg("cenital", pos_m=(0.15, 0.0, 1.30), look_at_m=(0.15, 0.0, 0.76)),
        CameraCfg("muñeca", pos_m=(0.0, 0.0, 0.0), look_at_m=(0.0, 0.0, 0.0),
                  width=480, height=360, focal_length_mm=12.0, mount="wrist", enabled=False),
    ]


def preset(name: str = "demo") -> SceneCfg:
    if name == "demo":
        pieces = [
            PieceCfg("rojo_1", "red"), PieceCfg("rojo_2", "red"),
            PieceCfg("azul_1", "blue"), PieceCfg("azul_2", "blue"), PieceCfg("azul_3", "blue"),
        ]
    elif name == "simple":
        pieces = [PieceCfg("rojo_1", "red", pos_xy_m=(0.17, -0.07), yaw_deg=0.0)]
    elif name == "formas":
        pieces = [
            PieceCfg("rojo_1", "red"), PieceCfg("azul_1", "blue"),
            PieceCfg("azul_2", "blue", shape="cylinder"), PieceCfg("verde_1", "green", shape="cylinder"),
        ]
    else:
        raise KeyError(f"preset desconocido: {name}")
    return SceneCfg(name=name, pieces=pieces, containers=[ContainerCfg("caja")], cameras=default_cameras())


def from_dict(data: dict) -> SceneCfg:
    if data.get("version") != VERSION:
        raise ValueError(f"versión de configuración {data.get('version')} != {VERSION}")
    d = copy.deepcopy(data)
    table = d.pop("table")
    robot = d.pop("robot")
    physics = d.pop("physics")
    return SceneCfg(
        version=d["version"], name=d.get("name", "custom"), seed=d.get("seed", 0),
        table=TableCfg(**{**table, "material": Material(**table["material"])}),
        robot=RobotCfg(**{**robot,
                          "arm_servo": ServoCfg(**robot["arm_servo"]),
                          "gripper_servo": ServoCfg(**robot["gripper_servo"]),
                          "jaw_material": Material(**robot["jaw_material"])}),
        pieces=[PieceCfg(**p) for p in d.get("pieces", [])],
        containers=[ContainerCfg(**c) for c in d.get("containers", [])],
        cameras=[CameraCfg(**c) for c in d.get("cameras", [])],
        physics=PhysicsCfg(**{**physics, "piece_material": Material(**physics["piece_material"])}),
        work_area=WorkArea(**d.get("work_area", {})),
        lighting=LightingCfg(**d.get("lighting", {})),
    )


def load(path: str | Path) -> SceneCfg:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate(cfg: SceneCfg) -> list[str]:
    """Errores de configuración legibles, en español. Lista vacía = válida."""
    errors: list[str] = []
    ids = [p.id for p in cfg.pieces]
    if len(set(ids)) != len(ids):
        errors.append("hay piezas con el mismo id")
    for p in cfg.pieces:
        if p.color not in COLORS:
            errors.append(f"{p.id}: color '{p.color}' no soportado")
        if p.shape not in SHAPES:
            errors.append(f"{p.id}: forma '{p.shape}' no soportada")
        elif p.shape == "prism":
            errors.append(f"{p.id}: los prismas todavía no están implementados (usa cubos o cilindros)")
        if not 0.012 <= p.size_m <= 0.032:
            errors.append(f"{p.id}: tamaño {p.size_m*1000:.0f} mm fuera de 12–32 mm (apertura de la pinza)")
        if p.mass_kg <= 0:
            errors.append(f"{p.id}: masa no positiva")
    if any(p.color == "yellow" for p in cfg.pieces):
        errors.append("aviso: el amarillo se confunde con el brazo (también amarillo); "
                      "sólo se distingue cuando el brazo no lo tapa")
    for c in cfg.containers:
        if min(c.inner_m) < 0.04:
            errors.append(f"{c.id}: hueco interior demasiado pequeño")
    if cfg.physics.dt > 1.0 / 120.0:
        errors.append("dt mayor que 1/120 s: el agarre por fricción no es estable")
    return errors


def piece_yaw_rad(p: PieceCfg) -> float:
    return math.radians(p.yaw_deg or 0.0)
