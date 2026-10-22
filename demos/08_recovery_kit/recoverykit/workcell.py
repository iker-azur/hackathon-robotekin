"""Construcción de la célula en Isaac Sim y acceso a su estado físico.

Sólo este módulo y ``audit.py`` leen poses del simulador. El agente recibe
imágenes y la propiocepción del robot (articulaciones y pinza) a través de
``Workcell.proprio()``; nunca la posición verdadera de una pieza.

Las únicas escrituras de pose están en ``reset_layout``: colocan la escena
antes de que el agente empiece, con el vigilante desarmado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import COLORS, SceneCfg
from .physics import Watchdog

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"


def resolve_usd(path: str) -> str:
    if path.startswith("LEISAAC:"):
        from huggingface_hub import snapshot_download

        root = snapshot_download("LightwheelAI/leisaac_env", local_files_only=True)
        return str(Path(root) / "assets" / path.split(":", 1)[1])
    return path


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def base_transform(cfg: SceneCfg) -> np.ndarray:
    """Pose 4×4 del eslabón base en el mundo; la misma que usa ``kinematics.fk``."""
    from .kinematics import yaw_transform

    r = cfg.robot
    return yaw_transform(r.base_xy_m[0], r.base_xy_m[1], cfg.table.top_z_m + r.base_z_offset_m,
                         math.radians(r.base_yaw_deg))


def _material_cfg(m):
    import isaaclab.sim as sim_utils

    return sim_utils.RigidBodyMaterialCfg(
        static_friction=m.static_friction, dynamic_friction=m.dynamic_friction, restitution=m.restitution,
        friction_combine_mode="average", restitution_combine_mode="average")


def _piece_spawn(cfg: SceneCfg, p):
    import isaaclab.sim as sim_utils

    ph = cfg.physics
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=ph.piece_position_iterations,
            solver_velocity_iteration_count=ph.piece_velocity_iterations,
            linear_damping=ph.piece_linear_damping,
            angular_damping=ph.piece_angular_damping,
            max_depenetration_velocity=ph.max_depenetration_m_s,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=p.mass_kg),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=ph.contact_offset_m, rest_offset=ph.rest_offset_m),
        physics_material=_material_cfg(ph.piece_material),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS[p.color], roughness=0.6),
    )
    if p.shape == "cube":
        return sim_utils.CuboidCfg(size=(p.size_m, p.size_m, p.height), **common)
    if p.shape == "cylinder":
        return sim_utils.CylinderCfg(radius=p.size_m / 2, height=p.height, axis="Z", **common)
    if p.shape == "prism":
        from .prism import TriangularPrismCfg

        return TriangularPrismCfg(side=p.size_m, height=p.height, **common)
    raise ValueError(p.shape)


def _static_box(prim_path: str, size, pos, color, material) -> None:
    import isaaclab.sim as sim_utils

    spawn = sim_utils.CuboidCfg(
        size=tuple(size),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=_material_cfg(material),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.8),
    )
    spawn.func(prim_path, spawn, translation=tuple(pos))


@dataclass
class Workcell:
    cfg: SceneCfg
    sim: object
    robot: object
    pieces: dict[str, object]
    cameras: dict[str, object]
    watchdog: Watchdog
    arm_ids: list[int] = field(default_factory=list)
    gripper_id: int = -1
    step_count: int = 0
    sim_time: float = 0.0
    # fallo de actuador inyectado: consigna de la pinza impuesta (perturbación C)
    gripper_override: float | None = None
    step_hooks: list = field(default_factory=list)

    # ---------------------------------------------------------------- pasos
    def step(self, render: bool = False) -> None:
        for hook in self.step_hooks:
            hook()
        if self.gripper_override is not None:
            t = torch.tensor([[self.gripper_override]], dtype=torch.float32, device=self.robot.device)
            self.robot.set_joint_position_target(t, joint_ids=[self.gripper_id])
        self.robot.write_data_to_sim()
        for obj in self.pieces.values():
            obj.write_data_to_sim()
        self.sim.step(render=render)
        dt = self.cfg.physics.dt
        self.robot.update(dt)
        for obj in self.pieces.values():
            obj.update(dt)
        self.step_count += 1
        self.sim_time += dt
        if self.watchdog.armed:
            pos, speed = self._piece_kinematics()
            self.watchdog.check(pos, speed)

    def render(self) -> None:
        self.sim.render()
        for cam in self.cameras.values():
            cam.update(0.0, force_recompute=True)

    # ---------------------------------------------------------------- robot
    def set_arm_targets(self, q: np.ndarray) -> None:
        t = torch.as_tensor(np.asarray(q, dtype=np.float32), device=self.robot.device).unsqueeze(0)
        self.robot.set_joint_position_target(t, joint_ids=self.arm_ids)

    def set_gripper_target(self, q: float) -> None:
        t = torch.tensor([[q]], dtype=torch.float32, device=self.robot.device)
        self.robot.set_joint_position_target(t, joint_ids=[self.gripper_id])

    def proprio(self) -> dict:
        """Lo que un SO-101 real reporta: posición, velocidad y carga de cada servo."""
        d = self.robot.data
        q = d.joint_pos[0].cpu().numpy()
        qd = d.joint_vel[0].cpu().numpy()
        effort = d.applied_torque[0].cpu().numpy()
        return {"arm_q": q[self.arm_ids].copy(), "arm_qd": qd[self.arm_ids].copy(),
                "gripper_q": float(q[self.gripper_id]), "gripper_qd": float(qd[self.gripper_id]),
                "gripper_effort": float(effort[self.gripper_id]), "arm_effort": effort[self.arm_ids].copy()}

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Pose de un eslabón del robot (cinemática directa del simulador)."""
        idx = self.robot.body_names.index(name)
        d = self.robot.data
        return d.body_pos_w[0, idx].cpu().numpy().copy(), d.body_quat_w[0, idx].cpu().numpy().copy()

    # ------------------------------------------------------- estado físico
    def _piece_kinematics(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        pos, speed = {}, {}
        for name, obj in self.pieces.items():
            pos[name] = obj.data.root_pos_w[0].cpu().numpy()
            speed[name] = float(torch.linalg.norm(obj.data.root_lin_vel_w[0]).item())
        return pos, speed

    def write_joint_state(self, q: np.ndarray) -> None:
        """Sólo para el reinicio: coloca el brazo sin pasar por la dinámica."""
        full = self.robot.data.default_joint_pos.clone()
        full[0, :] = torch.as_tensor(q, dtype=torch.float32, device=self.robot.device)
        zeros = torch.zeros_like(full)
        self.robot.write_joint_state_to_sim(full, zeros)
        self.robot.set_joint_position_target(full)

    def reset_layout(self, layout: dict[str, tuple[float, float, float]], settle_s: float = 0.6) -> None:
        """Coloca piezas y brazo en la pose de partida y deja asentar la escena.

        Es la única escritura de pose del demo. Las piezas sin sitio en
        ``layout`` se aparcan fuera de la mesa (y no cuentan para la escena).
        """
        self.watchdog.disarm()
        top = self.cfg.table.top_z_m
        for i, (name, obj) in enumerate(self.pieces.items()):
            p = self.cfg.piece(name)
            if name in layout:
                x, y, yaw = layout[name]
                z = self.support_z(x, y) + p.height / 2 + 0.0005
            else:
                x, y, yaw, z = -1.0 - 0.1 * i, 1.5, 0.0, 0.02
            pose = torch.tensor([[x, y, z, *yaw_quat(yaw)]], dtype=torch.float32, device=obj.device)
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=obj.device))
        q = np.zeros(len(self.robot.joint_names), dtype=np.float32)
        start = self.cfg.robot.start_joints
        for i, jid in enumerate(self.arm_ids):
            q[jid] = start[i]
        q[self.gripper_id] = start[5]
        self.write_joint_state(q)
        for _ in range(int(settle_s / self.cfg.physics.dt)):
            self.step()
        self.watchdog.arm(self._piece_kinematics()[0])

    # ------------------------------------------------- fuerzas (perturbaciones)
    def apply_force(self, name: str, force_n, torque_nm=(0.0, 0.0, 0.0)) -> None:
        """Fuerza en el centro de masa, marco del mundo; se aplica en cada paso hasta ``clear_force``."""
        obj = self.pieces[name]
        f = torch.tensor([[force_n]], dtype=torch.float32, device=obj.device)
        t = torch.tensor([[torque_nm]], dtype=torch.float32, device=obj.device)
        obj.set_external_force_and_torque(f, t, is_global=True)

    def clear_force(self, name: str) -> None:
        self.apply_force(name, (0.0, 0.0, 0.0))

    # -------------------------------------- verdad del simulador (no para el agente)
    def truth(self, name: str) -> dict:
        """Pose y velocidad reales de una pieza. Sólo ``audit.py`` y las herramientas de prueba."""
        d = self.pieces[name].data
        return {"pos": d.root_pos_w[0].cpu().numpy().copy(), "quat": d.root_quat_w[0].cpu().numpy().copy(),
                "lin_vel": d.root_lin_vel_w[0].cpu().numpy().copy(), "ang_vel": d.root_ang_vel_w[0].cpu().numpy().copy()}

    def viz_poses(self) -> dict:
        """Poses de eslabones y piezas para la vista 3D del panel (ventana a la simulación)."""
        d = self.robot.data
        links = {n: [d.body_pos_w[0, i].tolist(), d.body_quat_w[0, i].tolist()]
                 for i, n in enumerate(self.robot.body_names)}
        pieces = {}
        for n in self.pieces:
            t = self.truth(n)
            p = self.cfg.piece(n)
            pieces[n] = {"pos": t["pos"].tolist(), "quat": t["quat"].tolist(), "color": p.color,
                         "shape": p.shape, "size": p.size_m, "height": p.height}
        return {"t": round(self.sim_time, 3), "links": links, "pieces": pieces}

    def place_for_test(self, name: str, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Escritura de pose para montar una prueba física; desarma el vigilante."""
        self.watchdog.disarm()
        obj = self.pieces[name]
        pose = torch.tensor([[x, y, z, *yaw_quat(yaw)]], dtype=torch.float32, device=obj.device)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=obj.device))

    def arm_watchdog(self) -> None:
        self.watchdog.arm(self._piece_kinematics()[0])

    def support_z(self, x: float, y: float) -> float:
        """Altura de apoyo en (x, y): el suelo de una caja, el tablero o el suelo."""
        t = self.cfg.table
        top = t.top_z_m
        for c in self.cfg.containers:
            if abs(x - c.center_xy_m[0]) < c.inner_m[0] / 2 and abs(y - c.center_xy_m[1]) < c.inner_m[1] / 2:
                return top + c.floor_m
        if abs(x - t.center_xy_m[0]) <= t.size_m[0] / 2 and abs(y - t.center_xy_m[1]) <= t.size_m[1] / 2:
            return top
        return 0.0

    def camera_calibration(self, name: str) -> dict:
        """Intrínsecos y pose (marco óptico ROS → mundo); lo que daría una calibración real."""
        from .kinematics import quat_to_mat

        cam = self.cameras[name]
        return {"k": cam.data.intrinsic_matrices[0].cpu().numpy().copy(),
                "rot": quat_to_mat(cam.data.quat_w_ros[0].cpu().numpy()),
                "pos": cam.data.pos_w[0].cpu().numpy().copy(),
                "width": int(cam.cfg.width), "height": int(cam.cfg.height)}

    def frame(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Última imagen RGB (uint8) y profundidad (m) de una cámara; llamar tras ``render()``."""
        out = self.cameras[name].data.output
        rgb = out["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = out["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32)
        return rgb, depth

    def active_pieces(self) -> list[str]:
        """Piezas que están sobre la mesa (no aparcadas fuera de escena)."""
        pos, _ = self._piece_kinematics()
        top = self.cfg.table.top_z_m
        return [n for n, p in pos.items() if p[2] > top - 0.2]


def build(cfg: SceneCfg, sim) -> Workcell:
    """Crea la escena en el stage. Llamar después de crear ``SimulationContext``."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

    top = cfg.table.top_z_m
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    # luz medida para que las caras superiores no se saturen de blanco
    # (tools/capture_perception.py imprime la saturación de cada clase)
    light = sim_utils.DomeLightCfg(intensity=cfg.lighting.dome, color=(1.0, 1.0, 1.0))
    light.func("/World/light", light)
    key = sim_utils.DistantLightCfg(intensity=cfg.lighting.key, angle=2.0)
    key.func("/World/key_light", key, orientation=(0.94, 0.17, 0.26, 0.0))

    # mesa: tablero estático, gris medio para que los colores contrasten
    t = cfg.table
    _static_box("/World/table/top", t.size_m, (t.center_xy_m[0], t.center_xy_m[1], top - t.size_m[2] / 2),
                (0.36, 0.37, 0.39), t.material)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        leg = (t.center_xy_m[0] + sx * (t.size_m[0] / 2 - 0.04), t.center_xy_m[1] + sy * (t.size_m[1] / 2 - 0.04),
               (top - t.size_m[2]) / 2)
        _static_box(f"/World/table/leg_{i}", (0.05, 0.05, top - t.size_m[2]), leg, (0.25, 0.25, 0.27), t.material)

    # cajas: suelo y cuatro paredes estáticas, fijadas a la mesa
    for c in cfg.containers:
        cx, cy = c.center_xy_m
        ox, oy, oz = c.outer_m
        base = f"/World/containers/{c.id}"
        _static_box(f"{base}/floor", (ox, oy, c.floor_m), (cx, cy, top + c.floor_m / 2), c.color, t.material)
        h = oz - c.floor_m
        zc = top + c.floor_m + h / 2
        _static_box(f"{base}/wall_px", (c.wall_m, oy, h), (cx + ox / 2 - c.wall_m / 2, cy, zc), c.color, t.material)
        _static_box(f"{base}/wall_nx", (c.wall_m, oy, h), (cx - ox / 2 + c.wall_m / 2, cy, zc), c.color, t.material)
        _static_box(f"{base}/wall_py", (ox - 2 * c.wall_m, c.wall_m, h), (cx, cy + oy / 2 - c.wall_m / 2, zc),
                    c.color, t.material)
        _static_box(f"{base}/wall_ny", (ox - 2 * c.wall_m, c.wall_m, h), (cx, cy - oy / 2 + c.wall_m / 2, zc),
                    c.color, t.material)

    # robot
    r = cfg.robot
    ph = cfg.physics
    arm, grip = r.arm_servo, r.gripper_servo
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_usd(r.usd),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=ph.max_depenetration_m_s),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=ph.robot_position_iterations,
                solver_velocity_iteration_count=ph.robot_velocity_iterations,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=ph.contact_offset_m, rest_offset=ph.rest_offset_m),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(r.base_xy_m[0], r.base_xy_m[1], top + r.base_z_offset_m),
            rot=yaw_quat(math.radians(r.base_yaw_deg)),
            joint_pos={name: r.start_joints[i] for i, name in enumerate(ARM_JOINTS)} | {GRIPPER_JOINT: r.start_joints[5]},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINTS,
                effort_limit_sim=arm.stall_torque_nm, velocity_limit_sim=arm.no_load_speed_rad_s,
                stiffness=arm.stiffness, damping=arm.damping, armature=arm.armature, friction=arm.friction),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT],
                effort_limit_sim=grip.stall_torque_nm * r.gripper_effort_fraction,
                velocity_limit_sim=grip.no_load_speed_rad_s,
                stiffness=grip.stiffness, damping=grip.damping, armature=grip.armature, friction=grip.friction),
        },
    )
    robot = Articulation(robot_cfg)

    pieces: dict[str, object] = {}
    for i, p in enumerate(cfg.pieces):
        pieces[p.id] = RigidObject(RigidObjectCfg(
            prim_path=f"/World/pieces/{p.id}",
            spawn=_piece_spawn(cfg, p),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0 - 0.1 * i, 1.5, 0.02)),
        ))

    cameras = _build_cameras(cfg)

    sim.reset()
    _set_jaw_material(robot, r.jaw_material)

    arm_ids = [robot.joint_names.index(n) for n in ARM_JOINTS]
    gripper_id = robot.joint_names.index(GRIPPER_JOINT)
    for c_cfg in cfg.cameras:
        cam = cameras.get(c_cfg.name)
        if cam is not None and c_cfg.mount == "world" and not looks_down(c_cfg):
            cam.set_world_poses_from_view(
                torch.tensor([c_cfg.pos_m], dtype=torch.float32, device=cam.device),
                torch.tensor([c_cfg.look_at_m], dtype=torch.float32, device=cam.device))

    min_half = {}
    for p in cfg.pieces:
        min_half[p.id] = min(p.size_m * (math.sqrt(3) / 2 if p.shape == "prism" else 1.0), p.height) / 2
    cell = Workcell(cfg=cfg, sim=sim, robot=robot, pieces=pieces, cameras=cameras,
                    watchdog=Watchdog(dt=ph.dt, support=lambda x, y: 0.0, min_half=min_half),
                    arm_ids=arm_ids, gripper_id=gripper_id)
    cell.watchdog.support = cell.support_z
    return cell


def _build_cameras(cfg: SceneCfg) -> dict[str, object]:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    cams = {}
    for c in cfg.cameras:
        if not c.enabled:
            continue
        prim = "/World/cameras/" + _ascii(c.name)
        if c.mount == "wrist":
            prim = "/World/Robot/gripper/wrist_cam"
        cams[c.name] = Camera(CameraCfg(
            prim_path=prim,
            update_period=0.0,
            height=c.height,
            width=c.width,
            data_types=["rgb", "distance_to_image_plane"],
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=c.focal_length_mm, horizontal_aperture=c.horizontal_aperture_mm,
                clipping_range=(0.02, 4.0)),
            offset=CameraCfg.OffsetCfg(pos=c.pos_m, rot=camera_rotation_ros(c), convention="ros"),
        ))
    return cams


def looks_down(c) -> bool:
    d = np.asarray(c.look_at_m, float) - np.asarray(c.pos_m, float)
    return abs(d[2]) > 0.999 * float(np.linalg.norm(d))


def camera_rotation_ros(c) -> tuple[float, float, float, float]:
    """Orientación (w, x, y, z) del marco óptico ROS de una cámara cenital.

    Mirando en vertical el "mirar a" con z arriba no define el giro; se fija:
    eje óptico = −Z, derecha de la imagen = +X, abajo de la imagen = −Y. Es un
    giro de 180° alrededor de X.
    """
    return (0.0, 1.0, 0.0, 0.0) if looks_down(c) else (1.0, 0.0, 0.0, 0.0)


def _ascii(name: str) -> str:
    return name.replace("ñ", "n").replace(" ", "_")


def _set_jaw_material(robot, m) -> None:
    """Fija fricción y restitución de las formas de colisión de pinza y mordaza.

    Los colisionadores del asset son instancias USD y PhysX ignora un material
    enlazado sobre ellas (medido: la fuerza de arranque no cambiaba con μ,
    results/grip_capacity_*.json). Se escribe directamente en PhysX, forma por
    forma, como hace la aleatorización de materiales de Isaac Lab.
    """
    view = robot.root_physx_view
    counts = []
    for link_path in view.link_paths[0]:
        counts.append((link_path.rsplit("/", 1)[-1], robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes))
    if sum(c for _, c in counts) != view.max_shapes:
        raise RuntimeError("no se pudo asociar formas de colisión a eslabones")
    mats = view.get_material_properties()
    start = 0
    changed = 0
    for name, n in counts:
        if name in ("gripper", "jaw"):
            mats[:, start:start + n, 0] = m.static_friction
            mats[:, start:start + n, 1] = m.dynamic_friction
            mats[:, start:start + n, 2] = m.restitution
            changed += n
        start += n
    view.set_material_properties(mats, torch.arange(mats.shape[0]))
    print(f"[célula] material de mordazas (μ {m.static_friction}/{m.dynamic_friction}) en {changed} formas", flush=True)
