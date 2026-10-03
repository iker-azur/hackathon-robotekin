"""Everything in this demo that talks to Isaac Sim, in one place.

The functions here are extracted from ``demos/04_perception/09_camera_to_scene.py`` so
that demo 10 has its own copy and can change things demo 09 must not. The originals are
untouched and still work; where this file deliberately differs, the docstring says so.

Nothing else in this directory imports Isaac. ``planner.py`` is pure numpy, the command
model is torch, and this is the only module that needs a GPU and Kit to be running, so
the parts worth testing without a simulator stay testable.

``env_loader.py`` belongs to ``demos/01_simulation_basics`` and is imported by path
rather than copied, because it holds two workarounds this environment needs (import h5py
before Kit starts, accept the Omniverse EULA) and a second copy would only drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

import colors
import hollow
import pen as pen_module
import scene_spec
from scene import CameraModel, Frame

HERE = Path(__file__).resolve().parent
ENV_LOADER = HERE.parent / "01_simulation_basics" / "env_loader.py"

TASK = "LeIsaac-SO101-LiftCube-v0"

# How long the objects are given to fall onto the table before anything is looked at.
# Demo 05 uses 20 physics steps for the same job; a 6 mm cube dropped on a flat plane
# needs a few more because it can land on an edge.
SETTLE_STEPS = 40

# A sphere written to rest height is not at rest. The first contact resolves as a small pop,
# and these materials roll far enough to matter: measured, ``place_items`` left only 1.5 mm of
# drift but the balls were still creeping at 2 cm/s afterwards and had drifted 27 mm by the
# time the arm had parked. So settling is a loop with a stop condition, not a fixed count,
# and the residual velocity is zeroed once more afterwards - a ball dropped on a level table
# has no reason to move, so stopping it dead is not cheating.
SETTLE_MAX_STEPS = 600
SETTLE_STILL_M = 0.002

# Physics steps given to the arm to reach its parked pose. The SO-101 servos are
# position-controlled with stiffness 17.8, so 60 steps of 16.7 ms is a second, which is
# plenty for a 1.4 rad elbow move and is not enough for it to drift back.
ARM_PARK_STEPS = 60


def prepare_runtime(headless: bool) -> None:
    """Start Kit, through demos/01_simulation_basics/env_loader.py.

    That module is not copied here on purpose: it holds the two workarounds this
    environment needs (import h5py before Kit, accept the Omniverse EULA) and duplicating
    it would guarantee the two copies drift apart. Demo 09 does the same thing at its own
    line 89, and there are no __init__.py files anywhere in this repository, which is why
    it is loaded by path.
    """
    if not ENV_LOADER.exists():
        raise FileNotFoundError(f"{ENV_LOADER} is missing; demo 01 has to be present")
    spec = importlib.util.spec_from_file_location("demo_env_loader", ENV_LOADER)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {ENV_LOADER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.prepare_runtime(headless=headless)


def rigid_cfg(item: scene_spec.Item):
    """One scene_spec.Item as an IsaacLab RigidObjectCfg.

    Copied verbatim from demo 09, which documents three things about the spawner that are
    easy to get wrong and each cost a two-minute simulator start:

    * ``rigid_props`` must be present even when empty. ``_spawn_geom_from_prim_type``
      only calls ``define_rigid_body_properties`` when it is not None, so leaving it out
      spawns a prim with mass and collision but no ``RigidBodyAPI``, and the environment
      then fails with *"Failed to find a rigid body when resolving
      '/World/envs/env_.*/Scene/cube_blue'"*. Mass is a separate class again:
      ``mass_props=MassPropertiesCfg(mass=...)``.
    * ``collision_props`` goes on the *mesh* child, which is where the spawner puts it.
    * ``spawn_cuboid`` scales a unit cube, so ``size`` is per-axis, not a single edge.

    CHANGED HERE: a hollow tray is not one ``Item``, so this still builds a single solid
    body. ``hollow.py`` builds the real trays; this keeps the demo's *loose* objects.
    """
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    common = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=colors.COLORS[item.color].albedo, roughness=0.7
        ),
    }
    if item.shape == "sphere":
        spawn = sim_utils.SphereCfg(radius=item.size[0] / 2.0, **common)
    else:
        spawn = sim_utils.CuboidCfg(size=item.size, **common)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/" + item.prim,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(item.x, item.y, item.z), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


def build_env(camera: str, items: list[scene_spec.Item], pen: bool = False):
    """The shipped lift-cube task, plus our objects, plus our camera.

    Returns the unwrapped env and the name of the camera sensor that was set up.
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import leisaac.tasks  # noqa: F401  registers the gym task ids

    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)

    # The action manager is only installed by the teleop device, and without it the
    # arm_action term is missing and the environment cannot be built. Demo 05 does the
    # same; here the arm is driven for real, not only parked.
    cfg.use_teleop_device("so101_state_machine")
    cfg.seed = 0
    cfg.recorders = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None  # otherwise lifting an object ends the episode

    # The task randomises the cube by up to 7.5 cm and the camera by 5 mm on every
    # reset. That is right for training a policy and fatal for a measurement, and this
    # demo places everything explicitly instead.
    for name in ("domain_randomize_0", "domain_randomize_1"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)

    # The shipped cameras are removed: one is a metre from the robot base and the other
    # fails to load on this machine (isaacsim.sensors.rtx). Demo 09 notes the same three
    # reasons at its own lines 217-228.
    for term in ("front", "wrist"):
        if hasattr(cfg.scene, term):
            setattr(cfg.scene, term, None)
        if hasattr(cfg.observations.policy, term):
            setattr(cfg.observations.policy, term, None)

    for item in items:
        if item.prim == scene_spec.SHIPPED_CUBE:
            continue  # already a rigid body inside the scene USD
        if item.container:
            # A hollow tray, not a solid slab: one kinematic rigid body with a floor and
            # four walls as its collision shapes. See hollow.py for why this cannot be
            # five sibling objects.
            setattr(cfg.scene, item.prim, hollow.make_tray_asset_cfg(item))
        else:
            setattr(cfg.scene, item.prim, rigid_cfg(item))

    if pen:
        # The containment kerb for the "contained" variant. Same body shape as a tray, four
        # walls and no floor; see pen.py. Spawned last so it exists before anything is placed
        # inside it.
        setattr(cfg.scene, pen_module.PEN_PRIM, pen_module.make_pen_asset_cfg())

    sensor = ensure_camera(cfg, camera)
    return gym.make(TASK, cfg=cfg).unwrapped, sensor


def look_at_opengl(eye, target) -> tuple[float, float, float, float]:
    """A (w, x, y, z) quaternion aiming a camera from eye at target, OpenGL frame.

    Copied from demo 09, including its warning: this asks IsaacLab for the rotation
    matrix in opengl format (``-Z`` forward, ``+Y`` up), so ``up_axis`` is the *string*
    "Z", not a vector.
    """
    import torch
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    eyes = torch.tensor([list(eye)], dtype=torch.float32)
    targets = torch.tensor([list(target)], dtype=torch.float32)
    matrix = create_rotation_matrix_from_view(eyes, targets, "Z")
    quat = quat_from_matrix(matrix).squeeze(0)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def ensure_camera(cfg, name: str) -> str:
    """Make sure the sensor a camera choice names exists, and return its scene name.

    The pose goes in ``CameraCfg.offset`` rather than being applied afterwards, for the
    reason demo 09 records: ``Camera.set_world_poses_from_view`` returned without
    complaint and left the sensor reporting ``pos_w = [0, 0, 0]``, the prim's default, so
    the pose is set by the same call that creates the prim and cannot be lost.
    """
    sensor, add_it = scene_spec.CAMERA_CHOICES[name]
    if not add_it:
        return sensor

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    place = scene_spec.camera_poses()[sensor]
    setattr(
        cfg.scene,
        sensor,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Scene/" + sensor + "_camera",
            update_period=0.0,
            width=place["width"],
            height=place["height"],
            data_types=["rgb", scene_spec.DEPTH],
            offset=CameraCfg.OffsetCfg(
                pos=place["pos"],
                rot=look_at_opengl(place["pos"], place["look_at"]),
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=place["focal_length"],
                horizontal_aperture=38.11,
                clipping_range=(0.01, 20.0),
                lock_camera=True,
            ),
        ),
    )
    return sensor


def aim_camera(env, camera: str, sensor: str) -> None:
    """Check that the camera ended up where ``scene_spec`` said it should.

    The pose was already applied at spawn time, so this only reads it back and fails
    loudly. A camera that is not where the scene JSON claims it is makes every number in
    the report wrong in a way that is very hard to notice later.
    """
    if not scene_spec.CAMERA_CHOICES[camera][1]:
        return
    place = scene_spec.camera_poses()[sensor]
    wanted = np.asarray(place["pos"], dtype=float)
    back = env.scene[sensor].data.pos_w[0].detach().cpu().numpy()
    off = float(np.linalg.norm(back - wanted))
    print(f"  {sensor} camera asked for {np.round(wanted, 4).tolist()}, "
          f"sensor reports {np.round(back, 4).tolist()} (off by {off * 1000:.1f} mm)",
          flush=True)
    if off > 0.002:
        raise RuntimeError(
            f"the {sensor} camera is {off * 1000:.0f} mm from where scene_spec put it. "
            f"Every position would be measured against a camera that is not where it "
            f"claims to be, so this stops here rather than reporting numbers."
        )


def park_arm(env) -> None:
    """Fold the arm up and back, so it is not between the camera and the table.

    Copied from demo 09. The arm's home pose puts its upper links at z 0.2..0.45 directly
    over the working area, which is exactly where a camera looking down from the far side
    has to see: the first render of demo 09 was half yellow arm and half table, with the
    blue cube and the green ball completely hidden behind the gripper.

    Written as joint positions plus physics steps rather than a root pose, because this is
    an articulation. The angles are the ones demo 05 already drives through on every pick.
    """
    import torch

    robot = env.scene["robot"]
    names = list(robot.joint_names)
    folded = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 1.30,
        "elbow_flex": 1.45,
        "wrist_flex": 0.60,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }
    missing = [name for name in folded if name not in names]
    if missing:
        print(f"  NOTE: the arm has no {', '.join(missing)}; not parking it", flush=True)
        return
    positions = torch.tensor([[folded[name] for name in names]], device=env.device)
    with torch.inference_mode():
        for _ in range(ARM_PARK_STEPS):
            robot.write_joint_position_to_sim(positions)
            robot.write_joint_velocity_to_sim(torch.zeros_like(positions))
            env.sim.step(render=False)
        robot.write_joint_position_to_sim(positions)
    env.scene.update(dt=env.physics_dt)
    print(f"  arm parked: {', '.join(f'{k}={v}' for k, v in folded.items())}", flush=True)


def robot_frame(env) -> Frame:
    """The robot's own idea of where left is, rather than an assumption about it.

    Computed from the base quaternion so that moving the robot does not silently break
    left and right.
    """
    from scene import quat_to_matrix

    robot = env.scene["robot"]
    quat = robot.data.root_quat_w[0].detach().cpu().numpy().tolist()
    rotation = quat_to_matrix(quat)
    forward = rotation[:, 1]  # the base's own +y is where it faces
    up = rotation[:, 2]
    left = np.cross(up, forward)
    position = robot.data.root_pos_w[0].detach().cpu().numpy()
    return Frame.from_robot((float(position[0]), float(position[1])), left, forward)


def _rest(env, items: list[scene_spec.Item]) -> int:
    """Step physics until nothing on the table is moving. Returns the steps taken."""
    import torch

    for count in range(1, SETTLE_MAX_STEPS + 1):
        env.sim.step(render=False)
        worst = 0.0
        for item in items:
            speed = env.scene[item.prim].data.root_lin_vel_w[0]
            worst = max(worst, float(torch.linalg.norm(speed)))
        if worst < SETTLE_STILL_M:
            env.scene.update(dt=env.physics_dt)
            return count
    env.scene.update(dt=env.physics_dt)
    return SETTLE_MAX_STEPS


def place_items(env, items: list[scene_spec.Item]) -> None:
    """Write every object's pose and let it settle, the way demo 05 places the cube."""
    import torch

    device = env.device
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    zeros = torch.zeros(1, 6, device=device)
    for item in items:
        obj = env.scene[item.prim]
        position = torch.tensor([[item.x, item.y, item.z]], device=device)
        obj.write_root_pose_to_sim(torch.cat([position, upright], dim=-1))
        obj.write_root_velocity_to_sim(zeros)
    _rest(env, items)
    # Zero once more: the first contact leaves a residual roll that a level table will not
    # stop on its own. See SETTLE_STILL_M.
    for item in items:
        env.scene[item.prim].write_root_velocity_to_sim(zeros)
    _rest(env, items)


def capture(env, name: str) -> tuple[np.ndarray, np.ndarray | None, CameraModel]:
    """Render one frame from a camera sensor and read it as numpy.

    The sequence matters. ``sim.render()`` produces the frame, and only then is the sensor
    asked to refill its buffers with ``force_recompute=True``: without that last argument
    a sensor with an update period can hand back the previous frame.

    ``sim.render()`` really does draw in ``--headless`` mode, because passing
    ``enable_cameras=True`` to AppLauncher (which env_loader.prepare_runtime does) sets
    the render mode to PARTIAL_RENDERING rather than NO_RENDERING.
    """
    sensor = env.scene[name]
    env.sim.render()
    sensor.update(env.physics_dt, force_recompute=True)

    rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    bgr = rgb[..., ::-1].copy()  # the annotator gives RGB, OpenCV wants BGR

    depth = None
    if scene_spec.DEPTH in sensor.data.output:
        raw = sensor.data.output[scene_spec.DEPTH][0].detach().cpu().numpy()
        if not hasattr(capture, "_depth_shape_reported"):
            capture._depth_shape_reported = True
            print(f"  raw depth shape {raw.shape}, squeezing to 2-D")
        depth = np.squeeze(raw).astype(np.float64)
        if depth.ndim != 2:
            depth = depth.reshape(depth.shape[-2], depth.shape[-1])

    height, width = sensor.data.image_shape
    camera = CameraModel(
        pos_w=sensor.data.pos_w[0].detach().cpu().numpy().tolist(),
        quat_wxyz=sensor.data.quat_w_world[0].detach().cpu().numpy().tolist(),
        matrix=sensor.data.intrinsic_matrices[0].detach().cpu().numpy().tolist(),
        width=width,
        height=height,
    )
    return bgr, depth, camera