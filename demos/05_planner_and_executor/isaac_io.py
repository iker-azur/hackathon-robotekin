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
import math
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

# The RTX camera can publish a black or flat-grey render target for several renders after Kit
# starts, after the scene is reset, and occasionally after changing the active camera. The
# dashboard used to interpret that buffer as an empty table and abort a valid task. Re-render a
# small bounded number of times before perception is allowed to consume the frame.
CAMERA_CAPTURE_MAX_ATTEMPTS = 8
CAMERA_CAPTURE_RECOVERY_STEPS = 30
CAMERA_CAPTURE_REINITIALIZE_STEPS = 60
CAMERA_FRAME_MIN_LUMA_RANGE = 12.0

# Physics steps given to the arm to reach its parked pose. The SO-101 servos are
# position-controlled with stiffness 17.8, so 60 steps of 16.7 ms is a second, which is
# plenty for a 1.4 rad elbow move and is not enough for it to drift back.
ARM_PARK_STEPS = 60

# Damping, and why only for balls.
# ----------------------------------
# ``RigidBodyPropertiesCfg()`` with no arguments means zero linear and zero angular damping,
# which is a legal value and a useless ball. A sphere with neither has no reason to stop
# rolling: measured on this scene, a 10 cm/s shove sends it 353 mm along a clear stretch of
# table and it is STILL MOVING after 11.7 s; a 45 cm/s shove sends it 488 mm. That is the
# whole of "the balls move with anything touching them", and it is a property of the spawn
# rather than of the planner, so no amount of replanning can answer it.
#
# Measured sweep, ``tools/probe_ball_physics.py --section roll``, a 4 cm ball shoved along a
# clear 58 cm strip of table, distance rolled and time to a stop under 2 mm/s:
#
#     linear  angular  friction   10 cm/s   25 cm/s   45 cm/s   residual creep
#         0        0      0.5        353 mm    488 mm    488 mm    49 mm/s
#         0        2      0.5        270 mm    368 mm    408 mm    24 mm/s
#         2        8      0.5        149 mm    163 mm    181 mm     9 mm/s
#        10       40      0.5         33 mm     37 mm     45 mm    2.6 mm/s
#        25      100      0.5          9 mm     11 mm     14 mm    0.7 mm/s
#
# 10 and 40 are the pick. A shove of any speed the demo can produce moves the ball 33-45 mm:
# enough that being touched is visibly something, small enough that it stops where it was
# knocked and stays there. The next setting up, 25 and 100, only travels 9-14 mm, which is
# closer to a ball glued to the table than to one an arm can reposition.
#
# Two knobs that sound like they should help were measured and do not:
#
# * **Friction.** Raising it from 0.5 to 1.0 changed the roll to the millimetre at every
#   damping setting - 33/37/45 mm either way. A rolling ball is not held back by sliding
#   friction, which is the whole reason it rolls.
# * **PhysX's own rolling resistance**, ``torsional_patch_radius``. Set to 0.02 it returns
#   numbers identical to no patch at all, on every run. It is not inert by accident in this
#   build; it simply does not act on a sphere resting on a plane here.
#
# Only balls get damping. A cube rests on a flat face and does not roll, and the docs already
# measure cubes at 0.0 mm of drift under a park sweep while balls move, so damping the cubes
# would be a change with nothing to fix behind it.
BALL_LINEAR_DAMPING = 10.0
BALL_ANGULAR_DAMPING = 40.0


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

    rigid_props = (sim_utils.RigidBodyPropertiesCfg(
        linear_damping=BALL_LINEAR_DAMPING,
        angular_damping=BALL_ANGULAR_DAMPING)
        if item.kind == "ball" else sim_utils.RigidBodyPropertiesCfg())
    common = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
        "rigid_props": rigid_props,
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
        robot.write_joint_position_to_sim(positions)
        robot.write_joint_velocity_to_sim(torch.zeros_like(positions))
        for _ in range(ARM_PARK_STEPS):
            env.sim.step(render=False)
        robot.write_joint_position_to_sim(positions)
    env.scene.update(dt=env.physics_dt)
    print(f"  arm parked: {', '.join(f'{k}={v}' for k, v in folded.items())}", flush=True)


def hold_arm_pose(env, steps: int = 30, positions=None):
    """Freeze the arm at its measured pose while a scene is rewritten and settled.

    By default this does not choose a new arm configuration. It captures the current articulation
    state, clears residual velocity, and reasserts that state during raw physics ticks so an old
    IK target cannot sweep through freshly spawned objects. The next normal action owns the arm.

    ``positions`` is the exception: hold the arm at that joint configuration instead, which is
    how a later park is put back into the exact posture the first one produced.
    """
    import torch

    robot = env.scene["robot"]
    positions = robot.data.joint_pos.clone() if positions is None else positions.clone()
    zeros = torch.zeros_like(positions)
    with torch.inference_mode():
        # Updating qpos alone does not cancel the previous articulation controller target.
        # Once raw simulation ticks resume, that stale target can move the arm away from the
        # captured pose and sweep through objects placed during the hold.
        robot.set_joint_position_target(positions)
        for _ in range(steps):
            robot.set_joint_position_target(positions)
            robot.write_joint_position_to_sim(positions)
            robot.write_joint_velocity_to_sim(zeros)
            env.sim.step(render=False)
            robot.set_joint_position_target(positions)
            robot.write_joint_position_to_sim(positions)
            robot.write_joint_velocity_to_sim(zeros)
            env.scene.update(dt=env.physics_dt)
    return robot.data.joint_pos.clone()


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


def _rest(env, items: list[scene_spec.Item], held_joint_pos=None) -> int:
    """Step physics until nothing on the table is moving. Returns the steps taken."""
    import torch

    for count in range(1, SETTLE_MAX_STEPS + 1):
        if held_joint_pos is not None:
            robot = env.scene["robot"]
            with torch.inference_mode():
                robot.set_joint_position_target(held_joint_pos)
                robot.write_joint_position_to_sim(held_joint_pos)
                robot.write_joint_velocity_to_sim(torch.zeros_like(held_joint_pos))
        env.sim.step(render=False)
        if held_joint_pos is not None:
            # Keep the arm out of the newly populated objects while the old articulation action
            # target is explicitly replaced by the held scene-reset pose. The next deliberate
            # arm action takes ownership when settling returns.
            with torch.inference_mode():
                robot.set_joint_position_target(held_joint_pos)
                robot.write_joint_position_to_sim(held_joint_pos)
                robot.write_joint_velocity_to_sim(torch.zeros_like(held_joint_pos))
        worst = 0.0
        for item in items:
            speed = env.scene[item.prim].data.root_lin_vel_w[0]
            worst = max(worst, float(torch.linalg.norm(speed)))
        if worst < SETTLE_STILL_M:
            env.scene.update(dt=env.physics_dt)
            return count
    env.scene.update(dt=env.physics_dt)
    return SETTLE_MAX_STEPS


def root_is_kinematic(actor) -> bool | None:
    """Read the root body's USD kinematic flag; None means it could not be verified."""
    try:
        import omni.usd
        from pxr import UsdPhysics

        paths = actor.root_physx_view.prim_paths
        if not paths:
            return None
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(str(paths[0]))
        if not prim or not prim.IsValid():
            return None
        body = UsdPhysics.RigidBodyAPI(prim)
        if not body:
            return None
        attr = body.GetKinematicEnabledAttr()
        if not attr:
            return False
        value = attr.Get()
        return bool(value) if value is not None else False
    except Exception:
        return None


def set_root_kinematic(actor, enabled: bool) -> bool:
    """Explicitly author the root body's kinematic flag before an environment reset."""
    import omni.usd
    from pxr import UsdPhysics

    paths = actor.root_physx_view.prim_paths
    if not paths:
        raise RuntimeError("the rigid actor has no root PhysX prim path")
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(str(paths[0]))
    body = UsdPhysics.RigidBodyAPI(prim)
    if not prim or not prim.IsValid() or not body:
        raise RuntimeError(f"the actor root {paths[0]} has no UsdPhysics.RigidBodyAPI")
    attr = body.GetKinematicEnabledAttr()
    if not attr:
        attr = body.CreateKinematicEnabledAttr()
    return bool(attr.Set(bool(enabled)))


def zero_dynamic_root_velocity(actor, velocities) -> bool:
    """Clear residual motion only when USD confirms the body is dynamic.

    PhysX rejects velocity writes to kinematic bodies and emits native errors without
    raising Python exceptions. Unknown bodies are therefore left untouched.
    """
    if root_is_kinematic(actor) is not False:
        return False
    actor.write_root_velocity_to_sim(velocities)
    return True


def reset_scene_to_default_without_kinematic_velocity_writes(env, env_ids) -> None:
    """IsaacLab reset event that leaves fixed receptacles free of invalid velocity writes.

    IsaacLab's stock ``reset_scene_to_default`` writes root velocities to every rigid
    object, including the kinematic hollow trays used as Demo 11 destinations. PhysX
    rejects that write. Keep its pose reset and articulation/deformable behavior, while
    writing rigid velocities only when USD confirms the root is dynamic.
    """
    for rigid_object in env.scene.rigid_objects.values():
        default_state = rigid_object.data.default_root_state[env_ids].clone()
        default_state[:, 0:3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim(default_state[:, :7], env_ids=env_ids)
        if root_is_kinematic(rigid_object) is False:
            rigid_object.write_root_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)

    for articulation in env.scene.articulations.values():
        default_state = articulation.data.default_root_state[env_ids].clone()
        default_state[:, 0:3] += env.scene.env_origins[env_ids]
        articulation.write_root_pose_to_sim(default_state[:, :7], env_ids=env_ids)
        articulation.write_root_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)
        default_pos = articulation.data.default_joint_pos[env_ids].clone()
        default_vel = articulation.data.default_joint_vel[env_ids].clone()
        articulation.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

    for deformable in env.scene.deformable_objects.values():
        state = deformable.data.default_nodal_state_w[env_ids].clone()
        deformable.write_nodal_state_to_sim(state, env_ids=env_ids)


def place_items(env, items: list[scene_spec.Item]) -> None:
    """Write every object's pose and let it settle, the way demo 05 places the cube."""
    import torch

    device = env.device
    zeros = torch.zeros(1, 6, device=device)
    arm_hold = env.scene["robot"].data.joint_pos.clone()
    for item in items:
        obj = env.scene[item.prim]
        position = torch.tensor([[item.x, item.y, item.z]], device=device)
        half = math.radians(item.yaw_deg) / 2
        rotation = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], device=device)
        obj.write_root_pose_to_sim(torch.cat([position, rotation], dim=-1))
        zero_dynamic_root_velocity(obj, zeros)
    _rest(env, items, held_joint_pos=arm_hold)
    # Zero once more: the first contact leaves a residual roll that a level table will not
    # stop on its own. See SETTLE_STILL_M.
    for item in items:
        zero_dynamic_root_velocity(env.scene[item.prim], zeros)
    _rest(env, items, held_joint_pos=arm_hold)


def _camera_frame_is_ready(rgb: np.ndarray) -> bool:
    """Return false for the uniform black/grey buffers RTX emits while a camera warms up.

    Every supported Demo 11 environment includes a textured tabletop or other visible scene
    geometry, so a useful capture has measurable luminance variation. RTX can briefly return a
    softly shaded near-blank target (5th-95th percentile range 4-8) after the uniform warm-up
    buffer; the 12-level floor rejects those frames while ignoring a small number of outlier
    pixels. A real synchronized side-camera table frame in the seed-14000 regression had a range
    above 70.
    """
    if rgb.ndim != 3 or rgb.shape[-1] < 3 or rgb.size == 0:
        return False
    pixels = np.asarray(rgb[..., :3], dtype=np.float32)
    if not np.isfinite(pixels).all():
        return False
    luminance = (0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1]
                 + 0.0722 * pixels[..., 2])
    low, high = np.percentile(luminance, (5.0, 95.0))
    return bool(high - low >= CAMERA_FRAME_MIN_LUMA_RANGE)


def _render_camera_rgb(env, sensor, *, advance_physics_on_recovery: bool = False) -> tuple[np.ndarray, int]:
    """Render until the camera returns scene content, or fail explicitly after a short bound."""
    rgb = None
    for attempt in range(1, CAMERA_CAPTURE_MAX_ATTEMPTS + 1):
        env.sim.render()
        sensor.update(env.physics_dt, force_recompute=True)
        rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if _camera_frame_is_ready(rgb):
            return rgb, attempt
    # RTX render products can remain blank for render-only frames immediately after
    # scene population. Advance a few physics frames to synchronize Fabric and sensors.
    for recovery in range(1, CAMERA_CAPTURE_RECOVERY_STEPS + 1):
        if advance_physics_on_recovery:
            env.sim.step(render=True)
        else:
            env.sim.render()
        sensor.update(env.physics_dt, force_recompute=True)
        rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if _camera_frame_is_ready(rgb):
            return rgb, CAMERA_CAPTURE_MAX_ATTEMPTS + recovery
    # Replicator can leave an initialized render product attached but returning an all-zero
    # buffer. Resetting the Isaac Lab sensor clears its frame counter and reapplies its camera
    # pose; then give the render pipeline a longer synchronized warm-up before declaring the
    # camera unavailable. This is deliberately a camera reset, not a simulator reset, so the
    # active scene and robot state are preserved.
    reset = getattr(sensor, "reset", None)
    if callable(reset):
        reset()
    for recovery in range(1, CAMERA_CAPTURE_REINITIALIZE_STEPS + 1):
        if advance_physics_on_recovery:
            env.sim.step(render=True)
        else:
            env.sim.render()
        sensor.update(env.physics_dt, force_recompute=True)
        rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if _camera_frame_is_ready(rgb):
            return rgb, (CAMERA_CAPTURE_MAX_ATTEMPTS + CAMERA_CAPTURE_RECOVERY_STEPS
                         + recovery)
    low, high = np.percentile(
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2], (5.0, 95.0))
    raise RuntimeError(
        f"La c?mara sigui? vac?a tras {CAMERA_CAPTURE_MAX_ATTEMPTS + CAMERA_CAPTURE_RECOVERY_STEPS + CAMERA_CAPTURE_REINITIALIZE_STEPS} intentos "
        f"renders (rango de luminancia {high - low:.2f}); no se interpreta como mesa vacía.")


def capture_rgb(env, name: str) -> tuple[np.ndarray, CameraModel]:
    """One rendered frame as BGR plus its camera, and deliberately no depth.

    ``capture`` reads both buffers, and the depth half is a per-frame cost the dashboard's live
    picture does not want: a GPU-to-CPU copy of a 640x480 buffer and then ``astype("float64")``,
    which turns 307200 pixels into 2.4 MB of doubles for numbers the caller is going to
    downscale to 384 px wide and JPEG. ``capture``'s docstring explains the render/read ordering
    that has to be kept; this is that same sequence with the depth branch taken out.

    It exists for ``cam_pump``, which runs *inside* the executor's motion loops - that is, between
    two physics steps - at up to ``--cam-fps`` times a second. Anything it does not need is a
    stall the arm pays for to put a thumbnail on a screen. Perception still gets the depth it
    actually measures with, because it calls ``capture``.
    """
    sensor = env.scene[name]
    rgb, _attempts = _render_camera_rgb(env, sensor)
    bgr = rgb[..., ::-1].copy()  # the annotator gives RGB, OpenCV wants BGR
    height, width = sensor.data.image_shape
    camera = CameraModel(
        pos_w=sensor.data.pos_w[0].detach().cpu().numpy().tolist(),
        quat_wxyz=sensor.data.quat_w_world[0].detach().cpu().numpy().tolist(),
        matrix=sensor.data.intrinsic_matrices[0].detach().cpu().numpy().tolist(),
        width=width,
        height=height,
    )
    return bgr, camera


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
    rgb, attempts = _render_camera_rgb(env, sensor, advance_physics_on_recovery=True)
    if attempts > 1:
        print(f"  cámara lista tras {attempts} renders; se descartaron {attempts - 1} frames vacíos",
              flush=True)
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
