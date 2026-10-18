"""Build the SO-101 simulation environments used by the imitation-learning demos.

The demos all drive the same task, so the task id, the two control spaces, and
the geometry of the pick live in this one module and nowhere else.

Two control spaces, both from LeIsaac
------------------------------------

``so101_state_machine``
    LeIsaac's configuration for scripted control. ``arm_action`` is an inverse
    kinematics term (7 numbers: target pose of the ``gripper`` body) and
    ``gripper_action`` is a binary term (1 number), for an 8-dimensional action.
    Gravity is switched off on the arm. This is the space the *teacher* uses,
    and it is the same configuration ``demos/01_simulation_basics/
    05_cube_pick_and_place.py`` already runs on this machine.

joint follower
    The same environment with the action replaced by direct joint position
    targets (6 numbers, one per SO-101 joint). This is the space of the SO-101
    leader arm, the space of every LeRobot SO-101 dataset, and the space the
    *policy* executes in.

Keeping the two spaces separate is what lets the teacher be a script that knows
where the cube is, and the policy to be a network that has to work it out from
the camera.

Action and state units
---------------------

Both spaces are exported in the units LeIsaac itself exports to LeRobot:
``leisaac.utils.robot_utils.convert_leisaac_action_to_lerobot``, i.e. degrees
rescaled from the USD joint range onto the motor range of a real SO-101. That
keeps the produced dataset interchangeable with LeIsaac's own recordings and
with the SO-101 datasets already published on the Hugging Face Hub.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import degrees as to_degrees, radians
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_simulation_basics"))

from env_loader import prepare_runtime  # noqa: E402  (defined in demo 01)

TASK = "LeIsaac-SO101-LiftCube-v0"

# Joint order of the SO-101. Asserted against the articulation's own list at
# import time of the demos, because the dataset's state and action columns are
# named after this order and a silent mismatch would ruin training.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_NAMES = JOINT_NAMES[:5]

# Jaw opening of the gripper once "closed": the jaw touches the 3 cm cube at
# about 0.17 rad, so 0.15 rad is a light squeeze (demo 05, measured on the
# finger meshes).
JAW_CLOSED_RAD = 0.15
OPEN, CLOSE = 1.0, -1.0

# Where the 3 cm cube's centre sits in the gripper body frame when held, measured
# from the finger meshes by demo 05: the fixed finger's inner face is 0.8 cm back
# and the fingertips are 3 mm above the cube's bottom face.
HOLD = (0.007, 0.0, -0.092)

# Extra height while travelling (demo 05: higher and the wrist tilts).
ABOVE = 0.06

# Where the cube may be placed, in metres. Chosen from the arm's reach: the robot
# base sits at (0.35, -0.64) and demo 04 measured ``MAX_REACH_M = 0.33``, and the
# drop spot is another ``DROP_OFFSET_XY[0]`` further along x. The worst corner is
# (0.38, -0.365) dropping at (0.48, -0.365), which is 0.304 m from the base - an
# 8 % margin on the measured reach.
#
# This box was 7 x 6 cm and only ever sampled at two y values, which made the 25
# "varied" demonstrations 25 points on a line rather than a region. It is wider
# now so the demonstrations actually cover a surface; see ``sample_cube_spot``.
CUBE_BOX_X = (0.29, 0.38)
CUBE_BOX_Y = (-0.455, -0.365)

# How far the cube may be turned about its own axis, in radians. +-30 degrees is
# the same span that does not fail on this hardware: at +-40 to +-50 degrees the
# fixed finger of the jaw catches a corner of the cube and shoves it instead of
# lifting it, producing a failed demonstration out of a perfectly good plan.
CUBE_MAX_YAW = 0.5236

# The cube rests on the table top. Measured by demo 05 from the shipped USD, and
# the cube's own radius is 15 mm.
CUBE_REST_Z = 0.056

# The task's randomiser is frozen to zero. The demos place the cube themselves
# inside CUBE_BOX_X/Y, because the shipped task spreads it over +-75 mm - wide
# enough that the cube sometimes starts where the SO-101 cannot reach, and one
# unreachable start is enough to waste a demonstration. Demo 05 makes the same
# choice for the same reason; the difference here is that the cube is sampled
# over a small region instead of being fixed in one spot, so the demonstrations
# still vary.


def _halton(index: int, base: int) -> float:
    """The ``index``-th point of the Halton sequence in ``base``, in [0, 1)."""
    result, fraction = 0.0, 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def sample_cube_spot(episode: int, total: int) -> tuple[float, float]:
    """Where to put the cube for demonstration ``episode`` out of ``total``.

    A Halton low-discrepancy sweep, so consecutive demonstrations are far apart
    and the whole set covers a *surface* rather than a line. That is the fix for
    the workspace this used to sample: x swept linearly from one end to the other
    and y alternated between exactly two values, so 25 episodes gave 25 points on
    a one-dimensional line with the episode index, x and the cube's yaw all
    correlated 1:1. A plane-to-image fit could not even be solved on it.

    Still deterministic on purpose: the same dataset name and episode count always
    produce the same cube positions, so a re-run reproduces the earlier one and a
    colleague can check a claim about a specific episode.
    """
    del total  # the sequence is driven by the index alone
    x = CUBE_BOX_X[0] + _halton(episode + 1, 2) * (CUBE_BOX_X[1] - CUBE_BOX_X[0])
    y = CUBE_BOX_Y[0] + _halton(episode + 1, 3) * (CUBE_BOX_Y[1] - CUBE_BOX_Y[0])
    return x, y


def sample_cube_yaw(episode: int, total: int) -> float:
    """Yaw for demonstration ``episode``, in radians, over +-30 degrees.

    The cube is the only thing in the scene that moves between episodes, so it is
    the only source of visual variety. Without a turn the front camera sees
    nearly the same frame every time, which leaves the policy nothing in the image
    that tells it where the cube is.

    A third Halton dimension, so the turn is not tied to the position sweep the
    way it was when both were linear in the episode index.
    """
    del total
    return -CUBE_MAX_YAW + 2.0 * _halton(episode + 1, 5) * CUBE_MAX_YAW

ACTION_DIM = 8
STATE_DIM = 6

# The 8 numbers the IK action space carries, named the way LeIsaac names them
# when the action does not match the robot's joint order. The first three are the
# target position of the ``gripper`` body in the robot base frame, the next four
# its orientation, and the last one opens or closes the jaw.
ACTION_NAMES = ["dim_0", "dim_1", "dim_2", "dim_3", "dim_4", "dim_5", "dim_6", "dim_7"]

# The SO-101 joint range in the USD, in radians. Actions are clipped to this
# before being applied, because a policy trained on a handful of demonstrations
# can predict a value slightly outside the arm's range, and the simulator then
# quietly fights it. Mirrors ``SO101_FOLLOWER_USD_JOINT_LIMITS`` from
# ``leisaac.assets.robots.lerobot``, minus the gripper, whose two action terms
# each handle their own limits.
JOINT_LOWER_LIMITS = tuple(
    radians(limit) for limit in (-110.0, -100.0, -100.0, -95.0, -160.0, -10.0)
)
JOINT_UPPER_LIMITS = tuple(
    radians(limit) for limit in (110.0, 100.0, 90.0, 95.0, 160.0, 100.0)
)

# Pose every episode starts from, in the dataset's own unit (motor degrees).
#
# Why this is needed: ``env.reset()`` does not restore the robot's joints. Left
# alone, episode N therefore begins exactly where episode N-1 finished - measured
# on the 25-episode dataset, the elbow drifts monotonically from -39.4 to -64.5
# motor degrees across the run. That made the dataset's starting-state
# distribution drift with the episode index, and it made a re-run of demo 13
# produce a different dataset from the same seed.
#
# Zeros for the arm and fully open for the jaw: this is the pose the very first
# episode of the original dataset started from (measured ``[0, 0, 5.3, 0, 0,
# 9.1]`` - the arm at rest, the jaw not yet commanded), and that episode
# succeeded, so it is known to be a pose the plan can be executed from.
REST_MOTOR_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 61.0)


def reset_arm(env, settle_steps: int = 20) -> None:
    """Put the arm back on :data:`REST_MOTOR_DEG` and let it settle.

    Written as a joint state rather than as an action, because the action space
    is inverse kinematics in both environments: there is no joint-space command to
    send home with. Gravity is off on the arm, so once the positions are written
    it stays where it is put.

    Only the six SO-101 joints named in :data:`JOINT_NAMES` are touched. The USD
    carries extra joints for the gripper's fingers, and overwriting them with
    zeros would fold the jaw shut; they are left at whatever the simulation has.

    Call this right after ``env.reset()`` and before anything reads the arm's
    pose, or the plan is computed from wherever the previous episode stopped.
    """
    import torch

    from il_dataset import convert_to_sim_radians

    robot = env.scene["robot"]
    ids, _ = robot.find_joints(JOINT_NAMES)

    rest_motor = torch.tensor(REST_MOTOR_DEG, dtype=torch.float32, device=env.device)
    # ``convert_to_sim_radians`` hands back NumPy, so it goes through
    # ``as_tensor`` rather than straight into the joint write.
    rest_sim = torch.as_tensor(
        convert_to_sim_radians(rest_motor), dtype=torch.float32, device=env.device
    )

    joint_pos = robot.data.joint_pos.clone()
    joint_pos[:, ids] = rest_sim.unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(robot.data.joint_vel))

    # PhysX goes on driving towards whatever joint targets the last ``env.step``
    # left behind, so writing the position is not enough on its own - during the
    # settling steps below the arm walks straight back to the previous episode's
    # pose. That was measured: after a first attempt at this function, episode 0
    # started at the rest pose and episodes 1-39 still started all over the
    # workspace, with the wrist spread 100 motor degrees. Commanding the same
    # pose fixes it, and costs one line.
    robot.set_joint_position_target(rest_sim.unsqueeze(0), joint_ids=ids)
    # ``set_joint_position_target`` only fills an internal buffer. PhysX does not
    # see it until something flushes that buffer, and the usual flusher is
    # ``env.step()`` - which the settling loop below deliberately does not call.
    # Without this line the command never reaches the solver and the arm walks
    # back to the previous episode's pose; measured, see ``results/diagnosis.md``.
    robot.write_data_to_sim()

    # Settle with the action manager out of the way, so nothing recomputes the
    # targets from an inverse-kinematics command while the arm finds its pose.
    for _ in range(settle_steps):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)



def set_cube_randomization(cfg, cube_pose_range: dict | None = None) -> None:
    """Freeze or narrow the task's domain-randomisation events.

    LeIsaac adds its randomisation as dynamically named attributes
    (``domain_randomize_0``, ``domain_randomize_1``, ...) on the event manager
    config. They are edited where they sit, rather than removed and re-added:
    ``EventTermCfg`` objects are mutable and the manager builds from them when it
    is constructed, so a range written here is the range that is used.

    The term that carries the cube is told by its own parameters - a ``yaw`` key -
    so the loop does not depend on which index LeIsaac happened to assign.
    """
    import dataclasses

    if cube_pose_range is None:
        # Nothing randomised: the demos place the cube themselves.
        cube_pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    for field in dataclasses.fields(cfg.events):
        if not field.name.startswith("domain_randomize_"):
            continue
        pose_range = getattr(cfg.events, field.name).params.get("pose_range")
        if pose_range is None:
            continue

        if "yaw" in pose_range:
            pose_range.update(cube_pose_range)
        else:
            # The camera. Frozen instead of tuned: it changes the image the policy
            # must read, not the physics of the pick.
            for axis in pose_range:
                pose_range[axis] = (0.0, 0.0)


def _prepare_cfg(record: bool, device: str, num_envs: int = 1):
    """Build the LiftCube task config with recorders and randomisation handled."""
    import isaaclab_tasks.utils as isaaclab_utils
    import leisaac.tasks  # noqa: F401  registers the task ids

    cfg = isaaclab_utils.parse_env_cfg(TASK, device=device, num_envs=num_envs)
    # ``so101_state_machine``: IK action and gravity off on the arm. Both action
    # spaces below start from this one and only replace the action terms.
    cfg.use_teleop_device("so101_state_machine")
    cfg.actions.gripper_action.close_command_expr = {"gripper": JAW_CLOSED_RAD}
    cfg.recorders = None  # the demos write their own dataset

    if record:
        # The scripted teacher is a fixed-length plan, not an episode that ends
        # when the cube rises. Leaving the task's terms on would cut it short.
        cfg.terminations.time_out = None
        cfg.terminations.success = None

    set_cube_randomization(cfg)
    return cfg


def use_joint_follower(cfg) -> None:
    """Replace the IK action with direct joint position targets, 6 numbers.

    This is the control space of the SO-101 leader arm and of every LeRobot
    SO-101 dataset, so a policy trained on this dataset can be replayed on the
    real arm without a unit change. Gravity stays off, as the teacher env keeps
    it, so the two environments differ only in who computes the joints.

    Must be called on the config *before* the env is created.
    """
    import isaaclab.envs.mdp as mdp

    cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=1.0
    )
    cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["gripper"], scale=1.0
    )


def build_env(
    kind: str = "joint_follower",
    record: bool = False,
    device: str = "cuda:0",
    viewer: dict | None = None,
):
    """Create the LiftCube environment.

    Args:
        kind: ``"joint_follower"`` for a 6-dim joint target action (the policy's
            space, and the space the dataset uses), or ``"ik_teacher"`` for the
            8-dim inverse-kinematics action the scripted teacher drives.
        record: when True the task's own termination terms are disabled so a
            demonstration can run the full teacher plan.
        viewer: optional ``{"eye": (x, y, z), "lookat": (x, y, z)}`` for the
            window camera. Needed because the task's own viewer is aimed at
            ``(0.9, 0, -0.3)``, which is empty space in this scene.
    """
    import gymnasium as gym

    if kind not in ("joint_follower", "ik_teacher"):
        raise ValueError(f"kind must be 'joint_follower' or 'ik_teacher', got {kind!r}")

    cfg = _prepare_cfg(record=record, device=device)
    if kind == "joint_follower":
        # Has to happen before ``gym.make``: the action manager instantiates its
        # terms from the config it is constructed with, so editing the env's own
        # copy afterwards leaves the 8-dimensional IK term in place while the
        # evaluator feeds it 6 numbers.
        use_joint_follower(cfg)

    if viewer:
        if "eye" in viewer:
            cfg.viewer.eye = tuple(viewer["eye"])
        if "lookat" in viewer:
            cfg.viewer.lookat = tuple(viewer["lookat"])

    return gym.make(TASK, cfg=cfg).unwrapped


def default_viewer() -> dict:
    """A camera that actually frames this scene.

    The task's own viewer is ``eye=(-0.4, -0.6, 0.5)``, ``lookat=(0.9, 0, -0.3)``.
    Measured positions put the robot base at ``(0.35, -0.64, 0.01)``, the cube
    anywhere in ``CUBE_BOX_X/Y`` and the drop spot another 10 cm further along x,
    so the whole workspace is around ``x 0.2..0.5``, ``y -0.5..-0.3``,
    ``z 0..0.35``. The shipped look-at is well past it, and the window opens on
    empty floor.

    These two points were chosen to look at the workspace from in front of the
    robot, slightly above table height, which is the angle from which both the
    gripper and the cube are visible at once.
    """
    return {"eye": (0.05, -1.05, 0.55), "lookat": (0.38, -0.42, 0.12)}


def step_obs(env, action):
    """Step the environment and return only the observation.

    Isaac Lab's ``ManagerBasedEnv.step`` returns a 5-tuple
    (obs, reward, terminated, truncated, info), while a plain Gymnasium env
    returns a 4-tuple. The demos only ever want the observation, and the
    difference is not worth repeating in every script.
    """
    return env.step(action)[0]


def assert_joint_order(env) -> None:
    """Check that the articulation's joints are in the dataset's declared order.

    The state and action columns are named after ``JOINT_NAMES``. If the USD came
    back in a different order the dataset would be silently mislabelled, and the
    symptom would be a policy that moves the wrong joint. Fail here instead.
    """
    actual = list(env.scene["robot"].joint_names)[: len(JOINT_NAMES)]
    if actual != list(JOINT_NAMES):
        raise RuntimeError(f"the articulation's joints are {actual}, expected {list(JOINT_NAMES)}")
