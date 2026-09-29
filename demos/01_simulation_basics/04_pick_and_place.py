"""Pick and place with LeIsaac's own scripted state machine.

Earlier attempts at this demo hand-rolled an inverse kinematics loop and kept
failing for reasons that all trace back to not using the configuration LeIsaac
itself ships. ``leisaac.datagen.state_machine.PickOrangeStateMachine`` is a
working scripted pick and place, and the pieces it depends on are:

* ``env_cfg.use_teleop_device("so101_state_machine")``, which installs an action
  manager whose arm term is a ``DifferentialInverseKinematicsAction`` on the
  ``gripper`` body with ``command_type="pose"`` and ``ik_params={"lambda_val":
  0.04}``, and whose gripper term is a ``BinaryJointPositionAction`` with
  ``open_command_expr={"gripper": 1.0}`` and ``close_command_expr={"gripper":
  0.4}``. The action is therefore 8 numbers: target position, target orientation
  and one gripper command.
* The target pose is expressed **in the robot base frame**, not the world frame:
  ``quat_apply(quat_inv(base_quat), target_pos_w - base_pos)``.
* Gravity is switched off on the robot links. The actuators are position servos
  with stiffness 17.8, and with gravity on the arm sags a couple of centimetres
  below every commanded point, which is enough to miss a small object.
* The end effector is read as ``robot.data.body_pos_w[:, -1]``, the last body of
  the articulation, and the target sits ``0.10 m`` above the object because the
  fingers extend that far past the last body.

Rather than reimplement any of that, this demo runs the shipped state machine and
reports what it does. The task is ``LeIsaac-SO101-PickOrange-v0``, which picks
three oranges from a kitchen scene and places them on a plate, so it exercises
approach, grasp, lift, transport, lower, release and return home.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_TASK = "LeIsaac-SO101-PickOrange-v0"
STATE_MACHINE_DEVICE = "so101_state_machine"
ORANGES = 1

# Phase boundaries copied from PickOrangeStateMachine so the log can name the
# phase instead of just printing a step counter.
PHASES = (
    (120, "blend to hover"),
    (180, "move above orange"),
    (300, "hover above orange"),
    (360, "lower to orange"),
    (420, "grasp"),
    (500, "lift orange"),
    (550, "move above plate"),
    (600, "lower to plate"),
    (640, "release"),
    (680, "lift gripper"),
    (980, "return home"),
)


def phase_name(step: int, orange: int, total: int) -> str:
    for boundary, name in PHASES:
        if step < boundary:
            return f"orange {orange}/{total}: {name}"
    return f"orange {orange}/{total}: done"


def disable_robot_gravity(env) -> None:
    """Turn off gravity for the robot links.

    The arm is driven by position servos with stiffness 17.8. With gravity on, the
    commanded joint angles sag under the arm's own weight and the gripper settles
    a couple of centimetres below the target, which is enough to miss. LeIsaac's
    own data generation pipeline does the same thing.
    """
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    count = 0
    for prim in stage.Traverse():
        if "Robot" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
            count += 1
    print(f"  gravity disabled on {count} robot prims", flush=True)


def build_environment(task: str):
    """Create the task with the state machine action manager installed."""
    import gymnasium as gym
    import torch
    from isaaclab.managers import TerminationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import leisaac.tasks  # noqa: F401  registers the gym task ids

    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
    env_cfg.use_teleop_device(STATE_MACHINE_DEVICE)
    env_cfg.seed = 0
    env_cfg.recorders = None

    # Let the state machine decide when an episode ends.
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = TerminationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )

    return gym.make(task, cfg=env_cfg).unwrapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="LeIsaac gym task id")
    parser.add_argument("--oranges", type=int, default=ORANGES, help="Oranges to place")
    parser.add_argument("--headless", action="store_true", help="Run without a window")
    args = parser.parse_args()

    from env_loader import prepare_runtime

    try:
        prepare_runtime(headless=args.headless)
    except Exception as error:
        print(f"ERROR: could not start the simulator runtime: {error}")
        return 1

    import torch

    try:
        env = build_environment(args.task)
    except Exception:
        import traceback

        print("", flush=True)
        print("EXCEPTION while building the environment:", flush=True)
        traceback.print_exc()
        return 1

    try:
        from leisaac.datagen.state_machine import PickOrangeStateMachine
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim

        print(f"Action space: {env.action_space}", flush=True)

        disable_robot_gravity(env)
        if hasattr(env, "initialize"):
            env.initialize()

        machine = PickOrangeStateMachine(num_oranges=args.oranges)
        machine.setup(env)
        env.reset()
        machine.reset()

        robot = env.scene["robot"]
        ee_index = robot.num_bodies - 1
        print(f"EE body: {robot.body_names[ee_index]} (index {ee_index})", flush=True)

        last_phase = ""
        with torch.inference_mode():
            while not machine.is_episode_done:
                if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
                    dynamic_reset_gripper_effort_limit_sim(env, STATE_MACHINE_DEVICE)

                robot.write_joint_damping_to_sim(damping=10.0)
                machine.pre_step(env)
                action = machine.get_action(env)
                env.step(action)
                machine.advance()

                label = phase_name(machine.step_count, machine.orange_now, args.oranges)
                if label != last_phase:
                    last_phase = label
                    print(f"  step {machine.step_count:>4}  {label}", flush=True)

        success = machine.check_success(env)
        print("", flush=True)
        print("=" * 60)
        print(f"  task              : {args.task}")
        print(f"  oranges requested : {args.oranges}")
        print(f"  RESULT            : {'SUCCESS' if success else 'FAILED'}")
        print("=" * 60)
        return 0 if success else 2
    except Exception:
        import traceback

        print("", flush=True)
        print("EXCEPTION during the pick and place:", flush=True)
        traceback.print_exc()
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
