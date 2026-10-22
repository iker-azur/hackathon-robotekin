"""Pick up the red cube and move it 10 cm to the side. As simple as possible.

How it works:

* the arm is driven with LeIsaac's IK action (``so101_state_machine``): every
  step we send a target pose for the ``gripper`` body plus open/close;
* the cube is placed at a fixed, easy to reach spot, square to the robot
  (the task normally randomises it, sometimes out of comfortable reach);
* the motion is a list of waypoints relative to the cube, and we slide
  linearly from one to the next;
* once the gripper has closed, the cube is held at a fixed spot between the
  fingertips (its pose is written every step) until the gripper opens.
  Real friction grasping of the 3 cm cube was unreliable in this sim, so this
  keeps the demo about moving the cube, not about tuning contacts.

Run:

    python .\\demos\\01_simulation_basics\\05_cube_pick_and_place.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TASK = "LeIsaac-SO101-LiftCube-v0"
CUBE_START = (0.30, -0.42, 0.056)  # world metres, cube resting on the table
MOVE_CUBE_BY = (0.10, 0.0)  # metres in world x, y

# Where the 3 cm cube's centre sits in the gripper body frame when held,
# measured from the finger meshes: the fixed finger's inner face is at
# x = -0.8 cm, the fingers are centred on y = 0 and their tips reach
# z = -10.4 cm. So the cube rests against the fixed finger, with the
# fingertips 3 mm above its bottom face.
HOLD = (0.007, 0.0, -0.092)
ABOVE = 0.06  # extra height while travelling (higher and the wrist tilts)

# The IK action's gripper is binary: >= 0 opens to 1.0 rad, < 0 closes to
# JAW_CLOSED_RAD. LeIsaac's default close (0.4 rad) suits an 8 cm orange and
# leaves a 1 cm gap on the cube; fully shut (-0.17) makes the hooked tip ride
# over the cube's top edge. The jaw touches the 3 cm cube at ~0.17 rad, so
# 0.15 is a light squeeze.
OPEN, CLOSE = 1.0, -1.0
JAW_CLOSED_RAD = 0.15

# (name, x, y, z relative to the cube's start, gripper, steps)
DX, DY = MOVE_CUBE_BY
WAYPOINTS = [
    ("go above cube", 0.0, 0.0, ABOVE, OPEN, 150),
    ("go down", 0.0, 0.0, 0.0, OPEN, 100),
    ("close", 0.0, 0.0, 0.0, CLOSE, 50),
    ("lift", 0.0, 0.0, ABOVE, CLOSE, 80),
    ("carry", DX, DY, ABOVE, CLOSE, 120),
    ("put down", DX, DY, 0.0, CLOSE, 80),
    ("open", DX, DY, 0.0, OPEN, 40),
    ("go up", DX, DY, ABOVE, OPEN, 60),
]


def build_env():
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import leisaac.tasks  # noqa: F401  registers the task ids

    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    cfg.use_teleop_device("so101_state_machine")  # IK action + gravity off on the arm
    cfg.actions.gripper_action.close_command_expr = {"gripper": JAW_CLOSED_RAD}
    cfg.recorders = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None  # otherwise lifting the cube resets the episode
    return gym.make(TASK, cfg=cfg).unwrapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without a window")
    args = parser.parse_args()

    from env_loader import prepare_runtime

    prepare_runtime(headless=args.headless)

    import torch
    from isaaclab.utils.math import (
        combine_frame_transforms,
        quat_apply,
        quat_inv,
    )

    env = build_env()
    try:
        env.reset()
        robot = env.scene["robot"]
        cube = env.scene["cube"]
        gripper_id = robot.find_bodies("gripper")[0][0]
        dev = env.device

        base_pos = robot.data.root_pos_w
        base_quat = robot.data.root_quat_w

        # Put the cube on its spot, square to the world, and let it settle.
        upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        cube_start = torch.tensor([CUBE_START], device=dev)
        cube.write_root_pose_to_sim(torch.cat([cube_start, upright], dim=-1))
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        for _ in range(20):
            env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        grip_quat = quat_inv(base_quat)  # gripper square to the world, in the base frame
        hold = torch.tensor([HOLD], device=dev)
        grasp = cube_start - hold  # gripper body position at the grasp

        holding = False
        current = robot.data.body_pos_w[:, gripper_id].clone()

        with torch.inference_mode():
            for name, x, y, z, grip, steps in WAYPOINTS:
                print(f"  {name}", flush=True)
                goal = grasp + torch.tensor([[x, y, z]], device=dev)
                start = current
                for i in range(1, steps + 1):
                    current = start + (goal - start) * min(1.0, i / (0.8 * steps))
                    target = quat_apply(quat_inv(base_quat), current - base_pos)
                    action = torch.cat(
                        [target, grip_quat, torch.tensor([[grip]], device=dev)], dim=-1
                    )
                    env.step(action)

                    holding = grip == CLOSE and (holding or i == steps)
                    if holding:
                        pos, quat = combine_frame_transforms(
                            robot.data.body_pos_w[:, gripper_id],
                            robot.data.body_quat_w[:, gripper_id],
                            hold,
                            upright,
                        )
                        cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
                        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

            for _ in range(30):  # let the cube settle after release
                env.step(action)

        moved = (cube.data.root_pos_w - cube_start)[0]
        error = float((moved[:2] - torch.tensor(MOVE_CUBE_BY, device=dev)).norm())
        ok = error < 0.02
        print("")
        print(f"cube moved by ({float(moved[0]) * 100:+.1f}, {float(moved[1]) * 100:+.1f}) cm, "
              f"wanted ({DX * 100:+.1f}, {DY * 100:+.1f}) cm")
        print(f"RESULT: {'SUCCESS' if ok else 'FAILED'}")
        return 0 if ok else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
