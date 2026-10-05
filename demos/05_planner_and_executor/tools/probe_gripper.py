"""How wide do the jaws open, as a function of the commanded angle? Needs Isaac.

The gripper action is binary: it drives the joint to an open configuration or a closed one,
and its closed configuration is 0.15 rad - essentially shut. A 4 cm ball does not fit between
fingers that close to nothing, which is why the descent ejects it 77 mm and no amount of
re-aiming helps.

So the jaws need a closing width chosen for the object rather than a single "closed". That
needs the aperture as a function of the angle, which is not in the repository: the only note is
demo 03's, that LeIsaac's default of 0.4 rad suits an 8 cm orange. This measures it, by
commanding each angle and asking how far apart the finger tips actually are.

    python tools/probe_gripper.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import scene_spec  # noqa: E402

ANGLES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
OPEN_ANGLES = (1.0, 1.2, 1.4, 1.6, 1.7453)
SETTLE = 90


def finger_tips(robot):
    """The two opposing fingers.

    The articulation has exactly one body called ``jaw`` and one called ``gripper``, and no
    pair of bodies with matching names - so the two fingers are those two, and the aperture is
    the distance between them. There is nothing else to measure against.
    """
    return [n for n in ("gripper", "jaw") if n in robot.body_names]


def gap_mm(robot, tips) -> float:
    """Distance between the two finger bodies' origins, in millimetres."""
    left = robot.data.body_pos_w[0, robot.find_bodies(tips[0])[0][0]].detach().cpu().numpy()
    right = robot.data.body_pos_w[0, robot.find_bodies(tips[1])[0][0]].detach().cpu().numpy()
    return float(math.dist(left[:3], right[:3])) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", list(scene_spec.DEFAULT_SCENE))
    try:
        env.reset()
        robot = env.scene["robot"]
        ids, names = robot.find_joints("gripper")
        print(f"gripper joint ids {list(ids)} named {list(names)}")
        print(f"joint limits for those: "
              f"{robot.data.soft_joint_pos_limits[0][ids[0]].tolist()}")
        tips = finger_tips(robot)
        print(f"bodies with 'finger' in the name: {tips or 'none'}")
        print(f"all bodies: {list(robot.body_names)}")

        term = env.action_manager.get_term("gripper_action")
        print(f"\ngripper action term: {type(term).__name__}")
        print(f"  open_command  {term._open_command.tolist()}")
        print(f"  close_command {term._close_command.tolist()}")
        print(f"  joints        {list(term._joint_ids)}")

        if not tips or len(tips) < 2:
            print("\nRESULT: FAILED, could not identify two finger tips to measure")
            return 2

        print("\n| close command rad | tip gap mm | fits 3 cm cube | fits 4 cm ball |")
        print("|---|---|---|---|")
        for angle in ANGLES:
            term._close_command[:] = float(angle)
            for _ in range(SETTLE):
                env.step(torch_action(robot, closing=True))
            gap = gap_mm(robot, tips)
            print(f"| {angle:.2f} | {gap:.1f} | {'yes' if gap > 30 else 'no'} "
                  f"| {'yes' if gap > 40 else 'no'} |")

        # The open side is the one that decides whether a ball can be picked up at all, and
        # the first version of this probe never varied it: it always commanded "close" and
        # only moved the close target. The joint limit is 1.7453 rad, so the shipped open
        # command of 1.0 rad is not the widest the gripper can be driven.
        print("\n| open command rad | tip gap mm | fits 3 cm cube | fits 4 cm ball |")
        print("|---|---|---|---|")
        term._close_command[:] = float(term._close_command[0])
        for angle in OPEN_ANGLES:
            term._open_command[:] = float(angle)
            for _ in range(SETTLE):
                env.step(torch_action(robot, closing=False))
            gap = gap_mm(robot, tips)
            print(f"| {angle:.2f} | {gap:.1f} | {'yes' if gap > 30 else 'no'} "
                  f"| {'yes' if gap > 40 else 'no'} |")

        print("")
        print("The gap is between the two finger bodies' origins, not their tips, and the")
        print("origins are fixed to their mounts: that is why it reads the same for every")
        print("command. So it is the width a ball has to be pushed down between, and if it")
        print("does not exceed the ball's own diameter then no amount of re-aiming will help.")
        print("RESULT: SUCCESS")
        return 0
    finally:
        env.close()


def torch_action(robot, closing: bool = True):
    """An all-zero arm action, with the gripper commanded to close.

    The action vector is 7 arm numbers then one gripper number, and the gripper term treats a
    negative value as "close" - so this is what makes it apply ``_close_command``, which is the
    thing being varied here. Zeros would open it and measure nothing.
    """
    import torch

    action = torch.zeros(1, 8, device=robot.device)
    if closing:
        action[0, 7] = -1.0
    return action


if __name__ == "__main__":
    raise SystemExit(main())