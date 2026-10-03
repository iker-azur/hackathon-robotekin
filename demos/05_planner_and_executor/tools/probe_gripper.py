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
SETTLE = 90


def finger_tips(robot):
    """The two opposing fingers.

    The articulation has exactly one body called ``jaw`` and one called ``gripper``, and no
    pair of bodies with matching names - so the two fingers are those two, and the aperture is
    the distance between them. There is nothing else to measure against.
    """
    return [n for n in ("gripper", "jaw") if n in robot.body_names]


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

        print("\n| angle rad | tip gap mm | fits 3 cm cube | fits 4 cm ball |")
        print("|---|---|---|---|")
        for angle in ANGLES:
            term._close_command[:] = float(angle)
            for _ in range(SETTLE):
                env.step(torch_action(robot))
            left = robot.data.body_pos_w[0, robot.find_bodies(tips[0])[0][0]].detach().cpu().numpy()
            right = robot.data.body_pos_w[0, robot.find_bodies(tips[1])[0][0]].detach().cpu().numpy()
            gap = float(math.dist(left[:3], right[:3])) * 1000
            print(f"| {angle:.2f} | {gap:.1f} | {'yes' if gap > 30 else 'no'} "
                  f"| {'yes' if gap > 40 else 'no'} |")

        print("")
        print("The gap is between the two finger bodies' origins, not their tips, so it is an")
        print("upper bound on the aperture a ball must fit through. Treat it as indicative.")
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