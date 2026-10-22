"""Find a park pose the arm can be driven to AND drive away from. Needs Isaac.

The demo needs the arm out of the top camera's view while perception runs, and it needs the
arm drivable afterwards. Those two turned out to conflict:

* ``isaac_io.park_arm`` writes joint positions directly. The IK action manager then cannot
  drive the arm at all - the gripper sat 242 mm from every commanded pose and drifted.
* Driving to a guessed Cartesian "shoulder stand" instead did not help either: at
  ``(0.35, -0.74, 0.40)`` the tracking error was still 246 mm, and the arm hid one of the
  four objects from the camera.

Rather than guess again, this sweeps candidate poses and, for each one, measures the two
things that matter:

1. **visibility** - how many objects and containers perception still finds while parked;
2. **recoverability** - how far the gripper is from a commanded pose afterwards, which is
   demo 05's own grasp and is 0 mm when the arm has not been parked.

A pose is only a candidate if both are good.

    python tools/find_park_pose.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import perceive  # noqa: E402
import scene_spec  # noqa: E402

# What "drivable" means: demo 05's grasp pose, which tracks to 0 mm on an un-parked arm.
PROBE_POSE = (0.30, -0.42, scene_spec.TABLE_TOP_Z + 0.015 + executor_module.ABOVE)
TRACKING_OK_M = 0.02
VISIBLE_OBJECTS = 4
VISIBLE_CONTAINERS = 2

CANDIDATES = [
    ("over the base", 0.35, -0.66, 0.42),
    ("high over the base", 0.35, -0.66, 0.52),
    ("high and back", 0.35, -0.72, 0.50),
    ("upright, far back", 0.35, -0.70, 0.38),
    ("tucked at the base", 0.35, -0.62, 0.30),
    ("high over the front", 0.35, -0.58, 0.46),
    ("very high", 0.35, -0.64, 0.58),
    ("back and left", 0.28, -0.70, 0.44),
    ("back and right", 0.42, -0.70, 0.44),
]


def see(env, sensor, label):
    image, depth, camera = isaac_io.capture(env, sensor)
    scene = perceive.perceive(image, source=label, camera=camera,
                             frame=isaac_io.robot_frame(env),
                             plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                             looking_for=colors.colors(colors.DEFAULT_DETECT))
    return len(scene.objects), len(scene.containers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    items = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", items)
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        isaac_io.place_items(env, items)

        rows = []
        print(f"tracking tolerance {TRACKING_OK_M * 1000:.0f} mm, "
              f"want {VISIBLE_OBJECTS} objects and {VISIBLE_CONTAINERS} containers visible\n")
        print("| candidate | gripper | objects | containers | tracking error | verdict |")
        print("|---|---|---|---|---|---|")

        for label, x, y, z in CANDIDATES:
            env.reset()
            isaac_io.place_items(env, items)
            bot = executor_module.Executor(env)
            pose = np.array([x, y, z], dtype=float)
            bot.do("above", pose, executor_module.OPEN, steps=200)
            gripper = bot._gripper_pos()[0].detach().cpu().numpy()
            objects, containers = see(env, sensor, label)
            bot.do("above", np.asarray(PROBE_POSE), executor_module.OPEN, steps=200)
            after = bot._gripper_pos()[0].detach().cpu().numpy()
            error = float(np.linalg.norm(after - (np.asarray(PROBE_POSE)
                                                 - np.asarray(executor_module.HOLD))))
            visible = objects >= VISIBLE_OBJECTS and containers >= VISIBLE_CONTAINERS
            drivable = error < TRACKING_OK_M
            verdict = "USABLE" if (visible and drivable) else ("hides the table" if not visible else "cannot drive")
            rows.append((label, pose, visible and drivable, error))
            print(f"| {label} | ({gripper[0]:+.3f}, {gripper[1]:+.3f}, {gripper[2]:+.3f}) "
                  f"| {objects} | {containers} | {error * 1000:.0f} mm | {verdict} |")

        print("")
        good = [r for r in rows if r[2]]
        if good:
            best = min(good, key=lambda r: r[3])
            print(f"RESULT: SUCCESS, {len(good)} usable pose(s). Best: {best[0]} "
                  f"{tuple(round(float(v), 3) for v in best[1])}")
            print(f"Set PARK_POSE in executor.py to that.")
            return 0
        print("RESULT: FAILED, no candidate both clears the table and stays drivable.")
        print("Falling back to parking by joint angles and accepting that the arm must be")
        print("re-enabled after each perception pass.")
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())