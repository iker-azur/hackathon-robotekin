"""Does a parked arm disturb the scene, and does it matter? Needs Isaac.

check_executor.py found the red ball 49 mm from where the scene declares it, and printing the
simulator's own positions next to perception's showed perception was right to 9 mm: the ball
had physically rolled. Cubes had barely moved (2.5 mm); spheres had drifted up to 112 mm.

This measures the two facts that decide the demo's order of operations:

1. does parking move the balls (yes - the forearm sweeps the near half of the table);
2. does ``place_items`` put them back (yes, and exactly, once it settles until still).

Together they mean a round must park FIRST and populate SECOND, and capture after that.

    python tools/probe_ball_roll.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import scene_spec  # noqa: E402

MOVED = [i for i in scene_spec.DEFAULT_SCENE if not i.container and i.kind == "ball"]


def report(env, label):
    parts = []
    for item in MOVED:
        true = env.scene[item.prim].data.root_pos_w[0].detach().cpu().numpy()
        drift = ((true[0] - item.x) ** 2 + (true[1] - item.y) ** 2) ** 0.5
        parts.append(f"{item.prim} at ({true[0]:+.4f}, {true[1]:+.4f}) drift {drift * 1000:5.1f} mm")
    print(f"  {label:28s} " + " | ".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    items = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", items)
    try:
        print(f"declared: " + " | ".join(f"{i.prim} ({i.x:+.4f}, {i.y:+.4f}, z={i.z:+.4f})" for i in MOVED))
        print(f"SETTLE_STILL_M={isaac_io.SETTLE_STILL_M}, SETTLE_MAX_STEPS={isaac_io.SETTLE_MAX_STEPS}, "
              f"PARK_STEPS={executor_module.PARK_STEPS}\n")

        # Round 1: park, then populate. This is the order demo 10 must use.
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        bot = executor_module.Executor(env)
        bot.park()
        report(env, "round 1, after park()")

        isaac_io.place_items(env, items)
        report(env, "round 1, after place_items()")
        bot.settle(60)
        report(env, "round 1, after settle(60)")

        # Round 2: the arm is wherever the last round left it, over the table. Park from there
        # and populate again. This is the transition the demo has to survive every round.
        bot.park()
        report(env, "round 2, after park()")

        isaac_io.place_items(env, items)
        report(env, "round 2, after place_items()")
        bot.settle(60)
        report(env, "round 2, after settle(60)")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())