"""How far does the target move while the arm is reaching for it? Needs Isaac.

The demo grasps balls that are no longer where the plan said they were, and reports success.
The suspected mechanism is the kinematic fallback: ``Executor.grab`` writes the object's pose
from the gripper frame every step, so an object that is 20 mm away is picked up out of empty
air and carried, and nothing in the report says the jaws never touched it.

That only matters if the target really does move during the approach, which is what this
measures. For each object it plans a real pick from a real camera frame, drives ``above`` and
``down``, and compares where the object actually is with where the plan aimed:

* after ``above`` - drift caused by the transit across the table
* after ``down``  - drift caused by the descent onto it

The second number is the one that matters, because it is how far the jaws close from the
object they were aimed at.

    python tools/probe_approach_drift.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import dynamics  # noqa: E402
import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import perceive  # noqa: E402
import planner  # noqa: E402
import scene_spec  # noqa: E402


def where(env, prim):
    return env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--down-steps", type=int, default=None,
                        help="override the step count for the descent, to test whether a "
                             "gentler approach leaves the ball alone")
    args = parser.parse_args()

    base = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", base)
    try:
        print("drift in mm between where the plan aimed and where the object actually was\n")
        print("| round | object | after above | after down | jaw radius | verdict |")
        print("|---|---|---|---|---|---|")

        after_above: list[float] = []
        after_down: list[float] = []
        kinds_seen: list[str] = []
        stale = 0
        total = 0

        for round_index in range(args.rounds):
            layout = dynamics.randomize(base, round_index)
            env.reset()
            isaac_io.aim_camera(env, "top", sensor)
            bot = executor_module.Executor(env)
            bot.park()
            isaac_io.place_items(env, layout)

            image, depth, camera = isaac_io.capture(env, sensor)
            seen = perceive.perceive(image, source=f"probe{round_index}", camera=camera,
                                     frame=isaac_io.robot_frame(env),
                                     plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                     looking_for=colors.colors(colors.DEFAULT_DETECT))
            premise = {}
            claimed: set[str] = set()
            for thing in sorted(seen.objects, key=lambda t: t.id):
                best, best_gap = None, 0.025
                for item in layout:
                    if item.container or item.prim in claimed:
                        continue
                    gap = math.dist(thing.pos_m[:2], (item.x, item.y))
                    if gap < best_gap:
                        best, best_gap = item, gap
                if best is not None:
                    claimed.add(best.prim)
                    premise[thing.id] = best.prim

            for thing in seen.objects:
                prim = premise.get(thing.id)
                if prim is None:
                    continue
                plan = planner.plan_command(
                    {"intent": "pick_and_place",
                     "items": [{"kind": thing.kind, "name": None, "color": thing.color,
                                "quantity": 1, "position": None, "place": None}],
                     "place": {"kind": "box", "color": thing.color}, "sort_by": "none"},
                    seen)
                if not plan.moves or not plan.picks:
                    continue

                aimed = np.asarray(next(s.pose for s in plan.steps if s.action == "down"))
                start = where(env, prim)

                above_step = next(s for s in plan.steps if s.action == "above")
                bot.do("above", np.asarray(above_step.pose), executor_module.OPEN)
                drift_above = float(np.linalg.norm(where(env, prim)[:2] - start[:2])) * 1000

                down_step = next(s for s in plan.steps if s.action == "down")
                bot.do("down", np.asarray(down_step.pose), executor_module.OPEN,
                       steps=args.down_steps)
                now = where(env, prim)
                drift_down = float(np.linalg.norm(now[:2] - aimed[:2])) * 1000

                item = next(i for i in layout if i.prim == prim)
                jaw = item.size[0] * 500  # half the object, in mm: the jaw's reach
                total += 1
                after_above.append(drift_above)
                after_down.append(drift_down)
                kinds_seen.append(prim)
                missed = drift_down > jaw
                stale += int(missed)
                print(f"| {round_index} | {prim} | {drift_above:.1f} | {drift_down:.1f} "
                      f"| {jaw:.0f} | {'JAWS MISS' if missed else 'within reach'} |")

        print("")
        print(f"after above: median {np.median(after_above):.1f} mm, "
              f"worst {max(after_above):.1f} mm")
        print(f"after down : median {np.median(after_down):.1f} mm, "
              f"worst {max(after_down):.1f} mm")
        balls = [d for (p, d) in zip(kinds_seen, after_down) if p.startswith("ball")]
        if balls:
            print(f"balls only : median {np.median(balls):.1f} mm, worst {max(balls):.1f} mm "
                  f"({len(balls)} approach(es))")
        print(f"the jaws closed more than half an object away on {stale}/{total} approaches")
        print("")
        print("RESULT: SUCCESS" if stale == 0 else
              f"RESULT: FAILED, {stale}/{total} approaches aimed at stale positions")
        return 0 if stale == 0 else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())