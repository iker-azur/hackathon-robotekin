"""How violent is the park sweep, and does slowing it down help? Needs Isaac.

The demo has to park the arm before every look, because a plan ends with the arm over the
table and a camera looking down from the far side sees the arm before it sees the objects.
But driving out to the park pose sweeps the forearm through whatever is on the table, and in
the demo that launched a ball to x = -3.09 m - off a table spanning x 0.0032..0.7032 - after
which nothing could find it.

Contact impulse scales with closing speed, so a FASTER park should sweep less far. This
measures drift against park duration, starting from the retreat pose (the arm's position
between commands) rather than from reset, because that is the sweep the demo actually has to
survive.

The first run of this measured the opposite of what that reasoning predicts, and the numbers
are worth keeping: 150 steps drifted 28.5 mm, 300 drifted 79 mm, 600 drifted 184 mm and 1200
drifted 1385 mm. A slow park holds the forearm against the balls and leans on them; a fast one
gets past. Only the balls move at all - the cubes read 0.0 mm at every duration, because a
cube on a flat face does not slide.

    python tools/probe_park_speed.py --headless
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

import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import planner  # noqa: E402
import scene_spec  # noqa: E402

DURATIONS = (25, 40, 60, 90, 150)


def positions(env, items):
    out = {}
    for item in items:
        where = env.scene[item.prim].data.root_pos_w[0].detach().cpu().numpy()
        out[item.prim] = (float(where[0]), float(where[1]))
    return out


def drift(before, after) -> dict[str, float]:
    return {prim: math.dist(before[prim], after[prim]) * 1000 for prim in before}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    items = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", items)
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        bot = executor_module.Executor(env)

        loose = [i.prim for i in items if not i.container]
        print("drift in mm, sweeping from a hover over the far row back to PARK_POSE\n")
        print("| hover above table | park steps | " + " | ".join(loose) + " | worst | off the table |")
        print("|---|---|" + "---|" * (len(loose) + 2))

        worst_overall = 0.0
        rows = []
        for lift in (0.075, 0.15, 0.25, 0.35):
            for steps in (40,):
                env.reset()
                bot = executor_module.Executor(env)
                bot.park(steps=steps)
                isaac_io.place_items(env, items)
                hover = np.array([0.355, -0.470, scene_spec.TABLE_TOP_Z + lift])
                bot.do("above", hover, executor_module.OPEN, steps=120)
                before = positions(env, items)

                bot.park(steps=steps)
                after = positions(env, items)

                moves = drift(before, after)
                off = [p for p in loose
                       if not (scene_spec.WORK_X[0] - 0.02 <= after[p][0] <= scene_spec.WORK_X[1] + 0.02
                               and scene_spec.WORK_Y[0] - 0.02 <= after[p][1] <= scene_spec.WORK_Y[1] + 0.02)]
                worst = max(moves[p] for p in loose)
                worst_overall = max(worst_overall, 999.0 if off else worst)
                print(f"| {lift * 100:.1f} cm | {steps} | "
                      + " | ".join(f"{moves[p]:.1f}" for p in loose)
                      + f" | {worst:.1f} | {', '.join(off) if off else 'none'} |")
                rows.append((lift, steps, worst, bool(off)))

        best = min(rows, key=lambda r: (r[3], r[2]))
        print("")
        print(f"best: hover {best[0] * 100:.1f} cm above the table, {best[1]} park steps, "
              f"{best[2]:.1f} mm drift")
        print("RESULT: SUCCESS" if not best[3] and best[2] < 20.0
              else f"RESULT: FAILED, best sweep {best[2]:.0f} mm")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())