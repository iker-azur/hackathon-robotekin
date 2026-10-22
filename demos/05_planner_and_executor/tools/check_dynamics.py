"""Do randomised layouts still work? Needs Isaac.

dynamics.py claims a randomised layout is valid because ``conflicts`` is empty, and that a
shove actually moves something. Both are claims about the simulator, so this checks them over
many rounds rather than one:

* every randomised layout is free of overlaps and inside the reach envelope;
* perception still finds every object and container in it (the layout is only useful if the
  camera can read it);
* a mid-task shove moves something far enough to invalidate a planned pose.

Reports the worst case of each, not the mean: the demo has to survive the bad round.

    python tools/check_dynamics.py --headless
    python tools/check_dynamics.py --headless --rounds 20 --spread 0.03
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import dynamics  # noqa: E402
import isaac_io  # noqa: E402
import perceive  # noqa: E402
import scene_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--spread", type=float, default=dynamics.DEFAULT_SPREAD_M)
    parser.add_argument("--scene", default="top")
    args = parser.parse_args()

    import random

    base = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    # Built once with the default layout: prims are spawned from the item list, so the scene
    # cannot be rebuilt per round. Only positions change, which is what randomize() does.
    env, sensor = isaac_io.build_env(args.scene, base)

    problems: list[str] = []
    worst_detection = 0
    worst_missed = ""
    moved: list[float] = []
    total_moved = 0.0

    try:
        import executor as executor_module

        print(f"rounds {args.rounds}, spread {args.spread * 1000:.0f} mm, "
              f"minimum same-colour gap {dynamics.MIN_SAME_COLOUR_GAP_M * 1000:.0f} mm\n")
        print("| round | moved mm | missed | worst gap mm | disturbance (mm) | peak cm/s |")
        print("|---|---|---|---|---|---|")

        for seed in range(args.rounds):
            layout = dynamics.randomize(base, seed, spread=args.spread)
            conflicts = dynamics.conflicts(layout)
            if conflicts:
                problems.append(f"seed {seed}: " + "; ".join(conflicts))

            shifts = dynamics.displacement(base, layout)
            biggest = max((math.hypot(dx, dy) for dx, dy in shifts.values()), default=0.0)
            worst_same = _worst_same_colour_gap(layout)

            # Park first, then populate: driving out to the park pose sweeps the table, and
            # place_items is what puts everything back. See executor.PARK_POSE.
            env.reset()
            isaac_io.aim_camera(env, args.scene, sensor)
            bot = executor_module.Executor(env)
            bot.park()
            isaac_io.place_items(env, layout)

            image, depth, camera = isaac_io.capture(env, sensor)
            seen = perceive.perceive(image, source=f"seed{seed}", camera=camera,
                                     frame=isaac_io.robot_frame(env),
                                     plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                     looking_for=colors.colors(colors.DEFAULT_DETECT))
            want_objects = sum(1 for i in layout if not i.container)
            want_containers = sum(1 for i in layout if i.container)
            missed = (len(seen.objects), len(seen.containers)) != (want_objects, want_containers)
            if missed:
                problems.append(f"seed {seed}: saw {len(seen.objects)}/{want_objects} objects "
                                f"and {len(seen.containers)}/{want_containers} containers")

            # A mid-task shove, on an object the arm is not holding.
            shake = dynamics.disturb(env, layout, random.Random(seed))
            shake_mm = shake.moved_m * 1000 if shake else 0.0
            if shake is None or shake.moved_m < dynamics.DISTURB_MIN_MOVE_M:
                problems.append(f"seed {seed}: the shove moved "
                                f"{shake.moved_m * 1000:.1f} mm" if shake else
                                f"seed {seed}: nothing could be shoved")
            else:
                moved.append(shake.moved_m)
                total_moved += shake.moved_m

            shove = f"{shake.prim} {shake.method}" if shake else "-"
            shove_mm = f"{shake.moved_m * 1000:.1f}" if shake else "-"
            speeds = f"{shake.peak_m * 100:.0f}" if shake and shake.method == "shoved" else "-"
            print(f"| {seed} | {biggest * 1000:.1f} | {'MISSED' if missed else 'all'} "
                  f"| {worst_same * 1000:.1f} | {shove} {shove_mm} | {speeds} |")

        print("")
        if moved:
            print(f"shoves moved {min(moved) * 1000:.1f}-{max(moved) * 1000:.1f} mm, "
                  f"mean {total_moved / len(moved) * 1000:.1f} mm, over {len(moved)} round(s)")
        if problems:
            for line in problems:
                print(f"  PROBLEM: {line}")
            print(f"RESULT: FAILED ({len(problems)} problem(s))")
            return 2
        print(f"RESULT: SUCCESS, {args.rounds} round(s)")
        return 0
    finally:
        env.close()


def _worst_same_colour_gap(items) -> float:
    worst = math.inf
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            if first.color == second.color:
                worst = min(worst, dynamics._gap(first, second))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())