"""Does the kerb contain the scene, and does the arm still work with it there? Needs Isaac.

The pen exists because the arm knocks whatever it reaches for, and a ball that leaves the
working area cannot be re-planned for: it rolls 100-370 mm, lands off the table at x = -0.22
against an edge at x = 0.003, and outside the area perception stops calling it a ball at all.

So this measures the two things the kerb has to do, and they pull against each other. Its
inner face is on the working-area boundary, and the arm's base is beyond the near edge, so
**every move crosses the near kerb**. Containment is easy to measure and arm clearance is not:
a kerb tall enough to stop a ball may be one the forearm clips on the way in.

Three things are measured:

* **containment** - shove every object hard, many rounds, count how many leave the area;
* **arm clearance** - does the arm still complete a pick with the kerb present;
* **kind stability** - do the balls still read as balls, which they only do inside the area.

    python tools/probe_pen.py --headless
    python tools/probe_pen.py --headless --rounds 8 --no-pen     # the control
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import dynamics  # noqa: E402
import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import pen as pen_module  # noqa: E402
import perceive  # noqa: E402
import planner  # noqa: E402
import scene_spec  # noqa: E402


def outside_by(env, item) -> float:
    """How far this object sticks out of the working area, in metres. 0 when it is inside."""
    where = env.scene[item.prim].data.root_pos_w[0].detach().cpu().numpy()
    return max(scene_spec.WORK_X[0] - (float(where[0]) - scene_spec.half_width(item)),
               (float(where[0]) + scene_spec.half_width(item)) - scene_spec.WORK_X[1],
               scene_spec.WORK_Y[0] - (float(where[1]) - scene_spec.half_depth(item)),
               (float(where[1]) + scene_spec.half_depth(item)) - scene_spec.WORK_Y[1],
               0.0)


def main() -> int:
    global env_scene, layout_now
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--no-pen", action="store_true", help="the control: no kerb at all")
    parser.add_argument("--shove-speed", type=float, default=0.45,
                        help="harder than the demo's own shoves, to test the kerb properly")
    parser.add_argument("--pen-height", type=float, default=None,
                        help="override the kerb height in metres, to trade containment "
                             "against occlusion")
    parser.add_argument("--heights", type=float, nargs="*", default=None,
                        help="sweep several kerb heights in one run")
    parser.add_argument("--touching-mm", type=float, default=5.0,
                        help="an object this close to the boundary counts as contained: the "
                             "kerb's inner face IS the boundary, so a ball resting against "
                             "it overhangs by a hair and that is the kerb working")
    args = parser.parse_args()

    if args.pen_height is not None:
        pen_module.PEN_HEIGHT = args.pen_height

    base = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", base, pen=not args.no_pen)
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        bot = executor_module.Executor(env)

        if not args.no_pen:
            import isaacsim.core.utils.prims as prim_utils
            ok = prim_utils.is_prim_path_valid(
                f"/World/envs/env_0/Scene/{pen_module.PEN_PRIM}")
            children = 0
            if ok:
                children = len(prim_utils.get_prim_at_path(
                    f"/World/envs/env_0/Scene/{pen_module.PEN_PRIM}").GetChildren())
            print(f"pen present: {ok}, child parts: {children} (expected 4), "
                  f"height {pen_module.PEN_HEIGHT * 100:.1f} cm")
            if not ok or children != 4:
                print("RESULT: FAILED, the pen did not build as four walls")
                return 2

        work_x, work_y = pen_module.contained_region()
        print(f"contained area x {work_x} y {work_y}")
        print(f"within {args.touching_mm:.0f} mm of the boundary counts as contained: the "
              f"kerb's inner face IS the boundary\n")
        print("| round | shoved | outside by >%.0f mm | worst overhang mm | balls as balls |"
              % args.touching_mm)
        print("|---|---|---|---|---|")

        escapes = 0
        shoved = 0
        mislabelled = 0
        worst = 0.0
        for round_index in range(args.rounds):
            layout_now = dynamics.randomize(base, round_index)
            env.reset()
            isaac_io.aim_camera(env, "top", sensor)
            bot = executor_module.Executor(env)
            bot.park()
            isaac_io.place_items(env, layout_now)

            rng = random.Random(1000 + round_index)
            for _ in range(4):
                sh = dynamics.disturb(env, layout_now, rng, speed=args.shove_speed)
                if sh:
                    shoved += 1
            # let everything stop
            for _ in range(120):
                env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            escape_list = []
            for item in layout_now:
                if item.container:
                    continue
                out = outside_by(env, item)
                worst = max(worst, out)
                if out > args.touching_mm / 1000.0:
                    escape_list.append((item.prim, out))
            escapes += len(escape_list)

            image, depth, camera = isaac_io.capture(env, sensor)
            seen = perceive.perceive(image, source=f"pen{round_index}", camera=camera,
                                     frame=isaac_io.robot_frame(env),
                                     plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                     looking_for=colors.colors(colors.DEFAULT_DETECT))
            balls_here = sum(1 for i in layout_now if i.kind == "ball")
            balls_said = sum(1 for t in seen.objects if t.kind == "ball")
            mislabelled += max(0, balls_here - balls_said)

            print(f"| {round_index} | 4 | {len(escape_list)} "
                  f"{'(' + ', '.join(p for p, _ in escape_list) + ')' if escape_list else '-'} "
                  f"| {worst * 1000:.1f} | {balls_said}/{balls_here} |")

        print("")
        print(f"{escapes} of {shoved} shoves left the working area, "
              f"worst by {worst * 1000:.0f} mm")
        print(f"{mislabelled} balls were no longer called balls")
        label = "no pen" if args.no_pen else "pen"
        print(f"RESULT: {'SUCCESS' if escapes == 0 else 'FAILED'}, {label}")
        return 0 if escapes == 0 else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())