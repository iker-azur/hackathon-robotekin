"""Record the scenes the planner tests run against, from this demo's own layout. Needs Isaac.

``tests/fixtures/*.json`` are what ``tests/test_planner.py`` runs on, and they are recorded
camera frames reduced to objects and positions. They were recorded from an older layout - a 16
x 21 cm working area, 4 cm balls, objects in two columns - so once the layout changed they
described a table this demo no longer builds, and grounding correctly refused to plan anything
against them: every position in a recorded frame was outside the new working area.

The recorder in ``demos/04_perception`` cannot be reused for this, and that is deliberate rather
than an omission. It builds demo 04's own ``scene_spec``, and the whole point of demo 05 keeping
its own copy of every module is that demo 10 can change the scene without touching demo 09. So
this records from demo 05's ``isaac_io``, ``perceive`` and ``scene_spec`` and writes straight
into demo 05's ``tests/fixtures``.

    python tools/record_fixtures.py --headless --trials 20
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "tests" / "fixtures"))
    parser.add_argument("--spread", type=float, default=dynamics.DEFAULT_SPREAD_M)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    isaac_io.prepare_runtime(headless=args.headless)
    # No pen: the recorded scenes should be the plain table the planner reasons about, and a
    # kerb around the edge would put a black wall in every recorded frame.
    env, sensor = isaac_io.build_env("top", list(scene_spec.DEFAULT_SCENE), pen=False)
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)

        for trial in range(args.trials):
            layout = dynamics.randomize(list(scene_spec.DEFAULT_SCENE), trial,
                                        spread=args.spread)
            problems = dynamics.conflicts(layout)
            if problems:
                print(f"trial{trial:03d}: SKIPPED, invalid layout: {'; '.join(problems)}")
                continue
            env.reset()
            isaac_io.aim_camera(env, "top", sensor)
            # Park before populating, exactly as the demo does: the park sweep moves whatever
            # is already on the table, and a frame recorded before placing anything would be a
            # frame of an empty table with stale numbers in it.
            isaac_io.park_arm(env)
            isaac_io.place_items(env, layout)

            image, depth, camera = isaac_io.capture(env, sensor)
            seen = perceive.perceive(
                image, source=f"isaac:top:trial{trial}", camera=camera,
                frame=isaac_io.robot_frame(env), plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                looking_for=colors.colors(colors.DEFAULT_DETECT))

            path = out / f"trial{trial:03d}.json"
            seen.write(path)
            errors = []
            for thing in seen.objects:
                item = min(layout, key=lambda i: abs(i.x - (thing.pos_m or [9, 9])[0])
                           + abs(i.y - (thing.pos_m or [9, 9])[1]))
                if thing.pos_m:
                    errors.append(abs(thing.pos_m[0] - item.x) + abs(thing.pos_m[1] - item.y))
            worst = max(errors) if errors else 0.0
            print(f"trial{trial:03d}: {len(seen.objects)} object(s), "
                  f"{len(seen.containers)} container(s), worst position error "
                  f"{worst * 1000:.1f} mm -> {path.name}")

        print(f"\n{args.trials} scene(s) written to {out}")
        print("run tests/test_planner.py against them")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())