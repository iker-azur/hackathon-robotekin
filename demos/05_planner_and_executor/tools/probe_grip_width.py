"""Which jaw closing width lets a ball survive the approach? Needs Isaac.

The gripper action is binary and its closed configuration is 0.4 rad, set by the LeIsaac Hub
config this demo does not own. The aperture that produces cannot be measured from the
articulation: the two candidate finger bodies sit a constant 36.2 mm apart at every angle, so
neither of them is a moving finger and geometry will not answer it.

So this measures the outcome instead of the geometry: for each closing width, place a ball,
run the real pick, and report whether the ball was still where the jaws aimed when they closed
and whether it came up with the lift. That is the question the executor already asks, so the
numbers here are the same ones the demo acts on.

    python tools/probe_grip_width.py --headless
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

# The Hub default, then progressively wider. 1.0 is the open command, i.e. no grip at all,
# included as the control: if a ball "survives" at 1.0 then something else is holding it.
WIDTHS = (0.4, 0.5, 0.6, 0.7, 0.85, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--kind", default="ball", choices=["ball", "cube"])
    args = parser.parse_args()

    base = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", base)
    try:
        env.reset()
        term = env.action_manager.get_term("gripper_action")
        print(f"gripper close command as shipped: {term._close_command.tolist()} rad")
        print(f"asking for kind={args.kind}\n")
        print("| close rad | target | reached at close mm | followed lift % | verdict |")
        print("|---|---|---|---|---|")

        held = 0
        tried = 0
        for width in WIDTHS:
            term._close_command[:] = float(width)
            for round_index in range(args.rounds):
                layout = dynamics.randomize(base, round_index)
                env.reset()
                isaac_io.aim_camera(env, "top", sensor)
                bot = executor_module.Executor(env)
                bot.park()
                isaac_io.place_items(env, layout)

                image, depth, camera = isaac_io.capture(env, sensor)
                seen = perceive.perceive(image, source="grip", camera=camera,
                                         frame=isaac_io.robot_frame(env),
                                         plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                         looking_for=colors.colors(colors.DEFAULT_DETECT))
                thing = next((t for t in seen.objects if t.kind == args.kind), None)
                if thing is None:
                    print(f"| {width:.2f} | {args.kind} | - | - | not seen |")
                    continue
                prim = {"ball": "ball_red", "cube": "cube_blue"}[args.kind]

                plan = planner.plan_command(
                    {"intent": "pick_and_place",
                     "items": [{"kind": args.kind, "name": None, "color": thing.color,
                                "quantity": 1, "position": None, "place": None}],
                     "place": {"kind": "box", "color": thing.color}, "sort_by": "none"},
                    seen)
                if not plan.moves:
                    print(f"| {width:.2f} | {prim} | - | - | nothing to plan |")
                    continue

                # ---- let it settle, re-aim, then close ---------------------------
                # An earlier version measured the whole plan without waiting, and reported
                # ~100 mm at every width including fully open 1.0 rad - which says nothing
                # about the gripper, because at that moment the ball is still rolling from
                # the knock. So this waits for the roll to finish, re-aims at where the ball
                # came to rest, and only then asks whether the jaws can hold it.
                env.reset()
                isaac_io.aim_camera(env, "top", sensor)
                bot = executor_module.Executor(env)
                bot.park()
                isaac_io.place_items(env, layout)
                image, depth, camera = isaac_io.capture(env, sensor)
                seen = perceive.perceive(image, source="grip2", camera=camera,
                                         frame=isaac_io.robot_frame(env),
                                         plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                         looking_for=colors.colors(colors.DEFAULT_DETECT))
                thing = next((t for t in seen.objects if t.kind == args.kind), None)
                if thing is None:
                    print(f"| {width:.2f} | {prim} | (settled) | - | - | not seen |")
                    continue
                plan = planner.plan_command(
                    {"intent": "pick_and_place",
                     "items": [{"kind": args.kind, "name": None, "color": thing.color,
                                "quantity": 1, "position": None, "place": None}],
                     "place": {"kind": "box", "color": thing.color}, "sort_by": "none"},
                    seen)
                bot.do("above", np.asarray(next(s.pose for s in plan.steps
                                                if s.action == "above")),
                        executor_module.OPEN)
                steps_taken = bot.settle_object(prim)

                again, again_depth, again_cam = isaac_io.capture(env, sensor)
                after = perceive.perceive(again, source="grip3", camera=again_cam,
                                          frame=isaac_io.robot_frame(env),
                                          plane_z=scene_spec.TABLE_TOP_Z, depth=again_depth,
                                          looking_for=colors.colors(colors.DEFAULT_DETECT))
                reaimed = next((t for t in after.objects
                                if t.kind == args.kind and t.color == thing.color), None)
                if reaimed is None:
                    print(f"| {width:.2f} | {prim} | (settled {steps_taken}) | - | - "
                          f"| lost after the knock |")
                    continue
                true = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
                rest_error = float(np.linalg.norm(np.asarray(reaimed.pos_m[:2]) - true[:2])) * 1000
                grasp = np.array([reaimed.pos_m[0], reaimed.pos_m[1],
                                  scene_spec.TABLE_TOP_Z
                                  + planner.OBJECT_HEIGHT[args.kind] / 2.0])
                before = true.copy()
                lift = grasp + np.array([0, 0, executor_module.ABOVE])
                bot.do("down", grasp, executor_module.OPEN, steps=40)
                bot.do("close", grasp, executor_module.CLOSE, steps=60)
                off = float(np.linalg.norm(
                    env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()[:2]
                    - grasp[:2])) * 1000
                result = bot.do("lift", lift, executor_module.CLOSE, prim=prim,
                                on_close=before, check="follow")
                tried += 1
                ok = result.grasp == "real"
                held += int(ok)
                print(f"| {width:.2f} | {prim} | {rest_error:.0f} | "
                      f"{result.note.split('followed ')[1].split('%')[0] if 'followed' in result.note else '-'} "
                      f"| {'HELD' if ok else result.grasp} |")

        print("")
        print(f"a real friction grasp on {held}/{tried} attempts")
        print("RESULT: SUCCESS" if held else "RESULT: FAILED, no width gave a real grasp")
        return 0 if held else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())