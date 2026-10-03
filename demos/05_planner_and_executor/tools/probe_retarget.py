"""Can the arm re-aim at a ball it has already knocked moving? Needs Isaac.

The ball cannot be caught the way a cube is: the arm knocks it 57-62 mm on the way in and it
then rolls a further 100-370 mm while the gripper descends, so the jaws close on empty air.
Three fixes for that were measured and disproven (results/approach_drift.md). This tests the
remaining one, which is not to fight the roll but to join it: aim at where the ball *ended
up*.

That needs two things, and neither has been checked:

1. the ball comes to rest, rather than rolling indefinitely;
2. the camera can still see it once the arm is hovering above where the ball used to be.

If either is false the idea cannot work, so this measures them before any of it is built into
the executor. For each object it prints the time to rest, whether perception still finds it,
where it thinks the ball is, and how far the ball moves when the gripper then descends onto
that freshly measured position - which is the number that decides the whole approach.

    python tools/probe_retarget.py --headless
"""

from __future__ import annotations

import argparse
import math
import sys
import time
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

# Below this the ball counts as stopped. 1 mm/s is slow enough that it will not move
# perceptibly during the following camera capture.
STILL_M = 0.001

# How long to wait for it. The longest roll measured was 371 mm over a 1.7 s descent, so a
# few seconds is generous; the cap is here so a ball that never stops fails fast.
MAX_WAIT_STEPS = 400


def wait_for_rest(env, prim, cap: int = MAX_WAIT_STEPS):
    """Step until the object stops moving. Returns (steps, seconds, moved mm)."""
    start = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()
    taken = 0
    for taken in range(1, cap + 1):
        env.sim.step(render=False)
        speed = float(torch_norm(env.scene[prim].data.root_lin_vel_w[0]))
        if speed < STILL_M:
            break
    env.scene.update(dt=env.physics_dt)
    end = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
    return taken, taken * 0.0167, float(np.linalg.norm(end[:2] - start[:2])) * 1000


def torch_norm(tensor) -> float:
    import torch

    return float(torch.linalg.norm(tensor))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    base = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", base)
    try:
        print("re-aiming at a knocked ball: does it stop, can we see it, does it stay put\n")
        print("| round | object | rolled mm | rest s | seen again | re-aim error mm | "
              "moved on descent mm | verdict |")
        print("|---|---|---|---|---|---|---|---|")

        workable = 0
        attempted = 0
        rest_times: list[float] = []

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

            premise, claimed = {}, set()
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

                above = np.asarray(next(s.pose for s in plan.steps if s.action == "above"))
                grasp = np.asarray(next(s.pose for s in plan.steps if s.action == "down"))

                # 1. reach the hover, which knocks the target
                bot.do("above", above, executor_module.OPEN)
                steps, seconds, rolled = wait_for_rest(env, prim)
                rest_times.append(seconds)

                # 2. can the camera still see it, with the arm parked above where it was?
                again_image, again_depth, again_camera = isaac_io.capture(env, sensor)
                after = perceive.perceive(again_image, source="reaim", camera=again_camera,
                                          frame=isaac_io.robot_frame(env),
                                          plane_z=scene_spec.TABLE_TOP_Z, depth=again_depth,
                                          looking_for=colors.colors(colors.DEFAULT_DETECT))
                item = next(i for i in layout if i.prim == prim)
                reaimed = next((t for t in after.objects
                                if t.kind == thing.kind and t.color == thing.color), None)
                true = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
                seen_again = reaimed is not None
                reaim_error = (float(np.linalg.norm(np.asarray(reaimed.pos_m[:2]) - true[:2]))
                               * 1000) if seen_again else float("nan")

                if not seen_again:
                    # Say where it went and what the camera did see, because "not seen" is
                    # only useful if it distinguishes "hidden behind the arm" from "rolled off
                    # the table", and those need different fixes.
                    print(f"    {prim}: rolled to ({true[0]:+.4f}, {true[1]:+.4f}), "
                          f"camera saw {after.summary()}")
                    for other in after.objects:
                        print(f"      #{other.id} {other.kind} {other.color} "
                              f"({other.pos_m[0]:+.4f}, {other.pos_m[1]:+.4f})")
                    print(f"    gripper is at "
                          f"{[round(float(v), 4) for v in bot._gripper_pos()[0].detach().cpu().numpy()]}")
                    bot.park()
                    isaac_io.place_items(env, layout)
                    print(f"| {round_index} | {prim} | {rolled:.0f} | {seconds:.1f} | "
                          f"NO | - | - | not seen after the knock |")
                    continue

                # 3. descend onto the freshly measured position
                moved = float("nan")
                verdict = "not seen after the knock"
                if seen_again:
                    target = np.array([reaimed.pos_m[0], reaimed.pos_m[1],
                                       grasp[2]], dtype=float)
                    bot.do("above", target + np.array([0, 0, executor_module.ABOVE]),
                           executor_module.OPEN, steps=80)
                    bot.do("down", target, executor_module.OPEN)
                    landed = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
                    moved = float(np.linalg.norm(landed[:2] - target[:2])) * 1000
                    # Within reach is the same bar the executor uses: half the object.
                    reach = item.size[0] * 500
                    attempted += 1
                    if moved <= reach:
                        workable += 1
                        verdict = "RE-AIM WORKS"
                    else:
                        verdict = "still knocked away"
                    # Put it back for the next object in this round.
                    bot.park()
                    isaac_io.place_items(env, layout)

                print(f"| {round_index} | {prim} | {rolled:.0f} | {seconds:.1f} | "
                          f"{'yes' if seen_again else 'NO'} | {reaim_error:.0f} | {moved:.0f} | "
                          f"{verdict} |")
            # reset between rounds is done by the loop head

        print("")
        if rest_times:
            print(f"time to rest: {min(rest_times):.1f}-{max(rest_times):.1f} s "
                  f"(mean {sum(rest_times) / len(rest_times):.1f} s)")
        print(f"the ball stayed put for a re-aimed descent on {workable}/{attempted} approaches")
        print("")
        print("RESULT: SUCCESS" if attempted and workable == attempted else
              f"RESULT: FAILED, {workable}/{attempted} re-aimed descents held the object")
        return 0 if (attempted and workable == attempted) else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())