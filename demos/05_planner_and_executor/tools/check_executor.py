"""Does the executor still move a cube the way demo 05 does? Needs Isaac.

``demos/01_simulation_basics/05_cube_pick_and_place.py`` ends in::

    cube moved by (+10.2, -0.1) cm, wanted (+10.0, +0.0) cm
    RESULT: SUCCESS

That is the only motion result this repository had before demo 10, and it is measured against
the simulator's own physics rather than against a plan. The executor here is that loop pointed
at arbitrary poses, so before pointing it at camera-derived poses it has to reproduce the same
thing.

Two checks:

1. **Parity.** A hardcoded pick and move of 10 cm in x, run through ``Executor`` rather than
   demo 05's inline waypoint table. Success means within 2 cm of the target, which is demo
   05's own criterion.
2. **Camera-derived.** The same cycle but with the grasp pose and the drop pose read out of a
   recorded scene, so the error that matters - localisation - is in the loop.

    python tools/check_executor.py --headless
    python tools/check_executor.py --headless --scene trial013
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
import planner  # noqa: E402
import scene_spec  # noqa: E402
from scene import Frame, Scene  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"

# Demo 05's own numbers, for the parity check.
DEMO_05_START = (0.30, -0.42)
DEMO_05_MOVE_BY = (0.10, 0.0)
TOLERANCE_M = 0.02

# Where the other objects go while the parity lane is cleared: below the table, out of the way
# of the carry. They are put back from their stashed poses afterwards.
STASH_AT = (0.35, -0.60, -0.60)


def write_pose(env, prim, xyz) -> None:
    """Write one object's position directly, for stashing it out of the way."""
    import torch

    obj = env.scene[prim]
    pose = torch.tensor([[xyz[0], xyz[1], xyz[2], 1.0, 0.0, 0.0, 0.0]], device=env.device)
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))


def settle(env, steps: int = 25) -> None:
    for _ in range(steps):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def place_cube(env, x, y) -> None:
    import torch

    obj = env.scene[scene_spec.SHIPPED_CUBE]
    pose = torch.tensor([[x, y, scene_spec.TABLE_TOP_Z + 0.015, 1.0, 0.0, 0.0, 0.0]],
                        device=env.device)
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(20):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def cube_xyz(env):
    return env.scene[scene_spec.SHIPPED_CUBE].data.root_pos_w[0].detach().cpu().numpy()


def cube_xy(env):
    return cube_xyz(env)[:2]


def gripper_xyz(bot):
    return bot._gripper_pos()[0].detach().cpu().numpy()


def trace(bot, label, wanted):
    """Where the gripper ended up against where it was asked to go."""
    got = gripper_xyz(bot)
    off = float(np.linalg.norm(got - np.asarray(wanted)))
    print(f"    {label:9s} gripper at ({got[0]:+.4f}, {got[1]:+.4f}, {got[2]:+.4f})"
          f"   wanted ({wanted[0]:+.4f}, {wanted[1]:+.4f}, {wanted[2]:+.4f})"
          f"   off {off * 1000:.0f} mm")
    return off


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-park", action="store_true",
                        help="Skip park_arm before driving, which is what demo 05 does")
    parser.add_argument("--show-action", action="store_true", help="Print the first action vector")
    parser.add_argument("--scene", default="trial000",
                        help="Recorded scene for the camera-derived check")
    parser.add_argument("--regrasp", type=int, default=2,
                        help="re-perceive and replan a stale grasp this many times")
    args = parser.parse_args()

    items = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", items)
    problems = []
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        # Deliberately NOT isaac_io.park_arm: writing joints leaves the arm undrivable.
        # bot.park() drives there through the IK action manager instead.
        #
        # Park BEFORE placing the items. Driving out to the park pose sweeps the forearm across
        # the near half of the table, and it shoves the balls as it goes: measured with
        # tools/probe_ball_roll.py, they drift 1.5 mm during place_items but 27 mm by the first
        # 50 park steps, 90 mm by the end, and the scene no longer matches what it declares.
        # An empty table has nothing to knock over.
        bot = executor_module.Executor(env, log=print)
        if args.no_park:
            print("\n(no park: arm left where it reset, to show the difference)")
        else:
            bot.park()
        isaac_io.place_items(env, items)
        # No settle() here: place_items already steps until nothing is moving, and 60 more
        # steps of env.step() let the balls creep 11 mm, which is the whole error budget.

        # ---- 0. is the table visible while parked? ---------------------------
        print("\n== the parked arm must not hide the table ==")
        check_image, check_depth, check_camera = isaac_io.capture(env, sensor)
        parked = perceive.perceive(check_image, source="parked", camera=check_camera,
                                   frame=isaac_io.robot_frame(env),
                                   plane_z=scene_spec.TABLE_TOP_Z, depth=check_depth,
                                   looking_for=colors.colors(colors.DEFAULT_DETECT))
        print(f"  {parked.summary()}")
        print(f"  objects {[(t.id, t.kind, t.color) for t in parked.objects]}")
        print(f"  containers {[(t.id, t.kind, t.color) for t in parked.containers]}")
        if len(parked.objects) != 4 or len(parked.containers) != 2:
            problems.append(f"parked arm hides the table: {len(parked.objects)} objects and "
                            f"{len(parked.containers)} containers seen, expected 4 and 2")

        # ---- 1. parity with demo 05 ------------------------------------------
        print("\n== parity: hardcoded pick, move 10 cm in x ==")
        # The other five objects are moved out of the lane first, and put back afterwards.
        #
        # This is a motion test, not a scene test: it asks whether the executor can still do
        # demo 05's 10 cm carry. With the current layout the shipped cube sits at x = 0.290 and
        # ``ball_red`` at x = 0.360, so a 10 cm carry to x = 0.390 drags the cube straight
        # through the ball. That is a collision, not an executor fault, and it measured as
        # "+9.9, +2.5 cm, wanted +10.0, +0.0" - the 2.5 cm of y was the ball deflecting a cube
        # the jaws had never actually closed on (the lift reported "followed 3%").
        stashed = {}
        for item in scene_spec.DEFAULT_SCENE:
            if item.prim == scene_spec.SHIPPED_CUBE:
                continue
            stashed[item.prim] = env.scene[item.prim].data.root_pos_w[0].detach().cpu().numpy().copy()
            isaac_io.place_items(env, [item])  # keeps the layout bookkeeping honest
            write_pose(env, item.prim, STASH_AT)
        settle(env)
        place_cube(env, *DEMO_05_START)
        start = cube_xy(env).copy()

        grasp = np.array([start[0], start[1], scene_spec.TABLE_TOP_Z + 0.015])
        target = grasp + np.array([DEMO_05_MOVE_BY[0], DEMO_05_MOVE_BY[1], 0.0])
        above = grasp + np.array([0.0, 0.0, executor_module.ABOVE])
        prim = scene_spec.SHIPPED_CUBE
        before = cube_xyz(env).copy()

        worst = 0.0
        for name, pose, grip, prim_, close, check in (
            ("above", above, executor_module.OPEN, None, None, None),
            ("down", grasp, executor_module.OPEN, None, None, None),
            # The same two checks the executor's own run() applies: "reach" on the close,
            # "follow" on the lift. Passing them explicitly is what makes this parity loop
            # exercise the real grasp logic rather than a stripped-down copy of it.
            ("close", grasp, executor_module.CLOSE, prim, before, "reach"),
            ("lift", above, executor_module.CLOSE, prim, before, "follow"),
            ("carry", target + np.array([0.0, 0.0, executor_module.ABOVE]), executor_module.CLOSE, None, None, None),
            ("lower", target, executor_module.CLOSE, None, None, None),
            ("release", target, executor_module.OPEN, None, None, None),
            ("retreat", target + np.array([0.0, 0.0, executor_module.ABOVE]), executor_module.OPEN, None, None, None),
        ):
            result = bot.do(name, pose, grip, prim=prim_, on_close=close, check=check)
            if args.show_action and name == "down":
                action = bot._action(bot.current, grip)
                print(f"      action shape {tuple(action.shape)} dtype {action.dtype}")
                print(f"      action      {[round(float(v), 4) for v in action[0]]}")
                print(f"      env.action_manager terms: "
                      f"{sorted(getattr(env, 'action_manager', None).active_terms) if getattr(env, 'action_manager', None) else 'none'}")
            off = trace(bot, name, pose - np.asarray(executor_module.HOLD))
            worst = max(worst, off)
            if result.note:
                print(f"      {result.note}")
        bot.drop()
        bot.settle()
        print(f"  worst gripper tracking error: {worst * 1000:.0f} mm")

        # Put the scene back before anything else is measured against it.
        for prim, pose in stashed.items():
            write_pose(env, prim, pose)
        settle(env)

        moved = cube_xy(env) - start
        error = float(np.linalg.norm(moved - np.asarray(DEMO_05_MOVE_BY)))
        print(f"  grasp          : {result.grasp or 'none'}  {result.note}")
        print(f"  cube moved by ({moved[0] * 100:+.1f}, {moved[1] * 100:+.1f}) cm, "
              f"wanted ({DEMO_05_MOVE_BY[0] * 100:+.1f}, {DEMO_05_MOVE_BY[1] * 100:+.1f}) cm")
        print(f"  error {error * 100:.1f} cm, tolerance {TOLERANCE_M * 100:.0f} cm")
        if error >= TOLERANCE_M:
            problems.append(f"parity: cube moved {error * 100:.1f} cm from the target")
        print(f"  RESULT: {'SUCCESS' if error < TOLERANCE_M else 'FAILED'}")

        # ---- 2. the same cycle, driven by what the camera sees -------------------
        # Ground truth is THIS demo's own layout, from the item list we placed.
        #
        # It used to be demo 04's trial000.json, which is the wrong reference: that fixture is
        # a different layout (two cubes, no balls), so measuring live perception against it
        # reported a 32 mm "error" that was only the difference between two scenes.
        ball = next((i for i in items if i.prim == "ball_red"), None)
        if ball is None:
            print("\n== camera-derived: skipped, no ball_red in the scene ==")
        else:
            print("\n== camera-derived: pick the red ball, drop it in the red box ==")
            # Reset and park again. The parity test leaves the arm out over the table at the
            # retreat pose, which is directly above the red cube and the green ball: capturing
            # from there found 2 of 4 objects and put the red ball 39 mm out, because it was
            # photographing the arm's own shadow side of the table.
            env.reset()
            isaac_io.aim_camera(env, "top", sensor)
            # Park first, then populate: same reason as the setup above.
            bot = executor_module.Executor(env, log=print)
            bot.park()
            isaac_io.place_items(env, items)
            # No settle(), for the reason given above.
            image, depth, camera = isaac_io.capture(env, sensor)
            live = perceive.perceive(image, source="check", camera=camera,
                                    frame=isaac_io.robot_frame(env),
                                    plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                    looking_for=colors.colors(colors.DEFAULT_DETECT))
            print(f"  {live.summary()}")
            print("  ground truth in the simulator, for comparison:")
            for item in items:
                if item.container:
                    continue
                true = env.scene[item.prim].data.root_pos_w[0].detach().cpu().numpy()
                print(f"    {item.prim:12s} ({true[0]:+.4f}, {true[1]:+.4f}, {true[2]:+.4f}) "
                      f"declared ({item.x:+.4f}, {item.y:+.4f})")

            plan = planner.plan_command(
                {"intent": "pick_and_place",
                 "items": [{"kind": "ball", "name": None, "color": "red", "quantity": 1,
                            "position": None, "place": None}],
                 "place": {"kind": "box", "color": "red"}, "sort_by": "none"},
                live)
            print(f"  plan: {plan.summary}")
            for step in plan.steps:
                print(f"    {step.action:8s} {step.pose[0]:+.4f} {step.pose[1]:+.4f} "
                      f"{step.pose[2]:+.4f}   {step.note}")

            truth = np.array([ball.x, ball.y])
            if truth is None:
                problems.append("the scene has no red ball")
            else:
                seen = next((t for t in live.objects if t.color == "red" and t.kind == "ball"), None)
                if seen is None:
                    problems.append("the live frame has no red ball")
                else:
                    gap = float(np.linalg.norm(np.asarray(seen.pos_m[:2]) - truth))
                    print(f"  red ball at ({seen.pos_m[0]:+.4f}, {seen.pos_m[1]:+.4f}), "
                          f"ground truth ({truth[0]:+.4f}, {truth[1]:+.4f}), "
                          f"{gap * 1000:.1f} mm apart")
                    margin = planner.OBJECT_HEIGHT["ball"] / 2 - gap
                    print(f"  margin to the jaw edge: {margin * 1000:+.1f} mm")
                    if margin <= 0:
                        problems.append(f"the {gap * 1000:.0f} mm error leaves no margin to grasp")

                    # Locating the ball well enough to plan is not the same as picking it up.
                    # Run the plan for real, and re-perceive and replan if the arm knocks the
                    # ball out of the jaws' reach on the way in - which is exactly what the
                    # demo loop does, and what this check exists to hold to account.
                    print("\n  -- executing that plan --")
                    attempts = 1 + args.regrasp
                    run = None
                    for attempt in range(attempts):
                        run = bot.run(plan, resolve=lambda _i: "ball_red")
                        print(f"  {run.summary()}")
                        if not run.stale_grasps:
                            break
                        print(f"  stale grasp ({run.aborted_at}); "
                              f"re-planning, attempt {attempt + 2} of {attempts}")
                        bot.park()
                        isaac_io.place_items(env, items)
                        retry_image, retry_depth, retry_camera = isaac_io.capture(env, sensor)
                        live = perceive.perceive(retry_image, source="retry", camera=retry_camera,
                                                 frame=isaac_io.robot_frame(env),
                                                 plane_z=scene_spec.TABLE_TOP_Z,
                                                 depth=retry_depth,
                                                 looking_for=colors.colors(colors.DEFAULT_DETECT))
                        plan = planner.plan_command(
                            {"intent": "pick_and_place",
                             "items": [{"kind": "ball", "name": None, "color": "red",
                                        "quantity": 1, "position": None, "place": None}],
                             "place": {"kind": "box", "color": "red"}, "sort_by": "none"},
                            live)
                        if not plan.moves:
                            problems.append("nothing left to plan after a stale grasp")
                            break
                    if run is not None and not run.ok:
                        problems.append(f"the camera-derived plan did not run: {run.summary()}")

                    after = env.scene["ball_red"].data.root_pos_w[0].detach().cpu().numpy()
                    tray = next((c for c in live.containers
                                 if c.kind == "box" and c.color == "red"), None)
                    if tray is None:
                        problems.append("the live frame has no red tray")
                    else:
                        centre = np.asarray(tray.pos_m[:2])
                        offset = np.asarray(after[:2]) - centre
                        inner = np.asarray(tray.inner_rect_m[2:]) / 2 if tray.inner_rect_m else 0.028
                        radius = float(np.linalg.norm(offset))
                        print(f"  ball landed {radius * 1000:.1f} mm from the tray centre, "
                              f"at z {after[2]:.4f}")
                        # Outside the cavity in x or y means it missed or hit a wall.
                        inside_xy = (abs(offset[0]) <= inner[0] and abs(offset[1]) <= inner[1])
                        # A ball resting in the cavity sits on the 6 mm floor, so its centre is
                        # at TABLE_TOP + floor + radius. Resting on the table it is at
                        # TABLE_TOP + radius instead, 6 mm lower.
                        floor = scene_spec.TABLE_TOP_Z + 0.006 + planner.OBJECT_HEIGHT["ball"] / 2
                        print(f"  cavity floor puts a resting ball at z {floor:.4f}")
                        if not inside_xy:
                            problems.append(f"the ball landed {radius * 1000:.0f} mm from the tray "
                                            f"centre, outside the {inner[0] * 2000:.0f} mm cavity")
                        elif after[2] < floor - 0.004:
                            problems.append(f"the ball finished at z {after[2]:.4f}, below the "
                                            f"cavity floor at {floor:.4f}: it never left the table")
                        elif not problems:
                            print("  in the tray")

        print("")
        if problems:
            for line in problems:
                print(f"  PROBLEM: {line}")
            print(f"RESULT: FAILED ({len(problems)} problem(s))")
            return 2
        print("RESULT: SUCCESS")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())