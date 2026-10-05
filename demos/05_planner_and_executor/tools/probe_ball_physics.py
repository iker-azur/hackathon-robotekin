"""Why do the balls keep moving, and where can the arm actually reach? Needs Isaac.

Three complaints motivate this probe, and none of them can be fixed by reading the code:

1. **"the balls move with anything touching them"** - a 4 cm sphere on a plane with zero
   angular damping has no reason to stop. The roll-out is a physical property, not a bug in
   the planner, and the only way to choose damping is to measure how far a ball travels for
   a given shove.
2. **"they move and the arm is not close to them"** - either the arm really is closer than
   it looks (a forearm sweeping over a ball is contact), or something other than the arm
   is moving it: a pose written into an overlap, which PhysX resolves by launching the
   body. Both are distinguishable, by watching the arm's links and the ball every step.
3. **"the working area should be bigger"** - the area cannot be grown by arithmetic. The
   reach limit used for refusal is ``MAX_REACH_M`` from the base, and the interesting
   question is where the arm ACTUALLY gets to, because the IK tracks a commanded pose to
   within a few millimetres only well inside the envelope.

    python tools/probe_ball_physics.py --headless
    python tools/probe_ball_physics.py --headless --section roll
    python tools/probe_ball_physics.py --headless --section reach
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
import scene_spec  # noqa: E402

# Nudge speeds to push a ball at. The demo's own disturbance is 0.08..0.28 m/s
# (dynamics.DISTURB_SPEED_M) and the arm's transit across the table was measured moving a
# ball at up to 0.45 m/s, so the sweep brackets both.
NUDGE_SPEEDS_M = (0.10, 0.25, 0.45)

# A ball counts as stopped below this, the same threshold executor.settle_object uses.
STILL_M = 0.002

# And it gets this long to stop. 6.7 s was measured for the worst ball with no damping.
CAP_STEPS = 700

# Travel and hold steps for the reach map. The first version gave 90 steps to reach a point
# and read the position back immediately, and reported points as unreachable that this demo
# picks from every round: the arm had not finished converging. Both numbers are measured
# below - HOLD_STEPS is long enough for the servos to stop creeping.
TRAVEL_STEPS = 130
HOLD_STEPS = 70

# After it stops, it gets this many more steps: a ball that creeps is worse than one that
# rolls visibly, and 60 steps is 1 s.
DRIFT_WATCH_STEPS = 60


def balls_in(items):
    return [i for i in items if i.kind == "ball" and not i.container]


def xy(env, prim) -> tuple[float, float]:
    where = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
    return float(where[0]), float(where[1])


def speed(env, prim) -> float:
    import torch

    return float(torch.linalg.norm(env.scene[prim].data.root_lin_vel_w[0]))


def closest_arm_gap(env, prim) -> float:
    """Smallest distance from any robot link to this object, centre to centre, in metres.

    Centre to centre and not surface to surface, because the links have different sizes and
    guessing a radius per link would be an invented number. What matters here is only
    whether the answer is a few centimetres or several, so the link's own size does not
    change the conclusion.
    """
    links = env.scene["robot"].data.body_pos_w[0].detach().cpu().numpy()
    where = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
    return float(np.min(np.linalg.norm(links[:, :3] - where[None, :3], axis=1)))


def nudge(env, prim, speed_m: float, heading: float = 0.0) -> None:
    """Give one object a shove and nothing else: one write, then physics does the rest."""
    import torch

    direction = torch.tensor([[math.cos(heading), math.sin(heading), 0.0]], device=env.device)
    spin = torch.zeros(1, 3, device=env.device)
    zeros = torch.zeros(1, 6, device=env.device)
    env.scene[prim].write_root_velocity_to_sim(torch.cat([direction * speed_m, spin], dim=-1))
    del zeros


def clear_track(ball) -> tuple[float, float, float]:
    """A clear run for the roll measurement: an empty strip of table, shoved along +x.

    The first version shoved ``ball_red`` where it normally sits, at (0.27, -0.455), which is
    8.5 cm from the blue tray. A ball that rolls 300 mm hits the tray, and the number then
    reported is the tray, not the ball's rolling. This strip at y = -0.62 has 58 cm of
    clear table in front of it and nothing to bounce off.
    """
    return (0.10, -0.62, 0.0)


def set_friction(prim_path: str, value: float) -> bool:
    """Write the friction coefficients on whatever material this body's collider uses.

    Friction is a material, not a rigid-body property, so ``modify_collision_properties``
    cannot reach it - there is no friction field on CollisionPropertiesCfg. Returns whether
    it worked, so the probe reports an honest "could not change it" rather than a number
    that silently means nothing.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import PhysxSchema, Sdf, Usd, UsdShade

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    changed = False
    for child in Usd.PrimRange(prim):
        binding = UsdShade.MaterialBindingAPI(child).GetDirectBinding().GetMaterial()
        if not binding:
            continue
        material = binding.GetPrim()
        # PhysXMaterialAPI is applied to the same prim. The attribute names are set by name
        # rather than through the typed accessors, which differ between PhysX builds - and a
        # friction sweep that silently changed nothing would be worse than no sweep, so the
        # caller's report says whether this returned True.
        PhysxSchema.PhysxMaterialAPI.Apply(material)
        for name in ("staticFriction", "dynamicFriction"):
            material.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        changed = True
    return changed


def roll(env, items) -> None:
    """How far a ball travels for a shove, and how long it takes to stop.

    Swept over damping and friction, because both are choices and the default is not one.
    The demo's balls are spawned with ``RigidBodyPropertiesCfg()``, whose defaults are zero
    linear and zero angular damping, and with the friction default of 0.5: measured on this
    scene, a 10 cm/s shove sends the ball 388 mm and it is still moving after 11.7 s. A
    sphere with no damping has no reason to stop, so "the ball rolls away" is not something
    the planner can fix.

    Measured on a clear 58 cm strip of table rather than where the ball normally sits, and
    each setting is applied to the live body, so all of them are measured in one run.
    """
    from isaaclab.sim import schemas

    ball = balls_in(items)[0]
    prim_path = str(env.scene[ball.prim].root_physx_view.prim_paths[0])
    where = clear_track(ball)
    print(f"section: roll  ({ball.prim}, a {ball.size[0] * 100:.0f} cm ball at {prim_path})")
    print(f"  shoved from ({where[0]:+.2f}, {where[1]:+.2f}) along +x, clear table in front")

    # lin/ang damping, torsional patch radius, static/dynamic friction, sleep threshold.
    settings = [
        (0.0, 0.0, 0.0, 0.5, None),      # what the demo ships today
        (0.0, 2.0, 0.0, 0.5, None),
        (2.0, 8.0, 0.0, 0.5, None),
        (10.0, 40.0, 0.0, 0.5, None),
        (25.0, 100.0, 0.0, 0.5, None),
        (0.0, 0.0, 0.0, 1.0, None),      # friction alone
        (10.0, 40.0, 0.0, 1.0, None),
        (10.0, 40.0, 0.0, 0.5, 1e-4),    # plus PhysX sleeping
        (25.0, 100.0, 0.0, 1.0, 1e-4),
    ]
    print("| lin | ang | friction | sleep |"
          + "".join(f" {shove * 100:.0f} cm/s |" for shove in NUDGE_SPEEDS_M)
          + " worst creep mm |")
    print("|---|---|---|---|" + "---|" * (len(NUDGE_SPEEDS_M) + 1))

    for lin, ang, _patch, friction, sleep in settings:
        schemas.modify_rigid_body_properties(
            prim_path,
            schemas.RigidBodyPropertiesCfg(linear_damping=lin, angular_damping=ang,
                                           sleep_threshold=sleep))
        # The patch radius is swept too, once, to record that it changes nothing: PhysX 5
        # applies it only to contacts with a rotational component it cannot resolve here,
        # so the numbers come back identical to no patch at all.
        schemas.modify_collision_properties(
            prim_path, schemas.CollisionPropertiesCfg(torsional_patch_radius=0.0))
        try:
            set_friction(prim_path, friction)
        except Exception as complaint:  # a material API that differs between builds
            print(f"  (could not set friction to {friction}: {complaint})")

        cells = []
        worst_creep = 0.0
        for shove in NUDGE_SPEEDS_M:
            _park_the_ball(env, ball, where)
            nudge(env, ball.prim, shove, heading=0.0)

            taken = 0
            for taken in range(1, CAP_STEPS + 1):
                env.sim.step(render=False)
                if speed(env, ball.prim) < STILL_M:
                    break
            env.scene.update(dt=env.physics_dt)
            end = xy(env, ball.prim)
            rolled = math.dist(where[:2], end)

            here = end
            for _ in range(DRIFT_WATCH_STEPS):
                env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            creep = math.dist(here, xy(env, ball.prim))
            worst_creep = max(worst_creep, creep)

            stopped = f"{taken / 60.0:.2f}s" if taken < CAP_STEPS else "NEVER"
            cells.append(f" {rolled * 1000:4.0f}mm/{stopped} |")
        print(f"| {lin:.0f} | {ang:.0f} | {friction:.1f} | "
              + (f"{sleep:g} |" if sleep else "  -   |") + "".join(cells)
              + f" {worst_creep * 1000:5.1f} |")

    schemas.modify_rigid_body_properties(
        prim_path, schemas.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0,
                                                  sleep_threshold=None))
    set_friction(prim_path, 0.5)
    isaac_io.place_items(env, items)
    print()


def _park_the_ball(env, ball, where) -> None:
    """Put the ball on the clear strip at rest, moving nothing else."""
    import torch

    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    z = scene_spec.TABLE_TOP_Z + ball.height / 2.0
    obj = env.scene[ball.prim]
    obj.write_root_pose_to_sim(torch.cat(
        [torch.tensor([[where[0], where[1], z]], device=env.device), upright], dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(30):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def contact(env, items) -> None:
    """Is the arm actually touching a ball when the ball moves?"""
    import torch

    balls = balls_in(items)
    print("section: contact  (closest robot link to each ball while it parks, and how far "
          "each ball then goes)")
    bot = executor_module.Executor(env)

    isaac_io.place_items(env, items)
    bot.current = bot._gripper_pos().clone()
    before = {b.prim: xy(env, b.prim) for b in balls}
    worst = {b.prim: 9.9 for b in balls}

    # park() again, but stepping by hand so every link can be measured against every ball
    # on every step. park() itself is 40 steps of drive to PARK_POSE.
    goal = (torch.as_tensor(np.asarray(executor_module.PARK_POSE, dtype=np.float32),
                            device=bot.device) - bot._hold)
    start = bot.current.clone()
    with torch.no_grad():
        for index in range(1, executor_module.PARK_STEPS + 1):
            bot.current = start + (goal - start) * min(
                1.0, index / (0.8 * executor_module.PARK_STEPS))
            env.step(bot._action(bot.current, executor_module.OPEN))
            for b in balls:
                worst[b.prim] = min(worst[b.prim], closest_arm_gap(env, b.prim))
    env.scene.update(dt=env.physics_dt)

    after = {b.prim: xy(env, b.prim) for b in balls}
    for b in balls:
        print(f"  {b.prim:12s} closest link {worst[b.prim] * 1000:6.1f} mm, "
              f"moved {math.dist(before[b.prim], after[b.prim]) * 1000:6.1f} mm")
    print(f"  a 4 cm ball's own radius is {b.size[0] * 500:.0f} mm, so a gap much above that "
          f"is the arm nowhere near it\n")


def teleport(env, items) -> None:
    """Does writing a pose into an overlap launch the object?

    This is the other way an object moves with the arm nowhere near it: ``place_items``
    writes a pose outright, and PhysX resolves any overlap that creates by pushing the
    bodies apart at up to ``maxDepenetrationVelocity``, which defaults high.
    """
    ball = balls_in(items)[0]
    print("section: teleport  (writing a pose that overlaps a solid, and watching what escapes)")
    import torch

    isaac_io.place_items(env, items)
    for label, target in (
        ("onto the tray floor", (ball.x, ball.y, scene_spec.TABLE_TOP_Z + ball.height / 2 - 0.02)),
        ("0.5 mm into the table", (ball.x, ball.y, scene_spec.TABLE_TOP_Z + ball.height / 2 - 0.0005)),
    ):
        isaac_io.place_items(env, items)
        obj = env.scene[ball.prim]
        upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        obj.write_root_pose_to_sim(torch.cat(
            [torch.tensor([[target[0], target[1], target[2]]], device=env.device), upright], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

        peak = 0.0
        for _ in range(60):
            env.sim.step(render=False)
            peak = max(peak, speed(env, ball.prim))
        env.scene.update(dt=env.physics_dt)
        print(f"  {label:24s} peak speed {peak * 100:6.1f} cm/s")

    isaac_io.place_items(env, items)
    print()


def clear_the_table(env, items) -> None:
    """Move every object out of the way, so a reach measurement measures the arm.

    The first two reach maps drove the arm to each point with all six objects sitting on the
    table, and the unreachable patch came out exactly where the two 5 cm trays are - the arm
    was hitting them. A map of "where the arm cannot go" that is really "where the arm cannot
    go past the furniture" is the wrong measurement for sizing a working area, so the objects
    are dropped below the table first and the arm is measured in free space.
    """
    import torch

    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    away = torch.tensor([[0.35, -0.60, -0.60]], device=env.device)
    for item in items:
        env.scene[item.prim].write_root_pose_to_sim(torch.cat([away, upright], dim=-1))
        env.scene[item.prim].write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(20):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)


def reach(env) -> None:
    """Where does the arm actually get to, in free space? Commanded hold point, read back.

    The refusal logic and the pen are both sized from ``MAX_REACH_M``, which is a figure for
    the hardware. This is the figure for this simulation: the table is cleared, the gripper is
    driven to each point in a grid, held, and the position it actually reached is read back.
    A point is usable when the hold point lands within ``OK_M`` of where it was sent.
    """
    OK_M = 0.010
    bot = executor_module.Executor(env)
    items = list(scene_spec.DEFAULT_SCENE)
    isaac_io.place_items(env, items)
    bot.park()
    clear_the_table(env, items)

    height = scene_spec.TABLE_TOP_Z + 0.02  # ball height: the taller of the two objects
    xs = [0.15 + 0.05 * i for i in range(10)]
    ys = [-0.62 + 0.04 * i for i in range(8)]

    print("section: reach  (table cleared; hold point commanded, then read back; "
          f"ok if within {OK_M * 1000:.0f} mm)")
    print(f"  base at {scene_spec.ROBOT_XY}, MAX_REACH_M = {scene_spec.MAX_REACH_M}")
    print(f"  {TRAVEL_STEPS} steps to travel, then {HOLD_STEPS} to hold before reading back")
    print("        " + " ".join(f"{y:+.2f}" for y in ys))

    reached: dict[tuple[float, float], float] = {}
    # Serpentine, so consecutive points are neighbours and the arm is never asked to cross
    # the whole table between two measurements. The first attempt drove back to the parked
    # pose after every single point and never gave the arm time to converge, so a point that
    # is comfortably reachable was reported as missed.
    for row, x in enumerate(xs):
        columns = ys if row % 2 == 0 else list(reversed(ys))
        for y in columns:
            point = np.asarray((x, y, height), dtype=float)
            bot.do("down", point, executor_module.OPEN, steps=TRAVEL_STEPS)
            bot.do("down", point, executor_module.OPEN, steps=HOLD_STEPS)
            achieved = bot._gripper_pos()[0].detach().cpu().numpy() + bot._hold[0].cpu().numpy()
            reached[(x, y)] = float(np.linalg.norm(achieved - point))
        # End of each row: go back to the parked pose once, not once per column.
        bot.do("above", np.asarray(executor_module.PARK_POSE, dtype=float),
               executor_module.OPEN, steps=90)
        print(f"  x={x:+.2f} " + " ".join(
            ("o" if reached[(x, y)] <= OK_M else ".") for y in ys)
            + f"   worst {max(reached[(x, y)] for y in ys) * 1000:.0f} mm")
    print("  (o = reached, . = missed)")

    print("\nmeasured x band per y, against what MAX_REACH_M arithmetic predicts:")
    for y in ys:
        band = [x for x in xs if reached[(x, y)] <= OK_M]
        dy = abs(y - scene_spec.ROBOT_XY[1])
        span = math.sqrt(max(0.0, scene_spec.MAX_REACH_M ** 2 - dy ** 2))
        got = (f"x in [{min(band):+.3f}, {max(band):+.3f}] ({len(band)}/{len(xs)})"
               if band else "nothing reachable")
        print(f"  y={y:+.2f}  measured {got:40s} "
              f"predicted x in [{scene_spec.ROBOT_XY[0] - span:+.3f}, "
              f"{scene_spec.ROBOT_XY[0] + span:+.3f}]")

    # The number this section exists to produce: the biggest all-reachable rectangle, which is
    # what a walled working area has to fit inside. Reported rather than chosen here, because
    # picking the rectangle is scene_spec's job and it needs the grid to have been measured.
    best = None
    for i0 in range(len(xs)):
        for j0 in range(len(ys)):
            for i1 in range(i0 + 1, len(xs)):
                for j1 in range(j0 + 1, len(ys)):
                    if all(reached[(xs[i], ys[j])] <= OK_M
                           for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)):
                        area = (xs[i1] - xs[i0] + 0.05) * (ys[j1] - ys[j0] + 0.04)
                        if best is None or area > best[0]:
                            best = (area, xs[i0] - 0.025, xs[i1] + 0.025,
                                    ys[j0] - 0.02, ys[j1] + 0.02)
    if best:
        area, x0, x1, y0, y1 = best
        print(f"\nlargest fully reachable rectangle on this grid: "
              f"x [{x0:+.3f}, {x1:+.3f}] y [{y0:+.3f}, {y1:+.3f}] "
              f"= {(x1 - x0) * 100:.0f} x {(y1 - y0) * 100:.0f} cm = {area * 10000:.0f} cm2")
        print(f"  the current working area is "
              f"{(scene_spec.WORK_X[1] - scene_spec.WORK_X[0]) * 100:.0f} x "
              f"{(scene_spec.WORK_Y[1] - scene_spec.WORK_Y[0]) * 100:.0f} cm = "
              f"{(scene_spec.WORK_X[1] - scene_spec.WORK_X[0]) * (scene_spec.WORK_Y[1] - scene_spec.WORK_Y[0]) * 10000:.0f} cm2")
    print()


def robot_frame(env) -> None:
    """Which way the arm actually faces, and what that does to the reachable map.

    The map above is a crescent, not a rectangle, and it is wildly lopsided: the arm reaches
    x = 0.60 and cannot reach x = 0.25, at the same distance from the base. A disc of reach
    centred on the base cannot be that lopsided, so the base must be turned, and the scene is
    laid out in world x and y while the arm works in its own frame. This section reports the
    base's own axes and re-expresses the base position in them, because that is what decides
    where the working area can be.
    """
    frame = isaac_io.robot_frame(env)
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    origin = robot.data.root_pos_w[0].detach().cpu().numpy()
    print("section: robot frame")
    print(f"  base position   {np.round(origin, 4).tolist()}")
    print(f"  base quaternion {np.round(quat, 4).tolist()}")
    print(f"  front axis (its +y) {np.round(frame.front_axis, 4).tolist()}  "
          f"= {math.degrees(math.atan2(frame.front_axis[1], frame.front_axis[0])):+.1f} deg off "
          f"world +x")
    print(f"  left  axis (its +x) {np.round(frame.left_axis, 4).tolist()}")
    print(f"  scene_spec.ROBOT_XY = {scene_spec.ROBOT_XY}")
    print("  An identity quaternion means the arm faces world +y and its left is world -x, so "
          "the scene is")
    print("  already laid out in the arm's own frame. The crescent in the reach map is "
          "therefore NOT a")
    print("  rotated base: it is this arm's joint limits and self-collision, and it has to be "
          "measured,")
    print("  not computed from MAX_REACH_M.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--section", default="all",
                        choices=["all", "roll", "contact", "teleport", "frame", "reach"])
    args = parser.parse_args()

    items = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", items)
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        print()
        if args.section in ("all", "frame"):
            robot_frame(env)
        if args.section in ("all", "roll"):
            roll(env, items)
        if args.section in ("all", "contact"):
            contact(env, items)
        if args.section in ("all", "teleport"):
            teleport(env, items)
        if args.section in ("all", "reach"):
            reach(env)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())