"""How much lateral error can the jaws absorb, and does a wider close angle widen it? Needs Isaac.

    python tools/probe_grasp_centring.py --headless --position 0.445
    python tools/probe_grasp_centring.py --headless --position 0.281 --angles 0.24

**The question.** A 30 mm part between pads 36.2 mm apart has 3.1 mm of lateral slack, and demo 11
measured an aim error of 4-10 mm at the far row: the red cube is held and the blue one never is.
That is the symptom. This tool asks the question behind it - is the slack the binding constraint,
and is it fixed?

It sweeps two things against each other:

* **close angle** - how far the jaws are driven shut. A *wider* close leaves a bigger hole for the
  part to be in, so it should tolerate more lateral error, up to the point where the pads do not
  reach the part at all and nothing is held. ``tools/probe_grip_hold.py`` found 0.20 rad holds a
  cube at 100% and 0.30 rad holds it at 0%, so the window is somewhere between and this measures
  both ends of it.
* **lateral offset** - the aim is deliberately displaced along x by the given amount before
  descending, so the jaws are asked to close beside the part rather than on it.

Everything else is the demo's own path: real masses and friction, the 0.20 rad-era gripper
actuator stiffness, a closed-loop descent that follows the part, and the same 90 steps of close
that ``CLOSE_SETTLE_STEPS`` uses. The part is aimed at from its **true** simulator position, so
what is left in the measurement is the arm's own error and the part's own motion - not perception,
which ``results/grip_and_release.md`` measures separately.

A table of follow percentages is the answer: the widest close angle that still holds at zero
offset, and the largest offset each angle survives.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402

CLOSE_STEPS = 90
TRAVEL = 0.10  # how high the arm hovers before the descent, as ``track`` is given it
HOLD_FRACTION = 80.0  # percent; executor.GRASP_FOLLOW_MIN is 0.8


def demo11_module():
    """Demo 11's entry script by path, for its real-material ``build_env``.

    Its module body is guarded by ``__main__``, so importing it starts nothing; what is wanted is
    one function, and copying it here would put the real material table in two files.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo11_probe", ROOT / "11_real_sorting_with_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo11_probe"] = module
    spec.loader.exec_module(module)
    return module


def trial(env, term, robot, item, x: float, y: float, angle: float, offset_mm: float):
    """One approach: hover, descend on the part, close, lift. Returns what happened."""
    import torch

    term._close_command[:] = float(angle)
    isaac_io.place_items(env, [item])
    part = env.scene[item.prim]
    truth = part.data.root_pos_w[0].detach().cpu().numpy().copy()

    # The aim is the part's true position, displaced by the offset under test. Nothing here reads
    # the camera, so the number that comes out is the arm's error plus the part's own motion.
    aim_x = float(truth[0]) + offset_mm / 1000.0
    aim_y = float(truth[1])
    grasp = (aim_x, aim_y, scene_spec.TABLE_TOP_Z + item.size[2] / 2.0)
    above = (aim_x, aim_y, scene_spec.TABLE_TOP_Z + TRAVEL)

    bot = executor_module.Executor(env)
    bot.park(steps=executor_module.PARK_STEPS)
    bot.do("above", above, executor_module.OPEN, steps=60)

    def observe():
        here = part.data.root_pos_w[0].detach().cpu().numpy()
        return (float(here[0]), float(here[1]))

    descent = bot.track(observe=observe, target_z=grasp[2],
                        steps=executor_module.TRACK_DESCEND_STEPS, action="down",
                        grip=executor_module.OPEN, aim=(aim_x, aim_y))

    # Where the jaws actually are when they start closing, and where the part actually is. This
    # is the number the whole probe is about: the part must be within the pads' lateral slack of
    # the gripper's centre, and nothing in the plan can know that unless it is measured.
    gripper = bot._gripper_pos()[0].detach().cpu().numpy()
    at_close = part.data.root_pos_w[0].detach().cpu().numpy().copy()
    lateral = math.dist((float(at_close[0]), float(at_close[1])),
                        (float(gripper[0]), float(gripper[1])))
    # Where the part is relative to where the jaws were told to close, which is perception's error
    # plus whatever the descent moved it.
    aim_error = math.dist((float(at_close[0]), float(at_close[1])), (aim_x, aim_y))
    moved_by_descent = math.dist((float(at_close[0]), float(at_close[1])),
                                 (float(truth[0]), float(truth[1])))

    before = at_close.copy()
    lift = bot.do("lift", above, executor_module.CLOSE, steps=CLOSE_STEPS, prim=item.prim,
                  on_close=before, check="follow")
    follow = 0.0
    for token in lift.note.split():
        if token.endswith("%"):
            follow = float(token.rstrip("%"))
    held = lift.grasp == "real" or follow >= HOLD_FRACTION
    return {
        "angle": angle,
        "offset_mm": offset_mm,
        "lateral_mm": lateral * 1000.0,
        "aim_error_mm": aim_error * 1000.0,
        "descent_mm": moved_by_descent * 1000.0,
        "follow": follow,
        "grasp": lift.grasp or ("held" if lift.ok else "not held"),
        "held": held and descent.ok,
        "lost": lift.grasp == "lost",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--position", type=float, nargs="+", default=[0.445],
                        help="x positions to test, in metres")
    parser.add_argument("--y", type=float, default=-0.435)
    parser.add_argument("--angles", type=float, nargs="+", default=[0.20, 0.24, 0.28],
                        help="close commands to try, in radians")
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.0, 3.0, 6.0, 9.0],
                        help="lateral offsets to displace the aim by, in millimetres")
    parser.add_argument("--part", default="cube", choices=["cube", "ball"])
    args = parser.parse_args()

    demo11 = demo11_module()
    isaac_io.prepare_runtime(headless=args.headless)
    item = replace(next(i for i in scene_spec.DEFAULT_SCENE
                        if not i.container and i.kind == args.part),
                   x=args.position[0], y=args.y)
    env, _sensor = demo11.build_real_env(
        "top", [item] + [i for i in scene_spec.DEFAULT_SCENE if not i.container and i.prim != item.prim]
        + [i for i in scene_spec.DEFAULT_SCENE if i.container], pen=False,
        grip_stiffness=real_world.GRIPPER.actuator_stiffness)

    rows = []
    try:
        env.reset()
        robot = env.scene["robot"]
        term = env.action_manager.get_term("gripper_action")
        for x in args.position:
            where = replace(item, x=x, y=args.y)
            isaac_io.place_items(env, [where])
            for angle in args.angles:
                for offset in args.offsets:
                    row = trial(env, term, robot, where, x, args.y, angle, offset)
                    rows.append(row)
                    print(f"x={x:.3f} close={angle:.2f} offset={offset:+.0f}mm -> "
                          f"lateral {row['lateral_mm']:5.1f} mm, aim error "
                          f"{row['aim_error_mm']:5.1f} mm, descent moved "
                          f"{row['descent_mm']:4.1f} mm, follow {row['follow']:3.0f}% -> "
                          f"{'HELD' if row['held'] else 'not held'}", flush=True)
    finally:
        env.close()

    print("\n| x | close rad | offset mm | jaws-to-part mm | follow % | held |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {args.position[0]:.3f} | {row['angle']:.2f} | {row['offset_mm']:+.0f} | "
              f"{row['lateral_mm']:.1f} | {row['follow']:.0f} | {'yes' if row['held'] else 'no'} |")

    print("")
    for x in args.position:
        for angle in args.angles:
            group = [r for r in rows if r["angle"] == angle and r["x"] == x] if "x" in rows[0] \
                else [r for r in rows if r["angle"] == angle]
            if not group:
                continue
            held_offsets = [r["offset_mm"] for r in group if r["held"]]
            reach = max(held_offsets) if held_offsets else None
            print(f"x={x:.3f} close {angle:.2f} rad: held at offsets "
                  f"{', '.join(f'{o:+.0f}' for o in held_offsets) if held_offsets else 'none'}"
                  f" -> {'tolerates ' + f'{reach:+.0f} mm' if reach is not None else 'holds nothing'}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())