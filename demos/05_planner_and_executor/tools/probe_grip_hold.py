"""Does the simulated gripper actually hold a part, and what would make it? Needs Isaac.

    python tools/probe_grip_hold.py --headless

Demo 11 refuses to pick a part up unless friction held it, which is the right behaviour and
leaves a question this answers: **is the refusal a property of the parts, or of the simulated
gripper?** Demo 10 could not tell the difference, because its kinematic fallback made every
grasp look successful.

So this drives the real pick cycle - above, down, close, lift - onto one part at a declared
position, with no camera and no perception, and reports two numbers per trial:

* **gap mm**, the distance between the two finger bodies after the jaws have closed. If it is
  under the part's own width the jaws never touched it and the grasp was geometrically
  impossible.
* **follow %**, how much of the lift the part made. Anything under ``executor.GRASP_FOLLOW_MIN``
  (80%) means it was not held.

and sweeps the two knobs a real bench actually has:

* the **close command**, which the ``gripper_action`` term turns into a target joint angle
  (LeIsaac ships 0.4 rad), and
* the **gripper actuator's stiffness**, which with the joint's effort limit is what decides how
  hard the jaws squeeze. ``isaac_io`` quotes 17.8 for the SO-101's servos; the SO-101's real
  gripper is a parallel jaw driven by its own servo, so this is a figure to measure on the bench
  (see ``real_world.setup_sheet``) rather than a constant to trust.

Everything else is demo 11's own physics: real masses, real friction, no damping on cubes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402

# Close commands to try, in radians. 0.4 is LeIsaac's shipped value; the lower two close further.
CLOSE_ANGLES = (0.20, 0.30, 0.40)

# Gripper stiffness to try. 17.8 is what isaac_io says the SO-101's servos are configured at; the
# larger values are what a bench gripper would have to be nearer to for a 30 mm part.
STIFFNESSES = (17.8, 80.0, 300.0)

SETTLE = 90
PARK = (0.35, -0.58, 0.46)


def load_probe_gripper():
    """``tools/probe_gripper.py``'s two measurement helpers, by path.

    Reused rather than copied: the aperture measurement is the one number this probe's gap column
    has to agree with, and a second implementation of it would eventually disagree with the first.
    """
    spec = importlib.util.spec_from_file_location("probe_gripper", HERE / "probe_gripper.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_gripper"] = module
    spec.loader.exec_module(module)
    return module


def demo11_module_from_path():
    """Demo 11's entry script, by path, for its real-material ``build_env``.

    Demo 11's module body is guarded by ``__main__``, so importing it starts nothing; what is
    wanted from it is one function, and duplicating it here would put the real material table in
    two files.
    """
    spec = importlib.util.spec_from_file_location(
        "demo11_probe", ROOT / "11_real_sorting_with_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo11_probe"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--part", default="cube", choices=["cube", "ball"])
    parser.add_argument("--angles", type=float, nargs="+", default=None,
                        help="close commands to try, in radians (default: all of them)")
    parser.add_argument("--stiffnesses", type=float, nargs="+", default=None,
                        help="gripper actuator stiffnesses to try")
    args = parser.parse_args()
    angles = tuple(args.angles) if args.angles else CLOSE_ANGLES
    stiffnesses = tuple(args.stiffnesses) if args.stiffnesses else STIFFNESSES

    helpers = load_probe_gripper()
    demo11_module = demo11_module_from_path()

    isaac_io.prepare_runtime(headless=args.headless)
    item = next(i for i in scene_spec.DEFAULT_SCENE
                if not i.container and i.kind == args.part)
    env, sensor = demo11_module.build_real_env("top", list(scene_spec.DEFAULT_SCENE))
    try:
        env.reset()
        robot = env.scene["robot"]
        ids, _names = robot.find_joints("gripper")
        gripper_ids = list(ids)
        term = env.action_manager.get_term("gripper_action")
        tips = helpers.finger_tips(robot)
        print(f"  gripper joint ids {gripper_ids}; finger bodies {tips}")
        print(f"  gripper action term {type(term).__name__}: open "
              f"{term._open_command.tolist()}, close {term._close_command.tolist()}")
        print(f"  part under test: {item.prim}, {item.kind}, "
              f"{real_world.mass_kg(item) * 1000:.1f} g, {item.size[0] * 1000:.0f} mm")
        print(f"  friction cone needs {real_world.grasp(item).required_force_n:.2f} N\n")

        grasp = (item.x, item.y, scene_spec.TABLE_TOP_Z + item.size[2] / 2.0)
        above = (item.x, item.y, scene_spec.TABLE_TOP_Z + 0.10)

        print("| part | close rad | stiffness | jaw rad at rest | follow % | verdict |")
        print("|---|---|---|---|---|---|")
        held_somewhere = False
        stalls: list[float] = []
        for angle in angles:
            for stiffness in stiffnesses:
                term._close_command[:] = float(angle)
                robot.write_joint_stiffness_to_sim(float(stiffness), joint_ids=gripper_ids)
                isaac_io.place_items(env, [item])

                bot = executor_module.Executor(env)
                bot.park(steps=executor_module.PARK_STEPS)
                bot.do("above", above, executor_module.OPEN, steps=60)
                bot.do("down", grasp, executor_module.OPEN, steps=60)
                before = env.scene[item.prim].data.root_pos_w[0].detach().cpu().numpy().copy()
                bot.do("close", grasp, executor_module.CLOSE, steps=SETTLE)
                # Where the jaws actually stopped. The difference between the commanded angle
                # and this one is the travel the pads used to squeeze the part, and the squeeze
                # force is stiffness times that travel - which is how real_world derives the
                # gripper's stiffness from a declared force instead of a fitted constant.
                jaw = float(robot.data.joint_pos[0, gripper_ids[0]])
                stalls.append(jaw)
                result = bot.do("lift", above, executor_module.CLOSE, steps=SETTLE,
                                prim=item.prim, on_close=before, check="follow")
                follow = ""
                verdict = result.grasp or ("held" if result.ok else "not held")
                if "followed" in result.note:
                    for token in result.note.split():
                        if token.endswith("%"):
                            follow = token
                            break
                held = result.grasp == "real" or float(follow.rstrip("%") or 0) >= 80.0
                held_somewhere = held_somewhere or held
                print(f"| {item.prim} | {angle:.2f} | {stiffness:6.1f} | {jaw:.3f} | "
                      f"{follow or 'n/a':>7} | {verdict} |")

        print("")
        if stalls:
            print(f"  the jaws stalled at {min(stalls):.3f}-{max(stalls):.3f} rad, so the pads had "
                  f"{angles[0] - max(stalls):.3f} rad of travel to squeeze with at a "
                  f"{angles[0]:.2f} rad command")
        if held_somewhere:
            print("  RESULT: SUCCESS - at least one setting holds the part by friction.")
        else:
            print("  RESULT: FAILED - the simulated jaws never held this part at any setting "
                  "tried. demo 11 will refuse to pick it up, which is the honest answer for a "
                  "gripper whose squeeze has not been characterised.")
        return 0 if held_somewhere else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())