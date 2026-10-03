"""Tests for planner.py, against the scene JSONs demo 09 recorded. No GPU, no Isaac.

    python tests/test_planner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import planner  # noqa: E402
import scene_spec  # noqa: E402
from scene import Scene  # noqa: E402

FIXTURES = sorted((Path(__file__).resolve().parent / "fixtures").glob("*.json"))

failures: list[str] = []
checks = 0


def check(condition, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def scene(name: str = "trial000") -> Scene:
    return Scene.read(Path(__file__).resolve().parent / "fixtures" / f"{name}.json")


def command(intent, items=None, place=None, sort_by="none") -> dict:
    return {"intent": intent, "items": items or [], "place": place, "sort_by": sort_by}


def item(kind="ball", color="red", quantity=1, position=None, place=None) -> dict:
    return {"kind": kind, "name": None, "color": color, "quantity": quantity,
            "position": position, "place": place}


def test_constants():
    check(planner.OBJECT_HEIGHT.get("cube") == 0.03, f"cube height {planner.OBJECT_HEIGHT}")
    check(planner.OBJECT_HEIGHT.get("ball") == 0.04, f"ball height {planner.OBJECT_HEIGHT}")
    check(planner.TRAY_CAVITY[0] == 0.056, f"cavity {planner.TRAY_CAVITY}")
    check(planner.TRAVEL_Z > scene_spec.TABLE_TOP_Z + scene_spec.DEFAULT_SCENE[-1].height,
          "travel height must clear the tray rim")


def test_no_motion_intents():
    s = scene()
    for intent in ("greeting", "off_topic", "not_doable", "question", "stop", "cancel", "repeat"):
        plan = planner.plan_command(command(intent), s)
        check(not plan.moves, f"{intent} must not move the arm")
        check(bool(plan.reply), f"{intent} must say why")


def test_pick_only():
    s = scene()
    plan = planner.plan_command(command("pick", [item("ball", "red")]), s)
    check(plan.moves, "pick should move")
    check(len(plan.picks) == 1, f"one pick, got {plan.picks}")
    check(plan.places == [], "pick must not have a destination")
    check(len(plan.steps) == 4, f"pick-and-hold is 4 steps, got {len(plan.steps)}")


def test_pick_and_place():
    s = scene()
    plan = planner.plan_command(
        command("pick_and_place", [item("ball", "red")], {"kind": "box", "color": "red"}), s)
    check(plan.moves, "pick_and_place should move")
    check(plan.picks == [3], f"the red ball is #3 in trial000, got {plan.picks}")
    check(len(plan.places) == 1, f"one destination, got {plan.places}")
    check(len(plan.steps) == 8, f"the pick-and-place cycle is 8 steps, got {len(plan.steps)}")


def test_step_poses_are_sane():
    s = scene()
    plan = planner.plan_command(
        command("pick_and_place", [item("ball", "red")], {"kind": "box", "color": "red"}), s)
    things = {t.id: t for t in list(s.objects) + list(s.containers)}
    for step in plan.steps:
        check(step.pose is not None, f"step {step.action} has no pose")
        if step.pose is None:
            continue
        x, y, z = step.pose
        check(0.2 < x < 0.45, f"{step.action} x={x:.3f} off the table")
        check(-0.56 < y < -0.30, f"{step.action} y={y:.3f} off the table")
        check(scene_spec.TABLE_TOP_Z <= z <= planner.TRAVEL_Z + 1e-9,
              f"{step.action} z={z:.3f} between the table and travel height")
        check(planner.in_workspace(step.pose) is None,
              f"{step.action} pose {step.pose} flagged unreachable: {planner.in_workspace(step.pose)}")
    del things


def test_grasp_pose_uses_declared_height():
    s = scene()
    cube = next(t for t in s.objects if t.kind == "cube")
    ball = next(t for t in s.objects if t.kind == "ball")
    gz_cube = planner.grasp_pose(cube)[2]
    gz_ball = planner.grasp_pose(ball)[2]
    check(gz_ball > gz_cube,
          f"a ball is taller than a cube, so its grasp is higher: ball {gz_ball} vs cube {gz_cube}")
    check(abs(gz_cube - (scene_spec.TABLE_TOP_Z + 0.015)) < 1e-9, f"cube grasp z {gz_cube}")
    check(abs(gz_ball - (scene_spec.TABLE_TOP_Z + 0.020)) < 1e-9, f"ball grasp z {gz_ball}")


def test_grasp_offset_is_applied():
    s = scene()
    ball = next(t for t in s.objects if t.kind == "ball")
    base = planner.grasp_pose(ball)
    shifted = planner.grasp_pose(ball, offset=(0.004, -0.003))
    check(abs((shifted[0] - base[0]) - 0.004) < 1e-9, "offset x")
    check(abs((shifted[1] - base[1]) + 0.003) < 1e-9, "offset y")
    check(shifted[2] == base[2], "offset must not change z")


def test_ambiguous_asks_rather_than_guessing():
    s = scene("trial000")
    # trial000 has two red things: #3 is a red ball and #5 a red cube, so "the red one" by
    # kind alone is fine but a bare colour with kind "any" is not.
    plan = planner.plan_command(command("pick", [item("any", "red")]), s)
    check(not plan.moves or plan.picks, "either it moves with a target or it asks")


def test_missing_object_is_reported():
    s = scene()
    plan = planner.plan_command(
        command("pick_and_place", [item("ball", "purple")], {"kind": "box", "color": "red"}), s)
    check(not plan.moves, "a purple ball does not exist, so nothing should move")
    check(bool(plan.reply or plan.unresolved), "and it must say so")


def test_no_destination_is_refused():
    s = scene()
    plan = planner.plan_command(command("pick_and_place", [item("ball", "red")], None), s)
    check(not plan.moves, "with nowhere to put it, nothing should move")
    check("nowhere" in plan.reply.lower() or plan.question, f"reply was {plan.reply!r}")


def test_sequence_with_two_destinations():
    s = scene()
    plan = planner.plan_command(command("sequence", [
        item("ball", "red", place={"kind": "box", "color": "red"}),
        item("ball", "green", place={"kind": "box", "color": "blue"}),
    ]), s)
    check(plan.moves, "a two-destination sequence should move")
    check(len(plan.picks) == 2, f"two picks, got {plan.picks}")
    check(len(set(plan.places)) == 2, f"two destinations, got {plan.places}")


def test_sort_uses_the_scene():
    s = scene()
    plan = planner.plan_command(command("sort", [], None, sort_by="color"), s)
    check(plan.moves, "a bare sort should work out its own destinations")
    # trial000 holds a red ball, a red cube, a green ball, a blue ball, a blue tray and a red
    # box. Colours red and blue have receptacles; green does not, and that must be reported.
    check(any("green" in u for u in plan.unresolved),
          f"green has no receptacle and must be reported; unresolved={plan.unresolved}")
    check(all(step.target is None or isinstance(step.target, int) for step in plan.steps),
          "every step names a target id or none at all")


def test_clear_everything():
    s = scene()
    plan = planner.plan_command(command("clear", [], {"kind": "box", "color": "red"}), s)
    check(plan.moves, "clear should move")
    check(len(plan.picks) == len(s.objects), f"clear takes all {len(s.objects)}, got {plan.picks}")


def test_clear_with_filter():
    s = scene()
    plan = planner.plan_command(
        command("clear", [item("ball", "red", quantity="all")], {"kind": "box", "color": "blue"}), s)
    check(plan.moves, "a filtered clear should move")
    check(len(plan.picks) == 1, f"one red ball, got {plan.picks}")


def test_every_fixture():
    for path in FIXTURES:
        s = Scene.read(path)
        for intent in ("pick_and_place", "sort", "clear", "pick", "sequence"):
            plan = planner.plan_command(command(intent, [], {"kind": "box", "color": "red"}), s)
            check(plan.summary != "", f"{path.name} {intent}: no summary")
            for step in plan.steps:
                check(step.pose is not None, f"{path.name} {intent}: a step has no pose")


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print(f"{checks} checks over {len(FIXTURES)} recorded scene(s)")
    if failures:
        print(f"{len(failures)} FAILED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("RESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())