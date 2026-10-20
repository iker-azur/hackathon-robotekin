"""Tests for planner.py, against the scene JSONs demo 09 recorded. No GPU, no Isaac.

    python tests/test_planner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import planner  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402
from scene import Scene, Thing  # noqa: E402

FIXTURES = sorted((Path(__file__).resolve().parent / "fixtures").glob("*.json"))

failures: list[str] = []
checks = 0


def check(condition, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def put_trays_in_test_workspace(s: Scene) -> None:
    """Move recorded receptacles to where the arm can lower a part into them.

    The fixtures were recorded with trays at y -0.51..-0.52, 0.14-0.16 m from the base, nearer
    than ``scene_spec.MIN_PLACE_REACH_M``. These tests are about the planner's step structure, not
    about reach (that has its own tests in test_demo11.py), so keep each tray's x and bring it out
    to 0.19 m.
    """
    for tray in s.containers:
        if tray.pos_m is None:
            continue
        x = tray.pos_m[0]
        lateral = abs(x - scene_spec.ROBOT_XY[0])
        y = scene_spec.ROBOT_XY[1] + (0.19 ** 2 - min(lateral, 0.18) ** 2) ** 0.5
        if y > tray.pos_m[1]:
            tray.pos_m = [x, y, tray.pos_m[2]]
            tray.aim_xy = [x, y]


def scene(name: str = "trial000") -> Scene:
    s = Scene.read(Path(__file__).resolve().parent / "fixtures" / f"{name}.json")
    put_trays_in_test_workspace(s)
    return s


def put_red_ball_in_test_workspace(s: Scene) -> None:
    """Keep planner behavior tests out of the robot's measured near-base dead zone."""
    ball = next(t for t in s.objects if t.kind == "ball" and t.color == "red")
    ball.pos_m = [0.38, -0.435, ball.pos_m[2]]
    ball.aim_xy = [0.38, -0.435]


def command(intent, items=None, place=None, sort_by="none") -> dict:
    return {"intent": intent, "items": items or [], "place": place, "sort_by": sort_by}


def item(kind="ball", color="red", quantity=1, position=None, place=None) -> dict:
    return {"kind": kind, "name": None, "color": color, "quantity": quantity,
            "position": position, "place": place}


def test_constants():
    check(planner.OBJECT_HEIGHT.get("cube") == 0.03, f"cube height {planner.OBJECT_HEIGHT}")
    check(planner.OBJECT_HEIGHT.get("ball") == 0.03,
          f"ball height {planner.OBJECT_HEIGHT}: a 4 cm ball does not fit the 36.2 mm gripper")
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
    put_red_ball_in_test_workspace(s)
    plan = planner.plan_command(command("pick", [item("ball", "red")]), s)
    check(plan.moves, "pick should move")
    check(len(plan.picks) == 1, f"one pick, got {plan.picks}")
    check(plan.places == [], "pick must not have a destination")
    check(len(plan.steps) == 4, f"pick-and-hold is 4 steps, got {len(plan.steps)}")


def test_pick_and_place():
    s = scene()
    # The recorded red ball at (0.357, -0.451) is inside the newly measured SO-101 inner
    # dead zone. Put this planner-cycle test at the strict-physical validated central pose;
    # the dead-zone refusal is covered separately in test_demo11.py.
    ball = next(t for t in s.objects if t.kind == "ball" and t.color == "red")
    ball.pos_m = [0.38, -0.435, ball.pos_m[2]]
    ball.aim_xy = [0.38, -0.435]
    plan = planner.plan_command(
        command("pick_and_place", [item("ball", "red")], {"kind": "box", "color": "red"}), s)
    check(plan.moves, "pick_and_place should move")
    # Which id the red ball happens to be is a property of the recorded frame, not of the
    # planner: ids are assigned biggest-blob-first, so it moves whenever the layout does. Ask
    # the scene which id is the red ball instead of pinning a number that a re-recorded fixture
    # would invalidate for no reason.
    reds = [t.id for t in s.objects if t.color == "red" and t.kind == "ball"]
    check(plan.picks == reds, f"the red ball is {reds} in {s.source}, got {plan.picks}")
    check(len(plan.places) == 1, f"one destination, got {plan.places}")
    check(len(plan.steps) == 8, f"the pick-and-place cycle is 8 steps, got {len(plan.steps)}")
    ball = next(t for t in s.objects if t.kind == "ball" and t.color == "red")
    tray = next(t for t in s.containers if t.color == "red")
    drop = next(step.pose for step in plan.steps if step.action == "lower"
                and step.target == tray.id)
    centre = planner.place_pose(tray)
    compensation = real_world.jaw_sweep_approach_offset(ball.kind, ball.size_m)
    check(abs(drop[0] - centre[0] - compensation[0]) < 1e-9
          and abs(drop[1] - centre[1] - compensation[1]) < 1e-9,
          "ball release pose compensates the measured off-centre physical grasp")


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
        # The working area, read from the spec rather than written here: these bounds were
        # hardcoded to the old 16 cm-wide layout and went stale the moment it was resized.
        check(scene_spec.WORK_X[0] - 0.01 < x < scene_spec.WORK_X[1] + 0.01,
              f"{step.action} x={x:.3f} outside the working area {scene_spec.WORK_X}")
        check(scene_spec.WORK_Y[0] - 0.01 < y < scene_spec.WORK_Y[1] + 0.01,
              f"{step.action} y={y:.3f} outside the working area {scene_spec.WORK_Y}")
        step_high = planner.TRAVEL_Z
        if step.action in ("above", "lift") and step.target in things:
            step_high = max(step_high, planner.grasp_pose(things[step.target])[2]
                            + planner.GRASP_APPROACH_CLEARANCE_M)
        check(scene_spec.TABLE_TOP_Z <= z <= step_high + 1e-9,
              f"{step.action} z={z:.3f} within its safe approach/travel height")
        check(planner.in_workspace(step.pose) is None,
              f"{step.action} pose {step.pose} flagged unreachable: {planner.in_workspace(step.pose)}")
    del things


def test_grasp_pose_uses_declared_height():
    """The grasp height is the DECLARED height of the kind, read from the scene spec.

    Both kinds are 3 cm in this scene - the cubes always were, and the balls were changed
    from 4 cm because the gripper's fingers are 36.2 mm apart and cannot close on 40 mm
    (``tools/probe_gripper.py``). So this can no longer tell the two kinds apart by their
    size, and instead checks the thing that actually matters: that the z comes from
    ``planner.OBJECT_HEIGHT`` rather than from a constant, including for a kind that is not
    in the table at all.
    """
    s = scene()
    for kind in ("cube", "ball"):
        thing = next(t for t in s.objects if t.kind == kind)
        height = planner.OBJECT_HEIGHT.get(kind, planner.DEFAULT_HEIGHT)
        check(abs(planner.grasp_pose(thing)[2] - (scene_spec.TABLE_TOP_Z + height / 2.0)) < 1e-9,
              f"{kind} grasp z is the declared height + {height}")

    # A kind the spec does not know falls back to the cube's height rather than to None.
    unknown = next(t for t in s.objects if t.kind == "cube")
    unknown.kind = "dodecahedron"
    check(abs(planner.grasp_pose(unknown)[2]
              - (scene_spec.TABLE_TOP_Z + planner.DEFAULT_HEIGHT / 2.0)) < 1e-9,
          "an unknown kind uses DEFAULT_HEIGHT")
    unknown.kind = "cube"

    # And a taller declared kind is grasped higher, which is the property the old 4 cm ball
    # happened to demonstrate and the default scene can no longer show.
    taller = next(t for t in s.objects if t.kind == "ball")
    was = planner.OBJECT_HEIGHT.get("ball")
    planner.OBJECT_HEIGHT["ball"] = 0.08
    try:
        check(planner.grasp_pose(taller)[2] > planner.grasp_pose(unknown)[2],
              "a taller declared kind is grasped higher")
    finally:
        planner.OBJECT_HEIGHT["ball"] = was


def test_grasp_offset_is_applied():
    s = scene()
    ball = next(t for t in s.objects if t.kind == "ball")
    base = planner.grasp_pose(ball)
    shifted = planner.grasp_pose(ball, offset=(0.004, -0.003))
    check(abs((shifted[0] - base[0]) - 0.004) < 1e-9, "offset x")
    check(abs((shifted[1] - base[1]) + 0.003) < 1e-9, "offset y")
    check(shifted[2] == base[2], "offset must not change z")


def test_sim_assisted_cluster_offset_is_narrowly_scoped():
    triangle = Thing(id=1, kind="triangular_prism", color="red", centroid_px=[0, 0],
                     bbox_px=[0, 0, 10, 10], area_px=80, mean_hsv=[0, 0, 0],
                     confidence=.8, source="test", pos_m=[.344, -.444, .049],
                     size_m=[.020, .020, .016], aim_xy=[.345, -.445])
    neighbor = Thing(id=2, kind="ball", color="green", centroid_px=[0, 0],
                     bbox_px=[0, 0, 12, 12], area_px=100, mean_hsv=[0, 0, 0],
                     confidence=.8, source="test", pos_m=[.390, -.440, .056],
                     size_m=[.030, .030, .030], aim_xy=[.390, -.440])
    other = Thing(id=3, kind="box", color="red", centroid_px=[0, 0],
                  bbox_px=[0, 0, 20, 20], area_px=300, mean_hsv=[0, 0, 0],
                  confidence=.8, source="test", pos_m=[.485, -.505, .066],
                  aim_xy=[.485, -.505])
    peers = [triangle, neighbor, other]
    correction = planner.simulated_cluster_grasp_offset(
        triangle, peers, enabled=True, camera="top")
    check(correction == (-.013, .005),
          f"tested assisted 20 mm cluster gets the measured correction: {correction}")
    custom = planner.simulated_cluster_grasp_offset(
        triangle, peers, offset=(.002, -.001), enabled=True, camera="top")
    check(custom == (-.011, .004), "scene correction composes with the user's offset")
    check(planner.simulated_cluster_grasp_offset(
        triangle, peers, enabled=False, camera="top") == (0.0, 0.0),
        "strict physical execution never receives the simulated correction")
    for camera, supported in {
            "top": {"triangular_prism"},
            "front": {"cube", "cylinder", "triangular_prism", "hexagonal_prism"},
            "side": {"cube", "hexagonal_prism"},
    }.items():
        for kind in {"cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"}:
            triangle.kind = kind
            expected = (-.013, .005) if kind in supported else (0.0, 0.0)
            check(planner.simulated_cluster_grasp_offset(
                triangle, peers, enabled=True, camera=camera) == expected,
                f"{camera}/{kind} uses only its empirically tested assisted profile")
    triangle.kind = "triangular_prism"
    check(planner.simulated_cluster_grasp_offset(
        triangle, [triangle, other], enabled=True, camera="top") == (0.0, 0.0),
        "isolated triangles do not receive the cluster correction")
    for kind in ("cube", "hexagonal_prism"):
        triangle.kind = kind
        check(planner.simulated_cluster_grasp_offset(
            triangle, [triangle, other], enabled=True, camera="side") == (-.013, .005),
            f"measured side-view {kind} single-object miss receives its E2E-checked correction")
    triangle.kind = "ball"
    check(planner.simulated_cluster_grasp_offset(
        triangle, [triangle, other], enabled=True, camera="side") == (0.0, 0.0),
        "side-view ball single-object baseline remains unchanged")
    triangle.kind = "triangular_prism"
    triangle.size_m = [.030, .030, .024]
    check(planner.simulated_cluster_grasp_offset(
        triangle, peers, enabled=True, camera="top") == (0.0, 0.0),
        "other triangle sizes do not inherit the 20 mm assisted correction")


def test_jaw_sweep_offsets_are_object_specific_and_measured():
    s = scene()
    ball = next(t for t in s.objects if t.kind == "ball")
    base_xy = ball.aim_xy or scene_spec.aim_point(ball.pos_m)
    pose = planner.grasp_pose(ball)
    check(abs(pose[0] - base_xy[0]) < 1e-9
          and abs(pose[1] - base_xy[1] - .008) < 1e-9,
          "ball grasp aims 8 mm along the measured jaw-closing sweep")
    check(real_world.jaw_sweep_approach_offset("cylinder", (0.03, 0.03, 0.024))
          == (0.0, .008),
          "cylinders use the validated sweep compensation")
    check(real_world.jaw_sweep_approach_offset("ball", (0.02, 0.02, 0.02)) == (0.0, 0.0)
          and real_world.jaw_sweep_approach_offset("cylinder", (0.02, 0.02, 0.016))
          == (0.0, 0.0),
          "20 mm round objects do not inherit the unvalidated 30 mm grasp profile")
    check(real_world.jaw_sweep_approach_offset("triangular_prism") == (0.0, 0.0)
          and real_world.jaw_sweep_approach_offset(
              "hexagonal_prism", (.020, .020, .018)) == (0.0, 0.0)
          and real_world.jaw_sweep_approach_offset(
              "hexagonal_prism", (.030, .030, .024)) == (0.0, 0.0),
          "unvalidated prism sizes remain at the calibrated centre")


def test_grasp_lift_clearance_matches_strict_round_object_envelope():
    check(abs(planner.GRASP_APPROACH_CLEARANCE_M - .09) < 1e-9,
          "grasp lift stays at the 90 mm strict ball/cylinder carry height")


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
    put_red_ball_in_test_workspace(s)
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
    put_red_ball_in_test_workspace(s)
    plan = planner.plan_command(
        command("clear", [item("ball", "red", quantity="all")], {"kind": "box", "color": "blue"}), s)
    check(plan.moves, "a filtered clear should move")
    check(len(plan.picks) == 1, f"one red ball, got {plan.picks}")


def test_every_fixture():
    for path in FIXTURES:
        s = Scene.read(path)
        put_trays_in_test_workspace(s)
        for intent in ("pick_and_place", "sort", "clear", "pick", "sequence"):
            plan = planner.plan_command(command(intent, [], {"kind": "box", "color": "red"}), s)
            check(plan.summary != "", f"{path.name} {intent}: no summary")
            for step in plan.steps:
                check(step.pose is not None, f"{path.name} {intent}: a step has no pose")



def test_missing_destination_color_never_chooses_another_box():
    import ground
    s = scene()
    s.containers = [c for c in s.containers if c.color == "red"]
    result = ground.ground_place(s, {"kind": "box", "color": "green"})
    check(result.status != "ok" and result.place is None,
          "a missing green box must never select the red box")



def test_quantity_is_not_silently_reduced_and_clear_keeps_its_filter():
    import ground
    s=scene()
    put_red_ball_in_test_workspace(s)
    result=ground.ground(s,item("ball","red",quantity=2))
    check(not result.ok, "asking for two red balls cannot select the only one")
    result=ground.ground(s,item("ball",None,quantity="all"))
    check(result.ok and len(result.targets)==len([o for o in s.objects if o.kind=="ball"]), "all means all matching balls")
    plan=planner.plan_command(command("clear",[item("cube","purple")],{"kind":"box","color":"red"}),s)
    check(not plan.moves and bool(plan.question), "missing clear filter must not clear everything")


def test_named_container_and_declared_geometry():
    import ground
    s=scene(); tray=s.containers[0];tray.entity_id="bin_a";tray.size_m=[.07,.07,.03]
    result=ground.ground_place(s,{"entity_id":"bin_a","kind":"box","color":tray.color})
    check(result.ok and result.place is tray, "named container resolves by identity")
    check(abs(planner.place_pose(tray)[2]-(scene_spec.TABLE_TOP_Z+.03+planner.RELEASE_CLEARANCE))<1e-9,
          "release follows the configured container height")


def test_sort_filters_quantities_and_ambiguous_boxes():
    import copy
    s = scene()
    selected = command("sort", sort_by="color")
    selected["sort_filter"] = [item("cube", "red", "all")]
    plan = planner.plan_command(selected, s)
    check(bool(plan.tasks), "a supported sort filter produces a task")
    check(all(t.kind == "cube" and t.color == "red" for t in plan.tasks), "sort does not add unrequested balls or colours")
    selected["sort_filter"] = [item("ball", "red", 2)]
    plan = planner.plan_command(selected, s)
    check(bool(plan.question) and not plan.moves, "sort refuses a count that cannot be satisfied")
    selected["sort_filter"] = [item("ball", "purple", "all")]
    plan = planner.plan_command(selected, s)
    check(bool(plan.question) and not plan.moves, "sort does not silently discard an unknown colour filter")
    duplicate = copy.deepcopy(next(t for t in s.containers if t.color == "red")); duplicate.id=200
    s.containers.append(duplicate)
    plan = planner.plan_command(command("sort",sort_by="color"),s)
    check(bool(plan.unresolved), "same-colour containers require an explicit destination")


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
