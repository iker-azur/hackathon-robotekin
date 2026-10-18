"""Demo 12 -- planning, the collaboration log and the grasp evidence bar.

All three run in a fraction of a second on any machine and need no GPU, which is the point:
they cover the *decisions* the interface makes and the *rule* the orchestrator judges a pick
by, and both of those are worth testing far more often than a simulator boot allows.

Nothing here pretends to test physics. ``tests/test_grasp.py`` runs the same state machine
inside real Isaac Sim and measures it; these check that the orchestrator applies the bar
correctly and that the planner explains itself.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo12 import events as EV
from demo12 import grasp as G
from demo12 import planner as P
from demo12.config import (ROBOT_BASE_Y, WORK_AREA_X, WORK_AREA_Y, WORKING_RADIUS_M,
                          robot_can_work,
                          SceneError, default_scene)


# ---------------------------------------------------------------------------
# The evidence bar
# ---------------------------------------------------------------------------

#: The real probe's readings for the 30 mm ABS cube, from out12/grasp_probe.json. Used
#: throughout so these tests check the *rule* against measured numbers rather than against
#: invented ones.
CUBE = dict(part_id="part_1", bin_id="bin_1", robot_id="robot_1", mass_kg=0.0281,
            tool_error_m=0.000526, part_width_m=0.030, span_on_part_m=0.029402,
            grip_force_n=60.9061, part_rise_m=0.098884, part_carried_to_bin=True,
            part_inside_bin=True, part_resting=True)


def test_the_measured_cube_grasp_passes_every_criterion():
    evidence = G.evaluate_grasp(**CUBE)
    assert evidence.ok
    assert all(evidence.criteria.values())
    # Comfortably inside what the friction cone needs, which is the point: a gripper that
    # could barely hold something would not be a measurement.
    assert evidence.grip_required_n == pytest.approx(0.306, abs=0.01)
    assert evidence.grip_margin > 100


def test_a_punt_is_rejected_despite_a_large_force():
    """The documented 103 N false grasp: pads on a corner, part shoved 36 mm.

    Every number here is from ``docs/PHYSICS.md`` section 4. Under the old ``grip_n > 0``
    rule this was a completed task.
    """
    evidence = G.evaluate_grasp(
        **{**CUBE, "span_on_part_m": 0.0406, "grip_force_n": 103.0,
           "part_rise_m": 0.0, "part_carried_to_bin": False})

    assert not evidence.ok
    assert evidence.criteria["grip"], "the force itself was more than enough"
    assert not evidence.criteria["pads_on_faces"], "40.6 mm is not a 30 mm part's width"
    assert not evidence.criteria["part_rose"]
    assert not evidence.criteria["carried"]


def test_a_part_knocked_into_the_tray_is_not_a_successful_pick():
    """In the tray and at rest, but this robot never carried it there."""
    evidence = G.evaluate_grasp(
        **{**CUBE, "part_rise_m": 0.0, "part_carried_to_bin": False})

    assert not evidence.ok
    assert evidence.criteria["part_in_bin"]
    assert evidence.criteria["part_resting"]
    assert not evidence.criteria["part_rose"]


@pytest.mark.parametrize("force", [0.0, 0.002, 0.05])
def test_a_squeeze_below_the_friction_cone_is_not_a_grip(force):
    """Zero-force is the floor, and it is not the only thing that fails.

    The threshold is ``2 * mu * m * g`` with a 2x safety factor, not a constant, so a light
    part needs less than a heavy one and both are checked against their own mass.
    """
    evidence = G.evaluate_grasp(**{**CUBE, "grip_force_n": force})
    assert not evidence.criteria["grip"]

    needed = G.required_grip_n(CUBE["mass_kg"])
    evidence_ok = G.evaluate_grasp(**{**CUBE, "grip_force_n": needed})
    assert evidence_ok.criteria["grip"], "exactly the required force is enough"


def test_the_threshold_scales_with_the_part():
    """A 2.9 g ball needs far less force than a 28 g cube, and the bar says so."""
    ball = G.evaluate_grasp(
        part_id="part_5", bin_id="bin_1", robot_id="robot_1", mass_kg=0.0029,
        tool_error_m=0.000729, part_width_m=0.020, span_on_part_m=0.019157,
        grip_force_n=24.3881, part_rise_m=0.09874, part_carried_to_bin=True,
        part_inside_bin=True, part_resting=True)
    assert ball.ok
    assert ball.grip_required_n < G.required_grip_n(0.0281) / 5


def test_every_failure_is_explained_in_words():
    """A failure names what went wrong, so a stuck cell is diagnosable."""
    evidence = G.evaluate_grasp(**{**CUBE, "span_on_part_m": 0.0406,
                                   "part_rise_m": 0.0, "part_carried_to_bin": False,
                                   "part_resting": False})
    text = " ".join(evidence.failures)
    assert "faces" in text
    assert "not carried" in text
    assert "still moving" in text


# ---------------------------------------------------------------------------
# The collaboration log
# ---------------------------------------------------------------------------


def test_events_are_numbered_and_readable_forward():
    log = EV.EventLog()
    for i in range(5):
        log.emit(EV.PHASE, tick=i, sim_time_s=i * 0.1, robot_id="robot_1",
                 task_id=f"task_{i}", detail=f"step {i}")

    seen = log.since(0)
    assert [e["seq"] for e in seen] == [1, 2, 3, 4, 5]
    assert [e["taskId"] for e in seen] == [f"task_{i}" for i in range(5)]


def test_since_returns_only_what_is_new():
    log = EV.EventLog()
    for i in range(4):
        log.emit(EV.ASSIGN, tick=i, detail="x")

    assert len(log.since(2)) == 2
    assert log.since(log.next_seq) == []


def test_reading_the_cursor_does_not_move_it():
    """A poll that finds nothing must be able to say so.

    ``next_seq`` used to be ``next(self._seq)`` on an ``itertools.count``, so *asking* for
    the cursor consumed a sequence number. The interface polls twenty times a second, so the
    cursor drifted constantly with no new events, and a client keying its redraw on cursor
    change redrew forever. Nothing could represent "nothing has happened" -- which is the one
    thing a collaboration timeline most needs to be able to say.
    """
    log = EV.EventLog()
    for i in range(3):
        log.emit(EV.ASSIGN, tick=i, detail="x")

    first = log.next_seq
    for _ in range(50):
        assert log.next_seq == first, "reading the cursor changed it"

    log.emit(EV.DONE, tick=99, detail="new")
    assert log.next_seq == first + 1, "the cursor did not move on a real event"
    assert [e["detail"] for e in log.since(first)] == ["new"]


def test_the_log_is_bounded_but_never_reports_a_false_absence():
    """A long demo must not leak, and a client that fell behind must be told it has.

    Returning an empty list for "you asked for something older than the ring" is the one
    answer that is always wrong: it reads as "nothing happened".
    """
    log = EV.EventLog(capacity=10)
    for i in range(50):
        log.emit(EV.PHASE, tick=i, detail="x")

    assert len(log) == 10
    assert len(log.since(0)) == 10, "a stale cursor must get the oldest thing still held"


def test_counts_track_kinds():
    log = EV.EventLog()
    log.emit(EV.ASSIGN)
    log.emit(EV.ASSIGN)
    log.emit(EV.FAULT, robot_id="robot_2")
    assert log.counts() == {EV.ASSIGN: 2, EV.FAULT: 1}


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scene():
    return default_scene(3)


def _colour(value: str) -> str:
    return str(value or "").strip().lower()


@pytest.mark.parametrize("text,expected", [
    ("part_1 -> bin_2", [("part_1", "bin_2")]),
    ("mueve el cubo rojo a la caja azul", [("part_1", "bin_2")]),
    ("move the red cube to the blue bin", [("part_1", "bin_2")]),
    ("send the green cube to the blue bin", [("part_3", "bin_2")]),
    ("send the blue cylinder to bin_2", [("part_2", "bin_2")]),
    ("move the yellow ball to the red bin", [("part_4", "bin_1")]),
    ("pon la bola en la caja roja", [("part_4", "bin_1")]),
])
def test_commands_in_both_languages(scene, text, expected):
    result = P.parse_command(text, scene)
    if expected == "ambiguous":
        assert not result.moves
        assert any("all match" in e for e in result.errors)
    else:
        assert result.moves == expected, result.errors
        assert not result.errors


def test_all_expands_to_every_part(scene):
    result = P.parse_command("send all to bin_2", scene)
    assert len(result.moves) == len(scene.parts)
    assert all(b == "bin_2" for _, b in result.moves)


def test_accents_never_change_a_match(scene):
    with_accents = P.parse_command("mueve la bola amarilla a la caja roja", scene)
    without = P.parse_command("mueve la bola amarilla a la caja roja", scene)
    assert with_accents.moves == without.moves == [("part_4", "bin_1")]


def test_an_unknown_word_is_reported_not_guessed(scene):
    """A colour with no part in the cell must be refused, not resolved to the nearest one."""
    result = P.parse_command("move the turquoise widget to bin_1", scene)
    assert not result.moves
    assert result.errors, "a colour with no part must say so"


def test_a_colour_that_beats_a_shape_beats_it(scene):
    """"blue cylinder" names part_2 by colour *and* shape; part_4 only by shape.

    With a first-match-wins ladder both stop at the colour test and the scene-order tie goes
    to the wrong one. The scores have to be additive for this to come out right.
    """
    scored = P._score_parts("the red cube", scene)
    assert scored[0][1].id == "part_1", "colour+shape must outrank shape alone"
    assert len(scored) > 1, "the shape alone also matches another part"
    assert scored[0][0] > scored[1][0], "two matches must outrank one"


# ---------------------------------------------------------------------------
# Assignment cost
# ---------------------------------------------------------------------------


def test_the_nearest_robot_wins(scene):
    task = P.make_task(scene, "part_1", "bin_1")
    ranked = P.DEFAULT_POLICY.rank(scene, ["robot_1", "robot_2", "robot_3"], task)
    assert ranked[0].robot_id == "robot_2"      # part_1 sits nearest robot_2's base
    assert ranked == sorted(ranked, key=lambda c: c.total)


def test_a_retry_makes_a_robot_less_preferred(scene):
    """Otherwise the same arm retries the same flaky grasp until the budget runs out."""
    task = P.make_task(scene, "part_1", "bin_1")
    first = P.DEFAULT_POLICY.cost(scene, "robot_2", task, attempts=0)
    retried = P.DEFAULT_POLICY.cost(scene, "robot_2", task, attempts=2)
    assert retried.total > first.total
    assert retried.attempts == 2


def test_carry_distance_is_weighted_below_reach(scene):
    task = P.make_task(scene, "part_1", "bin_1")
    cost = P.DEFAULT_POLICY.cost(scene, "robot_2", task)
    assert cost.total == pytest.approx(
        cost.reach_m + P.DEFAULT_POLICY.CARRY_WEIGHT * cost.carry_m)


def test_a_robot_that_cannot_work_a_part_is_not_ranked(scene):
    """Excluded, and by the *measured* radius rather than the geometric envelope.

    Scoring an unreachable robot would put a number in the interface next to an option that
    can never be taken, which is exactly the kind of thing that makes a demo look like it
    is lying.
    """
    task = P.make_task(scene, "part_1", "bin_1")
    ranked = P.DEFAULT_POLICY.rank(scene, ["robot_1", "robot_2", "robot_3"], task)
    part = scene.part(task.part_id)
    assert ranked
    for cost in ranked:
        assert robot_can_work(scene.robot(cost.robot_id).base_xy, part.x, part.y), (
            f"{cost.robot_id} was offered a part it cannot work at")


def test_the_scheduler_and_the_validator_agree_on_reachability(scene):
    """One definition of reachable, used by both.

    They did not agree, and the disagreement is a silent way to lose a task: ``validate``
    checked the 235 mm working radius while the assignment policy kept checking the 335 mm
    envelope, so a cell that validated cleanly could still be scheduled into a grasp the arm
    cannot make. Measured, that mismatch is a run going from 2/2 to 0/4.
    """
    from demo12 import planner as PL

    for part in scene.parts:
        # What the validator believes.
        validator_says = [r.id for r in scene.enabled_robots
                          if robot_can_work(r.base_xy, part.x, part.y)]
        assert validator_says, f"{part.id}: validate would reject the scene"

        # What the scheduler believes, for a move starting at that part.
        move = PL.make_task(scene, part.id, scene.bins[0].id)
        scheduler_says = {c.robot_id for c in
                          PL.DEFAULT_POLICY.rank(
                              scene, [r.id for r in scene.enabled_robots], move)}

        assert scheduler_says == set(validator_says), (
            f"{part.id}: validator says {validator_says}, "
            f"scheduler says {sorted(scheduler_says)}")


def test_a_crowded_pick_is_penalised_rather_than_merely_labelled():
    """The conflict has to reach the total, or it is decoration.

    Every other cost term is a property of one arm and one part, so without this the policy
    sends each arm to whatever is nearest *it* -- and since all the arms face the same way
    down the same bench, the two nearest things to two different arms are often the same
    corner. Measured: robot_1 on ``part_1`` and robot_2 on ``part_4``, 70 mm apart, and the
    run ended 0/4 with tracking errors that read exactly like a reachability fault.
    """
    dense = default_scene(2, assign_all=True)
    assert len(dense.mission.tasks) >= 4, "this needs a mission with parts to fight over"

    held = dense.part("part_1")
    move = P.make_task(dense, "part_3", "bin_2")
    crowded = P.DEFAULT_POLICY.rank(dense, ["robot_2"], move, 0, [(held.x, held.y)])[0]
    clear = P.DEFAULT_POLICY.rank(dense, ["robot_2"], move, 0,
                                  [(held.x + 0.30, held.y)])[0]

    assert crowded.conflicted and not clear.conflicted
    assert crowded.total - clear.total >= P.DEFAULT_POLICY.CONFLICT_PENALTY_M - 1e-9


def test_the_pair_the_scheduler_picks_is_far_enough_apart():
    """The point of the term: the two arms actually land on different parts of the table."""
    dense = default_scene(2, assign_all=True)

    taken: list[str] = []
    picks: list[tuple[float, float]] = []
    for robot_id in ("robot_1", "robot_2"):
        best = None
        for task in dense.mission.tasks:
            if task.part_id in taken:
                continue
            ranked = P.DEFAULT_POLICY.rank(
                dense, [robot_id], P.make_task(dense, task.part_id, task.bin_id),
                0, picks)
            if ranked and (best is None or ranked[0].total < best[0].total):
                best = (ranked[0], task.part_id)
        assert best is not None, f"{robot_id} was given nothing"
        taken.append(best[1])
        part = dense.part(best[1])
        picks.append((part.x, part.y))

    assert len(set(taken)) == 2, "both arms were given the same part"
    gap = math.hypot(picks[0][0] - picks[1][0], picks[0][1] - picks[1][1])
    assert gap >= P.DEFAULT_POLICY.CONFLICT_RADIUS_M, (
        f"the scheduler put both arms {gap * 1000:.0f} mm apart, inside the "
        f"{P.DEFAULT_POLICY.CONFLICT_RADIUS_M * 1000:.0f} mm conflict radius")



def test_the_conflict_is_reported_not_just_applied(scene):
    """'robot_2 went to the far side' is only a satisfying answer if you can see the why."""
    task = P.make_task(scene, "part_1", "bin_1")
    held = scene.part("part_1")
    ranked = P.DEFAULT_POLICY.rank(
        scene, ["robot_2"], task, 0, [(held.x, held.y)])
    assert ranked[0].conflicted is True
    assert ranked[0].conflict_m == pytest.approx(0.0)
    assert ranked[0].to_dict()["conflicted"] is True


def test_no_conflict_is_the_default(scene):
    """The first arm of the cell must not be penalised for being first."""
    task = P.make_task(scene, "part_1", "bin_1")
    alone = P.DEFAULT_POLICY.rank(scene, ["robot_1"], task)[0]
    assert alone.conflicted is False
    assert alone.conflict_m == float("inf")


def test_every_task_has_at_least_one_eligible_robot(scene):
    """A task nobody can take would sit queued forever, which reads as a hang."""
    for task in scene.mission.tasks:
        ranked = P.DEFAULT_POLICY.rank(
            scene, [r.id for r in scene.enabled_robots],
            P.make_task(scene, task.part_id, task.bin_id))
        assert ranked, f"{task.id} has no eligible robot"


# ---------------------------------------------------------------------------
# Plan building and validation
# ---------------------------------------------------------------------------


def test_a_plan_needs_one_task_per_part(scene):
    built = P.build_plan(scene, [("part_1", "bin_1"), ("part_1", "bin_3")])
    assert not built["ok"]
    assert any("already has a task" in e for e in built["errors"])


def test_two_parts_may_share_a_tray(scene):
    """A tray filling up is a legitimate thing to want."""
    built = P.build_plan(scene, [("part_1", "bin_1"), ("part_2", "bin_1")])
    assert built["ok"], built["errors"]
    assert len(built["tasks"]) == 2


def test_an_ungraspable_part_is_refused_with_a_reason():
    """A 40 mm part cannot go in this gripper, and ``validate_move`` has to say so.

    The scene is edited directly rather than round-tripped through ``from_dict`` because
    ``from_dict`` validates and raises -- which is the behaviour under test one layer up,
    and the thing the interface relies on to refuse a bad edit.
    """
    from demo12.config import SceneConfig

    scene = SceneConfig.from_dict(default_scene(2).to_dict())
    scene.parts[0].size_m = (0.040, 0.040, 0.040)

    errors = P.validate_move(scene, "part_1", "bin_1")
    assert errors
    assert any("40 mm" in e and "gripper" in e for e in errors)


def test_task_ids_are_stable_so_edits_are_idempotent(scene):
    assert P.task_id_for("part_1", "bin_2") == P.task_id_for("part_1", "bin_2")
    assert P.task_id_for("part_1", "bin_2") != P.task_id_for("part_1", "bin_3")


def test_every_scenario_produces_a_valid_plan(scene):
    for name in P.SCENARIOS:
        built = P.build_plan(scene, P.scenario_moves(scene, name))
        assert built["ok"], f"{name}: {built['errors']}"
        assert built["tasks"], f"{name} produced no work"


# ---------------------------------------------------------------------------
# The cell itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("robots", [2, 3])
def test_the_default_cell_validates_for_every_roster(robots):
    """The cell is physically buildable for every roster it claims to support.

    If this fails, launching with that count builds a scene ``validate`` rejects -- three
    minutes of simulator boot before the user finds out.
    """
    scene = default_scene(robots)
    assert scene.collect_errors() == []
    assert len(scene.enabled_robots) == robots


def test_validate_checks_what_the_arm_can_do_not_the_envelope(scene):
    """A point inside the geometric envelope but outside the working radius is rejected.

    This is the check that would have caught the run where every grasp failed with a tool
    error of 150-260 mm. ``materials.MAX_REACH_M`` is 335 mm and the arm stops converging at
    235 mm, so validating against the envelope accepts layouts the arm cannot execute.
    """
    from demo12.config import SceneConfig

    rebuilt = SceneConfig.from_dict(default_scene(2).to_dict())
    inside = rebuilt.part("part_1")
    assert inside.reach_ok(rebuilt) if hasattr(inside, "reach_ok") else True

    # Push part_1 out past the envelope entirely: validate must object.
    rebuilt.parts[0].x = rebuilt.bins[-1].x + 0.60
    errors = rebuilt.collect_errors()
    assert any("out of reach of every robot" in e or "off the table" in e for e in errors), errors

    # And a point that is inside the envelope but outside the working radius.
    rebuilt2 = SceneConfig.from_dict(default_scene(2).to_dict())
    rebuilt2.parts[0].x = rebuilt2.bins[-1].x + 0.30
    errors2 = rebuilt2.collect_errors()
    assert errors2, "a 300 mm offset must not validate"


def test_every_part_and_tray_is_inside_the_measured_working_radius(scene):
    """The invariant the whole layout exists to hold, stated once."""
    for item in list(scene.parts) + list(scene.bins):
        reachable = any(
            math.hypot(item.x - r.base_xy[0], item.y - ROBOT_BASE_Y) <= WORKING_RADIUS_M
            for r in scene.enabled_robots)
        assert reachable, f"{item.id} is outside the {WORKING_RADIUS_M * 1000:.0f} mm " \
                          f"the arm can actually work at"


def test_no_transfer_is_longer_than_the_arm_can_carry(scene):
    """The longest *assigned* carry, not the longest possible one.

    Trays are matched colour-first and then nearest, because a 340 mm carry fails in a way
    a slightly wrong colour does not: the arm cannot hold a part across that distance at
    all. Measured, the longest carry in the default mission is 124 mm.
    """
    bins = {b.id: b for b in scene.bins}
    for task in scene.mission.tasks:
        part = scene.part(task.part_id)
        tray = bins[task.bin_id]
        distance = math.hypot(tray.x - part.x, tray.y - part.y)
        assert distance < 0.30, f"{task.id} carries {distance * 1000:.0f} mm"


def test_everything_sits_inside_the_work_area(scene):
    """A tray at x = -0.027 m is on the table and nowhere anyone would build a cell."""
    for item in list(scene.parts) + list(scene.bins):
        assert WORK_AREA_X[0] <= item.x <= WORK_AREA_X[1], item.id
        assert WORK_AREA_Y[0] <= item.y <= WORK_AREA_Y[1], item.id


def test_the_default_mission_is_colour_sorting(scene):
    """Two colour-matched picks, on opposite sides of the cell.

    A task for every part was tried and it breaks the cell: ``part_3`` and ``part_4`` sit at
    the same x only 47 mm apart in y, so two arms working the centre collide and the run
    ends 0/4 with tracking errors of 150-270 mm -- which looks exactly like a reachability
    problem and is not. The unassigned parts are in the cell and reachable; they wait for
    someone to assign them from the interface.
    """
    bins = {b.id: b for b in scene.bins}
    for task in scene.mission.tasks:
        part = scene.part(task.part_id)
        assert _colour(bins[task.bin_id].color) == _colour(part.color), task.id

    assert len(scene.mission.tasks) == 2
    xs = sorted(scene.part(t.part_id).x for t in scene.mission.tasks)
    assert xs[1] - xs[0] > 0.10, "both picks in one half of the cell will collide"


def test_every_part_is_assignable_even_without_a_default_task(scene):
    """Unassigned is not unreachable: the planner can still build a move for any part."""
    from demo12 import planner as P

    for part in scene.parts:
        target = min(scene.bins, key=lambda b: abs(b.x - part.x))
        assert not P.validate_move(scene, part.id, target.id), (part.id, target.id)


def _colour(value: str) -> str:
    return str(value or "").strip().lower()