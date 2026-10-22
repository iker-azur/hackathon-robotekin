"""Demo 12 -- replacing the plan while the arms are working.

Every one of these exists because of a bug found by running the live server, not by reading
the code: submitting a plan mid-pick froze the entire cell. `controlTicks` stuck at 350,
`simTimeS` stuck at 16.6, no error in `lastError`, no exception, and the two arms mid-pick
frozen where they stood.

Two defects, and both are required for the freeze:

1. `set_tasks` appended the in-flight task ids to `released`, emitted "plan replaced while in
   flight; requeued", and reported "2 requeued" -- then built the new queue from the done
   tasks and the submitted ones only. The task was announced as requeued and dropped.
2. The arm was never released. It kept `state="working"` and a `task_id` that no longer
   resolved, so `step` skipped the idle branch, found no runner, and `continue`d -- and
   `control_ticks`, which only advances on the path it never reached, stopped with it.

A freeze is the worst shape a bug can take: the process is alive, nothing is logged, and the
only symptom is that time stops.
"""

from __future__ import annotations

import pytest

from demo12.config import MoveTask, default_scene
from demo12.orchestrator import MultiRobotOrchestrator
from demo12.tests.test_orchestrator import FakeCell

ROBOT_COUNT = 2


def _working_cell(ticks: int = 120) -> tuple[MultiRobotOrchestrator, FakeCell]:
    """An orchestrator with both arms genuinely mid-pick, not merely idle."""
    scene = default_scene(ROBOT_COUNT)
    cell = FakeCell(scene, {})
    orch = MultiRobotOrchestrator(cell, scene)
    cell.bind(orch)
    orch.set_tasks(scene.mission.tasks)

    for _ in range(ticks):
        orch.step()

    busy = [r for r in orch.robots.values() if r.state == "working"]
    assert busy, "no arm picked anything up; the premise of this test is not established"
    return orch, cell


def test_replacing_the_plan_mid_pick_does_not_freeze_the_cell():
    """The bug, as a test. Before the fix this froze at the tick the plan was applied."""
    orch, _ = _working_cell()

    before = orch.control_ticks
    result = orch.set_tasks([
        MoveTask(id="move_part_3_to_bin_2", part_id="part_3", bin_id="bin_2", label="3"),
        MoveTask(id="move_part_4_to_bin_2", part_id="part_4", bin_id="bin_2", label="4"),
    ])
    assert result["ok"] is True

    for _ in range(400):
        orch.step()

    assert orch.control_ticks > before, (
        "the clock stopped when the plan was replaced -- the cell is frozen, not busy")
    assert orch.work_outstanding(), "there is queued work and the cell says there is none"


def test_a_task_released_in_flight_is_requeued_rather_than_dropped():
    """It was announced as requeued, and the plan event said "2 requeued"."""
    orch, _ = _working_cell()

    in_flight = [t.id for t in orch.tasks.values() if t.state != "done"]
    assert in_flight, "nothing was in flight"

    orch.set_tasks([MoveTask(id="move_part_3_to_bin_2", part_id="part_3",
                            bin_id="bin_2", label="3")])

    for task_id in in_flight:
        assert task_id in orch.tasks, (
            f"{task_id} was in flight and was announced as requeued, but it is not in the "
            f"queue; only {sorted(orch.tasks)} are")
        assert orch.tasks[task_id].state == "queued"
        assert orch.tasks[task_id].assigned_robot is None


def test_the_arm_holding_a_released_task_is_put_back_to_idle():
    """Otherwise it keeps a task id that resolves to nothing and never gets new work."""
    orch, _ = _working_cell()

    victim = next(r for r in orch.robots.values() if r.task_id)
    held = victim.task_id

    orch.set_tasks([MoveTask(id="move_part_3_to_bin_2", part_id="part_3",
                            bin_id="bin_2", label="3")])

    assert victim.task_id is None, f"{victim.id} still points at {held}"
    assert victim.state == "idle", f"{victim.id} is {victim.state}, not idle"

    for _ in range(200):
        orch.step()
    assert orch.tasks[held].assigned_robot == victim.id, (
        f"{held} was requeued but {victim.id} never picked it back up")


def test_a_robot_whose_task_vanished_cannot_stall_the_clock():
    """`step` must not be silent when it cannot find a task.

    The recovery path itself is defensive. It should be unreachable now that `set_tasks`
    releases the arm, and a test that reaches it deliberately is the only way to know it
    works rather than merely existing.
    """
    orch, _ = _working_cell()

    robot = next(r for r in orch.robots.values() if r.task_id)
    # Remove the task from under the arm, which is what the old `set_tasks` did.
    orch.tasks.pop(robot.task_id)

    before = orch.control_ticks
    for _ in range(20):
        orch.step()

    assert orch.control_ticks > before, "the clock stopped on a missing task"
    assert robot.state == "idle"
    assert robot.task_id is None


def test_submitting_a_plan_reports_what_it_did_to_the_old_one():
    """The interface shows this, and it is what made the bug so hard to see."""
    orch, _ = _working_cell()

    result = orch.set_tasks([MoveTask(id="move_part_3_to_bin_2", part_id="part_3",
                                      bin_id="bin_2", label="3")])
    assert result["tasksSubmitted"] == ["move_part_3_to_bin_2"]
    assert result["released"], "two arms were mid-pick; something should be reported"
    for task_id in result["released"]:
        assert task_id in orch.tasks, (
            f"released {task_id} is not in the queue, so 'released' is a misleading word")


def test_a_plan_applied_to_an_idle_cell_still_works():
    """The ordinary case, which must not have been broken by the fix."""
    scene = default_scene(ROBOT_COUNT)
    cell = FakeCell(scene, {})
    orch = MultiRobotOrchestrator(cell, scene)
    cell.bind(orch)
    orch.set_tasks([])

    orch.set_tasks([MoveTask(id="move_part_1_to_bin_1", part_id="part_1",
                            bin_id="bin_1", label="1")])
    for _ in range(3_000):
        if not orch.work_outstanding():
            break
        orch.step()

    assert orch.all_tasks_completed(), [t.to_dict() for t in orch.tasks.values()]
    assert "move_part_1_to_bin_1" in orch.tasks


@pytest.mark.parametrize("clear_done", [False, True])
def test_clearing_the_queue_does_not_freeze_the_cell(clear_done):
    """`clear_tasks` goes through `set_tasks` too, and an empty queue is legitimate."""
    orch, _ = _working_cell()

    orch.set_tasks([], clear_done=clear_done)
    before = orch.control_ticks
    for _ in range(50):
        orch.step()

    assert orch.control_ticks > before, (
        "the clock stopped after clearing the queue")
    assert not orch.work_outstanding(), (
        "an empty queue should report no outstanding work, or the loop spins forever")