"""Demo 12 -- orchestration logic, tested without Isaac Sim.

These run in a second on any machine and cover the parts of the orchestrator that are
about *decisions*: who takes which task, what happens when a grasp fails, what happens to
a robot that faults mid-task, and whether "completed" means completed.

They use a scripted fake cell, and that is a real limitation worth stating plainly. The fake
is told, per task, whether the grasp succeeded; it then reports the part where the physics
would have left it. So these tests prove the orchestrator **reacts correctly to outcomes**,
not that the outcomes happen. Nothing here can tell you a grasp works.

For that there is ``tests/test_grasp.py``, which runs the same state machine inside real
Isaac Sim and measures everything. The two belong together: these catch a wrong decision,
that one catches a broken gripper, and neither substitutes for the other.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo12 import materials as M
from demo12.config import (BinConfig, MoveTask, PartConfig, RobotConfig, SceneConfig,
                           default_scene)
from demo12.orchestrator import (CLOSING, DONE, FAILED, MAX_ATTEMPTS,
                                 MultiRobotOrchestrator, QUEUED, VERIFYING)


# ---------------------------------------------------------------------------
# A cell that reports scripted outcomes
# ---------------------------------------------------------------------------


class FakePartState:
    def __init__(self, x, y, z, resting=True):
        self.x, self.y, self.z = x, y, z
        self.lin_speed = 0.0 if resting else 0.5
        self.ang_speed = 0.0 if resting else 2.0

    @property
    def resting(self) -> bool:
        return self.lin_speed < 0.02 and self.ang_speed < 0.5


class FakeArm:
    """Accepts joint targets and slides towards them. No physics, no joints, no pads."""

    def __init__(self, cell, robot_id):
        self.cell = cell
        self.robot_id = robot_id
        self.joints = [0.0] * 8
        self.tool_pos = [0.0, 0.0, 0.3]
        self.jaws_open = True
        self.pad_offset = 0.023

    def set_jaws(self, open_: bool) -> None:
        self.jaws_open = open_

    def command_joints(self, targets) -> None:
        self.cell.ticks += 1

    def solve_ik(self, target, max_step_rad=None):
        # Move a fixed fraction of the way each solve, so tool_error_m converges the way
        # the real arm's does and the runner's staging check can pass.
        for i in range(3):
            self.tool_pos[i] += 0.6 * (target[i] - self.tool_pos[i])
        return [0.0] * 5

    def tool_error_m(self, target) -> float:
        return math.dist(self.tool_pos, target)

    def pad_offset_below_tool_m(self) -> float:
        return self.pad_offset

    def align_pad_axis(self, angle, max_ticks: int = 200) -> int:
        return 1

    def pad_axis_angle_rad(self) -> float:
        return 0.0


class FakeCell:
    """A cell whose physics is a script.

    ``fail_budget`` maps a task id to how many of its attempts should fail. A succeeding
    attempt ends with its part resting in the bin, which is exactly what real PhysX
    produces; a failing one reports no grip force and leaves the part on the table.
    """

    def __init__(self, scene: SceneConfig, outcomes: dict[str, bool]):
        self.scene = scene
        self.fail_budget = {tid: (0 if ok else 10**6) for tid, ok in outcomes.items()}
        self.robots = {r.id: object() for r in scene.robots}
        self.part_cfgs = {p.id: p for p in scene.parts}
        self.bins = {b.id: b for b in scene.bins}
        self.parts: dict[str, FakePartState] = {
            p.id: FakePartState(p.x, p.y, M.TABLE_TOP_Z_M + p.size_m[2] / 2.0)
            for p in scene.parts
        }
        self.arms = {rid: FakeArm(self, rid) for rid in self.robots}
        self.ticks = 0
        self.orch: MultiRobotOrchestrator | None = None
        self._resolved: set[tuple[str, int]] = set()

    def bind(self, orch: "MultiRobotOrchestrator") -> None:
        """Let the cell see the orchestrator's runners, so it can settle each outcome."""
        self.orch = orch

    def _attempt_fails(self, task) -> bool:
        return task.attempts <= self.fail_budget.get(task.id, 0)

    # -- the surface the orchestrator actually uses ----------------------

    def arm(self, robot_id):
        return self.arms[robot_id]

    def step(self) -> None:
        self.ticks += 1
        if self.orch is None:
            return
        # The moment a task starts judging its result is where the scripted physics takes
        # effect: either the part ends up in the bin, or it does not.
        for task in list(self.orch.tasks.values()):
            runner = task.runner
            if runner is None or runner.phase != VERIFYING or runner.tick != 0:
                continue
            key = (task.id, task.attempts)
            if key in self._resolved:
                continue
            self._resolved.add(key)
            if self._attempt_fails(task):
                continue
            bin_cfg = self.bins[task.bin_id]
            part_cfg = self.part_cfgs[task.part_id]
            self.parts[task.part_id] = FakePartState(
                bin_cfg.x, bin_cfg.y,
                bin_cfg.cavity_floor_z_m + part_cfg.size_m[2] / 2.0)

    def part_state(self, part_id):
        return self.parts[part_id]

    def grip_force_n(self, robot_id):
        """Zero if the task being run is scripted to fail, a real squeeze otherwise."""
        if self.orch is not None:
            for task in self.orch.tasks.values():
                if task.runner is not None and task.runner.robot_id == robot_id:
                    return 0.0 if self._attempt_fails(task) else 60.0
        return 0.0

    def jaw_pad_span_m(self, robot_id):
        return 0.030


def _scene_with(robot_count: int, task_part_ids: list[str]) -> SceneConfig:
    scene = default_scene(robot_count)
    tasks = [MoveTask(id=f"task_{i + 1}", part_id=pid,
                      bin_id=f"bin_{i % 2 + 1}", label=f"{pid} move")
             for i, pid in enumerate(task_part_ids)]
    scene.mission.tasks = tasks
    return scene


def _make(scene: SceneConfig, cell: FakeCell) -> MultiRobotOrchestrator:
    orch = MultiRobotOrchestrator(cell, scene, scene.mission.tasks)
    cell.bind(orch)
    return orch


def _run(orch: MultiRobotOrchestrator, max_ticks: int = 40_000) -> int:
    """Step until nothing is outstanding, or the budget runs out."""
    for n in range(max_ticks):
        if not orch.work_outstanding():
            return n
        orch.step()
    return max_ticks


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


def test_one_robot_takes_one_task():
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)

    _run(orch)

    assert orch.all_tasks_completed()
    assert orch.tasks["task_1"].assigned_robot == "robot_1"
    assert orch.control_ticks > 0


def test_tasks_are_split_across_robots():
    """Two robots, two tasks: each should take one, not stack on a single arm.

    This is the property the previous per-task orchestrator did not have -- it ran each
    pick to completion before starting the next, so one robot did all the work.
    """
    scene = _scene_with(2, ["part_1", "part_2"])
    cell = FakeCell(scene, {"task_1": True, "task_2": True})
    orch = _make(scene, cell)

    _run(orch)

    assert orch.all_tasks_completed()
    owners = {t.assigned_robot for t in orch.tasks.values()}
    assert owners == {"robot_1", "robot_2"}, f"expected both robots busy, got {owners}"


def test_no_task_is_left_unassigned():
    scene = _scene_with(3, ["part_1", "part_2", "part_3"])
    cell = FakeCell(scene, {f"task_{i}": True for i in (1, 2, 3)})
    orch = _make(scene, cell)

    _run(orch)

    assert all(t.assigned_robot is not None for t in orch.tasks.values())
    assert orch.all_tasks_completed()


# ---------------------------------------------------------------------------
# Completion means completion
# ---------------------------------------------------------------------------


def test_failed_task_does_not_report_success():
    """A task whose part never arrives must not be counted as done."""
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": False})
    orch = _make(scene, cell)

    ticks = _run(orch)

    assert not orch.all_tasks_completed()
    assert orch.tasks["task_1"].state == FAILED
    assert "failed" in orch.tasks["task_1"].note
    # It stopped rather than spinning: the loop must be able to tell "finished" from
    # "hopeless".
    assert ticks < 40_000


def test_queued_work_counts_as_outstanding():
    """A task nobody has started yet is outstanding: the run must not stop on it."""
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)

    assert orch.all_tasks_completed() is False
    assert orch.work_outstanding() is True


def test_empty_mission_is_not_complete():
    """Zero tasks is not success. Otherwise a misconfigured scene looks like a pass."""
    scene = default_scene(1)
    cell = FakeCell(scene, {})
    orch = _make(scene, cell)

    assert orch.all_tasks_completed() is False


# ---------------------------------------------------------------------------
# Retry and redistribution
# ---------------------------------------------------------------------------


def test_failed_grasp_is_retried_up_to_the_limit():
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": False})
    orch = _make(scene, cell)

    _run(orch)

    task = orch.tasks["task_1"]
    assert task.state == FAILED
    assert task.attempts == MAX_ATTEMPTS, "should retry exactly MAX_ATTEMPTS times"
    # The note has to say how many attempts were spent, or a task that used the whole
    # retry budget is indistinguishable from one that failed once.
    assert f"after {MAX_ATTEMPTS} attempts" in task.note
    assert "no grip force" in task.note


def test_retry_succeeds_when_the_second_attempt_works():
    """A task that fails once and then succeeds must end DONE, not FAILED.

    The retry budget exists so a transient failure does not write off a task, so this is
    the case that gives the budget its meaning.
    """
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    cell.fail_budget["task_1"] = 1          # fail the first attempt only
    orch = _make(scene, cell)

    _run(orch)

    assert orch.tasks["task_1"].attempts == 2
    assert orch.all_tasks_completed()


def test_faulted_robot_releases_its_task_and_it_is_retaken():
    scene = _scene_with(2, ["part_1", "part_2"])
    cell = FakeCell(scene, {"task_1": True, "task_2": True})
    orch = _make(scene, cell)

    # Let both robots pick something up.
    for _ in range(5):
        orch.step()

    victim = next(r.id for r in orch.robots.values() if r.task_id is not None)
    held = next(t.id for t in orch.tasks.values() if t.assigned_robot == victim)

    result = orch.inject_failure(victim)

    assert result["ok"] is True
    assert held in result["released"]
    assert orch.robots[victim].state == "faulted"
    task = orch.tasks[held]
    assert task.assigned_robot is None
    assert task.state == QUEUED
    assert victim in task.previous_robots, "history of who held it must be kept"


def test_task_moves_to_another_robot_after_a_fault():
    scene = _scene_with(2, ["part_1", "part_2"])
    cell = FakeCell(scene, {"task_1": True, "task_2": True})
    orch = _make(scene, cell)

    for _ in range(5):
        orch.step()
    victim = next(r.id for r in orch.robots.values() if r.task_id is not None)
    held = next(t.id for t in orch.tasks.values() if t.assigned_robot == victim)

    orch.inject_failure(victim)
    orch.repair_robot(victim)
    _run(orch)

    task = orch.tasks[held]
    assert orch.all_tasks_completed(), f"held task ended {task.state}: {task.note}"
    assert victim in task.previous_robots


def test_faulted_robot_is_not_given_more_work():
    scene = _scene_with(2, ["part_1", "part_2"])
    cell = FakeCell(scene, {"task_1": True, "task_2": True})
    orch = _make(scene, cell)

    orch.inject_failure("robot_1")
    _run(orch)

    assert orch.robots["robot_1"].state == "faulted"
    assert orch.robots["robot_2"].state == "idle"
    assert orch.all_tasks_completed(), "the healthy robot should finish everything"


def test_faulting_an_unknown_robot_is_reported_not_raised():
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)

    result = orch.inject_failure("no_such_robot")

    assert result["ok"] is False
    assert "no_such_robot" in result["reason"]


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


def test_remove_robot_releases_and_refuses_the_last_one():
    scene = _scene_with(2, ["part_1", "part_2"])
    cell = FakeCell(scene, {"task_1": True, "task_2": True})
    orch = _make(scene, cell)

    for _ in range(5):
        orch.step()
    victim = next(r.id for r in orch.robots.values() if r.task_id is not None)
    held = next(t.id for t in orch.tasks.values() if t.assigned_robot == victim)

    result = orch.remove_robot(victim)

    assert result["ok"] is True
    assert held in result["reassigned"]
    assert victim not in orch.robots
    assert orch.tasks[held].state == QUEUED


def test_cannot_remove_the_last_robot():
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)

    result = orch.remove_robot("robot_1")

    assert result["ok"] is False
    assert "last robot" in result["reason"]
    assert "robot_1" in orch.robots


def test_add_robot_refuses_a_duplicate_id():
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)

    result = orch.add_robot(RobotConfig(id="robot_1"))

    assert result["ok"] is False
    assert "already present" in result["reason"]


def test_add_robot_refuses_rather_than_faking_it():
    """Without a cell that can rebuild, adding a robot must fail loudly.

    A method that returns ``{"ok": true}`` without having added anything is worse than one
    that refuses: the interface would show a robot that does not exist.
    """
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)

    result = orch.add_robot(RobotConfig(id="robot_2"))

    assert result["ok"] is False
    assert "relaunch" in result["reason"]


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


def test_submit_order_adds_tasks():
    scene = _scene_with(1, [])
    cell = FakeCell(scene, {})
    orch = _make(scene, cell)

    result = orch.submit_order([
        MoveTask(id="task_x", part_id="part_1", bin_id="bin_1", label="x"),
        MoveTask(id="task_y", part_id="part_2", bin_id="bin_2", label="y"),
    ])

    assert result["ok"] is True
    assert set(orch.tasks) == {"task_x", "task_y"}
    assert orch.work_outstanding() is True

    _run(orch)
    assert all(t.state == DONE for t in orch.tasks.values())


def test_failed_task_is_not_retried_forever():
    """The retry budget has to actually bound the work."""
    scene = _scene_with(1, ["part_1"])
    cell = FakeCell(scene, {"task_1": False})
    orch = _make(scene, cell)

    ticks = _run(orch)

    assert orch.tasks["task_1"].attempts == MAX_ATTEMPTS
    assert orch.work_outstanding() is False
    assert ticks < MAX_ATTEMPTS * 20_000, "run should end, not grind"


# ---------------------------------------------------------------------------
# Scene validation, which is pure Python and needs no simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("robot_count", [1, 2, 3])
def test_default_scenes_validate(robot_count):
    scene = default_scene(robot_count)
    scene.validate()
    assert len(scene.robots) >= robot_count


@pytest.mark.parametrize("robot_count", [1, 2, 3])
def test_default_scene_parts_are_within_reach_of_every_enabled_robot(robot_count):
    scene = default_scene(robot_count)
    for part in scene.parts:
        for robot in scene.enabled_robots:
            assert robot.reach.contains((part.x, part.y)), (
                f"{part.id} at ({part.x}, {part.y}) is out of reach of {robot.id}")


def test_parts_do_not_start_inside_a_bin():
    for robot_count in (1, 2, 3):
        scene = default_scene(robot_count)
        for part in scene.parts:
            for bin_cfg in scene.bins:
                assert not bin_cfg.contains_footprint(part.x, part.y,
                                                      part.footprint_radius_m)


def test_every_mission_task_refers_to_a_real_part_and_bin():
    for robot_count in (1, 2, 3):
        scene = default_scene(robot_count)
        part_ids = {p.id for p in scene.parts}
        bin_ids = {b.id for b in scene.bins}
        for task in scene.mission.tasks:
            assert task.part_id in part_ids
            assert task.bin_id in bin_ids


def test_gripper_opens_wider_than_a_square_part_across_any_axis():
    """The pads must straddle a square part whatever the approach angle.

    This is the regression test for the corner-catch. At 34.8 deg to the part's faces a
    30 mm cube presents its diagonal, 41.8 mm, and the mechanism has to open wider than
    that or it closes on a corner and punts the part instead of gripping it.
    """
    size = 0.030
    diagonal = size * math.sqrt(2.0)
    open_span = 2.0 * (M.JAW_PAD_CLOSED_HALF_M + M.JAW_PAD_TRAVEL_M) - M.FINGER_PAD_THICK_M
    assert open_span > diagonal, (
        f"open span {open_span * 1000:.1f} mm must exceed the {diagonal * 1000:.1f} mm "
        f"diagonal of a {size * 1000:.0f} mm cube")
    shut_span = 2.0 * M.JAW_PAD_CLOSED_HALF_M - M.FINGER_PAD_THICK_M
    assert shut_span < size, "the mechanism must be able to close past the part's width"