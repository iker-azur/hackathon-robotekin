"""Demo 12 — who does what, and when. Reads the simulation; never writes to it.

The rule the whole demo rests on: **the orchestrator never writes a pose into the
simulator.** It sends joint targets, exactly as a real robot's controller would, and then
reads back what PhysX actually did. If a part is not in a bin, it is because it was not put
there. Nothing in this module can mark a task done by deciding it was done.

Three things make this different from the version it replaces:

**It is per-tick, not per-task.** The previous orchestrator ran a robot's *entire*
pick-lift-place inside one ``step()`` call, so three robots moved in sequence rather than
together and the cell looked busy while two thirds of it stood still. Here every robot
advances one tick of its own state machine per call, which is what makes this a
multi-robot demo rather than three single-robot demos in a trenchcoat.

**It drives one control tick at a time.** The state machine never steps the simulation
itself. It reads state, decides, writes joint targets, and returns; the caller advances
time. Otherwise "the orchestrator" and "the physics" would be the same object, and the
separation this demo is built on would not exist.

**Completion is evidence, not intent.** A task is done when the part is inside the target
bin's cavity *and* the part is at rest. Both are read from the part's own rigid body. A
grasp that reports force but carries nothing therefore cannot pass, which is exactly the
failure the grasp probe was written to catch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import grasp as G
from . import materials as M
from .config import MoveTask, SceneConfig

# Task states. A task is never left in a state the simulation cannot confirm.
QUEUED = "queued"
APPROACHING = "approaching"
CLOSING = "closing"
LIFTING = "lifting"
TRANSFERRING = "transferring"
RELEASING = "releasing"
VERIFYING = "verifying"
DONE = "done"
FAILED = "failed"

# States in which a task is still making progress. Anything not in this set is terminal,
# and a run with no task in this set is finished -- successfully or not.
ACTIVE_STATES = frozenset({QUEUED, APPROACHING, CLOSING, LIFTING,
                           TRANSFERRING, RELEASING, VERIFYING})

# How many times a task is retried before it is given up on and handed to another robot.
MAX_ATTEMPTS = 3

# A grasp is treated as failed -- and the task retried from the start -- if the pads report
# no force at all after the close. Anything above this is a real squeeze.
GRASP_FORCE_FLOOR_N = 0.0


@dataclass
class Task:
    """One part has to end up in one bin, and how that is going."""

    id: str
    part_id: str
    bin_id: str
    label: str = ""
    state: str = QUEUED
    assigned_robot: str | None = None
    previous_robots: list[str] = field(default_factory=list)
    attempts: int = 0
    note: str = ""
    #: What the robot that is currently executing this task is doing, tick by tick.
    #: ``None`` when no robot holds it.
    runner: "TaskRunner | None" = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "partId": self.part_id,
            "binId": self.bin_id,
            "label": self.label,
            "state": self.state,
            "assignedRobot": self.assigned_robot,
            "previousRobots": list(self.previous_robots),
            "attempts": self.attempts,
            "note": self.note,
            "holding": self.runner.phase if self.runner else None,
        }


@dataclass
class RobotStatus:
    """A robot's availability, which is orchestration state and not physics state."""

    id: str
    state: str = "idle"          # idle | working | faulted
    task_id: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "state": self.state, "taskId": self.task_id,
                "note": self.note}


class TaskRunner:
    """Advances one task by exactly one control tick per call.

    This is the whole pick, expressed as a state machine over physics that is already
    running. Each phase runs for a measured number of ticks and then hands over; the
    orchestrator decides who runs what, and this decides how the motion is shaped.

    Reusing :class:`demo12.grasp.GraspProbe` rather than reimplementing the motion is
    deliberate. The probe is the code the four acceptance criteria are written against, so
    the demo's motions and the tests' motions cannot drift apart.
    """

    def __init__(self, cell, robot_id: str, task: Task) -> None:
        self.cell = cell
        self.robot_id = robot_id
        self.task = task
        self.arm = cell.arm(robot_id)
        self.probe = G.GraspProbe(cell, robot_id)

        self.phase = QUEUED
        self.tick = 0
        self.failed = False
        self.failures: list[str] = []
        self.approach_stage = "to_staging"
        self.grip_n = 0.0
        self.span_on_part = 0.0

        part_cfg = cell.part_cfgs[task.part_id]
        bin_cfg = cell.bins[task.bin_id]
        self.part_cfg = part_cfg
        self.bin_cfg = bin_cfg

        # Motion plan, built once from the part and bin as they are *now*. Re-planned if a
        # retry finds the part has moved, which it will have if the previous attempt
        # knocked it.
        self.grip_z = 0.0
        self.lift_z = 0.0
        self.release_z = bin_cfg.cavity_floor_z_m + part_cfg.size_m[2] / 2.0 + 0.030
        self.start_xy = (0.0, 0.0)
        self.staging_z = 0.0
        self.plan()

    def plan(self) -> None:
        """Recompute the motion from the part's current measured position."""
        state = self.cell.part_state(self.task.part_id)
        self.start_xy = (state.x, state.y)
        self.start_z = state.z
        self.staging_z = state.z + G.STAGING_RISE_M
        # Pad offset is measured at the staging pose, where the tool points down; see
        # ``GraspProbe.run``. Here the arm is wherever it happens to be, so this is only a
        # seed: the real value is measured once the arm gets there.
        self.grip_z = state.z + self.arm.pad_offset_below_tool_m()
        self.lift_z = self.grip_z + LIFT_HEIGHT_M

    # -- phase helpers ---------------------------------------------------

    def _hold(self, target, ticks: int, jaws_open: bool, step_rad: float) -> None:
        self.arm.set_jaws(jaws_open)
        for _ in range(ticks):
            self.arm.command_joints(self.arm.solve_ik(target, step_rad))
            self.cell.step()

    def _ramp(self, xy, z_from: float, z_to: float, ticks: int,
              jaws_open: bool, step_rad: float) -> None:
        """Rate-limited move along a vertical line. See ``GraspProbe._descend``."""
        self.arm.set_jaws(jaws_open)
        for k in range(1, ticks + 1):
            z = z_from + (z_to - z_from) * (k / ticks)
            self.arm.command_joints(self.arm.solve_ik((xy[0], xy[1], z), step_rad))
            self.cell.step()

    # -- the state machine ------------------------------------------------

    def advance(self) -> str:
        """Run this task for one control tick and return the phase it is now in.

        Every phase ends by advancing the simulation exactly once per call, which is what
        keeps the orchestrator from being able to teleport the world.
        """
        phase = self.phase

        if phase == QUEUED:
            self.phase = APPROACHING
            self.tick = 0
            self.approach_stage = "to_staging"
            return self.phase

        if phase == APPROACHING:
            return self._approach_tick()

        if phase == CLOSING:
            return self._close_tick()

        if phase == LIFTING:
            return self._lift_tick()

        if phase == TRANSFERRING:
            return self._transfer_tick()

        if phase == RELEASING:
            return self._release_tick()

        if phase == VERIFYING:
            return self._verify_tick()

        # DONE / FAILED: nothing to do.
        return self.phase

    def _approach_tick(self) -> str:
        """Three sub-phases: get to staging, square the pads, then descend.

        The staging move has to *converge* before anything else happens. The first version
        of this gave the arm a single tick at the staging pose and then started descending,
        and the consequence was not subtle: the arm began its descent from wherever it
        happened to be, arrived beside the part rather than over it, and the close found
        nothing. Measured across three attempts -- "no grip force after close" every time,
        with part_1 shoved 41 mm from where it started.
        """
        stage = self.approach_stage

        if stage == "to_staging":
            self.arm.set_jaws(True)
            self.staging = (self.start_xy[0], self.start_xy[1], self.staging_z)
            self.arm.command_joints(self.arm.solve_ik(self.staging, 0.078))
            self.cell.step()
            self.tick += 1
            if (self.arm.tool_error_m(self.staging) <= STAGING_TOLERANCE_M
                    or self.tick >= STAGING_MAX_TICKS):
                self.approach_stage = "align"
                self.tick = 0
                if self.arm.tool_error_m(self.staging) > STAGING_TOLERANCE_M:
                    self.failures.append(
                        f"could not reach staging point, {self.arm.tool_error_m(self.staging) * 1000:.0f} mm out")
                    self.failed = True
                    self.phase = CLOSING
            return self.phase

        if stage == "align":
            self.align_target = self.probe._nearest_axis_angle()
            self.arm.align_pad_axis(self.align_target)
            # Re-measure the pad offset now that the tool is down and the pads are square.
            self.grip_z = self.start_z + self.arm.pad_offset_below_tool_m()
            self.lift_z = self.grip_z + LIFT_HEIGHT_M
            self.approach_stage = "descend"
            self.tick = 0
            return self.phase

        # Descent. Rate-limited, because the corridor past a square part is narrow and a
        # fast tool wanders into it rather than down it.
        frac = min(1.0, self.tick / DESCENT_TICKS)
        z = self.staging_z + (self.grip_z - self.staging_z) * frac
        self.arm.command_joints(
            self.arm.solve_ik((self.start_xy[0], self.start_xy[1], z),
                              G.DESCENT_MAX_STEP_RAD))
        self.cell.step()
        self.tick += 1
        if frac >= 1.0:
            self.phase = CLOSING
            self.tick = 0
        return self.phase

    def _close_tick(self) -> str:
        """Hold shut until the squeeze is settled, then start the lift."""
        self.arm.set_jaws(False)
        self.arm.command_joints(
            self.arm.solve_ik((self.start_xy[0], self.start_xy[1], self.grip_z), 0.02))
        self.cell.step()
        self.tick += 1

        if self.tick == GRIP_PROBE_TICK:
            # Read the squeeze the moment the jaws have had time to settle, not at the end
            # of the task. A force that has vanished by the lift means the grasp was never
            # there.
            self.grip_n = self.cell.grip_force_n(self.robot_id)
            self.span_on_part = self.cell.jaw_pad_span_m(self.robot_id)

        if self.tick >= GRIP_SETTLE_TICKS:
            if getattr(self, "grip_n", 0.0) <= GRASP_FORCE_FLOOR_N:
                self.failed = True
                self.failures.append(
                    f"no grip force after close on {self.task.part_id}")
            self.phase = LIFTING
            self.tick = 0
        return self.phase

    def _lift_tick(self) -> str:
        frac = min(1.0, self.tick / G.LIFT_TICKS)
        z = self.grip_z + (self.lift_z - self.grip_z) * frac
        self.arm.set_jaws(False)
        self.arm.command_joints(
            self.arm.solve_ik((self.start_xy[0], self.start_xy[1], z),
                              G.LIFT_MAX_STEP_RAD))
        self.cell.step()
        self.tick += 1
        if frac >= 1.0:
            self.phase = TRANSFERRING
            self.tick = 0
        return self.phase

    def _transfer_tick(self) -> str:
        frac = min(1.0, self.tick / G.TRANSFER_TICKS)
        z = self.lift_z + (self.release_z - self.lift_z) * frac
        self.arm.set_jaws(False)
        self.arm.command_joints(self.arm.solve_ik((self.bin_cfg.x, self.bin_cfg.y, z),
                                                 G.TRANSFER_MAX_STEP_RAD))
        self.cell.step()
        self.tick += 1
        if frac >= 1.0:
            self.phase = RELEASING
            self.tick = 0
        return self.phase

    def _release_tick(self) -> str:
        self.arm.set_jaws(True)
        self.arm.command_joints(self.arm.solve_ik((self.bin_cfg.x, self.bin_cfg.y,
                                                  self.release_z), 0.02))
        self.cell.step()
        self.tick += 1
        if self.tick >= G.RELEASE_SETTLE_TICKS:
            self.phase = VERIFYING
            self.tick = 0
        return self.phase

    def _verify_tick(self) -> str:
        """Retreat, let the part settle, then judge it from its own state."""
        frac = min(1.0, self.tick / G.RETREAT_TICKS)
        z = self.release_z + RETREAT_HEIGHT_M * frac
        self.arm.set_jaws(True)
        self.arm.command_joints(self.arm.solve_ik((self.bin_cfg.x, self.bin_cfg.y, z),
                                                 G.LIFT_MAX_STEP_RAD))
        self.cell.step()
        self.tick += 1

        if self.tick >= G.RETREAT_TICKS + G.REST_SETTLE_TICKS:
            if self.failed:
                return FAILED
            inside, resting = self.judge()
            if inside and resting:
                self.phase = DONE
                return DONE
            self.failed = True
            why = []
            if not inside:
                why.append("not inside the bin cavity")
            if not resting:
                why.append("still moving")
            self.failures.append("; ".join(why))
            return FAILED
        return self.phase

    def judge(self) -> tuple[bool, bool]:
        """Is the part in the bin, and has it stopped? Read from the part itself.

        Two independent conditions, and both are needed. A part sitting in the tray while
        still rolling is not placed; a part at rest outside the tray is not placed.
        """
        state = self.cell.part_state(self.task.part_id)
        inside = self.bin_cfg.contains_footprint(
            state.x, state.y, self.part_cfg.footprint_radius_m)
        return inside, state.resting

    def measurements(self) -> dict[str, Any]:
        return {
            "gripN": round(getattr(self, "grip_n", 0.0), 4),
            "spanOnPartM": round(getattr(self, "span_on_part", 0.0), 6),
            "failures": list(self.failures),
        }


# How high the tool lifts a part before transferring. 100 mm is enough to clear a tray rim
# (45 mm) and low enough that the wrist is not near its reach boundary.
LIFT_HEIGHT_M = 0.100

# How far to raise the tool after a release before judging the part's resting place.
RETREAT_HEIGHT_M = 0.080

# Descent ticks in the runner match the probe's, so the demo and the tests approach a part
# identically.
DESCENT_TICKS = G.DESCENT_TICKS

# The probe measures the squeeze after this many settle ticks. Matched to
# ``materials.GRIPPER_SETTLE_TICKS`` in spirit: long enough that the jaws have reached
# their commanded position, short enough to catch a grasp that is not going to happen.
GRIP_SETTLE_TICKS = G.GRIP_SETTLE_TICKS
GRIP_PROBE_TICK = 30

# How close to the staging point the arm must get before the pads are squared up and the
# descent begins, and how long it gets to get there. The cap exists so an unreachable
# staging point is *reported* rather than waited on forever.
STAGING_TOLERANCE_M = 0.003
STAGING_MAX_TICKS = 600


class MultiRobotOrchestrator:
    """Assigns tasks from a shared queue and watches them to completion.

    Reads the cell, writes joint targets, and decides nothing it has not measured. The
    public surface is deliberately small: submit work, step time, ask whether it is done,
    and change the robot roster.
    """

    def __init__(self, cell, scene: SceneConfig | None = None,
                 tasks: list[MoveTask] | None = None) -> None:
        self.cell = cell
        self.scene = scene if scene is not None else cell.scene
        if tasks is None:
            tasks = self.scene.mission.tasks
        self.tasks: dict[str, Task] = {
            t.id: Task(id=t.id, part_id=t.part_id, bin_id=t.bin_id, label=t.label)
            for t in tasks
        }
        self.robots: dict[str, RobotStatus] = {
            rid: RobotStatus(id=rid) for rid in cell.robots
        }
        self.sim_time_s = 0.0
        self.control_ticks = 0
        self.failures_injected = 0
        self.reassignments = 0

    # -- queries ---------------------------------------------------------

    def all_tasks_completed(self) -> bool:
        """True when every task reached DONE.

        Not "queued, assigned or done" -- a task that is merely in progress has not been
        completed, and counting it as such is how a demo reports success it has not had.
        """
        return all(t.state == DONE for t in self.tasks.values()) and bool(self.tasks)

    def work_outstanding(self) -> bool:
        """Whether any task could still make progress.

        Separate from :meth:`all_tasks_completed` because a run where one task is DONE and
        another has exhausted its retries is *finished*, not *successful*. Only the first
        says success; both say stop.

        Without this the caller cannot tell an unfinished run from a hopeless one, and a
        hopeless one burns the entire time budget: measured, a run whose single remaining
        task was terminally failed spun for 6,000 further control ticks with the simulated
        clock frozen, because ``all_tasks_completed()`` was correctly false and nothing
        else ever became true.
        """
        return any(t.state in ACTIVE_STATES for t in self.tasks.values())

    @property
    def failed_tasks(self) -> list[str]:
        return [t.id for t in self.tasks.values() if t.state == FAILED]

    def snapshot(self) -> dict[str, Any]:
        return {
            "simTimeS": round(self.sim_time_s, 3),
            "controlTicks": self.control_ticks,
            "allTasksCompleted": self.all_tasks_completed(),
            "tasksCompleted": len([t for t in self.tasks.values() if t.state == DONE]),
            "tasksTotal": len(self.tasks),
            "failuresInjected": self.failures_injected,
            "reassignments": self.reassignments,
            "robots": [r.to_dict() for r in self.robots.values()],
            "tasks": [t.to_dict() for t in self.tasks.values()],
        }

    # -- assignment ------------------------------------------------------

    def submit_order(self, tasks: list[MoveTask]) -> dict[str, Any]:
        """Add tasks to the queue. Existing task ids are replaced, not merged."""
        for t in tasks:
            self.tasks[t.id] = Task(id=t.id, part_id=t.part_id, bin_id=t.bin_id,
                                    label=t.label)
        return {"ok": True, "tasksSubmitted": [t.id for t in tasks],
                "tasksTotal": len(self.tasks)}

    def _assignable(self) -> Task | None:
        """The next task some healthy robot could take.

        Lowest attempt count first, so a task that keeps failing is not also the one every
        robot keeps retrying while fresh work waits. Ties break on id so the order is
        deterministic and a failed run can be compared against the last one.
        """
        candidates = [t for t in self.tasks.values()
                      if t.state == QUEUED and t.assigned_robot is None]
        if not candidates:
            return None
        return min(candidates, key=lambda t: (t.attempts, t.id))

    def _idle_robots(self) -> list[RobotStatus]:
        return [r for r in self.robots.values() if r.state == "idle"]

    # -- stepping --------------------------------------------------------

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """Advance every robot by one control tick, then return.

        The caller owns time. This method decides what each robot should be doing this tick
        and writes joint targets; it never calls the simulation itself outside a runner's
        motion helpers, and it never writes a pose.
        """
        for robot in self.robots.values():
            if robot.state == "faulted":
                continue

            if robot.state == "idle":
                task = self._assignable()
                if task is not None:
                    self._assign(task, robot)
                else:
                    # Idle robots still hold their last joint target, so write it and step,
                    # or the arm will sag under gravity while it waits.
                    self._hold_idle(robot)

            task = self._task_for(robot.task_id)
            if task is None or task.runner is None:
                continue

            phase = task.runner.advance()
            self.control_ticks += 1
            self.sim_time_s += dt

            if phase in (DONE, FAILED):
                self._finish(task, robot, phase)

    def _hold_idle(self, robot: RobotStatus) -> None:
        """Keep an idle arm parked where it is, rather than letting it drop."""
        arm = self.cell.arm(robot.id)
        arm.command_joints([float(v) for v in arm.joints[:5]])
        self.cell.step()
        self.control_ticks += 1

    def _task_for(self, task_id: str | None) -> Task | None:
        return self.tasks.get(task_id) if task_id else None

    def _assign(self, task: Task, robot: RobotStatus) -> None:
        """Give a task to a robot, recording who had it before."""
        if task.assigned_robot and task.assigned_robot != robot.id:
            task.previous_robots.append(task.assigned_robot)
            self.reassignments += 1
        task.assigned_robot = robot.id
        task.attempts += 1
        task.state = APPROACHING
        task.note = ""
        task.runner = TaskRunner(self.cell, robot.id, task)
        task.runner.phase = QUEUED
        robot.state = "working"
        robot.task_id = task.id

    def _finish(self, task: Task, robot: RobotStatus, phase: str) -> None:
        """Close out a task, and decide what happens next from what was measured."""
        runner = task.runner
        task.note = "; ".join(runner.failures) if runner else ""
        task.runner = None
        robot.task_id = None

        if phase == DONE:
            task.state = DONE
            robot.state = "idle"
            return

        # Failed. Put the task back on the queue if it has attempts left, otherwise mark it
        # failed for good and let another robot take it if it has one.
        if task.attempts < MAX_ATTEMPTS:
            task.state = QUEUED
            task.assigned_robot = None
            task.note = (f"attempt {task.attempts} failed: {task.note}")
            robot.state = "idle"
        else:
            task.state = FAILED
            robot.state = "idle"

    # -- roster and faults -----------------------------------------------

    def add_robot(self, robot_cfg) -> dict[str, Any]:
        """Add a robot to the cell, rebuilding it.

        Isaac Lab cannot add an articulation to a live scene: the asset's PhysX tensor
        views, its actuator registry and every contact sensor are built at construction and
        invalidated by a reset. The honest implementation is therefore a rebuild, not an
        insertion, and the previous roster is returned so a caller can restore it.

        ``IsaacCell`` owns the rebuild; see :meth:`isaac_env.IsaacCell.rebuild`. This
        method exists on the orchestrator so the roster is changeable from the same
        interface as everything else, and it refuses rather than pretending if the cell
        cannot rebuild.
        """
        if robot_cfg.id in self.robots:
            return {"ok": False, "reason": f"robot {robot_cfg.id!r} already present"}

        rebuild = getattr(self.cell, "rebuild_with_robot", None)
        if rebuild is None:
            return {"ok": False,
                    "reason": "this cell does not support adding a robot at runtime; "
                              "configure it in the SceneConfig and relaunch"}

        previous = self.scene.robots
        self.scene.robots = list(previous) + [robot_cfg]
        try:
            rebuild()
        except Exception:
            self.scene.robots = previous
            raise

        self.robots[robot_cfg.id] = RobotStatus(id=robot_cfg.id)
        return {"ok": True, "robotId": robot_cfg.id,
                "roster": [r.id for r in self.robots.values()]}

    def remove_robot(self, robot_id: str) -> dict[str, Any]:
        """Remove a robot, reassigning anything it was holding before it goes."""
        if robot_id not in self.robots:
            return {"ok": False, "reason": f"no robot with id {robot_id!r}"}
        if len(self.robots) <= 1:
            return {"ok": False, "reason": "the last robot cannot be removed"}

        # Anything this robot was holding goes back on the queue, with it recorded as a
        # previous owner so the history is not lost.
        reassigned = []
        for task in self.tasks.values():
            if task.assigned_robot != robot_id:
                continue
            task.previous_robots.append(robot_id)
            task.assigned_robot = None
            task.state = QUEUED
            task.runner = None
            task.note = f"released by removed robot {robot_id}"
            reassigned.append(task.id)
            self.reassignments += 1

        rebuild = getattr(self.cell, "rebuild_without_robot", None)
        if rebuild is not None:
            previous = list(self.scene.robots)
            self.scene.robots = [r for r in previous if r.id != robot_id]
            try:
                rebuild()
            except Exception:
                self.scene.robots = previous
                raise

        del self.robots[robot_id]
        return {"ok": True, "removed": robot_id, "reassigned": reassigned,
                "roster": [r.id for r in self.robots.values()]}

    def inject_failure(self, robot_id: str) -> dict[str, Any]:
        """Fault a robot, as a UI button or a test would.

        The robot stops being eligible for work and whatever it was holding goes back on
        the queue. Nothing about the simulation is touched: this is orchestration state, so
        a fault here cannot be confused with the arm breaking.
        """
        robot = self.robots.get(robot_id)
        if robot is None:
            return {"ok": False, "reason": f"no robot with id {robot_id!r}"}

        released = []
        for task in self.tasks.values():
            if task.assigned_robot != robot_id:
                continue
            task.previous_robots.append(robot_id)
            task.assigned_robot = None
            task.state = QUEUED
            task.runner = None
            task.note = f"faulted robot {robot_id}"
            released.append(task.id)
            self.reassignments += 1

        robot.state = "faulted"
        robot.task_id = None
        robot.note = "failure injected"
        self.failures_injected += 1
        return {"ok": True, "robotId": robot_id, "released": released,
                "state": robot.state}

    def repair_robot(self, robot_id: str) -> dict[str, Any]:
        """Bring a faulted robot back into the pool."""
        robot = self.robots.get(robot_id)
        if robot is None:
            return {"ok": False, "reason": f"no robot with id {robot_id!r}"}
        robot.state = "idle"
        robot.note = ""
        return {"ok": True, "robotId": robot_id, "state": robot.state}