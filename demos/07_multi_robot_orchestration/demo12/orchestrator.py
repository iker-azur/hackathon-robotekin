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

**Completion is evidence, not intent.** A task is done when *this robot demonstrably
carried this part* into the target bin and left it at rest. Every part of that is a
reading, checked by :func:`demo12.grasp.evaluate_grasp` -- the same function the offline
grasp probe uses, so a test and the demo cannot disagree about whether a grasp was real.
The part must rise with the jaw, the pads must have stalled at the part's own width on both
faces, and the squeeze must exceed what the friction cone needs for that mass. A gripper
reporting force while holding nothing, or a part that ended up in the tray because something
else knocked it there, both fail.

**Decisions are logged.** Every assignment, reassignment, fault and retry is recorded in
:mod:`demo12.events` with its simulated time and its reason, so the collaboration between
robots is something you can watch rather than infer from a table of columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import events as EV
from . import grasp as G
from . import materials as M
from . import planner as P
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

# The grip floor is no longer a constant. It comes from the friction cone --
# ``grasp.required_grip_n`` -- so it is a property of the part's mass and the pad material
# rather than a number someone picked. ``grasp.GRIP_MIN_N`` (0 N) remains the absolute
# floor underneath it, and both are used together: a task has to exceed the higher of the
# two.
#
# It used to be literally 0.0 with the comment "a squeeze below this is not carrying
# anything", which described an intention the code did not implement: any force above zero
# passed, including the 103 N that a punt against a corner produces.
GRASP_FORCE_FLOOR_N = G.GRIP_MIN_N


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
    #: The full grasp record from the last attempt: seven criteria, their measurements, and
    #: what was on the pads. ``None`` until an attempt finishes. This is what the interface
    #: shows instead of a green tick, so a completed task can be audited rather than
    #: believed.
    evidence: "G.GraspEvidence | None" = None
    #: Every attempt, as a short record. The last one is what counts; the earlier ones are
    #: how a task that failed twice and then worked explains itself.
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Simulated time this task first became active, for the timeline.
    started_tick: int = 0
    finished_tick: int = 0

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
            "progress": self.runner.progress if self.runner else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "history": list(self.history),
            "startedTick": self.started_tick,
            "finishedTick": self.finished_tick,
        }


@dataclass
class RobotStatus:
    """A robot's availability, which is orchestration state and not physics state."""

    id: str
    state: str = "idle"          # idle | working | faulted
    task_id: str | None = None
    note: str = ""

    # -- collaboration telemetry ----------------------------------------
    #
    # Everything below is here so the interface can explain what this arm is doing without
    # polling the simulation separately, and so the load chart can be drawn from counters
    # rather than by replaying events.
    #: Control ticks spent holding a task. The numerator of "how much of this cell was
    #: actually used", which is the number a multi-robot demo is actually judged on.
    busy_ticks: int = 0
    #: Control ticks spent parked, waiting for work.
    idle_ticks: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    #: Live reading of the pads, so the grip can be watched happening rather than inferred
    #: from a task's final state.
    grip_n: float = 0.0
    span_m: float = 0.0
    tool_xyz: list[float] = field(default_factory=list)
    joints_rad: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "state": self.state, "taskId": self.task_id,
                "note": self.note, "busyTicks": self.busy_ticks,
                "idleTicks": self.idle_ticks,
                "tasksCompleted": self.tasks_completed,
                "tasksFailed": self.tasks_failed,
                "gripN": round(self.grip_n, 3),
                "spanM": round(self.span_m, 5),
                "tool": [round(float(v), 4) for v in self.tool_xyz],
                "joints": [round(float(v), 4) for v in self.joints_rad],
                "utilisation": (round(self.busy_ticks / max(1, self.busy_ticks + self.idle_ticks), 3)
                                if (self.busy_ticks + self.idle_ticks) else 0.0)}


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

        # -- grasp evidence, accumulated one reading at a time -------------
        #
        # Each of these is filled in the phase that can measure it, and the set is turned
        # into one verdict by ``grasp.evaluate_grasp`` at the end. Keeping them as separate
        # fields rather than a running dict is what makes it obvious which phase is
        # responsible for which claim: if ``carried`` is ever false, the field was last
        # written in the transfer phase and nowhere else.
        self.tool_error_m = 0.0
        self.part_drift_on_close_m = 0.0
        self.part_rise_m = 0.0
        self.grip_n_during_lift = 0.0
        self.part_carried_to_bin = False
        self.start_z = 0.0
        self.evidence: "G.GraspEvidence | None" = None
        #: Set when the close was rejected in ``_check_close``. The part was never
        #: held, so the retreat has nowhere to go except back up.
        self.aborted = False
        #: Simulated time the attempt began, for the duration column.
        self.started_tick = 0

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

    # -- progress --------------------------------------------------------

    @property
    def progress(self) -> dict[str, Any]:
        """Where in its own pick this attempt is, as a fraction and a label.

        Derived from ticks elapsed against ticks required, so a task that is stuck in a
        long transfer shows a bar that is honestly stuck rather than one that keeps moving
        because a CSS animation does not know what happened.
        """
        stage = getattr(self, "approach_stage", "to_staging")
        key = stage if self.phase == APPROACHING else self.phase
        span = PHASE_TICKS.get(key) or PHASE_TICKS.get(self.phase, 1)
        fraction = min(1.0, self.tick / span) if span else 0.0
        return {"phase": self.phase, "stage": stage, "fraction": round(fraction, 3),
                "tick": self.tick, "spanTicks": span}

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
        """Shut the jaws, measure the squeeze, and refuse to go on with a bad grasp.

        The failure check used to run only at the end of the settle and only tested
        ``grip_n > 0``. Two things were wrong with that. A force of 0.001 N passed; and when
        it did fail, the runner still went ahead and spent a 260-tick lift and a 520-tick
        transfer before reporting it, so a task that could never succeed cost half a minute
        of the whole cell's time before anyone heard about it.

        Now the grasp is judged as soon as the jaws have settled, against the friction
        cone, *and* against where the pads actually landed. A close that punts the part --
        pads shut on a corner, span stalled wide of the part's width -- is rejected here,
        before the lift that would have looked successful while nothing was held.
        """
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

        if self.tick == GRASP_CHECK_TICK:
            verdict = self._check_close()
            if verdict is not None:
                return verdict

        if self.tick >= GRIP_SETTLE_TICKS:
            self.phase = LIFTING
            self.tick = 0
        return self.phase

    def _check_close(self) -> str | None:
        """Reject a close that did not actually take hold. Returns a phase if it failed.

        Four independent reasons to stop, checked cheapest-first and all of them measured:

        * **Squeeze.** Below the friction cone's requirement for this part's mass. This is
          the "the pads closed on air" case.
        * **Span.** The pads must have stalled at the part's own width. A span still near
          the free shut figure means they closed on something thinner than the part -- a
          corner, or nothing -- and this is the criterion that used to be missing entirely.
        * **Drift.** The part has to still be where it was. A close that shoves the part
          sideways has punted it, whatever the pads report.
        * **Tracking.** The tool has to have been where it was told to be, or the width
          match is meaningless.

        Returns ``VERIFYING`` rather than ``FAILED`` so the arm still retreats and the part
        still settles; the verdict is taken in ``_verify_tick`` from the same evidence, with
        these flags already set.
        """
        width = self.part_cfg.grasp_width_m
        needed = G.required_grip_n(self.part_cfg.mass_kg)

        if self.grip_n < max(GRASP_FORCE_FLOOR_N, needed):
            self.failures.append(
                f"grip {self.grip_n:.3f} N is below the {needed:.3f} N needed to hold "
                f"{self.part_cfg.mass_kg * 1000:.0f} g on rubber pads")
            self.failed = True
        elif not (self.span_on_part > G.SPAN_SHUT_MAX_M
                  and abs(self.span_on_part - width) <= G.ON_PART_SPAN_TOLERANCE_M):
            self.failures.append(
                f"pads stalled at {self.span_on_part * 1000:.1f} mm on a "
                f"{width * 1000:.1f} mm part, so they are not on both faces")
            self.failed = True

        state = self.cell.part_state(self.task.part_id)
        drift = math.dist((state.x, state.y), self.start_xy)
        self.part_drift_on_close_m = drift
        if drift > MAX_CLOSE_DRIFT_M:
            self.failures.append(
                f"the close pushed the part {drift * 1000:.1f} mm off its spot")
            self.failed = True

        self.tool_error_m = self.arm.tool_error_m((self.start_xy[0], self.start_xy[1],
                                                  self.grip_z))

        if self.failed:
            # Open the jaws and let go. There is nothing to lift, and driving the arm to the
            # destination with empty pads is both pointless and rude: it parks a gripper over
            # the table for another robot's benefit. ``_verify_tick`` reads this flag and
            # simply holds the grasp pose while the settle ticks run out.
            self.aborted = True
            self.arm.set_jaws(True)
            self.phase = VERIFYING
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
            # Did the part actually come with us? Read here, at the top of the lift, because
            # this is the single reading that separates "the robot picked something up" from
            # "the robot moved and the part stayed where it was". A grip that reports 60 N
            # while the part is still on the bench is the failure this whole module exists
            # to catch, and after the transfer the part's position is no longer evidence of
            # anything.
            state = self.cell.part_state(self.task.part_id)
            self.part_rise_m = state.z - self.start_z
            self.grip_n_during_lift = self.cell.grip_force_n(self.robot_id)
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
            # Is the part over the destination, still in the jaws? Recorded before the
            # release, because once the jaws open the part is falling and its position says
            # nothing about who carried it there.
            held = self.cell.part_state(self.task.part_id)
            self.part_carried_to_bin = math.hypot(held.x - self.bin_cfg.x,
                                                  held.y - self.bin_cfg.y) <= CARRY_TOLERANCE_M
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
        """Retreat, let the part settle, then judge the attempt from its own readings.

        The verdict comes from :func:`grasp.evaluate_grasp`, the same function the offline
        probe runs. This used to be ``if inside and resting``, which asked only where the
        part ended up and not whether this robot had put it there -- so a part knocked into
        the tray by another arm, or punted across the table into it, counted as a successful
        pick by whichever arm happened to be nearest. The evidence adds ``part_rose`` and
        ``carried``, which nothing but this arm carrying this part can satisfy.
        """
        # When the close was aborted there is nothing in the jaws, so the arm parks
        # above the part instead of driving to the destination with empty pads -- which
        # would also park a gripper over the table for whichever robot comes next. It
        # still waits out the settle ticks: the verdict comes from the same evidence
        # either way, and the part needs the same time to stop moving before anything
        # about it can be believed.
        frac = min(1.0, self.tick / G.RETREAT_TICKS)
        if self.aborted:
            target = (self.start_xy[0], self.start_xy[1],
                      self.grip_z + RETREAT_HEIGHT_M * frac)
        else:
            target = (self.bin_cfg.x, self.bin_cfg.y,
                      self.release_z + RETREAT_HEIGHT_M * frac)
        self.arm.set_jaws(True)
        self.arm.command_joints(self.arm.solve_ik(target, G.LIFT_MAX_STEP_RAD))
        self.cell.step()
        self.tick += 1

        if self.tick < G.RETREAT_TICKS + G.REST_SETTLE_TICKS:
            return self.phase

        final = self.cell.part_state(self.task.part_id)
        inside = self.bin_cfg.contains_footprint(
            final.x, final.y, self.part_cfg.footprint_radius_m)

        evidence = G.evaluate_grasp(
            part_id=self.task.part_id,
            bin_id=self.task.bin_id,
            robot_id=self.robot_id,
            mass_kg=self.part_cfg.mass_kg,
            tool_error_m=self.tool_error_m,
            part_width_m=self.part_cfg.grasp_width_m,
            span_on_part_m=self.span_on_part,
            grip_force_n=self.grip_n,
            part_rise_m=self.part_rise_m,
            part_carried_to_bin=self.part_carried_to_bin,
            part_inside_bin=inside,
            part_resting=final.resting,
            grip_force_during_lift_n=self.grip_n_during_lift,
            part_drift_on_close_m=self.part_drift_on_close_m,
        )
        self.evidence = evidence
        self.failures.extend(evidence.failures)
        self.failed = not evidence.ok
        self.phase = DONE if evidence.ok else FAILED
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
            "gripN": round(self.grip_n, 4),
            "gripRequiredN": round(G.required_grip_n(self.part_cfg.mass_kg), 4),
            "spanOnPartM": round(self.span_on_part, 6),
            "toolErrorM": round(self.tool_error_m, 6),
            "partRiseM": round(self.part_rise_m, 6),
            "partCarriedToBin": self.part_carried_to_bin,
            "partDriftOnCloseM": round(self.part_drift_on_close_m, 6),
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

# When the close is judged. Long enough that the prismatic pads have actually reached their
# commanded span and the contact sensors have reported, short enough that a grasp which is
# never going to take costs the cell 60 ticks rather than 780.
GRASP_CHECK_TICK = 45

# How far the part may be shoved by the close before that is a punt rather than a grasp.
# 10 mm is the same figure the offline probe uses, and the same figure it was measured with
# (the pads met one face 18.3 mm away and the other 26.3 mm away, and pushed the part
# 36 mm).
MAX_CLOSE_DRIFT_M = 0.010

# How close to the destination's centre the part must be, still held, for the robot to have
# carried it there. 20 mm is inside the bin's own inner width, so a part carried to the
# right tray but dropped off-centre still counts as carried; a part left behind on the table
# is 100 mm away and does not.
CARRY_TOLERANCE_M = 0.020


# How close to the staging point the arm must get before the pads are squared up and the
# descent begins, and how long it gets to get there. The cap exists so an unreachable
# staging point is *reported* rather than waited on forever.
STAGING_TOLERANCE_M = 0.003
STAGING_MAX_TICKS = 600

#: Control ticks each phase of a pick takes, so the interface can draw a progress bar
#: against something real rather than against an animation of its own invention. The counts
#: match the phases in :meth:`TaskRunner.advance` exactly; where one is a cap rather than a
#: fixed length (``to_staging``, ``align``) the cap is what is shown, which is the worst case
#: and therefore never a bar that quietly overstates progress.
#:
#: Module-level rather than a class attribute because the class body runs at import time,
#: before these constants are bound.
PHASE_TICKS: dict[str, int] = {
    "to_staging": STAGING_MAX_TICKS,
    "align": 200,                       # G.ROLL_ALIGN budget inside ArmController
    "descend": G.DESCENT_TICKS,
    "closing": GRIP_SETTLE_TICKS,
    "lifting": G.LIFT_TICKS,
    "transferring": G.TRANSFER_TICKS,
    "releasing": G.RELEASE_SETTLE_TICKS,
    "verifying": G.RETREAT_TICKS + G.REST_SETTLE_TICKS,
}


class MultiRobotOrchestrator:
    """Assigns tasks from a shared queue and watches them to completion.

    Reads the cell, writes joint targets, and decides nothing it has not measured. The
    public surface is deliberately small: submit work, step time, ask whether it is done,
    and change the robot roster.
    """

    def __init__(self, cell, scene: SceneConfig | None = None,
                 tasks: list[MoveTask] | None = None,
                 events: "EV.EventLog | None" = None) -> None:
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
        self.control_ticks = 0
        self.failures_injected = 0
        self.reassignments = 0
        self.roster_changes = 0

        self.events = events if events is not None else EV.EventLog()
        self.policy = P.DEFAULT_POLICY
        #: The cost breakdown behind each assignment, for the interface to show.
        self.assignment_costs: dict[str, dict[str, Any]] = {}

        self.events.emit(
            EV.PLAN, tick=0, sim_time_s=0.0,
            detail=f"{len(self.robots)} robot(s), {len(self.tasks)} task(s)",
            robots=list(self.robots), tasks=[t.id for t in self.tasks.values()])

    @property
    def sim_time_s(self) -> float:
        """Simulated seconds, read from the cell.

        This used to be a counter the orchestrator incremented by 1/60 per robot-tick,
        which is wrong twice over: the physics advances once per *robot*, not once per
        orchestrator step, and the idle branch incremented ``control_ticks`` without
        incrementing ``sim_time_s`` at all. A cell with three robots and one task reported a
        third of the time that had actually been simulated. In a demo whose whole argument is
        that the numbers come from the simulation rather than from the code's own opinion,
        the clock was the one number invented locally.

        ``IsaacCell.step`` accumulates ``sim_dt * substeps`` for every physics substep, so
        this is the real elapsed simulated time and nothing else has to maintain it.
        """
        return float(getattr(self.cell, "sim_time_s", 0.0))

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
            "rosterChanges": self.roster_changes,
            "robots": [r.to_dict() for r in self.robots.values()],
            "tasks": [t.to_dict() for t in self.tasks.values()],
            "assignmentCosts": dict(self.assignment_costs),
            "eventCursor": self.events.next_seq,
            "eventCounts": self.events.counts(),
            "metrics": self.metrics(),
        }

    def metrics(self) -> dict[str, Any]:
        """The numbers worth putting on a slide, all measured, none estimated.

        Every field is computed from the run's own history. ``graspSuccessRate`` in
        particular is the honest one: it is completions over *attempts*, so a run that
        needed four tries to place four parts reports 1.0 and not four.
        """
        attempts = sum(t.attempts for t in self.tasks.values())
        completed = len([t for t in self.tasks.values() if t.state == DONE])
        evidences = [t.evidence for t in self.tasks.values() if t.evidence]
        rises = [e.part_rise_m for e in evidences]
        grips = [e.grip_force_n for e in evidences]
        errors = [e.tool_error_m for e in evidences]
        sim_time = self.sim_time_s
        return {
            "tasksTotal": len(self.tasks),
            "tasksCompleted": completed,
            "tasksFailed": len(self.failed_tasks),
            "attempts": attempts,
            "graspSuccessRate": (round(completed / attempts, 3) if attempts else 0.0),
            "reassignments": self.reassignments,
            "failuresInjected": self.failures_injected,
            "controlTicks": self.control_ticks,
            "simTimeS": round(sim_time, 2),
            "tasksPerSimMinute": (round(completed / (sim_time / 60.0), 2)
                                  if sim_time > 0 else 0.0),
            "meanPartRiseMm": (round(sum(rises) / len(rises) * 1000.0, 1) if rises else 0.0),
            "meanGripN": (round(sum(grips) / len(grips), 2) if grips else 0.0),
            "meanToolErrorMm": (round(sum(errors) / len(errors) * 1000.0, 3) if errors else 0.0),
            "robotUtilisation": {rid: r.to_dict()["utilisation"]
                                 for rid, r in self.robots.items()},
            "criteriaPassed": self._criteria_tally(evidences),
        }

    def _criteria_tally(self, evidences: list["G.GraspEvidence"]) -> dict[str, dict[str, int]]:
        """How many of the seven grasp criteria held, across every attempt that finished.

        This is the number that shows whether the strictness actually cost anything. If
        every attempt passes all seven, the extra criteria are free. If ``pads_on_faces``
        is the one that fails, the cell has an alignment problem and the fix is in the
        grasp, not in the threshold.
        """
        tally: dict[str, dict[str, int]] = {}
        for e in evidences:
            for name, passed in e.criteria.items():
                slot = tally.setdefault(name, {"passed": 0, "failed": 0})
                slot["passed" if passed else "failed"] += 1
        return tally

    # -- assignment ------------------------------------------------------

    def set_tasks(self, tasks: list[MoveTask], *, clear_done: bool = False) -> dict[str, Any]:
        """Replace the queue with exactly these tasks, and say what happened to the old ones.

        This is the endpoint behind the interface's piece-to-box picker. Replacing rather
        than merging is the right shape for it: the user's mental model is "here is the job
        list I just built", and a queue that quietly accumulated tasks from earlier edits
        would not match anything on screen.

        Tasks already in flight are requeued rather than dropped, because the robot holding
        one is mid-pick and abandoning it would leave a part in the air. Tasks already DONE
        are kept only when ``clear_done`` is false -- and even then they are re-checked
        against the physics, because a part that was in the tray a minute ago may not be now.
        """
        keep_done: dict[str, Task] = {}
        for task in self.tasks.values():
            if task.state == DONE and not clear_done:
                keep_done[task.id] = task

        new_ids = {t.id for t in tasks}
        released: list[Task] = []
        dropped: list[str] = []

        for task in self.tasks.values():
            if task.id in new_ids or task.id in keep_done:
                continue
            if task.state in ACTIVE_STATES:
                if task.assigned_robot:
                    self.events.emit(
                        EV.RELEASE, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                        robot_id=task.assigned_robot, task_id=task.id,
                        part_id=task.part_id, bin_id=task.bin_id,
                        detail="plan replaced while in flight; requeued"
                               if tasks else "queue cleared while in flight; abandoned")
                # Requeue an in-flight pick when the plan is *replaced*, but honour an
                # explicit clear. Someone who empties the queue is asking the cell to stop,
                # and requeuing the work would quietly refuse -- and would leave the run loop
                # stepping forever against a queue the interface shows as empty.
                if tasks:
                    released.append(task)
                else:
                    dropped.append(task.id)
                    self._free_robot_holding(task.id)
            else:
                dropped.append(task.id)

        merged = dict(keep_done)

        # A task that was in flight is **requeued**, which is what the event above just
        # claimed and what the docstring has always promised. It used to be announced and
        # then dropped: `released` collected the ids, and `merged` was built from the done
        # tasks and the new ones only. Measured, submitting a plan mid-pick left the two
        # arms holding task ids that no longer existed -- see `step()` for what that did.
        for task in released:
            fresh = Task(id=task.id, part_id=task.part_id, bin_id=task.bin_id,
                         label=task.label)
            fresh.attempts = task.attempts
            fresh.previous_robots = list(task.previous_robots)
            fresh.note = "requeued: the plan was replaced while this pick was in flight"
            merged[fresh.id] = fresh
            self._free_robot_holding(task.id)

        merged.update({
            t.id: Task(id=t.id, part_id=t.part_id, bin_id=t.bin_id, label=t.label)
            for t in tasks
        })
        self.tasks = merged
        self.assignment_costs.clear()

        self.events.emit(
            EV.PLAN, tick=self.control_ticks, sim_time_s=self.sim_time_s,
            detail=(f"{len(tasks)} task(s) submitted"
                    + (f", {len(released)} requeued" if released else "")
                    + (f", {len(dropped)} dropped" if dropped else "")),
            tasks=[t.id for t in tasks], released=released, dropped=dropped,
            keptDone=sorted(keep_done))
        return {"ok": True, "tasksTotal": len(self.tasks),
                "tasksSubmitted": [t.id for t in tasks],
                "released": [t.id for t in released], "dropped": dropped,
                "keptDone": sorted(keep_done)}

    def _free_robot_holding(self, task_id: str) -> None:
        """Put any robot holding ``task_id`` back to idle.

        Without this the robot keeps ``state="working"`` and a ``task_id`` that no longer
        resolves, so `step` skips its idle branch, finds no runner, and quietly does nothing
        -- forever, because `control_ticks` only advances on the path it never reaches. The
        cell looks hung rather than crashed, and nothing is logged.
        """
        for robot in self.robots.values():
            if robot.task_id == task_id:
                robot.task_id = None
                robot.state = "idle"
                robot.note = f"released {task_id}: the plan was replaced"

    def submit_order(self, tasks: list[MoveTask]) -> dict[str, Any]:
        """Add tasks to the queue, keeping existing ones. See :meth:`set_tasks`."""
        merged = [MoveTask(id=t.id, part_id=t.part_id, bin_id=t.bin_id, label=t.label)
                  for t in self.tasks.values() if t.state != DONE]
        merged.extend(tasks)
        return self.set_tasks(merged, clear_done=True)

    def clear_tasks(self) -> dict[str, Any]:
        """Empty the queue. Every part stays where it is; only the work list goes."""
        return self.set_tasks([], clear_done=True)

    def rank_robots(self, task: Task) -> list[dict[str, Any]]:
        """Every healthy robot for this task, cheapest first, with the arithmetic.

        Served by ``/api/tasks`` so the interface can show "why this robot" next to the
        assignment it actually made, and so a judge can check the numbers rather than
        believing them.
        """
        idle = [r.id for r in self.robots.values() if r.state == "idle"]
        move = MoveTask(id=task.id, part_id=task.part_id, bin_id=task.bin_id)
        return [c.to_dict() for c in self.policy.rank(self.scene, idle, move, task.attempts)]

    def _busy_pick_positions(self, exclude: str | None = None) -> list[tuple[float, float]]:
        """Where every *other* working arm is currently reaching for.

        This is what makes the assignment collaborative rather than eight independent
        schedulers. Without it each arm simply takes whatever is nearest itself, and since
        all the arms face the same way down the same bench, the two nearest things to two
        different arms are frequently the same corner.

        Measured: with four tasks on the bench and no such term, robot_1 took ``part_1`` at
        x = 0.290 and robot_2 took ``part_4`` at x = 0.360 -- 70 mm apart. Two arms over one
        patch of table, and the run ended 0/4 with tracking errors of 146-208 mm that read
        exactly like a reachability fault and were not one.
        """
        out: list[tuple[float, float]] = []
        for other in self.robots.values():
            if other.state != "working" or not other.task_id:
                continue
            if exclude is not None and other.id == exclude:
                continue
            task = self.tasks.get(other.task_id)
            if task is None:
                continue
            try:
                part = self.scene.part(task.part_id)
            except Exception:                # noqa: BLE001 - a stale id is not fatal
                continue
            out.append((part.x, part.y))
        return out

    def _assignable(self, robot: RobotStatus | None = None) -> Task | None:
        """The next task this robot should take, by the cost policy.

        **Cheapest for this arm**, not lowest id. That distinction is the whole point of the
        policy: picking by id means every idle robot reaches for the same first task, so two
        arms queue behind one part while another sits idle, and the near arm is as likely to
        take the far part as the far one.

        Ties break on attempts then id, so a run is reproducible and a task that keeps
        failing does not monopolise the cell. A task this arm cannot physically reach is not
        a candidate at all -- not ranked last, because it is not an option.
        """
        candidates = [t for t in self.tasks.values()
                      if t.state == QUEUED and t.assigned_robot is None]
        if not candidates:
            return None

        if robot is None:
            return min(candidates, key=lambda t: (t.attempts, t.id))

        # What the other arms are already reaching for, so this one can stay out of it.
        busy = self._busy_pick_positions(exclude=robot.id)

        scored: list[tuple[float, int, str, Task]] = []
        for task in candidates:
            cost = self.policy.choose(self.scene, [robot.id], self._move_of(task),
                                      task.attempts, busy)
            if cost is None:
                continue                      # out of this arm's working radius
            scored.append((cost.total, task.attempts, task.id, task))
        if not scored:
            return None
        scored.sort(key=lambda s: (s[0], s[1], s[2]))
        return scored[0][3]

    def _move_of(self, task: Task) -> MoveTask:
        return MoveTask(id=task.id, part_id=task.part_id, bin_id=task.bin_id)

    def _idle_robots(self) -> list[RobotStatus]:
        return [r for r in self.robots.values() if r.state == "idle"]

    # -- stepping --------------------------------------------------------

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """Advance every robot by one control tick, then return.

        The caller owns time. This method decides what each robot should be doing this tick
        and writes joint targets; it never calls the simulation itself outside a runner's
        motion helpers, and it never writes a pose.

        Note that ``dt`` is now only used for the runners' own ramp fractions -- the
        simulated clock is read from the cell (see :attr:`sim_time_s`), because a counter
        maintained here would have to be kept in step with how many robots happened to be
        idle, and nothing enforces that.
        """
        for robot in self.robots.values():
            if robot.state == "faulted":
                continue

            if robot.state == "idle":
                task = self._assignable(robot)
                if task is not None:
                    self._assign(task, robot)
                else:
                    # Idle robots still hold their last joint target, so write it and step,
                    # or the arm will sag under gravity while it waits.
                    self._hold_idle(robot)
                    continue

            task = self._task_for(robot.task_id)
            if task is None:
                # A robot holding a task id that is not in the queue. That used to be a
                # silent `continue`, which froze the entire cell with no error anywhere --
                # `control_ticks` only advances below, on the path this skipped.
                #
                # Recover rather than stall: the robot is, functionally, idle, and it is
                # about to be told so. The note is what makes the cause findable if it ever
                # happens again.
                self._hold_idle(robot)
                self.control_ticks += 1
                robot.idle_ticks += 1
                if robot.state != "idle":
                    self.events.emit(
                        EV.RELEASE, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                        robot_id=robot.id, task_id=robot.task_id,
                        detail="task is not in the queue; the arm was returned to idle")
                    robot.state = "idle"
                robot.task_id = None
                robot.note = "released: its task was no longer in the queue"
                continue

            if task.runner is None:
                # Assigned but not started. Let the idle branch have it next tick rather
                # than sitting on it, and keep the clock moving.
                self._hold_idle(robot)
                self.control_ticks += 1
                robot.idle_ticks += 1
                robot.state = "idle"
                continue

            previous = task.runner.phase
            phase = task.runner.advance()
            self.control_ticks += 1
            robot.busy_ticks += 1
            self._read_robot_telemetry(robot)

            if phase != previous:
                self.events.emit(
                    EV.PHASE, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                    robot_id=robot.id, task_id=task.id,
                    part_id=task.part_id, bin_id=task.bin_id,
                    detail=f"{previous} -> {phase}",
                    stage=getattr(task.runner, "approach_stage", None))

            if phase in (DONE, FAILED):
                self._finish(task, robot, phase)

    def _read_robot_telemetry(self, robot: RobotStatus) -> None:
        """Pull the arm's own readings into the robot's status.

        Read straight from the articulation rather than from the runner's bookkeeping, so
        the grip figure on screen is the contact sensors' current output and not a number
        the orchestration remembered.
        """
        try:
            arm = self.cell.arm(robot.id)
        except Exception:                       # noqa: BLE001 - diagnostics must not stop a tick
            return
        robot.grip_n = float(self.cell.grip_force_n(robot.id))
        robot.span_m = float(self.cell.jaw_pad_span_m(robot.id))
        robot.tool_xyz = [float(v) for v in arm.tool_pos]
        robot.joints_rad = [float(v) for v in arm.joints]

    def _hold_idle(self, robot: RobotStatus) -> None:
        """Keep an idle arm parked where it is, rather than letting it drop."""
        arm = self.cell.arm(robot.id)
        arm.command_joints([float(v) for v in arm.joints[:5]])
        self.cell.step()
        self.control_ticks += 1
        robot.idle_ticks += 1
        robot.note = "waiting for work"

    def _task_for(self, task_id: str | None) -> Task | None:
        return self.tasks.get(task_id) if task_id else None

    def _assign(self, task: Task, robot: RobotStatus) -> None:
        """Give a task to a robot, recording who had it before and why this one.

        The cost breakdown is stored as well as logged, so ``/api/state`` can show the
        arithmetic next to the decision rather than asking anyone to trust it.
        """
        previous_owner = task.assigned_robot
        cost = self.policy.choose(self.scene, [robot.id], self._move_of(task), task.attempts,
                                  self._busy_pick_positions(exclude=robot.id))

        task.assigned_robot = robot.id
        task.attempts += 1
        task.state = APPROACHING
        task.note = ""
        task.runner = TaskRunner(self.cell, robot.id, task)
        task.runner.phase = QUEUED
        task.started_tick = self.control_ticks
        robot.state = "working"
        robot.task_id = task.id
        robot.note = ""

        kind = EV.REASSIGN if previous_owner and previous_owner != robot.id else EV.ASSIGN
        if kind == EV.REASSIGN:
            task.previous_robots.append(previous_owner)
            self.reassignments += 1
        if cost is not None:
            self.assignment_costs[task.id] = cost.to_dict()

        self.events.emit(
            kind, tick=self.control_ticks, sim_time_s=self.sim_time_s,
            robot_id=robot.id, task_id=task.id,
            part_id=task.part_id, bin_id=task.bin_id,
            detail=(f"attempt {task.attempts}"
                    + (f", after {previous_owner}" if kind == EV.REASSIGN else "")),
            attempt=task.attempts,
            cost=cost.to_dict() if cost else None)

    def _finish(self, task: Task, robot: RobotStatus, phase: str) -> None:
        """Close out a task, and decide what happens next from what was measured."""
        runner = task.runner
        evidence = runner.evidence if runner else None

        if evidence is not None:
            task.evidence = evidence
            task.history.append({
                "robotId": robot.id,
                "attempt": task.attempts,
                "ok": evidence.ok,
                "criteria": dict(evidence.criteria),
                "measurements": evidence.to_dict()["measurements"],
                "failures": list(evidence.failures),
                "startedTick": task.started_tick,
                "finishedTick": self.control_ticks,
            })

        task.note = "; ".join(runner.failures) if runner else ""
        task.runner = None
        task.finished_tick = self.control_ticks
        robot.task_id = None
        robot.note = ""

        if phase == DONE:
            task.state = DONE
            robot.state = "idle"
            robot.tasks_completed += 1
            self.events.emit(
                EV.DONE, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                robot_id=robot.id, task_id=task.id,
                part_id=task.part_id, bin_id=task.bin_id,
                detail=(f"{task.part_id} -> {task.bin_id} in {task.attempts} attempt(s)"
                        if task.attempts else f"{task.part_id} -> {task.bin_id}"),
                attempt=task.attempts,
                evidence=evidence.to_dict() if evidence else None)
            return

        # Failed. Put the task back on the queue if it has attempts left, otherwise mark it
        # failed for good and let another robot take it if it has one.
        robot.tasks_failed += 1
        will_retry = task.attempts < MAX_ATTEMPTS

        if will_retry:
            task.state = QUEUED
            task.assigned_robot = None
            task.note = (f"attempt {task.attempts} of {MAX_ATTEMPTS} failed: "
                         f"{task.note}")
            robot.state = "idle"
            self.events.emit(
                EV.RETRY, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                robot_id=robot.id, task_id=task.id,
                part_id=task.part_id, bin_id=task.bin_id,
                detail=task.note, attempt=task.attempts)
        else:
            task.state = FAILED
            # Keep the attempt count in the note. Without it the final note reads as a
            # single failure and the interface cannot distinguish a task that failed once
            # from one that used the whole retry budget -- which is exactly the distinction
            # someone looking at a stuck cell needs.
            task.note = (f"failed after {task.attempts} attempts: {task.note}")
            robot.state = "idle"
            self.events.emit(
                EV.FAILED, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                robot_id=robot.id, task_id=task.id,
                part_id=task.part_id, bin_id=task.bin_id,
                detail=task.note, attempt=task.attempts,
                evidence=evidence.to_dict() if evidence else None)

    # -- roster and faults -----------------------------------------------

    def add_robot(self, robot_cfg) -> dict[str, Any]:
        """Refuse, and say why.

        Isaac Lab cannot insert an articulation into a live scene, so the only route is to
        rebuild the whole cell. That route was built
        (:meth:`isaac_env.IsaacCell.rebuild_with_robot`) and **measured to hang**: a clean
        cell built and moved, then ``RemovePrim`` plus a second ``sim.reset()`` in the same
        Kit session never returned -- no exception, just the process at 370 CPU-seconds with
        the last log line still the banner before the call. So it is not wired up.

        The honest answer is that the roster is fixed when the cell is built. Returning
        ``{"ok": false}`` with the reason beats returning ``{"ok": true}`` for a robot that
        was never spawned, which would put a phantom in the interface.

        To change the robot count, relaunch with a different ``--robots``; the task ledger
        survives because it lives here and not in the physics.
        """
        if robot_cfg.id in self.robots:
            return {"ok": False, "reason": f"robot {robot_cfg.id!r} already present"}
        return {"ok": False,
                "reason": ("adding a robot at runtime needs an Isaac Lab cell rebuild, and "
                           "an in-process rebuild hangs on this build (measured: RemovePrim "
                           "+ sim.reset() in the same Kit session never returns). Relaunch "
                           "with a different --robots; the task queue survives, the "
                           "simulation state does not.")}

    def remove_robot(self, robot_id: str) -> dict[str, Any]:
        """Refuse, and say why. See :meth:`add_robot` -- same measured reason."""
        if robot_id not in self.robots:
            return {"ok": False, "reason": f"no robot with id {robot_id!r}"}
        if len(self.robots) <= 1:
            return {"ok": False, "reason": "the last robot cannot be removed"}
        return {"ok": False,
                "reason": ("removing a robot at runtime needs an Isaac Lab cell rebuild, "
                           "and an in-process rebuild hangs on this build (measured). "
                           "Relaunch with a different --robots.")}

    def release_for_relaunch(self) -> dict[str, Any]:
        """Put every in-flight task back on the queue, in preparation for a relaunch.

        The supported way to change the roster: requeue the ledger, exit, relaunch with a
        different ``--robots``. The parts do not survive that -- they respawn at their
        configured positions -- so a task mid-flight is no longer true and must be requeued
        rather than resumed.

        This is what an in-process rebuild would have needed too, which is why losing the
        physical state was never the part that made that approach attractive.
        """
        released: list[str] = []
        for task in self.tasks.values():
            if task.state not in ACTIVE_STATES:
                continue
            if task.assigned_robot:
                task.previous_robots.append(task.assigned_robot)
                self.reassignments += 1
            task.assigned_robot = None
            task.state = QUEUED
            task.runner = None
            task.note = "released for relaunch: the simulation state does not survive"
            released.append(task.id)
        self.events.emit(
            EV.SCENE, tick=self.control_ticks, sim_time_s=self.sim_time_s,
            detail=f"relaunch planned; {len(released)} task(s) requeued",
            released=released)
        return {"ok": True, "released": released,
                "note": "relaunch with --robots N to change the roster"}

    def inject_failure(self, robot_id: str) -> dict[str, Any]:
        """Fault a robot, as a UI button or a test would.

        The robot stops being eligible for work and whatever it was holding goes back on
        the queue. Nothing about the simulation is touched: this is orchestration state, so
        a fault here cannot be confused with the arm breaking.
        """
        robot = self.robots.get(robot_id)
        if robot is None:
            return {"ok": False, "reason": f"no robot with id {robot_id!r}"}

        released: list[str] = []
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
            self.events.emit(
                EV.REASSIGN, tick=self.control_ticks, sim_time_s=self.sim_time_s,
                task_id=task.id, part_id=task.part_id, bin_id=task.bin_id,
                detail=f"{robot_id} faulted; requeued for another arm",
                freedBy=robot_id)

        robot.state = "faulted"
        robot.task_id = None
        robot.note = "failure injected"
        self.failures_injected += 1
        self.events.emit(
            EV.FAULT, tick=self.control_ticks, sim_time_s=self.sim_time_s,
            robot_id=robot_id,
            detail=(f"out of service; released {len(released)} task(s)"
                    if released else "out of service"),
            released=released)
        return {"ok": True, "robotId": robot_id, "released": released,
                "state": robot.state}

    def repair_robot(self, robot_id: str) -> dict[str, Any]:
        """Bring a faulted robot back into the pool."""
        robot = self.robots.get(robot_id)
        if robot is None:
            return {"ok": False, "reason": f"no robot with id {robot_id!r}"}
        robot.state = "idle"
        robot.note = ""
        self.events.emit(
            EV.REPAIR, tick=self.control_ticks, sim_time_s=self.sim_time_s,
            robot_id=robot_id, detail="back in service")
        return {"ok": True, "robotId": robot_id, "state": robot.state}

    def fault_all(self, robot_ids: list[str] | None = None) -> dict[str, Any]:
        """Fault several robots at once.

        The interface's "what happens if the cell loses an arm" button. Faulting every arm
        is refused rather than allowed to deadlock the run: with nothing healthy the queue
        can never advance, and a demo that quietly hangs is worse than one that says the
        request does not make sense.
        """
        wanted = list(robot_ids) if robot_ids is not None else list(self.robots)
        healthy = [r for r in wanted
                   if (r in self.robots and self.robots[r].state != "faulted")]
        if not healthy:
            return {"ok": False, "reason": "no healthy robot to fault"}
        if len(healthy) == len(self.robots):
            return {"ok": False,
                    "reason": ("faulting every robot would leave nothing able to work; "
                               "keep at least one in service")}
        results = [self.inject_failure(r) for r in healthy]
        return {"ok": True, "faulted": healthy,
                "released": [t for r in results for t in r.get("released", [])]}