"""Demo 12 — is this grasp real?

Every question this module answers is answered by **reading the simulation**, never by
trusting a command. A gripper that has been told to close has not necessarily closed; a
pad touching a part has not necessarily squeezed it; and a part that has been lifted has
not necessarily been carried. Those are three different claims and each one is measured
here separately.

The four acceptance criteria for Demo 12's gripper, and what each is allowed to mean:

1. **Tracking** -- ``tool_error_m < 1 mm`` at the grasp pose. Read from the articulation's
   own ``body_pos_w``, compared against the commanded point.
2. **Span** -- pads >= 36 mm open, < 18 mm shut. Read from the pad bodies' separation
   projected on the jaw's own separation axis.
3. **Grip** -- ``grip_force_n > 0`` while closed on the part. Read from the per-pad contact
   sensors, i.e. from PhysX's own solver output.
4. **Lift and release** -- the part rises with the jaw, and after release it is still in
   the target bin and at rest. Read from the part's own rigid-body state.

Criterion 4 is the only one that proves a grasp happened. Criteria 1-3 can all pass while
the part is still sitting on the bench.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import materials as M

# ---------------------------------------------------------------------------
# Thresholds. Each one is a stated criterion, not a tuned-to-pass number.
# ---------------------------------------------------------------------------

TRACKING_TOLERANCE_M = 0.001          # criterion 1
SPAN_OPEN_MIN_M = 0.036               # criterion 2
SPAN_SHUT_MAX_M = 0.018               # criterion 2

# A squeeze below this is not carrying anything. A 30 mm ABS cube weighs 0.276 N and the
# pads are rubber at mu 0.9, so the threshold is set well under what is physically needed
# rather than at it: the point is to catch "the pads closed on air", not to fine-tune.
GRIP_MIN_N = 0.0

# How high a part must rise for the lift to count. The lift itself is 100 mm; a part that
# follows it must follow it essentially all the way, so this is a loose floor that exists
# to reject a part that merely twitches.
LIFT_MIN_RISE_M = 0.050

# Ticks to hold the jaws shut before measuring the squeeze. S2: at 50 the jaws were still
# being squeezed when the measurement was taken; 90 (1.5 s at 60 Hz) is where the linkage
# has actually reached its commanded position.
GRIP_SETTLE_TICKS = 90

# Ticks to hold the jaws open after a release before the arm retreats.
RELEASE_SETTLE_TICKS = 25

# Ticks to let a placed part come to rest before its position is believed. A 10 g PET ball
# on melamine takes noticeably longer than a cube.
REST_SETTLE_TICKS = 60

# How far above the part's centre the arm stages before descending. Generous on purpose:
# this is a safe parking height where the tool is already pointing down, so the pad offset
# can be measured, and it is nowhere near the bench.
STAGING_RISE_M = 0.120

# How many times to re-solve the grasp height and re-approach. Descending changes the arm
# pose, which changes the jaw's tilt, which changes the pad offset by a fraction of a
# millimetre. Two passes is enough for that to settle; the third would be noise.
GRASP_HEIGHT_REFINEMENTS = 2

# Per-tick joint cap while descending onto a part, and how long the descent takes.
#
# 0.012 rad/tick is 0.72 rad/s -- about a sixth of the servo's real 4.7 rad/s. The point is
# not speed, it is that the damped step stays small enough for the solver to re-aim the
# tool's lateral position on every single tick as it comes down.
DESCENT_MAX_STEP_RAD = 0.012
DESCENT_TICKS = 260

# The lift is rate-limited the same way and for the same reason: a fast lift levers a
# grasped part sideways out of the jaws.
LIFT_MAX_STEP_RAD = 0.012
LIFT_TICKS = 260

# Ticks to raise the tool clear of the tray after a release.
RETREAT_TICKS = 120

# The transfer from part to bin. Rate-limited for the same reason as the lift: a grasped
# part is held by friction, and a fast transfer spends that budget. This move covers about
# 100 mm laterally plus 85 mm vertically, so it gets more ticks than either.
TRANSFER_MAX_STEP_RAD = 0.012
TRANSFER_TICKS = 520

# How far the shut span may differ from the part's own width and still count as "the pads
# are on both faces". 2 mm is inside the alignment tolerance times the part's half-width, so
# anything beyond it means a pad is on a corner.
ON_PART_SPAN_TOLERANCE_M = 0.002

# Ticks to hold the mechanism open and shut, in free space, to measure its own travel.
FREE_SPAN_TICKS = 40

# Ticks to hold the pads open once they are level with the part, before closing. Long
# enough that the prismatic joints have actually reached their commanded open span rather
# than being measured mid-stroke.
PRE_GRASP_SETTLE_TICKS = 40


@dataclass
class GraspResult:
    """One attempt at one grasp, with every measurement that went into it."""

    ok: bool
    criteria: dict[str, bool] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "criteria": self.criteria,
            "failures": self.failures,
            "measurements": self.measurements,
        }


@dataclass
class GraspEvidence:
    """Everything one pick had to show, and whether it actually showed it.

    **This is the single definition of "the grasp was real".** Both :class:`GraspProbe` and
    the live ``demo12.orchestrator.TaskRunner`` build one of these and obey it, which is the
    only reason a task the demo reports as done and a grasp the test suite certifies are the
    same claim.

    They used not to be, and the gap was the demo's largest hole. ``GraspProbe.run`` judged a
    grasp on eight criteria. The live runner judged the same grasp on two -- ``grip_n > 0.0``
    and "in the bin and at rest" -- and never checked the other six. Concretely, the two
    documented ways this demo used to cheat a success both passed the live runner:

    * **The punt** (``docs/PHYSICS.md`` section 4). Pads landing on a corner instead of two
      faces stalled the span at 40.6 mm, shoved the part 36 mm along the axis, and reported
      **103 N** of contact. 103 > 0, so the runner accepted it. ``pads_on_faces`` is the
      criterion that rejects it and the runner never evaluated it.
    * **A part that never left the table.** The runner judged only where the part ended up.
      Anything that ended inside the tray and at rest counted, including a part another
      robot had knocked in, or one a fast close had punted across the table. ``part_rose``
      and ``carried`` are the criteria that require *this robot* to have done the carrying.

    Every field here is a reading. None of them is a command, and nothing in this module can
    make a grasp succeed that the physics did not produce.
    """

    part_id: str = ""
    bin_id: str = ""
    robot_id: str = ""

    # -- what the arm did ---------------------------------------------------
    tool_error_m: float = 0.0
    span_on_part_m: float = 0.0
    grip_force_n: float = 0.0
    grip_force_during_lift_n: float = 0.0
    part_drift_on_close_m: float = 0.0

    # -- what the part did --------------------------------------------------
    part_width_m: float = 0.0
    part_rise_m: float = 0.0
    part_carried_to_bin: bool = False
    part_inside_bin: bool = False
    part_resting: bool = False

    # -- the verdict --------------------------------------------------------
    criteria: dict[str, bool] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.criteria) and all(self.criteria.values())

    @property
    def grip_required_n(self) -> float:
        """Normal force the friction cone needs for this part.

        ``2 * mu * N >= m * g`` with a 2x safety factor, which is what
        :func:`required_grip_n` computes. Compared against the *sum* of both pads' contact
        forces, which is what ``IsaacCell.grip_force_n`` reports, so the two are the same
        physical quantity and not off by a factor of two.
        """
        return getattr(self, "_grip_required_n", 0.0)

    @property
    def grip_margin(self) -> float:
        """Measured grip as a multiple of the friction cone's requirement.

        Reported next to every grasp so a pass can be judged as comfortable or marginal
        rather than merely true or false. The demo's own two reference grasps sit at ~200x;
        a value near 1.0 would be a grip about to slip.
        """
        need = self.grip_required_n
        if need <= 0.0:
            return float("inf")
        return self.grip_force_n / need

    @property
    def measurements(self) -> dict[str, Any]:
        """The numbers behind the verdict, as plain floats.

        A property rather than a field so the derived quantities -- the friction cone's
        requirement and the margin over it -- cannot disagree with the raw readings they
        came from. ``to_dict`` is built from this.
        """
        margin = self.grip_margin
        return {
            "toolErrorM": round(self.tool_error_m, 6),
            "spanOnPartM": round(self.span_on_part_m, 6),
            "partWidthM": round(self.part_width_m, 6),
            "gripForceN": round(self.grip_force_n, 4),
            "gripRequiredN": round(self.grip_required_n, 4),
            "gripMargin": None if margin == float("inf") else round(margin, 1),
            "gripForceDuringLiftN": round(self.grip_force_during_lift_n, 4),
            "partDriftOnCloseM": round(self.part_drift_on_close_m, 6),
            "partRiseM": round(self.part_rise_m, 6),
            "partCarriedToBin": self.part_carried_to_bin,
            "partInBin": self.part_inside_bin,
            "partResting": self.part_resting,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "partId": self.part_id,
            "binId": self.bin_id,
            "robotId": self.robot_id,
            "criteria": dict(self.criteria),
            "failures": list(self.failures),
            "measurements": self.measurements,
        }


def evaluate_grasp(
    *,
    part_id: str,
    bin_id: str,
    robot_id: str,
    mass_kg: float,
    tool_error_m: float,
    part_width_m: float,
    span_on_part_m: float,
    grip_force_n: float,
    part_rise_m: float,
    part_carried_to_bin: bool,
    part_inside_bin: bool,
    part_resting: bool,
    grip_force_during_lift_n: float = 0.0,
    part_drift_on_close_m: float = 0.0,
) -> GraspEvidence:
    """Turn one pick's readings into a verdict. The only place a grasp is judged.

    Seven criteria, each a statement about a different thing, so a failure names what
    actually went wrong instead of collapsing to "grasp failed":

    1. ``tracking``     -- the tool was where it was told to be when it closed.
    2. ``pads_on_faces`` -- the pads stalled at the part's own width, on both flat faces.
    3. ``grip``          -- the squeeze exceeded what the friction cone needs for this mass.
    4. ``part_rose``      -- the part rose with the jaw, so it was carried and not nudged.
    5. ``carried``        -- it travelled to the destination still in the jaws.
    6. ``part_in_bin``    -- it ended fully inside the cavity.
    7. ``part_resting``   -- and it had stopped moving.

    (2), (4) and (5) are the three that the live runner used to omit.
    """
    required_n = required_grip_n(mass_kg)

    e = GraspEvidence(
        part_id=part_id, bin_id=bin_id, robot_id=robot_id,
        tool_error_m=tool_error_m,
        span_on_part_m=span_on_part_m,
        grip_force_n=grip_force_n,
        grip_force_during_lift_n=grip_force_during_lift_n,
        part_drift_on_close_m=part_drift_on_close_m,
        part_width_m=part_width_m,
        part_rise_m=part_rise_m,
        part_carried_to_bin=part_carried_to_bin,
        part_inside_bin=part_inside_bin,
        part_resting=part_resting,
    )
    e._grip_required_n = required_n          # noqa: SLF001 - derived, not a constructor arg

    c = e.criteria
    c["tracking"] = tool_error_m < TRACKING_TOLERANCE_M
    # The pads have to have stalled at the part's own width, on both flat faces. A span still
    # near the free shut figure means they closed on something thinner than the part.
    c["pads_on_faces"] = (
        span_on_part_m > SPAN_SHUT_MAX_M
        and abs(span_on_part_m - part_width_m) <= ON_PART_SPAN_TOLERANCE_M
    )
    # The friction cone, not "greater than zero". 0 N is the floor for "the pads closed on
    # air"; anything above it has to actually hold this part's weight with a margin.
    c["grip"] = grip_force_n >= max(GRIP_MIN_N, required_n)
    # The part rose with the jaw. Without this, a part that was knocked or punted into the
    # tray by anything at all would pass criteria 6 and 7.
    c["part_rose"] = part_rise_m >= LIFT_MIN_RISE_M
    c["carried"] = bool(part_carried_to_bin)
    c["part_in_bin"] = bool(part_inside_bin)
    c["part_resting"] = bool(part_resting)

    if not c["tracking"]:
        e.failures.append(
            f"tracking {tool_error_m * 1000:.2f} mm >= {TRACKING_TOLERANCE_M * 1000:.1f} mm")
    if not c["pads_on_faces"]:
        e.failures.append(
            f"span on part {span_on_part_m * 1000:.1f} mm is not the part's "
            f"{part_width_m * 1000:.1f} mm width, so the pads are not on both faces")
    if not c["grip"]:
        e.failures.append(
            f"grip force {grip_force_n:.3f} N below the {required_n:.3f} N the friction cone "
            f"needs for {mass_kg * 1000:.0f} g")
    if not c["part_rose"]:
        e.failures.append(
            f"part rose only {part_rise_m * 1000:.1f} mm on a 100 mm lift "
            f"(needs {LIFT_MIN_RISE_M * 1000:.0f} mm): it was not carried")
    if not c["carried"]:
        e.failures.append("the part was not over the destination while still held")
    if not c["part_in_bin"]:
        e.failures.append("the part did not end fully inside the destination cavity")
    if not c["part_resting"]:
        e.failures.append("the part was still moving when the task was judged")

    return e


class GraspProbe:
    """Runs one pick-lift-place against a live cell and reports what actually happened.

    Constructed by :meth:`demo12.isaac_env.IsaacCell` consumers, or directly in a probe:

        probe = GraspProbe(cell, robot_id)
        result = probe.run(part_id, bin_id)
        assert result.ok, result.failures

    Only joint targets are ever sent. No pose is written, no part is attached to anything,
    and nothing in here can make a grasp succeed that the physics did not produce.
    """

    def __init__(self, cell, robot_id: str) -> None:
        self.cell = cell
        self.robot_id = robot_id
        self.arm = cell.arm(robot_id)

    # -- helpers ---------------------------------------------------------

    def _nearest_axis_angle(self) -> float:
        """Which world axis to square the pads up to: whichever is closest.

        The parts are laid out axis-aligned, so presenting the pads along world +X or +Y
        both put a flat pad face against a flat part face. Choosing the nearer one keeps
        the wrist's travel small, which keeps the alignment cheap and keeps the pads well
        inside the joint's range.
        """
        current = self.arm.pad_axis_angle_rad()
        candidates = (0.0, math.pi / 2.0)
        best = min(candidates,
                   key=lambda a: abs((a - current + math.pi) % (2.0 * math.pi) - math.pi))
        return best

    def _hold(self, target, ticks: int, jaws_open: bool, max_step_rad: float) -> None:
        """Hold a point for a while, re-solving IK against the measured pose each tick."""
        self.arm.set_jaws(jaws_open)
        for _ in range(ticks):
            self.arm.command_joints(self.arm.solve_ik(target, max_step_rad))
            self.cell.step()

    def _approach(self, target, max_ticks: int = 400) -> int:
        """Move to a point with the pads open, arriving gently.

        Used for free-space motion, where a fast cap is fine. Never use it for the last
        descent onto a part -- see :meth:`_descend`.
        """
        self.arm.set_jaws(True)
        ticks = 0
        for ticks in range(1, max_ticks + 1):
            self.arm.command_joints(self.arm.solve_ik(target, 0.078))
            self.cell.step()
            if self.arm.tool_error_m(target) <= TRACKING_TOLERANCE_M:
                break
        return ticks

    def _descend(self, xy, z_from: float, z_to: float, ticks: int,
                 jaws_open: bool = True, max_step_rad: float = DESCENT_MAX_STEP_RAD) -> int:
        """Lower the tool along a fixed vertical line, re-aiming at the part every tick.

        A straight ``_approach`` onto the grasp point does not work, and the reason is
        geometric rather than a tuning problem. The mechanism's open span is 37 mm and the
        part is 30 mm, so each pad clears its face by 3.5 mm. That is a tight corridor to
        descend a 100 mm staging drop through, and at the arm's full 0.078 rad/tick the
        damped step is large enough that the tool wanders sideways while it comes down.

        Measured, before this method existed: the tool drifted 6.4 mm in x and 9.0 mm in y
        on the way down, the left pad landed on the part's **top face** rather than beside
        it and stalled at 60.3 N -- exactly ``materials.JAW_PAD_EFFORT_LIMIT_N`` -- while
        the right pad showed 0.0 N. The one-sided jam twisted ``wrist_roll`` to 0.089 rad,
        which rotated the separation axis further into the part. The jaws then closed to
        14.94 mm on the part's upper corner, the grip read 31 N, and on the lift the part
        did not move at all: the pads were shut on nothing.

        So the descent is rate-limited and lateral error is re-corrected every tick. The
        tool follows a straight line instead of an arc, and the pads pass down the part's
        sides where they belong.
        """
        self.arm.set_jaws(jaws_open)
        for k in range(1, ticks + 1):
            z = z_from + (z_to - z_from) * (k / ticks)
            self.arm.command_joints(self.arm.solve_ik((xy[0], xy[1], z), max_step_rad))
            self.cell.step()
        return ticks

    # -- the sequence ---------------------------------------------------

    def run(self, part_id: str, bin_id: str) -> GraspResult:
        """Approach, close, lift, place, release, and verify. Returns what was measured."""
        m: dict[str, Any] = {}
        criteria: dict[str, bool] = {}
        failures: list[str] = []

        part_cfg = self.cell.part_cfgs[part_id]
        bin_cfg = self.cell.bins[bin_id]

        # -- 0. starting state, measured -------------------------------------
        start = self.cell.part_state(part_id)
        m["part_start"] = {"x": start.x, "y": start.y, "z": start.z}

        # -- 1. approach with the pads open ----------------------------------
        #
        # The descent is staged, and it has to be.
        #
        # ``ArmController.tool_z_for_part_centre`` solves for the tool height from the
        # pads' *measured* offset below the tool, which is only meaningful once the tool is
        # pointing down. Asking it that question while the arm is still folded at its home
        # pose returns a garbage offset, and the first version of this probe did exactly
        # that: it asked at home, got 0.12 m, and commanded a descent 120 mm below the
        # tabletop. Measured: 147.8 mm of tracking error and a part that never moved.
        #
        # So: go to a staging point well above the part first, let the arm settle into a
        # tool-down pose, measure the offset there, and only then compute the real grasp
        # height. Two refinement passes, because descending changes shoulder_lift and
        # elbow_flex, which changes the tilt, which changes the offset by a little.
        staging = (start.x, start.y, start.z + STAGING_RISE_M)
        self._approach(staging, max_ticks=500)

        # Square the pads up to the part before descending.
        #
        # Whichever of the two world axes the pad axis is already closest to, so the wrist
        # takes the short way round. Rolling does not move the tool origin, so this costs
        # nothing in tracking -- but it is what turns a punt into a grip.
        target_angle = self._nearest_axis_angle()
        ticks = self.arm.align_pad_axis(target_angle)
        m["pad_axis_aligned_to_deg"] = round(math.degrees(target_angle), 2)
        m["pad_axis_remaining_deg"] = round(
            math.degrees(self.arm.pad_axis_angle_rad() - target_angle), 3)
        m["pad_axis_align_ticks"] = ticks
        m["roll_sign"] = self.arm._roll_sign
        m["roll_gain_measured"] = round(self.arm._roll_gain_measured, 4)

        # Measure the mechanism's own travel here, in free space with nothing between the
        # pads.
        #
        # This has to happen up here rather than at the end of the close. With a 30 mm part
        # between them the pads *cannot* reach their shut span -- they stall at the part's
        # width, which is correct behaviour and not a fault. Measured: a shut span of
        # 29.40 mm on a 30 mm cube, which is the grip working. Judging the mechanism's
        # travel against that number would mean judging a correct grip as a broken
        # linkage, so the free span and the on-part span are separate measurements and
        # criterion 2 uses the free one.
        self.arm.set_jaws(False)
        self._hold(staging, FREE_SPAN_TICKS, jaws_open=False, max_step_rad=0.02)
        m["span_free_shut_m"] = round(self.cell.jaw_pad_span_m(self.robot_id), 6)
        self.arm.set_jaws(True)
        self._hold(staging, FREE_SPAN_TICKS, jaws_open=True, max_step_rad=0.02)
        m["span_free_open_m"] = round(self.cell.jaw_pad_span_m(self.robot_id), 6)

        # Measure the pad offset from the staging pose, where the tool is pointing down and
        # the number means something.
        offset = self.arm.pad_offset_below_tool_m()
        grasp_z = start.z + offset
        self._descend((start.x, start.y), staging[2], grasp_z, DESCENT_TICKS)

        # Descending changed the arm pose, hence the jaw's tilt, hence the offset. Settle it
        # with one slow re-approach onto the part's level before calling the tracking good.
        offset = self.arm.pad_offset_below_tool_m()
        grasp_z = start.z + offset
        self._hold((start.x, start.y, grasp_z), PRE_GRASP_SETTLE_TICKS,
                   jaws_open=True, max_step_rad=DESCENT_MAX_STEP_RAD)

        grasp_target = (start.x, start.y, grasp_z)
        m["staging_target"] = list(staging)
        m["grasp_target"] = list(grasp_target)
        m["pad_offset_below_tool_m"] = round(offset, 6)
        m["descend_ticks"] = DESCENT_TICKS

        m["tool_error_m"] = round(self.arm.tool_error_m(grasp_target), 6)
        m["span_open_m"] = round(self.cell.jaw_pad_span_m(self.robot_id), 6)

        criteria["tracking"] = m["tool_error_m"] < TRACKING_TOLERANCE_M
        criteria["span_open"] = m["span_free_open_m"] >= SPAN_OPEN_MIN_M
        criteria["span_shut_free"] = m["span_free_shut_m"] < SPAN_SHUT_MAX_M
        if not criteria["tracking"]:
            failures.append(
                f"tracking {m['tool_error_m'] * 1000:.2f} mm >= "
                f"{TRACKING_TOLERANCE_M * 1000:.1f} mm")
        if not criteria["span_open"]:
            failures.append(
                f"span open {m['span_free_open_m'] * 1000:.1f} mm < "
                f"{SPAN_OPEN_MIN_M * 1000:.1f} mm")
        if not criteria["span_shut_free"]:
            failures.append(
                f"span shut (free) {m['span_free_shut_m'] * 1000:.1f} mm >= "
                f"{SPAN_SHUT_MAX_M * 1000:.1f} mm")

        # How far apart the pads actually are from the part's faces, which is what decides
        # whether the close straddles it or shoves it.
        pad_l, pad_r = self.cell.jaw_pad_positions(self.robot_id)
        m["pad_l_world"] = [round(float(v), 5) for v in pad_l]
        m["pad_r_world"] = [round(float(v), 5) for v in pad_r]

        # -- 2. close on the part --------------------------------------------
        self.arm.set_jaws(False)
        self._hold(grasp_target, GRIP_SETTLE_TICKS, jaws_open=False, max_step_rad=0.02)
        m["span_on_part_m"] = round(self.cell.jaw_pad_span_m(self.robot_id), 6)
        m["grip_force_n"] = round(self.cell.grip_force_n(self.robot_id), 4)
        m["pad_contacts"] = self.cell.pad_contact_forces(self.robot_id)

        # The on-part span and the squeeze are recorded here but *judged* at the end, by
        # ``evaluate_grasp`` -- the same function the live orchestrator runs. A probe and a
        # demo that scored a grasp by different rules could disagree about whether it was
        # real, and then a test would certify something the demo reported as a failure.
        on_part_width = part_cfg.grasp_width_m
        m["part_grasp_width_m"] = on_part_width

        # The part must still be where it was. A close that shoves the part away has not
        # grasped it, whatever the pads report.
        after_close = self.cell.part_state(part_id)
        moved = math.dist((after_close.x, after_close.y),
                          (start.x, start.y))
        m["part_drift_on_close_m"] = round(moved, 6)
        m["part_z_after_close"] = after_close.z
        if moved > 0.010:
            failures.append(
                f"closing pushed the part {moved * 1000:.1f} mm away; that is not a grasp")

        # -- 3. lift 100 mm --------------------------------------------------
        # Rate-limited for the same reason the descent is. A fast lift drags a grasped part
        # sideways against friction, and a part levered out of the jaws on the way up is a
        # failure that looks like a successful lift until the part's height is checked.
        lift_target = (start.x, start.y, grasp_target[2] + 0.100)
        self._descend((start.x, start.y), grasp_target[2], lift_target[2],
                      LIFT_TICKS, jaws_open=False, max_step_rad=LIFT_MAX_STEP_RAD)
        m["lift_ticks"] = LIFT_TICKS

        lifted = self.cell.part_state(part_id)
        rise = lifted.z - start.z
        m["part_rise_m"] = round(rise, 6)
        m["part_after_lift"] = {"x": lifted.x, "y": lifted.y, "z": lifted.z}
        m["grip_force_during_lift_n"] = round(self.cell.grip_force_n(self.robot_id), 4)

        # -- 4. place and release -------------------------------------------
        # Drop from a little above the rim so the part falls into the cavity rather than
        # being pushed over the wall.
        release_z = bin_cfg.cavity_floor_z_m + part_cfg.size_m[2] / 2.0 + 0.030
        place_target = (bin_cfg.x, bin_cfg.y, release_z)

        # Rate-limited, like the descent and the lift. This move used to run at the arm's
        # full speed, and the part fell out of the jaws on it: measured, the lift carried
        # the part to z = 0.1548 with 60.8 N still on the pads, and by the release the part
        # was back on the table at z = 0.0559 and 26.5 mm from where it started. Nothing
        # had failed -- the grip was fine, the part had simply been accelerated out of it.
        # A grasped part is held by friction, and friction is a budget the acceleration
        # spends.
        self._descend((bin_cfg.x, bin_cfg.y), lift_target[2], release_z,
                      TRANSFER_TICKS, jaws_open=False, max_step_rad=TRANSFER_MAX_STEP_RAD)

        above = self.cell.part_state(part_id)
        m["part_before_release"] = {"x": above.x, "y": above.y, "z": above.z}
        m["part_carried_to_bin"] = math.dist((above.x, above.y),
                                             (bin_cfg.x, bin_cfg.y)) < 0.020

        self.arm.set_jaws(True)
        self._hold(place_target, RELEASE_SETTLE_TICKS, jaws_open=True, max_step_rad=0.02)

        # Retreat straight up before judging the part's resting place, so the retreating
        # arm is not still hovering over the tray and knocking it.
        retreat = (bin_cfg.x, bin_cfg.y, release_z + 0.080)
        self._descend((bin_cfg.x, bin_cfg.y), place_target[2], retreat[2],
                      RETREAT_TICKS, jaws_open=True, max_step_rad=LIFT_MAX_STEP_RAD)

        self._hold(retreat, REST_SETTLE_TICKS, jaws_open=True, max_step_rad=0.02)

        # -- 5. verify from the part's own state ----------------------------
        final = self.cell.part_state(part_id)
        inside = bin_cfg.contains_footprint(final.x, final.y, part_cfg.footprint_radius_m)
        m["part_final"] = {"x": final.x, "y": final.y, "z": final.z}
        m["part_resting"] = final.resting
        m["part_inside_bin"] = inside
        m["part_lin_speed"] = final.lin_speed

        # -- 6. one verdict, shared with the live orchestrator ---------------
        evidence = evaluate_grasp(
            part_id=part_id, bin_id=bin_id, robot_id=self.robot_id,
            mass_kg=part_cfg.mass_kg,
            tool_error_m=m["tool_error_m"],
            part_width_m=on_part_width,
            span_on_part_m=m["span_on_part_m"],
            grip_force_n=m["grip_force_n"],
            part_rise_m=rise,
            part_carried_to_bin=m["part_carried_to_bin"],
            part_inside_bin=inside,
            part_resting=final.resting,
            grip_force_during_lift_n=m["grip_force_during_lift_n"],
            part_drift_on_close_m=m["part_drift_on_close_m"],
        )
        criteria.update(evidence.criteria)
        failures.extend(evidence.failures)
        m["grip_required_n"] = round(evidence.grip_required_n, 4)
        m["grip_margin"] = (None if evidence.grip_margin == float("inf")
                            else round(evidence.grip_margin, 1))
        m["evidence"] = evidence.to_dict()

        return GraspResult(
            ok=all(criteria.values()),
            criteria=criteria,
            measurements=m,
            failures=failures,
        )


def required_grip_n(part_mass_kg: float) -> float:
    """Normal force the pads need for this part, from the friction cone.

    Two pads, so ``2 * mu * N >= m * g``, with a 2x safety factor so the part is not held
    exactly at the edge of slipping -- carrying it also has to pay for the arm's own
    acceleration. Reported next to the measured grip so a pass can be judged as real or
    marginal rather than just true or false.
    """
    return M.required_normal_force_n(part_mass_kg, M.RUBBER_PAD.static_friction)