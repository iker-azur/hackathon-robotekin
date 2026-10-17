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

        # The on-part span is evidence about the *grasp*, not about the linkage: the pads
        # have to have stalled at the part's width, on both flat faces. A span still near
        # the free shut figure means they closed on something thinner than the part -- a
        # corner, or nothing.
        on_part_width = part_cfg.grasp_width_m
        m["part_grasp_width_m"] = on_part_width
        criteria["pads_on_faces"] = (
            m["span_on_part_m"] > SPAN_SHUT_MAX_M
            and abs(m["span_on_part_m"] - on_part_width) <= ON_PART_SPAN_TOLERANCE_M
        )
        criteria["grip"] = m["grip_force_n"] > GRIP_MIN_N
        if not criteria["pads_on_faces"]:
            failures.append(
                f"span on part {m['span_on_part_m'] * 1000:.1f} mm is not the part's "
                f"{on_part_width * 1000:.1f} mm width, so the pads are not on both faces")
        if not criteria["grip"]:
            failures.append(f"grip force {m['grip_force_n']:.4f} N <= {GRIP_MIN_N} N")

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

        criteria["part_rose"] = rise >= LIFT_MIN_RISE_M
        if not criteria["part_rose"]:
            failures.append(
                f"part rose only {rise * 1000:.1f} mm on a 100 mm lift "
                f"(needs {LIFT_MIN_RISE_M * 1000:.0f} mm)")

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

        criteria["part_in_bin"] = inside
        criteria["part_resting"] = final.resting
        if not criteria["part_in_bin"]:
            failures.append(
                f"part ended at ({final.x:.4f}, {final.y:.4f}), not inside "
                f"{bin_id} (centre {bin_cfg.x:.4f}, {bin_cfg.y:.4f})")
        if not criteria["part_resting"]:
            failures.append(
                f"part still moving at the end: {final.lin_speed:.4f} m/s")

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