"""Turn a planner Plan into robot motion. The only part of demo 10 that needs Isaac.

Extracted from ``demos/01_simulation_basics/05_cube_pick_and_place.py``, which is the
reference: that demo's ``pick.log`` ends in ``RESULT: SUCCESS`` for moving the shipped cube
10 cm in x, and ``tools/check_executor.py`` reproduces exactly that before any of this is
pointed at a camera-derived pose. The differences from demo 05 are deliberate and each is
noted below.

The action vector
-----------------
Eight numbers: the first seven are a gripper pose **in the robot's base frame**, the last is
the gripper command. That is the contract of the ``so101_state_machine`` teleop device, whose
arm term is a ``DifferentialInverseKinematicsAction`` with ``command_type="pose"``, and whose
gripper term is a ``BinaryJointPositionAction`` that opens on a positive value and closes on a
negative one. Demo 05's own comment: *"the target is in the robot base frame, not world"*.

Grasping
--------
Two mechanisms, because one is not enough.

A real close is attempted first. Then the lift is watched: if the object did not move with the
gripper, the grasp is re-made kinematically by writing the object's pose from the gripper
frame each step, exactly as demo 05 does. Demo 05 documents why it needs this at all: *"Real
friction grasping of the 3 cm cube was unreliable in this sim, so this keeps the demo about
moving the cube, not about tuning contacts."* The difference here is that the fallback is
*detected and reported* rather than assumed, so the run's report says which grasps were real.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

import planner

# Where a held object's centre sits in the gripper body frame. Demo 05 measures this off the
# finger meshes; it is the same arm, so the same numbers apply.
HOLD = (0.007, 0.0, -0.092)

# Extra height above the grasp pose while travelling, from demo 05.
ABOVE = 0.06

# The IK action's gripper is binary: a positive value opens it to 1.0 rad and a negative one
# closes it. The closing angle is NOT set here - it belongs to the BinaryJointPositionAction
# term configured by the LeIsaac Hub module that env_loader downloads, and measured as 0.4 rad
# by tools/probe_gripper.py. An earlier version of this file declared JAW_CLOSED_RAD = 0.15
# and nothing ever read it, which is how a README came to claim the jaws shut to 0.15 rad.
# They do not.
OPEN, CLOSE = 1.0, -1.0

# Physics steps per planner action. Demo 05's waypoint table, which is known to work on this
# arm: it spends 150 steps (2.5 s) travelling and 40 (0.7 s) opening the jaws.
STEP_COUNTS = {
    "above": 150,
    "down": 100,
    "close": 50,
    "lift": 80,
    "carry": 120,
    "lower": 80,
    "release": 40,
    "retreat": 60,
}
DEFAULT_STEPS = 100

# Closed-loop approach. Every approach and descent re-reads the camera and re-aims, instead of
# driving once at a position computed before the arm started moving.
#
# The open-loop version is what made ball handling look hopeless, and the reason is in the
# measurements this file already quotes: the arm's own transit knocks a target 57-62 mm and the
# descent knocks it a further 192-371 mm. A plan built from one camera frame is therefore wrong
# by 20-40 cm by the time the jaws close, and no amount of re-planning between steps fixes that
# because the object is being knocked *continuously* while the arm is moving. The old remedy was
# to re-aim once, after the hover; this re-aims throughout, so the jaws arrive where the object
# is rather than where it was.
# **The travel count was reduced and the descent count was put back, and that asymmetry is
# the finding.** The simulation runs at half real time - measured with the demo 11 environment,
# 33.0 ms per step against a 1/60 s timestep, a real-time factor of 0.505 - so 150 steps of
# travel were 5 s of wall clock for a move above the parts, with nothing at stake.
#
# 150 -> 120 on TRAVEL was kept: it is a move in free air, no part is touched, and every camera
# re-aim on the way is preserved (TRACK_EVERY is 20, so 120 steps is 6 re-aims).
#
# 200 -> 100 on DESCENT was tried in the same run and REVERTED, because it made the pick
# UNRELIABLE, not just faster: 14.9 s instead of 33.1 s, and one stale grasp - the arm
# reporting that the part was no longer between the jaws when they closed. ``grip_and_release.md``
# says why: the tool-centre servo keeps correcting down to LAST_CORRECTION_Z, and a descent that
# reaches the bottom before the servo has converged closes the jaws on where the part WAS.
# ``approach_drift.md`` sweeps the other direction - a descent that is too slow leans the forearm
# on the part - so there are two ways to fail and 200 steps is the one that was measured to
# work. A travel step that is too long wastes time; a descent step that is too short loses the

TRACK_TRAVEL_STEPS = 120
TRACK_DESCEND_STEPS = 200
# How often to re-look while moving. 20 steps is a third of a second: often enough to follow a
# roll, rare enough that a descent is not dominated by rendering.
TRACK_EVERY = 20
# The front camera is intentionally behind the arm. A previously identified target can be hidden
# briefly as the arm enters its sight line, so front-view tracking may coast on the locked scene
# pose for a bounded number of looks; top and side views fail closed on the first missing target.
TRACK_MAX_OCCLUDED = 8
# Move laterally to the target while staying at the current Z for this fraction of an "above"
# approach. Then descend vertically. Combining the IK XY correction with descent swept the arm
# through spheres in clustered scenes before the fingers reached their grasp height.
TRACK_ABOVE_HOVER_FRACTION = 0.65
# Closed-loop corrections during lift/carry/lower. The approach already uses the observed
# aperture; carrying used to revert to the open-loop command and accumulated 17-25 mm of error.
MOTION_FEEDBACK_EVERY = 10
MOTION_FEEDBACK_GAIN = 0.65
MOTION_FEEDBACK_MAX_CORRECTION_M = 0.025
MOTION_FEEDBACK_TOTAL_CORRECTION_M = 0.050
MOTION_FEEDBACK_FINAL_ERROR_M = 0.020
# Simulator-side object pose check after the horizontal carry. The gripper servo alone can reach
# the destination while a slipped object remains on the table; require the physical actor to
# follow the requested object-centre pose before lowering or releasing.
CARRY_OBJECT_FOLLOW_TOLERANCE_M = 0.025
SIM_ASSIST_CARRY_TOLERANCE_M = 0.035


def carry_follow_tolerance(*, simulator_assisted: bool) -> float:
    """Keep physical-slip tolerance tight while allowing bounded assisted compliance."""
    return (SIM_ASSIST_CARRY_TOLERANCE_M if simulator_assisted
            else CARRY_OBJECT_FOLLOW_TOLERANCE_M)


def object_pose_error(actual, target) -> float:
    """Euclidean object-centre error in world metres, or infinity for invalid poses."""
    try:
        actual_xyz = np.asarray(actual, dtype=float).reshape(-1)[:3]
        target_xyz = np.asarray(target, dtype=float).reshape(-1)[:3]
        if (actual_xyz.size != 3 or target_xyz.size != 3
                or not np.all(np.isfinite(actual_xyz))
                or not np.all(np.isfinite(target_xyz))):
            return math.inf
        return float(np.linalg.norm(actual_xyz - target_xyz))
    except (TypeError, ValueError):
        return math.inf


def bounded_feedback_offset(current, correction, max_norm: float =
                             MOTION_FEEDBACK_TOTAL_CORRECTION_M) -> np.ndarray:
    """Accumulate a feedback correction without allowing unreachable targets to run away."""
    offset = np.asarray(current, dtype=float) + np.asarray(correction, dtype=float)
    magnitude = float(np.linalg.norm(offset))
    if magnitude > max_norm > 0.0:
        offset *= max_norm / magnitude
    return offset


def _quat_mul_wxyz(a, b) -> np.ndarray:
    """Multiply scalar-first quaternions, as Isaac Lab stores them."""
    aw, ax, ay, az = np.asarray(a, dtype=float).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=float).reshape(4)
    return np.asarray((aw*bw-ax*bx-ay*by-az*bz,
                       aw*bx+ax*bw+ay*bz-az*by,
                       aw*by-ax*bz+ay*bw+az*bx,
                       aw*bz+ax*by-ay*bx+az*bw), dtype=float)


def _rotate_quaternion_wxyz(quaternion, vector) -> np.ndarray:
    """Rotate a 3-vector by a scalar-first unit quaternion."""
    q = np.asarray(quaternion, dtype=float).reshape(4)
    v = np.asarray(vector, dtype=float).reshape(3)
    qv = q[1:]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q[0] * v)


def relative_pose_wxyz(parent_position, parent_quaternion, child_position,
                       child_quaternion) -> tuple[np.ndarray, np.ndarray]:
    """Express a child pose in a parent frame without changing its world pose."""
    parent_position = np.asarray(parent_position, dtype=float).reshape(3)
    child_position = np.asarray(child_position, dtype=float).reshape(3)
    parent_quaternion = np.asarray(parent_quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(parent_quaternion))
    if norm < 1e-9:
        raise ValueError("parent quaternion must be nonzero")
    parent_quaternion = parent_quaternion / norm
    inverse = parent_quaternion.copy()
    inverse[1:] *= -1.0
    relative_position = _rotate_quaternion_wxyz(
        inverse, child_position - parent_position)
    relative_quaternion = _quat_mul_wxyz(
        inverse, np.asarray(child_quaternion, dtype=float).reshape(4))
    return relative_position, relative_quaternion


def lowering_start_aim(bot, body: np.ndarray, hold: np.ndarray) -> tuple[float, float]:
    """Return the measured aperture centre for a clearance-to-grasp descent.

    The calibrated body-to-tool offset is only a fallback. Starting the second tracking phase
    from that estimate can differ from the actual jaw midpoint after IK moves the arm, making a
    stationary object look like a large identity jump. The measured midpoint is the correct
    scene-linked reference for continuing the same descent.
    """
    measure = getattr(bot, "aperture_centre", None)
    if callable(measure):
        try:
            centre = measure()
            if centre is not None and len(centre) >= 2:
                x, y = float(centre[0]), float(centre[1])
                if math.isfinite(x) and math.isfinite(y):
                    return x, y
        except (TypeError, ValueError, IndexError):
            pass
    return float(body[0] + hold[0]), float(body[1] + hold[1])


def tracked_body_z(action: str, start_z: float, target_z: float, fraction: float) -> float:
    """Keep an above approach high while it aligns XY, then descend on a vertical segment."""
    progress = min(1.0, max(0.0, float(fraction)))
    if action == "above":
        descend = min(1.0, max(0.0,
                              (progress - TRACK_ABOVE_HOVER_FRACTION)
                              / (1.0 - TRACK_ABOVE_HOVER_FRACTION)))
    else:
        descend = progress
    return float(start_z) + (float(target_z) - float(start_z)) * descend

# Stop the per-step render while an Executor is driving. MEASURED, not assumed, with
# tools/probe_step_cost.py on this machine: a motion step (`env.step`) costs ~32 ms, while the
# same step with physics only (`env.sim.step(render=False)`) costs ~3.9 ms. The other ~28 ms is a
# ``sim.render()`` that ManagerBasedEnv.step issues on EVERY physics step while an RTX sensor is
# present - and that render is redundant here, because the dashboard's live picture and every
# tracked re-aim read the camera through ``isaac_io.capture``/``capture_rgb``, which call
# ``sim.render()`` themselves. That per-step render is the whole reason the simulation runs at
# half real time; removing it turns each motion step back into a physics step.
#
# Setting the render interval this high means the motion loop essentially never auto-renders (a
# command is a couple thousand steps at most), while the pump's explicit renders still publish
# frames at up to ``cam_fps``. Physics is untouched - render does not change the simulation - so
# grasp validation and placement verification behave exactly as before.
MOTION_RENDER_INTERVAL = 1_000_000

# How far the aim may be corrected in one update before it is treated as "this is a different
# object" rather than "this object got knocked". A shove moves a ball 33-45 mm
# (isaac_io.BALL_LINEAR_DAMPING); 120 mm is well past that and inside the distance between two
# same-coloured objects.
# Tracking happens above the part, before contact. Camera measurements should therefore stay
# close to the scene-linked target; a larger correction is evidence of an identity jump or a
# bad camera projection, not target motion. The old 120 mm allowance let a succession of small
# updates walk the tool away from the calibrated object pose.
TRACK_MAX_SHIFT_M = 0.025
TRACK_MAX_TOTAL_SHIFT_M = 0.025
TRACK_MAX_DESCENT_TARGET_DRIFT_M = 0.010
GRASP_CLOSE_CENTER_DRIFT_M = 0.008


def tracking_correction_is_safe(observed, current_aim, initial_aim) -> bool:
    """Reject a camera correction that is large per frame or accumulated from the plan pose."""
    point = (float(observed[0]), float(observed[1]))
    return (math.dist(point, current_aim) <= TRACK_MAX_SHIFT_M
            and math.dist(point, initial_aim) <= TRACK_MAX_TOTAL_SHIFT_M)


def tracked_target_motion_is_safe(observed, initial_aim, *, during_descent: bool) -> bool:
    """Fail closed if visual servo starts following an object pushed during approach."""
    point = (float(observed[0]), float(observed[1]))
    limit = TRACK_MAX_DESCENT_TARGET_DRIFT_M if during_descent else TRACK_MAX_TOTAL_SHIFT_M
    return math.dist(point, (float(initial_aim[0]), float(initial_aim[1]))) <= limit


def close_center_drift_is_safe(before_xy, after_xy) -> bool:
    """A close that shoves the target beyond camera error is a missed grasp, not a lift."""
    return (math.dist((float(before_xy[0]), float(before_xy[1])),
                      (float(after_xy[0]), float(after_xy[1])))
            <= GRASP_CLOSE_CENTER_DRIFT_M)


def tool_alignment_needs_correction(error_m: float, allowed_error_m: float) -> bool:
    """Continue visual centring only while the measured error exceeds grasp tolerance."""
    return float(error_m) > float(allowed_error_m)


def pregrasp_coast_supported(kind: str, size_m) -> bool:
    """Whether a non-contact hover may coast through brief segmentation mask jumps."""
    return (kind == "ball" or
            bool(size_m) and max(float(value) for value in size_m[:2]) <= 0.022)


def tracking_cadence_for(kind: str, default: int) -> int:
    """Bound visual-servo spacing so the hover can converge before the final descent."""
    return min(int(default), 10)


def tracking_steps_for(kind: str, size_m, action: str, default: int) -> int:
    """Slow difficult final approaches enough for feedback to correct before contact."""
    steps = max(1, int(default))
    if (action == "down" and kind == "ball" and size_m
            and 0.029 <= max(float(value) for value in size_m[:2]) <= 0.031):
        return max(steps, 400)
    if (action == "down" and kind == "cube" and size_m
            and round(max(float(value) for value in size_m[:2]) * 1000) == 30):
        # At the measured forward-pick reach, a 30 mm cube needs the quarter-turn wrist pose;
        # the top camera also begins to occlude it during the last part of descent. Extra
        # physics steps give the live aperture servo more bounded correction opportunities.
        return max(steps, 400)
    return steps

# The camera stops being asked once the gripper BODY is below this height, which is where the
# arm stops occluding the object it is reaching for: measured at the bottom of a descent, a blue
# cube was not in the frame at all.
#
# In BODY coordinates, not hold-point coordinates, because ``track`` compares it against the
# interpolated body height. The hover sits at a hold point of scene_spec.TABLE_TOP_Z + tray
# height + 0.04 = 0.131, so its body is at 0.223 - above this, and the bottom of a descent is at
# 0.148 - below it. That gap is where the arm follows the target and then locks in.
TRACK_OBSERVE_ABOVE_Z = 0.20

# How many steps to wait for a knocked object to stop rolling before re-aiming at it.
# Measured: 6.7 s (400 steps) for the worst ball, so the cap is generous rather than tight.
SETTLE_UNTIL_STILL_STEPS = 400

# Below this speed an object counts as stopped, so that it will not move appreciably during
# the camera capture that follows.
STILL_M = 0.001

# How many ``settle`` steps between two calls to ``settle_hook``. One frame is one
# ``sim.render()``, and a render between two physics steps is the most expensive
# thing the dashboard can ask for - so the waits that issue no action still publish a
# picture, but at 1/15th of their steps, which keeps the dashboard moving without
# paying for a picture of the same still table 30 times.
SETTLE_HOOK_EVERY = 15

# How close an object must still be to where the plan aimed when the jaws finish closing.
# This is the check that was missing, and its absence is why the demo reported successes it
# did not have. ``grab`` writes an object's pose from the gripper frame every step, so an
# object 371 mm away was picked out of empty air, carried, and placed correctly - and nothing
# in the report said the jaws had never touched it. The number comes from
# tools/probe_approach_drift.py: a ball is knocked 57-60 mm by the arm's own transit across
# the table and a further 130-310 mm by the descent. Not the jaw width: measured across
# closing angles from 0.4 to 1.0 rad the displacement is identical, including fully open.
# a 4 cm ball and squirt it out. A 3 cm cube moves 10-12 mm over the same approach, all of it
# perception error, and is well inside this.
#
# 20 mm is a ball's own radius, i.e. "the jaws could still physically be around it". Beyond
# that the grasp did not happen and must not be faked.
GRASP_REACH_M = 0.020

# What fraction of the gripper's own travel an object must follow during the lift to count as
# held. Absolute distance is not enough: a 4 cm ball nudged by the closing jaws follows only
# 46% of the lift and then drops out, which a "did it move at all" test calls a real grasp.
# A ball is nudged 30 mm of the lift's 60 mm and then drops out, which a "did it move at all"
# test happily calls a real grasp. The measured follow fraction is 0.51 in that case.
GRASP_FOLLOW_MIN = 0.8
# Tool and object motion are read from separate simulator buffers. Allow small contact/solver
# noise on a clean lift, while still rejecting objects launched well beyond the gripper path.
GRASP_LOST_OVERSHOOT = 1.05

# Below this the object did not move AT ALL during the lift, so nothing held it and the
# kinematic fallback must not be used on it.
#
# It is NOT on its own a failure, and this took one run to find out why. In this simulation
# nothing is ever held by friction - demo 05 says so of the cube it inherited - so the fallback
# is the intended mechanism and the object is *supposed* to sit still while the gripper rises.
# ``tools/check_executor.py`` passes a cube through the fallback on this exact test.
#
# What distinguishes the fallback working from the jaws missing is whether the object is still
# where the jaws were aimed. The ``close`` step has already refused anything further away than
# ``GRASP_REACH_M``, so by the time the lift is judged the object should be within 20 mm of the
# aim; if it has wandered off even that far while the gripper rose, nothing was grasped and
# carrying it would place a ghost. 8 mm is a third of the smallest object here, well clear of
# the solver jitter a resting object shows.
GRASP_TOUCHED_M = 0.008


# The park pose is MEASURED, not chosen. tools/find_park_pose.py sweeps nine candidates and,
# for each, checks the two things that matter: that perception still finds all four objects
# and both containers with the arm there, and that the arm can still be driven afterwards
# (demo 05's grasp pose, which tracks to 0 mm on an unparked arm). All nine passed; this is
# the best of them.
#
# Note what is NOT the problem. demo 09 parks with isaac_io.park_arm, which writes joint
# positions straight into the articulation, and a first version of this demo parked that way
# before driving. The arm could not then be driven at all - 242 mm from every commanded pose,
# drifting instead of tracking, cube 22 cm from the target. Parking by driving, with the IK
# action manager still in charge, gives 0 mm.
PARK_POSE = (0.35, -0.58, 0.46)

# The order of operations matters more than the path here.
#
# Any parking motion sweeps the forearm across the near half of the table and shoves whatever
# is on it: measured with tools/probe_ball_roll.py, parking a populated scene drifted the
# balls 79 mm. Parking in two legs - straight up to CLEAR_Z first, to lift the forearm over
# the objects - is worse, not better, at 169 mm: this arm hangs off a shoulder, so raising
# the gripper swings the elbow down and forward into the table instead.
#
# So parking does not avoid the sweep, and no pose avoids it. It does not need to:
# ``place_items`` rewrites every object's pose from the declared layout and now settles until
# nothing is moving, which measured 0.0 mm of drift. A round therefore parks first and
# populates second, and the camera capture comes after ``place_items``. Anything the sweep
# moved is put back before a single pixel is rendered.

# Park briskly. MEASURED, by tools/probe_park_speed.py, sweeping from the retreat pose - the
# arm's position between commands - back to PARK_POSE with the objects down:
#
#     25 steps (0.4 s)   10.4 mm worst drift, nothing left the table
#     40 steps (0.7 s)   16.6 mm
#     60 steps (1.0 s)   24.4 mm
#     90 steps (1.5 s)   36.5 mm, one ball out of the working area
#    150 steps (2.5 s)   60.4 mm, one ball out of the working area
#    300 steps           118.7 mm      600 steps  237.3 mm
#   1200 steps (20 s)   1893.3 mm
#
# The direction of that trend is the opposite of the obvious one, and it is worth stating: a
# SLOW park is far worse. The forearm rests against the balls and leans on them the whole
# time it is passing, so duration is roughly how long the balls are being pushed. Only the
# balls move; the cubes read 0.0 mm at every duration, because a cube on a flat face does not
# slide.
#
# 40 steps is the pick: 0.7 s is still a plausible arm motion, and 16.6 mm of drift is well
# inside the 51 mm smallest same-colour gap. The demo re-perceives after parking, so what the
# sweep moves is seen where it came to rest and planned for there.
PARK_STEPS = 40


@dataclass
class StepResult:
    action: str
    note: str
    ok: bool = True
    grasp: str = ""


@dataclass
class RunResult:
    steps: list[StepResult] = field(default_factory=list)
    aborted: bool = False
    aborted_at: str = ""
    real_grasps = 0
    kinematic_grasps = 0
    lost_grasps = 0
    # Grasps where the jaws shut on nothing because the arm had already knocked the target
    # away. These are the ones that used to be reported as successes.
    stale_grasps = 0
    stale_prims: list[str] = field(default_factory=list)
    # Times the plan was re-aimed after the arm knocked its own target.
    reaimed = 0
    # Approach and descent moves that re-read the camera while the arm was moving.
    tracked = 0

    @property
    def ok(self) -> bool:
        return not self.aborted and all(s.ok for s in self.steps)

    def summary(self) -> str:
        if self.aborted:
            return f"aborted at {self.aborted_at} after {len(self.steps)} step(s)"
        return (f"{len(self.steps)} step(s), {self.real_grasps} real grasp(s), "
                f"{self.kinematic_grasps} kinematic"
                + (f", {self.tracked} tracked" if self.tracked else "")
                + (f", {self.lost_grasps} lost" if self.lost_grasps else "")
                + (f", {self.stale_grasps} stale target/grasp(s)"
                   if self.stale_grasps else ""))


def failed_step_can_retarget(action: str, object_held: bool) -> bool:
    """Whether a fresh perception pass can safely rebuild a failed pick plan.

    A missed close or lift is recoverable only when the executor confirms that it is not
    holding the object. Re-reading the scene then lets the planner try a new grasp pose
    instead of continuing through stale waypoints.
    """
    return (action in ("above", "down")
            or (action in ("close", "lift") and not object_held))


class Executor:
    """Drives the arm through a plan. One instance per simulator run."""

    def __init__(self, env, items=None, log=lambda *_a: None):
        import torch
        from isaaclab.utils.math import quat_inv

        self.env = env
        self.items = list(items or [])
        self.log = log
        # Turn off the per-step render while we drive: see MOTION_RENDER_INTERVAL. The camera's
        # own renders (cam_pump -> isaac_io.capture_rgb, look -> isaac_io.capture) are explicit
        # and unaffected; physics is identical, so grasps and verification do not change.
        try:
            self.env.cfg.sim.render_interval = MOTION_RENDER_INTERVAL
        except Exception:
            pass
        # Called while physics runs but nothing is being commanded: ``settle`` waits
        # for a released part to come to rest, and a descent's re-aim ticks. Demo 11
        # uses it to keep the dashboard's picture moving, because those are exactly
        # the stretches where the camera used to go stale and the panel reported
        # "imagen sin actualizar" while the robot was measuring. ``do`` already had
        # ``frame_hook`` for the same reason; this is the hook for the waits that
        # issue no action at all.
        self.settle_hook = None
        # Optional safe-height approach, enabled only in explicit A/B trials. Its feasibility
        # depends strongly on the current IK pose, so the benchmark must establish a benefit
        # across positions before it can become the production path.
        self.approach_clearance_z: float | None = None
        # Experimental observation cadence for moving-target approach. The default retains
        # the measured production cadence; paired simulation trials can test faster updates.
        self.track_every_steps = TRACK_EVERY
        # Demo 11 connects this to its telemetry bus. It is checked between physics
        # steps, so a pause preserves the current gripper pose and never releases a part.
        self.pause_hook = None
        # Optional per-step target schedule for isolated gripper-contact experiments.
        self.gripper_step_hook = None
        self.torch = torch
        self.device = env.device
        self.robot = env.scene["robot"]
        self.gripper_id = self.robot.find_bodies("gripper")[0][0]
        # The WHOLE per-env tensors, not [0]. Demo 05 writes ``base_pos = robot.data.root_pos_w``
        # and ``base_quat = robot.data.root_quat_w``, which are (1, 3) and (1, 4). Taking [0]
        # makes them 1-D, and then torch.cat of the pose, the quaternion and the gripper fails
        # with "Tensors must have same number of dimensions: got 2 and 1".
        self.base_pos = self.robot.data.root_pos_w.clone()
        self._inverse = quat_inv(self.robot.data.root_quat_w.clone())
        # The commanded gripper orientation is the base's own inverse, which is what demo 05
        # sends. It is not "identity in world" - it is "aligned with the base frame".
        self.grip_quat = self._inverse
        self._upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        self._hold = torch.tensor([HOLD], device=self.device)
        self._zeros = torch.zeros(1, 6, device=self.device)
        self._held = None  # the object being carried, if any
        self._held_offset = None
        self._held_quaternion = None
        # Demo 11 supplies the measured midpoint between its finger pads. Kept optional so the
        # base executor remains usable by the earlier demos that do not expose that measurement.
        self.position_feedback = None
        self.current = self._gripper_pos().clone()

    # -- frames ---------------------------------------------------------------

    def _gripper_pos(self):
        return self.robot.data.body_pos_w[:, self.gripper_id].clone()

    def _action(self, world_point, grip: float):
        """A pose in the base frame plus the gripper command, as the action manager wants."""
        from isaaclab.utils.math import quat_apply

        target = quat_apply(self._inverse, world_point - self.base_pos)
        gripper = self.torch.tensor([[grip]], device=self.device)
        return self.torch.cat([target, self.grip_quat, gripper], dim=-1)

    def hold_offset_world(self):
        """Rotate the calibrated gripper-local hold offset into the world frame."""
        from isaaclab.utils.math import quat_apply, quat_mul

        world_grip_quat = quat_mul(self.robot.data.root_quat_w, self.grip_quat)
        return quat_apply(world_grip_quat, self._hold)

    def body_pose_from_current(self):
        """Return the most recently commanded gripper-body pose in world coordinates."""
        return self.current[0].detach().cpu().numpy().copy()

    def set_tool_yaw(self, yaw_deg: float) -> None:
        """Set a world-Z tool yaw while keeping the calibrated hold offset in the same frame."""
        import math
        import torch
        from isaaclab.utils.math import quat_mul

        if not hasattr(self, "_nominal_grip_quat"):
            self._nominal_grip_quat = self.grip_quat.clone()
            self._nominal_hold = self._hold.clone()
        yaw = float(yaw_deg)
        if abs(yaw) < 1e-6:
            self.grip_quat = self._nominal_grip_quat.clone()
            self._hold = self._nominal_hold.clone()
            return
        half = math.radians(yaw) / 2.0
        world_yaw = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]],
                                 dtype=self.grip_quat.dtype, device=self.device)
        self.grip_quat = quat_mul(self._inverse, world_yaw)
        self._hold = self._nominal_hold.clone()

    def gripper_pixel(self, camera):
        """Where the gripper is in the image, for masking it out of perception."""
        pos = self._gripper_pos()[0].detach().cpu().numpy()
        return camera.project([float(pos[0]), float(pos[1]), float(pos[2])])

    # -- grasping -------------------------------------------------------------

    def grab(self, prim: str) -> None:
        """Attach the object's measured pose to the gripper for simulator-only carrying.

        ``HOLD`` is a planning calibration, not the transform of every individual grasp.
        Capturing the relative pose here prevents the first carry tick from teleporting an
        object when the measured closed-jaw pose differs from the mesh estimate.
        """
        from isaaclab.utils.math import quat_inv, quat_apply, quat_mul

        gripper_position = self.robot.data.body_pos_w[:, self.gripper_id]
        gripper_quaternion = self.robot.data.body_quat_w[:, self.gripper_id]
        obj = self.env.scene[prim]
        object_position = obj.data.root_pos_w
        object_quaternion = obj.data.root_quat_w
        inverse = quat_inv(gripper_quaternion)
        self._held_offset = quat_apply(inverse, object_position - gripper_position).clone()
        self._held_quaternion = quat_mul(inverse, object_quaternion).clone()
        self._held = prim
        obj.write_root_velocity_to_sim(self._zeros)

    def drop(self) -> None:
        self._held = None
        self._held_offset = None
        self._held_quaternion = None

    @property
    def holding(self) -> str | None:
        """The prim currently being carried kinematically, if any."""
        return self._held

    def _carry_if_held(self) -> None:
        """Write the held object's pose from the gripper frame. Called every step."""
        if self._held is None:
            return
        from isaaclab.utils.math import combine_frame_transforms

        offset = self._held_offset if self._held_offset is not None else self._hold
        orientation = (self._held_quaternion if self._held_quaternion is not None
                       else self._upright)
        pos, quat = combine_frame_transforms(
            self.robot.data.body_pos_w[:, self.gripper_id],
            self.robot.data.body_quat_w[:, self.gripper_id],
            offset,
            orientation,
        )
        pose = self.torch.cat([pos, quat], dim=-1)
        self.env.scene[self._held].write_root_pose_to_sim(pose)
        self.env.scene[self._held].write_root_velocity_to_sim(self._zeros)

    def _moved(self, prim: str, since: np.ndarray) -> float:
        """How far an object has travelled since ``since``, in 3-D.

        3-D, not x-y: the lift is vertical, so an x-y measure reports the gripper's own travel
        as 0.0 mm and every grasp reads as "followed 0% of the lift".
        """
        now = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        return float(np.linalg.norm(now - since))

    # -- one step -------------------------------------------------------------

    def do(self, action: str, pose, grip: float, prim: str | None = None,
           on_close=None, steps: int | None = None, check: str | None = None,
           body_target: bool = False, feedback_target=None) -> StepResult:
        """Run one planner action to completion.

        Planner poses are world-frame hold points. A tracked approach can instead pass a direct
        world-frame gripper-body target with ``body_target=True``; this avoids converting frames
        twice and preserves the final closed-loop command for the close and lift.

        ``check`` is what to test at the END of this action, and both checks exist because a
        grasp can fail in two different ways:

        ``"reach"``
            on ``close``: is the object still where this step aimed? If it is not, the jaws
            shut on nothing and the run is stale. See ``GRASP_REACH_M``.
        ``"follow"``
            on ``lift``: did the object come up with the gripper? This is the friction test,
            and it is what decides real against kinematic.

        The interpolation is demo 05's: each step moves 1/0.8n of the remaining distance, so
        the arm covers the whole way in 80% of the steps and then holds. Without that the last
        few millimetres are taken at a crawl and the servos sag.

        ``torch.no_grad`` rather than demo 05's ``torch.inference_mode``. An inference tensor
        may not be used in an in-place operation outside inference mode, and this class
        interleaves its stepping with other calls - ``capture``, ``park_arm`` - that touch the
        same buffers. Demo 05 can afford inference_mode because nothing else runs between its
        steps.
        """
        steps = steps or STEP_COUNTS.get(action, DEFAULT_STEPS)
        # Everything here stays a tensor: demo 05 interpolates in torch and quat_apply wants a
        # tensor, so mixing in numpy fails on the first subtraction.
        goal = self.torch.as_tensor(np.asarray(pose, dtype=np.float32),
                                    device=self.device)
        if not body_target:
            goal = goal - self.hold_offset_world()
        # ``current`` is a command cache, not a measurement. Start each planned
        # segment from the live articulation pose so residual IK lag at a phase
        # boundary cannot be replayed as a lateral sweep during close or carry.
        try:
            start = self._gripper_pos().clone()
            if tuple(start.shape) != tuple(self.current.shape):
                start = self.current.clone()
        except (AttributeError, IndexError, RuntimeError, TypeError):
            start = self.current.clone()
        self.current = start.clone()
        result = StepResult(action=action, note="")
        # The SO-101 gripper is a linkage, not a true parallel jaw: closing its fingers
        # moves the measured pad midpoint laterally by several millimetres. Without a
        # compensating arm correction the fingers sweep round parts and push them away
        # before they can clamp. Keep the measured aperture over the target during close
        # as well as during transport. This is ordinary feedback from the robot state,
        # not an object-follow constraint.
        feedback_enabled = (getattr(self, "position_feedback", None) is not None
                            and action in ("close", "lift", "carry", "lower"))
        feedback_offset = self.torch.zeros_like(start) if feedback_enabled else None
        feedback_corrections = 0
        feedback_worst = 0.0
        feedback_final = None
        if feedback_target is None:
            feedback_target = (np.asarray(on_close, dtype=float)[:2]
                               if body_target and on_close is not None
                               else np.asarray(pose, dtype=float)[:2])
        else:
            feedback_target = np.asarray(feedback_target, dtype=float)[:2]

        with self.torch.no_grad():
            for index in range(1, steps + 1):
                if self.pause_hook is not None and not self.pause_hook():
                    result.ok = False
                    result.note = "parada solicitada entre pasos de física"
                    break
                self.current = start + (goal - start) * min(1.0, index / (0.8 * steps))
                if feedback_offset is not None:
                    self.current = self.current + feedback_offset
                if action == "close" and self.gripper_step_hook is not None:
                    self.gripper_step_hook(index, steps)
                self.env.step(self._action(self.current, grip))
                # Simulator-specific assists (for example, a collision-filtered sphere
                # held stationary while the gripper approaches) need to update after every
                # physics tick, including the ordinary close/lift actions below.
                physics_step_hook = getattr(self, "physics_step_hook", None)
                if physics_step_hook is not None:
                    physics_step_hook(action, prim)

                # A ball or prism can start rolling as soon as one jaw touches it.
                # Waiting until the whole close finishes lets the fingers keep
                # pushing it for dozens of physics ticks (observed displacements
                # exceeded 200 mm). Stop the close on the first measured drift past
                # the grasp-centre tolerance so recovery can re-localize the part.
                if (action == "close" and check == "reach" and prim
                        and feedback_target is not None):
                    try:
                        object_xy = (self.env.scene[prim].data.root_pos_w[0, :2]
                                     .detach().cpu().numpy())
                        close_drift = float(np.linalg.norm(
                            object_xy - feedback_target[:2]))
                    except (KeyError, AttributeError, IndexError, TypeError):
                        close_drift = 0.0
                    if close_drift > GRASP_CLOSE_CENTER_DRIFT_M:
                        result.ok = False
                        result.grasp = "stale"
                        result.note = (f"cierre desplazó la pieza {close_drift * 1000:.1f} mm; "
                                       "cierre detenido para volver a localizarla")
                        break

                # The arm's IK action does not arrive exactly where a command asks it to go.
                # Servo the measured jaw midpoint toward the requested hold point while an
                # object is being lifted, carried, or lowered. Approach already has its own
                # camera-and-aperture servo; release and retreat happen with empty jaws.
                feedback_every = 1 if action == "close" else MOTION_FEEDBACK_EVERY
                if (feedback_enabled and index % feedback_every == 0):
                    measured = self.position_feedback()
                    if measured is not None and len(measured) >= 2:
                        error = feedback_target - np.asarray(measured[:2], dtype=float)
                        error_norm = float(np.linalg.norm(error))
                        feedback_worst = max(feedback_worst, error_norm)
                        feedback_final = error_norm
                        if os.environ.get("DEMO11_TRACE_MOTION") == "1" and action in {
                                "close", "lift", "carry", "lower"}:
                            body_now = self.current[0].detach().cpu().numpy()
                            measured_body = self._gripper_pos()[0].detach().cpu().numpy()
                            self.log(f"      motion trace {action} step={index}/{steps} "
                                     f"body=({body_now[0]:+.3f},{body_now[1]:+.3f}) "
                                     f"measured_body=({measured_body[0]:+.3f},"
                                     f"{measured_body[1]:+.3f}) "
                                     f"aperture=({measured[0]:+.3f},{measured[1]:+.3f}) "
                                     f"target=({feedback_target[0]:+.3f},"
                                     f"{feedback_target[1]:+.3f}) "
                                     f"error={error_norm * 1000:.1f} mm")
                        if error_norm > 0.001:
                            correction = np.clip(
                                error * MOTION_FEEDBACK_GAIN,
                                -MOTION_FEEDBACK_MAX_CORRECTION_M,
                                MOTION_FEEDBACK_MAX_CORRECTION_M)
                            bounded = bounded_feedback_offset(
                                feedback_offset[0, :2].detach().cpu().numpy(), correction)
                            feedback_offset[0, :2] = self.torch.as_tensor(
                                bounded, dtype=feedback_offset.dtype, device=self.device)
                            feedback_corrections += 1

                if (action == "close" and prim
                        and os.environ.get("DEMO11_TRACE_CONTACT") == "1"
                        and index % 10 == 0):
                    try:
                        object_xy = (self.env.scene[prim].data.root_pos_w[0, :2]
                                     .detach().cpu().numpy())
                        aperture_xy = self.position_feedback()
                        gap_mm = (float(np.linalg.norm(object_xy - aperture_xy[:2])) * 1000
                                  if aperture_xy is not None else float("nan"))
                        jaw = ""
                        try:
                            joint_id = int(self.robot.find_joints("gripper")[0][0])
                            jaw = (f" jaw={float(self.robot.data.joint_pos[0, joint_id]):.3f}"
                                   f"->{float(self.robot.data.joint_pos_target[0, joint_id]):.3f} rad")
                            pads = self.finger_contact_points()
                            if pads is not None:
                                fixed, moving = pads["gripper"], pads["jaw"]
                                quat = self.robot.data.body_quat_w[0, self.gripper_id]
                                w, qx, qy, qz = (float(v) for v in quat.detach().cpu().numpy())
                                yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy),
                                                              1 - 2 * (qy * qy + qz * qz)))
                                jaw += (f" pads=({fixed[0]:+.3f},{fixed[1]:+.3f},{fixed[2]:+.3f})"
                                        f"/({moving[0]:+.3f},{moving[1]:+.3f},{moving[2]:+.3f})"
                                        f" sep={float(np.linalg.norm(fixed - moving)) * 1000:.0f} mm"
                                        f" body_yaw={yaw:+.0f} deg")
                        except Exception:  # noqa: BLE001 - the jaw readout is optional
                            pass
                        self.log(f"      contact trace close step={index}/{steps} "
                                 f"object=({object_xy[0]:+.3f},{object_xy[1]:+.3f}) "
                                 f"aperture=({aperture_xy[0]:+.3f},{aperture_xy[1]:+.3f}) "
                                 f"gap={gap_mm:.1f} mm{jaw}")
                    except Exception:  # diagnostics must not change the grasp outcome
                        pass

                if on_close is not None and index == steps and check == "reach":
                    # The jaws have finished shutting. Is the object still where this step
                    # aimed? If not, the descent knocked it away, the jaws closed on nothing,
                    # and the honest answer is that the grasp did not happen.
                    now = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
                    aimed = feedback_target
                    off = float(np.linalg.norm(now[:2] - aimed))
                    if off > GRASP_REACH_M:
                        result.grasp = "stale"
                        result.ok = False
                        result.note = (f"nothing to close on: the object is {off * 1000:.0f} mm "
                                       f"from where the jaws aimed, past the "
                                       f"{GRASP_REACH_M * 1000:.0f} mm the jaws can reach")

                if on_close is not None and index == steps and check == "follow":
                    # Judged as a fraction of how far the GRIPPER moved. A grasp that "took"
                    # because the closing jaws nudged the object is not a grasp; and an object
                    # that followed half the lift and then slipped out is not held either.
                    moved = self._moved(prim, on_close)
                    travel = float(self.torch.linalg.norm((goal - start)[0]))
                    follow = moved / travel if travel > 1e-6 else 0.0
                    if follow > GRASP_LOST_OVERSHOOT:
                        # More than 5% beyond the gripper path indicates a launch. A small
                        # overshoot is allowed because tool and object motion come from separate
                        # simulator buffers and differ by fractions of a millimetre on a clean lift.
                        #
                        # NOT carried, deliberately. Grabbing it would write its pose from the
                        # gripper frame - teleporting a ghost 64 m back onto the table and
                        # then dropping it somewhere plausible. Better to say the object was
                        # lost and let the caller report a placement that did not happen.
                        result.grasp = "lost"
                        result.ok = False
                        result.note = (f"the object was knocked out of reach, not held "
                                       f"(moved {moved * 100:.0f} cm against a "
                                       f"{travel * 100:.0f} cm lift); not carrying it")
                    elif follow < GRASP_FOLLOW_MIN:
                        # Still where the jaws were aimed, or not? That, and not whether it
                        # moved, is what says the jaws closed around it.
                        here = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
                        aimed = feedback_target
                        off = float(np.linalg.norm(here[:2] - aimed))
                        if moved < GRASP_TOUCHED_M and off > GRASP_REACH_M:
                            result.grasp = "stale"
                            result.ok = False
                            result.note = (f"the jaws closed and the object is not there: it "
                                           f"ended {off * 1000:.0f} mm from where they aimed, "
                                           f"past the {GRASP_REACH_M * 1000:.0f} mm they can "
                                           f"reach; not carrying it")
                        else:
                            self.grab(prim)
                            result.grasp = "kinematic"
                            result.note = (f"followed {follow * 100:.0f}% of the lift "
                                           f"(moved {moved * 1000:.1f} of "
                                           f"{travel * 1000:.1f} mm), "
                                           f"carrying kinematically")
                    else:
                        result.grasp = "real"
                        result.note = (f"followed {follow * 100:.0f}% of the lift "
                                       f"({moved * 1000:.1f} of {travel * 1000:.1f} mm)")

                if grip == CLOSE or self._held is not None:
                    self._carry_if_held()
            # Do not let a pause or early break retain a temporary close schedule.
            self.gripper_step_hook = None
        if feedback_corrections:
            feedback_note = (f"jaw feedback {feedback_corrections}x, "
                             f"worst observed error {feedback_worst * 1000:.1f} mm")
            result.note = f"{result.note}; {feedback_note}" if result.note else feedback_note
        if (action in {"carry", "lower"} and feedback_final is not None
                and feedback_final > MOTION_FEEDBACK_FINAL_ERROR_M):
            result.ok = False
            result.note = (f"{result.note}; transporte detenido: error final de pinza "
                           f"{feedback_final * 1000:.1f} mm supera "
                           f"{MOTION_FEEDBACK_FINAL_ERROR_M * 1000:.0f} mm")
        return result

    def track(self, observe, target_z: float, steps: int, action: str = "above",
              grip: float = OPEN, every: int = TRACK_EVERY,
              aim: tuple[float, float] | None = None) -> StepResult:
        """Move to ``target_z`` while re-aiming at a moving target the whole way.

        This is the difference between the arm arriving where the object is and arriving where
        it was. ``observe`` is called every ``every`` steps and returns the object's current
        x-y as the camera sees it, or None if it cannot be found; the aim is then corrected to
        match. Only x and y move: z keeps the same 1/0.8n schedule ``do`` uses, so the arm still
        accelerates in and slows into the target rather than creeping the last millimetres.

        A per-frame or cumulative correction larger than its bound is not followed. Tracking is
        performed before contact, so a target that appears to move more than 25 mm is an
        association/projection failure. The old per-frame-only 120 mm bound let several smaller
        errors accumulate until the gripper approached a different point.

        Returns the number of corrections and the largest one, so the run can report honestly
        that the arm followed its target instead of pretending it aimed correctly first time.
        """
        # ``current`` is the last commanded body pose. The IK/action controller can
        # still have residual tracking error when a planned phase ends, so carrying
        # that command into the next interpolation compounds the lag and can sweep
        # the jaws sideways during close or carry. Seed from articulation feedback
        # whenever the live gripper pose is available; retain the command only for
        # lightweight executors and isolated tests without a robot state buffer.
        try:
            start = self._gripper_pos().clone()
            if tuple(start.shape) != tuple(self.current.shape):
                start = self.current.clone()
        except (AttributeError, IndexError, RuntimeError, TypeError):
            start = self.current.clone()
        self.current = start.clone()
        aim = aim if aim is not None else (float(start[0][0]), float(start[0][1]))
        initial_aim = (float(aim[0]), float(aim[1]))
        # self.current is the GRIPPER BODY, but target_z is the HOLD POINT - the place the
        # object should be - and the two differ by HOLD. ``do`` handles this by computing
        # ``goal = pose - HOLD``; the first version of ``track`` interpolated z verbatim, which
        # drove the body 92 mm too low on every descent and put the fingers through the table.
        # That is why the tracked picks were landing 85-128 mm outside the cavity: the arm was
        # not reaching for the tray, it was ploughing into the table beside it.
        hold_z = float(self._hold[0][2])
        start_z = float(start[0][2])
        target_body_z = target_z - hold_z
        corrections = 0
        largest = 0.0
        lost = 0
        occluded = 0

        with self.torch.no_grad():
            for index in range(1, steps + 1):
                if self.pause_hook is not None and not self.pause_hook():
                    lost += 1
                    break
                fraction = min(1.0, index / (0.8 * steps))
                z = start_z + (target_body_z - start_z) * fraction
                # Only consult the camera while the gripper is high enough not to be standing
                # on top of the thing it is looking for. Measured: at the bottom of a descent
                # the blue cube was gone from the frame altogether, and the only thing the
                # camera had left to offer was a gripper-shaped error. The aim is locked in
                # from the hover instead, and the stale-grasp check after the jaws close is
                # what verifies it.
                if (index % every == 0 or index == steps) and z >= TRACK_OBSERVE_ABOVE_Z:
                    seen = observe()
                    if seen is None:
                        if (getattr(self, "allow_occluded_tracking", False)
                                and occluded < TRACK_MAX_OCCLUDED):
                            occluded += 1
                        else:
                            lost += 1
                    else:
                        occluded = 0
                        shift = math.dist((seen[0], seen[1]), aim)
                        if not tracking_correction_is_safe(seen, aim, initial_aim):
                            lost += 1
                        elif shift > 0.002:  # 2 mm: below this the camera is repeating itself
                            aim = (float(seen[0]), float(seen[1]))
                            corrections += 1
                            largest = max(largest, shift)
                    if lost:
                        # Do not complete a blind approach toward an unknown or different blob.
                        break
                self.current = self.torch.tensor([[aim[0], aim[1], z]], device=self.device)
                self.env.step(self._action(self.current, grip))

        note = f"aimed at ({aim[0]:+.3f}, {aim[1]:+.3f})"
        if corrections:
            note += f", re-aimed {corrections}x (largest {largest * 1000:.0f} mm)"
        if occluded:
            note += f", {occluded} occluded look(s), retained the last scene-linked aim"
        if lost:
            note += f", {lost} look(s) could not find it"
        return StepResult(action=action, note=note, ok=lost == 0)

    # -- a whole plan ---------------------------------------------------------

    def settle_object(self, prim: str, cap: int = SETTLE_UNTIL_STILL_STEPS) -> int:
        """Step physics until ``prim`` stops moving. Returns the steps taken.

        Needed because the arm knocks its own target: measured, a ball is displaced 157 mm
        and takes 6.7 s to come to rest, and aiming at where it *was* closes the jaws on empty
        air. Zeroing its velocity instead would be simpler and a lie - the ball would stop
        unnaturally fast, in a place nothing put it.
        """
        import torch

        taken = 0
        for taken in range(1, cap + 1):
            if self.pause_hook is not None and not self.pause_hook():
                return taken - 1
            self.env.sim.step(render=False)
            if float(torch.linalg.norm(self.env.scene[prim].data.root_lin_vel_w[0])) < STILL_M:
                break
        self.env.scene.update(dt=self.env.physics_dt)
        return taken

    def run(self, plan, resolve=None, keys=lambda: set(), fps=1, retarget=None,
            observe=None) -> RunResult:
        """Run a planner Plan. ``resolve`` maps a scene id to a prim name for grabbing.

        ``keys`` is called after every step and returns the keys pressed in the viewer; a
        ``q`` aborts the run rather than the process, so the demo can report what it managed
        before the user gave up.

        ``observe`` is the closed loop. Given a prim it returns where the camera says that
        object is right now, and every ``above`` and ``down`` step becomes a ``track`` that
        re-reads it as the arm moves. Without it the two approach steps are driven once at a
        pose computed from a frame that is already stale, which is worth 57-371 mm on a ball.

        ``retarget`` is the older, blunter remedy and is kept for the case ``track`` cannot
        cover: the object has left the camera's view entirely, so re-aiming has nothing to aim
        at and the plan has to be rebuilt against a new scene. It returns
        ``(steps, resolve)`` - the replacement steps AND the resolver that goes with them,
        because a re-planned step's ``target`` is an id from a *new* camera frame. Reusing the
        old resolver silently aimed later steps at the wrong object: a run once logged
        "above #2" followed by "down onto #4" because the fresh plan's ids were resolved with
        the stale frame's map. The demo owns ``retarget`` and needs perception to build it.
        """
        out = RunResult()
        steps_left = list(plan.steps)
        index = 0
        reaimed = False
        tracked: dict[str, int] = {}
        tracked_grasp_pose: dict[str, np.ndarray] = {}
        tracked_grasp_center: dict[str, np.ndarray] = {}
        held_object_prim: str | None = None
        held_object_center: np.ndarray | None = None
        while index < len(steps_left):
            step = steps_left[index]
            prim = None
            # Two different checks on two different steps, and both are needed:
            #
            # ``close`` asks whether the object is still where the jaws are aimed. This is
            # what catches the arm knocking its own target away on the way in, which for a ball
            # is 192-371 mm - far past anything the jaws can close around.
            #
            # Only those two carry a check. ``carry`` and ``lower`` also hold CLOSE, so keying
            # on the gripper alone checked four times per grasp and reported "3 real grasps" for
            # a plan containing one.
            check = "reach" if step.action == "close" else (
                "follow" if step.action == "lift" else None)
            # The prim is needed by three different things, so it is resolved for every step
            # that names a target - not only for the two that carry a check. Resolving it only
            # for those two left ``prim`` None on ``above`` and ``down``, which is exactly the
            # pair of steps the closed loop is for, so the tracking silently never ran and the
            # demo reported zero re-aims while behaving exactly as open-loop as before.
            if resolve is not None and step.target is not None:
                prim = resolve(step.target)
            if step.action == "above" and prim is not None:
                choose_yaw = getattr(self, "set_object_grasp_yaw", None)
                if choose_yaw is not None:
                    choose_yaw(prim, step.pose)
            on_close = None
            if check is not None and prim is not None:
                on_close = (self.env.scene[prim].data.root_pos_w[0]
                            .detach().cpu().numpy().copy())

            # The approach and the descent are the two moves that knock the target, so they are
            # the two that re-aim. Everything after the jaws close has a fixed target and is
            # driven open-loop. The object-aware benchmark records displacement during each
            # phase so a future trajectory change can be checked against the same start pose.
            if observe is not None and step.action in ("above", "down") and prim is not None \
                    and step.pose is not None:
                steps = (TRACK_TRAVEL_STEPS if step.action == "above"
                         else TRACK_DESCEND_STEPS)
                pose = np.asarray(step.pose, dtype=float)
                tracking_item = next((item for item in getattr(self, "layout", ())
                                      if item.prim == prim), None)
                tracking_every = tracking_cadence_for(
                    getattr(tracking_item, "kind", ""), self.track_every_steps)
                steps = tracking_steps_for(
                    getattr(tracking_item, "kind", ""),
                    getattr(tracking_item, "size", None), step.action, steps)
                target_z = float(pose[2])
                clearance = None
                safe_z = target_z
                if step.action == "above" and self.approach_clearance_z is not None:
                    body = self.current[0].detach().cpu().numpy()
                    hold = self.hold_offset_world()[0].detach().cpu().numpy()
                    current_hold_z = float(body[2] + hold[2])
                    safe_z = max(target_z, float(self.approach_clearance_z))
                    # The stage height is a fixed workspace plane. Do not promote it to the
                    # robot's current height: a parked arm can be much higher than the target,
                    # and translating while descending diagonally is exactly what sweeps the
                    # forearm through a ball. First move vertically at the current XY, then
                    # translate at the stage height, then descend at the target XY.
                    if abs(current_hold_z - safe_z) > 0.005:
                        clearance_pose = (float(body[0] + hold[0]),
                                          float(body[1] + hold[1]), safe_z)
                        clearance = self.do("above", np.asarray(clearance_pose), step.grip,
                                            prim=prim, steps=STEP_COUNTS["above"])
                if clearance is not None and not clearance.ok:
                    result = clearance
                    result.note = "vertical rise to clearance height failed; " + result.note
                else:
                    self._tracking_prim = prim
                    self._tracking_action = step.action
                    result = self.track(
                        observe=lambda p=prim: observe(p), target_z=safe_z,
                        steps=steps, action=step.action, grip=step.grip,
                        every=tracking_every,
                        aim=(float(pose[0]), float(pose[1])))
                    if result.ok and step.action == "above" and safe_z > target_z:
                        body = self.current[0].detach().cpu().numpy()
                        hold = self.hold_offset_world()[0].detach().cpu().numpy()
                        lower_aim = lowering_start_aim(self, body, hold)
                        self._tracking_action = "down"
                        lowering = self.track(
                            observe=lambda p=prim: observe(p), target_z=target_z,
                            steps=TRACK_DESCEND_STEPS, action="down", grip=step.grip,
                            every=tracking_every,
                            aim=lower_aim)
                        if not lowering.ok:
                            result = lowering
                            result.note = "vertical settle from clearance height failed; " + result.note
                        else:
                            result.note = "clearance-height approach; " + result.note
                    self._tracking_prim = None
                    self._tracking_action = None
                    if result.ok and step.action == "down":
                        # Keep the hold point reached by the closed-loop approach. Reusing the
                        # planner's original close/lift XY would undo the visual corrections and
                        # send the jaws away from the object immediately before contact.
                        tracked_grasp_pose[prim] = np.asarray(
                            self.body_pose_from_current(), dtype=float).copy()
                        tracked_grasp_center[prim] = np.asarray(
                            self.env.scene[prim].data.root_pos_w[0]
                            .detach().cpu().numpy()[:2], dtype=float).copy()
                tracked[prim] = tracked.get(prim, 0) + 1
                out.tracked += 1
            else:
                pose = np.asarray(step.pose, dtype=float)
                grasp_body = tracked_grasp_pose.get(prim) if prim is not None else None
                if grasp_body is not None and step.action == "close":
                    result = self.do(
                        step.action, grasp_body, step.grip, prim=prim, on_close=on_close,
                        check=check, body_target=True,
                        feedback_target=tracked_grasp_center.get(prim))
                elif grasp_body is not None and step.action == "lift":
                    # Keep the actual body pose reached after close, and lift vertically by the
                    # planner's requested hold-point delta. The fixed tool offset cancels from
                    # this delta, so no frame conversion is needed.
                    body_lift = np.asarray(self.body_pose_from_current(), dtype=float).copy()
                    body_lift[2] += float(np.asarray(step.pose)[2] -
                                          np.asarray(steps_left[index - 1].pose)[2])
                    result = self.do(
                        step.action, body_lift, step.grip, prim=prim, on_close=on_close,
                        check=check, body_target=True,
                        feedback_target=tracked_grasp_center.get(prim))
                elif (held_object_prim is not None and held_object_center is not None
                      and step.action in {"carry", "lower"}):
                    # Carry the grasped object by translating the measured body pose by the
                    # object's requested world displacement. This preserves the real grasp
                    # transform and avoids re-deriving it from a jaw linkage whose midpoint
                    # moves during close.
                    object_target = np.asarray(step.pose, dtype=float)
                    body_target = (np.asarray(self.body_pose_from_current(), dtype=float)
                                   + object_target - held_object_center)
                    result = self.do(
                        step.action, body_target, step.grip, prim=prim, on_close=on_close,
                        check=check, body_target=True,
                        feedback_target=object_target[:2])
                    if result.ok:
                        actual_object = (self.env.scene[held_object_prim].data.root_pos_w[0]
                                         .detach().cpu().numpy().copy())
                        carry_error = object_pose_error(actual_object, object_target)
                        carry_tolerance = carry_follow_tolerance(
                            simulator_assisted=(getattr(self, "holding", None)
                                                == held_object_prim))
                        if carry_error > carry_tolerance:
                            result.ok = False
                            result.grasp = "lost"
                            result.note = (
                                f"deslizamiento durante el transporte: el centro se desvió "
                                f"{carry_error * 1000:.1f} mm de la pose objetivo "
                                f"(límite {carry_tolerance * 1000:.0f} mm)")
                            failures = getattr(self, "failures", None)
                            if isinstance(failures, dict):
                                failures[held_object_prim] = failures.get(held_object_prim, 0) + 1
                            if hasattr(self, "lost"):
                                self.lost += 1
                            failure_hook = getattr(self, "failure_hook", None)
                            if failure_hook is not None:
                                geometry_hook = getattr(self, "part_geometry", None)
                                geometry = (geometry_hook(held_object_prim)
                                            if geometry_hook else None)
                                failure_hook(held_object_prim, "lost", geometry)
                        else:
                            # Carry feedback and the next waypoint should start from the
                            # measured actor pose, not the commanded pose, so small tolerated
                            # compliance errors do not accumulate into a false slip report.
                            held_object_center = actual_object
                            follow_note = (f"seguimiento físico del objeto: "
                                           f"error {carry_error * 1000:.1f} mm")
                            result.note = f"{result.note}; {follow_note}".strip("; ")
                else:
                    result = self.do(step.action, pose, step.grip,
                                     prim=prim, on_close=on_close, check=check)
            if step.action == "lift" and prim is not None:
                tracked_grasp_pose.pop(prim, None)
                tracked_grasp_center.pop(prim, None)
                if result.grasp in {"real", "kinematic"}:
                    held_object_prim = prim
                    held_object_center = np.asarray(
                        self.env.scene[prim].data.root_pos_w[0]
                        .detach().cpu().numpy(), dtype=float).copy()

            # A failed closed-loop approach is a recoverable target loss, not a terminal
            # generic motion error. The caller can retreat, take a fresh camera frame, and
            # rebuild the same identity's pick plan. Without this classification, a ball that
            # rolls or is briefly occluded during approach aborts the entire command even
            # though no grasp has been attempted yet.
            if (not result.ok and not result.grasp and prim is not None
                    and step.action in {"above", "down"}):
                result.grasp = "stale"
                result.note = ("objetivo perdido durante la aproximación; se requiere nueva "
                               "percepción y replanificación; " + result.note).strip("; ")

            # The object has left the camera's view: nothing to re-aim at, so rebuild the plan
            # and its resolver together.
            if (retarget is not None and not reaimed and observe is not None
                    and not result.ok
                    and failed_step_can_retarget(step.action, held_object_prim is not None)
                    and index + 1 < len(steps_left)):
                reaimed = True
                fresh = retarget(step, prim)
                if fresh:
                    replacement, resolve = fresh
                    out.reaimed += 1
                    if step.action in ("close", "lift"):
                        # A missed contact or physical lift may rebuild a plan whose first
                        # waypoint is already ``down`` (the arm is at hover). Apply the
                        # alternate object-aware wrist pose here so recovery does not skip
                        # the ``above`` hook that normally selects it.
                        choose_yaw = getattr(self, "set_object_grasp_yaw", None)
                        if choose_yaw is not None and prim is not None:
                            choose_yaw(prim, step.pose)
                    steps_left = steps_left[:index + 1] + list(replacement)
            if result.grasp == "real":
                out.real_grasps += 1
            elif result.grasp == "kinematic":
                out.kinematic_grasps += 1
            elif result.grasp == "lost":
                out.lost_grasps += 1
            elif result.grasp == "stale":
                out.stale_grasps += 1
                out.stale_prims.append(prim)

            if step.action == "release":
                reset_yaw = getattr(self, "set_tool_yaw", None)
                if reset_yaw is not None:
                    reset_yaw(0.0)
                held_object_prim = None
                held_object_center = None

            pending_pregrasp = getattr(self, "_pregrasp_kinematic", None)
            if (step.grip == OPEN and self._held is not None
                    and self._held != pending_pregrasp):
                self.drop()

            out.steps.append(result)
            self.log(f"  {result.action:8s} {step.note}"
                      + (f"  [{result.note}]" if result.note else ""))

            pressed = set(keys()) if callable(keys) else set(keys)
            if "q" in pressed or "escape" in pressed:
                out.aborted = True
                out.aborted_at = f"{result.action} ({step.note})"
                self.log(f"  ABORTED by the user at {result.action}")
                break
            if result.grasp == "stale":
                # Stop here. The target is no longer safe to approach or the jaws never closed
                # on it, so the remaining waypoints would move to a destination with no payload.
                # The caller re-perceives and replans instead.
                out.aborted = True
                out.aborted_at = (
                    f"{result.action}: target needs reacquisition"
                    if step.action in {"above", "down"}
                    else f"{result.action}: nothing to close on")
                break
            if not result.ok:
                # A failed approach, descent, or recovery is not a safe intermediate state.
                # Continuing the remaining open-loop waypoints can push a missed ball farther
                # away, close on empty space, or report a placement that never happened. Stop at
                # the failing waypoint and let the caller take a fresh observation before it
                # decides whether to retry or report failure.
                out.aborted = True
                out.aborted_at = f"{result.action}: {result.note or 'acción fallida'}"
                self.log(f"  ABORTED at {result.action}: {result.note or 'acción fallida'}")
                break
            index += 1
        return out

    # -- settling -------------------------------------------------------------

    def settle(self, steps: int = 30) -> None:
        """Let physics finish after a release. Demo 05 uses 30.

        ``settle_hook`` is called every ``SETTLE_HOOK_EVERY`` steps, not every step:
        a frame is a render, and a render between two physics steps is the single most
        expensive thing the dashboard asks the simulation to do. 15 steps is a quarter
        of a second at 60 Hz, which is what a published picture needs to read as live
        and is 15x cheaper than every step. ``park`` goes through ``do`` and already
        publishes at its own rate, so it needs nothing here.
        """
        action = self._action(self.current, OPEN)
        with self.torch.no_grad():
            for index in range(steps):
                if self.pause_hook is not None and not self.pause_hook():
                    return
                self.env.step(action)
                self._carry_if_held()
                if self.settle_hook is not None and (index + 1) % SETTLE_HOOK_EVERY == 0:
                    self.settle_hook()

    def park(self, steps: int = PARK_STEPS, pose=None) -> StepResult:
        """Fold the arm out of the camera's view, by driving it there.

        Call this BEFORE placing the items, not after: the sweep across the table moves
        whatever is already down, and ``place_items`` is what puts everything back. See the
        note on ``PARK_POSE``. A camera-specific pose may be supplied when the standard
        front-view park point lies on the sight line to the target area.
        """
        target = PARK_POSE if pose is None else pose
        return self.do("above", np.asarray(target, dtype=float), OPEN, steps=steps)

    def park_by_joints(self, steps: int = 60) -> None:
        """demo 09's way of parking. Kept only to demonstrate why ``park`` is not this."""
        import isaac_io

        isaac_io.park_arm(self.env)
        self.current = self._gripper_pos().clone()
        del steps
