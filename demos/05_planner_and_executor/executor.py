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
TRACK_MAX_SHIFT_M = 0.120

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
                + (f", {self.stale_grasps} stale (jaws closed on nothing)"
                   if self.stale_grasps else ""))


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

    def gripper_pixel(self, camera):
        """Where the gripper is in the image, for masking it out of perception."""
        pos = self._gripper_pos()[0].detach().cpu().numpy()
        return camera.project([float(pos[0]), float(pos[1]), float(pos[2])])

    # -- grasping -------------------------------------------------------------

    def grab(self, prim: str) -> None:
        """Start carrying an object kinematically. Demo 05's mechanism."""
        self._held = prim
        env = self.env
        env.scene[prim].write_root_velocity_to_sim(self._zeros)

    def drop(self) -> None:
        self._held = None

    @property
    def holding(self) -> str | None:
        """The prim currently being carried kinematically, if any."""
        return self._held

    def _carry_if_held(self) -> None:
        """Write the held object's pose from the gripper frame. Called every step."""
        if self._held is None:
            return
        from isaaclab.utils.math import combine_frame_transforms

        pos, quat = combine_frame_transforms(
            self.robot.data.body_pos_w[:, self.gripper_id],
            self.robot.data.body_quat_w[:, self.gripper_id],
            self._hold,
            self._upright,
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
           on_close=None, steps: int | None = None, check: str | None = None) -> StepResult:
        """Run one planner action to completion. ``pose`` is a world-frame gripper point.

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
                                    device=self.device) - self._hold
        start = self.current.clone()
        result = StepResult(action=action, note="")

        with self.torch.no_grad():
            for index in range(1, steps + 1):
                self.current = start + (goal - start) * min(1.0, index / (0.8 * steps))
                self.env.step(self._action(self.current, grip))

                if on_close is not None and index == steps and check == "reach":
                    # The jaws have finished shutting. Is the object still where this step
                    # aimed? If not, the descent knocked it away, the jaws closed on nothing,
                    # and the honest answer is that the grasp did not happen.
                    now = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
                    aimed = np.asarray(pose, dtype=float)[:2]
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
                    if follow > 1.0:
                        # Further than the gripper went, so nothing held it - the arm knocked
                        # it, and the object is no longer on the table. Measured twice at
                        # exactly 6417 cm, which is not a physical launch but the solver
                        # ejecting a ball that had been squeezed against a kinematic tray, so
                        # it is deterministic rather than chaotic.
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
                        aimed = np.asarray(pose, dtype=float)[:2]
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

        A correction larger than ``TRACK_MAX_SHIFT_M`` is not followed. Following it would mean
        the arm chasing a different object - the perception layer re-grounds on colour, kind and
        the last known position, so a blob belonging to something else can be picked up - and a
        grasp at the wrong place is worse than a refusal to move.

        Returns the number of corrections and the largest one, so the run can report honestly
        that the arm followed its target instead of pretending it aimed correctly first time.
        """
        start = self.current.clone()
        aim = aim if aim is not None else (float(start[0][0]), float(start[0][1]))
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

        with self.torch.no_grad():
            for index in range(1, steps + 1):
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
                        lost += 1
                    else:
                        shift = math.dist((seen[0], seen[1]), aim)
                        if shift > TRACK_MAX_SHIFT_M:
                            lost += 1
                        elif shift > 0.002:  # 2 mm: below this the camera is repeating itself
                            aim = (float(seen[0]), float(seen[1]))
                            corrections += 1
                            largest = max(largest, shift)
                self.current = self.torch.tensor([[aim[0], aim[1], z]], device=self.device)
                self.env.step(self._action(self.current, grip))

        note = f"aimed at ({aim[0]:+.3f}, {aim[1]:+.3f})"
        if corrections:
            note += f", re-aimed {corrections}x (largest {largest * 1000:.0f} mm)"
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
            on_close = None
            if check is not None and prim is not None:
                on_close = (self.env.scene[prim].data.root_pos_w[0]
                            .detach().cpu().numpy().copy())

            # The approach and the descent are the two moves that knock the target, so they are
            # the two that re-aim. Everything after the jaws close has a fixed target and is
            # driven open-loop, which is what it should be.
            if observe is not None and step.action in ("above", "down") and prim is not None \
                    and step.pose is not None:
                steps = (TRACK_TRAVEL_STEPS if step.action == "above"
                         else TRACK_DESCEND_STEPS)
                result = self.track(
                    observe=lambda p=prim: observe(p),
                    target_z=float(np.asarray(step.pose, dtype=float)[2]),
                    steps=steps, action=step.action, grip=step.grip,
                    aim=(float(step.pose[0]), float(step.pose[1])))
                tracked[prim] = tracked.get(prim, 0) + 1
                out.tracked += 1
            else:
                result = self.do(step.action, np.asarray(step.pose, dtype=float), step.grip,
                                 prim=prim, on_close=on_close, check=check)

            # The object has left the camera's view: nothing to re-aim at, so rebuild the plan
            # and its resolver together.
            if (retarget is not None and not reaimed and observe is not None
                    and not result.ok and step.action in ("above", "down")
                    and index + 1 < len(steps_left)):
                reaimed = True
                fresh = retarget(step, prim)
                if fresh:
                    replacement, resolve = fresh
                    out.reaimed += 1
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

            if step.grip == OPEN and self._held is not None:
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
                # Stop here. The rest of the plan is carry-lower-release for an object the
                # jaws never closed on, so running it would fly the gripper to the destination
                # and open on nothing - eight steps of motion and a placement that did not
                # happen. The caller re-perceives and replans instead.
                out.aborted = True
                out.aborted_at = f"{result.action}: nothing to close on"
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
                self.env.step(action)
                self._carry_if_held()
                if self.settle_hook is not None and (index + 1) % SETTLE_HOOK_EVERY == 0:
                    self.settle_hook()

    def park(self, steps: int = PARK_STEPS) -> StepResult:
        """Fold the arm out of the camera's view, by driving it there.

        Call this BEFORE placing the items, not after: the sweep across the table moves
        whatever is already down, and ``place_items`` is what puts everything back. See the
        note on ``PARK_POSE``.
        """
        return self.do("above", np.asarray(PARK_POSE, dtype=float), OPEN, steps=steps)

    def park_by_joints(self, steps: int = 60) -> None:
        """demo 09's way of parking. Kept only to demonstrate why ``park`` is not this."""
        import isaac_io

        isaac_io.park_arm(self.env)
        self.current = self._gripper_pos().clone()
        del steps
