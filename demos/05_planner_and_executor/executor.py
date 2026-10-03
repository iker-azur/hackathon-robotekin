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

# How many steps to wait for a knocked object to stop rolling before re-aiming at it.
# Measured: 6.7 s (400 steps) for the worst ball, so the cap is generous rather than tight.
SETTLE_UNTIL_STILL_STEPS = 400

# Below this speed an object counts as stopped, so that it will not move appreciably during
# the camera capture that follows.
STILL_M = 0.001

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
# is nudged 30 mm of the lift's 60 mm and then drops out, which a "did it move at all" test
# happily calls a real grasp. The measured follow fraction is 0.51 in that case.
GRASP_FOLLOW_MIN = 0.8


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

    @property
    def ok(self) -> bool:
        return not self.aborted and all(s.ok for s in self.steps)

    def summary(self) -> str:
        if self.aborted:
            return f"aborted at {self.aborted_at} after {len(self.steps)} step(s)"
        return (f"{len(self.steps)} step(s), {self.real_grasps} real grasp(s), "
                f"{self.kinematic_grasps} kinematic"
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
                        self.grab(prim)
                        result.grasp = "kinematic"
                        result.note = (f"followed {follow * 100:.0f}% of the lift "
                                       f"(moved {moved * 1000:.1f} of {travel * 1000:.1f} mm), "
                                       f"carrying kinematically")
                    else:
                        result.grasp = "real"
                        result.note = (f"followed {follow * 100:.0f}% of the lift "
                                       f"({moved * 1000:.1f} of {travel * 1000:.1f} mm)")

                if grip == CLOSE or self._held is not None:
                    self._carry_if_held()
        return result

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

    def run(self, plan, resolve=None, keys=lambda: set(), fps=1, retarget=None) -> RunResult:
        """Run a planner Plan. ``resolve`` maps a scene id to a prim name for grabbing.

        ``keys`` is called after every step and returns the keys pressed in the viewer; a
        ``q`` aborts the run rather than the process, so the demo can report what it managed
        before the user gave up.

        ``retarget`` is called once, after a hover that precedes a descent onto something.
        The arm knocks its own target on the way in - a ball is displaced 157 mm and rolls for
        6.7 s - so the descent needs re-aiming at where the object ended up rather than where
        it was seen. The callback is given the step about to run and returns a replacement for
        the rest of the plan, or None to carry on with the original. It is the demo's job, not
        the executor's, because it needs perception.
        """
        out = RunResult()
        steps_left = list(plan.steps)
        index = 0
        reaimed = False
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
            if check is not None and resolve is not None and step.target is not None:
                prim = resolve(step.target)
                on_close = (self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()
                            if prim else None)
            else:
                on_close = None

            result = self.do(step.action, np.asarray(step.pose, dtype=float), step.grip,
                             prim=prim, on_close=on_close, check=check)

            # Re-aim once, at the first descent, after the hover that preceded it. This is
            # the point where the arm has knocked the target and it has finished rolling.
            if (retarget is not None and not reaimed and step.action == "above"
                    and index + 1 < len(steps_left) and steps_left[index + 1].action == "down"):
                reaimed = True
                fresh = retarget(step, resolve(step.target) if (
                    resolve and step.target is not None) else None)
                if fresh:
                    out.reaimed += 1
                    steps_left = steps_left[:index + 1] + list(fresh)
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
        """Let physics finish after a release. Demo 05 uses 30."""
        action = self._action(self.current, OPEN)
        with self.torch.no_grad():
            for _ in range(steps):
                self.env.step(action)
                self._carry_if_held()

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