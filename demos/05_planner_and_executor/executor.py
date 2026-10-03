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

# The IK action's gripper is binary: >= 0 opens to 1.0 rad, < 0 closes to JAW_CLOSED_RAD.
# Demo 05's reasoning: LeIsaac's default of 0.4 rad suits an 8 cm orange and leaves a 1 cm
# gap on a 3 cm cube; fully shut (-0.17) makes the hooked tip ride over the cube's top edge.
OPEN, CLOSE = 1.0, -1.0
JAW_CLOSED_RAD = 0.15

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

# What fraction of the gripper's own travel an object must follow during the lift to count as
# held. Absolute distance is not enough: a 4 cm ball closed on by jaws that shut to 0.15 rad
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

# Long enough to converge. 150 steps was not enough for the farthest candidate tried and
# left the arm mid-motion when the next command arrived, which then fought it.
PARK_STEPS = 220


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

    @property
    def ok(self) -> bool:
        return not self.aborted and all(s.ok for s in self.steps)

    def summary(self) -> str:
        if self.aborted:
            return f"aborted at {self.aborted_at} after {len(self.steps)} step(s)"
        return (f"{len(self.steps)} step(s), {self.real_grasps} real grasp(s), "
                f"{self.kinematic_grasps} kinematic")


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
           on_close=None, steps: int | None = None) -> StepResult:
        """Run one planner action to completion. ``pose`` is a world-frame gripper point.

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

                if on_close is not None and index == steps:
                    # Checked at the END of the action, not during it, and judged as a
                    # fraction of how far the GRIPPER moved. A grasp that "took" because the
                    # closing jaws nudged the object is not a grasp; and an object that
                    # followed half the lift and then slipped out is not held either.
                    moved = self._moved(prim, on_close)
                    travel = float(self.torch.linalg.norm((goal - start)[0]))
                    follow = moved / travel if travel > 1e-6 else 0.0
                    if follow < GRASP_FOLLOW_MIN:
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

    def run(self, plan, resolve=None, keys=lambda: set(), fps=1) -> RunResult:
        """Run a planner Plan. ``resolve`` maps a scene id to a prim name for grabbing.

        ``keys`` is called after every step and returns the keys pressed in the viewer; a
        ``q`` aborts the run rather than the process, so the demo can report what it managed
        before the user gave up.
        """
        out = RunResult()
        for step in plan.steps:
            prim = None
            # The grasp is judged on the LIFT, not on the close. The close is where the jaws
            # shut and nudge whatever is between them, which reads as a grasp that is not one.
            # The lift is where "is it actually in the gripper" gets answered.
            #
            # Only the lift carries a grasp check. ``carry`` and ``lower`` also hold CLOSE, so
            # keying on the gripper alone counted one grasp four times over - a run reported
            # "3 real grasps" for a plan containing a single grasp.
            if step.action == "lift" and resolve is not None and step.target is not None:
                prim = resolve(step.target)
                on_close = (self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()
                            if prim else None)
            else:
                on_close = None

            result = self.do(step.action, np.asarray(step.pose, dtype=float), step.grip,
                             prim=prim, on_close=on_close)
            if result.grasp == "real":
                out.real_grasps += 1
            elif result.grasp == "kinematic":
                out.kinematic_grasps += 1

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