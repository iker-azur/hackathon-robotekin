"""Demo 12 — one SO-101 arm: real joints, real IK, real state.

This is the whole robot interface. It never touches a root pose: it sends joint targets,
exactly as a real robot's controller would, and reads back what PhysX did.

The cartesian path uses Isaac Lab's ``DifferentialIKController`` in absolute pose mode,
which is the same solver LeIsaac's own SO-101 tasks use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import materials as M

# The SO-101's five arm joints, in the order the articulation exposes them.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Where the grasp frame sits relative to the jaw body's origin.
#
# Deliberately NOT a constant. It used to be, and three different numbers were carried for
# the same physical quantity: materials.JAW_PAD_LOCAL_X_M (-0.026 m, jaw local X),
# materials.FINGER_PAD_CENTRE_BELOW_JAW_M (0.023 m) and this module's own
# FINGER_TIP_BELOW_TOOL_M (0.025 m). They disagreed. And once the pads were re-hung from a
# different jaw axis (materials.JAW_PAD_AXIS) all three became wrong at once, because the
# offset is a property of the *built* geometry rather than a number anyone can pick.
#
# ``pad_centre_z`` and ``tool_z_for_part_centre`` below therefore measure where the pads
# actually are and solve for the tool height. Nothing downstream can be wrong about it.

# The tool is commanded to point straight down at a target. The SO-101's wrist has two
# pitch joints that can do that; ``wrist_roll`` is held at zero because the jaws only close
# on a part symmetric about the tool's roll axis, and rotating them about it buys nothing.
TOOL_DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)

# How far a single 60 Hz tick may ask a joint to move. 0.078 rad per tick is 4.7 rad/s,
# the STS3215's real no-load speed (materials.py), so the IK never asks the arm for more
# than the servo can actually deliver.
MAX_JOINT_STEP_RAD = 4.7 / 60.0

# How close to a stop the arm may be before the solver starts pushing it back. 0.15 rad is
# about 8.6 deg, which is wider than the steady-state droop of these gains so a healthy
# pose is never disturbed by this term.
JOINT_LIMIT_MARGIN_RAD = 0.15

# Damped-least-squares regularisation. Larger trades accuracy for stability near the
# arm's reach boundary; 0.015 rad keeps the tool within a millimetre or so of the target
# in the working envelope while staying well away from the singularity.
IK_DAMPING = 0.015

# Wrist-roll calibration, for squaring the pads up to a part before closing. See
# ``ArmController.align_pad_axis``: the relation is a rigid rotation about the vertical, so
# only its sign needs measuring and 0.08 rad is far more than enough to see which way it goes.
ROLL_CALIBRATION_NUDGE_RAD = 0.08
ROLL_CALIBRATION_TICKS = 12

# How close the pad axis has to get to its commanded heading. 0.01 rad is 0.6 deg, which on a
# 30 mm part puts the residual skew across the pads under 0.3 mm -- below the solver's own
# tracking tolerance, so aligning tighter than this buys nothing and costs descent time.
ROLL_ALIGN_TOLERANCE_RAD = 0.01

# The wrist-angle relation that points the tool straight down, fitted against the
# articulation's own forward kinematics over 72 poses. The SO-101 chain is planar, so this
# is affine in the two arm joint angles; the three constants are the fit, and
# ``tests/test_tool_down.py`` re-measures it.
# Original fitted relation (handoff). Points the tool down approximately while
# keeping position tracking convergent. Do not replace with the exact-orientation
# fit: forcing X straight up costs ~25 mm of position error on this 5-DOF arm.
WRIST_DOWN_OFFSET_RAD = 1.502306
WRIST_DOWN_PER_LIFT = -0.858021
WRIST_DOWN_PER_ELBOW = -0.944549


class ArmError(RuntimeError):
    """The arm could not do what was asked, for a stated physical reason."""


@dataclass
class ArmStatus:
    """A short, honest description of what the arm is doing right now."""

    label: str
    note: str = ""


class ArmController:
    """Drives one SO-101 articulation and reports its measured state.

    Constructed by :meth:`demo12.isaac_env.IsaacCell.arm`.
    """

    def __init__(self, cell, robot_id: str) -> None:
        import torch
        from isaaclab.controllers import (DifferentialIKController,
                                          DifferentialIKControllerCfg)

        self.cell = cell
        self.robot_id = robot_id
        self.artic = cell.robots[robot_id]
        self.device = cell.device

        self.arm_joint_ids = cell._arm_joint_ids[robot_id]
        self.grip_joint_id = cell._grip_joint_id[robot_id]
        self.grip_body_id = cell._grip_body[robot_id]
        self.jaw_body_id = cell._jaw_body[robot_id]

        # The end-effector frame is the **jaw**, not the gripper housing.
        #
        # Measured with the tool pointing down, the jaw sits 35 mm to the side of the
        # gripper body and only 7.6 mm below it: the fingers splay outward from the
        # housing rather than hanging beneath it. Aiming the housing at a part therefore
        # aims ~35 mm past it, and the arm cannot descend the last 120 mm without the
        # housing colliding with the table. Aiming the jaw puts the fingers on the part.
        self.ee_body_id = self.jaw_body_id

        self._root_pos = cell._root_pos[robot_id]
        self._root_quat_inv = cell._root_quat_inv[robot_id]

        # lambda_val 0.005 is a damping term. Lower values track the target more closely
        # and jitter more; this is the value LeIsaac's own SO-101 tasks ship.
        self.ik = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.005},
            ),
            num_envs=1,
            device=self.device,
        )

        self.status = "idle"
        self.note = ""
        self._wrist_integral = 0.0

        # Measured on first use by ``_calibrate_roll``; None means "not yet measured".
        self._roll_sign: float | None = None
        self._roll_gain_measured = 0.0

        # The wrist_roll that ``solve_ik`` commands. Zero unless ``align_pad_axis`` has
        # squared the pads up to a part, which is what keeps a commanded alignment from
        # being undone on the very next solve -- see ``solve_ik``.
        self._roll_command: float = 0.0

        # The effector Jacobian is a per-body tensor on the articulation view. It is read
        # fresh each solve rather than cached, because PhysX owns it.

    def _jacobian(self):
        """Geometric Jacobian of the grasp frame, restricted to the five arm joints.

        ``get_jacobians()`` indexes its body axis without the articulation root, so the
        array has one fewer entry than the articulation has bodies. Indexing it with a raw
        body index therefore only works by coincidence: for the gripper housing it happens
        to line up, for the jaw -- index 6 of 7 -- it raises "index 6 is out of bounds for
        dimension 1 with size 6". The offset is computed rather than hard-coded so this
        keeps working if a link is added.
        """
        view = self.artic.root_physx_view
        js = view.get_jacobians()
        n_bodies = len(self.artic.body_names)
        offset = n_bodies - js.shape[1]
        idx = self.ee_body_id - offset
        if not (0 <= idx < js.shape[1]):
            raise IndexError(
                f"no Jacobian for body {self.ee_body_id} ({self.artic.body_names[self.ee_body_id]})"
                f": the Jacobian holds {js.shape[1]} entries for {n_bodies} bodies"
            )
        return js[:, idx, :, self.arm_joint_ids]

    @property
    def tool_pos(self):
        """World position of the grasp frame: the jaw, where the fingers actually are."""
        return self.artic.data.body_pos_w[0, self.ee_body_id]

    @property
    def tool_quat(self):
        """World orientation of the grasp frame, shaped (1, 4)."""
        return self.artic.data.body_quat_w[0:1, self.ee_body_id]

    # -- state -----------------------------------------------------------

    @property
    def joints(self):
        return self.artic.data.joint_pos[0]

    @property
    def housing_pos(self):
        """World position of the gripper housing. Diagnostics only -- never a target."""
        return self.artic.data.body_pos_w[0, self.grip_body_id]


    def pad_axis_angle_rad(self) -> float:
        """World heading of the jaw's pad separation axis, in the horizontal plane.

        Measured from the jaw's own quaternion, so it is the direction the pads really
        present rather than one inferred from the joint angles.
        """
        import math

        import torch
        from isaaclab.utils.math import quat_apply

        axis = quat_apply(self.tool_quat[0],
                          torch.tensor([[0.0, 0.0, 1.0]],
                                       dtype=torch.float32,
                                       device=self.device))[0]
        return math.atan2(float(axis[1].item()), float(axis[0].item()))

    def align_pad_axis(self, target_angle_rad: float, max_ticks: int = 200) -> int:
        """Roll the wrist until the pads' separation axis points along ``target_angle_rad``.

        Why this has to exist. The pads close along the jaw's local Z, and at the grasp pose
        that axis lies at 34.8 deg to the world axes -- the arm arrives wherever its IK
        solution puts it, and nothing squares it up with the object. A square part then
        presents not its width but a slanted chord across the pads, and because the two
        faces are *unequally* far away along that chord (18.3 mm on one side, 26.3 mm on the
        other, for a 30 mm cube) the pads meet one corner long before the other face. They
        never squeeze; they punt. Measured with a 48 mm open span: the jaws stalled at
        40.6 mm and shoved the part 36 mm along the axis while the grip sensor reported
        103 N of contact.

        A real cell squares the gripper up to the object before closing. ``wrist_roll`` is
        the joint that does it, so that is what this uses. Rolling does not move the tool
        origin, so it can be done before a descent without disturbing the approach.

        The relation between wrist_roll and the axis heading is a rigid rotation about the
        vertical: linear, gain 1 in magnitude. Its **sign** is the USD's joint-frame
        convention rather than a fact about this arm, so :meth:`_calibrate_roll` measures
        it on the live articulation the first time this runs and caches it.
        """
        if self._roll_sign is None:
            self._calibrate_roll()

        ticks = 0
        for ticks in range(1, max_ticks + 1):
            err = target_angle_rad - self.pad_axis_angle_rad()
            # Wrap to (-pi, pi] so a target one turn away is a short move, not a long one.
            err = (err + math.pi) % (2.0 * math.pi) - math.pi
            if abs(err) < ROLL_ALIGN_TOLERANCE_RAD:
                break
            roll = float(self.joints[4].item()) + self._roll_sign * err
            lo, hi = self.artic.data.soft_joint_pos_limits[0, 4]
            roll = max(float(lo) + 0.01, min(float(hi) - 0.01, roll))
            joints = [float(v) for v in self.joints[:5]]
            joints[4] = roll
            self.command_joints(joints)
            self.cell.step()

        # Keep commanding the heading that was reached. ``solve_ik`` uses this instead of a
        # hard zero, otherwise the next descent tick rolls the pads straight back off-axis
        # and the alignment is undone before the jaws ever close -- which is exactly what
        # happened the first time: the alignment converged to -0.44 deg and the span then
        # stalled at 40.58 mm, bit-identical to the unaligned run.
        self._roll_command = float(self.joints[4].item())
        return ticks

    def _calibrate_roll(self) -> None:
        """Measure how a positive wrist_roll command moves the pad-axis heading.

        Done once per arm, on the live articulation, because the sign is the USD's joint
        convention rather than something to assume. Two short nudges and a difference is
        enough: the relation is a rigid rotation, so anything more precise is wasted.
        The arm is returned to where it was found.
        """
        base = self.pad_axis_angle_rad()
        before = [float(v) for v in self.joints[:5]]

        nudged = list(before)
        nudged[4] = before[4] + ROLL_CALIBRATION_NUDGE_RAD
        self.command_joints(nudged)
        for _ in range(ROLL_CALIBRATION_TICKS):
            self.cell.step()
        after = self.pad_axis_angle_rad()

        self.command_joints(before)
        for _ in range(ROLL_CALIBRATION_TICKS):
            self.cell.step()

        delta = (after - base + math.pi) % (2.0 * math.pi) - math.pi
        self._roll_sign = 1.0 if delta >= 0.0 else -1.0
        self._roll_gain_measured = abs(delta) / ROLL_CALIBRATION_NUDGE_RAD

    @property
    def pad_centre_z(self) -> float:
        """World Z of the midpoint between the two finger pads, measured.

        Read from the pad bodies' own positions, so it accounts for every link in the
        wrist and for however the pads happen to be hung. This is the point the gripper
        actually grabs with.
        """
        pad_l, pad_r = self.cell.jaw_pad_positions(self.robot_id)
        return 0.5 * (float(pad_l[2].item()) + float(pad_r[2].item()))

    def pad_offset_below_tool_m(self) -> float:
        """How far the pad centre currently sits below the jaw origin, in world Z.

        Measured rather than declared. A small negative value is normal: the jaw tilts a
        couple of degrees at the grasp pose (measured local X world-Z of -0.032), so the
        pads sit very slightly *above* the tool origin, not below it.
        """
        return float(self.tool_pos[2].item()) - self.pad_centre_z

    def tool_z_for_part_centre(self, part_centre_z: float) -> float:
        """Tool height that puts the finger pads at a part's centre height.

        Solved from the measured pad offset rather than a hard-coded drop, so moving the
        pads to a different jaw axis, or resizing them, cannot silently desynchronise the
        approach from the grasp.
        """
        return part_centre_z + self.pad_offset_below_tool_m()

    def jaws_closed(self) -> bool:
        """Whether the jaws have reached their closed target.

        Read from the simulated joint, not from the command, so "closed" means the fingers
        are where they were told to go.
        """
        # The real gripper is the linkage pads, not the asset's gripper joint.
        try:
            span = self.cell.jaw_pad_span_m(self.robot_id)
        except Exception:
            actual = float(self.artic.data.joint_pos[0, self.grip_joint_id].item())
            return abs(actual - M.GRIPPER_CLOSE_RAD) < 0.08
        return span < 0.018

    def set_jaws(self, open_: bool) -> None:
        """Open or shut the fingers.

        This drives the **linkage joints**, not the asset's ``gripper`` joint. The shipped
        gripper joint rotates a single solid slab and cannot close on anything -- see
        ``isaac_env._author_jaw_pads`` -- so the pads are two rigid bodies on prismatic
        joints and those are what a position target belongs on.

        Keeping the name ``set_jaws`` means every caller in the orchestration layer reads the
        same as before, and there is exactly one place in the demo that knows how the
        fingers actually work.
        """
        self.cell.set_fingers(self.robot_id, open_)

    # -- motion ----------------------------------------------------------

    def command_joints(self, joint_targets) -> None:
        import torch

        values = torch.tensor([joint_targets[:5]], dtype=torch.float32, device=self.device)
        self.artic.set_joint_position_target(values, joint_ids=self.arm_joint_ids)

    def solve_ik(self, target_world, max_step_rad: float = MAX_JOINT_STEP_RAD) -> "list[float]":
        """Joint angles that move the tool toward a world point, pointing straight down.

        **Position-only DLS, then the wrist is set to point the tool down.**

        Asking the differential solver for position *and* orientation in one shot does not
        converge here, and the failure looks like a broken arm rather than a bad
        formulation: the solver satisfies the orientation by swinging ``shoulder_pan``,
        which walks the tool straight past the target. Measured, it reached 32 mm at 0.17 s
        and then left, ending 225 mm away with the base joint at 1.2 rad.

        Splitting the two is what a real pick-and-place controller does. The three position
        rows of the Jacobian are well conditioned and converge; the downward orientation
        is then imposed by hand on ``wrist_flex``, whose value is a pure function of the
        two arm joints and therefore cannot fight the position solve.
        """
        import torch

        target = torch.tensor([[target_world[0], target_world[1], target_world[2]]],
                              dtype=torch.float32, device=self.device)

        current = self.artic.data.joint_pos[0, self.arm_joint_ids]
        jacobian = self._jacobian()

        # Position-only differential step: J is (1, 6, 5); take the translational 3 rows.
        j_pos = jacobian[:, 0:3, :]
        error = target - self.tool_pos
        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
        jjt = torch.matmul(j_pos, j_pos.transpose(1, 2))
        damping = torch.eye(3, device=self.device) * (IK_DAMPING ** 2)
        step = torch.matmul(
            torch.matmul(j_pos.transpose(1, 2), torch.linalg.inv(jjt + damping)),
            error.unsqueeze(-1)).squeeze(-1)[0]

        # Cap the per-tick joint change at the servo's real top speed.
        delta = torch.clamp(step, -max_step_rad, max_step_rad)

        # Keep the command inside the joints' limits, by **projection**, not by adding a
        # repulsion term.
        #
        # This used to be `proposed = current + delta + push`, where `push` shoved any
        # joint within JOINT_LIMIT_MARGIN_RAD of a stop back off it. That looks equivalent
        # and is not, and the difference is a frozen arm rather than a small tracking error.
        #
        # The damped step out here is large: with the Jacobian's rows at ~0.15 m/rad and
        # lambda 0.015, a 136 mm error produces a step of -0.41, +0.32 and -1.22 rad on
        # three joints. Clamped to the servo's real 0.078 rad/tick that becomes
        # [-0.078, +0.078, -0.078] -- the same magnitude as the push terms. When a joint
        # is already near a stop, the push therefore *exactly* cancels the clamped step and
        # the solve is a fixed point: it commands the arm the pose it is already in, every
        # tick, forever.
        #
        # Measured, before the fix: the arm froze on tick ~100 and then held
        # shoulder_lift 0.073 rad off its lower limit and elbow_flex 0.071 rad off its upper
        # limit for the remaining 400 ticks, with 136 mm of position error unchanged to the
        # micrometre. A projection cannot fight anything -- a step that would cross a limit
        # is simply truncated there, and every other joint still moves.
        lo = self.artic.data.soft_joint_pos_limits[0, self.arm_joint_ids, 0]
        hi = self.artic.data.soft_joint_pos_limits[0, self.arm_joint_ids, 1]
        proposed = torch.clamp(current + delta,
                               lo + JOINT_LIMIT_MARGIN_RAD,
                               hi - JOINT_LIMIT_MARGIN_RAD)

        # Point the tool straight down. The SO-101 chain is planar, so the wrist pitch is
        # fixed by the two arm joints. ``wrist_roll`` is whatever ``align_pad_axis`` last
        # settled on -- zero by default, and deliberately *not* always zero: rolling the
        # wrist is how the pads get squared up to a part before they close, and pinning it
        # to 0 here would undo that alignment on the next solve.
        lift, elbow = float(proposed[1]), float(proposed[2])
        wrist = self._wrist_for_down(lift, elbow)
        return [float(proposed[0]), lift, elbow, wrist, self._roll_command]

    def _wrist_for_down(self, lift: float, elbow: float) -> float:
        """``wrist_flex`` that puts the tool's axis parallel to world Z.

        The fitted planar relation ``wrist ~ a + b*lift + c*elbow``, measured against the
        articulation's own forward kinematics over 72 poses (``tests/test_tool_down.py``):
        accurate to about 5 deg across the working envelope.

        There is deliberately no feedback term here. An integral correction on the
        measured tilt was tried and it wound up: the wrist command saturates at its limit,
        the integrator keeps accumulating while it is saturated, and the arm ends up
        driving the tool *along* the ground with +Z pointing sideways at
        (-0.42, -0.66, -0.25). A saturating command and a free-running integrator cannot
        coexist.
        """
        raw = (WRIST_DOWN_OFFSET_RAD
               + WRIST_DOWN_PER_LIFT * lift
               + WRIST_DOWN_PER_ELBOW * elbow)
        lo, hi = self.artic.data.soft_joint_pos_limits[0, 3]
        return max(float(lo) + 0.01, min(float(hi) - 0.01, raw))

    def _measured_tilt_rad(self) -> float:
        """Signed angle of the tool axis away from vertical, in the arm's swing plane.

        Positive means the tool has tipped toward the arm's own forward direction, so the
        wrist has to rotate back by the same amount.
        """
        import math

        w, x, y, z = (float(v) for v in self.tool_quat[0])
        # world direction of the tool's local +Z axis
        ax = 2.0 * (w * y + x * z)
        ay = 2.0 * (y * z - w * x)
        az = 1.0 - 2.0 * (x * x + y * y)
        return math.atan2(math.hypot(ax, ay), az)

    def move_to(self, cell, target_world, *, open_: bool = True, max_ticks: int = 300,
                tol_m: float = 0.003, max_step_rad: float = MAX_JOINT_STEP_RAD) -> int:
        """Drive the tool toward a world point. Returns the ticks it took.

        Servo-like: each tick re-solves IK against the measured pose and re-sends a joint
        target. There is no trajectory to fall behind, and the arm simply stops when it
        arrives.

        ``max_step_rad`` caps how far the joints may be asked to move in one tick. The
        approach to a part wants a slower cap than free motion: the pads clear an
        axis-aligned 30 mm part by 3.1 mm a side, and arriving at the full 4.7 rad/s servo
        speed measured the part being knocked away at 1.09 m/s. A gentler descent clears
        it.
        """
        import torch

        target = torch.tensor([[target_world[0], target_world[1], target_world[2]]],
                              dtype=torch.float32, device=self.device)
        self.set_jaws(open_)
        ticks = 0
        for ticks in range(1, max_ticks + 1):
            self.command_joints(self.solve_ik(target_world, max_step_rad))
            cell.step()
            if float((target - self.tool_pos).norm()) <= tol_m:
                break
        return ticks

    def move_joints(self, cell, joint_targets, *, open_: bool = True, ticks: int = 60) -> None:
        """Hold a joint pose for a while. Used for retracting and parking."""
        self.set_jaws(open_)
        for _ in range(ticks):
            self.command_joints(joint_targets)
            cell.step()

    # -- reporting -------------------------------------------------------

    def tool_error_m(self, target_world) -> float:
        return math.dist(
            (self.tool_pos[0].item(), self.tool_pos[1].item(), self.tool_pos[2].item()),
            (target_world[0], target_world[1], target_world[2]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.robot_id,
            "tool": [round(float(v), 5) for v in self.tool_pos],
            "jointPosRad": [round(float(v), 4) for v in self.joints],
            "jawsRad": round(float(self.joints[self.grip_joint_id].item()), 4),
            "jawsClosed": self.jaws_closed(),
            "status": self.status,
            "note": self.note,
        }

