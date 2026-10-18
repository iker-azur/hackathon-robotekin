"""The scripted expert that produces the demonstrations.

The expert is the same Cartesian waypoint controller as
``demos/01_simulation_basics/05_cube_pick_and_place.py``: move above the cube,
descend, close, lift, carry sideways, put down, open, retract. It drives the
``so101_state_machine`` action space (8 numbers), so the joint targets it ends up
commanding are what the dataset records.

Why a script and not a human: a demonstration dataset has to be perfect. One
episode where the operator fumbles the grasp is enough to teach the policy to
fumble. A script gives the same grasp every time, which is exactly the property
imitation learning needs, and it needs no leader arm plugged in.

Why the cube is held kinematically: real friction grasping of the 3 cm cube was
unreliable in this simulator (demo 05's note). Once the gripper is closed, the
cube's pose is written directly every step, which keeps the demonstration about
moving the cube rather than about tuning contacts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import il_env as env_cfg  # noqa: E402


@dataclass(frozen=True)
class Phase:
    """One straight-line segment of the plan, expressed relative to the grasp point."""

    name: str
    x: float  # metres, relative to the grasp point in world x
    y: float  # metres, relative to the grasp point in world y
    z: float  # metres, relative to the grasp point in world z
    gripper: float  # 1.0 open, -1.0 closed
    steps: int


# The plan, in the order it runs. Times are counted in simulator steps, not
# seconds, because that is what the loop below steps through. Each phase moves
# continuously from the previous target to its own; none of them plateaus, for the
# reason given in ``action``.
#
#   go above cube   lift to a safe height over the cube
#   go down         descend onto it
#   close           shut the jaw
#   lift            rise back to the safe height
#   carry           move sideways to the drop spot, still high
#   put down        descend at the drop spot
#   open            release
#   retract         rise clear of the cube
#
# Why the plan is shaped this way, including two attempts that made it worse
# -----------------------------------------------------------------------
#
# A reactive policy is shown one observation and asked for one action, so the
# expert's action has to be a *function* of what the policy can see. This plan
# was measured not to be, and the measurement is what this comment is for: the arm
# passes through the same configuration in different phases - frames 299 and 300
# matched to 8e-6 degrees while the plan switched from ``lift`` to ``carry``, and
# frames 391 and 420 matched to 0 degrees while it switched from ``carry`` to
# ``put down``. Counting every pair of frames sharing a pose within 1 degree but
# demanding targets more than 1 cm apart gives **153 ambiguous pairs per episode**
# (``tools/18_measure_ambiguity.py``). The best any single-observation policy can
# do on those is average the two options, which is a target that goes nowhere -
# and that is what the rollout did.
#
# **Two attempts to remove them geometrically both failed, and are recorded here
# so they are not tried a third time.** Both were measured on 10 recorded
# episodes:
#
# 1. *Give every phase its own altitude* - 283 ambiguous pairs per episode, worse
#    than the 153 it started from. It left ``lift`` and ``carry`` sharing both a
#    height and a direction.
# 2. *Offset the carry sideways in y so the vertical segments stop overlapping* -
#    214 per episode, also worse. It made ``carry`` and ``put down`` traverse the
#    same diagonal line together, taking that pair from 37 to 113 per episode.
#
# The reason both fail is visible in the attribution: every surviving collision
# is between two phases that share the **same jaw state**, so the jaw cannot
# separate them either -
#
#   carry vs put down   37   jaw shut / shut
#   close vs lift       51   jaw shut / shut
#   lift vs carry       44   jaw shut / shut
#
# Separating those would mean the arm never passes through the same pose twice,
# which for a plan that has to approach, grasp, carry and release means giving up
# on the natural shape of the task. The information that actually distinguishes
# them is **which way the arm is moving**, and that is not in a single
# observation - it is in the difference between this frame and the last one. So
# the fix is a policy that sees a history, which LeRobot's ACT refuses
# (``configuration_act.py:149``) and PI0Config accepts. This plan is left as the
# simplest one that works, and the ambiguity is documented rather than hidden.
PLAN = [
    Phase("go above cube", 0.0, 0.0, env_cfg.ABOVE, env_cfg.OPEN, 100),
    Phase("go down", 0.0, 0.0, 0.0, env_cfg.OPEN, 100),
    Phase("close", 0.0, 0.0, 0.0, env_cfg.CLOSE, 60),
    Phase("lift", 0.0, 0.0, env_cfg.ABOVE, env_cfg.CLOSE, 40),
    Phase("carry", 0.10, 0.0, env_cfg.ABOVE, env_cfg.CLOSE, 120),
    Phase("put down", 0.10, 0.0, 0.0, env_cfg.CLOSE, 80),
    Phase("open", 0.10, 0.0, 0.0, env_cfg.OPEN, 40),
    Phase("retract", 0.10, 0.0, env_cfg.ABOVE, env_cfg.OPEN, 100),
]

# Where the plan puts the cube down, relative to where it started. The carry
# phase above matches this, so the two are deliberately kept apart in one place.
DROP_OFFSET_XY = (0.10, 0.0)

# How close the cube has to end up, in metres, for the episode to count. The
# expert lands the cube within a couple of millimetres of this (measured over the
# demos: the recorded final error is ~1 cm), so the tolerance has slack but still
# separates a real demonstration from a dropped cube.
PLACEMENT_TOLERANCE_M = 0.03


# How far into its phase a command starts, as a fraction of the phase. Without
# this the interpolation runs ``t = phase_step / steps`` and the first step of
# every phase commands the pose the arm is *already* in - measured on the
# recorded dataset, the target moves 1.6 mm between frame 0 and frame 1.
#
# That is not a small detail. It makes the first pair of every episode
# ``(rest pose) -> (rest pose)``, and every evaluation starts from the rest pose,
# so the policy is being taught the correct answer "do nothing" at precisely the
# state it is asked about. The result is a stable fixed point: the rollout
# freezes on step 1 and stays frozen for all 800 steps. It was measured - see
# ``results/diagnosis.md``.
#
# Starting at t > 0 removes the self-loop without reintroducing the plateau that
# the "no plateau" rule above exists to prevent: the last step of a phase is
# still exactly t = 1, and nothing lingers at the end.
#
# 0.13 is deliberately small - enough that the first command is a real
# displacement (for "go above cube", about 8 mm) and the arm visibly moves, but
# small enough that the phase still covers its full distance.
PHASE_LEAD = 15.0


class Expert:
    """Drive the teacher action space from the cube's actual position.

    ``bind`` must be called after ``env.reset()``: it places the cube on the spot
    it is given, reads back where it ended up, and turns the relative plan above
    into world waypoints.
    """

    def __init__(self, env):
        self.env = env
        self.phases = list(PLAN)
        self.phase_index = 0
        self.phase_step = 0
        self.current = None  # world position of the gripper body
        self.base_pos = None
        self.base_quat = None
        self.grip_quat = None
        self.hold = None
        self.grasp = None
        self.cube_start = None
        self.holding = False
        self.done = False
        self.upright = None
        self.quat_at_grasp = None

    # -- setup ---------------------------------------------------------------

    def bind(self, spot: tuple[float, float] | None = None, yaw: float = 0.0) -> tuple[float, float, float]:
        """Place the cube and precompute the world waypoints.

        ``spot`` is the (x, y) the cube is put on, chosen by
        ``il_env.sample_cube_spot``. ``yaw`` turns the cube about its own axis, in
        radians, which is what makes one demonstration's front-camera image look
        different from another's: the cube is the only thing in the scene that
        moves between episodes, so without the yaw the policy is being asked to
        answer "where is the cube" from an image that always looks the same.

        Returns the cube's centre position, which the demos print for the record.
        """
        import torch

        from isaaclab.utils.math import quat_from_euler_xyz, quat_inv

        robot = self.env.scene["robot"]
        cube = self.env.scene["cube"]
        self.upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.env.device)
        gripper_id = robot.find_bodies("gripper")[0][0]
        self._gripper_id = gripper_id
        self.last_action = None

        if spot is not None:
            resting = quat_from_euler_xyz(
                torch.zeros(1, device=self.env.device),
                torch.zeros(1, device=self.env.device),
                torch.tensor([yaw], device=self.env.device),
            )
            cube.write_root_pose_to_sim(
                torch.cat(
                    [
                        torch.tensor([[spot[0], spot[1], env_cfg.CUBE_REST_Z]], device=self.env.device),
                        resting,
                    ],
                    dim=-1,
                )
            )
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))

        self.base_pos = robot.data.root_pos_w
        self.base_quat = robot.data.root_quat_w
        self.grip_quat = quat_inv(self.base_quat)  # gripper square to the world
        self.hold = torch.tensor([env_cfg.HOLD], device=self.env.device)

        cube_start = cube.data.root_pos_w.clone()
        self.cube_start = tuple(float(v) for v in cube_start[0])
        # The cube's own orientation at the grasp, remembered so the kinematic
        # hold puts it back the way it was picked up rather than snapping it to
        # the world's axes, which would read as a jump in the recorded images.
        self.quat_at_grasp = cube.data.root_quat_w.clone()
        # The gripper body sits one HOLD offset away from the cube centre, so the
        # grasp point is simply the cube minus that offset.
        self.grasp = cube_start - self.hold
        self.current = robot.data.body_pos_w[:, gripper_id].clone()

        # Let the cube settle on its spot, otherwise the first waypoints are aimed
        # at a pose that is still changing.
        for _ in range(20):
            self.env.sim.step(render=False)
        self.env.scene.update(dt=self.env.physics_dt)

        self.phase_index = 0
        self.phase_step = 1
        self.holding = False
        self.done = False
        return self.cube_start

    # -- stepping ------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        if self.phase_index >= len(self.phases):
            return self.phases[-1]
        return self.phases[self.phase_index]

    @property
    def progress(self) -> str:
        if self.done:
            return "finished"
        return f"{self.phase.name} {self.phase_step}/{self.phase.steps}"

    def action(self):
        """The next 8-number IK action, or None once the plan is finished."""
        import torch

        from isaaclab.utils.math import quat_apply, quat_inv

        if self.done:
            return None

        phase = self.phase
        goal = self.grasp + torch.tensor([[phase.x, phase.y, phase.z]], device=self.env.device)
        start = self.current
        # The target moves on every step of the phase. An earlier version held
        # still for the last fifth of each phase, which is what put a fifth of all
        # recorded transitions at "stay exactly where you are". A policy trained
        # on that parks on the first such pose and never leaves, because from its
        # point of view the demonstration says so - and it does, until the plan
        # moves on to the next phase, which the policy has no way to know. Any
        # reactive policy has this failure; removing the plateaus removes the
        # transitions that cause it.
        #
        # ``PHASE_LEAD`` handles the same failure at the *start* of a phase, which
        # is the one that actually stops the rollout: see its comment. The phase
        # still ends at exactly t = 1 on its last step.
        t = (self.phase_step - 1 + PHASE_LEAD) / (phase.steps - 1 + PHASE_LEAD)
        self.current = start + (goal - start) * t

        target = quat_apply(quat_inv(self.base_quat), self.current - self.base_pos)
        action = torch.cat(
            [target, self.grip_quat, torch.tensor([[phase.gripper]], device=self.env.device)], dim=-1
        )

        # Demo 05's rule, kept exactly: the cube is pinned from the last step of
        # any closing phase onwards, and released the moment the jaw is asked to
        # open again. That is what stops the cube slipping inside a closed jaw.
        if phase.gripper == env_cfg.CLOSE and (self.holding or self.phase_step == phase.steps):
            self.holding = True
        elif phase.gripper == env_cfg.OPEN:
            self.holding = False

        self.phase_step += 1
        if self.phase_step > phase.steps:
            self.phase_index += 1
            self.phase_step = 1
            if self.phase_index >= len(self.phases):
                self.done = True

        self.last_action = action
        return action

    def after_step(self) -> None:
        """Pin the cube if it is held. Call immediately after ``env.step``."""
        if self.holding:
            self.pin_cube()

    def pin_cube(self) -> None:
        """Write the cube to its held pose, expressed in the gripper frame."""
        import torch

        from isaaclab.utils.math import combine_frame_transforms

        upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.env.device)
        pose = combine_frame_transforms(
            self.env.scene["robot"].data.body_pos_w[:, self._gripper_id],
            self.env.scene["robot"].data.body_quat_w[:, self._gripper_id],
            self.hold,
            # The cube is held in the orientation it was picked up in, so it does
            # not snap sideways when the kinematic hold takes over from friction.
            self.quat_at_grasp,
        )
        self.env.scene["cube"].write_root_pose_to_sim(torch.cat([pose[0], pose[1]], dim=-1))
        self.env.scene["cube"].write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))

    # -- reading the result ---------------------------------------------------

    def cube_position(self) -> tuple[float, float, float]:
        pos = self.env.scene["cube"].data.root_pos_w[0]
        return tuple(float(v) for v in pos)

    def base_position(self) -> float:
        """Height of the robot's ``base`` body, the reference of the task's term."""
        robot = self.env.scene["robot"]
        index = robot.data.body_names.index("base")
        return float(robot.data.body_pos_w[0, index, 2])

    def target(self) -> tuple[float, float]:
        """The drop spot, in world x and y: the cube's start plus the carry offset."""
        return (self.cube_start[0] + DROP_OFFSET_XY[0], self.cube_start[1] + DROP_OFFSET_XY[1])

    def placement_error(self) -> float:
        """How far the cube ended from the drop spot, in metres."""
        end = self.cube_position()
        target = self.target()
        return float(((end[0] - target[0]) ** 2 + (end[1] - target[1]) ** 2) ** 0.5)

    def height_above_base(self) -> float:
        """The cube's height above the robot base body, the task's own measure."""
        return self.cube_position()[2] - self.base_position()

    def succeeded(self) -> bool:
        """The demonstration's own success: the cube ended at the drop spot.

        Note this is *not* the task's own success term. ``LeIsaac-SO101-LiftCube``
        declares success as the cube 20 cm above the base body, which the plan
        used here never reaches: it only carries the cube 6 cm above the table
        and then puts it back down. The demos therefore judge the demonstration
        the way this repository already does in demos 05 and 11 - by where the
        cube ended up - and report the task's measure alongside it.
        """
        return self.placement_error() < PLACEMENT_TOLERANCE_M
