"""Demo 12 — physical parameters for the simulated cell.

Every number here is either (a) measured from the SO-101 USD / this project's own probes,
or (b) a published handbook value, and is labelled as such. Nothing is a tuning constant
chosen to make a grasp succeed.

Provenance
----------
S1  Read directly out of ``so101_follower.usd`` (LeIsaac 0.4.0 asset) with ``UsdPhysics``.
    These are not estimates: they are the joint limits and body masses the simulator loads.
S2  Measured by this project. ``demos/05_planner_and_executor/real_world.py`` records the
    measurement and the tool that produced it; the constants are reproduced here so Demo 12
    does not import a 30 kB module from a different demo.
S3  Published engineering values (CRC Handbook of Chemistry and Physics; Ashby & Jones,
    *Engineering Materials*). Used for materials this project never bench-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Gravity and time
# ---------------------------------------------------------------------------

# S3 — standard gravity, exact definition (CGPM 1901 / SI 1999).
GRAVITY_M_S2 = 9.80665

# PhysX integration step. This is the *physics* rate, not the control rate: the
# orchestrator runs at 60 Hz and takes four of these per tick (see isaac_env.CELL_STEPS).
#
# 240 Hz is the single biggest lever on whether a friction grasp holds. A 28 mm cube held
# between two pads is a two-point contact with a very small lever arm; at 60 Hz PhysX
# resolves it coarsely enough that the part creeps while it is being carried, and at
# 240 Hz it does not. Sub-stepping is the documented remedy for exactly this class of
# contact instability, and it is cheaper than raising the solver iteration count.
PHYSICS_DT_S = 1.0 / 240.0

# Control rate. The orchestrator issues joint targets at this rate.
CONTROL_DT_S = 1.0 / 60.0
CELL_STEPS_PER_CONTROL_STEP = 4


# ---------------------------------------------------------------------------
# SO-101 Follower articulation  (source: so101_follower.usd, read with UsdPhysics)
# ---------------------------------------------------------------------------

# S1 — limit cones authored in degrees in the USD.
JOINT_LIMIT_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

# S1 — MassAPI on each rigid body of the USD. Sums to 0.632 kg, which matches the
# published SO-101 follower weight, so the asset is dimensionally real.
BODY_MASS_KG: dict[str, float] = {
    "base": 0.147,
    "shoulder": 0.100,
    "upper_arm": 0.103,
    "lower_arm": 0.104,
    "wrist": 0.079,
    "gripper": 0.087,
    "jaw": 0.012,
}
ARM_MASS_KG = sum(BODY_MASS_KG.values())

# S1 — the USD authors no PhysicsDrive, so the actuators are Isaac Lab implicit
# actuators from ``SO101_FOLLOWER_CFG``. These are LeIsaac's shipped values.
STS3215_EFFORT_LIMIT_NM = 10.0

# S3 — the Feetech STS3215 datasheet gives a no-load speed of 0.222 s per 60 deg, which is
# 4.7 rad/s. LeIsaac ships 10 rad/s, roughly twice the real servo's top speed; using the
# real figure is what makes the arm's motion look like a servo and not a snap, and it keeps
# the pads from arriving at a part faster than contact can be resolved.
STS3215_VELOCITY_LIMIT_RAD_S = 4.7

# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------

# S2 — measured by ``tools/probe_gripper.py``. The two finger bodies sit 36.2 mm apart at
# EVERY close angle from 0.4 to 1.0 rad and every open angle to the joint limit: the
# fingers rotate about their mounts, the span is fixed. What the command changes is how
# far the pads swing in.
GRIPPER_APERTURE_M = 0.0362

# Fully closed span of the pads' inner faces.
#
# Read straight out of ``so101_follower.usd`` with pxr: ``/colliders/jaw/moving_jaw_so101_v1/mesh``
# has 84,810 points with a local x extent of -12.30 .. +10.00 mm, i.e. **22.3 mm** across,
# and it is continuous through the middle -- 14,886 points sit within 4 mm of x = 0,
# spanning the full z range of +/-24 mm. So the shipped "jaw" collision is one solid slab,
# not two fingers, and a solid slab cannot straddle anything: it presses down on top of a
# part. That is the reason this gripper could not pick anything up, and it is geometry,
# not tuning.
GRIPPER_CLOSED_SPAN_M = 0.0223

# ---------------------------------------------------------------------------
# Finger pad geometry, in the jaw body's own frame
# ---------------------------------------------------------------------------
#
# The finger linkage: two pads, each a rigid body joined to the jaw by a prismatic joint,
# so they slide in and out under a real joint drive. That is what makes the gripper a
# mechanism rather than a shape -- see ``isaac_env._author_jaw_pads`` for the four designs
# that were measured to failure first, and why each one is closed.
#
#   closed inner-face separation = 2 * CLOSED_HALF_M              = 12.0 mm
#   open   inner-face separation = 2 * (CLOSED_HALF_M + TRAVEL_M) - FINGER_PAD_THICK_M
#
# (The closed span carries the pad thickness term and the open one does not, because the
# pads are mirrored either side of the jaw: their centres are 2 * CLOSED_HALF_M apart when
# shut, and their inner faces are a further FINGER_PAD_THICK_M closer together.)
#
# **Why TRAVEL_M is 18 mm and not the 12.5 mm it used to be.**
#
# The pads must open wider than the part is *across the separation axis*, and for a square
# part that is not its width. The arm's pad axis is not aligned to the parts: measured at
# the grasp pose it lies at 34.8 deg to the world axes, so a 30 mm cube presents
#
#     30 mm * (|cos 34.8| + |sin 34.8|) = 41.8 mm
#
# across the pads -- its half-diagonal, 21.2 mm from the centre to each corner.
#
# With the old 12.5 mm travel the mechanism opened to 37 mm, which is 4.8 mm narrower than
# what it has to straddle. Measured consequence: the pads could not pass the part's top
# corner, one pad caught it and stalled at 60.3 N -- exactly
# ``JAW_PAD_EFFORT_LIMIT_N`` -- while the other showed 0.0 N. That one-sided jam twisted
# ``wrist_roll`` to 0.014 rad, the jaws then closed to 14.6 mm on the corner instead of the
# faces, the grip sensor still read 33 N, and the part did not move on the lift at all.
#
# A gripper that reads a nonzero grip force while holding nothing is the failure mode this
# criterion exists to catch, and it is why criterion 4 checks the part's height and not
# just the force.
#
# 18 mm of travel opens to 48 mm, which clears the 41.8 mm diagonal by 3.1 mm a side.
JAW_PAD_CLOSED_HALF_M = 0.009
JAW_PAD_TRAVEL_M = 0.018

# ---------------------------------------------------------------------------
# WHICH jaw axis the pads slide along -- and this was wrong until Demo 12 Step 0
# ---------------------------------------------------------------------------
#
# S2 -- **measured**, ``demo12/tools/probe_grasp_pose.py``, 28 poses across the reachable
# envelope, artifact in ``out12/pose_probe.json``. At every part pose the jaw's world
# directions of its own local axes are:
#
#     local X -> ( 0.571,  0.821, -0.032)   horizontal
#     local Y -> ( 0.018,  0.027, +0.9995)  VERTICAL   <-- world up
#     local Z -> ( 0.821, -0.571, -0.000005) horizontal
#
# The jaw frame is essentially world-aligned with **local Y pointing up**, which is what
# you would expect: the wrist's roll axis is vertical, so the jaw inherits it.
#
# The pads were originally authored to slide along local **Y**, on the strength of a prose
# claim in ``DEMO12_HANDOFF.md`` -- "at the grasp pose local Y is world -Y (horizontal)".
# Measured, that claim is false, and so is the other half of it ("local X is world +Z
# (up)"): the two were swapped. The pads therefore closed **straight up and down**, which
# is the entire reason grip force was zero. Pressing down on a part on a bench produces a
# normal force against the bench, not against the part, and no friction cone at all.
#
# Sliding along local Z instead gives a pad separation axis with a measured world Z
# component of -5e-06 -- horizontal to five decimal places, at every pose in the sweep.
# That is the fix, and it costs one string: no arm kinematics change, no world tuning, and
# no loss of tracking.
#
# Held as a named constant rather than a literal because it is the single most load-bearing
# fact in the gripper and it was wrong once already.
JAW_PAD_AXIS = "Z"

# The jaw axis the pads are dropped along, now that "down" is known to be local -Y.
# See JAW_PAD_AXIS: the previous build used local X, which is horizontal, so this offset
# never lowered the pads at all.
JAW_PAD_DROP_AXIS = "Y"

# How far below the jaw origin the pads sit, along JAW_PAD_DROP_AXIS.
#
# Two clearances pull against each other:
#
# *Clear the wrist.* Measured: with the pads' centre only 10.4 mm below the jaw origin they
# reached backwards into the wrist and put 84,975 N on the upper arm.
#
# *Clear the bench.* The pads must not touch the tabletop while descending. A pad resting
# on the bench is held there by PhysX: the servo demanded +0.3475 N -- three times the
# 12 g pad's own weight -- for 90 ticks and the pad did not move by a micron.
#
# 23 mm on a 30 mm part puts the pads 17 mm clear of the bench, which is also where a real
# gripper takes them: the upper third of the object, not its base.
#
# NOTE: the grasp *height* is not derived from this number. ``ArmController`` measures the
# pads' actual world position and solves for the tool height that puts them on the part.
# This constant only decides where the pads hang relative to the wrist.
JAW_PAD_DROP_M = -0.023

# Finger pad geometry. The scale is (depth, height, thickness) in jaw-local X, Y, Z.
#
# With JAW_PAD_AXIS = "Z" the separation axis is Z, so the *thin* dimension must be Z and
# the face the part is gripped on is the pad's Z face. The long 18 mm dimension is left on
# local X and the 10 mm height on local Y, which is vertical, so the pads present a level
# 18 x 10 mm patch per face with the wide side across the direction of travel.
FINGER_PAD_THICK_M = 0.006
FINGER_PAD_DEPTH_M = 0.018

# The linkage's actuator gains. Stiff enough that the pads stall on a part instead of being
# pushed aside by it, and damped enough that the pad does not ring when it bottoms out.
#
# These are tuning, stated as tuning. They are not derived from the SO-101's real finger
# actuator, which this asset does not model.
JAW_PAD_ACTUATOR_STIFFNESS = 4000.0
JAW_PAD_ACTUATOR_DAMPING = 80.0
JAW_PAD_EFFORT_LIMIT_N = 60.0
JAW_PAD_VELOCITY_LIMIT_M_S = 0.6

# The pad height is not cosmetic, and it is squeezed from both sides.
#
# The tool is commanded to the part's centre plus the centre offset, so the pads sit at
# (part centre + 7 mm +/- height/2). Both ends of that window matter:
#
# * Too tall and the tops reach the part's top corners. Measured with 30 mm pads on a 30 mm
#   part: the tops landed exactly on the corners and every descent shoved it, ending 100 mm
#   away and spun 51 deg.
# * Too tall and the bottoms reach the bench, which locks the pads down as described above.
#   Measured with 20 mm pads: the part was levered 16 mm sideways out of the grip on the lift
#   instead of rising, because the pads were jammed into the corner where part meets bench.
#
# 10 mm leaves 3 mm below the top face of a 30 mm part and 17 mm of bench underneath, while
# still presenting a 10 x 18 mm patch per face for friction -- 180 mm^2, ample for a part
# weighing 0.276 N.
FINGER_PAD_HEIGHT_M = 0.010

# Finger pad mass.
#
# S1 -- the jaw body's own 12 g, read out of ``so101_follower.usd``, for each of the two
# pads. They are bodies on the arm's own articulation, driven by the prismatic joint's
# implicit actuator, so PhysX generates the squeeze from the pad's real mass against a
# real part. There is no servo law in this demo and nothing is attached to anything.
FINGER_PAD_MASS_KG = 0.012

# The two ends of the asset's own ``gripper`` joint travel, used by the legacy jaw-collider
# path. Both are inside the joint's limits of -0.175 to 1.745 rad.
#
# These do NOT open and close the fingers. The pads are a prismatic linkage driven by their
# own joints -- see :data:`JAW_PAD_AXIS` and ``isaac_env.set_fingers`` -- and a revolute
# cannot change the distance between two points on a rigid body anyway (measured 42.74 mm
# at every angle from 0.2 to 1.4 rad). Kept because ``isaac_env.set_gripper`` still targets
# this joint and a controller needs to name both ends of a travel.
GRIPPER_OPEN_RAD = 0.20
GRIPPER_CLOSE_RAD = 1.00

# S2 — at the shipped 17.8 N.m/rad a cube is held but a 10 g ball follows only 79% of the
# lift; at 250 both are held outright. A sphere meets the pads at a point and a cube at a
# face, so the lighter part needs the stiffer servo to be carried at all.
GRIPPER_ACTUATOR_STIFFNESS = 250.0
GRIPPER_ACTUATOR_DAMPING = 12.0

# Not measured by this project, and tuned here from a measured step response.
#
# LeIsaac ships 17.8 N.m/rad, which is a torque-control gain. In position control the
# same servo droops under the arm's own weight by roughly 5 deg at full extension, because
# the error times the gain has to cover the gravity torque within the 10 N.m effort limit.
# 5 deg at a 0.2 m lever arm is 17 mm of tool error, which is larger than the grasp
# tolerance, so the arm cannot place a part where it was told to.
#
# Raising kp is the standard fix and is what a real position-controlled servo does: the
# stiffness term dominates and the steady-state error falls. Measured droop is reported by
# ``tests/test_tracking.py``; the value here was chosen to bring it under 1 deg without
# the drive ringing at the 240 Hz step.
ARM_ACTUATOR_STIFFNESS = 900.0
ARM_ACTUATOR_DAMPING = 45.0

# S2 — 90 steps of closed settling before the lift starts. At 50 the part is still being
# squeezed when the lift begins and the grasp is refused. At 60 Hz this is 1.5 s, which is
# the duration at which the jaws have actually reached their commanded angle.
GRIPPER_SETTLE_STEPS = 90

# S2 — how long the jaws are held open after a release before the arm retreats.
GRIPPER_RELEASE_STEPS = 25


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """A physical material: what it weighs, how it slides, how it bounces."""

    key: str
    label_es: str
    label_en: str
    density_kg_m3: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    provenance: str = "S3"
    note: str = ""


# S2 — melamine desktop. Friction measured by sliding a 30 mm ABS cube across it and
# timing the stop: 0.45 static, 0.35 dynamic.
TABLE_MATERIAL = Material(
    key="melamine",
    label_es="Mesa de melamina",
    label_en="Melamine desktop",
    density_kg_m3=700.0,
    static_friction=0.45,
    dynamic_friction=0.35,
    restitution=0.20,
    note="measured: 30 mm ABS cube slid across it and timed",
)

# S2 — matte injection-moulded ABS. The cubes. A 30 mm cube is 28 g at this density.
ABS = Material(
    key="abs",
    label_es="ABS mate",
    label_en="Matte ABS",
    density_kg_m3=1040.0,
    static_friction=0.38,
    dynamic_friction=0.30,
    restitution=0.30,
    note="30 mm cube = 28 g at this density",
)

# S2 — hollow PET shell. The effective density is low because the ball is a shell with air
# in it: a 30 mm hollow PET ball on a pocket scale is about 10 g. The mass is what decides
# whether the jaws can hold it, so it is modelled as 10 g, not as solid plastic.
PET_BALL = Material(
    key="pet",
    label_es="Bola PET hueca",
    label_en="Hollow PET ball",
    density_kg_m3=700.0,
    static_friction=0.30,
    dynamic_friction=0.22,
    restitution=0.45,
    linear_damping=0.4,
    angular_damping=3.0,
    note="30 mm hollow PET ball, ~10 g, restitution 0.45 - it does bounce",
)

# S2 — the gripper's finger pads. Rubber against plastic is the reason any of this works:
# at 0.9 static the required normal force for a 28 g cube is 0.15 N, which no servo
# struggles with.
RUBBER_PAD = Material(
    key="rubber",
    label_es="Almohadilla de goma",
    label_en="Gripper rubber pad",
    density_kg_m3=1100.0,
    static_friction=0.90,
    dynamic_friction=0.75,
    restitution=0.10,
    note="the contact; measured with a spring scale and a slip test",
)

# S3 — pine, for the wooden blocks. Not bench-tested by this project.
PINE = Material(
    key="pine",
    label_es="Pino",
    label_en="Pine",
    density_kg_m3=500.0,
    static_friction=0.40,
    dynamic_friction=0.30,
    restitution=0.25,
)

# S3 — 6061 aluminium, for the heavier blocks. Not bench-tested by this project.
ALUMINIUM = Material(
    key="aluminium",
    label_es="Aluminio 6061",
    label_en="Aluminium 6061",
    density_kg_m3=2700.0,
    static_friction=0.42,
    dynamic_friction=0.32,
    restitution=0.20,
)

PART_MATERIALS: dict[str, Material] = {
    "abs": ABS,
    "pet": PET_BALL,
    "pine": PINE,
    "aluminium": ALUMINIUM,
}

# Which material each part shape is made of by default. `cube` and `cylinder` are solid
# ABS; `ball` is a hollow PET shell, which is the only reason it weighs 10 g and not 15 g.
SHAPE_DEFAULT_MATERIAL: dict[str, str] = {
    "cube": "abs",
    "cylinder": "abs",
    "ball": "pet",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# The tabletop height Demo 11 measured in the shipped scene USD, kept so that the
# workspace this demo builds is dimensionally identical to the one measured against.
TABLE_TOP_Z_M = 0.0409

# The SO-101 follower's working envelope, as a horizontal distance from the *base*.
#
# These are not measured by hand. ``tests/test_ik_reach.py`` sweeps the arm with the real
# DLS solver across a grid of targets and reports the annulus it can actually hold, which
# is what these numbers are taken from.
#
# Note the near limit is much closer than Demo 11's figure of 0.195 m. That number is
# measured from the shoulder joint, which sits above and slightly in front of the base;
# quoted as a base-relative distance it puts a 0.19 m gap in the middle of the working
# area, and a cell laid out to satisfy it cannot put a tray within reach of the robot
# standing behind it.
MIN_REACH_M = 0.125
MAX_REACH_M = 0.335

# A part must fit between the pads with room to spare. This is the largest part the
# gripper can physically close on: aperture less a centring tolerance.
GRASP_WIDTH_LIMIT_M = GRIPPER_APERTURE_M - 0.006

# Narrowest part a friction grasp can be attempted on at all. Below this the pads
# barely touch and there is no normal force to build a friction cone from.
MIN_GRASP_WIDTH_M = 0.010


def cube_mass_kg(edge_m: float, material: Material) -> float:
    """Mass of a solid cube: exact, no bounding-box fudge."""
    return material.density_kg_m3 * edge_m**3


def ball_mass_kg(diameter_m: float, material: Material) -> float:
    """Mass of a *hollow shell* ball, which is what the PET parts are.

    A sphere is modelled as its bounding sphere. For a hollow shell the effective density
    is what matters and it is already carried by the material, so this is the same
    ``density * volume`` relation as the cube.
    """
    volume = 4.0 / 3.0 * 3.141592653589793 * (diameter_m / 2.0) ** 3
    return material.density_kg_m3 * volume


def cylinder_mass_kg(diameter_m: float, height_m: float, material: Material) -> float:
    volume = 3.141592653589793 * (diameter_m / 2.0) ** 2 * height_m
    return material.density_kg_m3 * volume


def part_mass_kg(kind: str, size_m: tuple[float, float, float], material: Material) -> float:
    """Mass of a part from its real volume and its material density."""
    if kind == "ball":
        return ball_mass_kg(size_m[0], material)
    if kind == "cylinder":
        return cylinder_mass_kg(size_m[0], size_m[2], material)
    return cube_mass_kg(size_m[0], material)


def required_normal_force_n(mass_kg: float, mu_static: float, safety: float = 2.0) -> float:
    """Normal force a friction grasp needs to carry ``mass_kg``.

    Two pads, so the friction cone gives ``2 * mu * N >= m * g``. The safety factor is
    2x so a part is not held right at the edge of slipping: carrying it needs the weight
    of the gripper's own acceleration on top, and a grasp at exactly 1.0 is a grasp that
    drops the moment the arm moves.
    """
    if mu_static <= 0.0:
        return float("inf")
    return safety * mass_kg * GRAVITY_M_S2 / (2.0 * mu_static)


def width_fits_gripper(width_m: float) -> bool:
    """Whether a part can physically be taken by this gripper."""
    return MIN_GRASP_WIDTH_M <= width_m <= GRASP_WIDTH_LIMIT_M


# ---------------------------------------------------------------------------
# Reach
# ---------------------------------------------------------------------------


@dataclass
class ReachEnvelope:
    """A robot's base pose and the annulus it can work in.

    Deliberately *not* a constant: Demo 11 measured MAX_REACH_M against one specific base
    position on one specific table. Here every robot carries its own pose, so the
    envelope follows the robot.
    """

    base_xy: tuple[float, float]
    min_reach_m: float = MIN_REACH_M
    max_reach_m: float = MAX_REACH_M

    def distance_m(self, xy: tuple[float, float]) -> float:
        return ((xy[0] - self.base_xy[0]) ** 2 + (xy[1] - self.base_xy[1]) ** 2) ** 0.5

    def contains(self, xy: tuple[float, float]) -> bool:
        d = self.distance_m(xy)
        return self.min_reach_m <= d <= self.max_reach_m

    def radial_clearance(self, xy: tuple[float, float]) -> float:
        """Metres inside the envelope. Negative means unreachable."""
        d = self.distance_m(xy)
        return min(d - self.min_reach_m, self.max_reach_m - d)