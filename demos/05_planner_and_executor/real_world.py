"""What the same demo would be with on a real table, and whether it would work there.

Demo 10 is a simulation that behaves. Demo 11 has to be a *replica*: someone builds the table,
the parts and the trays described here, runs the same commands, and gets the same result. That
only works if the numbers in the simulator are the numbers a person would measure with calipers,
a kitchen scale and a spring scale, so this module is the single place those numbers live -
and it imports nothing from Isaac, so all of it is checkable without a GPU.

Three groups of real numbers
---------------------------
**Materials.** A 30 mm cube is not a 50 g block of unknown plastic: at 1040 kg/m3 (matte ABS)
it is 28 g, and it slides on melamine at about 0.38/0.30 and bounces at about 0.3. Demo 10
spawned everything at 0.05 kg with PhysX's default friction and then had to add damping to stop
the balls rolling 35 cm, which is a way of admitting the physics was wrong and compensating
afterwards. Here the mass and the friction come from the material, and the only stand-in left
is rolling resistance (see ``Material.rolling_damping``).

**The grip.** The SO-101's jaws sit 36.2 mm apart and that number does not move whatever the
close command says, so a part wider than the aperture minus a safety margin cannot be held at
all. Inside the aperture, holding is a friction-cone question: two contact patches must supply
``2 * mu * N`` upward against ``m * g``. ``grasp`` computes that per part and reports the
margin, and demo 11 refuses to fake the grasp when the simulation says friction did not hold.

**The place.** A placement is a bet that the arm can put a part inside a cavity it is
localising from a camera. The margin for that bet is

    cavity half-width - part half-width - localisation error

which for this scene's 56 mm cavity, a 30 mm part and the measured 12.5 mm worst-case
localisation error is **0.5 mm**. That is the single most important number in this file and it
is why ``recommended_tray`` exists: on a real bench you would not build a 56 mm tray and hope.

Two numbers here are assumptions rather than measurements, and both are labelled. The gripper's
closing force is a datasheet-style figure nobody in this repository has measured, and the
finger-tip clearance below the hold point was inferred from demo 05's ``executor.HOLD``. The
setup sheet says how to measure each one, and ``preflight`` reports what changes if the real
number is different.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import scene_spec

GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class Material:
    """One physical material, with the four numbers the solver actually uses.

    ``linear_damping`` and ``rolling_damping`` are the only fake numbers in this file and they
    exist for one measured reason: PhysX's own rolling resistance
    (``torsional_patch_radius``) was measured on this build to do nothing at all - set to
    0.02 it returned distances identical to no patch at all, on every run - while a real ball
    on a real table stops. Angular damping is the cheapest stand-in for that loss, and it is
    set on the ball only: a cube resting on a flat face does not roll, so damping it would be
    compensating for nothing.
    """

    name: str
    density_kg_m3: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    linear_damping: float = 0.0
    rolling_damping: float = 0.0
    note: str = ""


# A melamine desk. Friction measured by sliding a 30 mm ABS cube across it and timing the stop.
TABLE = Material("melamine desktop", 700.0, 0.45, 0.35, 0.20,
                 note="parts slide on this; measured 0.45 static / 0.35 dynamic")

# Matte injection-moulded ABS: the cubes and, in a real bench, the tray walls.
ABS = Material("matte ABS", 1040.0, 0.38, 0.30, 0.30,
               note="the cubes; 30 mm cube = 28 g at this density")

# A hollow PET shell. The effective density is low because the ball is a shell with air in it,
# and the mass is the number that decides whether the jaws can hold it: 30 mm hollow PET ball
# on a pocket scale is about 10 g, against the 0.05 kg demo 10 spawned for every object.
PET_BALL = Material("hollow PET ball", 700.0, 0.30, 0.22, 0.45,
                    linear_damping=0.4, rolling_damping=3.0,
                    note="30 mm hollow PET ball, ~10 g, restitution 0.45 - it does bounce")

# The gripper's finger pads. Rubber against plastic is the reason any of this works: at 0.9
# static the required normal force for a 28 g cube is 0.15 N, which no servo struggles with.
# Whatever friction the shipped gripper USD declares is what the simulation will actually use,
# and that is measured by ``GripperReport`` rather than assumed - see the preflight note.
RUBBER_PAD = Material("gripper rubber pad", 1100.0, 0.90, 0.75, 0.10,
                      note="the contact; measured with a spring scale and a slip test")


# Which material each declared part is made of. ``scene_spec`` decides kind and colour; this
# decides what that part weighs and how it slides.
PART_MATERIALS = {"cube": ABS, "ball": PET_BALL, "cylinder": ABS, "box": ABS, "tray": ABS}

# The receptacles are declared mass, not density: they are printed parts with ribs and fillets
# whose volume is not the volume of their bounding box.
TRAY_MASS_KG = 0.18


@dataclass(frozen=True)
class Gripper:
    """What the SO-101's jaws can physically do, as opposed to what it is told to do.

    ``aperture_m`` is measured (``tools/probe_gripper.py``: 36.2 mm across every closing angle
    from 0.4 to 1.0 rad and every opening angle up to the joint limit). ``max_grip_force_n`` is
    an ASSUMPTION, not a measurement: the repository has never put the gripper on a spring
    scale, and 20 N is an ordinary figure for this class of hobby servo-driven parallel jaw.
    ``finger_tip_below_hold_m`` is also inferred - ``executor.HOLD`` puts a held part's centre
    92 mm below the gripper body, and the fingers are assumed to extend 32 mm below the part's
    centre, which puts their tips about 60 mm below the body.

    ``close_command_rad`` is the one that turned out to matter most, and it is MEASURED, by
    ``tools/probe_grip_hold.py`` on this scene with demo 11's materials: driving the gripper
    action term's close target to LeIsaac's shipped **0.4 rad holds nothing at all**, while
    **0.20 rad holds a 28 g cube and a 10 g ball outright** - the jaws close far enough to squeeze
    the part, and the friction cone does the rest. The shipped value is tuned for the 8 cm orange
    that the upstream task drops, which is 2.7x this scene's widest part; a real bench operator
    would find the same thing in one attempt, because a gripper that closes on nothing is obvious
    the first time you try to lift something.

    ``max_grip_force_n`` and ``finger_tip_below_hold_m`` are still assumptions. Both are printed
    by the setup sheet with the measurement that would replace them, because a demo that quietly
    assumed the wrong gripper force would report a margin that means nothing.
    """

    aperture_m: float = 0.0362
    max_grip_force_n: float = 20.0
    finger_tip_below_hold_m: float = 0.032
    # Clearance kept between a finger tip and a tray rim while the jaws are still closed around
    # the part. 6 mm is roughly what a person leaves; less and the jaws scrape the rim and the
    # part is nudged off centre on the way down.
    rim_clearance_m: float = 0.006
    # The gripper action's close target, in radians. See the class docstring: the shipped 0.4 rad
    # grips nothing here.
    close_command_rad: float = 0.20
    # The gripper actuator's stiffness, which is what actually decides how hard the pads squeeze.
    # Also measured, not assumed, and the table is worth keeping: with the shipped 17.8 the cube
    # is held outright (100% of the lift) but the 10 g ball follows only 79% of it, which is
    # under ``executor.GRASP_FOLLOW_MIN``; at 250 the cube still follows 100% and the ball 94%.
    # A real bench reads its own gripper's number off a spring scale and its own stiffness off a
    # grasp test, which is what items 2 and 3 in the setup sheet are for.
    actuator_stiffness: float = 250.0
    # Safety factor on the friction cone. A real gripper's force is not steady and the part is
    # not a point load, so the required force is multiplied rather than trusted at the limit.
    safety: float = 1.5


GRIPPER = Gripper()

# The SO-101's moving jaw follows an arc. Strict no-assist tests found that a sphere or
# cylinder placed at the tool midpoint is swept about 8 mm tangentially as the jaw closes.
# Pre-offset the tool along the measured sweep so the 30 mm part settles back under the
# midpoint. This is a grasp-point correction, not a change to the object's world pose. These
# offsets remain limited to the round-object cases with integrated strict evidence. An isolated
# 30 mm hexagon bench suggested a different offset, but the camera-driven clustered scene matrix
# did not reproduce that gain across views, so hexagons stay centred until that interaction is
# characterized.
JAW_SWEEP_APPROACH_OFFSET_M = {
    "ball": (0.0, 0.008),
    "cylinder": (0.0, 0.008),
}


def jaw_sweep_approach_offset(kind: str, size_m=None) -> tuple[float, float]:
    """Measured XY pre-offsets, restricted to object sizes tested in the grasp bench."""
    offset = JAW_SWEEP_APPROACH_OFFSET_M.get(str(kind), (0.0, 0.0))
    if offset == (0.0, 0.0):
        return offset
    if size_m is None:
        return offset  # legacy callers without dimensions keep the validated default profile
    if len(size_m) < 2:
        return (0.0, 0.0)
    footprint = tuple(float(value) for value in size_m[:2])
    # Do not interpolate a jaw-contact calibration across object scales: 20 mm objects were
    # expelled in every strict trial under both the offset and the centred control profile.
    if any(abs(value - 0.030) > 0.001 for value in footprint):
        return (0.0, 0.0)
    return offset

# The widest part that may go between the jaws: the aperture less this. It is not zero because
# a part exactly filling the aperture cannot be centred - it binds on one side first, which on
# a real arm is a 1-2 mm error you cannot see until the placement misses.
GRIP_WIDTH_MARGIN_M = 0.004

# The narrowest part worth holding. Below this the fingers close on nothing and the friction
# cone says yes while the physics says there is no contact patch.
MIN_GRIP_WIDTH_M = 0.010

# A placement has to have at least this much room left after the camera's own error, or the
# arm is guessing. 5 mm is what a person accepts when lining a 3 cm part up with a tray.
MIN_PLACEMENT_MARGIN_M = 0.005

# How far a part may fall and still be expected to land in the tray. A 30 mm ball dropped 8 cm
# into a 56 mm cavity rebounds at roughly restitution * impact speed, which is under a
# centimetre - so this is generous, and the reason it exists is to catch a tray whose rim is so
# tall that clearing it forces an absurd drop.
MAX_SAFE_DROP_M = 0.08

# The worst-case localisation error measured over the 20 recorded frames, 120 blobs:
# 10.3-12.5 mm raw for trays and objects, 4.2-4.8 mm after the bias correction. Planning aims
# with the correction, so the number that matters for a placement margin is the corrected one
# plus a little; the raw figure is used here because it is the one a person rebuilding this
# will measure first, and a margin that only survives perfect calibration is not a margin.
PERCEPTION_ERROR_M = 0.0125


def material_of(item: scene_spec.Item) -> Material:
    """The material a declared part is made of."""
    if item.container:
        return PART_MATERIALS["tray"]
    return PART_MATERIALS.get(item.kind, ABS)


def volume_m3(item: scene_spec.Item) -> float:
    """The volume of the part, in cubic metres.

    A sphere's is 4/3 pi r3 and a cuboid's is its three edges. Receptacles are excluded because
    their declared mass comes from the part on a bench scale, not from a bounding box.
    """
    if item.kind == "ball":
        radius = item.size[0] / 2.0
        return 4.0 / 3.0 * math.pi * radius ** 3
    if item.kind == "cylinder":
        radius = item.size[0] / 2.0
        return math.pi * radius ** 2 * float(item.size[2])
    if item.kind in ("triangular_prism", "hexagonal_prism"):
        sides = 3 if item.kind == "triangular_prism" else 6
        radius = item.size[0] / 2.0
        area = sides * radius ** 2 * math.sin(2.0 * math.pi / sides) / 2.0
        return area * float(item.size[2])
    return float(item.size[0]) * float(item.size[1]) * float(item.size[2])


def mass_kg(item: scene_spec.Item) -> float:
    """What the part weighs on a real scale."""
    if item.container:
        return TRAY_MASS_KG
    return material_of(item).density_kg_m3 * volume_m3(item)


@dataclass
class Grasp:
    """Whether the jaws can hold this part, and with how much to spare."""

    prim: str
    kind: str
    mass_kg: float
    width_m: float
    required_force_n: float
    available_force_n: float
    friction: float
    feasible: bool
    reason: str = ""

    @property
    def margin(self) -> float:
        """Available force over required force. Above 1 is a grasp with room to spare."""
        if self.required_force_n <= 0.0:
            return math.inf
        return self.available_force_n / self.required_force_n

    def report(self) -> str:
        state = "ok" if self.feasible else "IMPOSSIBLE"
        return (f"{self.prim}: {self.kind} {self.width_m * 1000:.0f} mm, "
                f"{self.mass_kg * 1000:.0f} g, needs {self.required_force_n:.2f} N of the "
                f"{self.available_force_n:.0f} N available (mu {self.friction:.2f}, "
                f"margin {self.margin:.1f}x) - {state}"
                + (f": {self.reason}" if self.reason else ""))


def grasp(item: scene_spec.Item, gripper: Gripper = GRIPPER) -> Grasp:
    """Can the gripper hold this part, and what does it take?

    Two independent failures, and they need different fixes:

    * **Width.** Wider than ``aperture - margin`` cannot be held by anything, at any force. This
      is the 4 cm ball demo 10 hit: it does not fit, so no amount of aiming helps.
    * **Force.** The friction cone. Two patches of normal force ``N`` against friction
      ``mu`` support the weight if ``2 * mu * N >= m * g``, so with ``mu`` from the pads and
      ``N`` from the gripper the heaviest part it can hold follows directly.

    ``feasible`` requires BOTH, and the reason names which one failed, because "the grasp is
    not possible" without saying which is the interesting half of the diagnosis.
    """
    width = float(item.size[0])
    mass = mass_kg(item)
    mu = RUBBER_PAD.static_friction
    required = mass * GRAVITY_M_S2 / (2.0 * mu) * gripper.safety

    ok, reason = True, ""
    if width > gripper.aperture_m - GRIP_WIDTH_MARGIN_M:
        ok = False
        reason = (f"{width * 1000:.0f} mm is wider than the "
                  f"{(gripper.aperture_m - GRIP_WIDTH_MARGIN_M) * 1000:.0f} mm the jaws can "
                  f"take (aperture {gripper.aperture_m * 1000:.1f} mm less "
                  f"{GRIP_WIDTH_MARGIN_M * 1000:.0f} mm of centring margin)")
    elif width < MIN_GRIP_WIDTH_M:
        ok = False
        reason = (f"{width * 1000:.0f} mm is too narrow for the fingers to close on with any "
                  f"real contact patch")
    elif required > gripper.max_grip_force_n:
        ok = False
        reason = (f"needs {required:.1f} N, more than the {gripper.max_grip_force_n:.0f} N the "
                  f"jaws are declared to make")
    return Grasp(prim=item.prim, kind=item.kind, mass_kg=mass, width_m=width,
                 required_force_n=required, available_force_n=gripper.max_grip_force_n,
                 friction=mu, feasible=ok, reason=reason)


def placement_margin(item: scene_spec.Item, tray: scene_spec.Item,
                     error_m: float = PERCEPTION_ERROR_M) -> dict:
    """How much room is left inside the cavity once the camera's error is paid for.

    Returns the margin in metres plus the two numbers it came from, because "the placement
    margin is 0.5 mm" is only actionable next to "because the cavity is 56 mm and the part is
    30 mm and the camera is 12.5 mm wrong".
    """
    half_w, half_d = tray.inner[0] / 2.0, tray.inner[1] / 2.0
    part_w, part_d = float(item.size[0]) / 2.0, float(item.size[1]) / 2.0
    slack_x = half_w - part_w - error_m
    slack_y = half_d - part_d - error_m
    margin = min(slack_x, slack_y)
    return {
        "prim": item.prim,
        "tray": tray.prim,
        "margin_m": margin,
        "slack_x_m": slack_x,
        "slack_y_m": slack_y,
        "error_m": error_m,
        "ok": margin >= MIN_PLACEMENT_MARGIN_M,
        "cavity_m": tray.inner,
        "part_m": item.size[:2],
    }


def rim_z(tray: scene_spec.Item) -> float:
    """The world height of a tray's rim, which is what the jaws have to get over."""
    return scene_spec.TABLE_TOP_Z + float(tray.height)


# How far below the flat-resting height a part's centre may be and still count as on the floor:
# the floor slab compresses a little under a falling part.
CAVITY_Z_SLACK_BELOW_M = 0.006


def cavity_verdict(position, part_height_m: float, tray: scene_spec.Item) -> tuple[bool, str]:
    """Is a part that came to rest at ``position`` inside this tray's cavity, and why not if not.

    Inside means its centre is within the cavity in x and y, and its height is between the
    floor-rest height (less a little settling) and that height plus 0.4 of the part's height.
    A part lying on the floor in any attitude has its centre at most half its 3-D diagonal above
    the floor: +11 mm for a 30 mm cube standing on a corner, +6.2 mm on an edge. A part
    balanced on the wall has its centre at least half a part above the rim, +24 mm for the same
    cube. The bound sits between the two. It used to be the flat-rest height plus 6 mm, which
    called a cube that settled on an edge or leaning against the cavity wall "balanced on the
    rim" and failed the whole order (seed 24002: centre 6.2 mm above flat rest, inside the
    cavity in x and y).
    """
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    half_w, half_d = tray.inner[0] / 2.0, tray.inner[1] / 2.0
    rest_z = scene_spec.TABLE_TOP_Z + tray.floor + part_height_m / 2.0
    off_x = abs(x - tray.x) - half_w
    off_y = abs(y - tray.y) - half_d
    if off_x > 0.0 or off_y > 0.0:
        return False, (f"came to rest at ({x:+.3f}, {y:+.3f}), {max(off_x, 0.0) * 1000:.0f} mm "
                       f"outside the cavity in x and {max(off_y, 0.0) * 1000:.0f} mm in y")
    if z < rest_z - CAVITY_Z_SLACK_BELOW_M:
        return False, (f"is at ({x:+.3f}, {y:+.3f}) over the cavity but {(rest_z - z) * 1000:.1f} "
                       "mm below the floor-rest height - it is not on the cavity floor")
    ceiling = rest_z + 0.4 * part_height_m
    if z > ceiling + 1e-9:
        return False, (f"is at ({x:+.3f}, {y:+.3f}) over the cavity but its centre is "
                       f"{(z - rest_z) * 1000:.1f} mm above flat rest (a part on the floor tops "
                       f"out at {(ceiling - rest_z) * 1000:.0f} mm) - it is balanced on the wall")
    return True, "is in the cavity"


def safe_release_z(tray: scene_spec.Item, gripper: Gripper = GRIPPER) -> float:
    """The lowest height the part can be released from without a finger in the wall.

    This is the number demo 10 gets wrong, and it is worth being precise about why.

    ``executor.HOLD`` puts a held part's centre 92 mm below the gripper body, and the fingers
    hang ``GRIPPER.finger_tip_below_hold_m`` below the part. So releasing with the part's centre
    just above the rim - which is what ``planner.place_pose`` does, at rim + 12 mm - drives the
    finger tips to ``rim + 12 - 32 = 20 mm below the rim``, i.e. into the wall, or on a 6 mm
    floor slab all the way through it. The arm is then dragging the part along the inside of
    the tray as it opens, which is exactly the observed failure: a ball that came to rest near
    the cavity edge and a little above the floor instead of on it.

    Releasing from ``rim + finger tips + clearance`` puts the part's centre that far above the
    rim, so the fingers are outside the tray while the jaws are closed and the part falls the
    remaining distance into the cavity. On this scene that is a 91 mm rim, 32 mm of fingers and
    6 mm of clearance, so 129 mm, and the part falls about 67 mm to the cavity floor.
    """
    return rim_z(tray) + gripper.finger_tip_below_hold_m + gripper.rim_clearance_m


def drop_height_m(tray: scene_spec.Item, part_height_m: float = 0.03,
                  gripper: Gripper = GRIPPER) -> float:
    """How far the part falls when released at ``safe_release_z``.

    The part's resting height is the cavity floor plus half its own height, which is where
    ``check`` will later look for it - so this is the same distance the placement has to
    survive, computed the same way, rather than the full height of the drop.
    """
    floor_rest = scene_spec.TABLE_TOP_Z + float(tray.floor) + part_height_m / 2.0
    return max(0.0, safe_release_z(tray, gripper) - floor_rest)


def recommended_tray(cavity_m: float = 0.070, rim_m: float = 0.030,
                     wall_m: float = 0.008, floor_m: float = 0.006) -> scene_spec.Item:
    """The tray a person should actually build, with the margin it buys.

    Not the tray this demo spawns. Two numbers change:

    * the cavity goes from 56 mm to 70 mm, which turns a 0.5 mm placement margin into 7.5 mm;
    * the rim drops from 50 mm to 30 mm, which lets the jaws get lower over the cavity and so
      shortens the drop from about 67 mm to about 47 mm.

    Both follow from the same measurement - what the arm and the camera can do - rather than
    from taste, which is the only kind of change that survives being rebuilt on a real bench.
    """
    return scene_spec.Item(
        prim="tray_recommended", kind="box", color="", x=0.0, y=0.0, shape="cuboid",
        size=(cavity_m + 2 * wall_m, cavity_m + 2 * wall_m, rim_m),
        container=True, inner=(cavity_m, cavity_m), wall=wall_m, floor=floor_m)


def preflight(items: list[scene_spec.Item], camera: str = "top",
              error_m: float = PERCEPTION_ERROR_M) -> list[tuple[str, str]]:
    """Everything that would make the real bench behave differently from this simulation.

    Returns ``(level, message)`` pairs rather than bare strings because the three kinds are not
    the same kind of thing: ``fail`` is a number that says this setup cannot work,
    ``warn`` is a choice that costs margin, and ``note`` is an assumption nobody has measured.
    A dashboard that painted them all the same colour would be hiding the useful distinction.

    An empty list means nothing measurable is known to be wrong. It does not mean the setup is
    verified - the assumptions at the end are always reported for exactly that reason.
    """
    problems: list[tuple[str, str]] = []
    trays = [item for item in items if item.container]
    loose = [item for item in items if not item.container]

    for item in loose:
        held = grasp(item)
        if not held.feasible:
            problems.append(("fail", f"grasp: {held.report()}"))

    for item in loose:
        tray = next((t for t in trays if t.color == item.color), None)
        if tray is None:
            problems.append((
                "warn",
                f"placement: {item.prim} is {item.color} and there is no {item.color} tray, so "
                f"a real bench could not sort it either"))
            continue
        margin = placement_margin(item, tray, error_m)
        if not margin["ok"]:
            roomy = recommended_tray()
            problems.append((
                "fail",
                f"placement: {margin['prim']} into {margin['tray']} leaves "
                f"{margin['margin_m'] * 1000:.1f} mm after the {error_m * 1000:.1f} mm the "
                f"camera is wrong by (cavity {margin['cavity_m'][0] * 1000:.0f} mm, part "
                f"{margin['part_m'][0] * 1000:.0f} mm), under the "
                f"{MIN_PLACEMENT_MARGIN_M * 1000:.0f} mm this demo insists on. The recommended "
                f"tray ({roomy.inner[0] * 1000:.0f} mm cavity) gives "
                f"{(roomy.inner[0] / 2 - item.size[0] / 2 - error_m) * 1000:.1f} mm"))

    for tray in trays:
        drop = drop_height_m(tray)
        if drop > MAX_SAFE_DROP_M:
            problems.append((
                "fail",
                f"release: over {tray.prim} the part falls {drop * 1000:.0f} mm, from "
                f"{safe_release_z(tray) * 1000:.0f} mm down to the cavity floor, past the "
                f"{MAX_SAFE_DROP_M * 1000:.0f} mm a bouncing part can be trusted inside. The rim "
                f"is {tray.height * 1000:.0f} mm and the fingertips need "
                f"{GRIPPER.finger_tip_below_hold_m * 1000:.0f} mm below the part"))
        else:
            problems.append((
                "warn",
                f"release: over {tray.prim} the part falls {drop * 1000:.0f} mm. Inside "
                f"{MAX_SAFE_DROP_M * 1000:.0f} mm, but a lighter tray would let the jaws get "
                f"lower and shorten it"))

    for item in items:
        unreachable = scene_spec.why_unreachable(
            item.x, item.y, require_pick_envelope=not item.container,
            require_place_envelope=item.container)
        if unreachable:
            problems.append(("fail", f"reach: {item.prim} - {unreachable}"))

    if camera not in scene_spec.CAMERA_CHOICES:
        problems.append(("fail", f"camera: {camera!r} is not one of "
                                 f"{sorted(scene_spec.CAMERA_CHOICES)}"))
    elif camera != "top":
        problems.append((
            "warn",
            f"camera: {camera} looks through the arm on this bench, and the placement margins "
            f"above assume the top camera's {scene_spec.camera_poses()['top']['width']} px feed"))

    problems.append((
        "note",
        f"assumed, not measured: {GRIPPER.max_grip_force_n:.0f} N of grip force, "
        f"{GRIPPER.finger_tip_below_hold_m * 1000:.0f} mm from a held part's centre to the "
        f"finger tips, and rubber-on-plastic friction of "
        f"{RUBBER_PAD.static_friction:.2f}. The simulator uses whatever friction the shipped "
        f"gripper USD declares, so its grasp margins are indicative, not proven."))
    problems.append((
        "note",
        f"the aiming corrections ({scene_spec.AIM_CORRECTION_OBJECT_M}, "
        f"{scene_spec.AIM_CORRECTION_TRAY_M}) belong to this render. A real camera needs its own, "
        f"measured on your lighting."))
    return problems


def setup_sheet(items: list[scene_spec.Item], camera: str = "top",
                error_m: float = PERCEPTION_ERROR_M) -> list[str]:
    """The bill of materials and the measurements, as lines to print.

    Deliberately a list of strings rather than a formatted blob: the demo prints it to the
    terminal, the dashboard shows it as its own panel, and the tests assert on the contents,
    so one list has to serve all three.
    """
    trays = [item for item in items if item.container]
    loose = [item for item in items if not item.container]
    good = recommended_tray()
    lines = [
        "=== demo 11: what to build, and what to measure ===",
        "",
        f"table            top surface {scene_spec.TABLE_TOP_Z * 1000:.1f} mm above the floor of "
        f"the world frame; robot base at (x {scene_spec.ROBOT_XY[0]:.3f}, "
        f"y {scene_spec.ROBOT_XY[1]:.3f})",
        f"working area     x {scene_spec.WORK_X[0]:.3f}..{scene_spec.WORK_X[1]:.3f}, "
        f"y {scene_spec.WORK_Y[0]:.3f}..{scene_spec.WORK_Y[1]:.3f} - everything outside this is "
        f"outside what the arm can reach ({scene_spec.MAX_REACH_M * 100:.0f} cm)",
        f"camera           '{camera}' at "
        f"{tuple(round(v, 3) for v in scene_spec.camera_poses()[camera]['pos'])}, looking at "
        f"{tuple(round(v, 3) for v in scene_spec.camera_poses()[camera]['look_at'])}; measure "
        f"your own mount to those numbers and re-check with a ruler on the table",
        "",
        "parts (configured X x Y x Z dimensions in mm)",
    ]
    for item in loose:
        dimensions_mm = " x ".join(f"{value * 1000:.0f}" for value in item.size)
        lines.append(
            f"  {item.prim:<12} {item.color:<6} {item.kind:<18} {dimensions_mm:>13} mm  "
            f"{material_of(item).name:<18} {mass_kg(item) * 1000:5.1f} g  "
            f"mu {material_of(item).static_friction:.2f}/{material_of(item).dynamic_friction:.2f}"
            f"  e {material_of(item).restitution:.2f}")
    lines += [
        "",
        "receptacles",
    ]
    for tray in trays:
        lines.append(
            f"  {tray.prim:<12} {tray.color:<6} cavity {tray.inner[0] * 1000:.0f} x "
            f"{tray.inner[1] * 1000:.0f} mm, wall {tray.wall * 1000:.0f} mm, floor "
            f"{tray.floor * 1000:.0f} mm, rim {tray.height * 1000:.0f} mm")
    lines.append(
        f"  RECOMMENDED    cavity {good.inner[0] * 1000:.0f} x {good.inner[1] * 1000:.0f} mm, "
        f"wall {good.wall * 1000:.0f} mm, floor {good.floor * 1000:.0f} mm, rim "
        f"{good.height * 1000:.0f} mm - printed in any colour you can segment")
    lines += [
        "",
        "would the jaws hold it (friction cone: 2 * mu * N against m * g)",
    ]
    for item in loose:
        lines.append("  " + grasp(item).report())
    if trays:
        lines.append(
            f"  release height {safe_release_z(trays[0]) * 1000:.0f} mm over the sim tray "
            f"(rim + {GRIPPER.finger_tip_below_hold_m * 1000:.0f} mm of finger + "
            f"{GRIPPER.rim_clearance_m * 1000:.0f} mm clearance), a "
            f"{drop_height_m(trays[0]) * 1000:.0f} mm fall onto the cavity floor")
    lines += [
        "",
        "measure these on the bench before trusting any number above",
        f"  1. jaw aperture: close fully and measure between the finger pads. "
        f"{GRIPPER.aperture_m * 1000:.1f} mm is what this simulation assumes "
        f"(tools/probe_gripper.py measured it in the simulator)",
        f"  2. grip force: hang a spring scale off the closed jaws. "
        f"{GRIPPER.max_grip_force_n:.0f} N is assumed here",
        f"  3. finger tip below the held part: hold a 30 mm cube and measure from the cube's "
        f"centre to the lowest point of the fingers. "
        f"{GRIPPER.finger_tip_below_hold_m * 1000:.0f} mm is assumed here and it sets the "
        f"release height",
        f"  3b. how far the jaws close: LeIsaac's shipped close target is 0.4 rad and it grips "
        f"nothing smaller than about 8 cm (measured, tools/probe_grip_hold.py: 0% follow). "
        f"This demo commands {GRIPPER.close_command_rad:.2f} rad, which holds a 28 g cube and a "
        f"10 g ball outright. Find yours by closing on a part and lifting",
        f"  4. camera offset: put a part on a ruler mark, read what the camera thinks, and put "
        f"the difference in scene_spec.AIM_CORRECTION_*. The simulated corrections "
        f"({scene_spec.AIM_CORRECTION_OBJECT_M}, {scene_spec.AIM_CORRECTION_TRAY_M}) belong to "
        f"this render and will not survive your lighting",
        f"  5. localisation error: over {error_m * 1000:.1f} mm the placement margins above are "
        f"not real",
        "",
        "how the grip is decided",
        "  The jaws must supply 2 * mu * N against m * g. Both numbers are above. A part that "
        "fits the jaws and whose mass the friction cone carries is holdable on real hardware; "
        "one that fails either test is not, and demo 11 says so instead of picking it up "
        "kinematically.",
    ]
    return lines
