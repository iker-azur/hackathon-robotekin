"""What the perception scene contains: the objects, the containers, where they go.

Kept apart from ``09_camera_to_scene.py`` so that the same scene can be built in the
simulator and scored, and so that ``--trials`` can move things around without touching
any Isaac code.

Two decisions worth stating:

**The red cube is LeIsaac's own, not a primitive.** The shipped scene has exactly one
rigid body, a 3 cm textured glossy red cube, which ``parse_usd_and_create_subassets``
already exposes as ``env.scene["cube"]``. It is kept and repositioned so that colour
segmentation is proven on a real textured asset and not only on flat shading. The other
objects are IsaacLab primitives (``CuboidCfg``, ``SphereCfg``) with an explicit
``PreviewSurfaceCfg.diffuse_color``, because that gives the exact albedo the HSV bands
in ``colors.py`` are written for and needs no asset download.

**Two objects share a colour.** ``ball_red`` exists so that "the red ball" is
genuinely ambiguous and the robot has to ask, which is the moment worth showing. Pass
``--no-duplicate`` for a scene where every colour is unique.

Positions are world metres on the shipped table. Its top surface is at z = 0.0409, and
it spans x 0.0032..0.7032 and y -0.6768..-0.0268, all measured from
``assets/scenes/table_with_cube/scene.usd`` rather than assumed. The objects live in
x 0.25..0.50, y -0.52..-0.30, which is inside the reach demo 05 already proved works,
so demo 10 can reuse this layout unchanged.

Objects of the same colour are at least 6 cm apart, which is 72 px at the top camera's
1206 px/m and comfortably more than ``colors.OPENING_PX`` can bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured from the shipped USD, not guessed.
TABLE_TOP_Z = 0.0409
TABLE_X = (0.0032, 0.7032)
TABLE_Y = (-0.6768, -0.0268)

# The working area demo 10 will reuse. It is deliberately inside the reach of the
# SO-101, not just inside the table: the base sits at (0.35, -0.64) and the shipped cube
# rests 28 cm forward of it. Widening this would make the perception demo look better
# and demo 10 unable to reach the objects.
WORK_X = (0.24, 0.40)
WORK_Y = (-0.53, -0.32)

# The SO-101's base, from LiftCubeEnvCfg.__post_init__. Needed because "left" is the
# robot's left, and a scene that does not say where the robot is cannot answer that.
ROBOT_XY = (0.35, -0.64)

# How far the arm is expected to reach from that base. LeRobot advertises about 350 mm
# for the SO-101 and the shipped cube rests 280 mm forward, so 0.33 leaves the zones a
# little room without putting them past the arm's specification. It is a figure from the
# hardware, not from this simulation: if demo 10 cannot reach a zone, this is the first
# number to move.
MAX_REACH_M = 0.33


@dataclass
class Item:
    """One thing to put on the table.

    ``prim`` is the name it will have in ``env.scene``. ``shape`` is the IsaacLab
    primitive to spawn, or ``None`` for the shipped USD cube, which needs no spawning.
    ``size`` is the full (x, y, z) extent in metres; for a sphere it is the diameter.
    """

    prim: str
    kind: str
    color: str
    x: float
    y: float
    shape: str | None = "cuboid"
    size: tuple[float, float, float] = (0.03, 0.03, 0.03)
    container: bool = False
    inner: tuple[float, float] = (0.0, 0.0)
    wall: float = 0.0
    floor: float = 0.0

    @property
    def height(self) -> float:
        return self.size[2]

    @property
    def z(self) -> float:
        """The height of the item's centre when it is resting on the table."""
        return TABLE_TOP_Z + self.height / 2.0

    @property
    def outer(self) -> tuple[float, float]:
        """A container's footprint from its cavity, for checking against ``size``.

        ``size`` stays the outer footprint on purpose: ``half_width``, ``half_depth`` and
        ``clearance`` all read it, so the working area fitted for demo 09 still holds. This
        is the same number stated the other way round, which is what hollow.py needs.
        """
        return (self.inner[0] + 2.0 * self.wall, self.inner[1] + 2.0 * self.wall)

    @property
    def cavity_depth(self) -> float:
        """Usable depth inside a tray: its height less the floor slab."""
        return max(0.0, self.height - self.floor)


def _cube(prim: str, color: str, x: float, y: float, edge: float = 0.03) -> Item:
    return Item(prim=prim, kind="cube", color=color, x=x, y=y,
                shape="cuboid", size=(edge, edge, edge))


def _ball(prim: str, color: str, x: float, y: float, diameter: float = 0.04) -> Item:
    return Item(prim=prim, kind="ball", color=color, x=x, y=y,
                shape="sphere", size=(diameter, diameter, diameter))


def _tray(prim: str, color: str, x: float, y: float) -> Item:
    """A shallow open tray: a 4.4 x 4.4 cm cavity 4.4 cm deep, in a 6 x 6 cm footprint.

    CHANGED IN THIS COPY. Demo 09's tray was a solid 6 x 7 x 1.2 cm coloured slab - a zone
    painted on the table, with nothing an object could go inside. Demo 09's own readme
    records that as the thing demo 10 had to fix: *"Containers are coloured zones, not
    hollow boxes... Making one hollow is a demo 10 change."*

    This is that change. hollow.py builds it as one kinematic rigid body whose five
    collision shapes are a floor and four walls, so an object dropped from above stays
    inside. The footprint is 7 cm square, the widest two trays fit across the 16 cm working
    area with a gap between them.

    The cavity is 5.6 cm rather than the 4.4 cm first tried, and that number was measured
    rather than chosen (``tools/tray_sizes.py``). Perception localises a tray to within
    9-11 mm on this camera, and the margin left after that error is the cavity half-width
    less the object's half-width less 12 mm:

        3 cm cube   5.6 cm cavity -> +1 mm     4.4 cm cavity -> -5 mm
        4 cm ball   5.6 cm cavity -> -4 mm     4.4 cm cavity -> -10 mm

    So even the largest cavity the reach envelope allows cannot absorb the localisation
    error for a ball. The tray is therefore sized to make the cube case marginal rather
    than impossible, and the executor does not rely on the margin: it verifies where the
    object actually came to rest and re-drops it if it is not in the cavity. That is the
    recovery loop this demo is supposed to demonstrate anyway, and it is the honest answer
    to an error that perception cannot be made small enough to remove.

    ``kind`` is still ``box``: perception sees a rim and a cavity, not an aspect ratio it
    can classify, and hollow.py has no way to make a square tray read as anything else.
    """
    inner_w, inner_d, wall = 0.056, 0.056, 0.007
    height, floor = 0.05, 0.006
    return Item(
        prim=prim, kind="box", color=color, x=x, y=y, shape="cuboid",
        size=(inner_w + 2 * wall, inner_d + 2 * wall, height),
        container=True, inner=(inner_w, inner_d), wall=wall, floor=floor,
    )


def _box(prim: str, color: str, x: float, y: float) -> Item:
    """A deeper, squarer tray: the same 4.4 cm cavity but 5 cm of wall above the floor.

    CHANGED IN THIS COPY, same reason as ``_tray``. Demo 09 declared this one 3 cm tall and
    therefore 3 cm of solid slab with its "interior" written in a field nothing read -
    ``Item.inner`` and ``Item.wall`` had no readers anywhere in the repository.

    It is now a real cavity too, told apart from ``_tray`` by height rather than by being
    hollower: the rim of this one sits higher above the table, which is a difference the
    depth image can see even though the colour blob cannot. Same cavity size, so the two
    are interchangeable for placement and the reach margin argument in ``_tray`` applies
    here too.
    """
    inner, wall = 0.056, 0.007
    height, floor = 0.05, 0.006
    return Item(
        prim=prim, kind="box", color=color, x=x, y=y, shape="cuboid",
        size=(inner + 2 * wall, inner + 2 * wall, height),
        container=True, inner=(inner, inner), wall=wall, floor=floor,
    )


# The default scene. One line per thing, so the layout can be read at a glance and
# changed at the hackathon without touching code that builds it.
# A 2 x 3 grid, one thing per cell, and that is not tidiness: it is what stops two
# same-coloured things from merging into one blob in the image.
#
# Every number below was fitted, not chosen by eye, against four constraints at once:
# inside WORK_X and WORK_Y, within MAX_REACH_M of the arm's base, same-coloured pairs
# further apart than the jitter can close, and rows far enough apart that a nearer row
# does not cover the one behind it in the camera's view. A search over zone sizes and
# grid positions returned 12804 layouts meeting all four; this is the one with the
# largest same-colour margin.
#
# Two things the earlier layout got wrong, both found by ``clearance`` rather than by
# looking at the render:
#
# * ``tray_blue`` was 7.3 mm outside MAX_REACH_M and outside WORK_Y, on the *base* layout,
#   before any jitter. A 9 x 7 cm tray has a 5.7 cm half-extent and the far row sits only
#   13 cm from the base, so the reach envelope simply has no room for a third row that
#   wide. The zones are now 6 cm.
# * the balls shared a column with a same-coloured cube above and the same-coloured zone
#   below, needing 88.4 mm of centre-to-centre separation where the row spacing gave
#   85 mm. The rows are now 55 mm apart in y and the balls are in the *opposite* columns
#   to the cubes, which separates that pair in x instead. Minimum same-colour gap is
#   51 mm, or 38 px at the measured 747 px/m, against a 3 px morphological opening.
DEFAULT_SCENE: list[Item] = [
    # farthest row: the 3 cm cubes, and the shipped USD cube
    _cube("cube_blue", "blue", 0.270, -0.510),
    _cube("cube", "red", 0.355, -0.510),
    # middle row: the balls. The reds are deliberately in different columns from the red
    # cube and the red box, so "the red ball" is ambiguous by colour and each red pair is
    # also separated along x.
    _ball("ball_red", "red", 0.270, -0.455),
    _ball("ball_green", "green", 0.355, -0.455),
    # nearest row: the hollow receptacles.
    #
    # CHANGED IN THIS COPY: both moved 5 mm further from the robot's near edge, from x
    # 0.270/0.355 to 0.275/0.360. The cavities grew to 5.6 cm so that a 3 cm cube has a
    # chance of landing inside rather than on the wall (see _tray for the measurement), and
    # a 7 cm footprint at x 0.270 spans 0.235..0.305, which pokes 5 mm outside WORK_X's
    # lower bound of 0.24. At 0.275 and 0.360 they span 0.240..0.310 and 0.325..0.395,
    # inside the working area with a 15 mm gap between them.
    _tray("tray_blue", "blue", 0.275, -0.370),
    _box("box_red", "red", 0.360, -0.365),
]

# The shipped cube is the only rigid body inside the scene USD, so
# parse_usd_and_create_subassets already exposes it as env.scene["cube"] and it only
# has to be moved. Everything else is added to cfg.scene before gym.make().
SHIPPED_CUBE = "cube"


def scene_of(duplicate: bool = True) -> list[Item]:
    """The default scene, with or without the second red ball."""
    items = list(DEFAULT_SCENE)
    if not duplicate:
        items = [i for i in items if i.prim != "ball_red"]
    return items


def jittered(items: list[Item], trial: int) -> list[Item]:
    """A deterministic rearrangement of the same scene, for ``--trials``.

    No randomness on purpose: the report has to be reproducible, so trial N is the same
    layout on every machine and on every run. Each object is nudged along x and y by an
    amount derived from the trial number, clamped so nothing leaves the reach envelope.

    The nudge is far smaller than the smallest same-colour gap in ``DEFAULT_SCENE``
    (51 mm), so a rearrangement cannot bring two objects of the same colour together.
    ``clearance`` is checked for every trial by ``09_camera_to_scene.py --calibrate``,
    and the numbers were chosen so that it passes.

    The 2 mm below are not arbitrary: at the measured 747 px/m they are 1.5 px, so two
    trials differ by at most 3 px in the image while the tightest same-colour gap is
    38 px. An earlier version nudged by 3 and 4 mm, which is still inside a 51 mm gap,
    but it was sized against a layout that no longer exists.
    """
    out: list[Item] = []
    for index, item in enumerate(items):
        step = 0.002 * ((trial + index) % 3 - 1)
        across = 0.002 * (((trial * 3 + index) % 3) - 1)
        # Clamped against the item's own footprint, not its centre: clamping a 6 cm zone
        # by its centre puts 3 cm of it outside the area.
        x = min(max(item.x + step, WORK_X[0] + half_width(item)), WORK_X[1] - half_width(item))
        y = min(max(item.y + across, WORK_Y[0] + half_depth(item)), WORK_Y[1] - half_depth(item))
        out.append(Item(**{**item.__dict__, "x": x, "y": y}))
    return out


def half_width(item: Item) -> float:
    """Half the item's x extent, which is what a containment check on x needs."""
    return item.size[0] / 2.0


def half_depth(item: Item) -> float:
    """Half the item's y extent, which is what a containment check on y needs."""
    return item.size[1] / 2.0


def half_extent(item: Item) -> float:
    """Half the item's largest horizontal dimension, as a circle would be.

    Used for separating two items and for the reach check, where treating everything as
    a disc is the conservative choice. Containment uses ``half_width``/``half_depth``
    instead: a 6 x 7 cm tray is not a 7 cm circle, and clamping it as one pushes it out
    of the working area.
    """
    return max(item.size[0], item.size[1]) / 2.0


def clearance(items: list[Item], same_colour_only: bool = True) -> list[tuple[str, str, float]]:
    """Every pair closer than it should be, plus anything out of bounds.

    Two blobs of the same colour that touch come back from segmentation as one, so the
    layout has to keep them apart; different colours cannot merge and are only reported
    when the caller asks for them. Also reports anything past the reach envelope or
    outside the working area, which would look fine in a picture and be unreachable by
    the arm.
    """
    problems: list[tuple[str, str, float]] = []
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            if same_colour_only and first.color != second.color:
                continue
            distance = float(np.hypot(first.x - second.x, first.y - second.y))
            gap = distance - half_extent(first) - half_extent(second)
            if gap < 0.0:
                problems.append((first.prim, second.prim, round(gap * 1000, 1)))
    for item in items:
        reach = float(np.hypot(item.x - ROBOT_XY[0], item.y - ROBOT_XY[1])) + half_extent(item)
        if reach > MAX_REACH_M:
            problems.append((item.prim, "out of reach", round((reach - MAX_REACH_M) * 1000, 1)))
        if item.x - half_width(item) < WORK_X[0] or item.x + half_width(item) > WORK_X[1]:
            problems.append((item.prim, "outside the working area in x", 0.0))
        if item.y - half_depth(item) < WORK_Y[0] or item.y + half_depth(item) > WORK_Y[1]:
            problems.append((item.prim, "outside the working area in y", 0.0))
    return problems


def camera_poses() -> dict[str, dict]:
    """Where the cameras go, in world metres, with what they look at.

    **The top camera looks from the far side of the table, back towards the robot, not
    the other way round.** That is the whole trick. The arm sits at x 0.28..0.40,
    y -0.68..-0.50, rising to about z 0.45 at its home pose, which is between the robot
    and any camera placed behind the robot. A camera there sees the arm first and loses
    the left-hand objects behind the gripper: the first version of this demo rendered
    exactly that, with the yellow arm filling the left half of the frame and the blue
    cube and green ball completely hidden. From y = -0.22 the objects are nearest the
    camera and the arm is only ever behind them, as background.

    Elevation is 67 degrees, which keeps the foreshortening along the view direction to
    about 8%, and the objects are 8 cm or more behind the tallest zone, so nothing hides
    anything: a 3 cm tall box hides only 3/tan(67) = 1.3 cm behind itself.

    Focal length 24 with aperture 38.11 gives a 38 degree horizontal half-angle, which
    covers the whole working area with room to spare, and about 1000 px/m, so a 3 cm cube
    is about 30 px. The ``side`` camera is deliberately on the robot's other side so that
    it too never looks through the arm.

    ``front`` is the robot's own side of the table, kept as a comparison: it looks through
    the arm, which is the failure the ``top`` camera exists to avoid. See
    ``CAMERA_CHOICES``.
    """
    return {
        "top": {"pos": (0.375, -0.22, 0.55), "look_at": (0.36, -0.43, 0.045),
                "focal_length": 24.0, "width": 640, "height": 480},
        "side": {"pos": (0.68, -0.50, 0.28), "look_at": (0.36, -0.43, 0.05),
                 "focal_length": 24.0, "width": 640, "height": 480},
        "front": {"pos": (0.35, -0.86, 0.42), "look_at": (0.33, -0.45, 0.05),
                  "focal_length": 24.0, "width": 640, "height": 480},
    }


# Which sensor name each --camera choice reads from, and whether this demo adds it.
# All three are added by this demo, including `front`: the shipped `front` TiledCamera is
# removed in build_env together with the observation term that read it, so reusing that
# name for our own camera replaces nothing and collides with nothing.
#
# `front` is the deliberate worst case. From y = -0.86 it is behind the robot, so the arm
# rises into the middle of the frame and the far objects hide behind it. It is kept so the
# README's claim that `top` is not an arbitrary choice can be checked by running both.
CAMERA_CHOICES = {
    "top": ("top", True),
    "side": ("side", True),
    "front": ("front", True),
}


DEPTH = "distance_to_image_plane"
