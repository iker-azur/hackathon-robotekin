"""Where things are, and how that changes while the robot is working.

Three things move in this demo, and they are deliberately different:

**At startup and between rounds**, the whole layout is re-randomised (``randomize``). The
point is to prove the robot is reading the camera rather than memorising coordinates: if the
same command works across twenty layouts, nothing in the pipeline can be reading a hardcoded
position. ``scene_spec.jittered`` already did a version of this for demo 09, but it nudges
by at most 2 mm, which at the measured 747 px/m is 1.5 px - too small to change any decision
downstream. ``randomize`` moves things by centimetres.

**Mid-task**, a single object is disturbed (``disturb``): given a shove and left to slide or
roll. This is the case the recovery loop exists for. The arm is mid-plan with a pose computed
from the old position, so a disturbance means the plan is stale before it is executed. The
demo's answer is to re-perceive and replan rather than to finish a plan that is known to be
wrong.

Both are seeded. ``randomize`` takes an integer seed and ``disturb`` takes a
``random.Random``, so a run is reproducible from its seed alone, which is what makes a
failure report worth anything.

The layout constraints are not decoration. Two same-coloured things that touch come back from
segmentation as a single blob, so they must stay apart; two things of any colour that overlap
are interpenetrating at spawn and PhysX pops them apart, which is exactly what made the
balls appear to "roll" during development. Anything past ``MAX_REACH_M`` looks correct in a
picture and cannot be picked up. ``randomize`` therefore rejects a layout that breaks any of
these rather than handing it to the simulator and hoping.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

import scene_spec
from scene_spec import Item

# How far an item may move from its default position, per axis, when randomised.
#
# Centimetres, not the 2 mm of ``scene_spec.jittered``. The working area is 16 cm x 21 cm, so
# 18 mm is a real relocation - an object can change which row it is in - while still leaving
# every layout close enough to DEFAULT_SCENE that the command model's ambiguity cases (two
# reds, "the red ball") survive. Randomising freely instead is possible and is worth trying
# once the recovery loop is trusted; see ``randomize(spread=...)``.
DEFAULT_SPREAD_M = 0.018

# Same-coloured pairs must stay at least this far apart, CENTRE TO CENTRE.
#
# CHANGED from a gap between footprints to a distance between centres, and raised from 8 mm to
# 63 mm. The old number was a gap, so it allowed two same-coloured objects 9 mm apart in centre
# terms as long as their footprints did not literally touch - and 9 mm is nothing to
# segmentation: the blobs are one blob.
#
# 63 mm is measured, not chosen. The two blobs must clear each other in the image, and at the top
# camera's 747 px/m a tray's colour blob is about 47 px half-width and a loose object's about 16,
# so their centres need roughly 63 px of separation - which is 63 mm on the table. See
# ``scene_spec.DEFAULT_SCENE``, which lays the same-coloured pairs 80-150 mm apart for the same
# reason.
#
# This is what makes the demo deterministic under a random layout: rejection sampling means a
# layout that would merge two reds never reaches the simulator, instead of reaching it and
# producing a robot that confidently picks up the wrong thing.
MIN_SAME_COLOUR_GAP_M = 0.063

# Two items of any colour must not overlap at all.
MIN_GAP_M = 0.0

# How many layouts to try before giving up on a seed.
#
# Raised from 400 to 2000 because ``MIN_SAME_COLOUR_GAP_M`` went from a footprint gap to a
# 63 mm centre-to-centre separation, which is a much tighter constraint: the declared layout's two
# reds are 70 mm apart, so a jitter of ±18 mm can put them in the wrong order and only a fraction
# of random draws are valid. Measured: 300 seeds, none rejected and none fell back to the declared
# layout, with 2000 attempts the search never comes close to exhausting. Returning the input
# unchanged is still the fallback rather than raising, because a demo that dies on an unlucky draw
# is worse than one that runs the layout it already had.
MAX_ATTEMPTS = 2000

# How far a disturbed object ends up moving, in metres. Chosen to invalidate a planned pose
# without putting anything near the edge of the table: 2 cm is well inside the 8 mm
# same-colour gap times the 18 mm layout spread, and the smallest gap measured over 200
# randomised layouts was 13.7 mm, so a 20 mm move can - and should - be re-planned.
DEFAULT_DISTURB_M = 0.02

# A shove is aimed roughly INWARD, toward the middle of the working area, with this much
# random either side of that. Aimed any old direction it throws things off the table: a ball
# shoved away from the robot at 24 cm/s rolled 12.7 cm to y = -0.568, past the table's near
# edge at -0.6768 was fine but the same run produced x = -0.679, which is off the far side
# entirely, and an object that leaves the table cannot be re-planned for. Aimed inward, the
# longest roll measured over 16 rounds is 134 mm and stays between the trays.
DISTURB_AIM_JITTER_RAD = math.radians(70.0)


def _inward_heading(x: float, y: float, rng: random.Random) -> float:
    """A shove direction pointing at the middle of the working area, loosely.

    Aimed any old direction it throws things off the table: one ball shoved away from the
    robot rolled to x = -0.679, which is off the far side of a table that spans x 0.0032 to
    0.7032, and an object that leaves the table cannot be re-planned for. Aimed inward, the
    longest roll measured over 16 rounds is 134 mm and every object stayed between the trays.
    """
    centre_x = (scene_spec.WORK_X[0] + scene_spec.WORK_X[1]) / 2.0
    centre_y = (scene_spec.WORK_Y[0] + scene_spec.WORK_Y[1]) / 2.0
    inward = math.atan2(centre_y - y, centre_x - x)
    return inward + rng.uniform(-DISTURB_AIM_JITTER_RAD, DISTURB_AIM_JITTER_RAD)
#
# So a sphere is shoved, which is the interesting case because where it ends up is nobody's
# prediction, and anything else is displaced directly by DEFAULT_DISTURB_M, which is what a
# person does when they pick something up and move it. The demo still gets a mid-task
# disturbance for every object; it just does not pretend the two kinds are the same physics.
DISTURB_SPEED_M = (0.08, 0.28)

# How long to let a disturbed object come to rest before re-perceiving.
DISTURB_SETTLE_STEPS = 45

# A shove slower than this is not a disturbance, it is a rounding error.
DISTURB_MIN_MOVE_M = 0.005


@dataclass
class Disturbance:
    """One mid-task shove, and what it did."""

    prim: str
    from_xy: tuple[float, float]
    to_xy: tuple[float, float]
    speed_m: float
    steps: int
    # How it was disturbed: "shoved" for a sphere that rolled, "moved" for anything a shove
    # cannot budge. The report says which, so a reader is never left guessing why a cube
    # ended up exactly where it was asked to go.
    method: str = "shoved"
    # What the simulator did with the command. A shoved sphere reads back a speed that decays
    # to zero; these make the difference between "asked to move" and "moved" visible.
    applied_m: float = 0.0
    peak_m: float = 0.0

    @property
    def moved_m(self) -> float:
        return math.dist(self.from_xy, self.to_xy)

    def summary(self) -> str:
        detail = (f"shoved at {self.speed_m * 100:.0f} cm/s, peaked at "
                  f"{self.peak_m * 100:.0f} cm/s") if self.method == "shoved" else "moved"
        return (f"{self.prim} {detail}, went {self.moved_m * 100:.1f} cm to "
                f"({self.to_xy[0]:+.4f}, {self.to_xy[1]:+.4f})")


def _gap(first: Item, second: Item) -> float:
    """Clearance between two items' footprints, in metres. Negative means overlap.

    Axis-aligned boxes rather than discs, because a 7 cm square tray treated as a 7 cm circle
    would be reported as overlapping things it is actually clear of.
    """
    dx = abs(first.x - second.x) - scene_spec.half_width(first) - scene_spec.half_width(second)
    dy = abs(first.y - second.y) - scene_spec.half_depth(first) - scene_spec.half_depth(second)
    return max(dx, dy)


def conflicts(items: list[Item]) -> list[str]:
    """Every reason this layout would not work, in plain words.

    Two kinds of check, and they measure different things:

    * **Overlap** is a gap between footprints, checked for every pair. Interpenetration at spawn
      is a physics problem, not a perception one, and it is invisible in a picture.
    * **Separation** is a distance between CENTRES, and only for same-coloured pairs whose blobs
      would merge. A tray and the object that shares its colour do not have to be far apart in
      centre terms to be told apart in the image if they are at different depths - which is why
      the layout in ``scene_spec`` puts them in different rows - so this check is what forces a
      layout to do that rather than leaving it to luck.
    """
    problems: list[str] = []
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            gap = _gap(first, second)
            if gap < MIN_GAP_M:
                same = "same colour" if first.color == second.color else "different colours"
                problems.append(f"{first.prim} and {second.prim} ({same}) overlap by "
                                f"{(MIN_GAP_M - gap) * 1000:.1f} mm")
            if first.color != second.color or (first.container and second.container):
                continue
            # Two same-coloured things that are both receptacles are told apart by nothing but
            # their position, so they need the same separation as anything else.
            centre = math.hypot(first.x - second.x, first.y - second.y)
            if centre < MIN_SAME_COLOUR_GAP_M:
                problems.append(
                    f"{first.prim} and {second.prim} (same colour) are "
                    f"{centre * 1000:.0f} mm apart, and their colour blobs merge below "
                    f"{MIN_SAME_COLOUR_GAP_M * 1000:.0f} mm")
    for item in items:
        reach = (math.hypot(item.x - scene_spec.ROBOT_XY[0], item.y - scene_spec.ROBOT_XY[1])
                 + scene_spec.half_extent(item))
        if reach > scene_spec.MAX_REACH_M:
            problems.append(f"{item.prim} is {(reach - scene_spec.MAX_REACH_M) * 1000:.0f} mm "
                            f"out of reach")
        if (item.x - scene_spec.half_width(item) < scene_spec.WORK_X[0]
                or item.x + scene_spec.half_width(item) > scene_spec.WORK_X[1]
                or item.y - scene_spec.half_depth(item) < scene_spec.WORK_Y[0]
                or item.y + scene_spec.half_depth(item) > scene_spec.WORK_Y[1]):
            problems.append(f"{item.prim} is outside the working area")
    return problems


def _sample(items: list[Item], rng: random.Random, spread: float) -> list[Item]:
    """One candidate layout: every item nudged by up to ``spread`` and clipped to the area."""
    out: list[Item] = []
    for item in items:
        half_w, half_d = scene_spec.half_width(item), scene_spec.half_depth(item)
        x = min(max(item.x + rng.uniform(-spread, spread), scene_spec.WORK_X[0] + half_w),
                scene_spec.WORK_X[1] - half_w)
        y = min(max(item.y + rng.uniform(-spread, spread), scene_spec.WORK_Y[0] + half_d),
                scene_spec.WORK_Y[1] - half_d)
        out.append(replace(item, x=x, y=y))
    return out


def randomize(items: list[Item], seed: int, spread: float = DEFAULT_SPREAD_M,
              attempts: int = MAX_ATTEMPTS) -> list[Item]:
    """A valid layout for round ``seed``.

    Rejection sampling: keep drawing until nothing overlaps, nothing same-coloured is close,
    and everything is reachable and inside the working area. Deterministic in ``seed``, so the
    same round number is the same layout on every machine and every run - a failure report is
    only worth reading if someone else can reproduce it.

    Returns the input unchanged if no valid layout is found in ``attempts`` tries, rather than
    raising: a demo that dies because the 401st random draw was unlucky is worse than one that
    runs the layout it already had.
    """
    rng = random.Random(seed)
    for _ in range(attempts):
        candidate = _sample(items, rng, spread)
        if not conflicts(candidate):
            return candidate
    return list(items)


def displacement(items: list[Item], other: list[Item]) -> dict[str, tuple[float, float]]:
    """How far each item moved between two layouts, for the report."""
    by_prim = {item.prim: (item.x, item.y) for item in items}
    return {item.prim: (item.x - by_prim[item.prim][0], item.y - by_prim[item.prim][1])
            for item in other if item.prim in by_prim}


def disturb(env, items: list[Item], rng: random.Random, exclude: set[str] | None = None,
            distance: float = DEFAULT_DISTURB_M, settle: int = DISTURB_SETTLE_STEPS,
            speed: float | None = None) -> Disturbance | None:
    """Disturb one object that is not being held, and let it come to rest.

    How it is disturbed depends on what the object is, because the two kinds do not respond
    the same way and pretending otherwise produces a disturbance that only exists for half the
    scene. See ``DISTURB_SPEED_M`` for the measurements.

    Takes ``items`` rather than bare prim names so it can skip the containers itself. The
    hollow trays are kinematic rigid bodies (hollow.py builds them that way so an object stays
    inside), and a kinematic body cannot be moved by a velocity or a shove: writing one to a
    tray moves it 0.0 mm, which is how this function first looked broken on 5 rounds out of 6.

    Returns ``None`` when there is nothing safe to disturb, so a caller can treat a
    disturbance as optional rather than special-casing an empty table.
    """
    import torch

    skip = set(exclude or ())
    candidates = [item for item in items if not item.container and item.prim not in skip]
    if not candidates:
        return None

    item = candidates[rng.randrange(len(candidates))]
    prim = item.prim
    speed_m = rng.uniform(*DISTURB_SPEED_M) if speed is None else speed

    obj = env.scene[prim]
    start = obj.data.root_pos_w[0].detach().cpu().numpy()
    from_xy = (float(start[0]), float(start[1]))
    heading = _inward_heading(from_xy[0], from_xy[1], rng)
    direction = (math.cos(heading), math.sin(heading), 0.0)
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    zeros = torch.zeros(1, 6, device=env.device)

    peak = 0.0
    applied = 0.0
    taken = 0

    if item.kind == "ball":
        method = "shoved"
        obj.write_root_velocity_to_sim(
            torch.cat([torch.tensor([[direction[0] * speed_m, direction[1] * speed_m, 0.0]],
                                    device=env.device), zeros[:, 3:]], dim=-1))
        for taken in range(1, settle + 1):
            env.sim.step(render=False)
            applied = float(torch.linalg.norm(obj.data.root_lin_vel_w[0]))
            peak = max(peak, applied)
            if applied < 0.01:
                break
        env.scene.update(dt=env.physics_dt)
    else:
        method = "moved"
        shifted = start.copy()
        shifted[0] += direction[0] * distance
        shifted[1] += direction[1] * distance
        obj.write_root_pose_to_sim(torch.cat(
            [torch.as_tensor(shifted, dtype=torch.float32, device=env.device)[None, :],
             upright], dim=-1))
        obj.write_root_velocity_to_sim(zeros)
        for taken in range(1, settle + 1):
            env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

    end = obj.data.root_pos_w[0].detach().cpu().numpy()
    return Disturbance(prim=prim, from_xy=from_xy,
                       to_xy=(float(end[0]), float(end[1])),
                       speed_m=speed_m, steps=taken, applied_m=applied, peak_m=peak,
                       method=method)