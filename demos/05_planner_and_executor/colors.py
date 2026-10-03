"""Named colour ranges for HSV segmentation, and the helpers that measure them.

This is the file to edit at the hackathon, in the same way ``vocabulary.yaml`` is
the file to edit for the command model: it holds the closed set of things
perception can name, and nothing else in the demo hardcodes a colour.

Why HSV and not RGB: the dome light in the task config is
``DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0)``, so the same albedo
renders at a different brightness depending on how it faces the light and how far
it is from the camera. Hue is stable under a change of intensity, and saturation
separates a painted object from the light wood of the table, which is why both the
hue band and ``sat_min`` are thresholds and value is only used as a floor.

OpenCV's hue is ``0..179`` for ``0..360`` degrees, so a hue of 360 is 0.

NOT MEASURED YET. Every range below is the nominal band for the albedo in
``scene_spec.py``, not a band read off a rendered frame. Run

    python .\\demos\\04_perception\\09_camera_to_scene.py --calibrate

once, read ``results/calibration_report.md``, and paste the measured bands back
into ``COLORS``. The comments above each colour record what it is waiting for.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field

import numpy as np

# Blobs smaller than this are dropped. At the top camera's 1206 px/m a 3 cm cube is
# 36 px wide, i.e. about 1300 px of area; a 2 cm ball is about 1000 px. 150 px keeps
# the two clear of sensor noise while still rejecting specular speckle. Tune with
# --calibrate, which reports the area of every blob it found.
MIN_AREA_PX = 150

# Radius of the morphological opening that separates two touching blobs. Two objects
# of the same colour are always placed at least 6 cm apart in the scene spec, which
# is 72 px at 1206 px/m, so a 3 px opening cannot break a real pair apart but does
# remove the thin bridges that shading makes along a contact edge.
OPENING_PX = 3


@dataclass(frozen=True)
class Color:
    """One named colour: the albedo it is spawned with, and the band it is found in.

    ``hue`` is a list of bands because red wraps around 0 in OpenCV's hue circle.
    An empty ``hue`` list means "any hue", which is how white and black are handled:
    they have no hue of their own and are found by saturation and value instead.
    """

    name: str
    albedo: tuple[float, float, float]
    hue: list[tuple[int, int]] = field(default_factory=list)
    sat_min: int = 0
    sat_max: int = 255
    val_min: int = 0
    val_max: int = 255
    note: str = ""
    measured: bool = True
    usable: bool = True

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "albedo": list(self.albedo),
            "hue": [list(band) for band in self.hue],
            "sat": [self.sat_min, self.sat_max],
            "val": [self.val_min, self.val_max],
            "measured": self.measured,
            "usable": self.usable,
            "note": self.note,
        }


def hsv_of(albedo: tuple[float, float, float]) -> tuple[int, int, int]:
    """The HSV triple a flat albedo of this colour has before any lighting.

    Used to print the value to look for in a calibration report, not to classify:
    the render is never exactly this bright.
    """
    r, g, b = (max(0.0, min(1.0, c)) for c in albedo)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(round(h * 180)), int(round(s * 255)), int(round(v * 255))


# The closed set. Kept in one place so that colours, the scene spec and the
# command model's vocabulary.yaml stay in step: every name here also has to exist
# in vocabulary.yaml `colors`, or a detected object cannot be named in a command.
#
# MEASURED, not guessed, by `--calibrate` against a rendered frame of the top camera
# (Isaac Sim 5.1, RTX 5060 Ti, 1000-intensity dome light, 640x480). Two findings from
# those measurements changed this file:
#
# * The rendered colours are much less saturated than their albedos suggest. The blue
#   cube measured S = 75 and the green ball S = 68 at their centres, where the nominal
#   albedos would predict S = 235 and S = 222. A dome light of intensity 1000 blows the
#   highlights out and washes the chroma away, so sat_min has to be far lower than
#   intuition suggests. It is 60, not 110.
# * The wood table measures H = 18, S = 82 - inside the yellow hue band. Yellow is
#   therefore only separable by saturation, and it needs S >= 130 while the wood peaks
#   at 95. Yellow also is not used by the scene spec yet, so it is off by default; see
#   DEFAULT_DETECT below.
COLORS: dict[str, Color] = {
    "red": Color(
        name="red",
        albedo=(0.78, 0.06, 0.05),
        hue=[(0, 11)],
        sat_min=120,
        val_min=110,
        note="MEASURED: S 170-180 and H 0-2 at the red cube's centre, S 56-211 across "
             "the cube and the red ball. The low-saturation tail is the cube's shaded "
             "left face, which is why the band starts at 120 rather than 160. Red is "
             "also the one colour with a second population at H 170-179 (50 px), from "
             "the specular edge; the band is left open there rather than chasing it.",
    ),
    "green": Color(
        name="green",
        albedo=(0.08, 0.62, 0.16),
        hue=[(39, 90)],
        sat_min=100,
        val_min=140,
        note="MEASURED: H 17-19 and S 58-80 at the green ball's centre - which is NOT "
             "green, it is the wood table showing around the ball. The ball itself is "
             "the 625 px counted at H 38-84 with S 110-176 and V 165-215. The band is "
             "therefore set from the histogram, not from the centre sample, and "
             "sat_min is 100 because the lit edge of the ball desaturates.",
    ),
    "blue": Color(
        name="blue",
        albedo=(0.06, 0.16, 0.78),
        hue=[(104, 135)],
        sat_min=60,
        val_min=120,
        note="MEASURED: 4493 px at H 104-132, S 110-209, V 156-221. sat_min is 60, the "
             "lowest in the set, because the blue cube's centre measured S 75 and a "
             "higher floor misses the object entirely. This is the number to raise "
             "first if blue starts matching the wood.",
    ),
    "yellow": Color(
        name="yellow",
        albedo=(0.82, 0.72, 0.06),
        hue=[(22, 38)],
        sat_min=130,
        val_min=130,
        note="MEASURED AND DANGEROUS: the wood table is H 18, S 82, V 206, and the "
             "yellow hue band has to start above 18 to avoid it, which leaves H 22-38. "
             "The table's saturation peaks at 95, so sat_min 130 is what separates them, "
             "and it is thin. Nothing in the scene spec is yellow yet; if a yellow "
             "object is added, measure it before believing this band.",
    ),
    "white": Color(
        name="white",
        albedo=(0.90, 0.90, 0.90),
        hue=[],
        sat_min=0,
        sat_max=40,
        val_min=200,
        note="Found by low saturation and high value. The background beyond the table "
             "measures this, so white blobs in the report are usually the void, not an "
             "object.",
    ),
    "black": Color(
        name="black",
        albedo=(0.06, 0.06, 0.06),
        hue=[],
        sat_min=0,
        sat_max=60,
        val_max=90,
        note="Found by low saturation and low value. In this scene it matches the "
             "robot's own dark grey links.",
    ),

    # --- the three colours vocabulary.yaml can name and demo 09 could not see --------
    #
    # All nine bands below now come from a rendered frame, measured by
    # tools/calibrate_colours.py: a 5 cm probe cube per colour on one grid, one rendered
    # frame, and the pixels each probe's known footprint covers read in HSV. The table
    # and its reasoning are in results/colour_calibration.md.
    #
    # Two of the three are measured and NOT USABLE, and that is the result rather than a
    # failure of the measurement:
    #
    #   orange  H peak 24, S 170-182, V 217-222
    #   yellow  H peak 29, S 167-180, V 210-220
    #
    # A hue gap of 5 with overlapping saturation and overlapping value. Orange is
    # physically between red and yellow, and this scene already has red at H 1 and yellow
    # at H 29, so there is no room left for it. Widening one of the two neighbours would
    # cost a band that currently works.
    #
    #   pink    H peak 170, S 19-202, V 156-223
    #   red     H peak   1, S 168-186, V 210-220
    #
    # A hue gap of 11 across the wrap, and pink's saturation spans almost the whole
    # range, so saturation cannot separate them either.
    #
    # Both are kept in COLORS so that a detected object can still be *named* by a command
    # model that says the word, and both are left out of the detect set. Perception is
    # better off saying it cannot see an orange than finding the wood table's hue.
    "orange": Color(
        name="orange",
        albedo=(0.85, 0.42, 0.05),
        hue=[(20, 28)],
        sat_min=140,
        val_min=190,
        measured=True,
        usable=False,
        note="MEASURED AND NOT USABLE. H peak 24, S 170-182, V 217-222. Yellow measures "
             "H peak 29, S 167-180, V 210-220: a hue gap of 5 with both saturation and "
             "value overlapping. Orange sits between this scene's red (H 1) and yellow "
             "(H 29) and there is no room for it. Band kept for reference only; excluded "
             "from the detect set.",
    ),
    "purple": Color(
        name="purple",
        albedo=(0.72, 0.18, 0.62),
        hue=[(144, 163)],
        sat_min=85,
        val_min=160,
        measured=True,
        usable=True,
        note="MEASURED. H peak 153, S 94-132, V 172-214. 95% of its pixels agree on the peak "
             "hue. Separated by hue from every colour this can actually detect: red, "
             "green, blue and yellow are all >= 20 hue units away. Its one conflict is "
             "pink, at a hue gap of 17 where pink's saturation range (19-202) swallows "
             "purple's (94-132) so saturation does not separate them either - but pink "
             "is itself measured as unusable and is not detectable, so the conflict "
             "never arises in practice. Usable, though still not in the proven detect "
             "set until it has been through the same 20-trial check as red/green/blue.",
    ),
    "pink": Color(
        name="pink",
        albedo=(0.95, 0.62, 0.70),
        hue=[(163, 177)],
        sat_min=15,
        val_min=150,
        measured=True,
        usable=False,
        note="MEASURED AND NOT USABLE. H peak 170, S 19-202, V 156-223. Red measures H 1, "
             "S 168-186: a hue gap of 11 across the wrap, and pink's saturation spans "
             "almost the entire range so it cannot be separated by saturation either. "
             "Excluded from the detect set; kept only so the word is nameable.",
    ),
}

# What the perception half actually looks for, in order.
#
# UNCHANGED from demo 09 on purpose. red, green and blue are the only three measured over
# 20 trials of a full scene (117/120 kinds correct), and adding a fourth colour to this
# tuple changes what every mask in the frame contains, so it has to earn its place the
# same way those three did.
#
# What the nine-colour measurement (tools/calibrate_colours.py, results/
# colour_calibration.md) established, for whoever wants to widen this:
#
#   usable   red H1, green H69, blue H110, yellow H29, purple H153,
#            white (S0-1, V216-223), black (V67-69)
#   unusable orange  H24 - a hue gap of 5 from yellow, saturation and value overlapping
#            pink    H170 - a hue gap of 11 from red across the wrap, saturation 19-202
#
# White and black are separable, but not by hue: white has no saturation (0-1) and black
# has no value above 69, and both are far from every other colour's range. They are off
# by default for a different reason - the background beyond the table measures as white
# and the robot's own links measure as black, so including them turns the whole frame
# into detections. Yellow is off because it needs S >= 130 to clear the wood table, which
# its measured S 167-180 supports; purple is off because it has never been through a
# 20-trial check.
#
# So the next colour to try adding is purple, and `WIDER_DETECT` below is the list to
# measure against demo 09's baseline before believing it.
DEFAULT_DETECT = ("red", "green", "blue")

# Every colour the measurement says is separable from every other. NOT the detect set:
# this is the candidate list, and widening DEFAULT_DETECT to it has to be re-measured
# over the same 20 trials first.
WIDER_DETECT = ("red", "green", "blue", "yellow", "purple")

# Containers are the only large blobs, so they are found with the same colour masks and
# then told apart by area. Not a second colour set: a blue tray and a blue cube differ in
# size, not in hue.
#
# Measured on rendered frames at the top camera's 746 px/m, which is the number the
# demo actually reports:
#
#     3 cm cube   644-702 px      4 cm ball   633-702 px
#     6 x 7 tray  ~2000 px        6 x 6 box   ~1900 px
#
# 4000 was sized for the old 9 x 7 cm and 8.5 x 8.5 cm zones, which covered 5000 and
# 5700 px. Narrowing the zones to 6 cm to fit inside the arm's reach took them below the
# old threshold, so both zones were reported as ordinary objects and read as cubes.
#
# 1200 sits between the largest small object measured (702 px) and the smallest zone
# measured (~1900 px). It is a wide gap in ratio terms, but only 1.7x, so a layout change
# that shrinks the zones again has to re-measure this rather than assume it holds.
CONTAINER_MIN_AREA_PX = 1200


def colors(selected: list[str] | None = None) -> list[str]:
    """The colour names to look for, in a fixed order so reports are reproducible.

    ``None`` means every colour in ``COLORS``; a list restricts to those, in the order
    ``COLORS`` defines rather than the caller's, so the output order never depends on a
    command line.
    """
    if selected is None:
        return list(COLORS)
    wanted = {name.strip() for name in selected if name.strip()}
    unknown = wanted - set(COLORS)
    if unknown:
        raise ValueError(f"unknown colour(s) {sorted(unknown)}; expected any of {list(COLORS)}")
    return [name for name in COLORS if name in wanted]


def mask_of(hsv: np.ndarray, color: Color) -> np.ndarray:
    """A boolean mask of the pixels that belong to this colour.

    Hue bands are OR-ed together, because red occupies two of them, and an empty
    band list matches every hue so that white and black can be gated on saturation
    and value alone.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = (s >= color.sat_min) & (s <= color.sat_max)
    inside &= (v >= color.val_min) & (v <= color.val_max)
    if color.hue:
        in_hue = np.zeros(h.shape, dtype=bool)
        for low, high in color.hue:
            in_hue |= (h >= low) & (h <= high)
        inside &= in_hue
    return inside


def parse_detect(text: str) -> list[str]:
    """The colours named on the command line, in the order COLORS defines.

    Warns about any colour that was measured as unusable, rather than refusing it. Asking
    for one is a reasonable thing to do while investigating, and the point of the ``usable``
    flag is that someone finds out the band does not work instead of quietly discovering it
    through wrong detections.
    """
    import warnings

    wanted = colors([name for name in re.split(r"[,\s]+", text or "") if name])
    bad = [name for name in wanted if not COLORS[name].usable]
    if bad:
        warnings.warn(
            f"{', '.join(bad)} {'was' if len(bad) == 1 else 'were'} measured as not separable "
            f"from another colour and will match it. Expect wrong detections; see "
            f"results/colour_calibration.md.",
            stacklevel=2,
        )
    return wanted


def detection_note() -> str:
    """One line explaining why some colours are off by default, for the report.

    Three groups, because "off by default" means two different things. A colour can be
    separable but not yet proven over a full scene, or it can be measured as impossible to
    separate from another colour at all, and conflating them would invite someone to add
    an unusable band to the detect set because it was merely untested. ``usable`` is what
    separates the second group from the first.
    """
    off = [c for c in COLORS if c not in DEFAULT_DETECT]
    unusable = [c for c in off if not COLORS[c].usable]
    untested = [c for c in off if COLORS[c].usable]
    parts = [f"looking for: {', '.join(DEFAULT_DETECT)}."]
    if untested:
        parts.append(f"separable but off by default: {', '.join(untested)} "
                     f"(white and black match the background and the robot's own links; "
                     f"yellow shares its hue band with the wood table; purple has not been "
                     f"through the 20-trial check yet).")
    if unusable:
        parts.append(f"MEASURED AS NOT SEPARABLE, leave off: {', '.join(unusable)} "
                     f"(orange is 5 hue units from yellow, pink is 11 from red, and "
                     f"saturation separates neither). They stay in COLORS so a command model "
                     f"can still name them.")
    added = [c for c in WIDER_DETECT if c not in DEFAULT_DETECT]
    if added:
        parts.append(f"measured candidates to try next: {', '.join(added)}.")
    parts.append("Add one with --detect.")
    return " ".join(parts)


def overlapping(colors: list[str]) -> dict[str, list[str]]:
    """Which colour bands cover the same pixels, for the calibration report.

    A clash means two colours claim the same pixels, so one detection is reported under
    both names. This is printed rather than left to be discovered as a wrong detection.

    Hue bands are compared as plain intervals, not as an arc: OpenCV's hue is a number
    in 0..179 with no wrap, and red's two bands (0-8 and 172-179) are two intervals
    rather than one arc, so an ordinary interval overlap is exactly the right test.
    """
    clashes: dict[str, list[str]] = {}
    names = list(colors)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            a, b = COLORS[first], COLORS[second]
            if max(a.sat_min, b.sat_min) > min(a.sat_max, b.sat_max):
                continue
            if a.val_max < b.val_min or b.val_max < a.val_min:
                continue
            if a.hue and b.hue:
                shared = any(
                    min(h1_hi, h2_hi) >= max(h1_lo, h2_lo)
                    for h1_lo, h1_hi in a.hue
                    for h2_lo, h2_hi in b.hue
                )
                if not shared:
                    continue
            clashes.setdefault(first, []).append(second)
    return clashes


def describe() -> str:
    """A table of the bands with the HSV each albedo would have, for the report."""
    rows = [
        "colour   albedo               nominal HSV   hue band        sat      val",
        "-------  -------------------  ------------  --------------  -------  -------",
    ]
    for name, color in COLORS.items():
        h, s, v = hsv_of(color.albedo)
        bands = ",".join(f"{lo}-{hi}" for lo, hi in color.hue) or "(any)"
        albedo = " ".join(f"{c:.2f}" for c in color.albedo)
        rows.append(
            f"{name:<8} {albedo:<20} {f'{h:>3},{s:>3},{v:>3}':<13} {bands:<15} "
            f"{f'{color.sat_min}-{color.sat_max}':<8} {f'{color.val_min}-{color.val_max}'}"
        )
    return "\n".join(rows)
