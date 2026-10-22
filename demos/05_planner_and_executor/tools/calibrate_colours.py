"""Measure the HSV of each colour as the camera actually renders it, and print the bands.

Why this exists
---------------
``colors.py`` holds nine named bands and the file's own header admits most of them were
never measured against a rendered frame. The measurements that *were* taken are
uncomfortable: the blue cube's albedo predicts saturation 235 and rendered at 75, and the
green ball's centre sample came back as H 17-19, which is the wood table showing around
it rather than the ball at all. Nominal HSV from an albedo is therefore close to useless
except for hue, and even hue is only an assumption until it is measured.

The measurement
---------------
For one colour at a time: place a single object of it on the table at a known pose, render
the top camera, then read the HSV of the pixels the object occupies. The object's true
world pose is known because this demo places it, so the pixels belonging to it are known
exactly - ``camera.project`` on the object's top-face corners - and there is no need for a
mask, a band, or a detector to already work. That is what makes this usable as the first
step: it measures the bands that the rest of the perception depends on.

It reports, per colour, the measured hue/saturation/value percentiles inside that
footprint, how the proposed band compares, and the nearest other colour's distribution so
overlaps are visible instead of discovered later.

    python tools/calibrate_colours.py
    python tools/calibrate_colours.py --colours orange,purple,pink --headless
    python tools/calibrate_colours.py --reject-colour pink

Running it writes ``results/colour_calibration.md`` and prints the same table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import isaac_io  # noqa: E402
import perceive  # noqa: E402
import scene_spec  # noqa: E402

RESULTS = ROOT / "results"

# One object per colour is enough to measure a band, and measuring them one at a time is
# the point: if all nine were on the table at once, every band would also be measuring the
# other eight, and a colour that overlaps a neighbour could not be told from it.
REFERENCE_AREA = (0.32, -0.42)

# Edge of the reference object, metres. A 3 cm cube is only ~22 px across at the top
# camera's 746 px/m, so a 356 px sample is mostly boundary and the wood table shows
# through it: a first run at 3 cm measured the green ball's 5th percentile as H 1, which
# is the table, and the black cube's 95th as V 221, which is also the table. 5 cm gives
# about 1400 px, of which the interior is most of the frame.
PROBE_EDGE = 0.05

# Fraction trimmed off each side of the probe's footprint before sampling.
PROBE_INSET = 0.22

# The probes are laid out on one 3x3 grid and measured from a SINGLE rendered frame,
# rather than one at a time. Measuring them one at a time needs nine separate spawns, and
# the first version of this tool got it wrong in a way worth recording: it spawned all
# nine at one position and then moved them there in turn, so they were stacked on top of
# each other and every colour reported the same H 171, S 18-30, V 205-225 - which is the
# wood table, not any of the nine colours.
#
# 5.5 cm spacing puts the 3x3 grid inside scene_spec's working area (x 0.24..0.40,
# y -0.53..-0.32) with the cubes 5 mm clear of the boundary.
PROBE_GRID = (3, 3)
PROBE_SPACING = 0.055


def grid_positions(count: int) -> list[tuple[float, float]]:
    """``count`` table positions on a grid centred on the working area, row by row."""
    columns, rows = PROBE_GRID
    if count > columns * rows:
        raise ValueError(f"{count} probes do not fit on a {columns}x{rows} grid")
    out = []
    for index in range(count):
        column = index % columns
        row = index // columns
        out.append((
            REFERENCE_AREA[0] + (column - (columns - 1) / 2.0) * PROBE_SPACING,
            REFERENCE_AREA[1] + (row - (rows - 1) / 2.0) * PROBE_SPACING,
        ))
    return out


def footprint_pixels(camera, item, inset: float = PROBE_INSET):
    """The pixels the reference object's top face covers, from its known world pose.

    Projects the four corners of a square of side ``inset`` smaller than the object, so
    the sample stays off its own antialiased boundary, and clips to the frame.
    """
    half = (item.size[0] / 2.0) * (1.0 - inset)
    corners = [
        (item.x - half, item.y - half, item.z + item.height / 2.0),
        (item.x + half, item.y - half, item.z + item.height / 2.0),
        (item.x + half, item.y + half, item.z + item.height / 2.0),
        (item.x - half, item.y + half, item.z + item.height / 2.0),
    ]
    points = np.array([camera.project(p) for p in corners], dtype=float)
    if not np.isfinite(points).all():
        return None
    return points


def sample_footprint(hsv: np.ndarray, points) -> np.ndarray | None:
    """Every pixel inside the projected quadrilateral, as an (N, 3) uint8 array."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2_fill(mask, points)
    chosen = hsv[mask > 0]
    return chosen if chosen.size else None


def cv2_fill(mask: np.ndarray, points) -> None:
    import cv2

    cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 1)


def dominant_hue(samples: np.ndarray) -> tuple[int, float, int]:
    """The hue a colour actually is: the peak of its hue histogram.

    NOT a mean. Hue wraps, so an arithmetic mean is meaningless across 0 and 179, and a
    saturation-weighted circular mean is worse than useless here: it is dragged between
    two populations instead of reporting either. With a first attempt at this the green
    ball's footprint was bimodal - the wood table at H 1 and the ball at H 69 - and the
    mean landed in the empty gap between them at H 57, agreeing with 3% of its own
    samples. A histogram peak cannot do that: it reports the biggest pile, which is the
    object, because the table is a minority of a probe's own pixels.

    Returns the peak hue, the share of samples within 10 hue units of it, and the number
    of samples in the peak bin.
    """
    hues = samples[:, 0]
    counts = np.bincount(hues.astype(np.int64), minlength=180)
    peak = int(np.argmax(counts))
    total = float(len(hues))
    share = float(np.mean(np.abs((hues.astype(np.float64) - peak + 90) % 180 - 90) <= 10))
    return peak, share, int(counts[peak])


def band_at(samples: np.ndarray, hue: int) -> tuple[int, int, int, int]:
    """Saturation and value percentiles of the pixels that are actually this colour.

    Taken only from samples within 10 hue units of the peak, so the wood table and the
    antialiased edge do not set the saturation floor.
    """
    hues = samples[:, 0].astype(np.float64)
    chosen = samples[np.abs((hues - hue + 90) % 180 - 90) <= 10]
    if len(chosen) < 10:
        return 0, 0, 0, len(chosen)
    sat = chosen[:, 1].astype(np.float64)
    val = chosen[:, 2].astype(np.float64)
    return (
        int(round(np.percentile(sat, 5))),
        int(round(np.percentile(sat, 95))),
        int(round(np.percentile(val, 5))),
        int(round(np.percentile(val, 95))),
    )


def hue_distance(a: int, b: int) -> int:
    """Shortest distance between two hues on the wrapping hue circle."""
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--colours", default=",".join(colors.COLORS),
                        help="Comma-separated colours to measure (default: all nine)")
    parser.add_argument("--camera", default="top", choices=sorted(scene_spec.CAMERA_CHOICES))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reject-colour", default="",
                        help="Print the separation of every colour from this one, and exit")
    args = parser.parse_args()

    if args.reject_colour:
        return separation_only(args.reject_colour)

    wanted = [c for c in colors.parse_detect(args.colours)]
    if not wanted:
        print(f"ERROR: no colour asked for; expected any of {list(colors.COLORS)}")
        return 1

    print(f"measuring {len(wanted)} colour(s) as the {args.camera} camera renders them")
    print(f"{PROBE_EDGE * 100:.0f} cm probes on a {PROBE_GRID[0]}x{PROBE_GRID[1]} grid, "
          f"{PROBE_SPACING * 100:.1f} cm apart, one rendered frame")
    print("")

    isaac_io.prepare_runtime(headless=args.headless)

    positions = grid_positions(len(wanted))
    items = [
        scene_spec._cube(f"probe_{name}", name, x, y, edge=PROBE_EDGE)
        for name, (x, y) in zip(wanted, positions)
    ]

    env, sensor = isaac_io.build_env(args.camera, items)
    try:
        # reset() first: sensor.data.pos_w is still [0, 0, 0] until the scene is
        # initialised, so aim_camera would read the prim default and complain. Demo 09
        # has the same order at its own line 846.
        env.reset()
        isaac_io.aim_camera(env, args.camera, sensor)
        isaac_io.park_arm(env)
        isaac_io.place_items(env, items)

        image, _depth, camera = isaac_io.capture(env, sensor)
        hsv = perceive.to_hsv(image)
        perceive.save_png(ROOT / "out" / "colour_probes.png", image)
        lines = [
            f"# colour calibration, {args.camera} camera",
            "",
            f"One {PROBE_EDGE * 100:.0f} cm object per colour, all in one rendered frame, and the",
            "pixels each object's known footprint covers were read in HSV. No mask, band or",
            "detector is involved, so this measures the bands the rest of perception depends",
            "on rather than assuming them.",
            "",
            "| colour | px | H peak | share | S 5/95 at peak | V 5/95 at peak | current band | nominal H |",
            "|---|---|---|---|---|---|---|---|",
        ]
        rows = []
        for item in items:
            points = footprint_pixels(camera, item)
            samples = sample_footprint(hsv, points) if points is not None else None
            if samples is None:
                print(f"  {item.color:7s}: NOT VISIBLE (projected footprint off-frame)")
                lines.append(f"| {item.color} | 0 | - | - | - | - | off-frame | - |")
                continue

            hue, share, in_peak = dominant_hue(samples)
            sat_lo, sat_hi, val_lo, val_hi = band_at(samples, hue)
            band = ", ".join(f"{a}-{b}" for a, b in colors.COLORS[item.color].hue) or "n/a"
            nominal = colors.hsv_of(colors.COLORS[item.color].albedo)
            rows.append((item.color, hue, sat_lo, sat_hi))

            print(f"  {item.color:7s} {len(samples):5d} px   H peak {hue:3d} ({share * 100:3.0f}% agree, "
                  f"{in_peak} in bin)   S {sat_lo:3d}-{sat_hi:3d}   V {val_lo:3d}-{val_hi:3d}")
            lines.append(
                f"| {item.color} | {len(samples)} | {hue} | {share * 100:.0f}% | "
                f"{sat_lo}/{sat_hi} | {val_lo}/{val_hi} | {band} | {nominal[0]} |"
            )

        lines += ["", "## how far apart the colours are", "",
                  "| pair | hue gap | S floors | verdict |", "|---|---|---|---|"]
        for i, (name_a, hue_a, sat_a_lo, sat_a_hi) in enumerate(rows):
            for name_b, hue_b, sat_b_lo, sat_b_hi in rows[i + 1:]:
                distance = hue_distance(hue_a, hue_b)
                achromatic = name_a in ("white", "black") or name_b in ("white", "black")
                # Separation by saturation needs a *gap* between the two ranges, not
                # merely an overlap. Comparing interval intersection, as a first version
                # did, flags red (S 168-186) against blue (S 173-176) as conflicting,
                # which is noise: a 71 hue gap already separates those.
                sat_gap = max(sat_a_lo, sat_b_lo) - min(sat_a_hi, sat_b_hi)
                if achromatic:
                    verdict = ("no hue of its own; separable by S/V "
                               f"(S {sat_a_lo}-{sat_a_hi}, V below)")
                elif distance >= 20:
                    verdict = "separated by hue"
                elif distance >= 10 and sat_gap > 25:
                    verdict = f"hue gap {distance}, separated by saturation"
                elif distance >= 10:
                    verdict = f"hue gap {distance}, saturation overlaps: too close"
                else:
                    verdict = f"hue gap {distance}: TOO CLOSE, needs a different albedo"
                lines.append(f"| {name_a} / {name_b} | {distance} | "
                             f"{sat_a_lo}-{sat_a_hi} / {sat_b_lo}-{sat_b_hi} | {verdict} |")
                print(f"  {name_a:7s} vs {name_b:7s}  {verdict}")

        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "colour_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwritten: {RESULTS / 'colour_calibration.md'}")
        return 0
    finally:
        env.close()


def separation_only(name: str) -> int:
    """Print the stored bands that already conflict with ``name``, without starting Kit."""
    clash = colors.overlapping(colors.colors(list(colors.COLORS)))
    print(f"stored hue-band overlap: {clash or 'none'}")
    colour = colors.COLORS.get(name)
    if colour is None:
        print(f"ERROR: no colour {name!r}; expected any of {list(colors.COLORS)}")
        return 1
    print(f"{name}: band {colour.hue or 'none'}, sat {colour.sat_min}-{colour.sat_max}, "
          f"val {colour.val_min}-{colour.val_max}, measured={colour.measured}")
    nominal = colors.hsv_of(colour.albedo)
    print(f"  nominal hue from albedo {nominal[0]}, and that band "
          f"{'contains' if any(a <= nominal[0] <= b for a, b in colour.hue) else 'DOES NOT contain'} it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())