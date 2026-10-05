"""How big a tray's cavity has to be before placement is reliable. No Isaac Sim.

The hollow trays in this directory were first built with a 4.4 cm cavity, and
``tools/check_trays.py`` then measured that no available position estimate was accurate
enough to drop a 3 cm cube into them: the best one was 9-11 mm off, and the margin between a
3 cm cube and a 4.4 cm cavity is 7 mm per side. The object would have touched the wall.

That is a geometry problem, not a perception problem, so it is answered with arithmetic
here rather than by guessing a new cavity size and re-running the simulator.

    python tools/tray_sizes.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scene_spec  # noqa: E402

# The objects this demo moves, and the worst-case position error measured on real trays.
OBJECTS = (("3 cm cube", 0.03), ("3 cm ball", 0.03))
POSITION_ERROR_M = 0.012  # the worst of the three centre estimates, measured

CANDIDATES = [
    (0.044, 0.008),
    (0.050, 0.006),
    (0.055, 0.006),
    (0.060, 0.006),
    (0.060, 0.005),
    (0.065, 0.005),
]


def worst_corner(outer: float) -> float:
    """The farthest point of any tray-sized footprint from the robot base, in the working area."""
    rx, ry = scene_spec.ROBOT_XY
    worst = 0.0
    for tx in scene_spec.WORK_X:
        for ty in scene_spec.WORK_Y:
            for cx, cy in ((tx - outer / 2, ty - outer / 2), (tx + outer / 2, ty + outer / 2)):
                worst = max(worst, math.dist((rx, ry), (cx, cy)))
    return worst


def main() -> int:
    print(f"robot base {scene_spec.ROBOT_XY}, reach {scene_spec.MAX_REACH_M} m")
    print(f"working area x {scene_spec.WORK_X}, y {scene_spec.WORK_Y}")
    print(f"budgeting for the worst measured position error, {POSITION_ERROR_M * 1000:.0f} mm")
    print()
    print("| cavity | wall | footprint | margin, 3 cm cube | margin, 3 cm ball | farthest corner | verdict |")
    print("|---|---|---|---|---|---|---|")
    for inner, wall in CANDIDATES:
        outer = inner + 2 * wall
        margins = []
        for _name, size in OBJECTS:
            slack = inner / 2 - size / 2 - POSITION_ERROR_M
            margins.append(slack)
        reach = worst_corner(outer)
        if reach > scene_spec.MAX_REACH_M:
            verdict = "OUT OF REACH"
        elif min(margins) <= 0:
            verdict = "no margin: the object can touch a wall"
        elif min(margins) < 0.005:
            verdict = "thin"
        else:
            verdict = "OK"
        print(f"| {inner * 100:.0f} cm | {wall * 1000:.0f} mm | {outer * 100:.1f} cm | "
              f"{margins[0] * 1000:+6.1f} mm | {margins[1] * 1000:+6.1f} mm | "
              f"{reach:.3f} m | {verdict} |")

    print()
    print("A tray footprint also has to fit in the working area alongside the other tray and")
    print("the loose objects, which scene_spec.clearance() checks in full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())