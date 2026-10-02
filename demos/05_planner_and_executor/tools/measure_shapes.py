"""Measure what the blobs in captured frames actually look like, per shape class.

Every shape and container threshold in ``perceive.py`` is a measured number, and a
threshold measured against solid coloured zones stops being true the moment the zones
become hollow trays. This is the tool that keeps those numbers honest: it segments a
folder of frames and prints, per recognised kind, the range of area, elongation and
fill, so a threshold can be placed in a gap that the data agrees on rather than in one
that seems reasonable.

It also reports the fill ratio of every blob the container test would accept, because a
tray is recognised by *not* being a ball or a cube, and the margin between those two
decisions is what has to stay clear.

    python tools/measure_shapes.py
    python tools/measure_shapes.py --frames ..\04_perception\out\frames --detect red,green,blue
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import perceive  # noqa: E402

DEFAULT_FRAMES = ROOT.parent / "04_perception" / "out" / "frames"


def frames_of(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither a file nor a folder")
    found = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not found:
        raise FileNotFoundError(f"{path} has no images in it")
    return found


def span(values: list[float], digits: int = 3) -> str:
    if not values:
        return "no blobs"
    return f"{min(values):.{digits}f}..{max(values):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--detect", default=",".join(colors.DEFAULT_DETECT),
                        help="Comma-separated colour names to segment")
    parser.add_argument("--min-area", type=int, default=colors.CONTAINER_MIN_AREA_PX,
                        help="Area floor handed to perceive.is_container")
    args = parser.parse_args()

    frames = frames_of(args.frames)
    looking_for = colors.colors(colors.parse_detect(args.detect))
    rows: list[tuple[str, int, float, float, str, bool]] = []

    for path in frames:
        hsv = perceive.to_hsv(perceive.read_image(path))
        for name in looking_for:
            for blob in perceive.blobs_of(hsv, colors.COLORS[name], colors.OPENING_PX):
                kind = perceive.kind_of(blob.elongation, blob.fill)
                rows.append((blob.color, blob.area_px, blob.elongation, blob.fill, kind,
                             perceive.is_container(blob, args.min_area)))

    if not rows:
        print(f"ERROR: no blobs in {len(frames)} frame(s)")
        return 1

    print(f"{len(frames)} frame(s), {len(rows)} blob(s), looking for: "
          f"{', '.join(looking_for)}")
    print(f"container area floor: {args.min_area} px")
    print("")
    print("| kind | blobs | area px | elongation | fill |")
    print("|---|---|---|---|---|")
    by_kind = collections.defaultdict(list)
    for row in rows:
        by_kind[row[4]].append(row)
    for kind in sorted(by_kind):
        group = by_kind[kind]
        areas = [r[1] for r in group]
        print(f"| {kind} | {len(group)} | {min(areas)}..{max(areas)} | "
              f"{span([r[2] for r in group])} | {span([r[3] for r in group])} |")

    print("")
    trays = [r for r in rows if r[5]]
    if trays:
        print(f"{len(trays)} blob(s) called a container:")
        for colour, area, elong, fill, _kind, _flag in sorted(trays, key=lambda r: -r[1]):
            print(f"    {colour:6s} area={area:5d}  elongation={elong:.3f}  fill={fill:.3f}")
        print("")
        print("  a container should sit well below both:")
        print(f"    cube ceiling fill {perceive.CUBE_FILL} (measured 0.9562..0.9780)")
        print(f"    ball floor  fill {perceive.BALL_FILL} (measured 0.9985..1.0001)")
    else:
        print("no blob passed the container test at this area floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())