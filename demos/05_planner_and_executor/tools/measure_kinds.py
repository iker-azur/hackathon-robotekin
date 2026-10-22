"""Re-measure the cube/ball shape thresholds against the current scene, and set them.

Why this exists
---------------
``perceive.BALL_FILL`` was measured by demo 09 on a scene whose receptacles were flat
coloured slabs. Its own comment says what to do when that stops being true: *"If a future
scene moves the cubes closer to the camera, re-measure this rather than nudging it."*

The scene did move. The receptacles are now hollow trays, 7 cm across and 5 cm tall instead
of 6 cm and 1.2 cm, which changes what the camera sees around them, and the very first
frame of the new scene already breaks the threshold:

    red ball   fill 0.9845   ->  called a cube

That is 0.0005 under ``BALL_FILL`` of 0.985. Demo 09 measured balls at 0.9985-1.0001, so
one frame is not enough to move a constant on: this runs N jittered trials, labels every
blob against the simulator's own ground truth, and reports the two distributions so the
threshold can be placed in the gap that the data agrees on - the same way demo 09 set it.

    python tools/measure_kinds.py --trials 20 --headless
    python tools/measure_kinds.py --trials 20 --headless --out out/kind_trials

It writes results/kind_calibration.md and the frames it measured, so the numbers can be
re-derived rather than taken on trust.
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

# How close a blob's centroid has to be to a truth item's projected position, in PIXELS,
# for that item to be its label. Generous, because the projection of the item's top face is
# not the blob's centroid, and tight enough that one blob is never labelled two ways. A 3 cm
# object is about 22 px across at the top camera's 746 px/m, so 18 px is most of one.
MATCH_RADIUS_PX = 18.0


def label_of(centroid, truth: dict[str, tuple[float, float, str]]) -> str | None:
    """The ground-truth kind a blob is, by nearest projected truth position.

    Positions are held in a dict rather than stashed on the ``Item``: ``Item`` is a
    dataclass and ``scene_spec.jittered`` rebuilds one with ``Item(**item.__dict__)``, so
    an extra attribute set on an item would be passed to the constructor and raise.
    """
    best, best_d = None, MATCH_RADIUS_PX
    for cx, cy, kind in truth.values():
        d = ((centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2) ** 0.5
        if d < best_d:
            best, best_d = kind, d
    return best


def report(measured: dict[str, list[tuple[int, float, float]]]) -> list[str]:
    """The two distributions side by side, and where the gap between them is."""
    lines = [
        "| truth kind | blobs | area px | elongation | fill |",
        "|---|---|---|---|---|",
    ]
    for kind in sorted(measured):
        rows = measured[kind]
        if not rows:
            continue
        areas = [r[0] for r in rows]
        elongs = [r[1] for r in rows]
        fills = [r[2] for r in rows]
        lines.append(f"| {kind} | {len(rows)} | {min(areas)}..{max(areas)} | "
                     f"{min(elongs):.3f}..{max(elongs):.3f} | {min(fills):.4f}..{max(fills):.4f} |")

    cube_fills = [r[2] for r in measured.get("cube", [])]
    ball_fills = [r[2] for r in measured.get("ball", [])]
    lines += ["", "## where the threshold goes", ""]
    if not cube_fills or not ball_fills:
        lines.append("Not enough of both kinds to place a threshold.")
        return lines
    cube_top, ball_bottom = max(cube_fills), min(ball_fills)
    if ball_bottom > cube_top:
        lines.append(f"Cubes top out at {cube_top:.4f} and balls start at {ball_bottom:.4f}, so the "
                     f"two ranges do not touch. Any threshold in "
                     f"({cube_top:.4f}, {ball_bottom:.4f}) separates them; "
                     f"{perceive.BALL_FILL} currently sits at {perceive.BALL_FILL}.")
        mid = (cube_top + ball_bottom) / 2.0
        lines.append(f"The middle of that gap is {mid:.4f}.")
    else:
        overlap = [f for f in cube_fills if f >= min(ball_fills)]
        lines.append(
            f"**The ranges OVERLAP.** Cubes reach {cube_top:.4f} and balls start at "
            f"{ball_bottom:.4f}; {len(overlap)} of {len(cube_fills)} cubes sit at or above the "
            f"lowest ball. There is no fill threshold that separates them, so `kind_of` has to "
            f"use something else - elongation, or a size prior - or the threshold has to be "
            f"chosen against the error that matters most."
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--camera", default="top", choices=sorted(scene_spec.CAMERA_CHOICES))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "kind_trials")
    args = parser.parse_args()

    base = list(scene_spec.DEFAULT_SCENE)
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env(args.camera, base)
    measured: dict[str, list[tuple[int, float, float]]] = {"cube": [], "ball": []}
    wrong: list[str] = []
    seen = 0
    try:
        env.reset()
        isaac_io.aim_camera(env, args.camera, sensor)
        isaac_io.park_arm(env)

        for trial in range(args.trials):
            items = scene_spec.jittered(base, trial) if trial else base
            isaac_io.place_items(env, items)

            image, _depth, camera = isaac_io.capture(env, sensor)
            args.out.mkdir(parents=True, exist_ok=True)
            perceive.save_png(args.out / f"trial{trial:03d}.png", image)

            # Truth positions in pixels, so a blob can be labelled without relying on the
            # very estimator being measured.
            truth: dict[str, tuple[float, float, str]] = {}
            for item in items:
                if item.container:
                    continue
                point = camera.project([item.x, item.y, item.z + item.height / 2.0])
                if point is not None:
                    truth[item.prim] = (point[0], point[1], item.kind)

            hsv = perceive.to_hsv(image)
            for name in colors.colors(colors.DEFAULT_DETECT):
                for blob in perceive.blobs_of(hsv, colors.COLORS[name], colors.OPENING_PX):
                    kind = label_of(blob.centroid_px, truth)
                    if kind in measured:
                        measured[kind].append((blob.area_px, blob.elongation, blob.fill))
                        called = perceive.kind_of(blob.elongation, blob.fill)
                        seen += 1
                        if called != kind:
                            wrong.append(f"trial {trial} {name} blob at "
                                         f"({blob.centroid_px[0]:.0f}, {blob.centroid_px[1]:.0f}) px: "
                                         f"truth {kind}, called {called} "
                                         f"(area {blob.area_px} px, elong {blob.elongation:.3f}, "
                                         f"fill {blob.fill:.4f})")
            print(f"  trial {trial:3d}: {len(image[0])}x{len(image)} frame, "
                  f"{sum(len(v) for v in measured.values())} labelled blob(s) so far", flush=True)

        total = sum(len(v) for v in measured.values())
        print("")
        print(f"{total} labelled blob(s) over {args.trials} trial(s)")
        lines = report(measured)

        lines += ["", "## kind accuracy at the current thresholds", "",
                  f"`BALL_FILL` = {perceive.BALL_FILL}, `BALL_ELONGATION` = {perceive.BALL_ELONGATION}",
                  "",
                  f"- correct: {seen - len(wrong)}/{seen} "
                  f"({(seen - len(wrong)) / seen * 100:.1f}%)" if seen else "- nothing measured"]
        if wrong:
            lines.append("")
            lines += [f"- wrong: {len(wrong)}/{seen}"]
            lines += ["", "```"] + wrong[:20] + ["```"]
        print(f"kind accuracy: {seen - len(wrong)}/{seen} "
              f"({(seen - len(wrong)) / seen * 100:.1f}%)" if seen else "nothing measured")
        for line in wrong[:10]:
            print(f"  {line}")

        print("\n".join(lines))

        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "kind_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwritten: {RESULTS / 'kind_calibration.md'}")
        print(f"frames:  {args.out}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())