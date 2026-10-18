"""Measure the cube detector against the recorded demonstrations. No Isaac needed.

The demonstrations already carry the ground truth this needs: one line per episode
in ``13_collect_demonstrations_episodes.jsonl`` says where the cube was put, and
the video shows where it appeared in the frame. Comparing the two measures the
detector without a simulator, and without a training run in the way.

Run:

    python .\\demos\\06_imitation_learning\\tools\\17_validate_detector.py

Reports four numbers:

* **detection rate** - how often the cube is found at all
* **repeatability** - how far the centroid moves between frames of the same episode
* **linearity** - R2 of world x against centroid u, which is what says whether a
  policy can learn from the centroid without any calibration
* **calibration refusal** - whether the fit correctly declines to run on the two
  table rows this dataset happens to use
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import cube_detect  # noqa: E402

# Frames sampled from inside "go above cube", where the arm is still travelling
# above the table and the cube is resting on it.
WINDOW = (10, 50, 90, 150)


def load_spots(dataset_root: Path) -> dict[int, tuple[float, float]]:
    """Where the cube was placed in each episode, from the collection log."""
    log = dataset_root / "13_collect_demonstrations_episodes.jsonl"
    spots = {}
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                spots[record["episode"]] = tuple(record["cube_spot"])
    return spots


def collect(dataset_root: Path, spots: dict):
    """Decode the needed frames and detect the cube in each."""
    import av

    needed = {640 * episode + offset for episode in spots for offset in WINDOW}
    video = (
        dataset_root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    )
    detections: dict[int, tuple] = {}
    container = av.open(str(video))
    for index, frame in enumerate(container.decode(video=0)):
        if index in needed:
            detections[index] = frame.to_ndarray(format="rgb24")
        if index > max(needed):
            break
    container.close()

    rows = []
    for episode in sorted(spots):
        for offset in WINDOW:
            image = detections.get(640 * episode + offset)
            if image is None:
                continue
            found = cube_detect.detect_cube(image)
            rows.append((episode, offset, spots[episode], found))
    return rows


def main() -> int:
    import il_dataset

    root = il_dataset.default_dataset_dir()
    if not (root / "meta" / "info.json").exists():
        print(f"ERROR: no dataset at {root}. Run 13_collect_demonstrations.py first.")
        return 1

    spots = load_spots(root)
    print(f"dataset: {root}")
    print(f"episodes: {len(spots)}, sampling frames {WINDOW[0]}-{WINDOW[-1]} of each")
    print()

    rows = collect(root, spots)
    found = [row for row in rows if row[3] is not None]
    print(f"detection rate : {len(found)}/{len(rows)} ({100 * len(found) / len(rows):.1f}%)")

    if not found:
        print("the cube was never found; check the hue and saturation thresholds")
        return 1

    plausible = sum(1 for row in found if row[3].plausible)
    print(f"cube-shaped    : {plausible}/{len(found)}")
    area = np.array([row[3].area_px for row in found])
    print(f"blob area      : mean {area.mean():.0f} px, range {area.min()}-{area.max()} px")
    print()

    # Repeatability: the same cube position, seen in several frames.
    repeats = []
    for episode in sorted(spots):
        seen = [row[3] for row in found if row[0] == episode]
        if len(seen) > 1:
            us = [d.u for d in seen]
            vs = [d.v for d in seen]
            repeats.append(max(np.ptp(us), np.ptp(vs)))
    if repeats:
        print(
            f"repeatability  : median {np.median(repeats):.1f} px, "
            f"worst {max(repeats):.1f} px (same cube, different frames)"
        )
    print()

    # Linearity of the centroid against world position - the number that says a
    # policy can use the centroid directly, with no calibration step.
    for y_value in sorted({row[2][1] for row in found}):
        subset = [row for row in found if row[2][1] == y_value]
        if len(subset) < 4:
            continue
        xs = np.array([row[2][0] for row in subset])
        us = np.array([row[3].u for row in subset])
        design = np.c_[xs, np.ones(len(xs))]
        coefficients, *_ = np.linalg.lstsq(design, us, rcond=None)
        predicted = design @ coefficients
        residual = np.abs(predicted - us)
        total = ((us - us.mean()) ** 2).sum()
        r2 = 1 - ((us - predicted) ** 2).sum() / total if total > 0 else float("nan")
        ppm = abs(coefficients[0])
        print(
            f"row y={y_value:+.2f}: u = {coefficients[0]:.1f}*x + {coefficients[1]:.1f}"
            f"   R2={r2:.4f}   residual <= {residual.max():.1f} px"
            f"   ({residual.max() / ppm * 1000:.1f} mm at {ppm:.0f} px/m)"
        )
    print()

    # The calibration should refuse on this dataset, and say why.
    pixels = [(row[3].u, row[3].v) for row in found]
    table = [row[2] for row in found]
    distinct_rows = len({round(y, 3) for _, y in table})
    calibration = cube_detect.Calibration.fit(pixels, table)
    if calibration is None:
        print(
            f"calibration    : correctly refused - the demonstrations use only "
            f"{distinct_rows} distinct table rows, and a plane-to-image fit needs "
            f"{cube_detect.Calibration.fit.__defaults__[0]}+. Use the pixel centroid."
        )
    else:
        worst = calibration.residual_px(pixels, table)
        print(
            f"calibration    : fitted on {calibration.samples} samples, "
            f"worst residual {worst:.1f} px "
            f"({worst / calibration.pixels_per_metre * 1000:.1f} mm)"
        )
    print()
    print("placement tolerance is 30 mm; anything under ~10 mm here has room to spare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())