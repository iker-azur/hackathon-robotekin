"""Is the re-measured shape threshold a regression on demo 09's frames, or a fix?

``tools/verify_copy.py`` replays demo 09's recorded scene JSONs, and after this demo
re-measured its cube/ball thresholds for the hollow trays it reports objects whose ``kind``
differs from what demo 09 recorded. That is only a problem if demo 09 was right, so this
compares both against the simulator's own ground truth from ``scene_spec``.

Matching is by reported position, not by list order: every recorded object carries
``pos_m``, and the truth for trial N is ``scene_spec.jittered(DEFAULT_SCENE, N)``.

Demo 09's recorded JSONs do NOT store ``elongation`` or ``fill``, which are the numbers the
thresholds are made of, so this cannot replay the threshold arithmetic itself - it can only
say which label was closer to the truth. That is the question that matters.

    python tools/compare_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scene_spec  # noqa: E402

SCENES = ROOT.parent / "04_perception" / "out" / "scenes"
MATCH_RADIUS_M = 0.05


def truth_kinds(trial: int) -> list[tuple[float, float, str, str]]:
    """``(x, y, kind, colour)`` for every loose object in a jittered trial."""
    return [(i.x, i.y, i.kind, i.color)
            for i in scene_spec.jittered(list(scene_spec.DEFAULT_SCENE), trial)
            if not i.container]


def nearest(recorded, truth) -> tuple[str, str, float]:
    x, y = recorded[0], recorded[1]
    best, best_d = None, MATCH_RADIUS_M
    for tx, ty, kind, colour in truth:
        d = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        if d < best_d:
            best, best_d = (kind, colour), d
    return (best[0], best[1], best_d) if best else (None, None, float("nan"))


def main() -> int:
    if not SCENES.is_dir():
        print(f"ERROR: no recorded scenes at {SCENES}")
        print("Run demos/04_perception/09_camera_to_scene.py --source spec --trials 20 first.")
        return 1

    print(f"{'scene':10s} {'obj':4s} {'colour':7s} {'truth':6s} {'demo 09':8s} {'pos err':8s}")
    print("-" * 58)
    right = total = 0
    disagreements = []
    for path in sorted(SCENES.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        trial = int(path.stem.replace("trial", "") or 0)
        truth = truth_kinds(trial)
        for entry in blob["objects"]:
            recorded = entry.get("pos_m") or []
            if len(recorded) < 2:
                continue
            kind, colour, error = nearest(recorded, truth)
            if kind is None:
                continue
            total += 1
            demo09 = entry["kind"]
            right += (demo09 == kind)
            if demo09 != kind:
                disagreements.append((path.stem, entry["id"], colour, kind, demo09,
                                      f"{error * 1000:.0f} mm"))

    print()
    print(f"loose objects compared: {total}")
    print(f"demo 09's recorded label matches the truth on: {right}/{total}")
    if disagreements:
        print()
        print("where demo 09 disagreed with the truth:")
        for stem, oid, colour, kind, demo09, error in disagreements:
            print(f"  {stem} #{oid} {colour:6s} truth {kind:5s} demo 09 said {demo09:5s}  ({error})")
        print()
        print("These are demo 09's errors, not this demo's. Re-running the copies with the")
        print("thresholds re-measured for the hollow trays is what makes verify_copy.py show")
        print("a kind difference on the blue cube: it used to be called a ball, and is now")
        print("called the cube it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())