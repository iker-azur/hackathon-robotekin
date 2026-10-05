"""The signed perception bias on this camera, measured from the recorded fixtures.

No simulator: the fixtures were recorded from ``dynamics.randomize(DEFAULT_SCENE, trial)``,
which is a pure function of the trial number, so the layout each frame was taken from can be
recomputed here and compared against the position that frame recorded.

The plane-ray estimate assumes the blob centroid is on the table. A cube's blob centroid is on
its TOP face, 3 cm above the table, so the ray continues past it and lands about a centimetre
further from the camera. That is a fixed, correctable error and ``planner.grasp_pose`` accepts
an ``offset`` for it - which ships as (0, 0).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = HERE.parent
sys.path.insert(0, str(root))
import dynamics
import scene_spec
by_kind: dict[str, list[tuple[float, float]]] = {}

for path in sorted((root / "tests" / "fixtures").glob("*.json")):
    trial = int(path.stem.replace("trial", ""))
    layout = dynamics.randomize(list(scene_spec.DEFAULT_SCENE), trial,
                                spread=dynamics.DEFAULT_SPREAD_M)
    frame = json.loads(path.read_text(encoding="utf-8"))
    loose = [(i.x, i.y) for i in layout if not i.container]
    trays = [(i.x, i.y) for i in layout if i.container]

    def nearest(pos, truth):
        best, best_gap = None, None
        for tx, ty in truth:
            gap = abs(tx - pos[0]) + abs(ty - pos[1])
            if best_gap is None or gap < best_gap:
                best, best_gap = (tx, ty), gap
        return best, best_gap

    for thing in frame.get("objects", []):
        pos = thing.get("pos_m")
        if not pos:
            continue
        best, gap = nearest(pos, loose)
        if best is not None and gap <= 0.05:
            by_kind.setdefault(thing.get("kind") or "any", []).append(
                (pos[0] - best[0], pos[1] - best[1]))

    for thing in frame.get("containers", []):
        pos = thing.get("pos_m")
        if not pos:
            continue
        best, gap = nearest(pos, trays)
        if best is not None and gap <= 0.05:
            by_kind.setdefault("tray", []).append((pos[0] - best[0], pos[1] - best[1]))

print("signed perception bias, mm  (perceived minus truth)")
print(f"{'kind':8s} {'n':>4s} {'dx mean':>9s} {'dy mean':>9s} {'|err| mean':>11s} {'max':>7s}")
for kind, pairs in sorted(by_kind.items()):
    dx = [p[0] for p in pairs]
    dy = [p[1] for p in pairs]
    err = [(abs(a) + abs(b)) / 2 * 1000 for a, b in pairs]
    print(f"{kind:8s} {len(pairs):4d} {sum(dx) / len(dx) * 1000:9.1f} {sum(dy) / len(dy) * 1000:9.1f}"
          f" {sum(err) / len(err):11.1f} {max(err):7.1f}")

allpairs = [p for pairs in by_kind.values() for p in pairs]
dx = [p[0] for p in allpairs]
dy = [p[1] for p in allpairs]
print(f"\noverall: dx {sum(dx) / len(dx) * 1000:+.1f} mm, dy {sum(dy) / len(dy) * 1000:+.1f} mm")
print("a consistent dy of this size is the blob-centroid bias, and it is what --offset is for")