"""Prove the copies in this directory are faithful to the originals in demos/04_perception.

Demo 10 works on its own copies of the perception modules, so the first thing to prove
is that a copy still produces exactly what the original produced. This script takes a
scene JSON that ``demos/04_perception/09_camera_to_scene.py`` wrote from the simulator,
reads back the very frame it was written from, replays the perception with the camera
model and frame recorded in that JSON, and diffs the result against the JSON itself.

Any difference at all is a bug in the copying, not in the code, and has to be settled
before a single line of the copies is changed. No Isaac Sim, no torch.

Run it with no arguments to see everything that differs. Run it with the fields this demo
deliberately changed ignored to turn it into a regression test - that is the form to use
after any edit to the copies, because it then fails if anything moves that should not:

    python tools/verify_copy.py ^
        --ignore-field inner_rect_m ^
        --ignore-field elongation ^
        --ignore-field fill ^
        --ignore-field hollow ^
        --ignore-field notes

``inner_rect_m`` changed because demo 09 promised a wall inset and never applied one.
``elongation``, ``fill`` and ``hollow`` are new fields demo 10 added to Thing so that the
container threshold is measurable. ``notes`` changed because the detection note lists
which colours exist, and demo 10 measured three more.

    python tools/verify_copy.py --frames ..\\04_perception\\out\\frames --scenes ..\\04_perception\\out\\scenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import perceive  # noqa: E402
from scene import CameraModel, Frame, Scene  # noqa: E402

PERCEPTION_DIR = ROOT.parent / "04_perception"
DEFAULT_FRAMES = PERCEPTION_DIR / "out" / "frames"
DEFAULT_SCENES = PERCEPTION_DIR / "out" / "scenes"


def camera_from_json(blob: dict) -> CameraModel:
    """The camera model demo 09 recorded, rebuilt exactly."""
    return CameraModel.from_dict(blob["camera"])


def frame_from_json(blob: dict) -> Frame:
    """The world frame demo 09 recorded, rebuilt exactly."""
    return Frame(
        space=blob["frame"]["space"],
        left_axis=tuple(blob["frame"]["left_axis"]),
        front_axis=tuple(blob["frame"]["front_axis"]),
        robot_xy=tuple(blob["frame"]["robot_xy"]),
    )


def same(a, b, tol: float = 1e-9) -> bool:
    """Two JSON values equal, treating numbers as equal when they are this close."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def diff(recorded, replayed, path: str = "") -> list[str]:
    """Every leaf where two JSON documents differ, reported as ``path: recorded, replayed``."""
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        problems = []
        for key in sorted(set(recorded) | set(replayed)):
            if key not in recorded:
                problems.append(f"{path}{key}: only in the replay ({replayed[key]!r})")
            elif key not in replayed:
                problems.append(f"{path}{key}: missing from the replay (recorded {recorded[key]!r})")
            else:
                problems += diff(recorded[key], replayed[key], f"{path}{key}.")
        return problems
    if isinstance(recorded, list) and isinstance(replayed, list):
        if len(recorded) != len(replayed):
            return [f"{path[:-1]}: recorded {len(recorded)} entries, replayed {len(replayed)}"]
        problems = []
        for index, (a, b) in enumerate(zip(recorded, replayed)):
            problems += diff(a, b, f"{path}{index}.")
        return problems
    if same(recorded, replayed):
        return []
    return [f"{path[:-1]}: recorded {recorded!r}, replayed {replayed!r}"]


def compare(expected: dict, got: Scene, ignore: tuple[str, ...] = ()) -> list[str]:
    """Every place the replayed scene differs from the recorded one, as readable lines.

    ``pos_m_depth`` is expected to be absent from the replay and is dropped from the
    recorded side before comparing. It comes from the simulator's depth image
    (``perceive.perceive`` takes ``depth=``), and this gate replays RGB from a saved PNG,
    so there is no depth to recompute it from.

    Anything in ``ignore`` is dropped from both sides at every level. It exists so this
    doubles as a regression test for the deliberate changes this demo made to the copies:
    the fields listed there are the ones that are *meant* to differ, and if anything else
    moves the gate fails.
    """
    recorded = json.loads(json.dumps(expected))
    for entry in recorded.get("objects", []) + recorded.get("containers", []):
        if entry.pop("pos_m_depth", None) is not None:
            pass
    strip(recorded, ignore)
    replayed = got.to_dict()
    strip(replayed, ignore)
    return diff(recorded, replayed)


def strip(node, names: tuple[str, ...]) -> None:
    """Drop the given keys from every dict in a JSON document, in place."""
    if not names:
        return
    if isinstance(node, dict):
        for key in names:
            node.pop(key, None)
        for value in node.values():
            strip(value, names)
    elif isinstance(node, list):
        for value in node:
            strip(value, names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES, help="Folder of captured frames")
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES, help="Folder of recorded scene JSON")
    parser.add_argument("--ignore-field", action="append", default=[],
                        help="A JSON key to drop from both sides before comparing, repeatable. "
                             "Use for the fields this demo deliberately changed in its copies.")
    args = parser.parse_args()

    if not args.frames.is_dir() or not args.scenes.is_dir():
        print(f"ERROR: need frames and scenes; got {args.frames} and {args.scenes}")
        print("Run demos/04_perception/09_camera_to_scene.py --source spec --trials 20 first.")
        return 1

    looking_for = colors.colors(colors.DEFAULT_DETECT)
    scenes = sorted(args.scenes.glob("*.json"))
    if not scenes:
        print(f"ERROR: no scene JSON in {args.scenes}")
        return 1

    failures = 0
    print(f"replaying {len(scenes)} recorded scene(s) through this directory's copies")
    print("")
    for path in scenes:
        blob = json.loads(path.read_text(encoding="utf-8"))
        frame_png = args.frames / f"{path.stem}.png"
        if not frame_png.exists():
            print(f"  {path.stem}: SKIP, no frame at {frame_png}")
            failures += 1
            continue

        image = perceive.read_image(frame_png)
        replayed = perceive.perceive(
            image,
            source=blob["source"],
            camera=camera_from_json(blob),
            frame=frame_from_json(blob),
            plane_z=blob["plane_z"],
            looking_for=looking_for,
        )
        problems = compare(blob, replayed, tuple(args.ignore_field))
        if problems:
            failures += 1
            print(f"  {path.stem}: {len(problems)} difference(s)")
            for line in problems[:12]:
                print(f"      {line}")
            if len(problems) > 12:
                print(f"      ... and {len(problems) - 12} more")
        else:
            counted = f"{len(replayed.objects)} object(s), {len(replayed.containers)} container(s)"
            print(f"  {path.stem}: identical, {counted}")

    print("")
    print(f"{len(scenes) - failures}/{len(scenes)} recorded scene(s) reproduced exactly")
    print(f"RESULT: {'SUCCESS' if failures == 0 else 'FAILED'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())