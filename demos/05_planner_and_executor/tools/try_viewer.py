"""Open the viewer on a saved frame. No simulator, so this is quick to run.

    python tools/try_viewer.py --image ..\04_perception\out\frames\trial000.png --seconds 3
    python tools/try_viewer.py --headless          # no window at all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import perceive  # noqa: E402
import scene_spec  # noqa: E402
import viewer as viewer_module  # noqa: E402
from scene import CameraModel, Frame  # noqa: E402

DEFAULT_IMAGE = ROOT.parent / "04_perception" / "out" / "frames" / "trial000.png"
DEFAULT_SCENE = ROOT.parent / "04_perception" / "out" / "scenes" / "trial000.json"

# The camera demo 09 recorded for its top sensor, so this needs no simulator either.
RECORDED = ROOT.parent / "04_perception" / "out" / "scenes" / "trial000.json"


def camera_and_frame() -> tuple[CameraModel, Frame]:
    import json

    blob = json.loads(RECORDED.read_text(encoding="utf-8"))
    camera = CameraModel.from_dict(blob["camera"])
    frame = Frame(space=blob["frame"]["space"],
                  left_axis=tuple(blob["frame"]["left_axis"]),
                  front_axis=tuple(blob["frame"]["front_axis"]),
                  robot_xy=tuple(blob["frame"]["robot_xy"]))
    return camera, frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--seconds", type=float, default=3.0, help="Close the window after this long")
    parser.add_argument("--headless", action="store_true", help="No window; write a frame instead")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "viewer_try")
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: no image at {args.image}")
        print("Run demos/04_perception/09_camera_to_scene.py --source spec --trials 20 first.")
        return 1

    image = perceive.read_image(args.image)
    camera, frame = camera_and_frame()
    looking_for = colors.colors(colors.DEFAULT_DETECT)

    print(f"image  : {args.image.name} {image.shape[1]}x{image.shape[0]}")
    print(f"camera : fx={camera.matrix[0][0]:.1f} at {camera.pos_w}")

    if args.headless:
        writer = viewer_module.FrameWriter(args.out, every=1)
        writer.offer(image)
        print(f"headless: wrote {writer.written} frame(s) to {args.out}")
        return 0

    with viewer_module.Viewer(enabled=True) as win, \
            viewer_module.PerceptionWorker(camera, frame, scene_spec.TABLE_TOP_Z, looking_for) as worker:
        if not win.enabled:
            print(f"ERROR: the viewer is disabled ({win._why_disabled})")
            return 1
        print(f"viewer : enabled, {viewer_module.KEY_HELP}")
        print("        close the window, or press q, to stop early")

        deadline = time.perf_counter() + args.seconds
        shown = 0
        while time.perf_counter() < deadline:
            keys = win.show(image, "saved frame, no simulator")
            shown += 1
            worker.submit(image)
            got = worker.latest()
            if got is not None and shown % 5 == 0:
                win.show(got[0], f"annotated: {got[1].summary()[:70]}")
            if win.wants_quit(keys):
                print("\nquit pressed")
                break
            time.sleep(1.0 / max(1.0, args.fps))

        annotated, scene = worker.latest() if worker.latest() else (None, None)
        print(f"\nframes shown : {shown}")
        print(f"perceived    : {worker.finished} finished, {worker.dropped} dropped")
        print(f"scene        : {scene.summary() if scene else 'none'}")
        if scene is not None:
            print(f"objects      : {[(t.id, t.kind, t.color) for t in scene.objects]}")
            print(f"containers   : {[(t.id, t.kind, t.color) for t in scene.containers]}")
        if annotated is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            perceive.save_png(args.out / "annotated.png", annotated)
            print(f"saved        : {args.out / 'annotated.png'}")
    print("RESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())