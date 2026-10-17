"""Demo 12 -- does /api/frame.jpg actually produce an image?

The endpoint was written against Replicator's API and imports cleanly, which is not the
same as rendering a frame. This probe answers it, in both configurations the server can be
started in:

    # headless: an offscreen render
    ... -m demo12.tools.probe_frame

    # with a Kit window: the normal path
    ... -m demo12.tools.probe_frame --gui

Reports bytes, dimensions and whether the image is not simply black -- a camera pointed at
nothing returns a valid JPEG of a black rectangle, which is a worse failure than an error
because the interface would show it as a live render.

Writes ``out12/frame_probe.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_DEMO12_ROOT = _HERE.parents[1]
_PARENT_ROOT = _HERE.parents[2]
_DEMOS_ROOT = _HERE.parents[3]

for _p in (str(_DEMOS_ROOT / "07_multi_robot_orchestration"),
           str(_DEMOS_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import env_loader  # noqa: E402

OUT_DIR = _PARENT_ROOT / "out12"
OUT_PATH = OUT_DIR / "frame_probe.json"


def describe(image_bytes: bytes) -> dict:
    """Size and content statistics. A black frame is a failure, not a success."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes))
    pixels = image.convert("RGB")
    colours = pixels.getcolors(maxcolors=1_000_000)
    total = image.width * image.height
    mean = tuple(round(c / total, 1) for c, _ in
                 (sum(n * px[i] for n, px in colours) / total for i in range(3)))
    distinct = len(colours)
    brightest = max(count and px for count, px in colours) if False else max(
        max(px) for _count, px in colours)
    return {
        "width": image.width,
        "height": image.height,
        "distinctColours": distinct,
        "meanRGB": list(mean),
        "brightestPixel": brightest,
        "isBlank": distinct <= 1 or brightest <= 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo 12 frame probe")
    parser.add_argument("--gui", dest="headless", action="store_false", default=True,
                        help="run with a Kit window")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_loader.prepare_runtime(headless=args.headless)

    report: dict = {"headless": args.headless}
    print(f"probe_frame: headless={args.headless}", flush=True)

    # -- what the scene reports about rendering ------------------------------
    try:
        import omni.replicator.core as rep

        report["replicator"] = True
    except Exception as exc:                       # noqa: BLE001
        report["replicator"] = False
        report["replicator_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  replicator unavailable: {exc}", flush=True)

    try:
        import isaaclab.sim as sim_utils

        ctx = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 240.0, device="cuda:0",
                                    render_interval=1))
        report["has_gui"] = ctx.has_gui()
        report["has_primed_camera"] = bool(getattr(ctx, "has_primed_camera", lambda: False)())
        print(f"  has_gui={report['has_gui']} "
              f"has_primed_camera={report['has_primed_camera']}", flush=True)
    except Exception as exc:                       # noqa: BLE001
        report["context_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  context error: {exc}", flush=True)
        OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os._exit(0)

    # -- the function the server actually calls ------------------------------
    #
    # On a real cell, not a stub: ``_frame_bytes`` calls ``cell.render_once``, and more to
    # the point there has to be a scene in front of the camera. A stub would prove the
    # Replicator calls work and nothing about whether they produce an image of anything.
    from demo12.config import default_scene
    from demo12.isaac_env import IsaacCell
    from demo12.server import _frame_bytes

    scene = default_scene(1)
    real_cell = IsaacCell(scene, device="cuda:0", headless=args.headless)
    real_cell.settle(60)
    report["cell_built"] = True

    data, reason = _frame_bytes(real_cell)

    if data is None:
        report["frame"] = None
        report["frame_reason"] = reason
        print(f"  NO FRAME: {reason}", flush=True)
    else:
        report["frame"] = {"bytes": len(data), "via": reason}
        print(f"  pixel accessor: {reason}", flush=True)
        try:
            report["frame"].update(describe(data))
            print(f"  frame OK: {len(data)} bytes, "
                  f"{report['frame']['width']}x{report['frame']['height']}, "
                  f"{report['frame']['distinctColours']} distinct colours, "
                  f"mean={report['frame']['meanRGB']}", flush=True)
            if report["frame"]["isBlank"]:
                print("  WARNING: frame is blank/black -- the camera is seeing nothing",
                      flush=True)
        except Exception as exc:                   # noqa: BLE001
            report["frame"]["describe_error"] = f"{type(exc).__name__}: {exc}"

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()