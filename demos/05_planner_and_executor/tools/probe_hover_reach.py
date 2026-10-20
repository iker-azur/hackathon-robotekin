"""Measure how high above the table the arm can actually hold its grip point, by radius.

The planner asks for a pre-grasp hover some clearance above the part. How high that can be is a
property of the arm, not of the plan: the SO-101 holds its fingers vertical, and the higher the
tool sits the further the wrist has to fold back, until the pose no longer exists. Before this
probe the clearance was a constant (90 mm, or 150 mm for the 30 mm ball) and the strict-physical
logs showed it failing at 0.24-0.26 m from the base with "the aperture is still 150 mm from the
part".

    python tools/probe_hover_reach.py --headless
    python tools/probe_hover_reach.py --headless --radii 0.22 0.25 --clearances 0.06 0.09 0.15

For each target point and clearance it parks, runs the executor's own ``above`` motion to the
commanded grip point, and reports how far the *measured* grip point ended up from it. The
output is the table ``planner.max_reachable_clearance`` is calibrated from.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import executor  # noqa: E402
import isaac_io  # noqa: E402
import scene_spec  # noqa: E402

BASE_X, BASE_Y = scene_spec.ROBOT_XY
GRASP_Z = scene_spec.TABLE_TOP_Z + 0.015  # centre of a 30 mm part resting on the table


def demo11_module():
    spec = importlib.util.spec_from_file_location("demo11_hover_probe",
                                                  ROOT / "11_real_sorting_with_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def targets(radii, lateral_offsets):
    """Table points at the given distances from the base, at several lateral offsets."""
    out = []
    for radius in radii:
        for lateral in lateral_offsets:
            if abs(lateral) >= radius:
                continue
            out.append((BASE_X + lateral, BASE_Y + math.sqrt(radius ** 2 - lateral ** 2)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", nargs="+", type=float,
                        default=[.20, .225, .25, .275, .30])
    parser.add_argument("--lateral", nargs="+", type=float, default=[-.08, 0.0, .08],
                        help="x offsets from the base axis, in metres")
    parser.add_argument("--clearances", nargs="+", type=float,
                        default=[.03, .06, .09, .12, .15],
                        help="grip-point height above the part centre, in metres")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "hover_reach_20261019.json")
    args = parser.parse_args(argv)

    module = demo11_module()
    isaac_io.prepare_runtime(headless=args.headless)
    selected = [next(i for i in scene_spec.DEFAULT_SCENE if i.prim == scene_spec.SHIPPED_CUBE)]
    env, _sensor = module.build_real_env("top", selected, device=args.device)
    rows = []
    try:
        env.reset()
        bot = module.StrictExecutor(env, allow_kinematic=False).with_layout(selected)
        for x, y in targets(args.radii, args.lateral):
            radius = math.hypot(x - BASE_X, y - BASE_Y)
            for clearance in args.clearances:
                bot.park(steps=executor.PARK_STEPS)
                goal = np.array([x, y, GRASP_Z + clearance])
                result = bot.do("above", goal, executor.OPEN,
                                steps=executor.STEP_COUNTS["above"])
                body = bot._gripper_pos()[0].detach().cpu().numpy()
                hold = bot.hold_offset_world()[0].detach().cpu().numpy()
                error = body + hold - goal
                # The hold point is the executor's fitted model of where the fingers are; the
                # aperture centre is where the finger pads measurably are. They agree while the
                # tool is vertical and part when the wrist has to tilt to reach, which is the
                # error a camera-driven approach actually has to servo out.
                aperture = bot.aperture_centre()
                aperture_error = (None if aperture is None else
                                  float(np.hypot(aperture[0] - x, aperture[1] - y)))
                row = {"x": round(x, 4), "y": round(y, 4), "radius_m": round(radius, 4),
                       "clearance_m": clearance,
                       "xy_error_mm": round(float(np.hypot(error[0], error[1])) * 1000, 1),
                       "z_error_mm": round(float(error[2]) * 1000, 1),
                       "error_mm": round(float(np.linalg.norm(error)) * 1000, 1),
                       "aperture_xy_error_mm": (None if aperture_error is None
                                                else round(aperture_error * 1000, 1)),
                       "aperture_z_m": None if aperture is None else round(float(aperture[2]), 4),
                       "ok": bool(result.ok)}
                rows.append(row)
                print(f"r {radius:.3f} ({x:+.3f},{y:+.3f}) clearance {clearance * 1000:4.0f} mm: "
                      f"hold error {row['error_mm']:6.1f} mm (xy {row['xy_error_mm']:.1f}, "
                      f"z {row['z_error_mm']:+.1f}), aperture xy error "
                      f"{row['aperture_xy_error_mm']} mm", flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"grasp_z": GRASP_Z, "rows": rows}, indent=2),
                           encoding="utf-8")
    print(f"report: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
