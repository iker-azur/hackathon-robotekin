"""Probe: can the arm actually reach each part's staging and grasp pose?

The evidence from the 6-task run showed three tasks failing with a *tool error* of 156, 60 and
268 mm -- the arm never arrived, so there was nothing to grasp and none of the grasp numbers
mean anything. This measures that directly, per part and per bin, for every robot, so the
layout can be fixed against numbers instead of theory.

Run with the Isaac interpreter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# parents: [0]=tools [1]=demo12 [2]=07_multi_robot_orchestration [3]=demos
for _p in (str(_HERE.parents[2]), str(_HERE.parents[3] / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo 12 reach probe")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--out",
                        default=str(_HERE.parents[2] / "out12" / "reach_probe.json"))
    args = parser.parse_args()

    env_loader.prepare_runtime(headless=True)

    from demo12 import grasp as G
    from demo12.config import default_scene
    from demo12.isaac_env import IsaacCell

    scene = default_scene(args.robots)
    cell = IsaacCell(scene, device="cuda:0", headless=True)
    cell.settle(90)

    rows = []
    for robot_id in cell.robots:
        arm = cell.arm(robot_id)
        base = scene.robot(robot_id)

        for part in scene.parts:
            state = cell.part_state(part.id)
            staging = (state.x, state.y, state.z + G.STAGING_RISE_M)
            arm.set_jaws(True)
            arm.move_to(cell, staging, open_=True, max_ticks=500, tol_m=0.003)
            staging_err = arm.tool_error_m(staging)

            # Measure the pad offset only once the tool is pointing down, which is the whole
            # reason the probe is staged this way.
            offset = arm.pad_offset_below_tool_m()
            grasp = (state.x, state.y, state.z + offset)
            ticks = arm.move_to(cell, grasp, open_=True, max_ticks=400, tol_m=0.003)
            grasp_err = arm.tool_error_m(grasp)

            horizontal = math.hypot(state.x - base.base_xy[0], state.y - base.base_xy[1])
            rows.append({
                "robot": robot_id, "part": part.id,
                "horizontal_m": round(horizontal, 4),
                "staging_error_m": round(staging_err, 5),
                "grasp_error_m": round(grasp_err, 5),
                "reached": grasp_err < G.TRACKING_TOLERANCE_M,
                "pad_offset_m": round(offset, 5),
                "ticks": ticks,
                "joints_rad": [round(float(v), 4) for v in arm.joints],
            })
            print(f"{robot_id} {part.id:<7} horiz {horizontal*1000:6.1f} mm  "
                  f"staging {staging_err*1000:7.2f} mm  grasp {grasp_err*1000:7.2f} mm  "
                  f"{'OK' if rows[-1]['reached'] else 'UNREACHABLE'}", flush=True)

        for tray in scene.bins:
            target = (tray.x, tray.y, tray.cavity_floor_z_m + 0.03)
            arm.set_jaws(True)
            arm.move_to(cell, target, open_=True, max_ticks=500, tol_m=0.003)
            err = arm.tool_error_m(target)
            horizontal = math.hypot(tray.x - base.base_xy[0], tray.y - base.base_xy[1])
            print(f"{robot_id} {tray.id:<7} horiz {horizontal*1000:6.1f} mm  "
                  f"release  {err*1000:7.2f} mm  "
                  f"{'OK' if err < 0.010 else 'UNREACHABLE'}", flush=True)
            rows.append({"robot": robot_id, "bin": tray.id,
                         "horizontal_m": round(horizontal, 4),
                         "grasp_error_m": round(err, 5),
                         "reached": err < 0.010})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scene": scene.to_dict(), "rows": rows}, indent=2),
                   encoding="utf-8")

    bad = [r for r in rows if not r["reached"]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} targets reached", flush=True)
    for r in bad:
        what = r.get("part") or r.get("bin")
        print(f"  UNREACHABLE {r['robot']} -> {what} "
              f"({r['grasp_error_m']*1000:.0f} mm out, horiz {r['horizontal_m']*1000:.0f} mm)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())