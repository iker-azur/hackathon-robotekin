"""Demo 12 â€” Step 1: run the four-criterion grasp probe in real Isaac Sim.

Every number this writes was produced by PhysX. There is no fallback path, no simulated
mode and no way to get a pass without the gripper actually gripping.

    C:\\Users\\Usuario\\miniconda3\\envs\\hackathon-robotekin\\python.exe -m demo12.tools.probe_grasp
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

for _p in (str(_DEMOS_ROOT), str(_DEMOS_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402

env_loader.prepare_runtime(headless=True)

from demo12 import materials as M  # noqa: E402
from demo12.config import MissionConfig, SceneConfig, default_scene  # noqa: E402
from demo12.grasp import GraspProbe, required_grip_n  # noqa: E402
from demo12.isaac_env import ARM_BODIES_WITH_COLLISION, IsaacCell  # noqa: E402

OUT_DIR = _PARENT_ROOT / "out12"
OUT_PATH = OUT_DIR / "grasp_probe.json"


def arm_collision_report(cell) -> dict:
    """Verify that arm collision is actually enabled, rather than inferring it.

    The grasp probe alone cannot answer this. Turning arm collision on leaves every grasp
    number bit-identical -- 2 colliders kept or 7 -- because on a pick-and-place the arm
    never touches anything: the pads do all the work. So "the grasp still passes" shows
    collision is *harmless*, not that it is *enabled*.

    The first attempt at this was a behavioural test: drive the tool 30 mm below the
    tabletop and see whether it stops. It is inconclusive and was reported as a false
    positive -- the tool halted at 0.1136 m in **both** configurations, 73 mm above the
    surface, because that point is outside the arm's reachable set for a tool-down pose and
    the IK never commands it. The test measured reachability, not collision.

    So this asks the stage directly instead: how many arm colliders are enabled, and how
    many are authored pads. That is what "arm collision is on" has to mean for the claim to
    be true.
    """
    from pxr import Usd, UsdPhysics

    stage = cell.sim.stage
    enabled: list[str] = []
    disabled: list[str] = []
    for robot_id in cell.robots:
        root = stage.GetPrimAtPath(f"/World/Demo12/Robots/{robot_id}")
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            is_pad = "/pad_" in prim.GetPath().pathString
            on = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            if is_pad:
                enabled.append(prim.GetPath().pathString)
            elif on:
                enabled.append(prim.GetPath().pathString)
            else:
                disabled.append(prim.GetPath().pathString)

    arm_enabled = [p for p in enabled if "/pad_" not in p]
    return {
        "arm_colliders_enabled": len(arm_enabled),
        "arm_colliders_disabled": len(disabled),
        "authored_pads": len([p for p in enabled if "/pad_" in p]),
        "arm_collision_is_on": len(arm_enabled) > 0,
        "note": (
            "Verified from the stage, not inferred from the grasp numbers: a pick-and-place "
            "never touches the bench with the arm links, so the grasp probe cannot "
            "distinguish the two settings by behaviour. A drive-below-the-table test was "
            "tried and discarded -- that target is unreachable, so it measured the IK "
            "rather than collision."
        ),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # One robot, so a failure is unambiguously the gripper's and not contention between
    # three arms over the same parts.
    #
    # A **two**-robot scene, then trimmed to the parts and trays this arm can actually
    # serve. The isolation here is about not having another arm in the way, and a single
    # arm at x = 0.20 cannot reach part_2 at 0.430 -- 315 mm, well past the 235 mm it is
    # demonstrated to manage. That used to slip through because ``validate`` checked the
    # 335 mm geometric envelope instead of what the arm does, so this probe was building a
    # cell it had no hope of completing and calling the result a gripper verdict.
    scene = default_scene(2)
    robot_id = scene.robots[0].id     # `scene.robots` is a list of configs, not an id map
    reach = scene.robot(robot_id).reach

    def _usable(item) -> bool:
        from demo12.config import ROBOT_BASE_Y, WORKING_RADIUS_M
        import math as _math
        return (_math.hypot(item.x - scene.robot(robot_id).base_xy[0],
                            item.y - ROBOT_BASE_Y) <= WORKING_RADIUS_M)

    kept_bins = [b for b in scene.bins if _usable(b)]
    kept_parts = [p for p in scene.parts if _usable(p)]
    if not kept_bins or not kept_parts:
        raise SystemExit(f"{robot_id} cannot reach any part or tray in the default cell")

    trimmed = SceneConfig(
        table=scene.table,
        robots=[r for r in scene.robots if r.id == robot_id],
        parts=kept_parts,
        bins=kept_bins,
        mission=MissionConfig(tasks=[]),
    )
    trimmed.validate()

    cell = IsaacCell(trimmed, device="cuda:0", headless=True)
    cell.settle(90)

    

    # Try the two smallest, most tractable parts first: a 20 mm PET ball is the easiest
    # thing a friction gripper can hold, and part_1 is a 30 mm ABS cube.
    attempts: list[dict] = []

    # Iterate the parts the trimmed cell actually holds, and pick the destination from the
    # cell's own bins: the trimmed scene may contain only the trays this one arm can reach,
    # and asking for one it does not have is a KeyError that reads like a physics failure.
    for part_id in [p.id for p in trimmed.parts]:
        part_cfg = cell.part_cfgs[part_id]
        # Not every part has a mission task -- tasks are built by colour match, and the ball
        # has no bin of its colour. Fall back to the first bin rather than assuming a task
        # exists, so the probe can be pointed at any part in the cell.
        bin_id = next((t.bin_id for t in trimmed.mission.tasks if t.part_id == part_id),
                      trimmed.bins[0].id)

        print(f"\n--- {part_id} ({part_cfg.kind}, "
              f"{part_cfg.size_m[0] * 1000:.0f} mm, "
              f"{part_cfg.mass_kg * 1000:.1f} g) -> {bin_id} ---", flush=True)

        probe = GraspProbe(cell, robot_id)
        result = probe.run(part_id, bin_id)

        m = result.measurements
        print(f"  tracking      : {m['tool_error_m'] * 1000:7.2f} mm "
              f"(needs < 1.00)   {'PASS' if result.criteria['tracking'] else 'FAIL'}",
              flush=True)
        print(f"  pad axis      : aligned to {m.get('pad_axis_aligned_to_deg')} deg, "
              f"residual {m.get('pad_axis_remaining_deg')} deg "
              f"(roll sign {m.get('roll_sign')}, gain "
              f"{m.get('roll_gain_measured')})", flush=True)
        print(f"  span free open: {m['span_free_open_m'] * 1000:7.2f} mm "
              f"(needs >= 36.0)  {'PASS' if result.criteria['span_open'] else 'FAIL'}",
              flush=True)
        print(f"  span free shut: {m['span_free_shut_m'] * 1000:7.2f} mm "
              f"(needs < 18.0)   "
              f"{'PASS' if result.criteria['span_shut_free'] else 'FAIL'}",
              flush=True)
        print(f"  span on part  : {m['span_on_part_m'] * 1000:7.2f} mm "
              f"(part is {m['part_grasp_width_m'] * 1000:.1f} mm) "
              f"{'PASS' if result.criteria['pads_on_faces'] else 'FAIL'}",
              flush=True)
        print(f"  grip force    : {m['grip_force_n']:7.3f} N "
              f"(needs > 0)      {'PASS' if result.criteria['grip'] else 'FAIL'}"
              f"   [needs {required_grip_n(part_cfg.mass_kg):.2f} N normal "
              f"for a 2x margin]", flush=True)
        print(f"  drift on close: {m['part_drift_on_close_m'] * 1000:7.2f} mm", flush=True)
        print(f"  part rise     : {m['part_rise_m'] * 1000:7.2f} mm "
              f"(needs >= 50.0)  {'PASS' if result.criteria['part_rose'] else 'FAIL'}",
              flush=True)
        print(f"  carried to bin: {str(m['part_carried_to_bin']):>7}", flush=True)
        print(f"  ends in bin   : {str(result.criteria['part_in_bin']):>7}"
              f"   final=({m['part_final']['x']:.4f}, {m['part_final']['y']:.4f}, "
              f"{m['part_final']['z']:.4f})", flush=True)
        print(f"  resting       : {str(result.criteria['part_resting']):>7}"
              f"   v={m['part_lin_speed']:.5f} m/s", flush=True)

        for f in result.failures:
            print(f"  ! {f}", flush=True)

        attempts.append({"part_id": part_id, "bin_id": bin_id, **result.to_dict()})

        if not result.ok:
            # One failure is enough to stop: a second attempt would just be a second
            # sample of the same broken configuration, and its parts are already disturbed.
            break

    overall = all(a["ok"] for a in attempts) and len(attempts) > 0

    collision = arm_collision_report(cell)
    print(f"\n--- arm collision ({len(ARM_BODIES_WITH_COLLISION)} requested) ---",
          flush=True)
    print(f"  arm colliders enabled  : {collision['arm_colliders_enabled']}",
          flush=True)
    print(f"  arm colliders disabled : {collision['arm_colliders_disabled']}",
          flush=True)
    print(f"  authored pads enabled  : {collision['authored_pads']}", flush=True)
    print(f"  arm collision is on    : {collision['arm_collision_is_on']}", flush=True)

    report = {
        "geometry": {
            "JAW_PAD_AXIS": M.JAW_PAD_AXIS,
            "JAW_PAD_DROP_AXIS": M.JAW_PAD_DROP_AXIS,
            "JAW_PAD_DROP_M": M.JAW_PAD_DROP_M,
            "JAW_PAD_CLOSED_HALF_M": M.JAW_PAD_CLOSED_HALF_M,
            "JAW_PAD_TRAVEL_M": M.JAW_PAD_TRAVEL_M,
            "FINGER_PAD_THICK_M": M.FINGER_PAD_THICK_M,
            "ARM_BODIES_WITH_COLLISION": list(ARM_BODIES_WITH_COLLISION),
        },
        "overall_pass": overall,
        "arm_collision_check": collision,
        "attempts": attempts,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("", flush=True)
    print(f"STEP 1 GRASP PROBE: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"wrote {OUT_PATH}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()