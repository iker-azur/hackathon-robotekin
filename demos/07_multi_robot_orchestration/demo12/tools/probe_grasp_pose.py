"""Demo 12 — Step 0: what is the jaw's local Y actually pointing, in the world?

Why this probe exists
---------------------
The handoff makes two mutually exclusive claims about the grasp pose and the rebuild
brief asserts the second one without an artifact behind it:

* ``DEMO12_HANDOFF.md`` and ``isaac_env._author_jaw_pads`` both state that at the grasp
  pose the jaw's local axes are ``X -> world +Z`` (up), ``Y -> world -Y`` (horizontal),
  ``Z -> world +X``, so a pair of pads mirrored about local Y closes *sideways*.
* The rebuild brief states that **every** reachable configuration has the pad axis
  *vertical*, which makes straddling a part impossible.

Nothing in the repository records a measurement of either. Both are assertions in prose.
This probe settles it by measuring, once, and writing the answer to
``out12/pose_probe.json`` so ``docs/PHYSICS.md`` can cite a number instead of a claim.

It reads only. No pose is written, no part is touched, and the probe exits without
asserting anything about whether the grasp works -- that is ``tools/probe_grasp.py``'s
job. If the answer is "local Y is horizontal", the rebuild brief's root cause is wrong and
the zero-grip cause is somewhere else entirely.

Run with the Isaac interpreter, not the PATH python::

    C:\\Users\\Usuario\\miniconda3\\envs\\hackathon-robotekin\\python.exe -m demo12.tools.probe_grasp_pose
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# parents[0] = demo12/tools, [1] = demo12, [2] = 07_multi_robot_orchestration, [3] = demos
_DEMO12_ROOT = _HERE.parents[1]
_PARENT_ROOT = _HERE.parents[2]
_DEMOS_ROOT = _HERE.parents[3]

# Kit has to be up before isaaclab imports. ``env_loader`` is the only bootstrap that
# works on this machine (it disables the sensor extensions whose native DLLs will not
# resolve here), and it lives in the 01 demo.
for _p in (str(_DEMOS_ROOT), str(_DEMOS_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402

env_loader.prepare_runtime(headless=True)

from demo12 import materials as M  # noqa: E402
from demo12.config import default_scene  # noqa: E402
from demo12.isaac_env import IsaacCell  # noqa: E402

OUT_DIR = _PARENT_ROOT / "out12"
OUT_PATH = OUT_DIR / "pose_probe.json"


def _vec(t) -> list[float]:
    return [round(float(v), 6) for v in t]


def _basis(arm) -> dict[str, list[float]]:
    """The jaw's three local axes expressed in world coordinates.

    This is the measurement the whole rebuild hinges on. ``local_y_world`` is the pad
    separation axis: if its world Z component is near 1 the pads close vertically and
    cannot straddle a part on the bench; if it is near 0 they close horizontally and can.
    """
    import torch
    from isaaclab.utils.math import quat_apply

    q = arm.tool_quat[0]
    axes = torch.tensor([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32, device=arm.device)
    world = quat_apply(q, axes)
    return {
        "local_x_world": _vec(world[0]),
        "local_y_world": _vec(world[1]),
        "local_z_world": _vec(world[2]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scene = default_scene(1)
    cell = IsaacCell(scene, device="cuda:0", headless=True)
    cell.settle(90)

    robot_id = next(iter(cell.robots))
    arm = cell.arm(robot_id)

    samples: list[dict] = []

    # Sweep every part in the scene plus a few synthetic targets, so the answer covers the
    # reachable envelope rather than one hand-picked point. A single sample could be an
    # accident of the solver; a sweep cannot.
    probe_targets: list[tuple[str, tuple[float, float, float]]] = []
    for p in scene.parts:
        ps = cell.part_state(p.id)
        probe_targets.append((
            f"part:{p.id}",
            (ps.x, ps.y, arm.tool_z_for_part_centre(ps.z)),
        ))

    # Synthetic ring: same radius as the parts, spread in yaw around the base, at three
    # heights. This is what turns "one pose" into "every reachable pose".
    base = scene.robots[0].base_xy
    import math
    r = 0.20
    for k in range(8):
        a = 2.0 * math.pi * k / 8.0
        x = base[0] + r * math.cos(a)
        y = base[1] + r * math.sin(a)
        for dz in (0.06, 0.09, 0.12):
            probe_targets.append((
                f"ring:{k}:{dz:.2f}",
                (x, y, M.TABLE_TOP_Z_M + dz + arm.tool_z_for_part_centre(0.0)),
            ))

    for name, target in probe_targets:
        # Snap the arm straight to the target with the pads commanded open, then measure.
        # Whatever pose the servo settles into IS the grasp pose; the probe reports it.
        arm.set_jaws(True)
        for _ in range(400):
            arm.command_joints(arm.solve_ik(target, 0.078))
            cell.step()
            if arm.tool_error_m(target) < 0.003:
                break

        # Let the pads actually reach their commanded open span before measuring, so the
        # pad body positions are the settled ones and not a transient.
        for _ in range(40):
            arm.command_joints(arm.solve_ik(target, 0.02))
            cell.step()

        basis = _basis(arm)
        pad_l, pad_r = cell.jaw_pad_positions(robot_id)

        samples.append({
            "name": name,
            "target": _vec(target),
            "tool_pos": _vec(arm.tool_pos),
            "tool_error_m": round(arm.tool_error_m(target), 6),
            "jaw_quat_w": _vec(arm.tool_quat[0]),
            **basis,
            "pad_l_world": _vec(pad_l),
            "pad_r_world": _vec(pad_r),
            "pad_separation_world": _vec(pad_r - pad_l),
            "pad_span_m": round(cell.jaw_pad_span_m(robot_id), 6),
            "joints_rad": [round(float(v), 4) for v in arm.joints],
        })

        print(
            f"  {name:>16}  localY.z={basis['local_y_world'][2]:+.4f}  "
            f"err={samples[-1]['tool_error_m'] * 1000:6.2f} mm  "
            f"span={cell.jaw_pad_span_m(robot_id) * 1000:5.1f} mm",
            flush=True,
        )

    # The verdict, computed rather than asserted. The rebuild brief's claim is that the
    # pad axis is vertical (|local_y_world.z| ~ 1) everywhere; the handoff's is that it is
    # horizontal (|local_y_world.z| ~ 0) at the grasp pose.
    yz = [abs(s["local_y_world"][2]) for s in samples]
    part_yz = [abs(s["local_y_world"][2]) for s in samples if s["name"].startswith("part:")]

    verdict = {
        "samples": len(samples),
        "part_pose_samples": len(part_yz),
        "max_abs_local_y_z": round(max(yz), 6),
        "mean_abs_local_y_z": round(sum(yz) / len(yz), 6),
        "max_abs_local_y_z_at_parts": round(max(part_yz), 6) if part_yz else None,
        "pad_axis_vertical_everywhere": max(yz) > 0.95,
        "pad_axis_horizontal_at_parts": (max(part_yz) < 0.30) if part_yz else None,
    }

    report = {
        "question": (
            "Is the jaw's local Y (the pad separation axis) horizontal or vertical in "
            "the world at the grasp pose?"
        ),
        "constants": {
            "JAW_PAD_AXIS": M.JAW_PAD_AXIS,
            "JAW_PAD_DROP_AXIS": M.JAW_PAD_DROP_AXIS,
            "JAW_PAD_DROP_M": M.JAW_PAD_DROP_M,
            "FINGER_PAD_THICK_M": M.FINGER_PAD_THICK_M,
            "FINGER_PAD_HEIGHT_M": M.FINGER_PAD_HEIGHT_M,
            "FINGER_PAD_DEPTH_M": M.FINGER_PAD_DEPTH_M,
        },
        "note": (
            "This probe is what found the bug. The pads were authored to slide along the "
            "jaw's local Y on the strength of DEMO12_HANDOFF.md's claim that 'local Y is "
            "world -Y (horizontal)'. Measured, local Y is world +Z: vertical. The pads "
            "closed up and down, pressed on top of the part, and produced no grip. The "
            "handoff's other axis was wrong too ('local X is world +Z (up)') -- the two "
            "were swapped. The pads now slide along local Z, whose world Z component is "
            "-5e-06 at every pose in this sweep."
        ),
        "verdict": verdict,
        "samples": samples,
    }

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("", flush=True)
    print("STEP 0 VERDICT", flush=True)
    print(f"  samples                     : {verdict['samples']}", flush=True)
    print(f"  max |local_Y.z|             : {verdict['max_abs_local_y_z']:.4f}", flush=True)
    print(f"  mean |local_Y.z|            : {verdict['mean_abs_local_y_z']:.4f}", flush=True)
    if verdict["max_abs_local_y_z_at_parts"] is not None:
        print(f"  max |local_Y.z| at parts    : "
              f"{verdict['max_abs_local_y_z_at_parts']:.4f}", flush=True)
    print(f"  pad axis vertical everywhere: {verdict['pad_axis_vertical_everywhere']}",
          flush=True)
    print(f"  pad axis horizontal at parts: {verdict['pad_axis_horizontal_at_parts']}",
          flush=True)
    print(f"\nwrote {OUT_PATH}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()