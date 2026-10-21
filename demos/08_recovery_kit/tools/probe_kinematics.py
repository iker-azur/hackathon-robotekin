"""Mide la cinemática real del asset: eslabones, mordazas y apertura de la pinza.

    python tools/probe_kinematics.py --headless

Escribe ``results/kinematics_probe.json`` y una imagen de la cámara cenital.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402


def collider_aabb(stage, link: str):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    prim = stage.GetPrimAtPath(f"/World/Robot/{link}/collisions")
    if not prim.IsValid():
        prim = stage.GetPrimAtPath(f"/World/Robot/{link}")
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return [list(box.GetMin()), list(box.GetMax())]


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    args = parser.parse_args()
    rt = runtime.boot(args)

    import numpy as np
    import omni.usd
    from isaaclab.sim import SimulationContext

    from recoverykit import config, layout, physics, workcell

    cfg = config.preset("simple")
    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    cell.reset_layout(layout.sample(cfg))
    stage = omni.usd.get_context().get_stage()

    out: dict = {"joint_names": cell.robot.joint_names, "body_names": cell.robot.body_names}
    lim = cell.robot.data.soft_joint_pos_limits[0].cpu().numpy().tolist()
    out["joint_limits"] = dict(zip(cell.robot.joint_names, lim))
    out["bodies_at_start"] = {b: cell.body_pose(b)[0].tolist() for b in cell.robot.body_names}
    out["proprio_start"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cell.proprio().items()}

    sweep = []
    for q in np.linspace(-0.17, 1.7, 12):
        cell.set_gripper_target(float(q))
        for _ in range(240):
            cell.step()
        sweep.append({"q_cmd": float(q), "q": cell.proprio()["gripper_q"],
                      "jaw_aabb": collider_aabb(stage, "jaw"), "gripper_aabb": collider_aabb(stage, "gripper"),
                      "jaw_body": cell.body_pose("jaw")[0].tolist(),
                      "gripper_body": cell.body_pose("gripper")[0].tolist()})
    out["gripper_sweep"] = sweep

    reach = []
    for lift in (-1.2, -0.6, 0.0, 0.6):
        for elbow in (-0.6, 0.0, 0.8, 1.4):
            q = np.array([0.0, lift, elbow, 1.2, 0.0])
            cell.set_arm_targets(q)
            for _ in range(480):
                cell.step()
            pos, quat = cell.body_pose("gripper")
            reach.append({"q_cmd": q.tolist(), "q": cell.proprio()["arm_q"].tolist(),
                          "gripper_pos": pos.tolist(), "gripper_quat": quat.tolist(),
                          "jaw_pos": cell.body_pose("jaw")[0].tolist()})
    out["reach_samples"] = reach
    out["watchdog"] = cell.watchdog.summary()

    cell.set_arm_targets(np.array(cfg.robot.start_joints[:5]))
    for _ in range(480):
        cell.step()
    cell.render()
    cell.render()
    cam = cell.cameras["cenital"]
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    import cv2

    (ROOT / "results").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "results" / "probe_cenital.png"), cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR))
    out["camera_intrinsics"] = cam.data.intrinsic_matrices[0].cpu().numpy().tolist()
    (ROOT / "results" / "kinematics_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[probe] escrito results/kinematics_probe.json", flush=True)
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
