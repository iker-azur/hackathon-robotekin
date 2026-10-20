"""Compare USD-authored pad centres with Demo 11's measured finger contact frames.

Run after Isaac Sim setup with:
    python tools/probe_contact_pad_frames.py --headless --device cpu

This is a frame-calibration check only; it does not command a grasp.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isaac_io  # noqa: E402
import gripper_contact_pads  # noqa: E402
import scene_spec  # noqa: E402


def demo11_module():
    spec = importlib.util.spec_from_file_location(
        "demo11_pad_frame_probe", ROOT / "11_real_sorting_with_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cpu")
    args = parser.parse_args()

    module = demo11_module()
    isaac_io.prepare_runtime(headless=args.headless)
    env, _sensor = module.build_real_env(
        "top", list(scene_spec.DEFAULT_SCENE), device=args.device,
        experimental_contact_pads=True, experimental_contact_pad_shape="flat",
        experimental_contact_pad_thickness_m=gripper_contact_pads.PAD_THICKNESS_M)
    try:
        env.reset()
        bot = module.StrictExecutor(env).with_layout(list(scene_spec.DEFAULT_SCENE))
        measured = bot.finger_contact_points()
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        report = {}
        robot = env.scene["robot"]
        for name in ("gripper", "jaw"):
            candidates = [
                f"/World/envs/env_0/Robot/so101_new_calib/{name}/demo11_rubber_contact_pad",
                f"/World/envs/env_0/Robot/{name}/demo11_rubber_contact_pad",
            ]
            pad_path = next((p for p in candidates if stage.GetPrimAtPath(p)), None)
            if pad_path is None:
                raise RuntimeError(f"No se encontró el pad USD del enlace {name}")
            matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(pad_path))
            actual = matrix.ExtractTranslation()
            actual_xyz = [float(actual[i]) for i in range(3)]
            body_index = robot.find_bodies(name)[0][0]
            quat = robot.data.body_quat_w[0, body_index].detach().cpu().numpy()
            normal = module._rotate_local_offset(
                quat, gripper_contact_pads.PAD_NORMALS_LOCAL[name])
            expected_xyz = np.asarray(measured[name], dtype=float) - normal * (
                gripper_contact_pads.PAD_THICKNESS_M / 2.0)
            expected_xyz = [float(v) for v in expected_xyz]
            error = sum((a - b) ** 2 for a, b in zip(actual_xyz, expected_xyz)) ** 0.5
            report[name] = {"pad_prim": pad_path, "usd_world_m": actual_xyz,
                            "measured_contact_world_m": expected_xyz,
                            "centre_error_mm": error * 1000.0}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
