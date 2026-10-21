"""Graba imágenes de la cámara cenital con la verdad del simulador, para medir la percepción.

    python tools/capture_perception.py --headless --frames 20 --seed 17100

Cada fotograma: disposición aleatoria (semilla), algunas piezas ya dentro de la
caja y, en parte de los fotogramas, el brazo en una postura que tapa algo.
Escribe ``results/perception_frames/frame_<n>.npz``; ``tools/eval_perception.py``
los puntúa sin necesidad de Isaac.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402


def quat_yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17100)
    args = parser.parse_args()
    rt = runtime.boot(args)

    from isaaclab.sim import SimulationContext

    from recoverykit import config, layout, physics, skills, workcell
    from recoverykit.config import PieceCfg
    from recoverykit.control import Controller

    cfg = config.preset("demo")
    cfg.pieces.append(PieceCfg("verde_1", "green", shape="cylinder"))
    cfg.pieces.append(PieceCfg("verde_2", "green"))
    # la percepción no necesita el hueco de agarre: más piezas, más juntas, más vecinas
    cfg.work_area.min_gap_m = 0.04
    cfg.work_area.angle_deg = (-75.0, 15.0)
    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    box = cfg.containers[0]
    slots = skills.box_slots(box.center_xy_m, box.inner_m)
    out_dir = ROOT / "results" / "perception_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ctrl = Controller(cell)
    occluding = [np.array([0.0, 0.0, 0.3, 1.2, 0.0]), np.array([-0.5, 0.2, 0.2, 1.3, 0.5])]

    rgb = None
    for n in range(args.frames):
        lay = layout.sample(cfg, seed=args.seed + n)
        # 0–2 piezas ya dentro de la caja
        in_box = rng.sample(list(lay), rng.randint(0, 2))
        for k, pid in enumerate(in_box):
            sx, sy = slots[k]
            lay[pid] = (sx + rng.uniform(-0.005, 0.005), sy + rng.uniform(-0.005, 0.005), rng.uniform(-0.6, 0.6))
        cell.reset_layout(lay)
        if n % 4 == 3:
            ctrl.move_joints(occluding[(n // 4) % 2])
        for _ in range(3):
            cell.render()
        rgb, depth = cell.frame("cenital")
        cal = cell.camera_calibration("cenital")
        truth = []
        for p in cfg.pieces:
            t = cell.truth(p.id)
            truth.append({"id": p.id, "color": p.color, "shape": p.shape, "pos": t["pos"].tolist(),
                          "yaw_deg": math.degrees(quat_yaw(t["quat"])), "in_box": p.id in in_box})
        meta = {"pieces": truth, "arm_q": ctrl.q().tolist(), "table_z": cfg.table.top_z_m}
        np.savez_compressed(out_dir / f"frame_{n:03d}.npz", rgb=rgb, depth=depth, k=cal["k"], rot=cal["rot"],
                            pos=cal["pos"], truth_json=json.dumps(meta))
        if n % 4 == 3:
            ctrl.move_joints(np.array(cfg.robot.start_joints[:5]))
        print(f"[captura] {n}: {len(truth)} piezas, {len(in_box)} en caja", flush=True)
        if n == 0:
            # saturación/valor en el centro proyectado de cada pieza y en el robot
            import cv2

            from recoverykit.perception import Camera

            cam = Camera(cal["k"], cal["rot"], cal["pos"], rgb.shape[1], rgb.shape[0])
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            for t in truth:
                uv, _ = cam.project(np.array([t["pos"]]) + np.array([0, 0, 0.0125]))
                u, v = int(uv[0, 0]), int(uv[0, 1])
                if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
                    patch = hsv[max(0, v - 3):v + 4, max(0, u - 3):u + 4].reshape(-1, 3)
                    print(f"[captura] hsv {t['color']:7s} {np.median(patch, 0).astype(int).tolist()}", flush=True)
            robot = hsv[np.abs(depth - (cfg.cameras[0].pos_m[2] - cfg.table.top_z_m)) > 0.08]
            if len(robot):
                print(f"[captura] hsv robot (mediana) {np.median(robot, 0).astype(int).tolist()} "
                      f"p95 sat {int(np.percentile(robot[:, 1], 95))}", flush=True)
            table = hsv[np.abs(depth - (cfg.cameras[0].pos_m[2] - cfg.table.top_z_m)) < 0.001]
            print(f"[captura] hsv mesa (mediana) {np.median(table, 0).astype(int).tolist()}", flush=True)
    if rgb is not None:
        import cv2

        cv2.imwrite(str(ROOT / "results" / "perception_sample.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
