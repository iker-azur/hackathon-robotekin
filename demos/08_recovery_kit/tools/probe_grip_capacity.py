"""¿La pinza sujeta por fricción o por forma? Mide la fuerza de arranque.

    python tools/probe_grip_capacity.py --headless --mu-jaw 0.45
    python tools/probe_grip_capacity.py --headless --mu-jaw 0.20

Coge un cubo, lo levanta y tira de él hacia abajo con una fuerza que crece
despacio hasta que se desliza 5 mm respecto a la pinza. Si la sujeción es por
fricción, la fuerza de arranque escala con μ (≈ 2·μ·N); si no escala, la pieza
está enganchada por la geometría de los dedos.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    parser.add_argument("--mu-jaw", type=float, default=0.45)
    parser.add_argument("--mu-piece", type=float, default=None)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    rt = runtime.boot(args, cameras=False)

    from isaaclab.sim import SimulationContext

    from recoverykit import config, physics, skills, workcell
    from recoverykit.config import Material, PieceCfg
    from recoverykit.control import Controller

    cfg = config.preset("simple")
    cfg.cameras = []
    cfg.robot.jaw_material = Material(args.mu_jaw, args.mu_jaw * 0.85, 0.1)
    if args.mu_piece is not None:
        cfg.physics.piece_material = Material(args.mu_piece, args.mu_piece * 0.85, 0.15)
    cfg.pieces = [PieceCfg("cubo", "red")]
    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    top = cfg.table.top_z_m
    rows = []
    for x, y, yaw in [(0.17, -0.07, 0.0), (0.18, -0.12, 0.3), (0.16, -0.02, -0.4)][: args.trials]:
        cell.reset_layout({"cubo": (x, y, yaw)})
        ctrl = Controller(cell)
        t = cell.truth("cubo")
        w, xq, yq, zq = t["quat"]
        tgt = skills.Target(float(t["pos"][0]), float(t["pos"][1]),
                            math.atan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq)), 0.025, 0.025)
        plan = skills.plan_pick(ctrl.base, top, tgt, ctrl.q())
        out = skills.execute_pick(ctrl, plan)
        if not out.ok:
            rows.append({"pose": [x, y, yaw], "error": out.status})
            continue
        ctrl.wait(0.3)
        rel0 = cell.truth("cubo")["pos"] - cell.body_pose("gripper")[0]
        f_slip = None
        ramp_s, f_max = 4.0, 20.0
        for n in range(int(ramp_s / cfg.physics.dt)):
            f = f_max * n * cfg.physics.dt / ramp_s
            cell.apply_force("cubo", (0.0, 0.0, -f))
            cell.step()
            rel = cell.truth("cubo")["pos"] - cell.body_pose("gripper")[0]
            if np.linalg.norm(rel - rel0) > 0.005:
                f_slip = f
                break
        cell.clear_force("cubo")
        rows.append({"pose": [x, y, yaw], "slip_force_n": f_slip, "gripper_q": out.gripper_q})
        print(f"[sujeción] μ={args.mu_jaw}: arranque a {f_slip} N", flush=True)
        ctrl.gripper(plan.q_open)
        ctrl.move_joints(np.array(cfg.robot.start_joints[:5]))
    mu_c = (args.mu_jaw + cfg.physics.piece_material.static_friction) / 2
    squeeze = cfg.robot.gripper_servo.stall_torque_nm * cfg.robot.gripper_effort_fraction / 0.072
    slips = [r["slip_force_n"] for r in rows if r.get("slip_force_n") is not None]
    out = {"mu_jaw": args.mu_jaw, "mu_combined": mu_c, "squeeze_estimate_n": squeeze,
           "predicted_2muN_n": 2 * mu_c * squeeze, "trials": rows,
           "median_slip_force_n": float(np.median(slips)) if slips else None}
    out["mu_piece"] = cfg.physics.piece_material.static_friction
    path = ROOT / "results" / f"grip_capacity_mu{args.mu_jaw:.2f}_piece{out['mu_piece']:.2f}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[sujeción] mediana {out['median_slip_force_n']} N, predicción 2μN {out['predicted_2muN_n']:.2f} N -> {path}",
          flush=True)
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
