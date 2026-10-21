"""Fase 3: banco de agarre físico estricto (sin ninguna sujeción artificial).

    python tools/benchmark_grasp.py --headless --trials 30 --seed 17001

Cada ensayo: una pieza en una pose aleatoria (semilla fija) dentro de la zona
de trabajo, el brazo la coge por fricción, la lleva a un hueco de la caja y la
suelta. El objetivo se toma de la VERDAD del simulador para medir sólo la
manipulación; la percepción se mide aparte. Se registra también si el
veredicto propioceptivo del agente (mordaza bloqueada) coincide con la verdad.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
import sys
import time
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
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17001)
    parser.add_argument("--shapes", default="cube,cylinder")
    parser.add_argument("--size", type=float, default=0.025)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    rt = runtime.boot(args, cameras=False)

    from isaaclab.sim import SimulationContext

    from recoverykit import config, kinematics as K, physics, skills, workcell
    from recoverykit.config import PieceCfg
    from recoverykit.control import Controller

    shapes = args.shapes.split(",")
    cfg = config.preset("simple")
    cfg.cameras = []
    cfg.pieces = [PieceCfg(f"p_{s}", "red" if s == "cube" else "green", shape=s, size_m=args.size) for s in shapes]
    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    top = cfg.table.top_z_m
    box = cfg.containers[0]
    slots = skills.box_slots(box.center_xy_m, box.inner_m)
    rng = random.Random(args.seed)
    area = cfg.work_area
    trials: list[dict] = []
    t_wall = time.perf_counter()

    for shape in shapes:
        name = f"p_{shape}"
        piece = cfg.piece(name)
        for i in range(args.trials):
            r = math.sqrt(rng.uniform(area.r_m[0] ** 2, area.r_m[1] ** 2))
            a = math.radians(rng.uniform(*area.angle_deg))
            yaw = rng.uniform(-math.pi / 4, math.pi / 4)
            x, y = r * math.cos(a), r * math.sin(a)
            cell.reset_layout({name: (x, y, yaw)})
            ctrl = Controller(cell)
            truth = cell.truth(name)
            tgt = skills.Target(float(truth["pos"][0]), float(truth["pos"][1]), quat_yaw(truth["quat"]),
                                piece.size_m, piece.height, shape)
            rec = {"shape": shape, "trial": i, "pose": [round(x, 4), round(y, 4), round(math.degrees(yaw), 1)]}
            plan = skills.plan_pick(ctrl.base, top, tgt, ctrl.q())
            if isinstance(plan, str):
                rec.update(plan=plan, agent_verdict="inalcanzable", lifted_truth=False, carried_truth=False,
                           verdict_matches_truth=True, place_err_mm=None, max_slip_mm=None)
                trials.append(rec)
                print(f"[agarre] {shape} {i}: plan imposible — {plan}", flush=True)
                continue
            rec["plan"] = "ok"
            z0 = float(cell.truth(name)["pos"][2])
            t0 = cell.sim_time
            out = skills.execute_pick(ctrl, plan)
            rec["pick_status"] = out.status
            rec["pick_detail"] = out.detail
            rec["gripper_q"] = out.gripper_q
            rec["q_contact"] = plan.q_contact
            z_lift = float(cell.truth(name)["pos"][2])
            rec["lift_rise_m"] = z_lift - z0
            rec["lifted_truth"] = z_lift - z0 > 0.04
            rec["pick_s"] = cell.sim_time - t0
            slip = 0.0
            in_box = False
            place_err = None
            if out.ok:
                slot = slots[i % len(slots)]
                place = skills.plan_place(ctrl.base, top, slot, top + box.floor_m, piece.height, plan.local,
                                          skills._yaw_candidates(0.0, shape), ctrl.q())
                if isinstance(place, str):
                    rec["place_plan"] = place
                else:
                    # deslizamiento: distancia pieza–punto de agarre durante el transporte
                    def watch(info, _plan=plan):
                        nonlocal slip
                        tool = K.fk(dict(zip(K.ARM, info["proprio"]["arm_q"])), ctrl.base)["gripper"]
                        tcp = tool[:3, :3] @ _plan.local + tool[:3, 3]
                        if cell.truth(name)["pos"][2] > top + 0.03:
                            slip = max(slip, float(np.linalg.norm(cell.truth(name)["pos"] - tcp)))
                        return None

                    t1 = cell.sim_time
                    ctrl.monitor = watch
                    place_out = skills.execute_place(ctrl, plan, place)
                    ctrl.monitor = None
                    rec["place_status"] = place_out.status
                    rec["place_s"] = cell.sim_time - t1
                    ctrl.wait(0.5)
                    p = cell.truth(name)["pos"]
                    hx, hy = box.inner_m[0] / 2, box.inner_m[1] / 2
                    in_box = bool(abs(p[0] - box.center_xy_m[0]) < hx and abs(p[1] - box.center_xy_m[1]) < hy
                                  and p[2] < top + box.outer_m[2])
                    place_err = float(np.hypot(p[0] - slot[0], p[1] - slot[1]))
            rec["max_slip_mm"] = round(slip * 1000, 2)
            rec["carried_truth"] = bool(out.ok and in_box)
            rec["in_box_truth"] = in_box
            rec["place_err_mm"] = None if place_err is None else round(place_err * 1000, 2)
            rec["agent_verdict"] = "agarrada" if out.ok else out.status
            rec["verdict_matches_truth"] = bool(out.ok == rec["lifted_truth"])
            trials.append(rec)
            print(f"[agarre] {shape} {i}: {out.status} ({out.phase}) levantada={rec['lifted_truth']} "
                  f"en_caja={in_box} desliz={rec['max_slip_mm']} mm err={rec['place_err_mm']} mm "
                  f"q={out.gripper_q} contacto={plan.q_contact:.3f}", flush=True)
            ctrl.move_joints(np.array(cfg.robot.start_joints[:5]))

    summary = {}
    for shape in shapes:
        rows = [t for t in trials if t["shape"] == shape]
        n = max(1, len(rows))
        errs = [t["place_err_mm"] for t in rows if t.get("place_err_mm") is not None]
        slips = [t["max_slip_mm"] for t in rows if t.get("max_slip_mm") is not None]
        cycles = [t.get("pick_s", 0) + t.get("place_s", 0) for t in rows if t.get("carried_truth")]
        s = {
            "trials": len(rows),
            "plannable": sum(t.get("plan") == "ok" for t in rows),
            "lift_rate": sum(bool(t["lifted_truth"]) for t in rows) / n,
            "carry_rate": sum(bool(t["carried_truth"]) for t in rows) / n,
            "verdict_agreement": sum(bool(t["verdict_matches_truth"]) for t in rows) / n,
            "median_place_err_mm": float(np.median(errs)) if errs else None,
            "max_place_err_mm": max(errs) if errs else None,
            "max_slip_mm": max(slips) if slips else None,
            "mean_cycle_s": float(np.mean(cycles)) if cycles else None,
        }
        s["gate_pass"] = s["lift_rate"] >= 0.90 and s["carry_rate"] >= 0.85
        summary[shape] = s
        print(f"[agarre] RESUMEN {shape}: levantadas {s['lift_rate']:.0%}, en caja {s['carry_rate']:.0%}, "
              f"acuerdo veredicto {s['verdict_agreement']:.0%}, puerta {'SUPERADA' if s['gate_pass'] else 'NO'}",
              flush=True)
    result = {"generated_at": _dt.datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
              "trials_per_shape": args.trials, "size_m": args.size,
              "target_source": "verdad del simulador (sólo manipulación)",
              "config": {"physics": cfg.to_dict()["physics"], "robot": cfg.to_dict()["robot"]},
              "trials": trials, "summary": summary, "watchdog": cell.watchdog.summary(),
              "wall_s": time.perf_counter() - t_wall}
    tag = f"_{args.tag}" if args.tag else ""
    path = ROOT / "results" / f"grasp_benchmark{tag}.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"[agarre] -> {path}", flush=True)
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
