"""Fase 6: escenarios reproducibles del agente completo, sin interfaz.

    python tools/run_scenarios.py --headless --suite all --seed 17000
    python tools/run_scenarios.py --headless --suite kit --runs 10 --seed 17200 --tag aleatorio

Cada ejecución: disposición con semilla, pedido de kit, perturbación física
opcional (disparada por fases del agente), y al final el auditor compara la
caja REAL (verdad del simulador) con el pedido y con lo que el agente cree.
Escribe ``results/scenarios_<tag>.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402

KIT = "Prepara un kit con una pieza roja y dos piezas azules"
SUITES = {
    # nombre: (pedido, perturbación)
    "basico": ("una pieza roja", None),
    "kit": (KIT, None),
    "A_movida": (KIT, "empujar"),
    "B_golpe": (KIT, "golpe_cierre"),
    "B_tiron": (KIT, "tirón_subida"),
    "C_pinza": (KIT, "fallo_pinza"),
    "C_sacada": (KIT, "sacar_caja"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    parser.add_argument("--suite", default="all", help="all | nombre[,nombre]")
    parser.add_argument("--runs", type=int, default=1, help="repeticiones por escenario (semillas consecutivas)")
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    rt = runtime.boot(args)

    from isaaclab.sim import SimulationContext

    from recoverykit import audit, config, kit, layout, physics, workcell
    from recoverykit.agent import Agent
    from recoverykit.perturb import Perturber
    from recoverykit.telemetry import Telemetry

    names = list(SUITES) if args.suite == "all" else args.suite.split(",")
    cfg = config.preset("demo")
    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    tele = Telemetry()
    perturber = Perturber(cell, seed=args.seed)
    agent = Agent(cell, tele, perturber)
    runs = []
    for name in names:
        text, pert = SUITES[name]
        for k in range(args.runs):
            seed = args.seed + k
            perturber.clear()
            perturber.log.clear()
            perturber._rng = random.Random(seed)
            if pert:
                perturber.arm(pert)
            cell.reset_layout(layout.sample(cfg, seed=seed))
            agent.reset_memory()
            tele.clear_run()
            v0 = cell.watchdog.summary()["violations"]
            w0, s0 = time.perf_counter(), cell.sim_time
            res = agent.run(text)
            wall = time.perf_counter() - w0
            aud = audit.check(cell, kit.parse(text).goal)
            believed = sorted((t["color"], t["shape"]) for t in res.verified)
            real = sorted((c["color"], c["shape"]) for c in aud["contents"])
            row = {
                "scenario": name, "seed": seed, "command": text, "perturbation": pert,
                "status": res.status, "duration_s": round(res.duration_s, 2), "metrics": res.metrics,
                "audit": aud, "agent_agrees_with_audit": believed == real,
                "perturbation_log": list(perturber.log), "recoveries": tele.snapshot()["recoveries"],
                "events": [e["text"] for e in tele.snapshot()["events"]],
                "physics_violations": cell.watchdog.summary()["violations"] - v0,
                "wall_s": round(wall, 1), "real_time_factor": round((cell.sim_time - s0) / wall, 2),
            }
            runs.append(row)
            print(f"[escenario] {name} semilla {seed}: {res.status}, auditor {'OK' if aud['satisfied'] else 'NO'}, "
                  f"{res.duration_s:.1f} s, recuperaciones {res.metrics.get('recoveries')}, "
                  f"fallos {res.metrics.get('failures')}, perturbación {[p['kind'] for p in perturber.log]}, "
                  f"violaciones {row['physics_violations']}, acuerdo {row['agent_agrees_with_audit']}", flush=True)

    def rate(rows, key):
        return round(sum(1 for r in rows if key(r)) / len(rows), 3) if rows else None

    attempts = sum(r["metrics"].get("grasp_attempts", 0) for r in runs)
    with_fail = [r for r in runs if r["metrics"].get("recoveries")]
    summary = {
        "runs": len(runs),
        "task_completion_rate": rate(runs, lambda r: r["audit"]["satisfied"]),
        "agent_claims_complete_rate": rate(runs, lambda r: r["status"] == "completado"),
        "false_success": sum(1 for r in runs if r["status"] == "completado" and not r["audit"]["satisfied"]),
        "grasp_success_rate": round(sum(r["metrics"].get("grasps_ok", 0) for r in runs) / attempts, 3) if attempts else None,
        "runs_with_failures": len(with_fail),
        "recovery_success_rate": rate(with_fail, lambda r: r["audit"]["satisfied"]),
        "mean_duration_s": round(sum(r["duration_s"] for r in runs) / len(runs), 2) if runs else None,
        "physics_violations": sum(r["physics_violations"] for r in runs),
        "agent_audit_agreement": rate(runs, lambda r: r["agent_agrees_with_audit"]),
        "mean_real_time_factor": round(sum(r["real_time_factor"] for r in runs) / len(runs), 2) if runs else None,
        "by_scenario": {},
    }
    for name in dict.fromkeys(r["scenario"] for r in runs):
        rows = [r for r in runs if r["scenario"] == name]
        summary["by_scenario"][name] = {"runs": len(rows), "completed": rate(rows, lambda r: r["audit"]["satisfied"]),
                                        "mean_s": round(sum(r["duration_s"] for r in rows) / len(rows), 1)}
    out = {"generated_at": _dt.datetime.now().isoformat(timespec="seconds"), "seed": args.seed, "suite": args.suite,
           "runs": runs, "summary": summary}
    tag = args.tag or args.suite.replace(",", "+")
    path = ROOT / "results" / f"scenarios_{tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in summary.items():
        print(f"[escenario] {k}: {v}", flush=True)
    print(f"[escenario] -> {path}", flush=True)
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
