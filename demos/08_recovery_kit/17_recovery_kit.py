"""Demo 17 — RecoveryKit: un SO-101 prepara kits y se recupera cuando el mundo cambia.

    python 17_recovery_kit.py --open                 # panel en http://127.0.0.1:8770
    python 17_recovery_kit.py --headless --kit "una roja y dos azules"   # una ejecución sin panel

La simulación corre en el hilo principal; el panel web en otro hilo y sólo deja
órdenes en una cola. Cambiar la escena (piezas, caja, cámara, física) guarda
``out/scene_config.json`` y reinicia el proceso, porque Isaac no añade cuerpos a
una escena ya en marcha.
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402

SCENE_FILE = ROOT / "out" / "scene_config.json"
DEMO_KIT = "Prepara un kit con una pieza roja y dos piezas azules"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(parser)
    parser.add_argument("--preset", default=None, help="demo | simple | formas (ignora out/scene_config.json)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true", help="abrir el panel en el navegador")
    parser.add_argument("--kit", default=None, help="ejecutar este pedido y salir (sin panel)")
    parser.add_argument("--perturb", default=None, help="perturbación armada para --kit")
    parser.add_argument("--realtime", type=float, default=1.0, help="1 = tiempo real; 0 = lo más rápido posible")
    args = parser.parse_args()
    rt = runtime.boot(args)

    from isaaclab.sim import SimulationContext

    from recoverykit import audit, config, kit, layout, physics, workcell
    from recoverykit.agent import Agent
    from recoverykit.perturb import LABELS, Perturber
    from recoverykit.telemetry import Telemetry

    if args.preset:
        cfg = config.preset(args.preset)
    elif SCENE_FILE.exists():
        cfg = config.load(SCENE_FILE)
    else:
        cfg = config.preset("demo")
    if args.seed is not None:
        cfg.seed = args.seed
    for note in config.validate(cfg):
        print(f"[recoverykit] {note}", flush=True)

    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    tele = Telemetry()
    perturber = Perturber(cell, seed=cfg.seed)

    def viz():
        tele.push_poses(cell.viz_poses())

    agent = Agent(cell, tele, perturber, realtime=0.0 if args.kit else args.realtime, viz=viz)
    seed = cfg.seed

    def lay_out(new_seed: int) -> None:
        nonlocal seed
        seed = new_seed
        perturber.clear()
        perturber.log.clear()
        cell.reset_layout(layout.sample(cfg, seed=seed))
        agent.reset_memory()
        tele.clear_run()
        tele.set(status="esperando", seed=seed, config=cfg.to_dict(), armed_perturbations=[])
        agent.ctrl.move_joints(agent.look_q)
        agent.look("inicio")
        viz()

    def run_kit(text: str) -> None:
        tele.set(armed_perturbations=perturber.pending(), audit=None)
        res = agent.run(text)
        req = kit.parse(text)
        aud = audit.check(cell, req.goal) if req.ok else None
        tele.set(audit=aud, perturbations=list(perturber.log), armed_perturbations=perturber.pending())
        if aud is not None:
            agree = aud["satisfied"] == (res.status == "completado")
            print(f"[recoverykit] {res.status} en {res.duration_s:.1f} s · auditor del simulador: "
                  f"{'kit correcto' if aud['satisfied'] else 'kit incorrecto'} "
                  f"({'coincide' if agree else 'NO coincide'} con el agente)", flush=True)

    lay_out(seed)

    # ------------------------------------------------------ modo sin panel
    if args.kit:
        if args.perturb:
            perturber.arm(args.perturb)
        run_kit(args.kit)
        rt.close(0)

    # -------------------------------------------------------------- panel
    from server.app import start

    start(tele, lambda: cfg, LABELS, port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[recoverykit] panel en {url}", flush=True)
    if args.open:
        webbrowser.open(url)

    idle_render = 0.0
    while rt.running():
        try:
            cmd = tele.commands.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd is None:
            # en reposo la física sigue: si algo empuja una pieza, se mueve
            cell.step()
            now = time.perf_counter()
            if now - idle_render > 0.25:
                idle_render = now
                rgb, _ = agent._grab()
                agent._publish_frame(rgb, None)
                viz()
            time.sleep(cfg.physics.dt * 0.5)
            continue
        if cmd.kind == "kit":
            run_kit(cmd.payload["text"])
        elif cmd.kind == "demo":
            perturber.arm("empujar")
            tele.event("perturbación", f"Demo: perturbación preparada — {LABELS['empujar']}")
            run_kit(DEMO_KIT)
        elif cmd.kind == "perturb":
            perturber.arm(cmd.payload["kind"])
            tele.set(armed_perturbations=perturber.pending())
            tele.event("perturbación", f"Perturbación preparada: {LABELS[cmd.payload['kind']]}")
        elif cmd.kind == "reset":
            tele.clear_stop()
            lay_out(seed)
        elif cmd.kind == "randomize":
            lay_out(random.randint(1, 10**6))
        elif cmd.kind == "config":
            config.from_dict(cmd.payload["config"]).save(SCENE_FILE)
            print("[recoverykit] escena nueva guardada; reiniciando la simulación", flush=True)
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                      *[a for a in sys.argv[1:] if a != "--open"]])
    rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
