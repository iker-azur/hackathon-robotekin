"""Demo 18 — clasificación de piezas por voz: voz → visión → política → verificación, en Isaac Sim.

    python 18_voice_bin_picking.py --open                       # panel en http://127.0.0.1:8780, guion del experto
    python 18_voice_bin_picking.py --voice --view --open        # con micrófono y vista 3/4 en el panel
    python 18_voice_bin_picking.py --policy act:<carpeta pretrained_model> --open
    python 18_voice_bin_picking.py --policy act+guion:<carpeta> --open
    python 18_voice_bin_picking.py --headless --text "Clasifica las piezas rojas a la izquierda y las verdes a la derecha"

Política de agarre (``--policy``), y se muestra siempre en el panel:

* ``guion``            el experto programado (``recoverykit.skills``). **No es aprendizaje.** Es el valor por defecto
                       porque es lo único que existe hasta que se entrene un ACT (``colab/02_entrenar_act.ipynb``);
* ``act:<carpeta>``    la política ACT de LeRobot aprendida con demostraciones;
* ``act+guion:<carpeta>`` ACT, y si falla dos veces con una pieza, el guion para esa pieza (cada intento queda etiquetado).

La simulación corre en el hilo principal; el panel, en otro, y sólo deja órdenes en una cola. Whisper y el modelo
ACT se cargan **antes** de arrancar Isaac (dos librerías nativas pesadas después de Kit rompen el proceso).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import binpick  # noqa: E402,F401  (añade recoverykit al sys.path)
from binpick import commands, config as C, policy as POL, scene_gen as G  # noqa: E402
from binpick.orchestrator import POLICY_MODES, SortingTask  # noqa: E402
from binpick.telemetry import BinTelemetry  # noqa: E402

OUT = ROOT / "out"

# la consola de Windows usa cp1252, que no tiene «→» ni «✔»; hay que fijarlo antes de imprimir ayuda o registro
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_policy(spec: str):
    """``guion`` | ``act:<ruta>`` | ``act+guion:<ruta>`` → (política, modo del orquestador)."""
    if spec in ("guion", "scripted"):
        return POL.ScriptedPolicy(), "guion"
    mode, _, path = spec.partition(":")
    if mode not in ("act", "act+guion") or not path:
        raise SystemExit(f"--policy no válido: {spec!r} (usa guion, act:<carpeta> o act+guion:<carpeta>)")
    return POL.ACTPickPolicy(path), mode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from recoverykit import runtime

    runtime.add_args(ap)
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--open", action="store_true", help="abrir el panel en el navegador")
    ap.add_argument("--seed", type=int, default=C.BinPickCfg.seed)
    ap.add_argument("--pieces", type=int, default=None, help="número de piezas (4–8); por defecto aleatorio 4–6")
    ap.add_argument("--layout", default=None, choices=G.MODES, help="separadas | juntas | apiladas | mixto (por defecto aleatorio)")
    ap.add_argument("--segmenter", default="hsv", help="hsv | yolo:<pesos.pt> | yoloe:<pesos.pt>")
    ap.add_argument("--policy", default="guion", help="guion | act:<carpeta> | act+guion:<carpeta>")
    ap.add_argument("--obs-mode", default=None, choices=sorted(__import__("binpick.observation", fromlist=["x"]).OBS_MODES),
                    help="entradas visuales de ACT (por defecto, las del checkpoint)")
    ap.add_argument("--voice", action="store_true", help="activar el micrófono (Whisper, español)")
    ap.add_argument("--voice-model", default="small", help="tiny | base | small | medium")
    ap.add_argument("--view", action="store_true", help="cámara 3/4 para el panel (un render más por imagen)")
    ap.add_argument("--realtime", type=float, default=1.0, help="1 = tiempo real; 0 = lo más rápido posible")
    ap.add_argument("--text", default=None, help="ejecutar esta orden y salir (sin panel)")
    ap.add_argument("--default-box", default="derecha", choices=C.DESTINATIONS, help="caja si la orden no dice ninguna")
    args = ap.parse_args()

    # --- lo pesado se carga ANTES de arrancar Isaac
    voice = None
    if args.voice:
        from binpick.voice import VoiceService

        voice = VoiceService(args.voice_model)
        why = voice.problem()
        if why:
            print(f"[demo18] sin voz: {why}", flush=True)
            from binpick.voice import TextOnly

            voice = TextOnly(why)
        else:
            voice.load()
    policy, policy_mode = parse_policy(args.policy)
    if args.obs_mode and hasattr(policy, "obs_mode"):
        policy.obs_mode = args.obs_mode

    rt = runtime.boot(args)

    from isaaclab.sim import SimulationContext
    from recoverykit import physics

    from binpick.cell import BinCell
    from binpick.robot_port import SimRobot

    cfg = C.BinPickCfg(seed=args.seed, view_camera=args.view)
    scene = C.build_scene(cfg)
    sim = SimulationContext(physics.sim_cfg(scene))
    bcell = BinCell.build(cfg, sim)
    tele = BinTelemetry()
    tele.set(policy={"kind": policy.kind, "name": policy.name, "mode": policy_mode}, segmenter=args.segmenter,
             voice={"state": "inactiva" if voice and not getattr(voice, "why", None) else "apagada",
                    "detail": "" if voice and not getattr(voice, "why", None) else (getattr(voice, "why", "") or "")})
    robot = SimRobot(bcell, tele, args.segmenter, policy=policy, realtime=0.0 if args.text else args.realtime)
    robot.live_frames = not args.text
    seed = args.seed

    def lay_out(new_seed: int) -> None:
        nonlocal seed
        seed = new_seed
        tele.clear_run()
        tele.set(status="preparando", ready=False)
        spec = G.sample_scene(cfg, seed, n_pieces=args.pieces, mode=args.layout)
        info = robot.new_scene(spec)
        robot.home()
        robot.observe("inicio")
        tele.set(status="esperando", ready=True, capacity={d: robot.capacity(d) for d in C.DESTINATIONS})
        lay = info["layout"]
        tele.event("escena", f"Escena {seed}: {info['n']} piezas ({', '.join(f'{v} {k}' for k, v in info['colors'].items())}); "
                             f"{' · '.join(lay['categories']) or 'sin categorías'}; {lay['touching_pairs']} pares en contacto, "
                             f"{lay['stacked']} apiladas; asentada en {info['settle_s']:.1f} s ({info['attempts']} intento/s)")
        for p in info["problems"]:
            tele.event("escena", f"AVISO: {p}")

    def run_order(text: str, source: str = "texto") -> dict:
        order = commands.parse(text, default_dest=args.default_box)
        tele.clear_stop()
        tele.set(status="ejecutando", heard=text, order=order.as_dict(), report=None)
        tele.event("orden", f"({source}) «{text}» → {order.summary()}")
        task = SortingTask(robot, scene, sink=tele, policy_mode=policy_mode, stop=lambda: tele.stop_requested,
                           clock=lambda: robot.cell.sim_time)
        rep = task.run(order)
        aud = bcell.audit()
        in_boxes = sum(sum(v.values()) for k, v in aud["regions"].items() if k in C.DESTINATIONS)
        agree = in_boxes == rep.moved
        tele.set(status=rep.status, report=rep.as_dict())
        tele.event("auditor", f"Verdad del simulador: {in_boxes} piezas en las cajas · el robot cuenta {rep.moved} "
                              f"({'coincide' if agree else 'NO coincide'})")
        print(f"[demo18] {rep.status} · {rep.moved} movidas, {rep.remaining} sin mover, {rep.recoveries} recuperaciones, "
              f"{rep.duration_s:.0f} s simulados · auditor {'de acuerdo' if agree else 'EN DESACUERDO'}", flush=True)
        OUT.mkdir(exist_ok=True)
        snap = tele.snapshot()
        if args.text:
            for e in snap.get("events", []):
                print(f"  [{e.get('kind', '')}] {e.get('text', e)}", flush=True)
        (OUT / "last_run.json").write_text(json.dumps({"seed": seed, "order": order.as_dict(), "report": rep.as_dict(),
                                                       "audit": aud, "policy": snap["policy"], "events": snap.get("events", [])},
                                                      indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        return {"report": rep, "audit": aud}

    lay_out(seed)

    if args.text:
        out = run_order(args.text, "línea de órdenes")
        rt.close(0 if out["report"].status in ("completado", "nada que hacer") else 2)

    from binpick.server import start

    start(tele, voice, port=args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[demo18] panel en {url}", flush=True)
    if args.open:
        webbrowser.open(url)

    while rt.running():
        try:
            cmd = tele.commands.get_nowait()
        except queue.Empty:
            cmd = None
        if cmd is None:
            robot.idle_tick()
            time.sleep(C.PHYSICS_DT * 0.5)
            continue
        if cmd.kind == "orden":
            run_order(cmd.payload["text"], cmd.payload.get("source", "texto"))
        elif cmd.kind == "reiniciar":
            tele.clear_stop()
            lay_out(seed)
        elif cmd.kind == "aleatorizar":
            tele.clear_stop()
            lay_out(random.randint(1, 10 ** 6))
    rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
