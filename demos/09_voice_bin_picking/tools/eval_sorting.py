"""Evalúa la clasificación COMPLETA en Isaac con varias semillas: voz/texto → visión → política → verificación. REQUIERE ISAAC SIM.

    python tools/eval_sorting.py --headless --seeds 20                                   # guion del experto (NO aprendido)
    python tools/eval_sorting.py --headless --seeds 20 --policy act:outputs/act_v1/pretrained_model
    python tools/eval_sorting.py --headless --seeds 20 --policy act+guion:outputs/act_v1/pretrained_model --segmenter yolo:runs/piezas_seg.pt

Para cada semilla: escena nueva (física), una de las órdenes de ``--orders`` (por defecto, las cuatro del enunciado, rotando),
ejecución con la máquina de estados y **verdad del simulador** al final. El éxito se juzga con la verdad, no con lo que cree el
robot: ninguna pieza en una caja que no le toca y ninguna pieza que la orden pida moviendo aún por mover. Se cuenta aparte
el acuerdo entre lo que cree el robot y lo que hay de verdad.

El informe etiqueta SIEMPRE qué política hizo cada agarre (``act`` = aprendida, ``guion`` = experto programado) y la tasa de
agarre de cada una. Una ejecución con ``--policy guion`` evalúa la tubería, **no** el aprendizaje por imitación.
Escribe ``results/eval_sorting_<nombre>.json`` y ``.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import binpick  # noqa: E402,F401
from binpick import commands as K, config as C, policy as POL, scene_gen as G  # noqa: E402
from binpick.orchestrator import SortingTask  # noqa: E402

ORDERS = [
    "Coloca las piezas rojas en la caja izquierda.",
    "Pon las piezas azules en la caja derecha.",
    "Clasifica las piezas rojas a la izquierda y las verdes a la derecha.",
    "Recoge todas las piezas azules.",
]


_COLOR_WORDS = {"red": ("rojas", "rojos", "roja", "rojo"), "green": ("verdes", "verde"), "blue": ("azules", "azul")}
_PLURAL = {"red": "rojas", "green": "verdes", "blue": "azules"}


def adapt_order(text: str, colors: dict[str, int]) -> str:
    """Cambia los colores que pide la orden por colores que HAY en la escena.

    Con colores sorteados, una orden como «las azules a la derecha» cae en «nada que hacer» en una de cada tres escenas y no
    mide nada. Los colores que sí hay se dejan; los que faltan se sustituyen por los presentes que sobran (más numerosos primero).
    """
    import re

    present = [c for c, _ in sorted(colors.items(), key=lambda kv: (-kv[1], kv[0]))]
    pattern = r"\b(" + "|".join(w for ws in _COLOR_WORDS.values() for w in ws) + r")\b"
    asked: list[str] = []
    for m in re.finditer(pattern, text):
        c = next(k for k, ws in _COLOR_WORDS.items() if m.group(1) in ws)
        if c not in asked:
            asked.append(c)
    if not present or all(c in colors for c in asked):
        return text
    free = [c for c in present if c not in asked]
    mapping = {c: c if c in colors else (free.pop(0) if free else present[0]) for c in asked}
    return re.sub(pattern, lambda m: _PLURAL[mapping[next(k for k, ws in _COLOR_WORDS.items() if m.group(1) in ws)]], text)


def truth_check(order: K.SortOrder, where: dict[str, str], scene) -> dict:
    """Juicio con la verdad del simulador: piezas en caja equivocada y piezas que aún tocan mover."""
    info = {pid: (scene.piece(pid).color, scene.piece(pid).shape) for pid in where}
    rules = order.rules
    wrong = 0
    for pid, region in where.items():
        if region in C.DESTINATIONS:
            c, sh = info[pid]
            wrong += not any(r.dest == region and (r.selector.rest or r.selector.matches(c, sh)) for r in rules)
    pending = 0
    for r in rules:
        s = r.selector
        if s.rest:
            claimed = {pid for rr in rules if not rr.selector.rest for pid, (c, sh) in info.items() if rr.selector.matches(c, sh)}
            matching = [pid for pid in info if pid not in claimed]
        else:
            matching = [pid for pid, (c, sh) in info.items() if s.matches(c, sh)]
        in_dest = [p for p in matching if where[p] == r.dest]
        movable = [p for p in matching if where[p] in ("origen", "mesa")]
        need = len(matching) if s.count == "all" else min(int(s.count), len(matching))
        if movable:
            pending += max(0, need - len(in_dest))
    return {"wrong": wrong, "pending": pending, "ok": wrong == 0 and pending == 0}


def main() -> int:
    from recoverykit import runtime

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(ap)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=200000)
    ap.add_argument("--policy", default="guion")
    ap.add_argument("--segmenter", default="hsv")
    ap.add_argument("--orders", nargs="*", default=ORDERS)
    ap.add_argument("--keep-orders", action="store_true", help="no cambiar los colores de la orden por los que hay en la escena")
    ap.add_argument("--pieces", type=int, default=None)
    ap.add_argument("--layout", default=None, choices=G.MODES)
    ap.add_argument("--name", default=None)
    ap.add_argument("--default-box", default="derecha", choices=C.DESTINATIONS,
                    help="caja para las órdenes que no dicen ninguna (la 4.ª del enunciado); el robot en sí no adivina")
    args = ap.parse_args()

    if args.policy in ("guion", "scripted"):
        policy, mode = POL.ScriptedPolicy(), "guion"
    else:
        mode, _, path = args.policy.partition(":")
        if mode not in ("act", "act+guion") or not path:
            print("--policy: guion | act:<carpeta> | act+guion:<carpeta>")
            return 2
        policy = POL.ACTPickPolicy(path)

    from binpick.robot_port import open_session

    s = open_session(args, C.BinPickCfg(), segmenter=args.segmenter, policy=policy)
    robot, bcell = s.robot, s.bcell
    rows = []
    for i in range(args.seeds):
        seed = args.first_seed + i
        asked = args.orders[i % len(args.orders)]
        info = robot.new_scene(G.sample_scene(s.cfg, seed, n_pieces=args.pieces, mode=args.layout))
        text = asked if args.keep_orders else adapt_order(asked, info["colors"])
        robot.home()
        robot.observe("inicio")
        order = K.parse(text, default_dest=args.default_box)
        s.tele.clear_run()
        task = SortingTask(robot, s.scene, sink=s.tele, policy_mode=mode, clock=lambda: robot.cell.sim_time)
        rep = task.run(order)
        events = [f"[{e.get('kind', '')}] {e.get('text', '')}" for e in s.tele.snapshot().get("events", [])]
        aud = bcell.audit()
        tc = truth_check(order, aud["where"], s.scene)
        in_boxes = sum(sum(v.values()) for k, v in aud["regions"].items() if k in C.DESTINATIONS)
        rows.append({"seed": seed, "orden": text, "orden_pedida": asked, "escena": {"modo": info["mode"], "piezas": info["n"], "colores": info["colors"],
                                                              "contacto": info["layout"]["touching_pairs"], "apiladas": info["layout"]["stacked"]},
                     "estado": rep.status, "movidas": rep.moved, "sin_mover": rep.remaining, "recuperaciones": rep.recoveries,
                     "fallos": rep.failures, "agarres": rep.picks, "agarres_ok": rep.picks_ok, "t_sim_s": round(rep.duration_s, 1),
                     "verdad": tc,
                     # el robot «cree que acabó» si dice completado, o si no había nada que hacer
                     "el_robot_acierta_al_creerlo": (rep.status in ("completado", "nada que hacer")) == tc["ok"] and in_boxes == rep.moved,
                     "piezas_en_cajas_de_verdad": in_boxes, "notas": rep.notes[:3], "eventos": events})
        r = rows[-1]
        print(f"[{i + 1:3d}/{args.seeds}] semilla {seed} «{text[:46]}» → {rep.status}; verdad {'OK' if tc['ok'] else 'NO'} "
              f"(equivocadas {tc['wrong']}, pendientes {tc['pending']}); {rep.moved} movidas, {rep.recoveries} recuperaciones, "
              f"{rep.duration_s:.0f} s", flush=True)

    n = len(rows)
    picks = {k: sum(r["agarres"].get(k, 0) for r in rows) for k in ("act", "guion")}
    picks_ok = {k: sum(r["agarres_ok"].get(k, 0) for r in rows) for k in ("act", "guion")}
    summary = {
        "política": policy.name, "tipo": "APRENDIDA (ACT)" if policy.kind == "act" else "GUION programado (NO aprendizaje)", "modo": mode,
        "segmentador": args.segmenter, "ejecuciones": n,
        "éxito_según_la_verdad_del_simulador": f"{sum(r['verdad']['ok'] for r in rows)}/{n}",
        "estado_del_robot": dict(Counter(r["estado"] for r in rows)),
        "agarres_act": f"{picks_ok['act']}/{picks['act']}", "agarres_guion": f"{picks_ok['guion']}/{picks['guion']}",
        "piezas_en_caja_equivocada": sum(r["verdad"]["wrong"] for r in rows),
        "el_robot_acierta_al_creerlo": f"{sum(r['el_robot_acierta_al_creerlo'] for r in rows)}/{n}",
        "recuperaciones_totales": sum(r["recuperaciones"] for r in rows),
        "t_sim_medio_s": round(float(np.mean([r["t_sim_s"] for r in rows])), 1) if rows else None,
    }
    name = args.name or f"{mode}_{args.segmenter.split(':')[0]}"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"eval_sorting_{name}.json").write_text(json.dumps({"resumen": summary, "ejecuciones": rows}, indent=1, ensure_ascii=False, default=str),
                                                    encoding="utf-8")
    md = [f"# Clasificación completa en Isaac — {summary['tipo']}", "", "| medida | valor |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()]
    (out / f"eval_sorting_{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    s.rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
