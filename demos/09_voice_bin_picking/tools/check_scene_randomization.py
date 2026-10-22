"""¿Genera el simulador escenas aleatorias variadas y físicamente válidas? REQUIERE ISAAC SIM.

    python tools/check_scene_randomization.py --headless --scenes 30

Para cada semilla deja caer las piezas (física), y mide con la verdad del simulador y con la percepción:

* si la escena es válida (todo dentro del contenedor, nada atraviesa el suelo, todo quieto) y en cuántos intentos;
* qué configuración ha salido: separadas, tocándose, apiladas (pares en contacto, piezas con otra debajo), inclinación máxima;
* cuántas piezas ve la percepción, cuántas están tapadas (visibilidad < 85 %) y cuántas son **agarrables** de entrada
  (hueco para los dedos, sin otra encima, alcanzables por IK sin chocar);
* violaciones del vigilante físico del demo 08 (saltos de posición, atravesar la mesa, velocidades imposibles);
* tiempo de asentamiento.

Escribe ``results/scene_randomization.json`` y ``.md``. Es la comprobación que dice si la aleatorización cumple lo pedido
(separadas, tocándose, solapadas, parcialmente apiladas, tapadas). Hasta que se ejecute, no está medida en Isaac.
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
from binpick import commands as K, config as C, scene_gen as G, selection as SEL  # noqa: E402


def main() -> int:
    from recoverykit import runtime

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(ap)
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--layout", default=None, choices=G.MODES)
    ap.add_argument("--pieces", type=int, default=None)
    args = ap.parse_args()

    from binpick.robot_port import open_session

    s = open_session(args, C.BinPickCfg())
    robot = s.robot
    order = K.parse("las piezas a la izquierda")
    rows = []
    for i in range(args.scenes):
        seed = args.first_seed + i
        info = robot.new_scene(G.sample_scene(s.cfg, seed, n_pieces=args.pieces, mode=args.layout))
        robot.home()
        obs = robot.observe("inicio")
        sel = SEL.rank(obs.instances, order, robot.scene)
        reachable = sum(1 for c in sel.usable if robot.reach(c) is None)
        wd = s.bcell.cell.watchdog.summary()
        lay = info["layout"]
        rows.append({"seed": seed, "modo": info["mode"], "piezas": info["n"], "colores": info["colors"], "válida": not info["problems"],
                     "problemas": info["problems"], "intentos": info["attempts"], "asentamiento_s": info["settle_s"],
                     "pares_en_contacto": lay["touching_pairs"], "apiladas": lay["stacked"], "aisladas": lay["isolated"],
                     "inclinación_máx_deg": lay["max_tilt_deg"], "categorías": lay["categories"],
                     "vistas": len(obs.instances), "tapadas": sum(1 for x in obs.instances if x.visibility < 0.85),
                     "agarrables": len(sel.usable), "alcanzables_sin_choque": reachable,
                     "violaciones_físicas": wd["violations"]})
        print(f"[escena {seed}] {rows[-1]['modo']:10s} n={rows[-1]['piezas']} válida={rows[-1]['válida']} "
              f"contacto={rows[-1]['pares_en_contacto']} apiladas={rows[-1]['apiladas']} vistas={rows[-1]['vistas']} "
              f"tapadas={rows[-1]['tapadas']} agarrables={rows[-1]['agarrables']} alcanzables={rows[-1]['alcanzables_sin_choque']} "
              f"violaciones={rows[-1]['violaciones_físicas']}", flush=True)

    n = len(rows)
    summary = {
        "escenas": n, "válidas": sum(r["válida"] for r in rows), "con_contacto": sum(r["pares_en_contacto"] > 0 for r in rows),
        "con_apiladas": sum(r["apiladas"] > 0 for r in rows), "con_piezas_tapadas": sum(r["tapadas"] > 0 for r in rows),
        "modos": dict(Counter(r["modo"] for r in rows)),
        "piezas_agarrables_de_entrada": f"{sum(r['agarrables'] for r in rows)}/{sum(r['vistas'] for r in rows)}",
        "alcanzables_sin_choque": f"{sum(r['alcanzables_sin_choque'] for r in rows)}/{sum(r['vistas'] for r in rows)}",
        "violaciones_físicas_total": sum(r["violaciones_físicas"] for r in rows),
        "asentamiento_s_medio": round(float(np.mean([r["asentamiento_s"] for r in rows])), 2) if rows else None,
        # sólo colores y nº de piezas; las posiciones y giros difieren en todas las semillas por construcción
        "combinaciones_distintas_de_colores_y_numero_de_piezas": len({json.dumps(r["colores"], sort_keys=True) + str(r["piezas"]) for r in rows}),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "scene_randomization.json").write_text(json.dumps({"resumen": summary, "escenas": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Aleatorización de escenas (Isaac Sim)", "", "| medida | valor |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()]
    (out / "scene_randomization.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    s.rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
