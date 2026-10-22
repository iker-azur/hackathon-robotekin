"""Banco de agarres físicos dentro del contenedor: ¿qué falla, la apertura, la percepción o el modelo de colisión? REQUIERE ISAAC SIM.

    python tools/bench_grasp.py --headless --seeds 12
    python tools/bench_grasp.py --headless --seeds 12 --variants m12 m16 m20 m16_oraculo --layout separadas

Para cada semilla y variante: la misma escena (física), se elige la primera pieza agarrable y alcanzable, y se hace UN agarre con el
guion del experto. Variantes (``--variants``):

* ``m12``, ``m16``, ``m20``: holgura de apertura de la pinza (mm) sobre la anchura de la pieza, con la pose que mide la percepción;
* ``<m>_oraculo``: igual pero con la pose REAL de la pieza (separa el error de percepción del de geometría del agarre);
* ``m16_g3``: además agarra 3 mm más abajo (``grasp_lower_m``; ``_g6`` = 6 mm), contra el deslizamiento al levantar;
* ``dedos_sin_modelo``: las piezas que el modelo descarta por «los dedos no caben» (con el giro de más holgura y sin
  comprobar la pared) se intentan igualmente;
* ``pared_sin_modelo``: las piezas que el modelo de colisión descarta por «chocaría con la pared» se intentan igualmente
  (dice si el modelo es demasiado conservador).

De cada intento guarda estado, error de la percepción (mm y grados) y cuánto empujó la pieza si falló. Escribe
``results/bench_grasp.json``. No sustituye a ``eval_sorting.py``: mide el agarre, no la clasificación.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
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
from binpick import collision, commands as K, config as C, picking, scene_gen as G, selection as SEL  # noqa: E402


def _yaw_of(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _fold(a: float) -> float:
    """Giro de una cara de cubo en [-π/4, π/4)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def _nearest(truths, pos):
    return min(truths, key=lambda t: math.hypot(float(t.pos[0]) - float(pos[0]), float(t.pos[1]) - float(pos[1])))


def main() -> int:
    from recoverykit import runtime

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(ap)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--first-seed", type=int, default=300000)
    ap.add_argument("--layout", default="separadas", choices=G.MODES)
    ap.add_argument("--variants", nargs="+", default=["m12", "m16", "m20", "m16_oraculo", "pared_sin_modelo"])
    ap.add_argument("--shape", default=None, choices=["cube", "cylinder"], help="sólo agarrar piezas de esta forma")
    ap.add_argument("--name", default="bench_grasp")
    args = ap.parse_args()

    from binpick.robot_port import open_session

    s = open_session(args, C.BinPickCfg())
    robot, bcell = s.robot, s.bcell
    order = K.parse("las piezas a la izquierda")
    real_check = collision.check_pick
    rows = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        for var in args.variants:
            oracle = var.endswith("_oraculo")
            nofingers = var == "dedos_sin_modelo"
            nomodel = var == "pared_sin_modelo" or nofingers
            tokens = var.split("_")
            margin = 12 if nomodel else int(tokens[0][1:])
            lower_mm = next((int(t[1:]) for t in tokens[1:] if t[:1] == "g" and t[1:].isdigit()), 0)
            strategy = {"grasp_lower_m": lower_mm / 1000.0} if lower_mm else {}
            picking.OPEN_MARGIN_M = margin / 1000.0
            collision.check_pick = real_check
            robot.new_scene(G.sample_scene(s.cfg, seed, mode=args.layout))
            robot.home()
            obs = robot.observe("banco")
            sel = SEL.rank(obs.instances, order, robot.scene)
            if args.shape:
                sel.usable = [c for c in sel.usable if c.inst.shape == args.shape]
            cand = None
            if nofingers:
                for c in sel.blocked:
                    if c.blocked and "dedos" in c.blocked and c.inst.visibility >= 0.7 and c.grasp.yaws_all and c.grasp.covered_by is None:
                        c2 = dataclasses.replace(c, grasp=dataclasses.replace(c.grasp, yaws_ok=c.grasp.yaws_all[:2]), blocked=None)
                        collision.check_pick = lambda *a, **k: None
                        why = robot.reach(c2)
                        collision.check_pick = real_check
                        if why is None:
                            cand = c2
                            break
            elif nomodel:
                for c in sel.usable:
                    why = robot.reach(c)
                    if why and "pared" in why:
                        collision.check_pick = lambda *a, **k: None
                        if robot.reach(c) is None:
                            cand = c
                        collision.check_pick = real_check
                        break
            else:
                cand = next((c for c in sel.usable if robot.reach(c) is None), None)
            if cand is None:
                rows.append({"seed": seed, "variante": var, "estado": "sin candidata"})
                continue
            truths = bcell.piece_truths()
            t = _nearest(truths, cand.inst.pos)
            err_mm = 1000 * math.hypot(float(t.pos[0] - cand.inst.pos[0]), float(t.pos[1] - cand.inst.pos[1]))
            err_deg = abs(math.degrees(_fold(_yaw_of(t.quat) - cand.inst.yaw))) if t.shape == "cube" else 0.0
            if oracle:
                pos = np.array([t.pos[0], t.pos[1], t.pos[2]], float)
                cand = dataclasses.replace(cand, inst=dataclasses.replace(cand.inst, pos=pos, yaw=_fold(_yaw_of(t.quat))))
                robot._plans.pop(cand.track, None)
                if robot.reach(cand) is not None:
                    rows.append({"seed": seed, "variante": var, "estado": "sin plan con la pose real"})
                    continue
            if nomodel:
                collision.check_pick = lambda *a, **k: None
                robot._plans.pop(cand.track, None)
                robot.reach(cand)
                collision.check_pick = real_check
            p0 = np.array(t.pos, float)
            res = robot.pick(cand, strategy, "guion")
            t2 = _nearest(bcell.piece_truths(), p0)
            moved_mm = 1000 * float(np.linalg.norm(np.array(t2.pos, float)[:2] - p0[:2])) if t2.id == t.id else None
            rows.append({"seed": seed, "variante": var, "pieza": t.id, "z_mm": round(float(t.pos[2]) * 1000, 1),
                         "estado": "ok" if res.ok else res.status, "detalle": res.detail, "err_pos_mm": round(err_mm, 1),
                         "err_yaw_deg": round(err_deg, 1), "desplazada_mm": None if res.ok or moved_mm is None else round(moved_mm, 1)})
            r = rows[-1]
            print(f"[{seed} {var:17s}] {t.id:9s} {r['estado']:9s} err={err_mm:4.1f} mm / {err_deg:4.1f}° "
                  f"{'empujó %.0f mm' % moved_mm if (not res.ok and moved_mm is not None) else ''} {res.detail}", flush=True)
            robot.home()

    summary = {}
    for var in args.variants:
        v = [r for r in rows if r["variante"] == var and "err_pos_mm" in r]
        summary[var] = {"n": len(v), "ok": sum(r["estado"] == "ok" for r in v),
                        "estados": dict(Counter(r["estado"] for r in v)),
                        "sin_candidata": sum(1 for r in rows if r["variante"] == var and "err_pos_mm" not in r),
                        "err_pos_mm_medio": round(float(np.mean([r["err_pos_mm"] for r in v])), 2) if v else None,
                        "err_yaw_deg_medio": round(float(np.mean([r["err_yaw_deg"] for r in v])), 2) if v else None,
                        "desplazamiento_mm_medio_si_falla": round(float(np.mean([r["desplazada_mm"] for r in v if r.get("desplazada_mm") is not None] or [0])), 1)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"variantes": summary, "intentos": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    s.rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
