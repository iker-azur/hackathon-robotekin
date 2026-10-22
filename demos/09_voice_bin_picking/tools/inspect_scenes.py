"""Diagnóstico de escenas en Isaac: ¿qué hay de verdad, qué ve la percepción y por qué no se alcanza cada pieza? REQUIERE ISAAC SIM.

    python tools/inspect_scenes.py --headless --seeds 5 24 26 29
    python tools/inspect_scenes.py --headless --seeds 0 1 2 --layout apiladas

Por semilla: deja caer la escena, observa y empareja cada pieza REAL (proyección analítica de su silueta) con la instancia
segmentada de mayor IoU. Para cada pieza dice cuánto se ve de verdad, si la percepción la encontró, y si no es candidata o no
se alcanza, el motivo. Guarda ``out/inspect/s<semilla>_cam.png`` y ``_seg.png`` y ``out/inspect/report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import binpick  # noqa: E402,F401
from binpick import commands as K, config as C, scene_gen as G, segmentation as S, selection as SEL, viz  # noqa: E402


def main() -> int:
    from recoverykit import runtime

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(ap)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layout", default=None, choices=G.MODES)
    ap.add_argument("--pieces", type=int, default=None)
    args = ap.parse_args()

    from binpick.robot_port import open_session

    s = open_session(args, C.BinPickCfg())
    robot, bcell, cam = s.robot, s.bcell, s.robot.cam
    out = ROOT / "out" / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    order = K.parse("las piezas a la izquierda")
    report = []
    for seed in args.seeds:
        info = robot.new_scene(G.sample_scene(s.cfg, seed, n_pieces=args.pieces, mode=args.layout))
        robot.home()
        obs = robot.observe("inspección")
        rgb, _ = obs.frame
        sel = SEL.rank(obs.instances, order, robot.scene)
        truths = bcell.piece_truths()
        labels, sils = S.ground_truth(cam, truths)
        reasons = {}
        for c in sel.blocked:
            reasons[id(c.inst)] = f"bloqueada: {c.blocked}"
        for c in sel.usable:
            why = robot.reach(c)
            if why:
                reasons[id(c.inst)] = f"no alcanzable: {why}"
        rows = []
        for j, t in enumerate(truths):
            vis_mask = labels == j + 1
            full = max(int(sils[t.id].sum()), 1)
            vis = float(vis_mask.sum()) / full
            best, iou = None, 0.0
            for i in obs.instances:
                v = S.mask_iou(vis_mask, i.mask)
                if v > iou:
                    best, iou = i, v
            seen = best is not None and iou >= 0.3
            region = C.region_of(robot.scene, float(t.pos[0]), float(t.pos[1]), float(t.pos[2]))
            motivo = None
            if not seen:
                motivo = "no vista" + (" (casi tapada)" if vis < 0.35 else " — fallo de segmentación")
            else:
                motivo = reasons.get(id(best))
            rows.append({"id": t.id, "color": t.color, "shape": t.shape, "z_mm": round(float(t.pos[2]) * 1000, 1),
                         "visibilidad_real": round(vis, 2), "vista": bool(seen), "iou": round(iou, 2), "region": region,
                         "motivo": motivo})
            print(f"  {t.id:10s} {t.color:5s} {t.shape:8s} z={rows[-1]['z_mm']:5.1f}mm vis={vis:4.2f} "
                  f"{'vista iou=%.2f' % iou if seen else 'NO VISTA   '} {region:8s} {motivo or 'ok'}", flush=True)
        print(f"[escena {seed}] {info['mode']} n={info['n']} válida={not info['problems']} vistas={len(obs.instances)}", flush=True)
        cv2.imwrite(str(out / f"s{seed}_cam.png"), viz.annotate(rgb, obs.instances, robot.scene, cam, "rgb", rotate=False))
        cv2.imwrite(str(out / f"s{seed}_seg.png"), viz.annotate(rgb, obs.instances, robot.scene, cam, "seg", rotate=False))
        report.append({"seed": seed, "modo": info["mode"], "piezas": info["n"], "problemas": info["problems"], "filas": rows})
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    s.rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
