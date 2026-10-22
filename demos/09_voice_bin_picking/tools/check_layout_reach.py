"""¿Llega el SO-101 a todo lo que la escena le pide? (sin Isaac)

    python tools/check_layout_reach.py
    python tools/check_layout_reach.py --bin-x 0.21 --left-y 0.15

Comprueba con la cinemática inversa del demo 08 (la misma que usa el robot):

* agarre vertical de una pieza en una rejilla dentro del contenedor, en el suelo
  y apilada (segunda capa), con cualquiera de las orientaciones de la pinza;
* colocación en una rejilla dentro de cada caja de destino;
* que el brazo, plegado en su postura de observación, no tape contenedor ni cajas.

Escribe ``results/layout_reach.json``. Es un cálculo geométrico: no dice nada
de colisiones con las paredes del contenedor (eso lo miran los candidatos de
``selection.py``) ni de la física.
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

from binpick import config as C, picking  # noqa: E402
from recoverykit import kinematics as K, skills  # noqa: E402
from recoverykit.workcell import base_transform  # noqa: E402


def grid(box, step):
    x0, x1, y0, y1 = box
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    return [(float(x), float(y)) for x in xs for y in ys]


def pick_ok(base, table_z, floor_z, wall_top_z, xy, layer, q_seed):
    h = C.PIECE_SIZE_M
    center_z = floor_z + layer * h + h / 2
    why = ""
    for yaw in (0.0, math.radians(22.5), math.radians(45.0), math.radians(67.5)):
        t = skills.Target(xy[0], xy[1], yaw, h, h, "cube")
        p = picking.plan_pick_ex(base, table_z, t, center_z, wall_top_z, q_seed, floor_z=floor_z)
        if not isinstance(p, str):
            return True, ""
        why = p
    return False, why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-x", type=float, default=None)
    ap.add_argument("--bin-y", type=float, default=None)
    ap.add_argument("--left-y", type=float, default=None)
    ap.add_argument("--box-x", type=float, default=None)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--out", default=str(ROOT / "results" / "layout_reach.json"))
    args = ap.parse_args()

    cfg = C.BinPickCfg()
    if args.bin_x is not None or args.bin_y is not None:
        cfg.bin_center_xy_m = (args.bin_x if args.bin_x is not None else cfg.bin_center_xy_m[0],
                               args.bin_y if args.bin_y is not None else cfg.bin_center_xy_m[1])
    if args.left_y is not None or args.box_x is not None:
        bx = args.box_x if args.box_x is not None else cfg.left_center_xy_m[0]
        ly = args.left_y if args.left_y is not None else cfg.left_center_xy_m[1]
        cfg.left_center_xy_m, cfg.right_center_xy_m = (bx, ly), (bx, -ly)
    scene = C.build_scene(cfg)
    base = base_transform(scene)
    top = scene.table.top_z_m
    seed = np.array(scene.robot.start_joints[:5])
    out: dict = {"config": cfg.to_dict()}

    # --- contenedor de origen
    b = C.container(scene, C.BIN_ID)
    box = C.inner_margin_box(scene, C.BIN_ID, 0.016)
    rows = {}
    for layer in (0, 1):
        pts = grid(box, args.step)
        ok = 0
        bad = []
        for xy in pts:
            good, why = pick_ok(base, top, top + b.floor_m, top + b.outer_m[2], xy, layer, seed)
            ok += good
            if not good:
                bad.append({"xy": [round(xy[0], 3), round(xy[1], 3)],
                            "polar": [round(v, 2) for v in C.polar(*xy)], "why": why})
        rows[f"capa_{layer}"] = {"alcanzables": ok, "total": len(pts), "fraccion": round(ok / len(pts), 3),
                                 "fallos": bad[:8]}
    out["contenedor_origen"] = rows
    out["contenedor_polar_centro"] = [round(v, 3) for v in C.polar(*cfg.bin_center_xy_m)]

    # --- cajas de destino: colocación
    for name, cid in C.DEST_IDS.items():
        c = C.container(scene, cid)
        pts = grid(C.inner_margin_box(scene, cid, 0.02), args.step)
        reach = []
        for xy in pts:
            plan = skills.plan_place(base, top, xy, top + c.floor_m, C.PIECE_SIZE_M, K.tcp_local(C.PIECE_SIZE_M),
                                     skills._yaw_candidates(0.0, "cube"), seed)
            if not isinstance(plan, str):
                reach.append(xy)
        # rejilla 2×2 de huecos (la misma que usa place.py): ¿cuántos son alcanzables?
        cx, cy = c.center_xy_m
        slots = [(cx + sx * c.inner_m[0] / 4, cy + sy * c.inner_m[1] / 4) for sx in (-1, 1) for sy in (-1, 1)]
        slot_ok = []
        for xy in slots:
            plan = skills.plan_place(base, top, xy, top + c.floor_m, C.PIECE_SIZE_M, K.tcp_local(C.PIECE_SIZE_M),
                                     skills._yaw_candidates(0.0, "cube"), seed)
            slot_ok.append(not isinstance(plan, str))
        out[f"caja_{name}"] = {"alcanzables": len(reach), "total": len(pts),
                               "fraccion": round(len(reach) / max(len(pts), 1), 3),
                               "huecos_2x2_alcanzables": int(sum(slot_ok)), "huecos_2x2": slot_ok,
                               "polar_centro": [round(v, 2) for v in C.polar(cx, cy)]}

    # --- el brazo plegado no tapa nada
    poses = K.fk(dict(zip(K.ARM, scene.robot.start_joints[:5])), base)
    parts = []
    for link in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"):
        v = K.colliders()[link][::3]
        t = poses[link]
        parts.append(v @ t[:3, :3].T + t[:3, 3])
    pts = np.concatenate(parts)
    arm = {"x": [round(float(pts[:, 0].min()), 3), round(float(pts[:, 0].max()), 3)],
           "y": [round(float(pts[:, 1].min()), 3), round(float(pts[:, 1].max()), 3)],
           "z_max": round(float(pts[:, 2].max()) - top, 3), "solapa": {}}
    for c in scene.containers:
        inside = (np.abs(pts[:, 0] - c.center_xy_m[0]) < c.outer_m[0] / 2) & \
                 (np.abs(pts[:, 1] - c.center_xy_m[1]) < c.outer_m[1] / 2)
        arm["solapa"][c.id] = int(inside.sum())
    in_roi = (pts[:, 0] > cfg.roi_x_m[0]) & (pts[:, 0] < cfg.roi_x_m[1]) & (np.abs(pts[:, 1]) < cfg.roi_y_m[1])
    arm["puntos_dentro_del_ROI"] = int(in_roi.sum())
    out["brazo_plegado"] = arm

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
