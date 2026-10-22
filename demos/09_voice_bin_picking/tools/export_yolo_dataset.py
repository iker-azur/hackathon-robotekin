"""Exporta imágenes de la cámara cenital con máscaras de instancia reales para ajustar YOLO-seg. REQUIERE ISAAC SIM.

    python tools/export_yolo_dataset.py --headless --scenes 400            # ≈ 1 200 imágenes con --views 3
    python tools/export_yolo_dataset.py --headless --scenes 20 --views 1 --out yolo_humo --debug 6

Por escena (piezas que caen y se asientan por física: separadas, juntas, apiladas, a distinta altura) se toman varias
vistas: la escena sola y, en parte de ellas, con el brazo colocado sobre una pieza (para que la red aprenda a ignorar el
brazo). Las **etiquetas son analíticas**: la silueta de cada pieza proyectada con la calibración de la cámara, con las
más altas tapando a las bajas y el brazo descontado. No sustituyen a las anotaciones de un renderizador: con
``--debug N`` se guardan N imágenes con el contorno dibujado para comprobar a ojo que cuadra con el render.

Formato (ultralytics, una sola clase «pieza»; el color y la forma NO los decide la red, se miden en la máscara):

    data/<out>/images/{train,val}/*.jpg   RGB
    data/<out>/labels/{train,val}/*.txt   «0 x1 y1 x2 y2 …» normalizado, un polígono por pieza visible
    data/<out>/depth/*.npz                profundidad float16 (para evaluar el segmentador HSV con las mismas imágenes)
    data/<out>/gt/*.json                  por pieza: id, color, forma, posición, giro, visibilidad, polígono
    data/<out>/data.yaml, cam.json        configuración de ultralytics y calibración de la cámara

Separación por escena (val = una de cada ``--val-every``): ninguna escena aparece en los dos conjuntos.
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
from binpick import commands as K, config as C, scene_gen as G, segmentation as S, selection as SEL  # noqa: E402

MIN_AREA_PX = 40
MIN_VISIBILITY = 0.15


def polygon(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.8, True)
    return c[:, 0, :] if len(c) >= 3 else None


def main() -> int:
    from recoverykit import runtime

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(ap)
    ap.add_argument("--scenes", type=int, default=400)
    ap.add_argument("--views", type=int, default=3, help="vistas por escena (la primera, sin brazo)")
    ap.add_argument("--arm-fraction", type=float, default=0.5, help="fracción de las vistas extra con el brazo a la vista")
    ap.add_argument("--first-seed", type=int, default=100000, help="lejos de las semillas de las demostraciones")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--out", default="yolo_pieces")
    ap.add_argument("--debug", type=int, default=6, help="imágenes con el contorno dibujado para comprobar la alineación")
    args = ap.parse_args()

    out = ROOT / "data" / args.out
    if (out / "data.yaml").exists():
        print(f"ERROR: {out} ya existe; elige otro --out")
        return 1
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "depth", "gt", "debug"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    from binpick.robot_port import arm_pixels, open_session

    s = open_session(args, C.BinPickCfg())
    robot, bcell, cam = s.robot, s.bcell, s.robot.cam
    rng = np.random.default_rng(args.first_seed)
    n_img = n_inst = dbg = 0
    for i in range(args.scenes):
        seed = args.first_seed + i
        info = robot.new_scene(G.sample_scene(s.cfg, seed))
        if info["problems"]:
            continue
        split = "val" if i % args.val_every == 0 else "train"
        for v in range(args.views):
            arm_on = v > 0 and rng.random() < args.arm_fraction
            robot.home()
            if arm_on:
                obs = robot.observe("etiquetado")
                sel = SEL.rank(obs.instances, K.parse("las piezas a la izquierda"), robot.scene)
                rng.shuffle(sel.usable)
                for c in sel.usable:
                    if robot.reach(c) is None:
                        robot.pose_arm(robot._plans[c.track][0].q_pre)
                        break
            rgb, depth = robot._grab_settled()
            hide = arm_pixels(robot) > 0 if arm_on else np.zeros(rgb.shape[:2], bool)
            truths = bcell.piece_truths()
            labels, sils = S.ground_truth(cam, truths)
            name = f"s{seed}_v{v}"
            lines, gt = [], []
            for j, t in enumerate(truths):
                m = (labels == j + 1) & ~hide
                full = sils[t.id] & ~hide
                area = int(m.sum())
                vis = area / max(int(full.sum()), 1)
                if area < MIN_AREA_PX or vis < MIN_VISIBILITY:
                    continue
                poly = polygon(m)
                if poly is None:
                    continue
                h, w = m.shape
                lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly))
                gt.append({"id": t.id, "color": t.color, "shape": t.shape, "pos": [round(float(x), 5) for x in t.pos],
                           "visibility": round(vis, 3), "area": area, "polygon": poly.tolist(),
                           "region": C.region_of(robot.scene, float(t.pos[0]), float(t.pos[1]), float(t.pos[2]))})
            if not lines:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            np.savez_compressed(out / "depth" / f"{name}.npz", depth=depth.astype(np.float16))
            (out / "gt" / f"{name}.json").write_text(json.dumps({"seed": seed, "view": v, "split": split, "arm": bool(arm_on),
                                                                 "layout": info["layout"], "instances": gt}, ensure_ascii=False),
                                                     encoding="utf-8")
            n_img += 1
            n_inst += len(lines)
            if dbg < args.debug:
                dbg += 1
                img = bgr.copy()
                for g in gt:
                    cv2.polylines(img, [np.array(g["polygon"], np.int32)], True, (0, 255, 255), 1)
                cv2.imwrite(str(out / "debug" / f"{name}.png"), img)
        if (i + 1) % 20 == 0:
            print(f"[yolo] {i + 1}/{args.scenes} escenas · {n_img} imágenes · {n_inst} instancias", flush=True)

    cal = robot.cell.camera_calibration(C.CAMERA)
    (out / "cam.json").write_text(json.dumps({"k": cal["k"].tolist(), "rot": cal["rot"].tolist(), "pos": cal["pos"].tolist(),
                                              "width": cal["width"], "height": cal["height"], "table_z": robot.top,
                                              "config": s.cfg.to_dict()}, indent=1), encoding="utf-8")
    (out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: pieza\n", encoding="utf-8")
    print(f"[yolo] listo: {n_img} imágenes, {n_inst} instancias en {out}", flush=True)
    s.rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
