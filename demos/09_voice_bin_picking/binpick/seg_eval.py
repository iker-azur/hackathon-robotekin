"""Evaluación de segmentadores (HSV, YOLO, …) sobre el conjunto exportado por ``tools/export_yolo_dataset.py``.

Mismas imágenes y mismas etiquetas para todos, así que la comparación HSV ↔ YOLO ↔ YOLOE es justa. Métricas, por
instancia (una pieza = una máscara):

* recall y precisión con IoU ≥ 0,5, y **recall por visibilidad** (piezas medio tapadas son las difíciles) y por tipo de
  escena (separadas / tocándose / apiladas);
* IoU medio de las emparejadas;
* acierto de **color** y de **forma** entre las emparejadas (el color y la forma se miden en la máscara, no los da la red);
* error de **posición** en el plano (mm) entre la pieza estimada y la real;
* latencia por imagen.

Las etiquetas son analíticas (siluetas proyectadas); un acuerdo perfecto no prueba nada sobre imágenes reales de otra
cámara, sólo sobre este render (``docs/03_limitaciones.md``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import segmentation as S
from recoverykit import perception as P  # noqa: E402

VIS_BINS = ((0.0, 0.5, "<50 %"), (0.5, 0.85, "50–85 %"), (0.85, 1.01, "≥85 %"))


def cfg_from_dict(d: dict) -> C.BinPickCfg:
    cfg = C.BinPickCfg()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def load_context(root: Path) -> tuple[S.Context, dict]:
    cam_j = json.loads((Path(root) / "cam.json").read_text(encoding="utf-8"))
    cam = P.Camera(np.array(cam_j["k"]), np.array(cam_j["rot"]), np.array(cam_j["pos"]), cam_j["width"], cam_j["height"])
    scene = C.build_scene(cfg_from_dict(cam_j.get("config", {})))
    return S.Context(cam, cam_j["table_z"], lambda x, y, z: C.region_of(scene, x, y, z)), cam_j


def iter_samples(root: Path, split: str = "val", limit: int | None = None):
    root = Path(root)
    names = sorted(p.stem for p in (root / "images" / split).glob("*.jpg"))
    for name in names[:limit]:
        bgr = cv2.imread(str(root / "images" / split / f"{name}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(root / "depth" / f"{name}.npz")["depth"].astype(np.float32)
        gt = json.loads((root / "gt" / f"{name}.json").read_text(encoding="utf-8"))
        yield name, rgb, depth, gt


def gt_mask(inst: dict, shape_hw: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape_hw, np.uint8)
    cv2.fillPoly(m, [np.array(inst["polygon"], np.int32)], 1)
    return m.astype(bool)


def match(gt_masks: list[np.ndarray], preds: list, thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Emparejamiento voraz por IoU descendente: ``[(índice gt, índice pred, IoU)]``."""
    pairs = []
    for g, gm in enumerate(gt_masks):
        for p, pi in enumerate(preds):
            iou = S.mask_iou(gm, pi.mask)
            if iou >= thr:
                pairs.append((iou, g, p))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, g, p in pairs:
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        out.append((g, p, iou))
    return out


def evaluate(root: Path, segmenter, split: str = "val", thr: float = 0.5, limit: int | None = None) -> dict:
    tp = fp = fn = 0
    ious, pos_err, times = [], [], []
    color_ok = color_n = shape_ok = shape_n = 0
    by_vis = {lab: [0, 0] for _, _, lab in VIS_BINS}
    by_cat: dict[str, list[int]] = {}
    n_img = 0
    for _name, rgb, depth, gt in iter_samples(root, split, limit):
        n_img += 1
        t0 = time.perf_counter()
        preds = segmenter.segment(rgb, depth)
        times.append(time.perf_counter() - t0)
        gts = gt["instances"]
        masks = [gt_mask(g, rgb.shape[:2]) for g in gts]
        pairs = match(masks, preds, thr)
        matched_g = {g for g, _p, _i in pairs}
        tp += len(pairs)
        fn += len(gts) - len(pairs)
        fp += len(preds) - len(pairs)
        cats = gt.get("layout", {}).get("categories", []) or ["sin_categoria"]
        for gi, g in enumerate(gts):
            hit = gi in matched_g
            for lo, hi, lab in VIS_BINS:
                if lo <= g["visibility"] < hi:
                    by_vis[lab][0] += hit
                    by_vis[lab][1] += 1
            for c in cats:
                by_cat.setdefault(c, [0, 0])
                by_cat[c][0] += hit
                by_cat[c][1] += 1
        for g, p, iou in pairs:
            ious.append(iou)
            color_n += 1
            color_ok += preds[p].color == gts[g]["color"]
            if preds[p].shape not in (None, "?"):
                shape_n += 1
                shape_ok += preds[p].shape == gts[g]["shape"]
            if preds[p].pos is not None:
                pos_err.append(float(np.linalg.norm(np.array(preds[p].pos[:2]) - np.array(gts[g]["pos"][:2]))) * 1000)
    rate = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {
        "segmentador": getattr(segmenter, "name", str(segmenter)), "conjunto": split, "imágenes": n_img, "iou_umbral": thr,
        "instancias_reales": tp + fn, "recall": rate(tp, tp + fn), "precisión": rate(tp, tp + fp),
        "falsos_positivos": fp, "falsos_negativos": fn, "iou_medio": round(float(np.mean(ious)), 3) if ious else None,
        "acierto_de_color": rate(color_ok, color_n), "acierto_de_forma": rate(shape_ok, shape_n),
        "formas_no_decididas": color_n - shape_n,
        "error_de_posición_mm": {"media": round(float(np.mean(pos_err)), 2), "p95": round(float(np.percentile(pos_err, 95)), 2)}
        if pos_err else None,
        "recall_por_visibilidad": {k: {"recall": rate(v[0], v[1]), "n": v[1]} for k, v in by_vis.items()},
        "recall_por_tipo_de_escena": {k: {"recall": rate(v[0], v[1]), "n": v[1]} for k, v in by_cat.items()},
        "latencia_ms": {"media": round(1000 * float(np.mean(times)), 1), "p95": round(1000 * float(np.percentile(times, 95)), 1)}
        if times else None,
    }


def render_markdown(r: dict) -> str:
    L = [f"# Segmentación — {r['segmentador']} (conjunto {r['conjunto']}, {r['imágenes']} imágenes, IoU ≥ {r['iou_umbral']})", "",
         f"- Instancias reales: {r['instancias_reales']} · recall **{r['recall']}** · precisión **{r['precisión']}** "
         f"(FP {r['falsos_positivos']}, FN {r['falsos_negativos']}) · IoU medio {r['iou_medio']}",
         f"- Color acertado: {r['acierto_de_color']} · forma acertada: {r['acierto_de_forma']} "
         f"({r['formas_no_decididas']} marcadas «?» por estar tapadas)",
         f"- Error de posición en el plano (mm): {r['error_de_posición_mm']} · latencia por imagen (ms): {r['latencia_ms']}", "",
         "| visibilidad de la pieza | recall | n |", "|---|---|---|"]
    L += [f"| {k} | {v['recall']} | {v['n']} |" for k, v in r["recall_por_visibilidad"].items()]
    L += ["", "| tipo de escena | recall | n |", "|---|---|---|"]
    L += [f"| {k} | {v['recall']} | {v['n']} |" for k, v in r["recall_por_tipo_de_escena"].items()]
    return "\n".join(L) + "\n"
