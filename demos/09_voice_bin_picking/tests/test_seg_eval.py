"""Evaluación de segmentadores sobre un conjunto exportado (aquí, sintético: mismo formato que ``export_yolo_dataset``)."""

import json

import cv2
import numpy as np
import pytest

from binpick import config as C
from binpick import scene_gen as G
from binpick import seg_eval as E
from binpick import segmentation as S
from synth import render, scene_and_ctx

SCENE, CTX = scene_and_ctx()


def poly_of(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 0.8, True)
    return c[:, 0, :].tolist()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("yolo")
    for sub in ("images/val", "depth", "gt"):
        (root / sub).mkdir(parents=True)
    cam = CTX.cam
    cfg = C.BinPickCfg()
    (root / "cam.json").write_text(json.dumps({"k": cam.k.tolist(), "rot": cam.rot.tolist(), "pos": cam.pos.tolist(),
                                               "width": cam.width, "height": cam.height, "table_z": CTX.table_z,
                                               "config": cfg.to_dict()}), encoding="utf-8")
    n = 0
    for seed in range(6):
        spec = G.sample_scene(cfg, seed, n_pieces=5, mode="mixto")
        placed = G.approximate_settle(spec, SCENE)
        truths = []
        for d in spec.drops:
            pos, quat, _ = placed[d.piece_id]
            pc = SCENE.piece(d.piece_id)
            truths.append(S.PieceTruth(d.piece_id, pc.color, pc.shape, pc.size_m, pc.height, pos, quat))
        rgb, depth = render(SCENE, CTX, truths)
        labels, sils = S.ground_truth(cam, truths)
        gt = []
        for j, t in enumerate(truths):
            m = labels == j + 1
            if m.sum() < 40:
                continue
            gt.append({"id": t.id, "color": t.color, "shape": t.shape, "pos": t.pos.tolist(),
                       "visibility": float(m.sum() / max(sils[t.id].sum(), 1)), "polygon": poly_of(m)})
        cv2.imwrite(str(root / "images/val" / f"s{seed}_v0.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])
        np.savez_compressed(root / "depth" / f"s{seed}_v0.npz", depth=depth.astype(np.float16))
        (root / "gt" / f"s{seed}_v0.json").write_text(json.dumps({"seed": seed, "split": "val", "instances": gt,
                                                                    "layout": G.analyze_layout({t.id: (t.pos, t.quat) for t in truths}).as_dict()}),
                                                      encoding="utf-8")
        n += len(gt)
    return root, n


def test_hsv_on_the_exported_set(dataset):
    root, n = dataset
    ctx, _ = E.load_context(root)
    r = E.evaluate(root, S.HsvSegmenter(ctx))
    assert r["imágenes"] == 6 and r["instancias_reales"] == n
    assert r["precisión"] == 1.0 and r["falsos_positivos"] == 0               # nunca inventa una pieza
    assert r["recall_por_visibilidad"]["≥85 %"]["recall"] == 1.0              # ve todas las que se ven enteras
    assert r["recall"] >= 0.7                                                  # lo que pierde son piezas casi enterradas (<35 % visibles)
    assert r["acierto_de_color"] == 1.0
    assert r["error_de_posición_mm"]["media"] < 4
    assert set(r["recall_por_visibilidad"]) == {"<50 %", "50–85 %", "≥85 %"}
    md = E.render_markdown(r)
    assert "recall" in md and "tipo de escena" in md


def test_a_degraded_segmenter_scores_lower(dataset):
    root, _ = dataset
    ctx, _ = E.load_context(root)

    class Half:
        name = "mitad"

        def __init__(self):
            self.inner = S.HsvSegmenter(ctx)

        def segment(self, rgb, depth):
            return self.inner.segment(rgb, depth)[::2]

    half = E.evaluate(root, Half())
    base = E.evaluate(root, S.HsvSegmenter(ctx))
    assert half["recall"] < base["recall"] - 0.2 and half["falsos_negativos"] > base["falsos_negativos"]


def test_matching_is_one_to_one_and_respects_the_threshold():
    a = np.zeros((40, 40), bool); a[5:20, 5:20] = True
    b = np.zeros((40, 40), bool); b[6:21, 6:21] = True
    far = np.zeros((40, 40), bool); far[30:38, 30:38] = True

    class I:  # noqa: E742
        def __init__(self, m):
            self.mask = m

    pairs = E.match([a, far], [I(b), I(a)])
    assert len(pairs) == 1 and pairs[0][0] == 0 and pairs[0][1] == 1                # a con a (IoU 1), no con b
    assert E.match([a], [I(far)]) == []
