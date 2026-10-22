"""Evalúa un segmentador sobre el conjunto exportado por ``export_yolo_dataset.py``. NO necesita Isaac.

    python tools/eval_segmentation.py --data data/yolo_pieces --segmenter hsv
    python tools/eval_segmentation.py --data data/yolo_pieces --segmenter yolo:runs/piezas_seg.pt     # tras entrenar en Colab
    python tools/eval_segmentation.py --data data/yolo_pieces --segmenter yoloe:yoloe-11s-seg.pt      # sin entrenar, por texto

Escribe ``results/segmentation_<segmentador>.json`` y ``.md``. Mismas imágenes para todos: la comparación es justa.
Lee: ``<data>/images/<split>/*.jpg``, ``depth/*.npz``, ``gt/*.json`` (``instances[{id,color,shape,pos,visibility,polygon}]``)
y ``cam.json`` (calibración). Para YOLO hace falta ``pip install ultralytics --no-deps`` (ver README).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import binpick  # noqa: E402,F401
from binpick import seg_eval as E, segmentation as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--segmenter", default="hsv")
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = Path(args.data)
    ctx, _ = E.load_context(root)
    seg = S.make_segmenter(args.segmenter, ctx)
    result = E.evaluate(root, seg, args.split, args.iou, args.limit)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.segmenter)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"segmentation_{tag}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    md = E.render_markdown(result)
    (out / f"segmentation_{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
