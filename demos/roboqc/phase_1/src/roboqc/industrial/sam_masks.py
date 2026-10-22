"""SAM candidate export utilities. No model loading, downloads or inference."""

import json
from pathlib import Path

import cv2
import numpy as np

from .vision_inference import mask_geometry


def save_candidates(rgb, annotations, output, minimum_area=100, maximum_fraction=.85, seed=42):
    """Preserve all raw masks; export filtered candidates without claiming object labels."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected a uint8 RGB image")
    if minimum_area < 1 or not 0 < maximum_fraction <= 1:
        raise ValueError("Require minimum_area >= 1 and 0 < maximum_fraction <= 1")
    height,width = rgb.shape[:2]
    masks = [np.asarray(a["segmentation"],dtype=bool) for a in annotations]
    if any(m.shape != (height,width) for m in masks):
        raise ValueError("Every mask must match the original image shape")
    out = Path(output)
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out / "raw_masks.npz",masks=np.stack(masks) if masks else np.zeros((0,height,width),bool))
    rows = []
    for index,(annotation,mask) in enumerate(zip(annotations,masks)):
        area = int(mask.sum())
        reason = "below_minimum_area" if area < minimum_area else "above_maximum_fraction" if area > maximum_fraction*height*width else None
        ys,xs = np.nonzero(mask)
        box = [int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)] if area else None
        filename = f"mask_{index:04d}.png" if reason is None else None
        if filename:
            if not cv2.imwrite(str(out / filename),mask.astype(np.uint8)*255):
                raise OSError(f"Could not save mask: {out / filename}")
        rows.append({"id":index,"raw_mask_index":index,"area_px":area,"bbox_xywh":box,
            "model_predicted_iou":float(annotation["predicted_iou"]) if annotation.get("predicted_iou") is not None else None,
            "stability_score":float(annotation["stability_score"]) if annotation.get("stability_score") is not None else None,
            "detector_score":float(annotation["detector_score"]) if annotation.get("detector_score") is not None else None,
            "prompt_label":annotation.get("prompt_label"),
            "kept":reason is None,"filter_reason":reason,"mask_file":filename,
            "category":"unknown","quality":"unknown",**mask_geometry(mask)})
    colors = np.random.default_rng(seed).integers(40,240,size=(len(rows),3))
    overlay = rgb.copy()
    # Smaller nested regions are drawn last, but all original overlapping masks remain separate.
    for row in sorted((r for r in rows if r["kept"]),key=lambda r:r["area_px"],reverse=True):
        mask = masks[row["id"]]
        overlay[mask] = (.55*overlay[mask] + .45*colors[row["id"]]).astype(np.uint8)
    for row in rows:
        if row["kept"]:
            x,y,w,h = row["bbox_xywh"]
            cv2.rectangle(overlay,(x,y),(x+w-1,y+h-1),(0,170,0),1)
            cv2.putText(overlay,str(row["id"]),(x,max(14,y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(180,0,0),1)
    if not cv2.imwrite(str(out / "overlay.png"),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)):
        raise OSError("Could not save overlay")
    result = {"schema_version":"sam-candidates-1.0","image_size":[height,width],
        "raw_masks":len(rows),"kept_candidates":sum(r["kept"] for r in rows),"candidates":rows,
        "filters":{"minimum_area":minimum_area,"maximum_fraction":maximum_fraction},
        "accuracy":"not_measured","warning":"Candidates may overlap or describe subparts/background, not unique physical objects"}
    (out / "candidates.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result,overlay
