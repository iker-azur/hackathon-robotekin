"""CPU segmentation, planar geometry and optional visible-quality classifier.

This module intentionally imports no GUI or robot library and is embedded in Colab.
"""

import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["bolt", "nut", "washer", "bracket"]
QUALITY = ["good", "defect_1", "defect_2"]


def mask_geometry(mask, symmetric=False, clearance_px=3):
    mask = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return {"center_px": None, "orientation_deg": None, "grasp_candidates": []}
    center = [float(xs.mean()), float(ys.mean())]
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(np.column_stack([xs, ys]), rowvar=False))
    orientation = None
    if not symmetric and eigenvalues[-1] > 1.3 * max(eigenvalues[0], 1e-9):
        axis = eigenvectors[:, -1]
        orientation = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, point = cv2.minMaxLoc(distance)
    candidates = []
    if radius >= clearance_px:
        candidates.append({"position_px": list(point), "clearance_px": float(radius),
                           "orientation_deg": orientation, "validated": False})
    return {"center_px": center, "orientation_deg": orientation, "grasp_candidates": candidates}


def classify_mask_color(frame, mask, minimum_fraction=.6):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Erode edges to reduce background contamination; classify only saturated pixels.
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    values = hsv[interior]
    if not len(values):
        return None, 0.0
    ranges = {"red": ((values[:, 0] <= 8) | (values[:, 0] >= 170)),
              "green": (values[:, 0] >= 35) & (values[:, 0] <= 85),
              "blue": (values[:, 0] >= 86) & (values[:, 0] <= 130)}
    votes = {key: float(np.mean(region & (values[:, 1] >= 70) & (values[:, 2] >= 45))) for key, region in ranges.items()}
    color = max(votes, key=votes.get)
    confidence = votes[color]
    return (color if confidence >= minimum_fraction else None), confidence


def quality_crop(frame, mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty instance mask")
    patch = np.where(mask[..., None] > 0, frame, 127)[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.cvtColor(cv2.resize(patch, (224, 224)), cv2.COLOR_BGR2RGB)


class Segmenter:
    def __init__(self, bundle, prefer_onnx=True, confidence=.35):
        from ultralytics import YOLO
        self.root = Path(bundle)
        self.metadata = json.loads((self.root / "vision_manifest.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != "industrial-vision-1.0":
            raise ValueError("Unsupported vision artifact schema")
        self.classes = self.metadata["classes"]
        filename = "segmentation.onnx" if prefer_onnx and (self.root / "segmentation.onnx").exists() else "segmentation.pt"
        if filename.endswith(".pt") and self.metadata.get("model_family") == "yoloe":
            from ultralytics import YOLOE
            self.model = YOLOE(str(self.root / filename), task="segment")
        else:
            self.model = YOLO(str(self.root / filename), task="segment")
        self.confidence = confidence
        self.quality_session = None
        if self.metadata.get("quality_enabled"):
            import onnxruntime as ort
            self.quality_session = ort.InferenceSession(str(self.root / "quality.onnx"), providers=["CPUExecutionProvider"])

    def predict(self, frame):
        # PT otherwise uses minimal rectangular padding, unlike static square ONNX.
        # YOLOE also defaults to agnostic NMS while YOLO(onnx) does not.
        prediction = self.model.predict(frame, device="cpu", imgsz=self.metadata.get("imgsz", 640),
            conf=self.confidence, rect=False, agnostic_nms=self.metadata.get("model_family") == "yoloe",
            iou=.7, max_det=300, retina_masks=False, augment=False, verbose=False)[0]
        objects = []
        if prediction.masks is None:
            return objects
        masks = prediction.masks.data.cpu().numpy()
        height, width = frame.shape[:2]
        for i, (box, cls, confidence) in enumerate(zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.cls.cpu().numpy(), prediction.boxes.conf.cpu().numpy())):
            # Ultralytics masks may be in letterboxed coordinates, not frame coordinates.
            from ultralytics.utils.ops import scale_masks
            import torch
            restored = scale_masks(torch.as_tensor(masks[i:i+1, None],dtype=torch.float32), (height, width))[0, 0].numpy()
            mask = (restored > .5).astype(np.uint8)
            if not mask.any():
                continue
            x1, y1, x2, y2 = box.tolist()
            if int(cls) != cls or not 0 <= int(cls) < len(self.classes):
                raise ValueError("Prediction class ID does not match the exported class mapping")
            object_type = self.classes[int(cls)]
            color, color_confidence = classify_mask_color(frame, mask)
            geometry = mask_geometry(mask, symmetric=object_type in ("washer", "nut"))
            if geometry["center_px"] is None:
                continue
            quality, quality_confidence = "unknown", 0.0
            if self.quality_session is not None:
                crop = quality_crop(frame, mask).astype(np.float32) / 255
                x = ((crop - [.485, .456, .406]) / [.229, .224, .225]).transpose(2, 0, 1)[None].astype(np.float32)
                logits = self.quality_session.run(None, {self.quality_session.get_inputs()[0].name: x})[0][0]
                p = np.exp(logits - logits.max()); p /= p.sum()
                quality_confidence = float(p.max())
                if quality_confidence >= self.metadata.get("quality_threshold", .8):
                    quality = QUALITY[int(p.argmax())]
            partial = x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1
            if partial:
                quality = "unknown"
                geometry["grasp_candidates"] = []
            objects.append({"type": object_type, "confidence": float(confidence), "mask": mask,
                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)], "color": color,
                "color_confidence": color_confidence, "quality": quality,
                "quality_confidence": quality_confidence, "partial": partial, **geometry})
        return objects


def draw_instances(frame, objects):
    out = frame.copy()
    for i, obj in enumerate(objects):
        mask = obj["mask"] > 0
        color = np.array([(53*i+60) % 255, (97*i+100) % 255, (131*i+180) % 255])
        out[mask] = (.55*out[mask] + .45*color).astype(np.uint8)
        x, y, w, h = obj["bbox"]
        cv2.rectangle(out, (x, y), (x+w, y+h), color.tolist(), 2)
        cv2.putText(out, f"{obj['type']} {obj['confidence']:.2f} {obj['quality']}", (x, max(16,y-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
        for point in obj["grasp_candidates"]:
            cv2.circle(out, tuple(map(int, point["position_px"])), 4, (0,255,255), -1)
    return out
