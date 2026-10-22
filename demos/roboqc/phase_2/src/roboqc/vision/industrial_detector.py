"""Adapter from exported industrial segmentation to existing tracking records."""

import numpy as np

from roboqc.industrial.vision_inference import Segmenter
from .detector import Detection


class IndustrialDetector:
    def __init__(self, cfg):
        settings = cfg["industrial"]
        self.segmenter = Segmenter(settings["model_dir"], settings.get("prefer_onnx", True), settings.get("confidence", .35))
        self.roi = cfg.get("roi")

    def detect(self, frame):
        height,width = frame.shape[:2]
        x0,y0,w,h = self.roi or [0,0,width,height]
        if x0 < 0 or y0 < 0 or w <= 0 or h <= 0 or x0+w > width or y0+h > height:
            raise ValueError("Source ROI must be inside the image")
        instances = self.segmenter.predict(frame[y0:y0+h,x0:x0+w])
        detections = []
        for obj in instances:
            x,y,bw,bh = obj["bbox"]
            full_mask = np.zeros((height,width),np.uint8)
            full_mask[y0:y0+h,x0:x0+w] = obj["mask"]
            points = []
            for candidate in obj["grasp_candidates"]:
                candidate = dict(candidate)
                candidate["position_px"] = [candidate["position_px"][0]+x0,candidate["position_px"][1]+y0]
                points.append(candidate)
            detections.append(Detection(color=obj["color"],color_confidence=obj["color_confidence"],
                shape="industrial",object_type=obj["type"],shape_confidence=obj["confidence"],
                bbox=(x+x0,y+y0,bw,bh),centroid=(obj["center_px"][0]+x0,obj["center_px"][1]+y0),
                area_px=float(obj["mask"].sum()),orientation_deg=obj["orientation_deg"],
                solidity=1.0,vertices=0,touches_border=obj["partial"],
                mask=full_mask,industrial_quality=obj["quality"],grasp_candidates=points,
                class_confidence=obj["confidence"]))
        return detections
