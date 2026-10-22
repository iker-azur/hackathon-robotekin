"""Foreground segmentation and per-object extraction (OpenCV, CPU only).

Single-frame detection only: identities come from tracker.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .color import classify_color, color_fractions, normalize_ranges
from .shape import classify_shape, compute_features


@dataclass
class Detection:
    color: Optional[str]
    color_confidence: float  # fraction of object pixels inside the winning color ranges
    shape: str
    object_type: str
    shape_confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h in full-frame pixels
    centroid: Tuple[float, float]  # full-frame pixels
    area_px: float
    orientation_deg: Optional[float]
    solidity: float
    vertices: int
    touches_border: bool  # object is cut by the frame or ROI edge: partially visible
    mask: Optional[np.ndarray] = None
    industrial_quality: Optional[str] = None
    grasp_candidates: Optional[list] = None
    class_confidence: Optional[float] = None


def normalize_roi(roi: Optional[Sequence[int]]) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise ValueError(f"ROI invalida: {roi}")
    return x, y, w, h


# Shading inside one object: a V spread this large means the side faces are darker than the top.
TOP_FACE_MIN_CONTRAST = 40
# The lit top face must cover at least this share of the silhouette to be used for the shape.
TOP_FACE_MIN_SHARE = 0.25
# Two same-color objects are split where the neck between their centres is below this share of the peak.
SPLIT_PEAK_RATIO = 0.7


def _split_touching(blob: np.ndarray) -> List[np.ndarray]:
    """Split one blob where two objects touch along a narrow neck (distance-transform watershed)."""
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    count, markers = cv2.connectedComponents((dist >= SPLIT_PEAK_RATIO * dist.max()).astype(np.uint8))
    if count <= 2:  # background and one peak: a single object
        return [blob]
    markers = markers.astype(np.int32)
    markers[blob == 0] = count  # background seed, so the flood stays inside the blob
    depth = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    landscape = cv2.cvtColor(255 - depth, cv2.COLOR_GRAY2BGR)
    cv2.watershed(landscape, markers)
    return [np.where(markers == k, 255, 0).astype(np.uint8) for k in range(1, count)]


class ObjectDetector:
    def __init__(self, cfg: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV no esta instalado (pip install opencv-python).")
        self.cfg = cfg
        self.ranges = normalize_ranges(cfg["color"]["ranges"])
        self.roi = normalize_roi(cfg.get("roi"))
        k = int(cfg["morphology"]["kernel_size"])
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def foreground_mask(self, hsv: np.ndarray) -> np.ndarray:
        fg = self.cfg["foreground"]
        mask = cv2.inRange(hsv, (0, fg["min_saturation"], fg["min_value"]), (179, 255, 255))
        if fg["detect_dark"]:
            mask |= cv2.inRange(hsv, (0, 0, 0), (179, 255, fg["dark_value_max"]))
        if fg["detect_bright_achromatic"]:
            mask |= cv2.inRange(
                hsv, (0, 0, fg["bright_value_min"]), (179, fg["achromatic_saturation_max"], 255)
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel,
                                iterations=self.cfg["morphology"]["open_iterations"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel,
                                iterations=self.cfg["morphology"]["close_iterations"])
        return mask

    def _contours_by_color(self, hsv: np.ndarray, support_hsv: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Outlines per color. Touching objects of different colors never merge into one contour."""
        fg = self.foreground_mask(hsv)
        support = self.foreground_mask(support_hsv) if support_hsv is not None else fg
        contours: List[np.ndarray] = []
        for ranges in self.ranges.values():
            in_color = np.zeros(fg.shape, dtype=np.uint8)
            for h0, h1, s0, s1, v0, v1 in ranges:
                in_color |= cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
            mask = cv2.morphologyEx(cv2.bitwise_and(in_color, fg), cv2.MORPH_CLOSE, self._kernel,
                                    iterations=self.cfg["morphology"]["close_iterations"])
            # Blurring must not grow the object's silhouette into the background.
            mask = cv2.bitwise_and(mask, support)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for k in range(1, count):
                x, y, w, h, _ = stats[k]
                blob = np.where(labels[y:y + h, x:x + w] == k, 255, 0).astype(np.uint8)
                for piece in _split_touching(blob):
                    found, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours.extend(c + np.array([[x, y]]) for c in found)
        return contours

    def _top_face_contour(self, patch_hsv: np.ndarray, mask: np.ndarray, outline: np.ndarray) -> np.ndarray:
        """Outline of the lit top face. Darker side faces would otherwise turn a cube into a hexagon."""
        value = patch_hsv[:, :, 2]
        inside = value[mask > 0]
        if inside.size == 0 or int(inside.max()) - int(inside.min()) < TOP_FACE_MIN_CONTRAST:
            return outline
        threshold, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top = np.zeros(mask.shape, dtype=np.uint8)
        top[(value >= threshold) & (mask > 0)] = 255
        # Closing bridges dim patches inside the top face (uneven light); the hull restores its corners.
        size = max(3, min(mask.shape) // 6) | 1
        top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        found, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            return outline
        hull = cv2.convexHull(max(found, key=cv2.contourArea))
        if cv2.contourArea(hull) < TOP_FACE_MIN_SHARE * cv2.contourArea(outline):
            return outline
        return hull

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        frame_h, frame_w = frame_bgr.shape[:2]
        x0, y0, w0, h0 = self.roi if self.roi else (0, 0, frame_w, frame_h)
        crop = frame_bgr[y0:y0 + h0, x0:x0 + w0]
        hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)
        contours = self._contours_by_color(hsv, cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))

        shape_cfg = self.cfg["shape"]
        min_fraction = self.cfg["color"]["min_pixel_fraction"]
        detections: List[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < shape_cfg["min_area_px"] or area > shape_cfg["max_area_px"]:
                continue
            x, y, w, h = cv2.boundingRect(contour)

            # Color is voted over the object's own pixels, not its bounding box.
            local = (contour - np.array([[x, y]])).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [local], -1, 255, thickness=cv2.FILLED)
            top = self._top_face_contour(hsv[y:y + h, x:x + w], mask, local)
            feats = compute_features(top, shape_cfg["approx_epsilon_ratio"])
            color, color_conf = classify_color(
                color_fractions(hsv[y:y + h, x:x + w], mask, self.ranges), min_fraction
            )

            shape, object_type, shape_conf = classify_shape(
                feats, shape_cfg["circularity_min"], shape_cfg["square_aspect_tolerance"]
            )
            m = cv2.moments(contour)
            cx = m["m10"] / m["m00"] if m["m00"] else x + w / 2.0
            cy = m["m01"] / m["m00"] if m["m00"] else y + h / 2.0
            detections.append(Detection(
                color=color,
                color_confidence=float(color_conf),
                shape=shape,
                object_type=object_type,
                shape_confidence=float(shape_conf),
                bbox=(x + x0, y + y0, w, h),
                centroid=(cx + x0, cy + y0),
                area_px=area,
                orientation_deg=None if shape == "circulo" else feats.angle_deg,
                solidity=feats.solidity,
                vertices=feats.vertices,
                touches_border=x == 0 or y == 0 or x + w >= w0 or y + h >= h0,
            ))
        detections.sort(key=lambda d: (d.centroid[1], d.centroid[0]))
        return detections
