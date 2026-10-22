"""Geometric features and shape classes from object contours.

From an overhead camera a cylinder and a sphere both look like a circle, so
circles are reported as the type "circular" rather than guessing one of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ShapeFeatures:
    area: float
    perimeter: float
    vertices: int
    circularity: float
    solidity: float
    rect_long: float  # longest side of the minimum-area rectangle, px
    rect_short: float  # shortest side, px
    angle_deg: float  # orientation in [0, 90)


def compute_features(contour: np.ndarray, epsilon_ratio: float) -> ShapeFeatures:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True) if perimeter > 0 else contour
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
    (_, _), (w, h), angle = cv2.minAreaRect(contour)
    return ShapeFeatures(
        area=area,
        perimeter=perimeter,
        vertices=len(approx),
        circularity=circularity,
        solidity=solidity,
        rect_long=float(max(w, h)),
        rect_short=float(min(w, h)),
        angle_deg=float(angle % 90.0),
    )


def classify_shape(f: ShapeFeatures, circularity_min: float, square_aspect_tolerance: float) -> Tuple[str, str, float]:
    """Return (shape, object_type, confidence). Unknown shapes return "desconocido"."""
    if f.rect_short <= 0:
        return "irregular", "desconocido", 0.0
    if f.circularity >= circularity_min:
        return "circulo", "circular", min(1.0, f.circularity)
    aspect = f.rect_long / f.rect_short
    if f.vertices == 3:
        return "triangulo", "prisma", 0.6
    if f.vertices == 4 and f.solidity >= 0.9:
        if aspect - 1.0 <= square_aspect_tolerance:
            return "cuadrado", "cubo", 0.9
        return "rectangulo", "prisma", 0.8
    return "irregular", "desconocido", 0.0
