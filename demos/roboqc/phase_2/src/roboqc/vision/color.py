"""Color classification from HSV pixel statistics (OpenCV HSV scale)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - reported by camera.open_source with a clear message
    cv2 = None

from ..commands.entities import normalize_color

HsvRange = Tuple[int, int, int, int, int, int]  # h_min, h_max, s_min, s_max, v_min, v_max


def normalize_ranges(raw: Dict[str, Sequence[Sequence[int]]]) -> Dict[str, List[HsvRange]]:
    """Convert JSON lists to tuples, validating each range."""
    out: Dict[str, List[HsvRange]] = {}
    for name, ranges in raw.items():
        canonical = normalize_color(name) or name
        parsed = []
        for r in ranges:
            if len(r) != 6:
                raise ValueError(f"Rango HSV de '{name}' debe tener 6 valores: {r}")
            h0, h1, s0, s1, v0, v1 = (int(v) for v in r)
            if not (0 <= h0 <= h1 <= 179 and 0 <= s0 <= s1 <= 255 and 0 <= v0 <= v1 <= 255):
                raise ValueError(f"Rango HSV fuera de limites para '{name}': {r}")
            parsed.append((h0, h1, s0, s1, v0, v1))
        out[canonical] = parsed
    return out


def color_fractions(hsv_patch: np.ndarray, mask: np.ndarray, ranges: Dict[str, List[HsvRange]]) -> Dict[str, float]:
    """Fraction of masked pixels that fall inside each color's ranges."""
    total = int(cv2.countNonZero(mask))
    if total == 0:
        return {}
    out: Dict[str, float] = {}
    for name, rs in ranges.items():
        hit = np.zeros(mask.shape, dtype=np.uint8)
        for h0, h1, s0, s1, v0, v1 in rs:
            hit |= cv2.inRange(hsv_patch, (h0, s0, v0), (h1, s1, v1))
        out[name] = cv2.countNonZero(cv2.bitwise_and(hit, mask)) / total
    return out


def classify_color(fractions: Dict[str, float], min_fraction: float) -> Tuple[Optional[str], float]:
    """Pick the dominant color. Returns (None, fraction) when no color dominates."""
    if not fractions:
        return None, 0.0
    name = max(fractions, key=fractions.get)
    best = fractions[name]
    if best < min_fraction:
        return None, best
    return name, best
