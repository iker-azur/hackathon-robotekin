"""Draw detections on a frame. Labels are ASCII because cv2.putText cannot render accents."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .world_state import ObjectRecord

# BGR colors per quality verdict.
QUALITY_BGR = {
    "GOOD": (60, 200, 60),
    "DEFECTIVE": (50, 50, 230),
    "UNCERTAIN": (0, 180, 255),
}
ROI_BGR = (255, 200, 0)
SELECTED_BGR = (255, 255, 255)
LABEL_SCALE = 0.5
LABEL_BG_BGR = (20, 20, 20)


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _free_label_top(x: int, y: int, width: int, height: int, occupied: List[Tuple[int, int, int, int]],
                    frame_h: int) -> int:
    """Top edge for a label above its object, moved down past any label already placed there."""
    top = max(0, y - height - 2)
    while top + height < frame_h and any(_overlaps((x, top, width, height), box) for box in occupied):
        top += height + 2
    return top


def draw_overlay(
    frame: np.ndarray,
    objects: Iterable[ObjectRecord],
    roi: Optional[Sequence[int]] = None,
) -> np.ndarray:
    out = frame.copy()
    occupied: List[Tuple[int, int, int, int]] = []
    if roi:
        x, y, w, h = (int(v) for v in roi)
        cv2.rectangle(out, (x, y), (x + w, y + h), ROI_BGR, 1)
    for obj in objects:
        if not obj.in_view:
            continue
        x, y, w, h = obj.bounding_box
        color = QUALITY_BGR.get(obj.quality, (200, 200, 200))
        if obj.mask is not None:
            pixels = obj.mask > 0
            out[pixels] = (.6*out[pixels] + .4*np.asarray(color)).astype(np.uint8)
        for point in obj.grasp_candidates:
            cv2.circle(out, tuple(map(int,point["position_px"])), 4, (0,255,255), -1)
        cv2.rectangle(out, (x, y), (x + w, y + h), SELECTED_BGR if obj.selected else color,
                      3 if obj.selected else 2)
        label = f"{obj.object_id} {obj.object_type} {obj.color or 'color?'} {obj.quality}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, 1)
        box_w, box_h = tw + 8, th + base + 8
        left = max(0, min(x, out.shape[1] - box_w))
        top = _free_label_top(left, y, box_w, box_h, occupied, out.shape[0])
        occupied.append((left, top, box_w, box_h))
        # Dark background so the text stays readable on any object color or table.
        cv2.rectangle(out, (left, top), (left + box_w, top + box_h), LABEL_BG_BGR, -1)
        cv2.putText(out, label, (left + 4, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, color, 1,
                    cv2.LINE_AA)
        cx, cy = int(obj.position_px[0]), int(obj.position_px[1])
        cv2.circle(out, (cx, cy), 3, color, -1)
    return out


def draw_text_lines(frame: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (10, 24)) -> np.ndarray:
    out = frame.copy()
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return out
