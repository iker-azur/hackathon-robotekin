"""Tell whether a piece lies in the tray, from the front camera.

The tray floor is saturated orange; the pieces are bare grey or silver metal. Inside a rectangle of the floor, a
blob of low-saturation pixels is therefore a piece. Measured on the front-camera frames of the MikelRubi recordings
made with the final framing (``so101-pick-cube_20261024_22*`` and ``up-down*_20261024_22*``: the first frame of 67
pick episodes against the last frame of 54 up-down episodes, where the tray is empty), the largest blob is 347 px or
more with a piece and 58 px or less without, so ``MIN_BLOB`` sits in the middle of a six-fold gap. Earlier sessions
framed the tray differently (the same rule then gave false alarms on the empty tray); if the camera is moved, pass
the new ``roi`` (``30_full_demo_app.py --tray-roi``) and check the rectangle drawn on the front view.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# x0, y0, x1, y1 as fractions of the frame: the tray floor, away from the rim and from the arm at rest.
DEFAULT_ROI = (0.06, 0.34, 0.78, 0.90)
MAX_SATURATION = 60  # the orange floor is far above this; metal is below
MIN_VALUE = 60  # skip the dark shadows at the tray edge
MIN_BLOB = 150  # pixels of the largest blob, at 640x480, after removing speckle


@dataclass(frozen=True)
class TrayReading:
    present: bool
    area: int  # pixels of the largest blob
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 of the searched rectangle, in pixels


def parse_roi(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(part) for part in text.split(","))
    if len(values) != 4 or not all(0 <= v <= 1 for v in values) or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("tray ROI must be x0,y0,x1,y1 as fractions with x0<x1 and y0<y1, e.g. 0.06,0.34,0.78,0.90")
    return values


def detect_piece(frame_rgb: np.ndarray, roi=DEFAULT_ROI, min_blob: int = MIN_BLOB) -> TrayReading:
    """``frame_rgb`` is an HxWx3 uint8 RGB frame, as ``robot.get_observation()`` returns it."""
    height, width = frame_rgb.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * width), int(roi[1] * height), int(roi[2] * width), int(roi[3] * height)
    scale = (width * height) / (640 * 480)  # the blob threshold is stated at 640x480
    hsv = cv2.cvtColor(np.ascontiguousarray(frame_rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] < MAX_SATURATION) & (hsv[..., 2] > MIN_VALUE)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    area = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    return TrayReading(area >= min_blob * scale, area, (x0, y0, x1, y1))


class PieceWatcher:
    """Debounce: a piece counts as present once it has been seen for ``stable_s`` seconds without a gap."""

    def __init__(self, stable_s: float = 1.0) -> None:
        self.stable_s, self.since = stable_s, None

    def update(self, present: bool, now: float) -> bool:
        if not present:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.stable_s

    def reset(self) -> None:
        self.since = None


def draw_overlay(frame_rgb: np.ndarray, reading: TrayReading) -> np.ndarray:
    """A copy of the frame with the searched rectangle: green with a piece, grey without."""
    shown = frame_rgb.copy()
    x0, y0, x1, y1 = reading.box
    colour = (40, 220, 60) if reading.present else (170, 170, 170)
    cv2.rectangle(shown, (x0, y0), (x1, y1), colour, 2)
    cv2.putText(shown, f"pieza: {reading.area} px" if reading.present else "bandeja vacia",
                (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return shown
