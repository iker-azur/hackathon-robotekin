"""Frame-to-frame tracking with stable object IDs.

Association uses centroid distance solved with the Hungarian algorithm, so two
same-colored pieces that are close together keep their own IDs. A track is only
reported after `min_hits` matched frames, which suppresses one-frame noise.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


@dataclass
class FrameEvents:
    entered: List[str] = field(default_factory=list)
    moved: List[str] = field(default_factory=list)
    disappeared: List[str] = field(default_factory=list)


@dataclass
class Track:
    object_id: str
    first_seen: float
    last_seen: float
    detection: Detection
    history_length: int
    hits: int = 1
    missed: int = 0
    moving: bool = False
    confirmed: bool = False
    observations: Deque[Tuple[Optional[str], str, str]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.observations = deque(maxlen=self.history_length)
        self._record(self.detection)

    def _record(self, det: Detection) -> None:
        self.observations.append((det.color, det.shape, det.object_type))

    def update(self, det: Detection, timestamp: float, motion_threshold: float) -> None:
        dx = det.centroid[0] - self.detection.centroid[0]
        dy = det.centroid[1] - self.detection.centroid[1]
        self.moving = (dx * dx + dy * dy) ** 0.5 >= motion_threshold
        self.detection = det
        self.last_seen = timestamp
        self.hits += 1
        self.missed = 0
        self._record(det)

    @property
    def stable_color(self) -> Optional[str]:
        return _mode([c for c, _, _ in self.observations if c is not None])

    @property
    def stable_shape(self) -> str:
        return _mode([s for _, s, _ in self.observations if s != "irregular"]) or self.detection.shape

    @property
    def stable_type(self) -> str:
        return _mode([t for _, _, t in self.observations if t != "desconocido"]) or self.detection.object_type

    @property
    def status(self) -> str:
        if self.missed > 0:
            return "OCCLUDED"
        if self.detection.touches_border:
            return "PARTIAL"
        return "VISIBLE"


def _mode(values: Sequence[str]) -> Optional[str]:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


class Tracker:
    def __init__(self, cfg: Dict[str, Any]):
        self.max_distance = float(cfg["max_distance_px"])
        self.max_missed = int(cfg["max_missed_frames"])
        self.motion_threshold = float(cfg["motion_threshold_px"])
        self.min_hits = int(cfg["min_hits"])
        self.history_length = int(cfg["history_length"])
        self._tracks: List[Track] = []
        self._next_number = 1

    def _new_id(self) -> str:
        object_id = f"OBJ_{self._next_number:03d}"
        self._next_number += 1
        return object_id

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks)

    def update(self, detections: Sequence[Detection], timestamp: float) -> FrameEvents:
        events = FrameEvents()
        matches = self._associate(detections)
        matched_tracks = set()
        matched_dets = set()

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            was_confirmed = track.confirmed
            track.update(detections[det_idx], timestamp, self.motion_threshold)
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            if not was_confirmed and track.hits >= self.min_hits:
                track.confirmed = True
                events.entered.append(track.object_id)
            if track.confirmed and track.moving:
                events.moved.append(track.object_id)

        survivors: List[Track] = []
        for idx, track in enumerate(self._tracks):
            if idx not in matched_tracks:
                track.missed += 1
                track.moving = False
            if track.missed > self.max_missed:
                if track.confirmed:
                    events.disappeared.append(track.object_id)
                continue
            survivors.append(track)
        self._tracks = survivors

        for det_idx, det in enumerate(detections):
            if det_idx in matched_dets:
                continue
            self._tracks.append(Track(
                object_id=self._new_id(),
                first_seen=timestamp,
                last_seen=timestamp,
                detection=det,
                history_length=self.history_length,
                confirmed=self.min_hits <= 1,
            ))
            if self.min_hits <= 1:
                events.entered.append(self._tracks[-1].object_id)
        return events

    def _associate(self, detections: Sequence[Detection]) -> List[Tuple[int, int]]:
        if not self._tracks or not detections:
            return []
        cost = np.zeros((len(self._tracks), len(detections)), dtype=np.float64)
        for i, track in enumerate(self._tracks):
            tx, ty = track.detection.centroid
            for j, det in enumerate(detections):
                dx, dy = det.centroid[0] - tx, det.centroid[1] - ty
                cost[i, j] = (dx * dx + dy * dy) ** 0.5
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= self.max_distance]
