"""Thread-safe world state: the single source of truth for visible objects.

This module imports neither Qt nor the UI. The future robot planner reads
`WorldState.snapshot()` and nothing else from the vision stack.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .calibration import WorkspaceCalibration
from .detector import Detection
from .quality import QualityRule, QualityRuleSet
from .tracker import FrameEvents, Tracker

TRACKED_STATUSES = ("VISIBLE", "PARTIAL")  # OCCLUDED objects are remembered but not in view


@dataclass(frozen=True)
class ObjectRecord:
    object_id: str
    color: Optional[str]
    object_type: str  # cubo | prisma | circular | desconocido
    shape: str  # cuadrado | rectangulo | triangulo | circulo | irregular
    position_px: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    area_px: float
    orientation_deg: Optional[float]
    quality: str  # GOOD | DEFECTIVE | UNCERTAIN
    quality_reasons: Tuple[str, ...]
    detection_status: str  # VISIBLE | PARTIAL | OCCLUDED
    motion: str  # NEW | STATIONARY | MOVING
    confidence: float
    first_seen: float
    last_seen: float
    world_position: Optional[Dict[str, Any]]
    selected: bool
    mask: Optional[Any] = None
    industrial_quality: Optional[str] = None
    grasp_candidates: Tuple[Dict[str, Any], ...] = ()
    class_confidence: Optional[float] = None

    @property
    def in_view(self) -> bool:
        return self.detection_status in TRACKED_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "type": self.object_type,
            "color": self.color,
            "shape": self.shape,
            "position_px": [round(self.position_px[0], 1), round(self.position_px[1], 1)],
            "bounding_box": list(self.bounding_box),
            "area_px": round(self.area_px, 1),
            "orientation_deg": None if self.orientation_deg is None else round(self.orientation_deg, 1),
            "quality": self.quality,
            "quality_reasons": list(self.quality_reasons),
            "detection_status": self.detection_status,
            "motion": self.motion,
            "confidence": round(self.confidence, 3),
            "world_position": self.world_position,
            "selected": self.selected,
            "industrial_quality": self.industrial_quality,
            "grasp_candidates": list(self.grasp_candidates),
            "class_confidence": self.class_confidence,
            "mask_rle": self._mask_rle(),
        }

    def _mask_rle(self):
        if self.mask is None:
            return None
        from itertools import groupby
        flat = (self.mask.ravel(order="C") > 0).tolist()
        counts = [sum(1 for _ in group) for _,group in groupby(flat)]
        if flat and flat[0]:
            counts.insert(0,0)
        return {"size":list(self.mask.shape),"order":"C","counts":counts}


@dataclass(frozen=True)
class WorldSnapshot:
    timestamp: float
    objects: Tuple[ObjectRecord, ...]
    rules: Tuple[QualityRule, ...]
    events: FrameEvents = field(default_factory=FrameEvents)

    def by_id(self, object_id: str) -> Optional[ObjectRecord]:
        return next((o for o in self.objects if o.object_id == object_id), None)

    def visible(self) -> List[ObjectRecord]:
        return [o for o in self.objects if o.in_view]

    def counts(self) -> Dict[str, int]:
        visible = self.visible()
        return {
            "visible": len(visible),
            "tracked": len(self.objects),
            "GOOD": sum(o.quality == "GOOD" for o in visible),
            "DEFECTIVE": sum(o.quality == "DEFECTIVE" for o in visible),
            "UNCERTAIN": sum(o.quality == "UNCERTAIN" for o in visible),
        }


class WorldState:
    def __init__(self, tracker: Tracker, quality: QualityRuleSet, calibration: Optional[WorkspaceCalibration] = None):
        self._lock = threading.Lock()
        self._tracker = tracker
        self._quality = quality
        self._calibration = calibration
        self._selected: List[str] = []
        self._timestamp = 0.0
        self._events = FrameEvents()

    def update(self, detections: Sequence[Detection], timestamp: float) -> FrameEvents:
        with self._lock:
            self._events = self._tracker.update(detections, timestamp)
            self._timestamp = timestamp
            return self._events

    def snapshot(self) -> WorldSnapshot:
        """Build an immutable view. Quality is evaluated here, so rule changes apply at once."""
        with self._lock:
            records = []
            for track in self._tracker.tracks:
                if not track.confirmed:
                    continue
                det = track.detection
                color = track.stable_color
                object_type = track.stable_type
                evaluation = self._quality.evaluate(
                    color=color,
                    object_type=object_type,
                    solidity=det.solidity,
                    area_px=det.area_px,
                    color_confidence=det.color_confidence,
                    partial=det.touches_border,
                )
                x, y = det.centroid
                records.append(ObjectRecord(
                    object_id=track.object_id,
                    color=color,
                    object_type=object_type,
                    shape=track.stable_shape,
                    position_px=(x, y),
                    bounding_box=det.bbox,
                    area_px=det.area_px,
                    orientation_deg=det.orientation_deg,
                    quality=("GOOD" if det.industrial_quality == "good" else "DEFECTIVE" if det.industrial_quality in ("defect_1","defect_2") else "UNCERTAIN") if det.industrial_quality is not None else evaluation.quality,
                    quality_reasons=("industrial visible-quality classifier: " + det.industrial_quality,) if det.industrial_quality is not None else tuple(evaluation.reasons),
                    detection_status=track.status,
                    motion="MOVING" if track.moving else ("STATIONARY" if track.hits > 1 else "NEW"),
                    confidence=min(det.color_confidence, det.shape_confidence),
                    first_seen=track.first_seen,
                    last_seen=track.last_seen,
                    world_position=self._calibration.describe(x, y) if self._calibration else None,
                    selected=track.object_id in self._selected,
                    mask=det.mask,
                    industrial_quality=det.industrial_quality,
                    grasp_candidates=tuple(det.grasp_candidates or []),
                    class_confidence=det.class_confidence,
                ))
            return WorldSnapshot(
                timestamp=self._timestamp,
                objects=tuple(records),
                rules=tuple(self._quality.rules),
                events=self._events,
            )

    def add_rule(self, rule: QualityRule) -> None:
        with self._lock:
            self._quality.add_rule(rule)

    def clear_rules(self) -> None:
        with self._lock:
            self._quality.clear()

    def set_selection(self, object_ids: Sequence[str]) -> None:
        with self._lock:
            self._selected = list(object_ids)

    def selected_ids(self) -> List[str]:
        with self._lock:
            return list(self._selected)
