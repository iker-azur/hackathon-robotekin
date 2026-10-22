"""Vision pipeline: frame -> detections -> tracks -> world state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .calibration import WorkspaceCalibration
from .detector import Detection, ObjectDetector
from .quality import QualityRuleSet
from .tracker import FrameEvents, Tracker
from .world_state import WorldSnapshot, WorldState


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    detections: List[Detection]
    events: FrameEvents
    snapshot: WorldSnapshot
    processing_ms: float


class VisionPipeline:
    """Owns the detector and the world state. Call process() from one thread only."""

    def __init__(self, cfg: Dict[str, Any], calibration: Optional[WorkspaceCalibration] = None):
        self.cfg = cfg
        if cfg.get("industrial", {}).get("enabled"):
            from .industrial_detector import IndustrialDetector
            self.detector = IndustrialDetector(cfg)
        else:
            self.detector = ObjectDetector(cfg)
        self.world = WorldState(Tracker(cfg["tracking"]), QualityRuleSet(cfg["quality"]), calibration)
        self.frame_index = 0

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> FrameResult:
        t0 = time.perf_counter()
        ts = time.monotonic() if timestamp is None else timestamp
        detections = self.detector.detect(frame)
        events = self.world.update(detections, ts)
        snapshot = self.world.snapshot()
        self.frame_index += 1
        return FrameResult(
            frame_index=self.frame_index,
            timestamp=ts,
            detections=detections,
            events=events,
            snapshot=snapshot,
            processing_ms=(time.perf_counter() - t0) * 1000.0,
        )
