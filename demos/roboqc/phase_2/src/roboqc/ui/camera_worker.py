"""Camera capture and vision processing off the UI thread."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PySide6.QtCore import QThread, Signal

from roboqc.vision.calibration import WorkspaceCalibration
from roboqc.vision.camera import CameraError, FpsMeter, open_source
from roboqc.vision.overlay import draw_overlay
from roboqc.vision.pipeline import VisionPipeline


class CameraWorker(QThread):
    """Reads frames, runs the vision pipeline and emits (QImage, WorldSnapshot).

    Processing runs at camera speed, but emission is throttled to display_fps so the
    UI thread never has to convert every frame.
    """

    updated = Signal(object, object)  # QImage, WorldSnapshot
    error = Signal(str)

    def __init__(self, source_spec, cfg: Dict[str, Any], calibration: Optional[WorkspaceCalibration] = None, parent=None):
        super().__init__(parent)
        self.source_spec = source_spec
        self.cfg = cfg
        self.pipeline = VisionPipeline(cfg, calibration=calibration)
        self.fps = 0.0
        self.latency_ms = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        import cv2
        from PySide6.QtGui import QImage

        cam = self.cfg["camera"]
        try:
            src = open_source(self.source_spec, cam["width"], cam["height"])
        except CameraError as e:
            self.error.emit(str(e))
            return
        meter = FpsMeter()
        interval = 1.0 / max(1, int(cam["display_fps"]))
        last_emit = 0.0
        try:
            with src:
                while not self._stop_event.is_set():
                    t0 = time.perf_counter()
                    frame = src.read()
                    read_seconds = time.perf_counter() - t0
                    if frame is None:
                        self.error.emit("La camara no devuelve imagenes.")
                        break
                    meter.tick(read_seconds)
                    self.fps, self.latency_ms = meter.fps, meter.latency_ms
                    result = self.pipeline.process(frame)
                    if time.monotonic() - last_emit < interval:
                        continue
                    last_emit = time.monotonic()
                    view = draw_overlay(frame, result.snapshot.objects, roi=self.cfg.get("roi"))
                    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                    h, w, channels = rgb.shape
                    image = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy()
                    self.updated.emit(image, result.snapshot)
        except Exception as e:  # keep the UI alive and show what failed
            self.error.emit(f"Error de procesamiento de video: {e}")

    def stop(self) -> None:
        self._stop_event.set()
