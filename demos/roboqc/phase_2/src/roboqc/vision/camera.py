"""Frame sources: webcam, video file or still image (OpenCV only).

Every failure to open a source raises CameraError with a message the operator can act on.
"""

from __future__ import annotations

import collections
import os
import time
from typing import Deque, List, Optional, Union

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class CameraError(RuntimeError):
    pass


def _require_cv2() -> None:
    if cv2 is None:
        raise CameraError("OpenCV no esta instalado. Ejecuta: pip install opencv-python")


def _backend() -> int:
    # DirectShow opens webcams faster on Windows; other platforms use the default backend.
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


class FrameSource:
    description = "fuente"

    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None when no frame is available."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class WebcamSource(FrameSource):
    def __init__(self, index: int, width: Optional[int] = None, height: Optional[int] = None):
        _require_cv2()
        self.index = index
        self.description = f"camara {index}"
        self._cap = cv2.VideoCapture(index, _backend())
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(
                f"No se puede abrir la camara {index}. Comprueba que esta conectada, que ningun "
                "otro programa la usa, o lista las disponibles con 05_test_camera.py --list."
            )
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class VideoFileSource(FrameSource):
    def __init__(self, path: str, loop: bool = True):
        _require_cv2()
        self.description = f"video {os.path.basename(path)}"
        self._loop = loop
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(f"No se puede abrir el video: {path}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        if not ok and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class ImageSource(FrameSource):
    """A still image: every read returns the same frame, so the pipeline can run on it."""

    def __init__(self, path: str):
        _require_cv2()
        self.description = f"imagen {os.path.basename(path)}"
        self._frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if self._frame is None:
            raise CameraError(f"No se puede leer la imagen: {path}")

    def read(self) -> Optional[np.ndarray]:
        return self._frame.copy()


def open_source(spec: Union[int, str], width: Optional[int] = None, height: Optional[int] = None) -> FrameSource:
    """Open a webcam index ("0", 0), a video or image file, or a simulated scene ("sim:demo", "sim3d:demo3d")."""
    text = str(spec).strip()
    if text.lower().startswith("sim:"):
        from .simulation import SIM_PREFIX, SimulatedSource
        return SimulatedSource(text[len(SIM_PREFIX):])
    if text.lower().startswith("sim3d:"):
        from .sim3d import SIM3D_PREFIX, Sim3DSource
        return Sim3DSource(text[len(SIM3D_PREFIX):])
    if text.isdigit():
        return WebcamSource(int(text), width, height)
    if not os.path.exists(text):
        raise CameraError(f"No existe el archivo: {text}")
    ext = os.path.splitext(text)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return ImageSource(text)
    if ext in VIDEO_EXTENSIONS:
        return VideoFileSource(text)
    raise CameraError(f"Formato no soportado: {ext or 'sin extension'}")


def list_cameras(max_index: int = 4) -> List[int]:
    """Probe webcam indices 0..max_index-1. Slow on some drivers; run it on demand only."""
    _require_cv2()
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _backend())
        if cap.isOpened():
            found.append(index)
        cap.release()
    return found


class FpsMeter:
    """Rolling FPS and average read latency over the last `window` frames."""

    def __init__(self, window: int = 30):
        self._times: Deque[float] = collections.deque(maxlen=window)
        self._latencies: Deque[float] = collections.deque(maxlen=window)

    def tick(self, read_seconds: float = 0.0) -> None:
        self._times.append(time.perf_counter())
        self._latencies.append(read_seconds)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def latency_ms(self) -> float:
        return 1000.0 * sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
