"""A live window showing the camera and what perception makes of it, and a key reader.

Two rules decide the shape of this file.

**The window belongs to the main thread.** ``cv2.imshow`` and ``waitKey`` are not thread-safe,
and Isaac's simulation loop has to be on the main thread too, so the window is pumped from
inside that loop rather than from a thread of its own. Two processes on one GPU would also
work but doubles the start-up cost of a two-minute simulator launch.

**Perception does not.** An open-vocabulary detector takes long enough to be visible as a
freeze, so frames are handed to a worker thread and the loop keeps pumping. The worker
returns an annotated frame through a one-slot queue and never touches the window. If it is
still busy when the next frame arrives, that frame is dropped: for a live view the newest
frame is the one worth having, and a growing queue would show the user stale images.

Every entry point is a no-op when disabled, so ``--headless`` runs the same code with no
window and no worker.

Requires ``opencv-python``, not ``opencv-python-headless``: the headless wheel has no
``imshow`` at all. Installing both at once breaks the import, so the headless one has to go.

    python tools/try_viewer.py --image ..\04_perception\out\frames\trial000.png --seconds 3
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# What each key does, printed in the window so the demo is usable without documentation.
KEY_HELP = "[q]uit  [space]pause  [n]step  [r]reshuffle  [s]save  [a]arm occlusion  [d]detect"

QUIT = "q"
PAUSE = " "
STEP = "n"
RESHUFFLE = "r"
SAVE = "s"
ARM_MASK = "a"
DETECT = "d"

KEYS = (QUIT, PAUSE, STEP, RESHUFFLE, SAVE, ARM_MASK, DETECT)


class Viewer:
    """A window, and the keys pressed in it. Does nothing when ``enabled`` is False."""

    def __init__(self, enabled: bool = True, name: str = "leisaac demo 10", scale: float = 1.0):
        self.enabled = bool(enabled and hasattr(cv2, "imshow"))
        self.name = name
        self.scale = scale
        self.opened = False
        self.frames = 0
        self._pressed: set[str] = set()
        self._why_disabled = None
        if not enabled:
            self._why_disabled = "disabled"
        elif not hasattr(cv2, "imshow"):
            self._why_disabled = "this OpenCV has no imshow (opencv-python-headless?)"

    # -- the window ------------------------------------------------------------

    def _fit(self, image: np.ndarray) -> np.ndarray:
        if self.scale == 1.0:
            return image
        height, width = image.shape[:2]
        return cv2.resize(image, (int(width * self.scale), int(height * self.scale)),
                         interpolation=cv2.INTER_AREA)

    def show(self, image: np.ndarray, caption: str = "") -> set[str]:
        """Put a frame up, read the keys, and return the ones pressed since last time.

        ``caption`` is drawn as a strip along the bottom together with the key help, so the
        user can see which camera and which phase they are looking at.
        """
        if not self.enabled:
            return set()
        frame = image
        if caption or self.enabled:
            frame = self._caption(image, caption)
        try:
            cv2.imshow(self.name, self._fit(frame))
            self.opened = True
        except cv2.error as error:
            # No display, or the window was closed by the window manager. Stop trying rather
            # than raising on every frame for the rest of the run.
            self.enabled = False
            self._why_disabled = f"imshow failed: {str(error).splitlines()[0][:80]}"
            return set()
        return self._drain_keys()

    def _caption(self, image: np.ndarray, caption: str) -> np.ndarray:
        strip = 22
        out = image.copy()
        cv2.rectangle(out, (0, out.shape[0] - strip), (out.shape[1], out.shape[0]), (25, 25, 25), -1)
        cv2.putText(out, caption, (6, out.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(out, KEY_HELP, (6, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (170, 200, 120), 1, cv2.LINE_AA)
        return out

    def _drain_keys(self) -> set[str]:
        """Every key pressed since the last call. waitKey(1) is what makes the window pump."""
        pressed: set[str] = set()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                break
            try:
                pressed.add(chr(key))
            except ValueError:
                continue
            if len(pressed) > 8:  # a stuck key can spin here
                break
        self._pressed |= pressed
        self.frames += 1
        got, self._pressed = self._pressed, set()
        return got

    def wants_quit(self, keys: set[str]) -> bool:
        return QUIT in keys or "escape" in keys

    def close(self) -> None:
        if self.opened and hasattr(cv2, "destroyAllWindows"):
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        self.opened = False

    def __enter__(self) -> "Viewer":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class PerceptionWorker:
    """Runs ``perceive`` on a worker thread so the window keeps moving.

    ``submit`` never blocks. If a frame is already in flight it is dropped, and the number of
    drops is counted so a slow detector is visible rather than silent.
    """

    def __init__(self, camera, frame, plane_z, looking_for, detect=None, enabled: bool = True):
        self.enabled = bool(enabled)
        self.camera = camera
        self.frame = frame
        self.plane_z = plane_z
        self.looking_for = looking_for
        self.detect = detect
        self.submitted = 0
        self.dropped = 0
        self.finished = 0
        self._lock = threading.Lock()
        self._pending = False
        self._latest: tuple[np.ndarray, object] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perceive") if self.enabled else None

    def submit(self, image_bgr: np.ndarray) -> bool:
        """Hand over a frame. False means the worker was busy and this one was dropped."""
        if not self.enabled:
            return False
        with self._lock:
            if self._pending:
                self.dropped += 1
                return False
            self._pending = True
        self.submitted += 1
        self._pool.submit(self._work, image_bgr.copy())
        return True

    def _work(self, image_bgr: np.ndarray) -> None:
        import perceive

        try:
            scene = perceive.perceive(
                image_bgr, source="live", camera=self.camera, frame=self.frame,
                plane_z=self.plane_z, detect=self.detect, looking_for=self.looking_for,
            )
            annotated = perceive.annotate(image_bgr, scene)
            with self._lock:
                self._latest = (annotated, scene)
        except Exception:  # noqa: BLE001 - a bad frame must not kill the demo
            import traceback

            traceback.print_exc()
        finally:
            with self._lock:
                self._pending = False
                self.finished += 1

    def latest(self):
        """The most recent annotated frame and its scene, or None if nothing is ready."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)

    def __enter__(self) -> "PerceptionWorker":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class FrameWriter:
    """Writes frames to disk instead of a window, so ``--headless`` still shows something."""

    def __init__(self, folder, every: int = 10, prefix: str = "frame"):
        import pathlib

        self.folder = pathlib.Path(folder)
        self.every = max(1, every)
        self.prefix = prefix
        self.written = 0
        self.seen = 0

    def offer(self, image: np.ndarray) -> None:
        import perceive

        self.seen += 1
        if self.seen % self.every:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        perceive.save_png(self.folder / f"{self.prefix}{self.written:04d}.png", image)
        self.written += 1
