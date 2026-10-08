"""Bounded latest-frame JPEG worker; rendering remains on the Isaac main thread."""
import threading
import time


class FrameEncoder:
    def __init__(self, bus, on_error):
        self.bus, self.on_error = bus, on_error
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.thread = threading.Thread(target=self.run, name="camera-jpeg", daemon=True)
        self.thread.start()

    def offer(self, image, meta, quality, width=None):
        # Sensor arrays may be overwritten by the next render.
        packet = (image.copy(), dict(meta), quality, width)
        with self.condition:
            if self.closed:
                return
            if self.pending is not None:
                self.bus.add_to_metric("camera_dropped_frames", 1)
            self.pending = packet
            self.condition.notify()

    def run(self):
        import cv2
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.pending is None:
                    return
                image, meta, quality, width = self.pending
                self.pending = None
            try:
                started = time.perf_counter()
                if width and image.shape[1] > width:
                    image = cv2.resize(image, (width, max(1, round(image.shape[0]*width/image.shape[1]))))
                ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    raise ValueError("No se pudo codificar la imagen.")
                self.bus.set_metric("jpeg_ms", round((time.perf_counter()-started)*1000, 2))
                self.bus.put_frame(encoded.tobytes(), meta)
                self.bus.tick("camera")
                self.bus.set_state(camera_error="")
            except Exception as exc:
                self.on_error(str(exc))

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=2)
