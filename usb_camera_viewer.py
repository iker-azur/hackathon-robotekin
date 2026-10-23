import sys, os, re, glob, subprocess, time
from datetime import datetime
from collections import deque

import cv2
import usb.core
import usb.util

from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QListWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QComboBox
)


class CameraScanWorker(QObject):
    """Probe camera devices off the UI thread."""

    finished = Signal(object)

    def run(self):
        cameras = []
        if sys.platform.startswith("linux"):
            for path in sorted(glob.glob("/dev/video*"), key=USBCameraApp.video_number):
                if USBCameraApp.valid_camera(path):
                    number = USBCameraApp.video_number(path)
                    name = USBCameraApp.camera_name(path)
                    cameras.append((f"{number} - {name} ({path})", (path, cv2.CAP_V4L2)))
        else:
            # OpenCV does not expose camera names consistently on Windows.
            # Probe a bounded range and expose only devices that return a frame.
            for index in range(10):
                backend = USBCameraApp.probe_camera(index)
                if backend is not None:
                    cameras.append((f"Camera {index}", (index, backend)))
        self.finished.emit(cameras)


class USBCameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USB Device Scanner and Camera Viewer")
        self.resize(900, 700)

        self.camera = None
        self.prev_sample = None
        self.last_change = time.monotonic()
        self.last_frame = None

        # FPS reales
        self.frame_times = deque(maxlen=60)

        # Detección de posible congelación
        self.FREEZE_TIMEOUT = 3.0
        self.CHANGE_THRESHOLD = 0.02

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.device_list = QListWidget()
        self.camera_combo = QComboBox()
        self.camera_combo.setEnabled(False)
        self._camera_scan_thread = None
        self._camera_scan_worker = None
        self._close_after_camera_scan = False

        self.video_label = QLabel("Camera preview will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background:black; color:white;")

        scan_btn = QPushButton("Scan USB Devices")
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.start_btn = QPushButton("Start Camera")
        stop_btn = QPushButton("Stop Camera")
        self.start_btn.setEnabled(False)

        scan_btn.clicked.connect(self.scan_usb_devices)
        self.refresh_btn.clicked.connect(self.refresh_camera_list)
        self.start_btn.clicked.connect(self.start_camera)
        stop_btn.clicked.connect(self.stop_camera)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.refresh_btn)

        button_layout = QHBoxLayout()
        for button in (scan_btn, self.start_btn, stop_btn):
            button_layout.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Connected USB Devices:"))
        layout.addWidget(self.device_list)
        layout.addLayout(camera_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.video_label)

        self.scan_usb_devices()
        self.video_label.setText("Buscando cámaras disponibles…")
        # Let Qt display the window before probing devices. Camera discovery can
        # take several seconds when no camera is connected.
        QTimer.singleShot(0, self.refresh_camera_list)

    def scan_usb_devices(self):
        self.device_list.clear()
        try:
            devices = list(usb.core.find(find_all=True) or [])
        except Exception as exc:
            self.device_list.addItem(f"USB device scan unavailable: {exc}")
            return

        if not devices:
            self.device_list.addItem("No USB devices found")
            return

        for dev in devices:
            try:
                manufacturer = usb.util.get_string(dev, dev.iManufacturer)
            except Exception:
                manufacturer = "Unknown manufacturer"

            try:
                product = usb.util.get_string(dev, dev.iProduct)
            except Exception:
                product = "Unknown product"

            self.device_list.addItem(
                f"{manufacturer} - {product} "
                f"(Vendor ID: {dev.idVendor:04x}, Product ID: {dev.idProduct:04x})"
            )

    @staticmethod
    def video_number(path):
        match = re.search(r"video(\d+)$", path)
        return int(match.group(1)) if match else 9999

    @staticmethod
    def camera_name(path):
        try:
            with open(
                f"/sys/class/video4linux/{os.path.basename(path)}/name",
                encoding="utf-8"
            ) as f:
                return f.read().strip()
        except Exception:
            return "Unknown camera"

    @staticmethod
    def valid_camera(path):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--device", path, "--info"],
                capture_output=True,
                text=True,
                timeout=2
            )

            output = result.stdout + result.stderr

            if result.returncode != 0:
                return False

            if "Metadata Capture" in output and "Video Capture" not in output:
                return False

        except Exception:
            pass

        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()

        return ok and frame is not None and frame.size > 0

    @staticmethod
    def camera_backends():
        """Return capture backends in the order that works best per platform."""
        if sys.platform == "win32":
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        if sys.platform.startswith("linux"):
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    @classmethod
    def probe_camera(cls, index):
        """Find a backend that can open the camera index and read a frame."""
        for backend in cls.camera_backends():
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    cap.release()
                    return backend
            cap.release()
        return None

    def refresh_camera_list(self):
        if self._camera_scan_thread is not None and self._camera_scan_thread.isRunning():
            return
        self.stop_camera()
        self.camera_combo.clear()
        self.camera_combo.addItem("Buscando cámaras disponibles…", None)
        self.camera_combo.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.video_label.setText("Buscando cámaras disponibles…")

        self._camera_scan_thread = QThread(self)
        self._camera_scan_worker = CameraScanWorker()
        self._camera_scan_worker.moveToThread(self._camera_scan_thread)
        self._camera_scan_thread.started.connect(self._camera_scan_worker.run)
        self._camera_scan_worker.finished.connect(self._camera_scan_finished)
        self._camera_scan_worker.finished.connect(self._camera_scan_thread.quit)
        self._camera_scan_worker.finished.connect(self._camera_scan_worker.deleteLater)
        self._camera_scan_thread.finished.connect(self._camera_scan_thread_finished)
        self._camera_scan_thread.finished.connect(self._camera_scan_thread.deleteLater)
        self._camera_scan_thread.start()

    def _camera_scan_finished(self, cameras):
        self.camera_combo.clear()
        if cameras:
            for label, device in cameras:
                self.camera_combo.addItem(label, device)
            self.camera_combo.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.video_label.setText("Cámara detectada. Pulsa «Start Camera» para ver la imagen.")
        else:
            self.camera_combo.addItem("No se encontraron cámaras disponibles", None)
            self.camera_combo.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.video_label.setText(
                "No se encontraron cámaras. Conecta o activa una cámara y pulsa «Refresh Cameras»."
            )
        self.refresh_btn.setEnabled(True)

    def _camera_scan_thread_finished(self):
        self._camera_scan_worker = None
        self._camera_scan_thread = None
        if self._close_after_camera_scan:
            self._close_after_camera_scan = False
            QTimer.singleShot(0, self.close)

    def start_camera(self):
        self.stop_camera()

        device = self.camera_combo.currentData()

        if not device:
            QMessageBox.warning(
                self,
                "Camera Error",
                "No working camera selected."
            )
            return

        source, backend = device
        self.camera = cv2.VideoCapture(source, backend)

        if not self.camera.isOpened():
            self.camera = None
            QMessageBox.warning(
                self,
                "Camera Error",
                f"Could not open camera: {source}"
            )
            return

        self.prev_sample = None
        self.last_frame = None
        self.last_change = time.monotonic()
        self.frame_times.clear()

        self.video_label.clear()
        self.video_label.setText("Conectando con la cámara…")
        self.timer.start(30)

    def frame_changed(self, frame):
        sample = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        if self.prev_sample is None:
            self.prev_sample = sample
            return True

        diff = cv2.absdiff(sample, self.prev_sample).mean()
        self.prev_sample = sample

        return diff > self.CHANGE_THRESHOLD

    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0.0

        elapsed = self.frame_times[-1] - self.frame_times[0]

        if elapsed <= 0:
            return 0.0

        return (len(self.frame_times) - 1) / elapsed

    def draw_overlay(self, frame, live=True):
        h, w = frame.shape[:2]

        # Fuente fina y discreta
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1

        # ---------------------------------------------------------
        # ESTADO + FPS - ARRIBA IZQUIERDA
        # ---------------------------------------------------------

        color = (0, 220, 0) if live else (0, 0, 255)
        fps = self.get_fps() if live else 0.0

        # Punto de estado
        cv2.circle(
            frame,
            (22, 24),
            6,
            color,
            -1,
            cv2.LINE_AA
        )

        # FPS
        cv2.putText(
            frame,
            f"{fps:.1f} FPS",
            (38, 30),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # ---------------------------------------------------------
        # FECHA Y HORA - ARRIBA DERECHA
        # ---------------------------------------------------------

        date_text = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

        (text_width, _), _ = cv2.getTextSize(
            date_text,
            font,
            font_scale,
            thickness
        )

        cv2.putText(
            frame,
            date_text,
            (w - text_width - 15, 30),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            c * w,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(pixmap)

    def update_frame(self):
        if self.camera is None:
            return

        ok, frame = self.camera.read()
        now = time.monotonic()

        # No llega ningún frame nuevo
        if not ok or frame is None:
            if self.last_frame is not None:
                frame = self.last_frame.copy()
                self.draw_overlay(frame, live=False)
                self.show_frame(frame)
            return

        self.frame_times.append(now)
        self.last_frame = frame.copy()

        # Comprueba si la imagen realmente está cambiando
        if self.frame_changed(frame):
            self.last_change = now

        live = (now - self.last_change) < self.FREEZE_TIMEOUT

        self.draw_overlay(frame, live)
        self.show_frame(frame)

    def stop_camera(self):
        self.timer.stop()

        if self.camera:
            self.camera.release()

        self.camera = None
        self.prev_sample = None
        self.last_frame = None
        self.frame_times.clear()

        self.video_label.clear()
        self.video_label.setText("Camera stopped")

    def closeEvent(self, event):
        if self._camera_scan_thread is not None and self._camera_scan_thread.isRunning():
            self._close_after_camera_scan = True
            event.ignore()
            return
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = USBCameraApp()
    window.show()

    sys.exit(app.exec())
