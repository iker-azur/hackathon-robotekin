#!/usr/bin/env python3
"""RoboQC demo 08: Phase 1 and Phase 2 in one window, one tab each.

Tab 1 "Fase 1" is the Phase 1 command window (same as phase_1/04_app.py).
Tab 2 "Fase 2" is the inspection workstation (camera, inventory, commands).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
except Exception as e:
    print("ERROR: PySide6 no esta disponible. Instala con: pip install PySide6")
    print(str(e))
    sys.exit(1)

from roboqc.ui.main_window import MainWindow
from roboqc.ui.workstation import WorkstationWindow
from roboqc.vision.calibration import load_calibration
from roboqc.vision.config import DEFAULT_CONFIG_PATH, load_config


class CommandWindow(MainWindow):
    """Phase 1 window unchanged. Every classified command is also shown in the Fase 2 result panel."""

    def __init__(self, phase2: WorkstationWindow, **kwargs):
        super().__init__(**kwargs)
        self._phase2 = phase2

    def _on_classify_finished(self, cmd, topk):
        super()._on_classify_finished(cmd, topk)
        self._phase2.show_result(cmd)


class HostWindow(QMainWindow):
    """Hosts both windows as tabs. Each child window is closed with the host."""

    def __init__(self, phase1: QMainWindow, phase2: QMainWindow):
        super().__init__()
        self.setWindowTitle("RoboQC")
        self.resize(1280, 860)
        self._children = (phase1, phase2)
        tabs = QTabWidget()
        for widget, title in ((phase1, "Comandos"), (phase2, "Inspeccion")):
            widget.setWindowFlags(Qt.Widget)  # embed as a plain widget, not a separate window
            tabs.addTab(widget, title)
        self.setCentralWidget(tabs)

    def closeEvent(self, event) -> None:
        for widget in self._children:
            widget.close()  # runs each window's own cleanup (camera thread, workers)
        super().closeEvent(event)


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboQC: Fase 1 y Fase 2 en una ventana con pestanas.")
    ap.add_argument("--source", default=None,
                    help="Camara (indice), video, imagen, sim:escena o sim3d:escena. Por defecto: camera.index de la configuracion.")
    ap.add_argument("--config", default=None, help="Fichero de configuracion de vision JSON.")
    ap.add_argument("--calibration", default=None, help="JSON de calibracion opcional.")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    model_dir = os.environ.get("ROBOQC_MODEL_DIR", os.path.join(ROOT, "..", "phase_1", "models"))
    preferred = os.environ.get("ROBOQC_MODEL", None)

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH)
    source = args.source if args.source is not None else cfg["camera"]["index"]
    calibration = load_calibration(args.calibration or cfg.get("calibration_path"))

    phase2 = WorkstationWindow(source, cfg, calibration, model_dir=model_dir, preferred_model=preferred,
                               commands_enabled=False)
    phase1 = CommandWindow(phase2, model_dir=model_dir, preferred_model=preferred)
    host = HostWindow(phase1, phase2)
    host.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
