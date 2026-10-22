#!/usr/bin/env python3
"""PySide6 desktop application entry point (RoboQC), Phase 1."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

try:
    from PySide6.QtWidgets import QApplication
except Exception as e:
    print("ERROR: PySide6 no esta disponible. Instala con: pip install PySide6")
    print(str(e))
    sys.exit(1)

from roboqc.ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    # Resolved from this file, so the app works from any working directory.
    model_dir = os.environ.get("ROBOQC_MODEL_DIR", os.path.join(ROOT, "models"))
    preferred = os.environ.get("ROBOQC_MODEL", None)
    w = MainWindow(model_dir=model_dir, preferred_model=preferred)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
