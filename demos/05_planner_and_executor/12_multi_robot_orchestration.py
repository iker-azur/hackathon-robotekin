"""Launcher wrapper for Demo 12 — Multi-Robot Autonomous Orchestration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DEMO12_DIR = Path(__file__).resolve().parent.parent / "07_multi_robot_orchestration"
DEMO12_MAIN = DEMO12_DIR / "12_multi_robot_orchestration.py"

if str(DEMO12_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO12_DIR))

spec = importlib.util.spec_from_file_location("demo12_main", DEMO12_MAIN)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar {DEMO12_MAIN}")
module = importlib.util.module_from_spec(spec)
sys.modules["demo12_main"] = module
spec.loader.exec_module(module)

if __name__ == "__main__":
    raise SystemExit(module.main())
