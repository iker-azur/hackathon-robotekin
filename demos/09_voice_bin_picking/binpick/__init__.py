"""Demo 18 — clasificación de piezas por voz con visión y aprendizaje por imitación.

Este paquete reutiliza el taller físico validado del demo 08 (``recoverykit``:
cinemática, control, agarre por fricción, cámara cenital calibrada). Se añade al
``sys.path`` al final para no tapar nada propio.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
RECOVERYKIT_ROOT = DEMO_ROOT.parent / "08_recovery_kit"

if RECOVERYKIT_ROOT.is_dir() and str(RECOVERYKIT_ROOT) not in sys.path:
    sys.path.append(str(RECOVERYKIT_ROOT))
