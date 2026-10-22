"""Comprueba un dataset de demostraciones antes de entrenar. NO necesita Isaac ni GPU.

    python tools/validate_dataset.py data/bin_pick_v1

Claves y formas, NaN, unidades de LeRobot, que las acciones se mueven y no copian el estado, marcas de tiempo a 30 Hz,
reparto de capas y tasa de éxito del experto (de ``meta/binpick_episodes.jsonl``). Sale con código 1 si hay problemas.
Lee ``meta/info.json``, ``meta/binpick_info.json`` y ``data/*/*.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import binpick  # noqa: E402,F401
from binpick import dataset_check as DC  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rep = DC.check(sys.argv[1])
    print(DC.render_report(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
