"""Calibration sweep for the cube pick-and-place.

Runs demo 05 headless over a small grid of grasp aims and read-out the result
line. Each simulation run takes about 2-4 minutes; a Kit boot crash (the
flaky 'CUDA error: an illegal memory access' / 'access violation' failing the
environment build) is retried once per combo before the combo is recorded
as CRASHED.

Usage:

    python .\\demos\\01_simulation_basics\\sweep_pick_offsets.py

Results are appended to sweep_results.tsv next to this file and printed live.
The sweep stops as soon as a combo reports PICKED AND PLACED.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
from pathlib import Path

PYTHON_EXE = sys.executable
DEMO_SCRIPT = Path(__file__).resolve().parent / "05_cube_pick_and_place.py"
RESULTS_PATH = Path(__file__).resolve().parent / "sweep_results.tsv"
RUN_TIMEOUT_S = 900
RETRIES_PER_COMBO = 2

# (pick_z_offset metres, pick_y_offset metres, grasp_hold steps). The z grid
# sits around the measured sweet spot (shipped +0.1 lowered by ~2.5-4 cm puts
# the closing fingertips at cube height); the hold gives the slow gripper
# servo (effort-limited to mass/0.15) enough travel time to the -0.17 stop.
GRID = tuple(itertools.product((0.025, 0.03, 0.04), (0.0, -0.01), (100, 200)))

RESULT_RE = re.compile(r"RESULT\s*:\s*(.+)")


def run_combo(python_exe: str, script: Path, pick_z: float, pick_y: float, hold: int) -> str:
    """Run demo 05 with the combo and return its parsed result verdict."""
    cmd = [
        python_exe,
        str(script),
        "--headless",
        "--pick-z-offset",
        str(pick_z),
        "--pick-y-offset",
        str(pick_y),
        "--grasp-hold",
        str(hold),
    ]
    print(f"  running: {' '.join(cmd)}", flush=True)
    try:
        for attempt in range(RETRIES_PER_COMBO):
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
            match = RESULT_RE.search(proc.stdout)
            if match:
                return match.group(1).strip()
            print(f"  sm crashed (attempt {attempt + 1}), retrying...", flush=True)
        return "CRASHED"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def main() -> int:
    with RESULTS_PATH.open("a", encoding="utf-8") as sink:
        sink.write("# pick_z\tpick_y\tgrasp_hold\tresult\n")
        for pick_z, pick_y, hold in GRID:
            result = run_combo(PYTHON_EXE, DEMO_SCRIPT, pick_z, pick_y, hold)
            line = f"{pick_z}\t{pick_y}\t{hold}\t{result}"
            print(f"  -> {line}", flush=True)
            sink.write(line + "\n")
            sink.flush()
            if "PICKED AND PLACED" in result:
                print("\nSUCCESS combo found.", flush=True)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
