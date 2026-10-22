"""Hace importable ``binpick`` (y, por él, ``recoverykit`` del demo 08) y ``synth`` desde pytest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import pytest


@pytest.fixture(scope="session")
def recorded_dataset(tmp_path_factory):
    """Dataset REAL de LeRobot con 2 episodios, grabado en el simulador cinemático (se comparte entre módulos de prueba)."""
    pytest.importorskip("lerobot")
    from binpick import collect as COL, config as C, recorder as R
    from binpick.robot_port import SimRobot
    from binpick.telemetry import BinTelemetry
    from fake_cell import FakeBinCell

    cfg = C.BinPickCfg()
    root = tmp_path_factory.mktemp("recorded") / "ds"
    fb = FakeBinCell(cfg)
    robot = SimRobot(fb, BinTelemetry(), "hsv", realtime=0.0)
    robot.live_frames = False
    ds = R.open_dataset(root, "binpick_shared", robot.spec, resume=False, threads=0)
    R.write_info(root, robot.spec, "hsv", cfg, R.NoiseCfg(0.02, 0.5))
    COL.collect(robot, root, ds, cfg, episodes=2, first_seed=30, noise=R.NoiseCfg(0.02, 0.5), say=lambda m: None)
    ds.finalize()
    return root, robot
