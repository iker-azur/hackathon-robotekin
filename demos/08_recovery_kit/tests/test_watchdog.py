"""El vigilante detecta saltos de posición y piezas que atraviesan su apoyo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recoverykit.physics import Watchdog  # noqa: E402

DT = 1 / 240


def support(x, y):
    return 0.76 if abs(x) < 0.5 and abs(y) < 0.5 else 0.0


def make():
    return Watchdog(dt=DT, support=support, min_half={"a": 0.0125, "parked": 0.0125})


def test_legit_motion_is_clean():
    w = make()
    p = {"a": np.array([0.2, 0.0, 0.7725]), "parked": np.array([-1.0, 1.5, 0.0125])}
    w.arm(p)
    for _ in range(240):
        p = {"a": p["a"] + np.array([0.5 * DT, 0.0, 0.0]), "parked": p["parked"]}
        w.check(p, {"a": 0.5, "parked": 0.0})
    assert w.violations == []


def test_teleport_and_sinking_are_caught():
    w = make()
    p = {"a": np.array([0.2, 0.0, 0.7725]), "parked": np.array([-1.0, 1.5, 0.0125])}
    w.arm(p)
    w.check({"a": np.array([0.25, 0.0, 0.7725]), "parked": p["parked"]}, {})
    w.check({"a": np.array([0.25, 0.0, 0.765]), "parked": p["parked"]}, {})
    kinds = [v.kind for v in w.violations]
    assert "teleport" in kinds and "penetración" in kinds


def test_disarmed_ignores_reset():
    w = make()
    w.check({"a": np.array([5.0, 0.0, 0.0]), "parked": np.zeros(3)}, {})
    assert w.violations == []
