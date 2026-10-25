"""Party mode motion for ``30_full_demo_app.py``: a slow rise to a standard pose, then a dance inside what was recorded.

``party_profile.json`` (made by ``analyze_party_ranges.py`` from the demonstrations) gives the standard "up" pose
(``home``: the pose the pick demonstrations end in), the envelope the arm has already moved in around it (``low`` /
``high``) and how far each joint may swing (``swing``). Nothing here talks to the robot: both functions return the
joint targets for a given time, so the app can send them with its usual per-command caps.

* ``rise_target``: from wherever the arm is to ``home``, a smoothstep slow enough that its peak speed is ``RISE_SPEED``.
* ``Choreography``: home plus a swing per joint, every one a sine locked to the beat, faded in over ``RAMP_S`` seconds.
  Targets are clamped to the envelope and the swings are sized so no joint exceeds ``MAX_SPEED`` (the demonstrations
  reach 60-97 deg/s at the 95th percentile, so this is slower than the arm normally moves).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

PROFILE_PATH = Path(__file__).with_name("party_profile.json")
BPM = 90.0  # 96 made the gripper pulse peak at 60.3 per second, over MAX_SPEED
RISE_SPEED = 25.0  # deg/s, peak, on the way up
RISE_MIN_S = 5.0
RAMP_S = 4.0  # the swings fade in over this long, so the dance starts from the standard pose
MAX_SPEED = 60.0  # deg/s, any joint, while dancing
# One sine per joint: (cycles per beat, phase in radians), in the joint order of the profile. The lift and the elbow
# bob against each other, the wrist nods on every second beat and the roll turns and the gripper pulses with it.
MOVES = (
    (1 / 8, 0.0),  # shoulder_pan: a sway every eight beats
    (1 / 4, 0.0),  # shoulder_lift
    (1 / 4, math.pi),  # elbow_flex, against the lift
    (1 / 2, math.pi / 2),  # wrist_flex: the nod
    (1 / 8, math.pi / 2),  # wrist_roll
    (1 / 2, 0.0),  # gripper
)


def load_profile(path: Path = PROFILE_PATH) -> dict:
    raw = json.loads(Path(path).read_text())
    profile = {key: np.asarray(raw[key], dtype=np.float32) for key in ("home", "low", "high", "swing")}
    if any(len(value) != 6 for value in profile.values()):
        raise ValueError(f"{path}: home, low, high and swing need six values (five joints and the gripper)")
    if np.any(profile["home"] < profile["low"]) or np.any(profile["home"] > profile["high"]):
        raise ValueError(f"{path}: the home pose is outside the recorded envelope")
    return profile


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def rise_seconds(start, home) -> float:
    """Long enough that the fastest joint peaks at ``RISE_SPEED`` (a smoothstep peaks at 1.5 times its mean speed)."""
    distance = float(np.max(np.abs(np.asarray(home, dtype=np.float32) - np.asarray(start, dtype=np.float32))))
    return max(RISE_MIN_S, 1.5 * distance / RISE_SPEED)


def rise_target(start, home, elapsed: float, seconds: float) -> np.ndarray:
    start, home = np.asarray(start, dtype=np.float32), np.asarray(home, dtype=np.float32)
    return start + (home - start) * smoothstep(elapsed / seconds)


class Choreography:
    def __init__(self, profile: dict, bpm: float = BPM, ramp_s: float = RAMP_S) -> None:
        self.home, self.low, self.high = profile["home"], profile["low"], profile["high"]
        self.swing, self.bpm, self.ramp_s = profile["swing"], bpm, ramp_s
        self.cycles = np.array([cycles for cycles, _ in MOVES], dtype=np.float64)
        self.phase = np.array([phase for _, phase in MOVES], dtype=np.float64)

    def beat(self, seconds: float) -> float:
        return seconds * self.bpm / 60.0

    def peak_speeds(self) -> np.ndarray:
        """Highest speed each joint's sine reaches at full swing, deg/s."""
        return self.swing * 2.0 * math.pi * self.cycles * self.bpm / 60.0

    def target(self, seconds: float) -> np.ndarray:
        fade = smoothstep(seconds / self.ramp_s)
        wave = np.sin(2.0 * math.pi * self.cycles * self.beat(seconds) + self.phase)
        return np.clip(self.home + fade * self.swing * wave, self.low, self.high).astype(np.float32)
