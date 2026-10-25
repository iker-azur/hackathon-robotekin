"""Has the pick arrived? Reach detection and firm hold for the pick policy of ``30_full_demo_app.py``.

``24_pick_and_classify_app.pose_stop_status`` only asks "has the arm stopped near the policy targets for 1.5 s, after
5 s?". Replayed on the 394 recorded picks it fires on 60 of them (15 %); the rest would wait for the time limit. This
detector asks what a pick looks like instead (``analyze_pick_reach.py`` measures it on the recordings):

1. the gripper opens and then closes on the piece: it fell by ``drop`` points from an opening of at least ``open_min``
   (a gripper that opens again starts the question over: a new grasp);
2. the arm lifted at least ``lift`` degrees away from where it grasped;
3. the arm has been still for ``window`` seconds and is where the policy wants it (``motion`` / ``error`` /
   ``gripper_motion``: the same robust measures ``pose_stop_status`` uses).

Stillness alone (what ``pose_stop_status`` checks) also fires while the arm hovers or lingers over the tray before it
has the piece; the first two conditions are what make an early decision safe. ``hold_target`` is the firm hold: the
demonstrations hold the piece with the target a few degrees past the measured pose (that gap is the motor torque), while
commanding the measured pose, as the app used to, holds with none. It returns the policy's own target over the still
window, clipped to the gap the demonstrations never exceeded.

Nothing here talks to the robot.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

PROFILE_PATH = Path(__file__).with_name("pick_reach_profile.json")
ARM = 5  # joints before the gripper


def load_profile(path: Path = PROFILE_PATH) -> dict:
    """Every threshold the detector and the app's default settings use, validated."""
    thresholds = json.loads(Path(path).read_text())["thresholds"]
    profile = {key: float(thresholds[key]) for key in
               ("open_min", "drop", "lift", "window", "motion", "error", "gripper_motion")}
    profile["hold_gap"] = np.asarray(thresholds["hold_gap"], dtype=np.float32)
    if len(profile["hold_gap"]) != ARM or np.any(profile["hold_gap"] <= 0):
        raise ValueError(f"{path}: hold_gap needs {ARM} positive values, one per arm joint")
    if min(profile["window"], profile["lift"], profile["drop"]) <= 0:
        raise ValueError(f"{path}: window, lift and drop must be positive")
    return profile


class ReachDetector:
    def __init__(self, profile: dict, window=None, motion=None, error=None, gripper_motion=None, min_time=0.0) -> None:
        """The profile gives the pick pattern; the keyword arguments override the stillness limits (the app's
        "Parada automática" settings) and ``min_time`` is the earliest second a reach may be declared."""
        self.open_min, self.drop, self.lift = profile["open_min"], profile["drop"], profile["lift"]
        self.hold_gap = np.asarray(profile["hold_gap"], dtype=np.float32)
        self.window = profile["window"] if window is None else window
        self.motion = profile["motion"] if motion is None else motion
        self.error = profile["error"] if error is None else error
        self.gripper_motion = profile["gripper_motion"] if gripper_motion is None else gripper_motion
        self.min_time = min_time
        self.reset()

    def reset(self) -> None:
        self.samples: deque = deque()
        self.peak, self.low = 0.0, 0.0
        self.grasp_pose = None
        self.displacement, self.lifted = 0.0, False

    # -- the pattern ------------------------------------------------------------------------------------------------

    def _track(self, state: np.ndarray) -> None:
        gripper = float(state[ARM])
        if self.grasp_pose is None:
            self.peak = max(self.peak, gripper)
            if self.peak >= self.open_min and self.peak - gripper >= self.drop:
                self.grasp_pose, self.low = state[:ARM].copy(), gripper
        else:
            self.low = min(self.low, gripper)
            if gripper - self.low >= self.drop:  # it opened again: that grasp was not it
                self.grasp_pose, self.peak = None, gripper
        if self.grasp_pose is None:
            self.displacement, self.lifted = 0.0, False
        else:
            self.displacement = float(np.abs(state[:ARM] - self.grasp_pose).max())
            self.lifted = self.displacement >= self.lift

    def update(self, elapsed: float, state, target) -> tuple[bool, str]:
        """Feed one control step (seconds since the pick started, measured joints, raw policy target)."""
        state, target = np.asarray(state, dtype=np.float32), np.asarray(target, dtype=np.float32)
        if not (np.isfinite(state).all() and np.isfinite(target).all()):
            return False, "Alcance: medida o predicción no válida"
        self.samples.append((elapsed, state, target))
        while len(self.samples) > 1 and self.samples[1][0] <= elapsed - self.window:
            self.samples.popleft()
        self._track(state)
        if elapsed < self.min_time:
            return False, f"Alcance: se activa en {self.min_time - elapsed:.1f} s"
        if self.grasp_pose is None:
            return False, "Alcance: esperando el agarre (pinza abierta y luego cerrada)"
        if not self.lifted:
            return False, f"Alcance: esperando a que levante la pieza ({self.displacement:.0f}/{self.lift:g}°)"
        if len(self.samples) < 3 or elapsed - self.samples[0][0] < self.window:
            return False, "Alcance: pieza levantada; comprobando que el brazo se ha parado"
        states = np.array([s for _, s, _ in self.samples])
        targets = np.array([t for _, _, t in self.samples])
        low, high = np.percentile(states, [10, 90], axis=0)  # central ranges tolerate an isolated spike
        motion = np.maximum(high - low, np.abs(states[-1] - np.median(states, axis=0)))
        arm_motion, grip_motion = float(motion[:ARM].max()), float(motion[ARM])
        errors = np.abs(targets[:, :ARM] - states[:, :ARM])
        arm_error = float(max(np.percentile(errors, 80, axis=0).max(), errors[-1].max()))
        reached = (arm_motion <= self.motion and arm_error <= self.error and grip_motion <= self.gripper_motion)
        return reached, (f"Alcance: movimiento {arm_motion:.1f}/{self.motion:g}°; distancia al objetivo "
                         f"{arm_error:.1f}/{self.error:g}°; pinza {grip_motion:.1f}/{self.gripper_motion:g}%")

    # -- the firm hold ----------------------------------------------------------------------------------------------

    def hold_target(self, state) -> np.ndarray:
        """Arm targets that hold the piece with the torque the demonstrations used: the policy's median target over
        the still window, never further from the measured pose than the demonstrated hold gap."""
        state = np.asarray(state, dtype=np.float32)[:ARM]
        if not self.samples:
            return state
        median = np.median(np.array([t[:ARM] for _, _, t in self.samples]), axis=0)
        return (state + np.clip(median - state, -self.hold_gap, self.hold_gap)).astype(np.float32)
