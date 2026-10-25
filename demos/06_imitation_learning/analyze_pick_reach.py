"""Measure the pick-cube recordings to set the reach detector and the firm hold of ``30_full_demo_app.py``.

For the 30 fps ``so101-pick-cube_*`` recordings under ``data/real_so101/raw`` this script

* replays every pick through ``reach_detect.ReachDetector`` (with the thresholds below) and through the rule the apps
  used so far (``pose_stop_status``: still for 1.5 s, not before 5 s, near the targets) and reports how many picks each
  one calls "arrived", how long after the arm really arrived, and how often it fired while the arm was still far from
  where the pick ended ("premature"). The truth is the moment the arm gets within 4 degrees of its final pose and stays;
* measures the gap between the target and the measured pose while the piece is held up: that gap is the motor torque,
  and its 99th percentile per joint is the largest hold gap the firm hold may command.

The recordings stop about a second after the arm arrives, so a rule cannot be seen to fire later than that on them:
"fired" is a floor for what a rollout, which can hold as long as it likes, would show.

    python analyze_pick_reach.py [--data data/real_so101/raw] [--out pick_reach_profile.json]

The data folder is not in git: the result is written to ``pick_reach_profile.json`` (committed) for ``reach_detect.py``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FPS = 30
ARM = 5
THRESHOLDS = {  # chosen on the grid in the README: shorter windows fire more but more often too early
    "open_min": 15.0,  # the gripper opened at least this far (points) before closing on the piece
    "drop": 10.0,  # and then fell by this much: 99 % of the recorded picks do, even holding a thick piece
    "lift": 30.0,  # degrees the arm moved after the grasp: the smallest recorded lift is 43
    "window": 0.75,  # seconds the arm must be still
    "motion": 2.0,  # degrees of motion tolerated in that window (10th-90th percentile range)
    "error": 6.0,  # degrees between the policy target and the measured pose (80th percentile)
    "gripper_motion": 5.0,  # gripper points of motion tolerated
}
ARRIVAL_TOLERANCE = 4.0  # degrees from the final pose that count as "there"
PREMATURE = 8.0  # firing this far from the final pose is too early


def load_detector():
    spec = importlib.util.spec_from_file_location("reach_detect", HERE / "reach_detect.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_picks(root: Path):
    """[(states, actions)] per pick episode of at least 3 s, 30 fps repositories only."""
    import pandas as pd

    picks = []
    for repo in sorted(os.listdir(root)):
        folder = root / repo
        info = folder / "meta" / "info.json"
        if not repo.startswith("so101-pick-cube") or not info.is_file() or json.loads(info.read_text()).get("fps") != FPS:
            continue
        for part in sorted(glob.glob(str(folder / "data" / "chunk-000" / "*.parquet"))):
            for _, group in pd.read_parquet(part).groupby("episode_index"):
                states = np.stack(group["observation.state"].to_numpy())
                if len(states) >= 3 * FPS:
                    picks.append((states, np.stack(group["action"].to_numpy())))
    return picks


def arrival_frame(states: np.ndarray) -> int:
    far = np.where(np.abs(states[:, :ARM] - states[-15:, :ARM].mean(0)).max(1) > ARRIVAL_TOLERANCE)[0]
    return int(far[-1] + 1) if len(far) else 0


def old_rule(states, actions, window=1.5, minimum=5.0, arm_motion=2.0, arm_error=6.0, gripper_motion=5.0) -> int:
    """``pose_stop_status`` replayed on a recording: the first frame it would stop at, or -1."""
    size = int(round(window * FPS))
    for t in range(int(minimum * FPS), len(states)):
        if t < size:
            continue
        seen, wanted = states[t - size + 1:t + 1], actions[t - size + 1:t + 1]
        low, high = np.percentile(seen, [10, 90], axis=0)
        motion = np.maximum(high - low, np.abs(seen[-1] - np.median(seen, axis=0)))
        errors = np.abs(wanted[:, :ARM] - seen[:, :ARM])
        error = max(np.percentile(errors, 80, axis=0).max(), errors[-1].max())
        if motion[:ARM].max() <= arm_motion and error <= arm_error and motion[ARM] <= gripper_motion:
            return t
    return -1


def new_rule(module, states, actions) -> int:
    detector = module.ReachDetector({**THRESHOLDS, "hold_gap": np.ones(ARM)})
    for t in range(len(states)):
        if detector.update(t / FPS, states[t], actions[t])[0]:
            return t
    return -1


def evaluate(picks, rule) -> dict:
    delays, premature = [], 0
    for states, actions in picks:
        t = rule(states, actions)
        if t < 0:
            continue
        delays.append((t - arrival_frame(states)) / FPS)
        premature += bool(np.abs(states[t, :ARM] - states[-15:, :ARM].mean(0)).max() > PREMATURE)
    fired = len(delays)
    return {"episodes": len(picks), "fired": fired, "fired_share": round(fired / len(picks), 3),
            "premature": premature, "premature_share_of_fired": round(premature / max(fired, 1), 3),
            "delay_p50_s": round(float(np.median(delays)), 2) if delays else None,
            "delay_p90_s": round(float(np.percentile(delays, 90)), 2) if delays else None}


def hold_gaps(picks) -> dict:
    gaps = np.array([(a[-12:] - s[-12:]).mean(0) for s, a in picks])
    rounded = lambda values: [round(float(v), 2) for v in values]
    return {"signed_p50": rounded(np.percentile(gaps, 50, 0)), "signed_p5": rounded(np.percentile(gaps, 5, 0)),
            "abs_p99": rounded(np.percentile(np.abs(gaps), 99, 0))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "real_so101" / "raw")
    parser.add_argument("--out", type=Path, default=HERE / "pick_reach_profile.json")
    args = parser.parse_args()
    picks = load_picks(args.data)
    module = load_detector()
    gaps = hold_gaps(picks)
    thresholds = {**THRESHOLDS, "hold_gap": [math.ceil(v * 2) / 2 for v in gaps["abs_p99"][:ARM]]}
    profile = {
        "source": {"pick_episodes": len(picks), "fps": FPS, "frames": int(sum(len(s) for s, _ in picks))},
        "thresholds": thresholds,
        "evaluation": {"today": evaluate(picks, old_rule), "new": evaluate(picks, lambda s, a: new_rule(module, s, a))},
        "hold_gap_stats": gaps,
    }
    args.out.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
