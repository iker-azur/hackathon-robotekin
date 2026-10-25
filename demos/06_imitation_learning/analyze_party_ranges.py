"""Derive the party-mode envelope of ``30_full_demo_app.py`` from the recorded SO-101 demonstrations.

The dance must not leave what the arm has already done safely. From the 30 fps recordings under
``data/real_so101/raw`` this script finds

* the "up" pose: the pose the pick demonstrations end in (the piece raised to the camera), as a real demonstrated
  pose (the episode closest to the median) and then the median of every frame near it (the *raised family*);
* the envelope: for the arm joints and the gripper, the 2nd-98th percentile that family reached; for the wrist roll,
  the 1st-99th percentile of all frames (the roll spins without moving the arm, and the raised family is bimodal);
* the dance amplitude per joint: a nominal swing, cut to stay inside the envelope with a margin;
* how far the straight joint-space rise from the rest pose passes from any demonstrated pose, next to how far the
  demonstrations themselves are from each other, and the speeds the follower reached.

The data folder is not in git, so the result is written to ``party_profile.json`` (committed) and read by
``party_dance.py``.

    python analyze_party_ranges.py [--data data/real_so101/raw] [--out party_profile.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NOMINAL_SWING = [14.0, 10.0, 12.0, 10.0, 35.0, 12.0]  # degrees (percentage points for the gripper)
MARGIN = 2.0  # the dance stays this far inside the envelope
FAMILY_WINDOW = (12.0, 15.0, 15.0)  # lift, elbow, wrist flex: how near the up pose counts as "raised"
ARM = slice(0, 5)


def load_states(root: Path):
    """[(task, states)] per episode, 30 fps repositories only (the 10 fps ones add nothing)."""
    import pandas as pd

    episodes = []
    for repo in sorted(os.listdir(root)):
        folder = root / repo
        info = folder / "meta" / "info.json"
        if not info.is_file() or json.loads(info.read_text()).get("fps") != 30:
            continue
        for part in sorted(glob.glob(str(folder / "data" / "chunk-000" / "*.parquet"))):
            frame = pd.read_parquet(part)
            for _, group in frame.groupby("episode_index"):
                episodes.append((repo.split("_")[0], np.stack(group["observation.state"].to_numpy())))
    return episodes


def analyze(episodes) -> dict:
    from scipy.spatial import cKDTree

    states = np.concatenate([s for _, s in episodes])
    picks = [s for task, s in episodes if task == "so101-pick-cube"]
    rest = np.median([s[0] for s in picks], axis=0)
    held = np.array([s[-15:].mean(0) for s in picks])  # the pose each pick ends in
    median = np.median(held[:, :4], axis=0)
    medoid = held[np.argmin(np.abs(held[:, :4] - median).sum(1))]

    window = np.array(FAMILY_WINDOW)
    family = states[np.all(np.abs(states[:, 1:4] - medoid[1:4]) <= window, axis=1)]
    low, high = np.percentile(family, 2, 0), np.percentile(family, 98, 0)
    low[4], high[4] = np.percentile(states[:, 4], [1, 99])
    home = np.median(family, axis=0)
    home[4] = rest[4]  # neutral twist: the roll is a free joint and its raised-family range is bimodal
    home[5] = (low[5] + high[5]) / 2
    room = np.minimum(home - low, high - home) - MARGIN
    swing = np.clip(np.minimum(NOMINAL_SWING, room), 0.0, None)

    tree = cKDTree(states[::2, ARM])
    reference, _ = tree.query(states[1::2][::20, ARM])  # how far a demonstrated frame is from other demonstrations
    path = np.array([rest + (home - rest) * k for k in np.linspace(0, 1, 101)])
    along, _ = tree.query(path[:, ARM])
    speed = np.concatenate([np.abs(s[3:] - s[:-3]) / 0.1 for _, s in episodes])  # deg/s over 0.1 s

    def rounded(values, digits=1):
        return [round(float(v), digits) for v in values]

    return {
        "source": {"frames": int(len(states)), "episodes": len(episodes), "fps": 30,
                   "pick_episodes": len(picks), "raised_family_frames": int(len(family))},
        "joints": JOINTS,
        "rest": rounded(rest),
        "home": rounded(home),
        "low": rounded(low),
        "high": rounded(high),
        "swing": rounded(swing),
        "speed_p50": rounded(np.percentile(speed, 50, 0)),
        "speed_p95": rounded(np.percentile(speed, 95, 0)),
        "rise_path": {"max_distance_to_a_demonstrated_pose": round(float(along.max()), 1),
                      "demonstrated_frames_p99": round(float(np.percentile(reference, 99)), 1),
                      "home_distance_to_a_demonstrated_pose": round(float(tree.query(home[ARM][None])[0][0]), 1)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "real_so101" / "raw")
    parser.add_argument("--out", type=Path, default=HERE / "party_profile.json")
    args = parser.parse_args()
    profile = analyze(load_states(args.data))
    args.out.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
