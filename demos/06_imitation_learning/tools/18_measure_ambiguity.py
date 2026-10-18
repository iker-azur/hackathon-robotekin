"""Measure whether the expert's plan is learnable from a single observation.

Why this exists
---------------

An imitation policy is shown one observation and asked for one action. That only
works if the expert's action is a **function** of what the policy can see. When
it is not, the policy is being asked for two different answers to the same
question, and the best it can do is average them - which is a target that goes
nowhere, and looks exactly like a policy that has frozen.

This measures that directly, from a recorded dataset, without a simulator and
without a training run:

* **ambiguous pairs** - frames in two different phases whose arm poses are within
  1 degree of each other while their commanded targets differ by more than 1 cm.
  Each one is a question with two right answers.
* **revisit** - the worst single collision, so a fix can be checked against the
  specific frames that caused it.

Run:

    python .\\demos\\06_imitation_learning\\tools\\18_measure_ambiguity.py
    python .\\demos\\06_imitation_learning\\tools\\18_measure_ambiguity.py --dataset so101_lift_cube_demos

A plan that scores 0 here cannot be learned by any single-observation policy,
however much data or training it is given. Fix ``expert.py`` and measure again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import expert as expert_module  # noqa: E402
import il_dataset  # noqa: E402

# Two frames count as the same pose within this many degrees of motor travel.
SAME_POSE_DEG = 1.0
# ...and their targets count as different beyond this many metres.
DIFFERENT_TARGET_M = 0.01


def phase_bounds(frames_per_episode: int | None = None) -> list[tuple[int, int, str]]:
    """Phase boundaries as ``(first, last, name)``, clipped to what was recorded.

    ``expert_module.PLAN`` is the *current* plan, but the dataset on disk may have
    been recorded by an earlier one - and the two have different lengths (the plan
    below runs 660 steps, an earlier one 640). Reading the bounds off the recorded
    frame count rather than off the plan keeps the tool honest about whichever
    dataset it is pointed at, instead of indexing past the end of an episode.
    """
    rows, start = [], 0
    for phase in expert_module.PLAN:
        end = start + phase.steps
        if frames_per_episode is not None and start >= frames_per_episode:
            break
        rows.append((start, min(end, frames_per_episode), phase.name))
        start = end
    return rows


def measure(dataset_root: Path, episodes: int | None = None, verbose: bool = True):
    import pandas as pd

    parquet = (
        dataset_root / "data" / "chunk-000" / "file-000.parquet"
    )
    if not parquet.exists():
        raise SystemExit(f"ERROR: no dataset at {dataset_root}")

    frame = pd.read_parquet(parquet)
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(frame["action"].to_numpy()).astype(np.float32)
    episode_index = frame["episode_index"].to_numpy()

    wanted = sorted(set(episode_index))[:episodes] if episodes else sorted(set(episode_index))
    frames_per_episode = int((episode_index == wanted[0]).sum())
    bounds = phase_bounds(frames_per_episode)

    total_ambiguous = 0
    worst = (0.0, None)
    per_episode = []

    for episode in wanted:
        mask = episode_index == episode
        poses = state[mask]
        targets = action[mask][:, :3]
        ambiguous = 0

        for i, (first, last, _) in enumerate(bounds):
            for other_first, other_last, _ in bounds[i + 1 :]:
                here = poses[first:last]
                there = poses[other_first:other_last]
                distance = np.linalg.norm(here[:, None, :] - there[None, :, :], axis=2)
                close = distance < SAME_POSE_DEG
                if not close.any():
                    continue
                here_targets = targets[first:last]
                there_targets = targets[other_first:other_last]
                gap = np.linalg.norm(
                    here_targets[:, None, :] - there_targets[None, :, :], axis=2
                )
                conflicting = (gap > DIFFERENT_TARGET_M) & close
                count = int(conflicting.sum())
                ambiguous += count
                if count:
                    worst_gap = float(gap[conflicting].max())
                    if worst_gap > worst[0]:
                        i_flat, j_flat = np.unravel_index(
                            np.argmax(np.where(conflicting, gap, -1)), conflicting.shape
                        )
                        worst = (
                            worst_gap,
                            (
                                episode,
                                first + int(i_flat),
                                bounds[i][2],
                                other_first + int(j_flat),
                                bounds[i + 1][2],
                            ),
                        )

        total_ambiguous += ambiguous
        per_episode.append(ambiguous)

    if verbose:
        names = [name for _, _, name in bounds]
        print(f"dataset: {dataset_root.name}   episodes measured: {len(wanted)}")
        print(f"phases:   {', '.join(names)}")
        print("")
        print(f"pose tolerance      : {SAME_POSE_DEG} deg of motor travel")
        print(f"target tolerance    : {DIFFERENT_TARGET_M * 1000:.0f} mm")
        print("")
        print(f"ambiguous pairs per episode: "
              f"{np.mean(per_episode):.0f} average, {max(per_episode)} worst")
        print(f"total ambiguous pairs      : {total_ambiguous}")
        if worst[1]:
            episode, frame_a, phase_a, frame_b, phase_b = worst[1]
            print("")
            print(f"worst collision: {worst[0] * 100:.1f} mm apart")
            print(f"  episode {episode}, frame {frame_a} in '{phase_a}' "
                  f"vs frame {frame_b} in '{phase_b}'")
            print("  same arm pose, different commanded target: the policy cannot")
            print("  know which phase it is in from the observation alone.")
        print("")
        verdict = "LEARNABLE from a single observation" if total_ambiguous == 0 else (
            "NOT learnable from a single observation - fix expert.py"
        )
        print(verdict)
    return total_ambiguous, np.mean(per_episode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="so101_lift_cube_demos_v2")
    parser.add_argument("--episodes", type=int, default=None,
                        help="measure only the first N episodes (faster)")
    args = parser.parse_args()

    root = il_dataset.default_dataset_dir(args.dataset)
    try:
        total, average = measure(root, episodes=args.episodes)
    except SystemExit as failure:
        print(failure)
        return 1
    return 0 if total == 0 else 0  # a non-zero count is a finding, not a tool failure


if __name__ == "__main__":
    raise SystemExit(main())