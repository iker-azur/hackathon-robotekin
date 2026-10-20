"""Look inside a demonstration dataset: what the robot saw and what it did.

This is step 0 of the playground. Before training anything it is worth knowing
what is actually in a dataset - what the columns mean, what a typical trajectory
looks like, and whether the demonstrations vary enough to teach something.

Run:

    python .\\demos\\06_imitation_learning\\12_inspect_demonstrations.py

Needs a dataset. Collect one with demo 13, or point ``--dataset`` at any
LeRobot dataset on disk. The script prints:

* the dataset's own metadata: features, frames, episodes, fps
* per-episode statistics from the recorded frames
* a summary of the action distribution: how much of each joint's range the
  demonstrations actually use, which is the quickest way to spot a dataset that
  is all one static pose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def episode_bounds(dataset) -> list[tuple[int, int]]:
    """(first, last) frame index of every episode, from the dataset's own metadata."""
    import glob

    import pyarrow.parquet as pq

    pattern = str(
        Path(dataset.root) / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    matches = glob.glob(pattern)
    if not matches:
        matches = sorted(glob.glob(str(Path(dataset.root) / "meta" / "*" / "*.parquet")))
    if not matches:
        return []

    table = pq.read_table(matches[0], columns=["dataset_from_index", "dataset_to_index"])
    starts = table.column("dataset_from_index").to_pylist()
    ends = table.column("dataset_to_index").to_pylist()
    return list(zip(starts, ends))


def describe_features(dataset, verbose: bool) -> None:
    print("features")
    for key, feature in dataset.features.items():
        shape = getattr(feature, "shape", None)
        dtype = getattr(feature, "dtype", None)
        info = f"{key}: dtype={dtype} shape={shape}"
        if verbose:
            names = feature.get("names")
            if names:
                info += f" names={names}"
        print(f"  {info}")


def summarise(dataset, episodes: int) -> None:
    import numpy as np

    total = len(dataset)
    print("dataset")
    print(f"  frames: {total}, episodes: {dataset.num_episodes}, fps: {dataset.fps}")
    print(f"  robot type: {dataset.meta.robot_type}")
    print(f"  camera(s): {dataset.meta.camera_keys}")
    print(f"  task(s): {dataset.meta.tasks}")

    if total == 0:
        print("")
        print("The dataset is empty. Collect some demonstrations with demo 13 first.")
        return

    bounds = episode_bounds(dataset)
    print("")
    for episode, (first, last) in enumerate(bounds[:episodes]):
        actions = np.stack([dataset[i]["action"].numpy() for i in range(first, last)])
        duration = (last - first) / dataset.fps
        movement = float(np.linalg.norm(actions[-1] - actions[0])) if last > first else 0.0
        distinct = len({tuple(np.round(action, 1)) for action in actions})
        print(
            f"  episode {episode}: {last - first} frames, {duration:.1f}s, "
            f"first-to-last action change {movement:.1f}, distinct actions {distinct}"
        )
    print("")

    # The ranges say whether the demonstrations actually move the arm. A narrow
    # band on a joint means the demonstrations never used it - the classic symptom
    # of a dataset recorded with the robot parked.
    states = np.stack([dataset[i]["observation.state"].numpy() for i in range(total)])
    actions = np.stack([dataset[i]["action"].numpy() for i in range(total)])
    print("state ranges, in motor degrees")
    print(f"  {'joint':<16}{'min':>10}{'max':>10}")
    state_names = dataset.features["observation.state"]["names"] or [
        f"state_{i}" for i in range(states.shape[1])
    ]
    for index, name in enumerate(state_names):
        print(f"  {name:<16}{states[:, index].min():>10.1f}{states[:, index].max():>10.1f}")

    # The action is a different quantity from the state, so it gets its own table
    # rather than sharing one with it. Printing them side by side invites reading
    # a control-space number as if it were a joint angle.
    print("")
    print("action ranges (the dataset's own control space)")
    print(f"  {'dimension':<16}{'min':>10}{'max':>10}")
    action_names = dataset.features["action"]["names"] or [
        f"action_{i}" for i in range(actions.shape[1])
    ]
    for index, name in enumerate(action_names):
        print(f"  {name:<16}{actions[:, index].min():>10.3f}{actions[:, index].max():>10.3f}")

    # The fastest way to see whether a dataset contains any motion at all.
    print("")
    print("share of steps in which each action dimension changes by less than 0.001")
    for index, name in enumerate(action_names):
        if actions.shape[0] > 1:
            steps = np.abs(np.diff(actions[:, index]))
            print(f"  {name:<16}{float((steps < 0.001).mean()) * 100:>9.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="so101_lift_cube_physical",
                        help="dataset directory name under data/, or a hub repo id")
    parser.add_argument("--episodes", type=int, default=3, help="episodes to summarise")
    parser.add_argument("--verbose", action="store_true", help="print feature names too")
    args = parser.parse_args()

    import il_dataset as data

    root = data.default_dataset_dir(args.dataset)
    hub = args.dataset.count("/") == 1
    if hub or not root.exists():
        where = "the Hugging Face Hub" if hub else str(root)
        print(f"loading {args.dataset} from {where}")
        dataset = data.load_dataset(args.dataset, root=None if hub else root)
    else:
        dataset = data.load_dataset(args.dataset, root=root)

    print(f"dataset: {dataset.repo_id}")
    print("")
    describe_features(dataset, args.verbose)
    print("")
    try:
        summarise(dataset, args.episodes)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
