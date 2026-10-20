"""Build the training set for the clock + cube policy from a recorded dataset.

Demo 13 records, on every frame, the plan clock and the cube's pixel position
(``observation.plan_clock``, ``observation.cube_uv``). This stacks them into the
one auxiliary input LeRobot's ACT has, ``observation.environment_state``::

    [plan_clock, cube_u, cube_v]        each scaled to [-1, 1]

and the policy is trained on **only** that. No camera image goes in, and the joint
state does not either, so the policy cannot read the arm's own pose - the shortcut
that made the earlier camera + joint-state policies drift off the plan
(``results/diagnosis.md``). A side effect is that there is no video to decode, so
training takes minutes instead of the better part of an hour.

One honest wrinkle: the copy still *contains* the ``observation.state`` column.
LeRobot 0.4.4's ACT reads ``batch["observation.state"].device`` even when no robot
state is configured (``modeling_act.py``, in both the VAE and non-VAE branches), so
a batch without that key crashes. Demo 15 therefore declares the environment state
as the policy's only input feature (the state is in the batch, not in the network),
and demo 16 hands the policy a zero placeholder under that key.

The recorded dataset is not touched; the copy is written next to it, and an
existing copy is never overwritten.

Run from the repository root:

    python .\\demos\\06_imitation_learning\\tools\\21_build_clock_cube_dataset.py ^
        --source so101_lift_cube_physical --output so101_lift_cube_clock_cube
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import il_dataset  # noqa: E402

ENV_STATE_KEY = il_dataset.ACT_ENV_STATE_KEY
# ``observation.state`` stays only so the batch has the key LeRobot's ACT reads the
# device from; it is not a policy input (see the module docstring).
KEPT_FEATURES = (
    "action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index",
)


def column_matrix(table: pa.Table, name: str, width: int) -> np.ndarray:
    """A parquet column as an (rows, width) float32 array, whether stored as a list or a scalar."""
    values = table[name].combine_chunks()
    if pa.types.is_fixed_size_list(values.type) or pa.types.is_list(values.type):
        array = np.asarray(values.to_pylist(), dtype=np.float32)
    else:
        array = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return array.reshape(len(table), width)


def stats_for(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def build(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"no dataset at {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    # Everything except the pictures: the copy never needs them.
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("videos", "images", "*.jsonl"))

    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for key in (il_dataset.PLAN_CLOCK_KEY, il_dataset.CUBE_UV_KEY):
        if key not in info["features"]:
            raise ValueError(f"{source.name} has no {key}; record it with demo 13 first")

    shards = sorted((output / "data").glob("*/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no frame parquet under {output / 'data'}")

    stacked_all, action_all = [], []
    for shard in shards:
        table = pq.read_table(shard)
        clock = column_matrix(table, il_dataset.PLAN_CLOCK_KEY, 1)
        uv = column_matrix(table, il_dataset.CUBE_UV_KEY, 2)
        stacked = np.concatenate([clock, uv], axis=1).astype(np.float32)
        stacked_all.append(stacked)
        action_all.append(column_matrix(table, "action", info["features"]["action"]["shape"][0]))

        keep = [name for name in table.column_names if name in KEPT_FEATURES]
        table = table.select(keep)
        env_state = pa.FixedSizeListArray.from_arrays(
            pa.array(stacked.reshape(-1), type=pa.float32()), list_size=3
        )
        table = table.append_column(ENV_STATE_KEY, env_state)
        pq.write_table(table, shard, compression="zstd")

    stacked_all = np.concatenate(stacked_all)
    action_all = np.concatenate(action_all)

    features = {name: spec for name, spec in info["features"].items() if name in KEPT_FEATURES}
    features[ENV_STATE_KEY] = {
        "dtype": "float32",
        "shape": [3],
        "names": il_dataset.ENV_KINDS["clock_cube"],
    }
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    stats_path = output / "meta" / "stats.json"
    old = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = {name: value for name, value in old.items() if name in KEPT_FEATURES}
    stats[ENV_STATE_KEY] = stats_for(stacked_all)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")

    for episodes_path in (output / "meta" / "episodes").glob("*/*.parquet"):
        episodes = pq.read_table(episodes_path)
        stale = [n for n in episodes.column_names
                 if n.startswith("stats/") and n.split("/")[1] not in features]
        stale += [n for n in episodes.column_names if n.startswith("videos/")]
        if stale:
            pq.write_table(episodes.drop(stale), episodes_path, compression="zstd")

    (output / "meta" / il_dataset.INPUTS_FILE).write_text(
        json.dumps(
            {
                "env_state": "clock_cube",
                "names": il_dataset.ENV_KINDS["clock_cube"],
                "source": source.name,
                "note": "plan_clock is PRIVILEGED (a real robot is not told where it is in the "
                        "plan); cube_u/cube_v come from the colour detector, once, at the start.",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(stacked_all)} rows to {output}")
    print(f"  environment_state range: min {stacked_all.min(0).round(3).tolist()} "
          f"max {stacked_all.max(0).round(3).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="so101_lift_cube_physical",
                        help="dataset name under data/ recorded by demo 13")
    parser.add_argument("--output", default="so101_lift_cube_clock_cube",
                        help="dataset name under data/ to create")
    args = parser.parse_args()
    build(il_dataset.default_dataset_dir(args.source), il_dataset.default_dataset_dir(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
