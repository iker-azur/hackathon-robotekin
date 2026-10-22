"""Build an ACT-ready dataset variant with the cube detector as environment state.

The detector returns image-centroid coordinates. Store normalized ``u``, ``v``,
and a visibility bit as ``observation.environment_state`` so LeRobot ACT encodes
them through its supported environment-state input.

Run from the repository root:

    python .\demos\06_imitation_learning\tools\20_add_cube_detect_state.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import cube_detect  # noqa: E402

PHASE_KEY = "observation.phase"
ENV_STATE_KEY = "observation.environment_state"
IMAGE_KEY = "observation.images.front"
STATE_NAMES = ["cube_u_normalized", "cube_v_normalized", "visible"]


def detect_frame(image: np.ndarray) -> tuple[float, float, float]:
    detection = cube_detect.detect_cube(image)
    if detection is None:
        return (0.0, 0.0, 0.0)
    return (
        2.0 * detection.u / (image.shape[1] - 1) - 1.0,
        2.0 * detection.v / (image.shape[0] - 1) - 1.0,
        1.0,
    )


def image_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))] * values.shape[1],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def build(source: Path, output: Path, workers: int) -> None:
    import av

    source = source.resolve()
    output = output.resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset metadata not found under {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset variant: {output}")

    print(f"copying dataset to {output}", flush=True)
    shutil.copytree(source, output)

    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    video_rel = info["video_path"].format(
        video_key=IMAGE_KEY, chunk_index=0, file_index=0
    )
    video_path = output / video_rel
    video_path = video_path.resolve()
    parquet_paths = sorted((output / "data").glob("*/*.parquet"))
    if len(parquet_paths) != 1:
        raise ValueError(f"Expected one frame parquet shard, found {len(parquet_paths)}")

    values: list[tuple[float, float, float]] = []
    with av.open(str(video_path)) as container, concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        stream = container.streams.video[0]
        frames = container.decode(stream)
        while True:
            chunk = []
            try:
                for _ in range(64):
                    chunk.append(next(frames).to_ndarray(format="rgb24"))
            except StopIteration:
                pass
            if not chunk:
                break
            values.extend(pool.map(detect_frame, chunk))
            if len(values) % 4096 == 0:
                print(f"detected {len(values)} frames", flush=True)

    expected = int(info["total_frames"])
    if len(values) != expected:
        raise ValueError(f"Decoded {len(values)} frames, metadata declares {expected}")

    values_np = np.asarray(values, dtype=np.float32)
    table = pq.read_table(parquet_paths[0])
    if table.num_rows != expected:
        raise ValueError(f"Parquet has {table.num_rows} rows, expected {expected}")
    if PHASE_KEY not in table.column_names:
        raise ValueError(f"Missing source phase column {PHASE_KEY}")
    table = table.drop([PHASE_KEY])
    env_state = pa.FixedSizeListArray.from_arrays(
        pa.array(values_np.reshape(-1), type=pa.float32()), list_size=3
    )
    table = table.append_column(ENV_STATE_KEY, env_state)
    pq.write_table(table, parquet_paths[0], compression="zstd")

    features = info["features"]
    features.pop(PHASE_KEY, None)
    features[ENV_STATE_KEY] = {
        "dtype": "float32",
        "shape": [3],
        "names": STATE_NAMES,
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    stats_path = output / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats.pop(PHASE_KEY, None)
    stats[ENV_STATE_KEY] = image_stats(values_np)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")

    for episodes_path in (output / "meta" / "episodes").glob("*/*.parquet"):
        episodes = pq.read_table(episodes_path)
        stale = [
            name
            for name in episodes.column_names
            if name.startswith(f"stats/{PHASE_KEY}/")
        ]
        if stale:
            pq.write_table(episodes.drop(stale), episodes_path, compression="zstd")

    visible_rate = float(values_np[:, 2].mean())
    print(
        f"wrote {expected} environment-state rows; cube visible in "
        f"{visible_rate:.1%} of frames; output={output}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=DEMO_DIR / "data" / "so101_lift_cube_phase",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEMO_DIR / "data" / "so101_lift_cube_cube_detect",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.source, args.output, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
