"""Upload finalized local SO-101 datasets that are missing from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-prefix",
        default="MikelRubi/so101-pick-cube",
        help="owner and base dataset name without its timestamp suffix",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=HF_LEROBOT_HOME,
        help="LeRobot local cache directory (defaults to HF_LEROBOT_HOME)",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if "/" not in args.repo_prefix:
        parser.error("--repo-prefix must have the form OWNER/DATASET_NAME")

    owner, base_name = args.repo_prefix.split("/", 1)
    owner_dir = args.root.expanduser() / owner
    local_datasets = sorted(owner_dir.glob(f"{base_name}_*")) if owner_dir.is_dir() else []
    if not local_datasets:
        print(f"No local datasets found under {owner_dir}/{base_name}_*")
        return 0

    api = HfApi()
    uploaded = 0
    skipped = 0
    failed = 0

    for local_root in local_datasets:
        info_path = local_root / "meta" / "info.json"
        if not info_path.is_file():
            print(f"Skip {local_root.name}: missing meta/info.json")
            skipped += 1
            continue

        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skip {local_root.name}: cannot read metadata ({exc})")
            skipped += 1
            continue

        if info.get("total_episodes", 0) < 1 or info.get("total_frames", 0) < 1:
            print(f"Skip {local_root.name}: no finalized episodes")
            skipped += 1
            continue

        local_files = {
            path.relative_to(local_root).as_posix(): path.stat().st_size
            for path in local_root.rglob("*")
            if path.is_file() and not path.relative_to(local_root).as_posix().startswith("images/")
        }
        parquet_paths = sorted(local_root.rglob("*.parquet"))
        parquet_rows = {}
        try:
            for path in parquet_paths:
                parquet_rows[path] = pq.read_metadata(path).num_rows
        except Exception as exc:
            print(f"Skip {local_root.name}: invalid Parquet {path.relative_to(local_root)} ({exc})")
            skipped += 1
            continue

        data_rows = sum(
            rows
            for path, rows in parquet_rows.items()
            if path.relative_to(local_root).parts[0] == "data"
        )
        episode_rows = sum(
            rows
            for path, rows in parquet_rows.items()
            if path.relative_to(local_root).parts[:2] == ("meta", "episodes")
        )
        if data_rows != info["total_frames"] or episode_rows != info["total_episodes"]:
            print(
                f"Skip {local_root.name}: Parquet row counts do not match metadata "
                f"(frames {data_rows}/{info['total_frames']}, episodes {episode_rows}/{info['total_episodes']})"
            )
            skipped += 1
            continue

        video_keys = [
            name for name, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
        ]
        has_data = any(name.startswith("data/") and name.endswith(".parquet") for name in local_files)
        has_episode_metadata = any(
            name.startswith("meta/episodes/") and name.endswith(".parquet") for name in local_files
        )
        has_videos = all(
            any(name.startswith(f"videos/{key}/") and name.endswith(".mp4") for name in local_files)
            for key in video_keys
        )
        if (
            "meta/stats.json" not in local_files
            or not has_data
            or not has_episode_metadata
            or not has_videos
        ):
            print(f"Skip {local_root.name}: dataset metadata is incomplete")
            skipped += 1
            continue

        repo_id = f"{owner}/{local_root.name}"
        try:
            remote_info = api.dataset_info(repo_id, files_metadata=True)
        except RepositoryNotFoundError:
            remote_info = None
        except Exception as exc:
            print(f"ERROR {repo_id}: cannot inspect Hugging Face ({exc})")
            failed += 1
            continue

        if remote_info is not None:
            remote_files = {item.rfilename: item.size for item in remote_info.siblings}
            if all(remote_files.get(name) == size for name, size in local_files.items()):
                print(f"Skip {repo_id}: all local files are already on Hugging Face")
                skipped += 1
                continue

        print(f"Uploading {repo_id} ({info['total_episodes']} episodes) from {local_root}...")
        try:
            dataset = LeRobotDataset(repo_id=repo_id, root=local_root, video_backend="pyav")
            dataset.push_to_hub(private=args.private)
            print(f"Uploaded https://huggingface.co/datasets/{repo_id}")
            uploaded += 1
        except Exception as exc:
            print(f"ERROR {repo_id}: upload failed ({exc})")
            failed += 1

    print(f"Finished: {uploaded} uploaded, {skipped} skipped, {failed} failed.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
