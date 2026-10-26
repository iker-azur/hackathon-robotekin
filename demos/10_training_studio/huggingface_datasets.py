"""List and download the user's LeRobot datasets from the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def emit_result(**payload) -> None:
    print("@@HF_RESULT@@" + json.dumps(payload, ensure_ascii=False), flush=True)


def list_datasets(author: str) -> int:
    from huggingface_hub import HfApi

    rows = []
    for info in HfApi().list_datasets(author=author, limit=100, full=True):
        modified = getattr(info, "last_modified", None)
        rows.append({
            "id": info.id,
            "private": bool(getattr(info, "private", False)),
            "downloads": int(getattr(info, "downloads", 0) or 0),
            "last_modified": modified.isoformat() if modified else "",
            "used_storage": getattr(info, "used_storage", None),
        })
    emit_result(ok=True, datasets=rows, author=author)
    return 0


def download_dataset(repo_id: str, author: str, data_root: Path) -> int:
    from huggingface_hub import HfApi, snapshot_download

    owner = repo_id.split("/", 1)[0]
    if owner.casefold() != author.casefold():
        raise ValueError(f"Only datasets owned by {author} are available in this browser.")

    api = HfApi()
    info = api.dataset_info(repo_id)
    local_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id.replace("/", "__"))
    target = data_root / local_name
    if (target / "meta" / "info.json").is_file():
        emit_result(ok=True, already_downloaded=True, repo_id=repo_id, local_name=local_name,
                    destination=str(target))
        return 0
    if target.exists():
        raise FileExistsError(
            f"A local folder already exists at {target} but is not a complete LeRobot dataset. "
            "Move or rename it before downloading this repository again."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    staging = data_root / f".{local_name}.partial"
    print(f"Downloading {repo_id} to {staging}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=info.sha,
        local_dir=str(staging),
    )

    metadata_path = staging / "meta" / "info.json"
    if not metadata_path.is_file():
        raise ValueError(
            "The repository downloaded, but it does not contain meta/info.json. "
            "This studio can train LeRobot ACT datasets, not arbitrary image archives."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("features") or {}
    if not isinstance(features, dict) or "action" not in features:
        raise ValueError("The downloaded LeRobot dataset has no 'action' feature required for ACT training.")
    image_features = [key for key in features if str(key).startswith("observation.images.")]
    if not image_features:
        raise ValueError("The downloaded LeRobot dataset has no camera image features for bin picking.")

    manifest = {
        "repo_id": repo_id,
        "revision": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "owner": author,
    }
    (staging / "huggingface_dataset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    staging.replace(target)
    emit_result(ok=True, repo_id=repo_id, local_name=local_name, destination=str(target),
                episodes=metadata.get("total_episodes"), frames=metadata.get("total_frames"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("list", "download"))
    parser.add_argument("--author", default="MikelRubi")
    parser.add_argument("--repo-id")
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "list":
            return list_datasets(args.author)
        if not args.repo_id:
            raise ValueError("A dataset repository ID is required.")
        return download_dataset(args.repo_id, args.author, args.data_root)
    except Exception as error:  # Surface Hub/auth/network errors in the app's status panel.
        print(f"Hugging Face error: {error}", file=sys.stderr, flush=True)
        emit_result(ok=False, error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
