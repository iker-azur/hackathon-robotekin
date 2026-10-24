"""Build one local ACT training dataset from the real SO-101 recordings on Hugging Face.

One run builds the training dataset of ONE policy, chosen with ``--kind``:

    pick   ``so101-pick-cube_*``    (17b key q: pick the piece and present it to the camera)
    good   ``up-down_*``            (17b key w: carry it from the camera to the good box)
    bad    ``up-down-bad_*``        (17b key a: carry it from the camera to the bad box)

By default only sessions recorded with ``17b_record_real_so101_live.py`` are used (they
carry ``meta/recording_notes.json``). The earlier ``pick-cube`` sessions are the same task
(pick a piece from the tray and present it to the camera: their end poses and last frames
match the 17b ones), only with many pieces in the tray instead of one, and
``--include-legacy`` adds them (``--kind pick`` only). The earlier ``up-down`` sessions
look like the same carry to the good box, but that has not been confirmed, so they are not
offered.

The recordings were made in incompatible configurations, so LeRobot cannot merge
them as they are:

    10 fps, 1280 x 720 (16:9, wide view of the tray)       15 datasets
    30 fps,  640 x 480 ( 4:3, tighter view of the tray)    25 datasets

This script puts every episode on one common footing and writes a single
LeRobot v3.0 dataset that ``20_train_act_so101.py`` trains on:

* **fps** - everything goes to ``--fps`` (default 30). The 30 fps recordings are used as
  they are. The 10 fps recordings are *up*-sampled by 3: each camera frame is held for
  three output frames and the joint positions are interpolated linearly between the
  recorded samples. That adds no information (the real 10 fps data stays 10 fps data
  underneath, with the image trailing the joints by up to 0.1 s) but puts every episode
  on the same time axis. With ``--fps 10`` the opposite happens: a 30 fps episode becomes
  three 10 fps episodes (frames 0::3, 1::3, 2::3), which come from the same take, so
  never put some in train and others in validation. ``meta/sources.json`` records the
  source episode, offset and up-sampling of every output episode.
* **frame shape** - 16:9 frames are center-cropped to 4:3, then resized to
  ``--height`` x ``--width`` (default 480 x 640, so the 30 fps footage stays native).
  The crop is a first approximation of the tighter view, not a calibrated match.
* **outliers** - episodes longer than ``--max-seconds`` are dropped (default 20 s: the
  first test runs, a 60 s and a 37 s episode, and one 21 s episode that was 50 % idle).
* **held-out session** - the sessions in ``--holdout`` (default: the last session of the
  kind) are flagged ``holdout`` in ``sources.json``; the training script never trains on
  them and validates on them. ``--holdout`` with no names keeps nothing out.

Nothing is guessed about success: the recordings carry no success label, so every kept
episode is treated as a good demonstration.

Run from the repository root, in the ``hackathon-robotekin`` environment:

    python .\\demos\\06_imitation_learning\\19_build_so101_act_dataset.py --kind pick

Downloads go to ``data/real_so101/raw`` and are reused on later runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "data" / "real_so101"
RAW_DIR = DATA_DIR / "raw"

DEFAULT_OWNER = "MikelRubi"
# Session names of each kind. "up-down" must not match "up-down-bad" or "up-down-small".
KIND_PATTERNS = {
    "pick": r"so101-pick-cube.*",
    "good": r"up-down_\d{8}_\d{6}",
    "bad": r"up-down-bad_\d{8}_\d{6}",
}
NOTES_FILE = "meta/recording_notes.json"  # written only by 17b_record_real_so101_live.py
CAMERAS = ("front", "wrist")
COMPLETE_MARKER = ".complete"
DATASET_REPO_ID = "local/so101_act"


def use_windows_certificates() -> None:
    """Make Python trust the Windows certificate store.

    On this PC Python alone rejects huggingface.co ("unable to get local issuer
    certificate") although the browser and curl accept it. ``truststore`` fixes that
    without turning verification off: ``python -m pip install --no-deps truststore``.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


def discover(owner: str, download: bool, kind: str, include_legacy: bool) -> list[str]:
    """Names of the recording datasets of one kind: from the Hub, or from what is already downloaded."""
    if download:
        from huggingface_hub import HfApi

        api = HfApi()
        names = [d.id.split("/", 1)[1] for d in api.list_datasets(author=owner)]
    else:
        names = [p.name for p in RAW_DIR.glob("*") if (p / COMPLETE_MARKER).exists()]
    names = sorted(n for n in names if re.fullmatch(KIND_PATTERNS[kind], n))
    if include_legacy:
        return names
    if download:
        return [n for n in names if NOTES_FILE in api.list_repo_files(f"{owner}/{n}", repo_type="dataset")]
    return [n for n in names if (RAW_DIR / n / NOTES_FILE).exists()]


def fetch(owner: str, name: str) -> Path:
    root = RAW_DIR / name
    if not (root / COMPLETE_MARKER).exists():
        from huggingface_hub import snapshot_download

        print(f"  downloading {owner}/{name}", flush=True)
        snapshot_download(repo_id=f"{owner}/{name}", repo_type="dataset", local_dir=str(root))
        (root / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
    return root


def read_episodes(root: Path) -> list[dict]:
    """The per-episode metadata rows, without the large per-episode statistics columns."""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = [c for c in parquet.schema_arrow.names if not c.startswith("stats/")]
        rows += parquet.read(columns=columns).to_pylist()
    return sorted(rows, key=lambda r: r["episode_index"])


def read_joints(root: Path, row: dict):
    import numpy as np
    import pyarrow.parquet as pq

    path = root / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
    table = pq.read_table(path).to_pylist()
    rows = sorted((r for r in table if r["episode_index"] == row["episode_index"]),
                  key=lambda r: r["frame_index"])
    action = np.asarray([r["action"] for r in rows], dtype=np.float32)
    state = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
    return action, state


def fit(frame, height: int, width: int):
    """Center-crop to the target aspect ratio, then resize (never stretch)."""
    import cv2

    h, w = frame.shape[:2]
    want = width / height
    if w / h > want + 1e-3:
        new_w = round(h * want)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif w / h < want - 1e-3:
        new_h = round(w / want)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h]
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def decode(root: Path, camera: str, row: dict, fps: int, height: int, width: int) -> list:
    """The episode's frames for one camera, already fitted to the target shape."""
    import av

    key = f"observation.images.{camera}"
    path = root / "videos" / key / f"chunk-{row[f'videos/{key}/chunk_index']:03d}" / (
        f"file-{row[f'videos/{key}/file_index']:03d}.mp4")
    start = float(row[f"videos/{key}/from_timestamp"])
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(max(start - 1.0, 0.0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) < start - 0.5 / fps:
                continue
            frames.append(fit(frame.to_ndarray(format="rgb24"), height, width))
            if len(frames) == row["length"]:
                break
    if len(frames) != row["length"]:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames, the episode has {row['length']}")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--kind", required=True, choices=sorted(KIND_PATTERNS),
                        help="which policy's data to build: pick, good (box) or bad (box)")
    parser.add_argument("--name", default=None,
                        help="dataset folder under data/real_so101 (default act_<kind>; an existing one is refused)")
    parser.add_argument("--include-legacy", action="store_true",
                        help="also use sessions recorded before 17b (only for --kind pick)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--holdout", nargs="*", default=None,
                        help="source datasets kept out of training, for validation "
                             "(default: the last session of the kind; no names: keep none out)")
    parser.add_argument("--no-download", action="store_true",
                        help="use only what is already under data/real_so101/raw (offline)")
    args = parser.parse_args()
    if args.include_legacy and args.kind != "pick":
        parser.error("--include-legacy only applies to --kind pick")
    args.name = args.name or f"act_{args.kind}"

    use_windows_certificates()
    final = DATA_DIR / args.name
    if final.exists():
        print(f"ERROR: {final} already exists; pass another --name or delete it.")
        return 1
    partial = DATA_DIR / (args.name + ".partial")
    if partial.exists():
        print(f"removing the unfinished build {partial}")
        shutil.rmtree(partial)

    names = discover(args.owner, not args.no_download, args.kind, args.include_legacy)
    if not names:
        print(f"ERROR: no '{args.kind}' recordings found; they must come from 17b_record_real_so101_live.py "
              "(pass --include-legacy for older pick-cube sessions)")
        return 1
    if args.holdout is None:
        args.holdout = names[-1:] if len(names) > 1 else []
        if not args.holdout:
            print("WARNING: only one session, so nothing is held out for validation")
    missing = [n for n in args.holdout if n not in names]
    if missing:
        print(f"ERROR: held-out session(s) not among the {len(names)} '{args.kind}' datasets: {missing}")
        return 1
    print(f"{len(names)} '{args.kind}' recording datasets, held out: {', '.join(args.holdout) or 'none'}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = None
    sources, skipped = [], []
    started = time.time()
    for name in names:
        root = fetch(args.owner, name) if not args.no_download else RAW_DIR / name
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        src_fps = info["fps"]
        if max(src_fps, args.fps) % min(src_fps, args.fps):
            print(f"ERROR: {name} is {src_fps} fps; --fps {args.fps} is not a multiple or divisor of it")
            return 1
        stride = max(src_fps // args.fps, 1)
        upsample = max(args.fps // src_fps, 1)
        height, width = info["features"]["observation.images.front"]["shape"][:2]
        regime = f"{src_fps}fps-{height}p"
        joint_names = info["features"]["action"]["names"]

        if dataset is None:
            image = {"dtype": "video", "shape": (args.height, args.width, 3),
                     "names": ["height", "width", "channels"]}
            features = {
                "action": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
                "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
                **{f"observation.images.{cam}": dict(image) for cam in CAMERAS},
            }
            dataset = LeRobotDataset.create(
                repo_id=DATASET_REPO_ID, fps=args.fps, features=features, root=partial,
                robot_type=info.get("robot_type"), use_videos=True, image_writer_threads=8,
                vcodec="h264",
            )
        elif joint_names != features["action"]["names"]:
            print(f"ERROR: {name} has different joints: {joint_names}")
            return 1

        for row in read_episodes(root):
            seconds = row["length"] / src_fps
            if seconds > args.max_seconds:
                skipped.append(f"{name} ep {row['episode_index']} ({seconds:.0f} s)")
                continue
            t0 = time.time()
            action, state = read_joints(root, row)
            images = {cam: decode(root, cam, row, src_fps, args.height, args.width) for cam in CAMERAS}
            task = row["tasks"][0]
            last = row["length"] - 1
            for offset in range(stride):
                # (source frame, fraction of the way to the next one) for every output frame
                if upsample > 1:
                    steps = [(i, k / upsample) for i in range(row["length"]) for k in range(upsample)]
                else:
                    steps = [(i, 0.0) for i in range(offset, row["length"], stride)]
                for i, frac in steps:
                    nxt = min(i + 1, last)
                    dataset.add_frame({
                        "action": action[i] + frac * (action[nxt] - action[i]),
                        "observation.state": state[i] + frac * (state[nxt] - state[i]),
                        "task": task,
                        **{f"observation.images.{cam}": images[cam][i] for cam in CAMERAS},
                    })
                sources.append({
                    "episode_index": dataset.meta.total_episodes, "source_repo": f"{args.owner}/{name}",
                    "source_episode": row["episode_index"], "offset": offset, "stride": stride,
                    "upsample": upsample, "regime": regime, "group": args.kind, "task": task,
                    "holdout": name in args.holdout, "frames": len(steps),
                })
                dataset.save_episode()
            print(f"  {name} ep {row['episode_index']}: {row['length']} frames at {src_fps} fps -> "
                  f"{stride} episode(s) of {len(steps)} frames at {args.fps} fps ({time.time() - t0:.0f} s)",
                  flush=True)

    dataset.finalize()
    (partial / "meta" / "sources.json").write_text(json.dumps(sources, indent=1) + "\n", encoding="utf-8")
    os.replace(partial, final)

    print("")
    print(f"dataset: {final}  ({len(sources)} episodes, {sum(s['frames'] for s in sources)} frames at {args.fps} fps, "
          f"{args.width}x{args.height})")
    print(f"dropped: {len(skipped)} episode(s) over {args.max_seconds:g} s: {', '.join(skipped) or 'none'}")
    print("                              train episodes / held-out")
    for regime in sorted({s["regime"] for s in sources}):
        pick = [s for s in sources if s["regime"] == regime]
        held = sum(s["holdout"] for s in pick)
        print(f"  {regime:12s} {args.kind:10s} {len(pick) - held:5d} / {held}")
    print(f"built in {time.time() - started:.0f} s; train with 20_train_act_so101.py --dataset {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
