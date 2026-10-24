"""Build the two-camera dataset for the piece / piece-with-pin classifier.

Input: ``pin_labels.json`` (next to this script): for each pick-cube episode, whether the
piece held at the end has a pin (perno / tornillo) or not. The labels were made by looking
at the last frame of both cameras; ``confidence`` is ``clear`` or ``moderate`` (a blurred or
partial view), and episodes marked ``unsure`` or ``exclude`` are skipped.

For every labelled episode this takes the last ``--frames`` frames of the front and the
wrist camera (the last one, N-1, N-2, ...). They all get the episode's label: the arm is
nearly still at the end, so they are near-duplicates, and the training script keeps them
together when it splits train from test. Frames are centre-cropped to 4:3 and resized to
``--height`` x ``--width`` (never stretched).

Output, in ``data/real_so101/<name>`` (an existing folder is refused):

    pin_cls.npz      front / wrist (N, H, W, 3) uint8 RGB, label (N,) 0 = nopin, 1 = pin
    samples.json     per sample: session, episode, k (0 = last frame), confidence, fps
    review_pin.jpg   one thumbnail pair (front | wrist, last frame) per episode, to check the
    review_nopin.jpg labels by eye

Needs the raw recordings that ``19_build_so101_act_dataset.py`` downloads
(``data/real_so101/raw``); missing sessions are downloaded unless ``--no-download``.

Run from the repository root, in the ``hackathon-robotekin`` environment:

    python .\\demos\\06_imitation_learning\\21_build_pin_dataset.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
LABELS_PATH = DEMO_DIR / "pin_labels.json"
CAMERAS = ("front", "wrist")
CLASSES = {"nopin": 0, "pin": 1}


def load_build_module():
    """19_build_so101_act_dataset.py already knows where the raw recordings are and how to read them."""
    spec = importlib.util.spec_from_file_location("build_act", DEMO_DIR / "19_build_so101_act_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def last_frames(root: Path, camera: str, row: dict, fps: int, count: int) -> list:
    """The last ``count`` RGB frames of one episode and camera, newest first."""
    import av

    key = f"observation.images.{camera}"
    path = root / "videos" / key / f"chunk-{row[f'videos/{key}/chunk_index']:03d}" / (
        f"file-{row[f'videos/{key}/file_index']:03d}.mp4")
    start = float(row[f"videos/{key}/from_timestamp"])
    first = start - 0.5 / fps  # frames of this episode lie in [first, end)
    end = start + (row["length"] - 0.5) / fps
    kept = deque(maxlen=count)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(max(end - (count + 2) / fps - 1.0, 0.0) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t >= end:
                break
            if t >= first:
                kept.append(frame.to_ndarray(format="rgb24"))
    if len(kept) < count:
        raise RuntimeError(f"{path}: only {len(kept)} of the last {count} frames could be decoded")
    return list(kept)[::-1]


def write_review(path: Path, items: list[tuple[str, object, object]]) -> None:
    """A grid of (front | wrist) thumbnails with a caption, for checking labels by eye."""
    import cv2
    import numpy as np

    thumb_w, thumb_h, per_row = 160, 120, 4
    cells = []
    for caption, front, wrist in items:
        pair = np.hstack([cv2.resize(front, (thumb_w, thumb_h)), cv2.resize(wrist, (thumb_w, thumb_h))])
        pair = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
        cv2.putText(pair, caption, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        cells.append(pair)
    while len(cells) % per_row:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--name", default="pin_cls", help="folder under data/real_so101")
    parser.add_argument("--frames", type=int, default=3, help="last N frames per episode (1 = only the last)")
    parser.add_argument("--height", type=int, default=168)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--owner", default=None, help="Hugging Face owner of the recordings (default: MikelRubi)")
    parser.add_argument("--no-download", action="store_true",
                        help="use only what is already under data/real_so101/raw (offline)")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")

    build = load_build_module()
    final = build.DATA_DIR / args.name
    if final.exists():
        print(f"ERROR: {final} already exists; pass another --name or delete it.")
        return 1
    build.use_windows_certificates()

    import numpy as np

    episodes = [e for e in json.loads(args.labels.read_text(encoding="utf-8"))["episodes"] if e["label"] in CLASSES]
    by_session: dict[str, list[dict]] = {}
    for e in episodes:
        by_session.setdefault(e["session"], []).append(e)
    owner = args.owner or build.DEFAULT_OWNER

    arrays = {cam: [] for cam in CAMERAS}
    labels, samples, review = [], [], {name: [] for name in CLASSES}
    for session in sorted(by_session):
        root = build.RAW_DIR / session if args.no_download else build.fetch(owner, session)
        fps = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
        rows = {r["episode_index"]: r for r in build.read_episodes(root)}
        for e in by_session[session]:
            frames = {cam: [build.fit(f, args.height, args.width)
                            for f in last_frames(root, cam, rows[e["episode"]], fps, args.frames)]
                      for cam in CAMERAS}
            for k in range(args.frames):
                for cam in CAMERAS:
                    arrays[cam].append(frames[cam][k])
                labels.append(CLASSES[e["label"]])
                samples.append({"session": session, "episode": e["episode"], "k": k,
                                "confidence": e["confidence"], "fps": fps})
            review[e["label"]].append((f"{session[-15:]} e{e['episode']} {e['confidence']}",
                                       frames["front"][0], frames["wrist"][0]))
        print(f"  {session}: {len(by_session[session])} episode(s)", flush=True)

    partial = final.with_name(final.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(partial / "pin_cls.npz", front=np.stack(arrays["front"]), wrist=np.stack(arrays["wrist"]),
                        label=np.asarray(labels, dtype=np.int64))
    (partial / "samples.json").write_text(json.dumps(samples, indent=1) + "\n", encoding="utf-8")
    for name, items in review.items():
        if items:
            write_review(partial / f"review_{name}.jpg", items)
    partial.rename(final)

    print("")
    print(f"dataset: {final} ({len(samples)} samples from {len(episodes)} episodes, "
          f"{args.frames} frame(s) each, {args.width}x{args.height})")
    for name, value in CLASSES.items():
        mine = [s for s, y in zip(samples, labels) if y == value]
        clear = sum(s["confidence"] == "clear" for s in mine)
        print(f"  {name:6s} {len(mine) // args.frames:3d} episodes ({clear // args.frames} clear, "
              f"{(len(mine) - clear) // args.frames} moderate)")
    print(f"check the labels by eye: {final / 'review_pin.jpg'} and {final / 'review_nopin.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
