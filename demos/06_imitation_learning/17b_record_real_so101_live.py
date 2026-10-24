"""Record SO-101 bin-picking demonstrations with a live camera view and review shortcuts.

Same recording pipeline as ``17_record_real_so101.py`` (leader/follower teleop, front +
wrist camera, LeRobot v3.0, local save then upload), reshaped around the advice in the
LeRobot "Imitation Learning on real-world robots" guide: keep the cameras fixed, record
about 10 episodes per object location, and throw away takes that were not clean.

Three policies, one dataset (and one key) each:

    q   pick the bracket and hold it up to the camera      -> MikelRubi/so101-pick-cube_<time>
    w   carry it from the camera to the good box           -> MikelRubi/up-down_<time>
    a   carry it from the camera to the bad box            -> MikelRubi/up-down-bad_<time>

The camera's good/bad decision happens between the pick take and the w/a take: you press
w or a according to the result.

A window shows both cameras and a status bar (phase, kind, saved episodes, location
counter, timer). Keys work from the terminal, from a global listener, or from that window:

    q / w / a       READY: start recording that kind.  RECORDING: stop, keep for review.
                    REVIEW: keep the take now and start recording that kind right away.
    Enter/Space/n/Right   READY: start recording the kind used last (pick at first).
                    RECORDING: stop.  REVIEW: keep it now.
    r / Left        RECORDING: discard and go back to READY.  REVIEW: discard.
    l / Up          next object location (the counters restart at 0)
    e / Esc         finish and upload (a take still being recorded is dropped)
    p / s           READY or REVIEW (between takes, teleop still on): save the front and wrist
                    images as they are now, labelled "with pin" (p, perno) or "without pin"
                    (s, sin perno). Move the arm so the piece is in front of the cameras first.

Flow of one episode: READY (teleop on, not recording) -> RECORDING (stops by itself after
``--episode-time``) -> REVIEW (``--review-seconds``; the take is only written to its
dataset when this window ends or you keep it, so ``r`` still throws it away) -> READY.
Ctrl-C finishes without uploading; ``18_upload_pending_so101.py`` uploads later.

Besides each dataset, ``meta/recording_notes.json`` stores the kind, location number and
length of every saved episode, so a training script can hold out whole locations.

The p / s snapshots are JPEG pairs (``pin_001_front.jpg`` / ``pin_001_wrist.jpg``, ``nopin_...``)
in ``pin_snapshots/`` inside the pick dataset, listed in ``pin_snapshots/index.json`` with their
label and location; they are uploaded with that dataset (which needs at least one pick episode).

Run on the recording PC, in the LeRobot environment:

    python 17b_record_real_so101_live.py
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

PREVIEW_INTERVAL = 1 / 10
DEBOUNCE_S = 0.3  # the window and the global listener can both report one key press

KEY_ACTIONS = {
    "enter": "advance", "\n": "advance", "\r": "advance", "space": "advance", " ": "advance",
    "n": "advance", "right": "advance",
    "q": "kind:pick", "w": "kind:good", "a": "kind:bad",
    "r": "discard", "left": "discard",
    "l": "location", "up": "location",
    "p": "snap:pin", "s": "snap:nopin",
    "e": "finish", "esc": "finish",
}
TK_KEYSYMS = {
    "Return": "enter", "space": "space", "Left": "left", "Right": "right", "Up": "up",
    "Escape": "esc",
}
KIND_LABELS = {"pick": "pick + present (q)", "good": "to good box (w)", "bad": "to bad box (a)"}
PLACE_TASKS = {
    "good": "Carry the bracket from the camera to the good box and release it",
    "bad": "Carry the bracket from the camera to the bad box and release it",
}
PLACE_REPOS = {"good": "up-down", "bad": "up-down-bad"}
PHASE_COLORS = {"ready": "#1f7a3a", "recording": "#b3261e", "review": "#9a6700"}
CAMERA_IDS = {
    "front": "*Wed_Camera*video-index0",
    "wrist": "*USB2.0_CAM1*video-index0",
}


def find_camera(role: str) -> Path:
    candidates = sorted((Path("/dev/v4l/by-id")).glob(CAMERA_IDS[role]))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely detect the {role} camera using /dev/v4l/by-id/{CAMERA_IDS[role]}. "
            f"Found {len(candidates)}; pass --camera-{role} with its device path."
        )
    return candidates[0]


class LivePreview:
    """Both camera views plus a status bar in a Tk window; keys pressed in it go to ``on_key``."""

    def __init__(self, on_key) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("SO-101 recording")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.Label(
            self.root, font=("TkDefaultFont", 16, "bold"), fg="white", anchor="w", justify="left"
        )
        self.status.pack(fill="x")
        self.picture = tk.Label(self.root)
        self.picture.pack()
        self.root.bind("<Key>", lambda e: on_key(TK_KEYSYMS.get(e.keysym) or e.char))
        self.open = True
        self._image = None  # Tk drops an image that Python no longer references

    def show(self, frames, lines, color) -> None:
        if not self.open:
            return
        import numpy as np

        picture = np.ascontiguousarray(np.hstack(frames))  # RGB uint8, same height
        height, width = picture.shape[:2]
        try:
            self._image = self._tk.PhotoImage(data=b"P6 %d %d 255\n" % (width, height) + picture.tobytes())
            self.picture.configure(image=self._image)
            self.status.configure(text="\n".join(lines), bg=color)
            self.root.update()
        except self._tk.TclError:  # window closed by the user
            self.open = False

    def close(self) -> None:
        self.open = False
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="MikelRubi/so101-pick-cube")
    parser.add_argument("--task", default="Pick a bracket from the crate and hold it up to the camera")
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--camera-front", default="auto", help="front camera path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-wrist", default="auto", help="wrist camera path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30,
                        help="camera and dataset frame rate; both cameras must support it")
    parser.add_argument("--episode-time", type=float, default=30,
                        help="seconds after which a take stops by itself")
    parser.add_argument("--review-seconds", type=float, default=4,
                        help="seconds after a take in which r still discards it")
    parser.add_argument("--per-location", type=int, default=10,
                        help="episodes of each kind to record per object location (the status bar counts it)")
    parser.add_argument("--no-preview", action="store_true", help="do not open the live window")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.episode_time <= 0 or args.review_seconds <= 0 or args.per_location <= 0:
        parser.error("--episode-time, --review-seconds and --per-location must be positive")
    if "/" not in args.repo_id:
        parser.error("--repo-id must have the form OWNER/DATASET_NAME")

    import cv2
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import (
        LeRobotDataset,
        VideoEncodingManager,
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: F401
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig  # noqa: F401
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.keyboard_input import create_key_listener
    from lerobot.utils.robot_utils import precise_sleep

    register_third_party_plugins()
    for role in ("front", "wrist"):
        option = f"camera_{role}"
        if getattr(args, option) == "auto":
            camera = find_camera(role)
            setattr(args, option, str(camera))
            print(f"Detected {role} camera: {camera} -> {camera.resolve()}", flush=True)

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=Path(args.camera_front), fps=args.fps,
            width=args.camera_width, height=args.camera_height, warmup_s=5
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=Path(args.camera_wrist), fps=args.fps,
            width=args.camera_width, height=args.camera_height, warmup_s=5
        ),
    }
    camera_names = list(cameras)
    robot = make_robot_from_config(
        SOFollowerRobotConfig(port=args.follower_port, id="mi_follower", cameras=cameras)
    )
    teleop = make_teleoperator_from_config(
        SOLeaderTeleopConfig(port=args.leader_port, id="mi_leader")
    )
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    owner = args.repo_id.split("/", 1)[0]
    tasks = {"pick": args.task, **PLACE_TASKS}
    repo_ids = {"pick": args.repo_id, **{kind: f"{owner}/{name}" for kind, name in PLACE_REPOS.items()}}
    recordings: dict[str, dict] = {}

    def create_recording(kind: str) -> dict:
        config = DatasetRecordConfig(
            repo_id=repo_ids[kind],
            single_task=tasks[kind],
            fps=args.fps,
            push_to_hub=True,
            private=args.private,
            streaming_encoding=True,
            encoder_threads=2,
        )
        config.rgb_encoder = RGBEncoderConfig(vcodec="auto")
        if config.rgb_encoder.vcodec == "h264_nvenc":
            config.rgb_encoder.preset = "p1"
        elif config.rgb_encoder.vcodec == "libsvtav1":
            config.rgb_encoder.preset = 10  # SVT-AV1 maps preset 12 to 10 on this host.
        config.stamp_repo_id()
        dataset = LeRobotDataset.create(
            repo_id=config.repo_id,
            fps=config.fps,
            robot_type=robot.name,
            features=features,
            use_videos=True,
            video_backend="pyav",
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
            batch_encoding_size=config.video_encoding_batch_size,
            rgb_encoder=config.rgb_encoder,
            depth_encoder=config.depth_encoder,
            encoder_threads=config.encoder_threads,
            streaming_encoding=config.streaming_encoding,
            encoder_queue_maxsize=config.encoder_queue_maxsize,
        )
        recordings[kind] = {"config": config, "dataset": dataset, "needs_resume": False, "notes": []}
        print(f"Local dataset ({kind}): {config.repo_id} at {dataset.root}", flush=True)
        return recordings[kind]

    create_recording("pick")

    advance = threading.Event()
    discard = threading.Event()
    next_location = threading.Event()
    finish_recording = threading.Event()
    upload_requested = threading.Event()
    last_press: dict[str, float] = {}
    state = {"phase": "ready", "kind": "pick", "t0": time.perf_counter()}
    location = {"index": 1}
    counts: dict[tuple[str, int], int] = {}
    chosen = {"kind": "pick"}
    kind_pressed = threading.Event()
    snapshot_requests: queue.SimpleQueue = queue.SimpleQueue()
    snapshot_counts = {"pin": 0, "nopin": 0}
    snapshot_index: list[dict] = []
    last_snapshot = {"text": ""}
    events = {"advance": advance, "discard": discard, "location": next_location}

    def on_key(key: str) -> None:
        action = KEY_ACTIONS.get(key.lower())
        if action is None:
            return
        now = time.perf_counter()
        if now - last_press.get(action, -DEBOUNCE_S) < DEBOUNCE_S:
            return
        last_press[action] = now
        if action.startswith("snap:"):
            if state["phase"] in ("ready", "review"):
                snapshot_requests.put(action.split(":", 1)[1])
            else:
                print("Snapshots are only taken between takes (READY or REVIEW).", flush=True)
            return
        if action == "finish":
            upload_requested.set()
            finish_recording.set()
        elif action.startswith("kind:"):
            chosen["kind"] = action.split(":", 1)[1]
            kind_pressed.set()
        else:
            events[action].set()

    def set_phase(name: str) -> None:
        state["phase"] = name
        state["t0"] = time.perf_counter()
        advance.clear()
        discard.clear()
        kind_pressed.clear()
        if name == "recording":  # a press made just before the take must not fire during it
            while not snapshot_requests.empty():
                snapshot_requests.get()

    def hud() -> tuple[list[str], str]:
        name = state["phase"]
        kind = state["kind"]
        count = counts.get((kind, location["index"]), 0)
        full = "  (location full: press l)" if count >= args.per_location else ""
        saved = sum(len(recording["notes"]) for recording in recordings.values())
        head = (f"{name.upper()} {KIND_LABELS[kind]}   saved {saved}   "
                f"location {location['index']}: {count}/{args.per_location}{full}")
        elapsed = time.perf_counter() - state["t0"]
        if name == "recording":
            second = (f"{elapsed:4.1f} / {args.episode_time:g} s   "
                      "Enter/n stop   r discard   e finish")
        elif name == "review":
            second = (f"keeping in {max(args.review_seconds - elapsed, 0):.0f} s   "
                      "r discard   Enter keep   q/w/a keep + record that kind")
        else:
            second = "q pick   w good box   a bad box   Enter repeat last   l next location   e finish"
        lines = [head, second]
        if name != "recording":
            lines.append(f"snapshots: {snapshot_counts['pin']} with pin / {snapshot_counts['nopin']} without   "
                         f"p = with pin   s = without pin{last_snapshot['text']}")
        return lines, PHASE_COLORS[name]

    def select_recording(kind: str, encoding_manager) -> None:
        if kind not in recordings:
            create_recording(kind)
        recording = recordings[kind]
        if recording["needs_resume"]:
            config = recording["config"]
            recording["dataset"] = LeRobotDataset.resume(
                repo_id=config.repo_id,
                root=recording["dataset"].root,
                video_backend="pyav",
                batch_encoding_size=config.video_encoding_batch_size,
                rgb_encoder=config.rgb_encoder,
                depth_encoder=config.depth_encoder,
                encoder_threads=config.encoder_threads,
                streaming_encoding=config.streaming_encoding,
                encoder_queue_maxsize=config.encoder_queue_maxsize,
            )
            recording["needs_resume"] = False
        state["kind"] = kind
        encoding_manager.dataset = recording["dataset"]

    def commit_episode(kind: str, seconds: float) -> None:
        recording = recordings[kind]
        dataset = recording["dataset"]
        episode_index = dataset.num_episodes
        dataset.save_episode()
        key = (kind, location["index"])
        counts[key] = counts.get(key, 0) + 1
        recording["notes"].append({
            "episode_index": episode_index,
            "kind": kind,
            "location": location["index"],
            "seconds": round(seconds, 2),
        })
        # Close the Parquet footers before the next episode.
        dataset.finalize()
        recording["needs_resume"] = True
        notes_path = Path(dataset.root) / "meta" / "recording_notes.json"
        notes_path.write_text(
            json.dumps({"task": tasks[kind], "fps": args.fps, "episodes": recording["notes"]}, indent=2),
            encoding="utf-8",
        )
        print(
            f"Saved {kind} episode {len(recording['notes'])} (location {location['index']}: "
            f"{counts[key]}/{args.per_location}) at {dataset.root}.",
            flush=True,
        )

    def save_snapshot(label: str, observation) -> None:
        folder = Path(recordings["pick"]["dataset"].root) / "pin_snapshots"
        folder.mkdir(exist_ok=True)
        snapshot_counts[label] += 1
        prefix = f"{label}_{snapshot_counts[label]:03d}"
        files = {name: f"{prefix}_{name}.jpg" for name in camera_names}
        for name, file in files.items():
            if not cv2.imwrite(str(folder / file), cv2.cvtColor(observation[name], cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write {folder / file}")
        snapshot_index.append({"label": label, "location": location["index"], **files})
        (folder / "index.json").write_text(json.dumps({"images": snapshot_index}, indent=2), encoding="utf-8")
        last_snapshot["text"] = f"   last: {prefix}"
        print(f"Saved snapshot {prefix} (front + wrist) in {folder}", flush=True)

    preview = None

    def run_control(record: bool, seconds: float | None = None, stop_events=()) -> None:
        start = time.perf_counter()
        deadline = None if seconds is None else start + seconds
        control_interval = 1 / 30
        record_interval = 1 / args.fps
        next_record_time = start
        next_preview_time = start
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if finish_recording.is_set() or any(event.is_set() for event in stop_events):
                break
            if next_location.is_set():
                next_location.clear()
                location["index"] += 1
                print(f"Location {location['index']}: move the pieces / crate to a new layout.",
                      flush=True)
            cycle_start = time.perf_counter()
            record_frame = record and cycle_start >= next_record_time
            show_frame = preview is not None and preview.open and cycle_start >= next_preview_time
            # Avoid a follower-bus read on control cycles that are neither stored nor shown.
            observation = robot.get_observation() if (record_frame or show_frame) else None
            action = teleop.get_action()
            processed_action = teleop_action_processor((action, observation))
            robot_action = robot_action_processor((processed_action, observation))
            robot.send_action(robot_action)
            while not record and not snapshot_requests.empty():
                if observation is None:
                    observation = robot.get_observation()
                save_snapshot(snapshot_requests.get(), observation)

            if record_frame:
                dataset = recordings[state["kind"]]["dataset"]
                processed_observation = robot_observation_processor(observation)
                observation_frame = build_dataset_frame(dataset.features, processed_observation, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, processed_action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": tasks[state["kind"]]})
                next_record_time = cycle_start + record_interval
            if show_frame:
                lines, color = hud()
                preview.show([observation[name] for name in camera_names], lines, color)
                next_preview_time = cycle_start + PREVIEW_INTERVAL

            precise_sleep(max(control_interval - (time.perf_counter() - cycle_start), 0.0))

    listener = None
    try:
        teleop.connect()
        robot.connect()
        listener = create_key_listener(
            on_key,
            controls_help="q/w/a=record pick/good/bad, Enter=repeat or stop, r=discard, l=next location, e=finish/upload",
        )
        if listener is None:
            raise RuntimeError("Keyboard controls need an interactive terminal.")
        if not args.no_preview:
            try:
                preview = LivePreview(on_key)
            except Exception as exc:  # no display, or Tk not installed
                print(f"Warning: live preview unavailable ({exc}); recording without it.", flush=True)

        print("Teleoperation active. q = pick, w = good box, a = bad box, Enter = same as last.", flush=True)

        with VideoEncodingManager(recordings["pick"]["dataset"]) as encoding_manager:
            try:
                kind = "pick"
                start_now = False
                while not finish_recording.is_set():
                    set_phase("ready")
                    if not start_now:
                        run_control(record=False, stop_events=(advance, kind_pressed))
                        if finish_recording.is_set():
                            break
                        if kind_pressed.is_set():
                            kind = chosen["kind"]
                    start_now = False

                    select_recording(kind, encoding_manager)
                    dataset = recordings[kind]["dataset"]
                    set_phase("recording")
                    print(f"Recording {kind} ({recordings[kind]['config'].repo_id})... "
                          "Enter/n/q/w/a stops, r discards, e finishes.", flush=True)
                    started = time.perf_counter()
                    run_control(record=True, seconds=args.episode_time,
                                stop_events=(advance, kind_pressed, discard))
                    seconds = time.perf_counter() - started

                    if finish_recording.is_set():
                        dataset.clear_episode_buffer()
                        print("Take in progress dropped.", flush=True)
                        break
                    if discard.is_set() or not dataset.has_pending_frames():
                        dataset.clear_episode_buffer()
                        print("Episode discarded.", flush=True)
                        continue

                    if seconds >= args.episode_time - 0.2:
                        print(f"Take hit the {args.episode_time:g} s limit.", flush=True)
                    elif seconds < 2:
                        print(f"Warning: very short take ({seconds:.1f} s).", flush=True)
                    set_phase("review")
                    print(f"Take of {seconds:.1f} s: r discards, Enter keeps now, q/w/a keeps and "
                          f"records that kind, otherwise kept in {args.review_seconds:g} s.", flush=True)
                    run_control(record=False, seconds=args.review_seconds,
                                stop_events=(advance, kind_pressed, discard))
                    if discard.is_set():
                        dataset.clear_episode_buffer()
                        print("Episode discarded.", flush=True)
                        continue
                    commit_episode(kind, seconds)
                    if kind_pressed.is_set() and not finish_recording.is_set():
                        kind = chosen["kind"]
                        start_now = True
            except KeyboardInterrupt:
                finish_recording.set()
                upload_requested.clear()
                dataset = recordings[state["kind"]]["dataset"]
                if state["phase"] == "review" and not discard.is_set() and dataset.has_pending_frames():
                    commit_episode(state["kind"], time.perf_counter() - state["t0"])
                else:
                    dataset.clear_episode_buffer()
                print("Interrupted safely; finalizing saved episodes locally.", flush=True)

        print("Finalizing...", flush=True)
        if snapshot_index:
            print(f"  snapshots: {snapshot_counts['pin']} with pin, {snapshot_counts['nopin']} without", flush=True)
        for (kind, index), count in sorted(counts.items()):
            print(f"  {kind}, location {index}: {count} episode(s)", flush=True)
    finally:
        state["phase"] = "finished"
        if preview is not None:
            preview.close()
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:
                print(f"Warning: keyboard listener cleanup failed: {exc}", flush=True)
        if robot.is_connected:
            try:
                robot.disconnect()
            except Exception as exc:
                print(
                    f"Warning: follower disconnect failed ({exc}). Check the arm and gripper "
                    "for a jam or overload.", flush=True
                )
        if teleop.is_connected:
            try:
                teleop.disconnect()
            except Exception as exc:
                print(f"Warning: leader disconnect failed: {exc}", flush=True)
        for recording in recordings.values():
            dataset = recording["dataset"]
            repo_id = recording["config"].repo_id
            dataset_finalized = False
            try:
                dataset.finalize()
                dataset_finalized = True
            except Exception as exc:
                print(f"ERROR: finalization failed for {repo_id}: {exc}", flush=True)
            if dataset.num_episodes:
                if upload_requested.is_set() and dataset_finalized:
                    print(f"Uploading {dataset.num_episodes} episode(s) to {repo_id}...", flush=True)
                    try:
                        dataset.push_to_hub(private=args.private)
                        print(f"Dataset: https://huggingface.co/datasets/{repo_id}", flush=True)
                    except Exception as exc:
                        print(f"ERROR: dataset upload failed for {repo_id}: {exc}", flush=True)
                        print("The finalized local dataset is kept; retry with 18_upload_pending_so101.py.", flush=True)
                else:
                    print(
                        f"Kept {dataset.num_episodes} episode(s) locally at {dataset.root}. "
                        "Run 18_upload_pending_so101.py to upload later.",
                        flush=True,
                    )
            else:
                print(f"No episodes saved for {repo_id}.", flush=True)
                if recording is recordings["pick"] and snapshot_index:
                    print(f"  its {len(snapshot_index)} snapshot pair(s) stay local in "
                          f"{Path(dataset.root) / 'pin_snapshots'}; save one pick episode to upload them.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
