"""Record SO-101 demonstrations with Enter, q, r, and e controls."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="MikelRubi/so101-pick-cube")
    parser.add_argument("--task", default="Pick up the cube and place it in the bowl")
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--camera-front", default="/dev/video2")
    parser.add_argument("--camera-wrist", default="/dev/video0")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10,
                        help="camera and dataset frame rate; both cameras must support it")
    parser.add_argument("--reset-seconds", type=float, default=10,
                        help="time to reposition the arm between episodes")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if args.reset_seconds < 0:
        parser.error("--reset-seconds must be non-negative")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.dataset import DatasetRecordConfig
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
    dataset_cfg = DatasetRecordConfig(
        repo_id=args.repo_id,
        single_task=args.task,
        fps=args.fps,
        push_to_hub=True,
        private=args.private,
        streaming_encoding=True,
        encoder_threads=2,
    )
    dataset_cfg.rgb_encoder.preset = 10  # SVT-AV1 maps preset 12 to 10 on this host.
    dataset_cfg.stamp_repo_id()
    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.repo_id,
        fps=dataset_cfg.fps,
        robot_type=robot.name,
        features=features,
        use_videos=True,
        video_backend="pyav",
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
        batch_encoding_size=dataset_cfg.video_encoding_batch_size,
        rgb_encoder=dataset_cfg.rgb_encoder,
        depth_encoder=dataset_cfg.depth_encoder,
        encoder_threads=dataset_cfg.encoder_threads,
        streaming_encoding=dataset_cfg.streaming_encoding,
        encoder_queue_maxsize=dataset_cfg.encoder_queue_maxsize,
        metadata_buffer_size=1,
    )

    phase = {"name": "waiting"}
    start_positioning = threading.Event()
    start_episode = threading.Event()
    end_episode = threading.Event()
    discard_episode = threading.Event()
    finish_recording = threading.Event()
    upload_requested = threading.Event()

    def on_key(key: str) -> None:
        key = key.lower()
        if key in ("enter", "\n", "\r"):
            if phase["name"] == "initial_wait":
                start_positioning.set()
            elif phase["name"] in ("positioning", "waiting"):
                start_episode.set()
        elif key == "q" and phase["name"] == "recording":
            end_episode.set()
        elif key == "r" and phase["name"] == "recording":
            discard_episode.set()
            end_episode.set()
        elif key == "e":
            upload_requested.set()
            finish_recording.set()
            end_episode.set()

    listener = None
    saved_episodes = 0

    def run_control(seconds: float | None, record: bool, stop_event=None) -> None:
        start = time.perf_counter()
        deadline = None if seconds is None else start + seconds
        control_interval = 1 / 30
        record_interval = 1 / dataset_cfg.fps
        next_record_time = start
        while deadline is None or time.perf_counter() < deadline:
            if (finish_recording.is_set() or (record and end_episode.is_set())
                    or (stop_event is not None and stop_event.is_set())):
                break
            cycle_start = time.perf_counter()
            observation = robot.get_observation()
            processed_observation = robot_observation_processor(observation)
            action = teleop.get_action()
            processed_action = teleop_action_processor((action, observation))
            robot_action = robot_action_processor((processed_action, observation))
            robot.send_action(robot_action)

            if record and cycle_start >= next_record_time:
                observation_frame = build_dataset_frame(dataset.features, processed_observation, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, processed_action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": args.task})
                next_record_time = cycle_start + record_interval

            precise_sleep(max(control_interval - (time.perf_counter() - cycle_start), 0.0))

    try:
        teleop.connect()
        robot.connect()
        phase["name"] = "initial_wait"
        start_episode.clear()
        listener = create_key_listener(
            on_key, controls_help="Enter=position then record, q=save locally, r=discard, e=finish/upload"
        )
        if listener is None:
            raise RuntimeError("Keyboard controls need an interactive terminal.")

        print(f"Dataset: {dataset_cfg.repo_id}")
        print(f"Local path: {dataset.root}")
        print("Press Enter to enable teleoperation and position the arms; press Enter again to record.", flush=True)

        with VideoEncodingManager(dataset):
            try:
                while not start_positioning.wait(0.1):
                    if finish_recording.is_set():
                        break

                should_record = False
                if not finish_recording.is_set():
                    phase["name"] = "positioning"
                    print("Teleoperation active; position the arms, then press Enter to start recording.", flush=True)
                    run_control(None, record=False, stop_event=start_episode)
                    start_episode.clear()
                    should_record = not finish_recording.is_set()

                while not finish_recording.is_set():
                    if not should_record:
                        phase["name"] = "waiting"
                        start_episode.clear()
                        while not start_episode.wait(0.1):
                            if finish_recording.is_set():
                                break
                        if finish_recording.is_set():
                            break

                    phase["name"] = "recording"
                    should_record = False
                    end_episode.clear()
                    discard_episode.clear()
                    print("Recording... q saves locally, r discards, e finalizes and uploads.", flush=True)
                    run_control(None, record=True)

                    if discard_episode.is_set():
                        dataset.clear_episode_buffer()
                        print("Episode discarded.", flush=True)
                    elif dataset.has_pending_frames():
                        dataset.save_episode()
                        saved_episodes += 1
                        print(f"Saved episode {saved_episodes} locally at {dataset.root}.", flush=True)
                    if finish_recording.is_set():
                        break

                    if args.reset_seconds:
                        phase["name"] = "reset"
                        print(f"Reset the arm now ({args.reset_seconds:g} seconds)...", flush=True)
                        run_control(args.reset_seconds, record=False)
                    print("Reset window finished. Press Enter for the next episode, or e to finish.", flush=True)
            except KeyboardInterrupt:
                finish_recording.set()
                if discard_episode.is_set():
                    dataset.clear_episode_buffer()
                elif dataset.has_pending_frames():
                    dataset.save_episode()
                    saved_episodes += 1
                print("Interrupted safely; finalizing saved episodes locally.", flush=True)

        print(f"Finalizing {saved_episodes} episode(s)...", flush=True)
    finally:
        phase["name"] = "finished"
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
        dataset_finalized = False
        try:
            dataset.finalize()
            dataset_finalized = True
        except Exception as exc:
            print(f"ERROR: dataset finalization failed: {exc}", flush=True)
        if dataset.num_episodes:
            if upload_requested.is_set() and dataset_finalized:
                print(f"Uploading {dataset.num_episodes} episode(s) to {dataset_cfg.repo_id}...", flush=True)
                try:
                    dataset.push_to_hub(private=args.private)
                    print(f"Dataset: https://huggingface.co/datasets/{dataset_cfg.repo_id}", flush=True)
                except Exception as exc:
                    print(f"ERROR: dataset upload failed: {exc}", flush=True)
                    print("The finalized local dataset is kept; retry with 18_upload_pending_so101.py.", flush=True)
            else:
                print(
                    f"Kept {dataset.num_episodes} episode(s) locally at {dataset.root}. "
                    "Run 18_upload_pending_so101.py to upload later.",
                    flush=True,
                )
        else:
            print("No episodes were saved; nothing was uploaded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
