"""Run the Up-down BAD ACT policy on the SO-101 follower (LeRobot 0.6.1), or replay it on a recorded episode.

    # robot (moves the arm!): start with the bracket already held in front of the inspection camera
    python run_policy.py --port COM5 --front-camera 0 --wrist-camera 1 --robot-id robotekin_follower

    # no hardware: feed a recorded episode through the same code path and compare with the demonstration
    python run_policy.py --dry-run-dataset ../../../demos/09_voice_bin_picking/data/up-down-bad_all --episode 4

The loop stops when the arm is back at the rest pose after releasing, or after --max-seconds.
See HANDOFF.md for the full contract (start pose, cameras, units, stop condition).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FPS = 30
TASK = "Carry the bracket from the camera to the bad box and release it"
# Mean final follower pose of the demonstrations (std ~4-10 per joint); gripper is the last joint.
REST_POSE = np.array([6.2, -100.1, 28.9, 99.5, -5.2, 0.9], dtype=np.float32)
REST_TOLERANCE = np.array([20, 15, 15, 10, 20, 8], dtype=np.float32)
GRIPPER_RELEASED = 20.0


def load_policy(policy_dir: Path, device: str, n_action_steps: int):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_dir)
    cfg.device = device
    cfg.n_action_steps = min(n_action_steps, cfg.chunk_size)
    policy = ACTPolicy.from_pretrained(policy_dir, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(policy_dir),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def step(policy, pre, post, features, observation: dict, device: str) -> dict:
    """Raw robot observation (joint '.pos' floats + 'front'/'wrist' HxWx3 uint8) -> robot action dict."""
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    frame = build_inference_frame(observation, torch.device(device), features, task=TASK, robot_type="so_follower")
    with torch.no_grad():
        action = post(policy.select_action(pre(frame)))
    return make_robot_action(action, features)


def finished(state: np.ndarray, released: bool) -> bool:
    return released and bool(np.all(np.abs(state - REST_POSE) <= REST_TOLERANCE))


def run_robot(args, policy, pre, post, features) -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    def camera(source: str):
        return OpenCVCameraConfig(index_or_path=int(source) if source.isdigit() else source, fps=FPS, width=640, height=480)

    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, calibration_dir=args.calibration_dir,
        max_relative_target=args.max_relative_target,
        cameras={"front": camera(args.front_camera), "wrist": camera(args.wrist_camera)},
    ))
    names = features["action"]["names"]
    robot.connect()
    released, start = False, time.perf_counter()
    try:
        while time.perf_counter() - start < args.max_seconds:
            tick = time.perf_counter()
            observation = robot.get_observation()
            state = np.array([observation[name] for name in names], dtype=np.float32)
            released |= state[-1] > GRIPPER_RELEASED
            if finished(state, released):
                print(f"Done: back at rest after releasing ({time.perf_counter() - start:.1f} s).")
                return
            robot.send_action(step(policy, pre, post, features, observation, args.device))
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
        print(f"Stopped: --max-seconds {args.max_seconds} reached without confirming the rest pose.")
    finally:
        robot.disconnect()


def run_dry(args, policy, pre, post, features) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.dry_run_dataset).resolve()
    dataset = LeRobotDataset(root.name, root=str(root), episodes=[args.episode], video_backend="pyav")
    names = features["action"]["names"]
    errors, released = [], False
    for index in range(len(dataset)):
        item = dataset[index]
        state = item["observation.state"].numpy()
        observation = {name: float(value) for name, value in zip(names, state)}
        for view in ("front", "wrist"):
            observation[view] = (item[f"observation.images.{view}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        action = step(policy, pre, post, features, observation, args.device)
        errors.append(np.abs(np.array([action[name] for name in names]) - item["action"].numpy()))
        released |= state[-1] > GRIPPER_RELEASED
    errors = np.array(errors)
    print(json.dumps({
        "episode": args.episode, "frames": len(errors),
        "mae_per_joint": dict(zip(names, errors.mean(0).round(2).tolist())),
        "mae": round(float(errors.mean()), 2),
        "stop_condition_reached_on_recording": finished(state, released),
    }, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, default=HERE / "latest" / "pretrained_model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-action-steps", type=int, default=25,
                    help="actions executed per prediction before re-observing (<= chunk 100); lower = more closed-loop")
    ap.add_argument("--port")
    ap.add_argument("--robot-id", default="robotekin_follower")
    ap.add_argument("--calibration-dir", type=Path, default=None)
    ap.add_argument("--front-camera", help="OpenCV index or device path of the camera recorded as 'front'")
    ap.add_argument("--wrist-camera", help="OpenCV index or device path of the camera recorded as 'wrist'")
    ap.add_argument("--max-relative-target", type=float, default=5.0, help="max change per command per joint")
    ap.add_argument("--max-seconds", type=float, default=15.0, help="demonstrations last 5-9 s (max 20 s)")
    ap.add_argument("--dry-run-dataset", type=Path)
    ap.add_argument("--episode", type=int, default=4)
    args = ap.parse_args()

    features = json.loads((args.policy.parent / "dataset_features.json").read_text(encoding="utf-8"))
    policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps)
    if args.dry_run_dataset:
        run_dry(args, policy, pre, post, features)
    elif args.port and args.front_camera is not None and args.wrist_camera is not None:
        run_robot(args, policy, pre, post, features)
    else:
        ap.error("pass --port, --front-camera and --wrist-camera for the robot, or --dry-run-dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
