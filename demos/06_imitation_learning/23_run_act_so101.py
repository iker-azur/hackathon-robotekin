"""Run a trained ACT policy on the real SO-101 follower (no leader arm).

Loads the policy from a local ``pretrained_model`` folder or from a Hugging Face repo, e.g.

    outputs/act_pick_all/checkpoints/006000/pretrained_model     (a checkpoint of ``20_train_act_so101.py``)
    IkerAzurmendi/act-pick-6000                                      (the same folder uploaded to the hub)

and drives the follower at the dataset rate: read both cameras + joints, predict, send the
action. It uses the same ports, cameras and 640x480 / 30 fps as ``17b_record_real_so101_live.py``,
so run it on the recording PC with the cameras where they were during recording.

Put the piece in the tray, keep a hand near the power supply, and try ``--dry-run`` first: it
does everything except sending the action, so you can compare the predicted action with the
real state. Ctrl-C stops; the follower's torque is released on disconnect, so hold the arm.

    python 23_run_act_so101.py --policy IkerAzurmendi/act-pick-6000 --dry-run
    python 23_run_act_so101.py --policy IkerAzurmendi/act-pick-6000 --seconds 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERA_IDS = {  # same USB IDs as 17b_record_real_so101_live.py
    "front": "*Wed_Camera*video-index0",
    "wrist": "*USB2.0_CAM1*video-index0",
}


def find_camera(role: str, path: str) -> Path:
    if path != "auto":
        return Path(path)
    candidates = sorted(Path("/dev/v4l/by-id").glob(CAMERA_IDS[role]))
    if len(candidates) != 1:
        raise RuntimeError(f"Could not uniquely detect the {role} camera in /dev/v4l/by-id "
                           f"({len(candidates)} found); pass --camera-{role} with its device path.")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="pretrained_model folder or Hugging Face repo id")
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--camera-front", default="auto", help="device path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-wrist", default="auto", help="device path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=30, help="stop after this long")
    parser.add_argument("--max-step", type=float, default=8,
                        help="largest joint move (degrees) the follower accepts per step; lower is safer")
    parser.add_argument("--dry-run", action="store_true", help="predict and print, but do not move the arm")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = parser.parse_args()
    if args.fps <= 0 or args.seconds <= 0 or args.max_step <= 0:
        parser.error("--fps, --seconds and --max-step must be positive")

    import numpy as np
    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: F401
    from lerobot.utils.robot_utils import precise_sleep

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = PreTrainedConfig.from_pretrained(args.policy)
    config.device = device
    policy = ACTPolicy.from_pretrained(args.policy, config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy.reset()
    print(f"policy {args.policy} on {device}; chunk {config.chunk_size}, "
          f"action steps {config.n_action_steps}", flush=True)

    cameras = {
        name: OpenCVCameraConfig(index_or_path=find_camera(name, path), fps=args.fps, width=args.camera_width,
                                 height=args.camera_height, warmup_s=5)
        for name, path in (("front", args.camera_front), ("wrist", args.camera_wrist))
    }
    robot = make_robot_from_config(SOFollowerRobotConfig(
        port=args.follower_port, id="mi_follower", cameras=cameras, max_relative_target=args.max_step))

    robot.connect()
    steps = 0
    try:
        period = 1 / args.fps
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds:
            cycle = time.perf_counter()
            obs = robot.get_observation()
            state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
            batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).to(device)}
            for name in cameras:
                image = torch.from_numpy(obs[name]).float().div(255).permute(2, 0, 1)  # HWC uint8 -> CHW [0,1]
                batch[f"observation.images.{name}"] = image.unsqueeze(0).to(device)
            batch["task"], batch["robot_type"] = "", ""
            with torch.inference_mode():
                action = post(policy.select_action(pre(batch))).squeeze(0).cpu().numpy()
            if not args.dry_run:
                robot.send_action({f"{j}.pos": float(a) for j, a in zip(JOINTS, action)})
            steps += 1
            if steps % args.fps == 0:
                print(f"t={time.perf_counter() - start:5.1f}s  max |action-state|="
                      f"{np.abs(action - state).max():5.1f}  gripper action {action[-1]:5.1f}", flush=True)
            precise_sleep(max(period - (time.perf_counter() - cycle), 0.0))
        elapsed = time.perf_counter() - start
        print(f"done: {steps} steps in {elapsed:.1f} s ({steps / elapsed:.1f} steps/s; the target is {args.fps})")
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
