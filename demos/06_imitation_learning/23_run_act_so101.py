"""Run a trained ACT policy on the real SO-101 follower (no leader arm).

Loads the policy from a local ``pretrained_model`` folder or from a Hugging Face repo, e.g.

    outputs/act_pick_all/checkpoints/006000/pretrained_model     (a checkpoint of ``20_train_act_so101.py``)
    IkerAzurmendi/act-pick-6000                                      (the same folder uploaded to the hub)

and drives the follower at the dataset rate: read both cameras + joints, predict, send the
action. It uses the same ports, cameras and 640x480 / 30 fps as ``17b_record_real_so101_live.py``,
so run it on the recording PC with the cameras where they were during recording.

Put the piece in the tray, keep a hand near the power supply, and try ``--dry-run`` first: it
reads the robot and cameras without configuring motors, changing torque, or writing calibration.
It stops if the motor EEPROM does not match the saved follower calibration. Ctrl-C stops; the
normal run releases the follower's torque on disconnect, so hold the arm.

    python 23_run_act_so101.py --policy IkerAzurmendi/act-pick-6000 --dry-run
    python 23_run_act_so101.py --policy IkerAzurmendi/act-pick-6000 --seconds 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_FIRST_GAP = 60  # degrees; a larger first action means the input is not what the policy was trained on
DEFAULT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E090058-if00"
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


def disconnect_read_only(robot) -> None:
    for camera in robot.cameras.values():
        if camera.is_connected or camera.thread is not None:
            try:
                camera.disconnect()
            except Exception as exc:
                print(f"Warning: camera disconnect failed: {exc}", flush=True)
    if robot.bus.is_connected:
        try:
            robot.bus.disconnect(disable_torque=False)
        except Exception as exc:
            print(f"Warning: motor bus disconnect failed: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="pretrained_model folder or Hugging Face repo id")
    parser.add_argument("--follower-port", default=DEFAULT_FOLLOWER_PORT,
                        help="follower serial path (default: this robot's stable USB ID)")
    parser.add_argument("--camera-front", default="auto", help="device path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-wrist", default="auto", help="device path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=60.0, help="stop after this many seconds (default: 60)")
    parser.add_argument("--max-step", type=float, default=3.0,
                        help="target offset from measured position: joint degrees, gripper percentage points")
    parser.add_argument("--save-frames", default=None,
                        help="folder to save the first front and wrist frame the policy sees (JPEG)")
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

    steps = 0
    try:
        print(f"follower port: {args.follower_port} -> {Path(args.follower_port).resolve()}", flush=True)
        if args.dry_run:
            robot.bus.connect()
            if not robot.bus.is_calibrated:
                raise RuntimeError(
                    "Follower motor EEPROM does not match mi_follower calibration; "
                    "dry-run will not write calibration or configure motors."
                )
            for camera in robot.cameras.values():
                camera.connect()
            print("dry-run: read-only connection; motor torque and settings are unchanged.", flush=True)
        else:
            robot.connect()

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
            if steps == 0:
                if args.save_frames:
                    import cv2
                    Path(args.save_frames).mkdir(parents=True, exist_ok=True)
                    for name in cameras:
                        cv2.imwrite(str(Path(args.save_frames) / f"first_{name}.jpg"),
                                    cv2.cvtColor(obs[name], cv2.COLOR_RGB2BGR))
                print(f"joints {JOINTS}", flush=True)
                if not args.dry_run and np.abs(action - state).max() > MAX_FIRST_GAP:
                    raise RuntimeError(
                        f"the first action is {np.abs(action - state).max():.0f} degrees from the arm's "
                        f"position (state {state.round().tolist()}, action {action.round().tolist()}); "
                        "not moving. Run with --dry-run and --save-frames to see why.")
            sent_action = None
            if not args.dry_run:
                sent_action = robot.send_action({f"{j}.pos": float(a) for j, a in zip(JOINTS, action)})
            steps += 1
            if steps == 1 or steps % args.fps == 0:
                print(f"t={time.perf_counter() - start:5.1f}s  max arm |prediction-state|="
                      f"{np.abs(action[:-1] - state[:-1]).max():5.1f} deg  "
                      f"gripper prediction {action[-1]:5.1f}%", flush=True)
                print(f"  state {state.round().astype(int).tolist()}  "
                      f"action {action.round().astype(int).tolist()}", flush=True)
                if sent_action is not None:
                    tracking = "  ".join(
                        f"{j}={measured:.1f}->{sent_action[f'{j}.pos']:.1f}"
                        for j, measured in zip(JOINTS, state)
                    )
                    print(f"  measured->sent (joints deg, gripper %): {tracking}", flush=True)
            precise_sleep(max(period - (time.perf_counter() - cycle), 0.0))
        elapsed = time.perf_counter() - start
        print(f"done: {steps} steps in {elapsed:.1f} s ({steps / elapsed:.1f} steps/s; the target is {args.fps})")
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        if args.dry_run:
            disconnect_read_only(robot)
        elif robot.is_connected:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
