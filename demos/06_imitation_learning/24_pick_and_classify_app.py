"""Window that shows both cameras, runs the pick ACT policy and tells whether the piece has the pin.

One Tk window (same approach as ``17b_record_real_so101_live.py``) over the real SO-101 follower:

    - the front and wrist camera, live;
    - SPACE starts the pick policy (``23_run_act_so101.py`` does the same without a window). It runs for
      at most ``--pick-seconds``. After at least 5 s, an arm that is nearly stationary and close to
      policy targets for 1.5 s finishes early. Brief prediction/encoder noise is tolerated. The app
      then holds the measured arm pose and last gripper target with torque
      kept enabled, so the gripper keeps its grip on the piece;
    - only after that hold command, ``classification_enabled`` enables the pin classifier of
      ``22_train_pin_classifier.py``. After a brief settling pause, it averages 1.5 s of fresh
      front/wrist frames and shows the RESULT. No ACT targets are sent during classification.

S holds the current pose and cancels the pick/classification without producing a result.

Keys (the window must have focus):  SPACE start the pick   S stop it   Q / Esc quit.

The classifier is a visual judgement trained on about 1000 labelled frames and was about 80 % right on
unseen sessions in cross-validation, so treat the result as a hint. Quitting disconnects the follower,
which releases its torque: hold the arm. Starting a pick from a position other than the rest pose
(folded, above the tray) is refused when the first action is more than 60 degrees away.

    python 24_pick_and_classify_app.py

With no arguments it downloads (once) the pick policy ``IkerAzurmendi/act-pick-12000`` and the classifier
``IkerAzurmendi/pin-classifier`` from the Hugging Face hub; ``--policy`` / ``--classifier`` also take a local
folder / ``model.pt``.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from collections import deque
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PREVIEW_INTERVAL = 1 / 6  # camera view and classifier refresh
SETTLE_S = 0.5  # let the arm and camera view settle after the hold command
RESULT_WINDOW_S = 1.5  # classify only stationary frames captured after settling
MIN_PICK_S = 5.0  # avoid declaring completion at the initial rest pose
POSE_WINDOW_S = 1.5
STOP_ARM_MOTION_DEG = 2.0
STOP_ARM_ERROR_DEG = 6.0
STOP_GRIPPER_MOTION = 5.0  # percentage points; contact prevents full closure on a held piece
COLORS = {"idle": "#1f7a3a", "running": "#b3261e", "classifying": "#1565c0", "result": "#9a6700"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.split(".")[0], DEMO_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PinClassifier:
    """Saved ``22_train_pin_classifier.py`` model: probability of "pin" for one front + wrist frame pair."""

    def __init__(self, path: Path, device: str) -> None:
        trainer, builder = load_script("22_train_pin_classifier.py"), load_script("19_build_so101_act_dataset.py")
        self.model, saved = trainer.load_model(path, device)
        self._predict, self._fit, self._device = trainer.predict_proba, builder.fit, device
        self._size = (saved["height"], saved["width"])

    def __call__(self, front, wrist) -> float:
        return float(self._predict(self.model, self._fit(front, *self._size)[None],
                                   self._fit(wrist, *self._size)[None], self._device)[0])


def resolve_classifier(spec: str) -> Path:
    """A local ``model.pt``, or a Hugging Face repo id whose ``model.pt`` is downloaded once and cached."""
    if Path(spec).is_file():
        return Path(spec)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(spec, "model.pt"))


def label(p: float) -> str:
    return f"PIN ({p:.0%})" if p >= 0.5 else f"NO PIN ({1 - p:.0%})"


def pose_stop_status(samples, elapsed: float) -> tuple[bool, str]:
    """Detect a measured plateau near raw policy targets and explain any remaining blocker."""
    import numpy as np

    while len(samples) > 1 and samples[1][0] <= elapsed - POSE_WINDOW_S:
        samples.popleft()
    if elapsed < MIN_PICK_S:
        return False, f"Auto-stop arms in {MIN_PICK_S - elapsed:.1f} s"
    if len(samples) < 3 or elapsed - samples[0][0] < POSE_WINDOW_S:
        return False, "Auto-stop: collecting stationary-pose measurements"
    states = np.array([state for _, state, _ in samples])
    targets = np.array([target for _, _, target in samples])
    if not (np.isfinite(states).all() and np.isfinite(targets).all()):
        return False, "Auto-stop: invalid joint measurement or policy prediction"

    # Central ranges tolerate isolated old spikes, but the newest sample must also be stable.
    low, high = np.percentile(states, [10, 90], axis=0)
    motion = np.maximum(high - low, np.abs(states[-1] - np.median(states, axis=0)))
    arm_motion, gripper_motion = float(motion[:-1].max()), float(motion[-1])
    errors = np.abs(targets[:, :-1] - states[:, :-1])
    arm_error = float(max(np.percentile(errors, 80, axis=0).max(), errors[-1].max()))
    reached = (arm_motion <= STOP_ARM_MOTION_DEG and arm_error <= STOP_ARM_ERROR_DEG
               and gripper_motion <= STOP_GRIPPER_MOTION)
    status = (
        f"Auto-stop: arm motion {arm_motion:.1f}/{STOP_ARM_MOTION_DEG:g} deg; "
        f"target gap {arm_error:.1f}/{STOP_ARM_ERROR_DEG:g} deg; "
        f"gripper motion {gripper_motion:.1f}/{STOP_GRIPPER_MOTION:g}%"
    )
    return reached, status


class Window:
    """Both camera views and three text lines; keys pressed in it go to ``on_key``."""

    def __init__(self, on_key) -> None:
        import tkinter as tk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("SO-101 pick + pin classifier")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.Label(self.root, font=("TkDefaultFont", 16, "bold"), fg="white", anchor="w")
        self.status.pack(fill="x")
        self.picture = tk.Label(self.root)
        self.picture.pack()
        self.live = tk.Label(self.root, font=("TkDefaultFont", 18), anchor="w")
        self.live.pack(fill="x")
        self.result = tk.Label(self.root, font=("TkDefaultFont", 28, "bold"), anchor="w")
        self.result.pack(fill="x")
        self.root.bind("<Key>", lambda e: on_key(e.keysym.lower()))
        self.open = True
        self._image = None  # Tk drops an image that Python no longer references

    def show(self, frames, status: str, color: str, live: str, result: str) -> None:
        if not self.open:
            return
        import numpy as np

        picture = np.ascontiguousarray(np.hstack([f[::2, ::2] for f in frames]))  # half size: faster to draw
        height, width = picture.shape[:2]
        try:
            self._image = self._tk.PhotoImage(data=b"P6 %d %d 255\n" % (width, height) + picture.tobytes())
            self.picture.configure(image=self._image)
            self.status.configure(text=status, bg=color)
            self.live.configure(text=live)
            self.result.configure(text=result)
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
    runner = load_script("23_run_act_so101.py")  # JOINTS, DEFAULT_FOLLOWER_PORT, find_camera, MAX_FIRST_GAP
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="IkerAzurmendi/act-pick-12000",
                        help="pick pretrained_model folder or Hugging Face repo id")
    parser.add_argument("--classifier", default="IkerAzurmendi/pin-classifier",
                        help="classifier model.pt, or the Hugging Face repo id that holds it")
    parser.add_argument("--follower-port", default=runner.DEFAULT_FOLLOWER_PORT,
                        help="follower serial path (default: the one 23_run_act_so101.py uses)")
    parser.add_argument("--camera-front", default="auto", help="device path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-wrist", default="auto", help="device path, or 'auto' to detect by USB ID")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pick-seconds", type=float, default=20,
                        help="maximum pick duration; a settled pose can finish earlier")
    parser.add_argument("--max-step", type=float, default=8.0,
                        help="target offset from measured position: joint degrees, gripper percentage points")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = parser.parse_args()
    if args.fps <= 0 or args.pick_seconds <= 0 or args.max_step <= 0:
        parser.error("--fps, --pick-seconds and --max-step must be positive")

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
    classifier = PinClassifier(resolve_classifier(args.classifier), device)
    config = PreTrainedConfig.from_pretrained(args.policy)
    config.device = device
    policy = ACTPolicy.from_pretrained(args.policy, config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    cameras = {
        name: OpenCVCameraConfig(index_or_path=runner.find_camera(name, path), fps=args.fps,
                                 width=args.camera_width, height=args.camera_height, warmup_s=5)
        for name, path in (("front", args.camera_front), ("wrist", args.camera_wrist))
    }
    robot = make_robot_from_config(SOFollowerRobotConfig(
        port=args.follower_port, id="mi_follower", cameras=cameras, max_relative_target=args.max_step))

    keys = []
    window = Window(keys.append)
    phase, run_start, first, history, result, note = "idle", 0.0, False, [], "", ""
    classification_enabled, classify_start, stop_requested = False, 0.0, False
    last_gripper_target = None
    pose_history = deque()
    stop_status, next_stop_report = "Auto-stop waits for the pick to start", 0.0
    next_show = 0.0
    robot.connect()
    try:
        period = 1 / args.fps
        while window.open:
            cycle = time.perf_counter()
            while keys:
                key = keys.pop(0)
                if key in ("q", "escape"):
                    window.close()
                elif key == "space" and phase in ("idle", "result"):
                    policy.reset()
                    phase, run_start, first, history, result, note = "running", cycle, True, [], "", ""
                    classification_enabled = False
                    last_gripper_target = None
                    pose_history.clear()
                    stop_status, next_stop_report = "Auto-stop: waiting for movement", cycle
                elif key == "s" and phase in ("running", "classifying"):
                    stop_requested = True
            if not window.open:
                break

            obs = robot.get_observation()
            timed_out = phase == "running" and cycle - run_start >= args.pick_seconds
            reached_pose = False
            if phase == "running" and not stop_requested and not timed_out:
                state = np.array([obs[f"{j}.pos"] for j in runner.JOINTS], dtype=np.float32)
                batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
                         "task": "", "robot_type": ""}
                for name in cameras:
                    image = torch.from_numpy(obs[name]).float().div(255).permute(2, 0, 1)
                    batch[f"observation.images.{name}"] = image.unsqueeze(0).to(device)
                with torch.inference_mode():
                    action = post(policy.select_action(pre(batch))).squeeze(0).cpu().numpy()
                gap = float(np.abs(action - state).max())
                if first and gap > runner.MAX_FIRST_GAP:
                    phase, note = "idle", (f"not moving: the first action is {gap:.0f} degrees from the "
                                           "arm (is it at the rest pose, above the tray?)")
                else:
                    pose_history.append((cycle - run_start, state.copy(), action.copy()))
                    reached_pose, stop_status = pose_stop_status(pose_history, cycle - run_start)
                    if not reached_pose:
                        sent_action = robot.send_action({f"{j}.pos": float(a) for j, a in zip(runner.JOINTS, action)})
                        last_gripper_target = sent_action["gripper.pos"]
                first = False

            if stop_requested or (phase == "running" and (timed_out or reached_pose)):
                hold_action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in runner.JOINTS}
                # Preserve the closing target so holding the arm does not release gripper pressure.
                if last_gripper_target is not None:
                    hold_action["gripper.pos"] = last_gripper_target
                robot.send_action(hold_action)
                history, result = [], ""
                classification_enabled = not stop_requested
                phase = "classifying" if classification_enabled else "idle"
                reason = "position reached" if reached_pose else "time limit reached"
                note = f"{reason}; holding pose" if classification_enabled else "stopped; holding pose"
                print(f"Pick stopped: {note}", flush=True)
                if reached_pose:
                    print(stop_status, flush=True)
                classify_start = cycle + SETTLE_S
                stop_requested = False

            if cycle >= next_show:
                live = {"idle": "Classifier waits until the pick finishes",
                        "running": stop_status,
                        "classifying": "Holding pose; waiting for the camera view to settle",
                        "result": "Classification finished; motors holding pose"}[phase]
                if phase == "running" and cycle >= next_stop_report:
                    print(stop_status, flush=True)
                    next_stop_report = cycle + 1.0
                if classification_enabled and cycle >= classify_start:
                    p = classifier(obs["front"], obs["wrist"])
                    history.append((cycle, p))
                    live = "classifier now: " + label(p)
                    if cycle - history[0][0] >= RESULT_WINDOW_S:
                        result = f"RESULT: {label(sum(q for _, q in history) / len(history))}"
                        phase, classification_enabled = "result", False
                status = {"idle": "READY - SPACE starts the pick",
                          "running": f"PICKING {cycle - run_start:4.1f} s / {args.pick_seconds:.0f} s - S stops",
                          "classifying": "CLASSIFYING - motors holding pose - S cancels",
                          "result": "DONE - SPACE picks again"}[phase]
                window.show([obs["front"], obs["wrist"]], f"{status}   {note}", COLORS[phase],
                            live, result)
                next_show = cycle + PREVIEW_INTERVAL
            precise_sleep(max(period - (time.perf_counter() - cycle), 0.0))
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        robot.disconnect()
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
