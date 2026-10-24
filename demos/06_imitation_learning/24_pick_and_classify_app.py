"""Window that shows both cameras, runs the pick ACT policy and tells whether the piece has the pin.

One Tk window (same approach as ``17b_record_real_so101_live.py``) over the real SO-101 follower:

    - the front and wrist camera, live;
    - the pin classifier of ``22_train_pin_classifier.py`` on those two frames, about 6 times a second,
      shown as "PIN / NO PIN" with its probability;
    - SPACE starts the pick policy (``23_run_act_so101.py`` does the same without a window). It runs for
      ``--pick-seconds`` or until you press S; the arm then keeps its position and the RESULT is the
      classifier averaged over the last 1.5 s, i.e. while the piece is held up to the cameras.

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
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PREVIEW_INTERVAL = 1 / 6  # camera view and classifier refresh
RESULT_WINDOW_S = 1.5  # the result averages the classifier over this long at the end of the pick
COLORS = {"idle": "#1f7a3a", "running": "#b3261e", "result": "#9a6700"}


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
    parser.add_argument("--pick-seconds", type=float, default=20, help="the pick stops by itself after this long")
    parser.add_argument("--max-step", type=float, default=3,
                        help="largest joint move (degrees) the follower accepts per step")
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
                elif key == "space" and phase != "running":
                    policy.reset()
                    phase, run_start, first, history, result, note = "running", cycle, True, [], "", ""
                elif key == "s" and phase == "running":
                    phase, note, result = "result", "stopped by you", None
            if not window.open:
                break

            obs = robot.get_observation()
            if phase == "running":
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
                    robot.send_action({f"{j}.pos": float(a) for j, a in zip(runner.JOINTS, action)})
                first = False
                if phase == "running" and cycle - run_start >= args.pick_seconds:
                    phase, note, result = "result", "pick finished", None

            if cycle >= next_show:
                p = classifier(obs["front"], obs["wrist"])
                history.append((cycle, p))
                if phase == "result" and result is None:
                    recent = [q for t, q in history if t >= cycle - RESULT_WINDOW_S] or [p]
                    result = f"RESULT: {label(sum(recent) / len(recent))}"
                status = {"idle": "READY - SPACE starts the pick",
                          "running": f"PICKING {cycle - run_start:4.1f} s / {args.pick_seconds:.0f} s - S stops",
                          "result": "DONE - SPACE picks again"}[phase]
                window.show([obs["front"], obs["wrist"]], f"{status}   {note}", COLORS[phase],
                            "classifier now: " + label(p), result or "")
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
