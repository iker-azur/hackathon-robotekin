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
      front/wrist frames and shows the RESULT, then runs the configured post-classification policy
      on the same connected follower and cameras.

After the result, ``policy_routes.json`` selects a trajectory policy. The window lets the operator
choose the Up-down BAD checkpoint used for the white box. S holds the current pose and stops the
active pick, classifier, or route.

Keys (the window must have focus):  SPACE start the pick   S stop it   P print measured joints   Q / Esc quit.

The classifier is a visual judgement trained on about 1000 labelled frames and was about 80 % right on
unseen sessions in cross-validation, so treat the result as a hint. Quitting disconnects the follower,
which releases its torque: hold the arm. Starting a pick from a position other than the rest pose
(folded, above the tray) is refused when the first action is more than 60 degrees away.

    python 24_pick_and_classify_app.py

With no arguments it selects the first ``IkerAzurmendi/act-pick-*`` checkpoint of the list (v3 super, 6,000
steps) and the classifier ``IkerAzurmendi/pin-classifier``. The pick selector lists all 40 checkpoints by run
and step with their offline error (``analyze_pick_policies.py``); both selectors show upload times in Madrid.
``--policy`` / ``--classifier`` also take a local folder / ``model.pt``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

DEMO_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DEMO_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
PREVIEW_INTERVAL = 1 / 6  # camera view and classifier refresh
SETTLE_S = 0.5  # let the arm and camera view settle after the hold command
RESULT_WINDOW_S = 1.5  # classify only stationary frames captured after settling
MIN_PICK_S = 5.0  # avoid declaring completion at the initial rest pose
POSE_WINDOW_S = 1.5
STOP_ARM_MOTION_DEG = 2.0
STOP_ARM_ERROR_DEG = 6.0
STOP_GRIPPER_MOTION = 5.0  # percentage points; contact prevents full closure on a held piece
COLORS = {"idle": "#1f7a3a", "running": "#b3261e", "classifying": "#1565c0",
          "routing": "#7b4ca0", "result": "#9a6700", "manual": "#9a6700", "homing": "#9a6700"}
HOME_APPROACH_TARGET = {"shoulder_pan": 10.59, "shoulder_lift": -25.36, "elbow_flex": -59.30,
                        "wrist_flex": 48.04, "wrist_roll": -0.66, "gripper": 1.1}
HOME_TARGET = {"shoulder_pan": 2.59, "shoulder_lift": -103.52, "elbow_flex": 19.12,
               "wrist_flex": 95.87, "wrist_roll": 0.40, "gripper": 2.8}
HOME_WAYPOINTS = (HOME_APPROACH_TARGET, HOME_TARGET)
HOME_ARM_TOLERANCE_DEG = 2.0
HOME_GRIPPER_TOLERANCE_PERCENT = 3.0
HOME_TIMEOUT_S = 45.0
# Keep the home target cap aligned with the app's usual 7-degree step setting.
# This limits position error; it does not change the servo's torque limit.
HOME_MAX_STEP = 7.0
_ROUTE_CLEARANCE_NOTICE = None


class HmiSortingRuleMonitor:
    """Poll the HMI rule off the control loop and expose only the latest update."""

    def __init__(self, url: str, initial_settings: dict):
        base_url = url.rstrip("/")
        self.url = base_url + "/api/sorting-rule"
        self.start_url = base_url + "/api/robot/poll"
        self.updates = queue.Queue()
        self.start_requests = queue.Queue()
        self.home_requests = queue.Queue()
        self.home_stop_requests = queue.Queue()
        self.control_requests = queue.Queue()
        self.phase_lock = threading.Lock()
        self.phase = "starting"
        self.report = {"note": "", "result": "", "destination": None,
                       "settings": initial_settings.copy(),
                       "defaults": initial_settings.copy(), "config_revision": 0}
        self.config_revision = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        previous_revision = None
        previous_mode = None
        while not self.stop.is_set():
            try:
                with urlopen(self.url, timeout=0.5) as response:
                    state = json.loads(response.read())
                config = {"rule": state.get("rule"), "active": state.get("active", False),
                          "settings": state.get("settings"), "defaults": state.get("defaults"),
                          "revision": state.get("revision", 0)}
                if config["active"]:
                    routes = config["rule"].get("routes", {})
                    if any(routes.get(key, {}).get("destination") not in ("green", "white")
                           for key in ("with_pin", "without_pin")):
                        raise ValueError("invalid sorting destinations")
                if config["revision"] != previous_revision:
                    self.updates.put(config)
                    previous_revision = config["revision"]
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
            try:
                with self.phase_lock:
                    phase = self.phase
                    report = self.report.copy()
                request = Request(self.start_url,
                                  data=json.dumps({"phase": phase, "report": report}).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=0.5) as response:
                    command = json.loads(response.read())
                configuration = command.get("configuration") or {}
                if configuration.get("revision") == self.config_revision:
                    settings = configuration.get("settings")
                    if isinstance(settings, dict):
                        with self.phase_lock:
                            self.report["settings"] = settings
                if command.get("start"):
                    self.start_requests.put(True)
                if command.get("home"):
                    self.home_requests.put(True)
                if command.get("stop_home"):
                    self.home_stop_requests.put(True)
                requested_mode = command.get("control_mode")
                if requested_mode in ("automatic", "manual") and requested_mode != previous_mode:
                    self.control_requests.put(requested_mode)
                    previous_mode = requested_mode
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            self.stop.wait(0.2)

    def latest(self):
        update = None
        while True:
            try:
                update = self.updates.get_nowait()
            except queue.Empty:
                return update

    def set_state(self, phase, note, result, destination, settings, defaults,
                  leader_available=False, control_mode="automatic", leader_status=""):
        with self.phase_lock:
            self.phase = phase
            self.report = {"note": note, "result": result, "destination": destination,
                           "settings": settings, "defaults": defaults,
                           "config_revision": self.config_revision,
                           "leader_available": leader_available, "control_mode": control_mode,
                           "leader_status": leader_status}

    def acknowledge_configuration(self, revision):
        with self.phase_lock:
            self.config_revision = revision

    def take_start_request(self):
        try:
            self.start_requests.get_nowait()
            return True
        except queue.Empty:
            return False

    def take_home_request(self):
        try:
            self.home_requests.get_nowait()
            return True
        except queue.Empty:
            return False

    def take_home_stop_request(self):
        try:
            self.home_stop_requests.get_nowait()
            return True
        except queue.Empty:
            return False

    def take_control_mode(self):
        mode = None
        while True:
            try:
                mode = self.control_requests.get_nowait()
            except queue.Empty:
                return mode

    def close(self):
        self.stop.set()


def find_leader_port(requested_port):
    if requested_port:
        return requested_port
    candidates = sorted(Path("/dev/serial/by-id").glob("*5A7A054669*"))
    return str(candidates[0]) if len(candidates) == 1 else None


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.split(".")[0], DEMO_DIR / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


motion_controls = load_script("motion_controls.py")


def set_action_steps(policy, requested: int | None, model_default: int) -> int:
    steps = min(requested if requested is not None else model_default, policy.config.chunk_size)
    policy.config.n_action_steps = steps
    policy.reset()  # ACT rebuilds its action queue with the chosen length.
    return steps


def load_script_path(path: Path):
    spec = importlib.util.spec_from_file_location(f"route_policy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load policy runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_policy_registry(config_path: Path):
    return load_script("policy_router.py").PolicyRouteRegistry(config_path)


def load_pick_components(source: str, device: str, revision: str | None = None):
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint = (source if Path(source).is_dir() else
                  snapshot_download(source, revision=revision,
                                    allow_patterns=["config.json", "*.safetensors", "policy_*.json"]))
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def load_route_components(route, device: str):
    checkpoint = route.checkpoint
    if getattr(route, "hub_repo", None):
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(route.hub_repo, revision=route.hub_revision,
                                     allow_patterns=[f"{route.hub_path}/*"])
        checkpoint = Path(snapshot) / route.hub_path
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"Policy {route.name!r} is missing {model_path}")
    lfs_pointers = []
    for path in checkpoint.glob("*.safetensors"):
        with path.open("rb") as model_file:
            if model_file.read(64).startswith(b"version https://git-lfs.github.com/spec/"):
                lfs_pointers.append(path.name)
    if lfs_pointers:
        raise RuntimeError(
            f"Policy {route.name!r} has Git LFS pointer files instead of weights: "
            f"{', '.join(lfs_pointers)}; "
            "install Git LFS and run `git lfs pull`."
        )

    runner = load_script_path(route.runner)
    features_path = getattr(route, "dataset_features", None) or checkpoint.parent / "dataset_features.json"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    policy, pre, post = runner.load_policy(checkpoint, device, route.n_action_steps)
    policy.reset()
    return runner, policy, pre, post, features


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


def label(p: float, threshold: float = 0.5) -> str:
    return f"PIN ({p:.0%})" if p >= threshold else f"NO PIN ({1 - p:.0%})"


@lru_cache(maxsize=1)
def load_route_kinematics():
    return load_script_path(DEMO_DIR.parent / "08_recovery_kit/recoverykit/kinematics.py")


def add_route_clearance(action: dict[str, float], released: bool,
                        measured_state, settings=None, destination="white") -> dict[str, float]:
    """Apply the greatest reachable lift without replacing policy motion with a hold."""
    global _ROUTE_CLEARANCE_NOTICE
    import numpy as np

    settings = settings or motion_controls.default_settings()
    if not settings[f"{destination}_clearance"]:
        return action
    kinematics = load_route_kinematics()
    arm_joints = kinematics.ARM
    target_q = np.deg2rad([float(action[f"{joint}.pos"]) for joint in arm_joints])
    measured_q = np.deg2rad(np.asarray(measured_state[:5], dtype=float))
    policy_pose = kinematics.fk(dict(zip(arm_joints, target_q)))["gripper"]
    measured_pose = kinematics.fk(dict(zip(arm_joints, measured_q)))["gripper"]

    closed = settings[f"{destination}_closed"]
    release = settings[f"{destination}_released"]
    opening = min(max((float(action["gripper.pos"]) - closed) / (release - closed), 0.0), 1.0)
    requested_lift = 0.0 if released else settings[f"{destination}_height_cm"] / 100.0 * (1.0 - opening)
    for scale in (1.0, 0.5, 0.25, 0.0):
        target_pose = policy_pose.copy()
        target_pose[2, 3] += requested_lift * scale
        target_pose[2, 3] = max(target_pose[2, 3], measured_pose[2, 3]
                                - settings[f"{destination}_descent_mm"] / 1000.0)
        result = kinematics.ik(target_pose, np.eye(4), target_q, iters=100)
        if result.ok:
            if scale < 1.0:
                notice = f"Route lift limited to {requested_lift * scale * 100:.1f} cm by reachability"
                if notice != _ROUTE_CLEARANCE_NOTICE:
                    print(notice, flush=True)
                    _ROUTE_CLEARANCE_NOTICE = notice
            return {**action, **{f"{joint}.pos": float(value)
                                for joint, value in zip(arm_joints, np.rad2deg(result.q))}}

    notice = "Route lift unreachable; applying policy action without height correction"
    if notice != _ROUTE_CLEARANCE_NOTICE:
        print(notice, flush=True)
        _ROUTE_CLEARANCE_NOTICE = notice
    return action


def pose_stop_status(samples, elapsed: float, settings=None) -> tuple[bool, str]:
    """Detect a measured plateau near raw policy targets and explain any remaining blocker."""
    import numpy as np

    settings = settings or motion_controls.default_settings()
    window, minimum = settings["pose_window"], settings["min_pick"]
    arm_limit, error_limit = settings["arm_motion"], settings["arm_error"]
    grip_limit = settings["gripper_motion"]
    while len(samples) > 1 and samples[1][0] <= elapsed - window:
        samples.popleft()
    if not settings["auto_stop"]:
        return False, "Auto-stop disabled; pick stops at the configured time limit"
    if elapsed < minimum:
        return False, f"Auto-stop arms in {minimum - elapsed:.1f} s"
    if len(samples) < 3 or elapsed - samples[0][0] < window:
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
    reached = (arm_motion <= arm_limit and arm_error <= error_limit and gripper_motion <= grip_limit)
    status = (
        f"Auto-stop: arm motion {arm_motion:.1f}/{arm_limit:g} deg; "
        f"target gap {arm_error:.1f}/{error_limit:g} deg; "
        f"gripper motion {gripper_motion:.1f}/{grip_limit:g}%"
    )
    return reached, status


class Window:
    """Both camera views and three text lines; keys pressed in it go to ``on_key``."""

    def __init__(self, on_key, checkpoint_choices=(), pick_choices=(), initial_pick=None,
                 initial_motion=None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self.root = tk.Tk()
        self.root.title("SO-101 pick + pin classifier")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.Label(self.root, font=("TkDefaultFont", 16, "bold"), fg="white", anchor="w")
        self.status.pack(fill="x")
        self._selectors, self._running = [], None
        self.pick_checkpoint = tk.StringVar(value=initial_pick or (pick_choices[0] if pick_choices else ""))
        if pick_choices:
            pick_row = tk.Frame(self.root)
            pick_row.pack(fill="x", padx=8, pady=6)
            tk.Label(pick_row, text="Recogida · checkpoint:").pack(side="left")
            selector = ttk.Combobox(pick_row, textvariable=self.pick_checkpoint, values=pick_choices,
                                   state="readonly", width=95, height=16)
            selector.pack(side="left", padx=4)
            self._selectors.append(selector)
        destinations = tk.Frame(self.root)
        destinations.pack(fill="x", padx=8, pady=6)
        self.pin_destination = tk.StringVar(value="Blanca")
        self.no_pin_destination = tk.StringVar(value="Blanca")
        self._destination_selectors = []
        self._voice_rule_active = False
        self._manual_destinations = ("Blanca", "Blanca")
        self.voice_rule_status = tk.Label(self.root, text="Regla HMI: sin conectar", anchor="w")
        self.voice_rule_status.pack(fill="x", padx=8)
        tk.Label(destinations, text="Con pin → caja:").pack(side="left")
        selector = ttk.Combobox(destinations, textvariable=self.pin_destination, values=("Verde", "Blanca"),
                               state="readonly", width=8)
        selector.pack(side="left", padx=(4, 16))
        self._selectors.append(selector)
        self._destination_selectors.append(selector)
        tk.Label(destinations, text="Sin pin → caja:").pack(side="left")
        selector = ttk.Combobox(destinations, textvariable=self.no_pin_destination, values=("Verde", "Blanca"),
                               state="readonly", width=8)
        selector.pack(side="left", padx=4)
        self._selectors.append(selector)
        self._destination_selectors.append(selector)
        self.checkpoint = tk.StringVar(value=checkpoint_choices[0] if checkpoint_choices else "")
        if checkpoint_choices:
            checkpoint_row = tk.Frame(self.root)
            checkpoint_row.pack(fill="x", padx=8, pady=(0, 6))
            tk.Label(checkpoint_row, text="Caja blanca · Up-down BAD:").pack(side="left")
            selector = ttk.Combobox(checkpoint_row, textvariable=self.checkpoint, values=checkpoint_choices,
                                   state="readonly", width=62, height=12)
            selector.pack(side="left", padx=4)
            self._selectors.append(selector)
        self.motion = motion_controls.MotionControls(self.root, initial_motion)
        self.picture = tk.Label(self.root)
        self.picture.pack()
        self.live = tk.Label(self.root, font=("TkDefaultFont", 18), anchor="w")
        self.live.pack(fill="x")
        self.result = tk.Label(self.root, font=("TkDefaultFont", 28, "bold"), anchor="w")
        self.result.pack(fill="x")
        def window_key(event):
            if (not self._running and event.widget.winfo_class() == "TCombobox"
                    and str(event.widget.cget("state")) == "normal"):
                return
            on_key(event.keysym.lower())

        self.root.bind("<Key>", window_key)
        self.open = True
        self._image = None  # Tk drops an image that Python no longer references

    def destination_for(self, classifier_label: str) -> str:
        choice = self.pin_destination if classifier_label == "PIN" else self.no_pin_destination
        return "green" if choice.get() == "Verde" else "white"

    def apply_hmi_configuration(self, update):
        if update is None or not self.open:
            return
        active = update.get("active")
        rule = update.get("rule") if active else None
        if active:
            if not self._voice_rule_active:
                self._manual_destinations = (self.pin_destination.get(), self.no_pin_destination.get())
            with_pin = rule["routes"]["with_pin"]["destination"]
            without_pin = rule["routes"]["without_pin"]["destination"]
            self.pin_destination.set("Verde" if with_pin == "green" else "Blanca")
            self.no_pin_destination.set("Verde" if without_pin == "green" else "Blanca")
            self.voice_rule_status.configure(text="Regla HMI activa · se aplicará a la siguiente recogida")
        elif self._voice_rule_active:
            self.pin_destination.set(self._manual_destinations[0])
            self.no_pin_destination.set(self._manual_destinations[1])
            self.voice_rule_status.configure(text="Regla HMI desactivada · destinos manuales restaurados")
        else:
            self.voice_rule_status.configure(text="Regla HMI conectada · sin regla activa")
        self._voice_rule_active = active
        settings = update.get("settings")
        if settings:
            for key, value in settings.items():
                if key in self.motion.variables:
                    self.motion.variables[key].set(self.motion.display(value))
            if not active:
                self.voice_rule_status.configure(text="Ajustes HMI recibidos · próximo ciclo")
        for selector in self._destination_selectors:
            selector.configure(state="disabled" if self._running or active else "readonly")

    def checkpoint_for(self, destination: str) -> str | None:
        return self.checkpoint.get() or None

    def pick_checkpoint_for(self) -> str:
        return self.pick_checkpoint.get()

    def motion_settings(self) -> dict:
        return self.motion.read()

    def set_running(self, running: bool):
        if not self.open or running == self._running:
            return
        self._running = running
        for selector in self._selectors:
            selector.configure(state="disabled" if running or
                               (self._voice_rule_active and selector in self._destination_selectors)
                               else "readonly")
        self.motion.set_enabled(not running)

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
    parser.add_argument("--policy", default=None,
                        help="initial pick pretrained_model folder or Hugging Face repo id; default: the first of the list (v3 super, 6,000 steps)")
    parser.add_argument("--classifier", default="IkerAzurmendi/pin-classifier",
                        help="classifier model.pt, or the Hugging Face repo id that holds it")
    parser.add_argument("--policy-routes", type=Path, default=DEMO_DIR / "policy_routes.json",
                        help="JSON mapping classifier labels to post-classification policies")
    parser.add_argument("--follower-port", default=runner.DEFAULT_FOLLOWER_PORT,
                        help="follower serial path (default: the one 23_run_act_so101.py uses)")
    parser.add_argument("--leader-port", default=None,
                        help="SO-101 leader serial path (default: unique /dev/serial/by-id entry containing 5A7A054669)")
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
    parser.add_argument("--hmi-url", default="http://127.0.0.1:8790",
                        help="base URL of the HMI that publishes the active voice sorting rule")
    args = parser.parse_args()
    if args.fps <= 0 or args.pick_seconds <= 0 or args.max_step <= 0:
        parser.error("--fps, --pick-seconds and --max-step must be positive")

    import numpy as np
    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: F401
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
    from lerobot.utils.robot_utils import precise_sleep
    from comandos_so101.hmi_autopublish import install as install_hmi_telemetry

    install_hmi_telemetry(SOFollower)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classifier = PinClassifier(resolve_classifier(args.classifier), device)
    policy_registry = load_policy_registry(args.policy_routes)
    for refresh in (policy_registry.refresh_checkpoints, policy_registry.refresh_pick_checkpoints):
        try:
            refresh()
        except Exception as exc:
            print(f"Could not refresh Hugging Face checkpoints; using the local catalog: {exc}", flush=True)
    pick_choices = policy_registry.pick_checkpoint_options(args.policy)
    if not pick_choices:
        pick_choices = policy_registry.pick_checkpoint_options("IkerAzurmendi/act-pick-12000")
    initial_pick = next((label for label, source in pick_choices.items() if source == args.policy),
                        next(iter(pick_choices)))
    loaded_pick = pick_choices[initial_pick]
    policy, pre, post = load_pick_components(loaded_pick, device, policy_registry.pick_revision(loaded_pick))
    model_pick_actions = policy.config.n_action_steps
    cameras = {
        name: OpenCVCameraConfig(index_or_path=runner.find_camera(name, path), fps=args.fps,
                                 width=args.camera_width, height=args.camera_height, warmup_s=5)
        for name, path in (("front", args.camera_front), ("wrist", args.camera_wrist))
    }
    robot = make_robot_from_config(SOFollowerRobotConfig(
        port=args.follower_port, id="mi_follower", cameras=cameras, max_relative_target=args.max_step))

    keys = []
    run_settings = motion_controls.default_settings(pick_seconds=args.pick_seconds, pick_step=args.max_step,
                                                   control_hz=args.fps)
    default_settings = run_settings.copy()
    window = Window(keys.append, policy_registry.checkpoint_options("white"),
                    tuple(pick_choices), initial_pick, run_settings)
    sorting_rule_monitor = HmiSortingRuleMonitor(args.hmi_url, default_settings)
    cycle_destinations, cycle_checkpoints = None, {}
    phase, run_start, first, history, result, note = "idle", 0.0, False, [], "", ""
    control_mode = "automatic"
    home_started = None
    home_return_step = None
    home_waypoint_index = None
    manual_first_action = False
    manual_blocked = False
    leader = None
    leader_available = False
    leader_status = "No se ha detectado el leader con ID USB 5A7A054669."
    classification_enabled, classify_start, stop_requested = False, 0.0, False
    last_gripper_target = None
    route_runtime, route_started, route_released = None, 0.0, False
    route_last_gripper_target = None
    pose_history = deque()
    leader_read_error_reported = False
    stop_status, next_stop_report = "Auto-stop waits for the pick to start", 0.0
    next_show = 0.0
    obs = None
    robot.connect()
    leader_port = find_leader_port(args.leader_port)
    if leader_port:
        try:
            leader = SOLeader(SOLeaderTeleopConfig(port=leader_port, id="mi_leader"))
            leader.bus.connect()
            if leader.is_calibrated:
                leader_available = True
                leader_status = "Leader conectado; calibración verificada."
                print(f"Leader conectado en modo lectura: {leader_port}", flush=True)
            else:
                leader_status = "La calibración del leader no coincide con mi_leader; modo manual bloqueado."
                print(leader_status, flush=True)
                leader.bus.disconnect(disable_torque=False)
                leader = None
        except Exception as exc:
            leader_status = f"No se pudo abrir el leader: {exc}"
            print(f"{leader_status}; la aplicación automática sigue activa.", flush=True)
            if leader is not None and leader.bus.is_connected:
                leader.bus.disconnect(disable_torque=False)
            leader = None
    else:
        print(f"{leader_status} Conecta ese USB o indica --leader-port.", flush=True)
    try:
        period = 1 / args.fps
        while window.open:
            cycle = time.perf_counter()
            destination = route_runtime.get("destination") if route_runtime else None
            try:
                current_settings = window.motion_settings()
            except ValueError:
                current_settings = run_settings.copy()
            sorting_rule_monitor.set_state(phase, note, result, destination,
                                           current_settings, default_settings,
                                           leader_available, control_mode, leader_status)
            update = sorting_rule_monitor.latest()
            if update is not None:
                window.apply_hmi_configuration(update)
                sorting_rule_monitor.acknowledge_configuration(update["revision"])
            if sorting_rule_monitor.take_start_request():
                if phase in ("idle", "result"):
                    keys.append("space")
                    print("Starting one pick cycle after Telegram confirmation", flush=True)
                else:
                    print("Ignored Telegram start request: the pick application is busy", flush=True)
            if sorting_rule_monitor.take_home_request():
                if phase in ("idle", "result") and control_mode == "automatic":
                    home_started = cycle
                    home_return_step = robot.config.max_relative_target
                    home_waypoint_index = 0
                    robot.config.max_relative_target = min(float(current_settings["pick_step"]), HOME_MAX_STEP)
                    phase, note = "homing", "Volviendo a la posición de casa solicitada por Telegram."
                    classification_enabled = False
                    print("Returning follower to the saved home pose", flush=True)
                else:
                    print("Ignored home request: automatic mode and an idle app are required", flush=True)
            if sorting_rule_monitor.take_home_stop_request() and phase == "homing":
                stop_requested = True
            requested_mode = sorting_rule_monitor.take_control_mode()
            if requested_mode == "manual" and leader_available and phase in ("idle", "result"):
                control_mode = "manual"
                manual_first_action, manual_blocked = True, False
                phase, note = "manual", "El follower sigue al maestro; el ciclo automático está pausado."
                classification_enabled = False
                stop_requested = False
                robot.config.max_relative_target = run_settings["pick_step"]
                print("Control manual con leader activado desde la HMI", flush=True)
            elif requested_mode == "automatic" and phase == "manual":
                control_mode = "automatic"
                manual_first_action, manual_blocked = False, False
                phase, note = "idle", "Control automático listo; el follower mantiene su posición."
                print("Control manual detenido desde la HMI", flush=True)
            while keys:
                key = keys.pop(0)
                if key in ("q", "escape"):
                    window.close()
                    break
                elif key == "p":
                    if obs is None:
                        print("No follower observation available yet", flush=True)
                    else:
                        positions = [f"{joint}={float(obs[f'{joint}.pos']):.2f}"
                                     for joint in runner.JOINTS[:-1]]
                        positions.append(f"gripper={float(obs['gripper.pos']):.1f}")
                        print("Measured follower positions (arm: degrees, gripper: percent): "
                              + ", ".join(positions), flush=True)
                elif key == "space" and phase in ("idle", "result"):
                    try:
                        run_settings = window.motion_settings()
                    except ValueError as exc:
                        phase, note = "idle", f"Ajustes no válidos: {exc}"
                        print(note, flush=True)
                        continue
                    cycle_destinations = ({label: window.destination_for(label) for label in ("PIN", "NO PIN")}
                                          if hasattr(window, "destination_for") else None)
                    cycle_checkpoints = ({destination: window.checkpoint_for(destination)
                                          for destination in ("green", "white")}
                                         if cycle_destinations is not None else {})
                    selected_pick = pick_choices[window.pick_checkpoint_for()]
                    window.set_running(True)
                    if selected_pick != loaded_pick:
                        note = f"Loading pick policy: {selected_pick}"
                        if obs is not None:
                            window.show([obs["front"], obs["wrist"]], note, COLORS["idle"], "", "")
                        try:
                            policy, pre, post = load_pick_components(
                                selected_pick, device, policy_registry.pick_revision(selected_pick))
                            loaded_pick = selected_pick
                            model_pick_actions = policy.config.n_action_steps
                        except Exception as exc:
                            phase, note = "idle", f"Pick not started: {exc}"
                            print(note, flush=True)
                            window.set_running(False)
                            continue
                        cycle = time.perf_counter()
                    actions = set_action_steps(policy, run_settings["pick_actions"], model_pick_actions)
                    period = 1 / run_settings["control_hz"]
                    robot.config.max_relative_target = run_settings["pick_step"]
                    print(f"Pick policy selected: {loaded_pick}; {run_settings['pick_seconds']:g} s, "
                          f"max step {run_settings['pick_step']:g}, {run_settings['control_hz']:g} Hz, "
                          f"{actions} actions per prediction", flush=True)
                    phase, run_start, first, history, result, note = "running", cycle, True, [], "", ""
                    classification_enabled = False
                    last_gripper_target = None
                    pose_history.clear()
                    stop_status, next_stop_report = "Auto-stop: waiting for movement", cycle
                elif key == "s" and phase in ("running", "classifying", "routing", "homing"):
                    stop_requested = True
            if not window.open:
                break

            obs = robot.get_observation()
            if phase == "homing" and not stop_requested:
                home_pose = HOME_WAYPOINTS[home_waypoint_index]
                arm_error = max(abs(float(obs[f"{joint}.pos"]) - home_pose[joint])
                                for joint in runner.JOINTS[:-1])
                gripper_error = abs(float(obs["gripper.pos"]) - home_pose["gripper"])
                if arm_error <= HOME_ARM_TOLERANCE_DEG and gripper_error <= HOME_GRIPPER_TOLERANCE_PERCENT:
                    if home_waypoint_index + 1 < len(HOME_WAYPOINTS):
                        home_waypoint_index += 1
                        note = "Punto de paso alcanzado; continuando a casa."
                        print(note, flush=True)
                        home_pose = HOME_WAYPOINTS[home_waypoint_index]
                        robot.send_action({f"{joint}.pos": home_pose[joint] for joint in runner.JOINTS})
                    else:
                        robot.send_action({f"{joint}.pos": float(obs[f"{joint}.pos"]) for joint in runner.JOINTS})
                        robot.config.max_relative_target = home_return_step
                        phase, note, home_started = "idle", "Posición de casa alcanzada.", None
                        home_return_step = None
                        home_waypoint_index = None
                        print(note, flush=True)
                elif cycle - home_started >= HOME_TIMEOUT_S:
                    robot.send_action({f"{joint}.pos": float(obs[f"{joint}.pos"]) for joint in runner.JOINTS})
                    robot.config.max_relative_target = home_return_step
                    phase, note, home_started = "idle", "Vuelta a casa detenida por tiempo; follower en espera.", None
                    home_return_step = None
                    home_waypoint_index = None
                    print(note, flush=True)
                else:
                    robot.send_action({f"{joint}.pos": home_pose[joint] for joint in runner.JOINTS})
            if phase == "manual" and not manual_blocked:
                try:
                    action = leader.get_action()
                    action_gap = max(abs(float(action[f"{joint}.pos"]) - float(obs[f"{joint}.pos"]))
                                     for joint in runner.JOINTS)
                    if manual_first_action and action_gap > runner.MAX_FIRST_GAP:
                        manual_blocked = True
                        note = (f"Modo manual bloqueado: maestro y follower difieren {action_gap:.0f}; "
                                "alinea ambos brazos, selecciona Automático y vuelve a Manual.")
                        print(note, flush=True)
                    else:
                        sent_action = robot.send_action({f"{joint}.pos": float(action[f"{joint}.pos"])
                                                         for joint in runner.JOINTS})
                        last_gripper_target = sent_action["gripper.pos"]
                        leader_read_error_reported = False
                    manual_first_action = False
                except Exception as exc:
                    note = f"Sin lectura del leader; no se envían consignas: {exc}"
                    if not leader_read_error_reported:
                        print(note, flush=True)
                        leader_read_error_reported = True
            timed_out = phase == "running" and cycle - run_start >= run_settings["pick_seconds"]
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
                if first and gap > run_settings["first_gap"]:
                    phase, note = "idle", (f"not moving: the first action is {gap:.0f} degrees from the "
                                           "arm (is it at the rest pose, above the tray?)")
                else:
                    pose_history.append((cycle - run_start, state.copy(), action.copy()))
                    reached_pose, stop_status = pose_stop_status(pose_history, cycle - run_start, run_settings)
                    if not reached_pose:
                        sent_action = robot.send_action({f"{j}.pos": float(a) for j, a in zip(runner.JOINTS, action)})
                        last_gripper_target = sent_action["gripper.pos"]
                first = False

            if phase == "routing" and stop_requested:
                hold_action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in runner.JOINTS}
                if route_last_gripper_target is not None:
                    hold_action["gripper.pos"] = route_last_gripper_target
                robot.send_action(hold_action)
                robot.config.max_relative_target = run_settings["pick_step"]
                phase, stop_requested = "result", False
                note = f"{route_runtime['route'].name} stopped; holding pose"
                print(note, flush=True)
            elif phase != "routing" and (stop_requested or (phase == "running" and (timed_out or reached_pose))):
                hold_action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in runner.JOINTS}
                # Preserve the closing target so holding the arm does not release gripper pressure.
                if last_gripper_target is not None:
                    hold_action["gripper.pos"] = last_gripper_target
                robot.send_action(hold_action)
                history, result = [], ""
                classification_enabled = not stop_requested
                phase = "classifying" if classification_enabled else "idle"
                if home_return_step is not None:
                    robot.config.max_relative_target = home_return_step
                home_started = None
                home_return_step = None
                home_waypoint_index = None
                reason = "position reached" if reached_pose else "time limit reached"
                note = f"{reason}; holding pose" if classification_enabled else "stopped; holding pose"
                print(f"Pick stopped: {note}", flush=True)
                if reached_pose:
                    print(stop_status, flush=True)
                classify_start = cycle + run_settings["settle_seconds"]
                stop_requested = False

            if cycle >= next_show:
                live = {"idle": "Classifier waits until the pick finishes",
                        "running": stop_status,
                        "classifying": "Holding pose; waiting for the camera view to settle",
                        "routing": f"Running post-classification policy: {route_runtime['route'].name if route_runtime else 'policy'}",
                        "result": "Classification finished; motors holding pose",
                        "manual": "Follower controlled by the leader; automatic cycle paused",
                        "homing": "Moving to the saved home pose"}[phase]
                if phase == "running" and cycle >= next_stop_report:
                    print(stop_status, flush=True)
                    next_stop_report = cycle + 1.0
                if classification_enabled and cycle >= classify_start:
                    p = classifier(obs["front"], obs["wrist"])
                    history.append((cycle, p))
                    threshold = run_settings["pin_threshold"] / 100.0
                    live = "classifier now: " + label(p, threshold)
                    if cycle - history[0][0] >= run_settings["classify_seconds"]:
                        probability = sum(q for _, q in history) / len(history)
                        classifier_label = label(probability, threshold).split(" (")[0]
                        result = f"RESULT: {label(probability, threshold)}"
                        classification_enabled = False
                        try:
                            if cycle_destinations is not None:
                                destination = cycle_destinations[classifier_label]
                                route = policy_registry.resolve_destination(
                                    destination, cycle_checkpoints[destination])
                            else:
                                destination = "white"
                                route = policy_registry.resolve(classifier_label)
                            route_started, route_released = cycle, False
                            route_last_gripper_target = last_gripper_target
                            route_runtime = {"route": route, "runner": None, "policy": None,
                                             "pre": None, "post": None, "features": None,
                                             "destination": destination,
                                             "max_seconds": run_settings[f"{destination}_seconds"] or route.max_seconds}
                            live = f"Loading policy: {route.name}"
                            window.show([obs["front"], obs["wrist"]], f"CLASSIFIED · {note}",
                                        COLORS["routing"], live, result)
                            (route_runtime["runner"], route_runtime["policy"], route_runtime["pre"],
                             route_runtime["post"], route_runtime["features"]) = load_route_components(route, device)
                            if run_settings[f"{destination}_actions"] is not None:
                                set_action_steps(route_runtime["policy"], run_settings[f"{destination}_actions"],
                                                 route_runtime["policy"].config.n_action_steps)
                            obs = robot.get_observation()
                            cycle = route_started = time.perf_counter()
                            robot.config.max_relative_target = (run_settings[f"{destination}_step"]
                                                                or route.max_relative_target)
                            phase = "routing"
                            note = f"{classifier_label} -> {destination} -> {route.name}"
                            print(f"Policy selected after classification: {note}; "
                                  f"{route_runtime['max_seconds']:g} s, "
                                  f"max step {robot.config.max_relative_target:g}", flush=True)
                        except Exception as exc:
                            phase = "result"
                            note = f"Policy not started; holding pose: {exc}"
                            print(note, flush=True)
                status = {"idle": "READY - SPACE starts the pick",
                          "running": f"PICKING {cycle - run_start:4.1f} s / {run_settings['pick_seconds']:g} s - S stops",
                          "classifying": "CLASSIFYING - motors holding pose - S cancels",
                          "routing": f"ROUTING {route_runtime['route'].name if route_runtime else 'policy'} - S stops",
                          "result": "DONE - SPACE picks again",
                          "manual": "LEADER CONTROL - change mode in the HMI",
                          "homing": "RETURNING HOME - S stops"}[phase]
                window.set_running(phase in ("running", "classifying", "routing", "manual", "homing"))
                window.show([obs["front"], obs["wrist"]], f"{status}   {note}", COLORS[phase],
                            live, result)
                next_show = cycle + PREVIEW_INTERVAL

            if phase == "routing" and not stop_requested:
                route = route_runtime["route"]
                route_runner = route_runtime["runner"]
                destination = route_runtime["destination"]
                state = np.array([obs[f"{j}.pos"] for j in runner.JOINTS], dtype=np.float32)
                route_released = route_released or state[-1] > run_settings[f"{destination}_released"]
                elapsed = cycle - route_started
                finished = run_settings[f"{destination}_finish"] and route_runner.finished(state, route_released)
                timed_out = elapsed >= route_runtime["max_seconds"]
                if finished or timed_out:
                    hold_action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in runner.JOINTS}
                    if route_last_gripper_target is not None:
                        hold_action["gripper.pos"] = route_last_gripper_target
                    robot.send_action(hold_action)
                    robot.config.max_relative_target = run_settings["pick_step"]
                    phase = "result"
                    route_status = "finished" if finished else "time limit reached"
                    note = f"{route.name} {route_status}; holding pose"
                    print(note, flush=True)
                else:
                    action = route_runner.step(route_runtime["policy"], route_runtime["pre"],
                                               route_runtime["post"], route_runtime["features"], obs, device)
                    action = add_route_clearance(action, route_released, state, run_settings, destination)
                    sent_action = robot.send_action(action)
                    route_last_gripper_target = sent_action["gripper.pos"]
            window.set_running(phase in ("running", "classifying", "routing", "manual", "homing"))
            precise_sleep(max(period - (time.perf_counter() - cycle), 0.0))
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        sorting_rule_monitor.close()
        if leader is not None and leader.bus.is_connected:
            leader.bus.disconnect(disable_torque=False)
        robot.disconnect()
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
