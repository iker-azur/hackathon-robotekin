"""The full bin-sorting demo in one window: voice order -> wait for a piece -> pick -> pin classifier -> box policy.

One Tk window over the real SO-101 follower, built on ``24_pick_and_classify_app.py`` (it reuses its model loaders,
classifier wrapper, auto-stop and route clearance) and adding the rest of the pipeline:

    1. LISTEN   V (or the button) records one order; faster-whisper detects Spanish / English / Basque and writes it
                down; ``IkerAzurmendi/bin-intent-gliner2`` gives the probability of every option and the order is
                kept ("good pieces to the white box, bad ones to the green box", "all pieces to the green box",
                "two good pieces to the green box", "stop"). Keys 1 / 2 / 3 set the usual orders without voice.
                The same sentence can be typed in the "Orden escrita" box, or sent to a Telegram bot as text or a
                voice note (``--telegram``, see ``demo_telegram.py``); the verdict is replied in the chat.
    2. WAIT     with an order active, the front camera is watched; the robot does not move until a piece has lain
                in the tray for a second (``tray_detect.py``). An empty tray means waiting, indefinitely.
    3. PICK     the pick ACT policy (default ``IkerAzurmendi/act-pick-v2-30000``) lifts the piece to the camera;
                the arm holds the pose and the gripper keeps its grip. The pick is over when ``reach_detect.py`` sees
                the grasp (gripper opened, then closed), a lift of 30 degrees and 0.75 s of stillness: about a
                second after the arm stops, not at the time limit; the hold then pushes with the policy's own target
                (the torque the demonstrations used) and about 1.8 s of classifier frames are averaged. If the time
                limit is reached with nothing lifted (the arm stays down, trying), the attempt failed: the arm
                walks back to rest, waits
                ``--retry-wait`` (3 s) and tries again, ``--pick-attempts`` (3) times in all; after the last one the
                order is cleared and the app asks you to check the tray. Needs the "Parada automática" setting on
                (it is by default): without it the time limit is the normal end of a pick and nothing is retried.
    4. CLASSIFY the two-camera pin classifier averages 1.5 s of frames: PIN or NO PIN.
    5. ROUTE    the order decides the box: green (big) -> ``up-down-good``, white (small) -> ``up-down-bad``. A piece
                with no rule is held until an order that covers it arrives.
    6. REPEAT   back at rest, the loop waits for the next piece, until the order is finished or you say stop.

Keys (the window must have focus, not the typed-order box):
    V listen   S stop and hold   R resume (the arm returns to rest; take the piece out of the gripper first)
    F pick now without waiting for the tray detector   C clear the order   1 / 2 / 3 preset orders   Q / Esc quit
    P party mode (also any sentence with party / fiesta / festa / jai..., which skips the intent model): the arm rises
      slowly to a standard pose taken from the demonstrations and dances inside their envelope (``party_dance.py``,
      ``party_profile.json``) until S or "stop"; it then walks back to rest. Only when no piece is being handled.

Motion: the apps used to cap every command at 3-8 degrees from the measured pose. In the demonstrations the follower
trails the leader by 10-29 degrees (95th-99th percentile) and the motor torque is proportional to that gap, so the cap
starved the motors: slow joints and a weak grip. This app caps at ``--arm-step`` (30) and ``--gripper-step`` (45)
per command, eased in over the first 0.7 s of each policy segment. See the README for the measurements.

    python 30_full_demo_app.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
PREVIEW_INTERVAL = 1 / 6  # camera view and classifier refresh, as in 24
DETECT_INTERVAL = 0.15
PIECE_STABLE_S = 1.0  # a piece must lie in view this long before the robot moves
RAMP_START_DEG, RAMP_S = 8.0, 0.7  # the command cap grows from 8 degrees to the full cap over this time
RETURN_STEP, RETURN_SECONDS = 15.0, 6.0
REST_SLACK = 1.5  # times the up-down policies' own rest tolerance
FAILURES_BEFORE_PAUSE = 2
DEFAULT_PICK = "IkerAzurmendi/act-pick-v2-30000"
DEFAULT_CLASSIFIER = "IkerAzurmendi/pin-classifier"
DEFAULT_INTENT = "IkerAzurmendi/bin-intent-gliner2"
SETTLE_S, CLASSIFY_S = 0.3, 1.5  # after the reach: a short pause, then 1.5 s of classifier frames, averaged
DEFAULT_PICK_SECONDS = 15.0  # 97% of the recorded picks (rest to piece raised to the camera) take less
PICK_ATTEMPTS, RETRY_WAIT_S = 3, 3.0  # a pick that times out goes back to rest, waits, and tries again
PARTY = ("party_up", "party")
PARTY_STEP = 12.0  # largest command offset while rising and dancing: the targets are smooth, so it never bites
PARTY_LAG_DEG, PARTY_LAG_S = 20.0, 1.0  # an arm joint this far behind its target this long is blocked: stop
PARTY_SETTLE_S = 1.0  # after the rise time, how long to wait before the dance starts fading in
MOVING = ("picking", "classifying", "holding", "routing", "returning", *PARTY)
PHASES = {
    "retry_wait": ("REINTENTO: en reposo, esperando para volver a intentarlo", "#ef6c00"),
    "party_up": ("FIESTA: subiendo despacio a la posición de baile", "#d81b60"),
    "party": ("FIESTA: bailando · S o «stop» para terminar", "#8e24aa"),
    "loading": ("CARGANDO MODELOS", "#555555"),
    "wait_order": ("ESPERANDO ORDEN - pulsa V y di qué hacer con las piezas", "#1f7a3a"),
    "wait_piece": ("ESPERANDO PIEZA EN LA BANDEJA", "#8a6d00"),
    "picking": ("RECOGIENDO", "#b3261e"),
    "classifying": ("CLASIFICANDO: con / sin perno", "#1565c0"),
    "holding": ("PIEZA SIN REGLA - di a qué caja va", "#c2185b"),
    "routing": ("LLEVANDO A LA CAJA", "#7b4ca0"),
    "returning": ("VOLVIENDO A REPOSO", "#00796b"),
    "paused": ("PARADO - R reanuda", "#444444"),
}


def load_script(name: str, key: str | None = None):
    spec = importlib.util.spec_from_file_location(key or name.split(".")[0], DEMO_DIR / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app24 = load_script("24_pick_and_classify_app.py", "pick_and_classify_app")
motion_controls = app24.motion_controls
tray_detect = load_script("tray_detect.py")
demo_orders = load_script("demo_orders.py")
demo_voice = load_script("demo_voice.py")
demo_telegram = load_script("demo_telegram.py")
party_dance = load_script("party_dance.py")
reach_detect = load_script("reach_detect.py")
smooth_motion = load_script("smooth_motion.py")
BUS_ERRORS_BEFORE_PAUSE, BUS_ERRORS_BEFORE_EXIT = 3, 100  # failed control cycles in a row (the loop waits 0.1 s each)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def step_limits(arm: float, gripper: float) -> dict:
    """Per-joint ``max_relative_target``: the gripper may squeeze further than the arm may lead."""
    return {**{joint: float(arm) for joint in JOINTS[:-1]}, "gripper": float(max(arm, gripper))}


def ramped(full: float, elapsed: float) -> float:
    return full if full <= RAMP_START_DEG else min(full, RAMP_START_DEG + (full - RAMP_START_DEG) * elapsed / RAMP_S)


def format_probabilities(reading: dict, width: int = 3) -> str:
    lines = []
    for task, options in reading["probabilities"].items():
        ranked = sorted(options.items(), key=lambda item: -item[1])[:width]
        lines.append(f"{task:8}" + "   ".join(f"{label} {p:.0%}" for label, p in ranked))
    lines.append(f"estado {reading['status']}   confianza {reading['confidence']:.0%}")
    return "\n".join(lines)


class Models:
    """Everything that takes seconds to load, loaded once in a thread while the window is already up."""

    def __init__(self, args, device, registry, initial_pick, white_label):
        self.args, self.device, self.registry = args, device, registry
        self.initial_pick, self.white_label = initial_pick, white_label
        self.ready, self.error, self.status = False, None, "iniciando"
        self.classifier = None
        self._pick, self._routes = {}, {}
        self.lock = threading.Lock()
        self.smooth = None if getattr(args, "no_smooth", True) else {"stride": args.blend_stride, "coeff": args.blend_coeff}

    def _blend(self, policy):
        """ACT policies send the average of their overlapping chunks, predicted off the control thread."""
        if self.smooth and hasattr(policy, "predict_action_chunk"):
            return smooth_motion.BlendedPolicy(policy, self.smooth["stride"], self.smooth["coeff"], threaded=True)
        return policy

    def start(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.status = "clasificador de perno"
            self.classifier = app24.PinClassifier(app24.resolve_classifier(self.args.classifier), self.device)
            self.status = "política de recogida"
            self.pick(self.initial_pick)
            for destination in ("green", "white"):
                self.status = f"política de la caja {destination}"
                self.route(destination, self.white_label if destination == "white" else None)
            self.ready = True
        except Exception as exc:
            traceback.print_exc()
            self.error = f"{type(exc).__name__}: {exc}"

    def pick(self, source):
        """(policy, pre, post, the checkpoint's own actions per prediction)"""
        with self.lock:
            if source not in self._pick:
                policy, pre, post = app24.load_pick_components(source, self.device, self.registry.pick_revision(source))
                policy = self._blend(policy)
                self._pick[source] = (policy, pre, post, policy.config.n_action_steps)
            return self._pick[source]

    def route(self, destination, label):
        with self.lock:
            key = (destination, label)
            if key not in self._routes:
                route = self.registry.resolve_destination(destination, label)
                runner, policy, pre, post, features = app24.load_route_components(route, self.device)
                policy = self._blend(policy)
                self._routes[key] ={"route": route, "runner": runner, "policy": policy, "pre": pre, "post": post,
                                     "features": features, "default_actions": policy.config.n_action_steps}
            return self._routes[key]

    def classify(self, front, wrist) -> float:
        return self.classifier(front, wrist)


class Demo:
    """The cycle, advanced by ``tick``; it owns no window, microphone or hardware setup, so a test can drive it."""

    def __init__(self, robot, models, orders, settings, choices, detect, rest_pose, rest_tolerance, device,
                 gripper_step=45.0, clock=None, log=print, party_profile=None, pick_attempts=PICK_ATTEMPTS,
                 retry_wait=RETRY_WAIT_S, reach_profile=None, target_filter=None):
        self.robot, self.models, self.orders = robot, models, orders
        self.target_filter, self._state = target_filter, None
        self.reach_profile, self.reach = reach_profile or reach_detect.load_profile(), None
        self.pick_attempts, self.retry_wait, self.attempts = pick_attempts, retry_wait, 0
        self.dance = party_dance.Choreography(party_profile or party_dance.load_profile())
        self.party_requested = self.party_end = False
        self.settings, self.choices, self.detect = settings, choices, detect
        self.rest_pose, self.rest_tolerance = np.asarray(rest_pose, dtype=np.float32), np.asarray(rest_tolerance)
        self.device, self.gripper_step, self.clock, self.log = device, gripper_step, clock or time.perf_counter, log
        self.phase, self.note, self.live, self.result = "loading", "", "", ""
        self.period = 1 / 30
        self.tray = None
        self.watcher = tray_detect.PieceWatcher(PIECE_STABLE_S)
        self.next_detect, self.piece_ready = 0.0, False
        self.force = self.stop_requested = self.resume_requested = False
        self.failures = 0
        self.delivered = {"green": 0, "white": 0}
        self.piece_class = None
        self.last_gripper = None

    # -- requests from the window and the voice thread -------------------------------------------------------

    def request_stop(self):
        if self.phase in PARTY:  # nothing is held: finish the dance and walk back to rest instead of freezing
            self.party_end = True
        elif self.phase in MOVING:
            self.stop_requested = True

    def request_party(self) -> bool:
        """Party mode starts only when the arm holds nothing and no cycle is running; False says it was refused."""
        if self.phase not in ("wait_order", "wait_piece"):
            return False
        self.party_requested = True
        return True

    def request_resume(self):
        if self.phase == "paused":
            self.resume_requested = True

    def request_force(self):
        if self.phase in ("wait_order", "wait_piece"):
            self.force = True
            self.note = "recogida forzada"

    def bus_error(self, note):
        """The motor bus stopped answering: stop commanding; the cycle resumes with R once the arm answers again."""
        if self.phase not in ("paused", "loading"):
            self._go("paused", note)

    # -- helpers ----------------------------------------------------------------------------------------------

    def status_text(self) -> str:
        label = PHASES[self.phase][0]
        if self.phase == "loading":
            label += f" ({self.models.status})"
        return f"{label}   {self.note}".strip()

    @property
    def color(self) -> str:
        return PHASES[self.phase][1]

    def near_rest(self, state, slack=REST_SLACK) -> bool:
        return bool(np.all(np.abs(state - self.rest_pose) <= self.rest_tolerance * slack))

    def _hold_action(self, obs, arm_target=None) -> dict:
        """Hold where the arm is. With ``arm_target`` (the firm hold) the arm joints are commanded a little past the
        measured pose, which is what makes the motors push: a target equal to the pose commands no torque at all."""
        action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS}
        if arm_target is not None:
            action.update({f"{j}.pos": float(v) for j, v in zip(JOINTS, arm_target)})
        if self.last_gripper is not None:  # keep the closing target so holding does not release the grip
            action["gripper.pos"] = self.last_gripper
        return action

    def _send(self, action, step, smooth=False):
        """``smooth``: a policy's command, low-passed before it is sent (starting from the measured pose)."""
        if smooth and self.target_filter is not None:
            if self.target_filter.x is None:
                self.target_filter.reset(self._state)
            vector = self.target_filter([action[f"{j}.pos"] for j in JOINTS], self.period)
            action = {f"{j}.pos": float(v) for j, v in zip(JOINTS, vector)}
        self.robot.config.max_relative_target = step_limits(step, self.gripper_step)
        sent = self.robot.send_action(action)
        self.last_gripper = sent["gripper.pos"]
        return sent

    def _go(self, phase, note=""):
        self.phase, self.note = phase, note
        self.log(f"[{phase}] {note}" if note else f"[{phase}]", flush=True)

    def _rewatch(self):
        self.watcher.reset()
        self.piece_ready = self.force = False

    # -- the cycle ---------------------------------------------------------------------------------------------

    def tick(self, obs):
        now = self.clock()
        state = self._state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        if self.phase == "loading":
            if self.models.error:
                self.note = f"no se pudo cargar: {self.models.error}"
                return
            if not self.models.ready:
                return
            self._go("wait_order")
        if self.party_end and self.phase in PARTY:
            self.party_end = False
            self._begin_return(state, "fin de la fiesta; vuelvo a reposo")
            return
        if self.party_requested and self.phase in ("wait_order", "wait_piece"):
            self.party_requested = False
            self._start_party(state)
            return
        if self.stop_requested:
            self.stop_requested = False
            self.attempts = 0
            self.robot.config.max_relative_target = step_limits(RETURN_STEP, self.gripper_step)
            self.robot.send_action(self._hold_action(obs))
            self._go("paused", "parado; manteniendo la pose")
            return
        getattr(self, f"_tick_{self.phase}")(obs, state, now)

    def _tick_wait_order(self, obs, state, now):
        if self.orders.active:
            self._rewatch()
            self._go("wait_piece")
        else:
            self.live = "Sin orden: el robot no se mueve"

    def _tick_wait_piece(self, obs, state, now):
        if not self.orders.active:
            self._go("wait_order")
            return
        if now >= self.next_detect:
            self.tray = self.detect(obs["front"])
            self.next_detect = now + DETECT_INTERVAL
            self.piece_ready = self.watcher.update(self.tray.present, now)
        at_rest = self.near_rest(state)
        if not (self.piece_ready or self.force):
            self.live = "Bandeja vacía: esperando una pieza"
        elif not (at_rest or self.force):
            self.live = "Pieza a la vista, pero el brazo no está en reposo (F fuerza la recogida)"
        else:
            self._start_pick()

    def _start_pick(self):
        try:
            settings = self.settings()
        except ValueError as exc:
            self.note = f"Ajustes no válidos: {exc}"
            self.log(self.note, flush=True)
            return
        source = self.choices()[0]
        self._go("picking", f"intento {self.attempts + 1}/{self.pick_attempts}" if self.attempts else "")
        policy, pre, post, default_actions = self.models.pick(source)
        actions = app24.set_action_steps(policy, settings["pick_actions"], default_actions)
        if isinstance(policy, smooth_motion.BlendedPolicy) and settings["pick_actions"]:
            policy.stride = int(settings["pick_actions"])  # blended: a new prediction every this many steps
        if self.target_filter is not None:
            self.target_filter.reset()
        self.run, self.pick_models = settings, (policy, pre, post)
        self.period = 1 / settings["control_hz"]
        self.start, self.first = self.clock(), True
        self.reach = reach_detect.ReachDetector(
            self.reach_profile, window=settings["pose_window"], motion=settings["arm_motion"],
            error=settings["arm_error"], gripper_motion=settings["gripper_motion"], min_time=settings["min_pick"])
        self.last_gripper, self.result, self.piece_class = None, "", None
        self._rewatch()
        self.live = "esperando movimiento"
        self.log(f"Pick policy {source}: {settings['pick_seconds']:g} s, step {settings['pick_step']:g}, "
                 f"{settings['control_hz']:g} Hz, {actions} actions per prediction", flush=True)

    def _pick_action(self, obs, state):
        import torch

        policy, pre, post = self.pick_models
        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device), "task": "", "robot_type": ""}
        for name in ("front", "wrist"):
            image = torch.from_numpy(obs[name]).float().div(255).permute(2, 0, 1)
            batch[f"observation.images.{name}"] = image.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return post(policy.select_action(pre(batch))).squeeze(0).cpu().numpy()

    def _tick_picking(self, obs, state, now):
        s, elapsed = self.run, now - self.start
        reached = False
        if elapsed < s["pick_seconds"]:
            action = self._pick_action(obs, state)
            gap = float(np.abs(action - state).max())
            if self.first and gap > s["first_gap"]:
                self.failures += 1
                self._rewatch()
                note = (f"no me muevo: la primera acción está a {gap:.0f}° del brazo "
                        "(¿está en la pose de reposo, sobre la bandeja?)")
                self._go("paused" if self.failures >= FAILURES_BEFORE_PAUSE else "wait_piece", note)
                return
            self.first = False
            if s["auto_stop"]:
                reached, self.live = self.reach.update(elapsed, state, action)
            else:
                self.live = "Parada automática desactivada: la recogida dura el tiempo máximo"
            if not reached:
                self._send({f"{j}.pos": float(a) for j, a in zip(JOINTS, action)}, ramped(s["pick_step"], elapsed),
                           smooth=True)
                return
        lifted = bool(s["auto_stop"] and self.reach.lifted)
        if s["auto_stop"] and not (reached or lifted):  # time ran out and the arm never lifted anything: still down
            self._pick_failed(state)
            return
        self.robot.config.max_relative_target = step_limits(s["pick_step"], self.gripper_step)
        # at the top the arm holds the piece firmly: the policy's own target, not the pose it happens to be in
        self.robot.send_action(self._hold_action(obs, self.reach.hold_target(state) if s["auto_stop"] else None))
        self.cls_history, self.cls_start, self.next_classify = [], now + s["settle_seconds"], 0.0
        self.failures = self.attempts = 0
        self._go("classifying", f"posición alcanzada a los {elapsed:.1f} s" if reached else
                 "tiempo máximo con la pieza levantada" if lifted else "tiempo máximo")

    def _pick_failed(self, state):
        """The pick policy used its whole time without finishing: back to rest, then try again, ``pick_attempts`` in all."""
        self.attempts += 1
        seconds = f"{self.run['pick_seconds']:g} s"
        if self.attempts >= self.pick_attempts:
            self.attempts = 0
            self._begin_return(state, f"{self.pick_attempts} intentos de {seconds} sin coger la pieza; vuelvo al inicio",
                               then="give_up")
        else:
            self._begin_return(state, f"intento {self.attempts}/{self.pick_attempts} sin éxito tras {seconds}; "
                                      "vuelvo al inicio", then="retry")

    def _tick_retry_wait(self, obs, state, now):
        if not self.orders.active:
            self.attempts = 0
            self._go("wait_order")
            return
        if now >= self.next_detect:
            self.tray = self.detect(obs["front"])
            self.next_detect = now + DETECT_INTERVAL
        waited = now - self.retry_start
        self.live = f"Reintento {self.attempts + 1}/{self.pick_attempts} en {max(self.retry_wait - waited, 0.0):.0f} s"
        if waited < self.retry_wait:
            return
        if self.tray is not None and self.tray.present:
            self._start_pick()
        else:  # the piece is gone (fell out of view, or somebody took it): it is a new situation
            self.attempts = 0
            self._rewatch()
            self._go("wait_piece", "la pieza ya no se ve en la bandeja")

    def _tick_classifying(self, obs, state, now):
        s = self.run
        if now < self.cls_start:
            self.live = "Sujetando la pieza; esperando a que la imagen se estabilice"
            return
        if now < self.next_classify:
            return
        self.next_classify = now + PREVIEW_INTERVAL
        p = self.models.classify(obs["front"], obs["wrist"])
        self.cls_history.append((now, p))
        threshold = s["pin_threshold"] / 100.0
        self.live = "clasificador ahora: " + app24.label(p, threshold)
        if now - self.cls_history[0][0] < s["classify_seconds"]:
            return
        probability = sum(q for _, q in self.cls_history) / len(self.cls_history)
        self.piece_class = app24.label(probability, threshold).split(" (")[0]
        self.result = f"RESULTADO: {app24.label(probability, threshold)}"
        self._decide()

    def _decide(self):
        destination = self.orders.destination_for(self.piece_class)
        if destination is None:
            self._go("holding", f"pieza {self.piece_class} sin regla: dime a qué caja va")
        elif destination == demo_orders.HOLD:
            self.orders.completed(self.piece_class)
            self._go("paused", "pieza recogida (orden: solo recoger); quítala y pulsa R")
        else:
            self._start_route(destination)

    def _tick_holding(self, obs, state, now):
        self.live = "Sujetando la pieza: esperando una orden que la cubra"
        if self.orders.destination_for(self.piece_class) is not None:
            self._decide()

    def _start_route(self, destination):
        s = self.run
        label = self.choices()[1] if destination == "white" else None
        try:
            runtime = self.models.route(destination, label)
        except Exception as exc:
            self._go("paused", f"no se pudo cargar la política de la caja {destination}: {exc}")
            return
        route, policy = runtime["route"], runtime["policy"]
        app24.set_action_steps(policy, s[f"{destination}_actions"] or runtime["default_actions"],
                               runtime["default_actions"])
        if isinstance(policy, smooth_motion.BlendedPolicy) and s[f"{destination}_actions"]:
            policy.stride = int(s[f"{destination}_actions"])
        if self.target_filter is not None:
            self.target_filter.reset()
        self.rt = {**runtime, "destination": destination, "start": self.clock(), "released": False,
                   "seconds": s[f"{destination}_seconds"] or route.max_seconds,
                   "step": s[f"{destination}_step"] or route.max_relative_target}
        self._go("routing", f"{self.piece_class} -> caja {destination}")
        self.live = f"Política {route.name}"
        self.log(f"Route {route.name}: {self.rt['seconds']:g} s, step {self.rt['step']:g}", flush=True)

    def _tick_routing(self, obs, state, now):
        s, rt = self.run, self.rt
        destination, elapsed = rt["destination"], now - rt["start"]
        rt["released"] = rt["released"] or bool(state[-1] > s[f"{destination}_released"])
        finished = bool(s[f"{destination}_finish"] and rt["runner"].finished(state, rt["released"]))
        if finished or elapsed >= rt["seconds"]:
            self._route_done(obs, state, finished)
            return
        action = rt["runner"].step(rt["policy"], rt["pre"], rt["post"], rt["features"], obs, self.device)
        action = app24.add_route_clearance(action, rt["released"], state, s, destination)
        self._send(action, ramped(rt["step"], elapsed), smooth=True)

    def _route_done(self, obs, state, finished):
        rt = self.rt
        if rt["released"]:
            self.orders.completed(self.piece_class)
            self.delivered[rt["destination"]] += 1
        if finished:
            self._rewatch()
            self._go("wait_piece" if self.orders.active else "wait_order",
                     f"entregadas: verde {self.delivered['green']} · blanca {self.delivered['white']}")
        elif rt["released"]:
            self._begin_return(state, "tiempo agotado tras soltar; vuelvo a reposo")
        else:
            self.robot.send_action(self._hold_action(obs))
            self._go("paused", "la pieza no se soltó: retírala de la pinza y pulsa R")

    def _begin_return(self, state, note, then=None):
        self.ret = {"from": state.copy(), "start": self.clock(), "seconds": RETURN_SECONDS, "then": then}
        self._go("returning", note)

    def _tick_returning(self, obs, state, now):
        ret = self.ret
        alpha = min(1.0, (now - ret["start"]) / ret["seconds"])
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        target = ret["from"] + (self.rest_pose - ret["from"]) * eased
        self._send({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)}, RETURN_STEP)
        self.live = f"Volviendo a reposo: {alpha:.0%}"
        if alpha >= 1.0 and (self.near_rest(state, 1.0) or now - ret["start"] > ret["seconds"] + 3.0):
            self._rewatch()
            if ret["then"] == "retry":
                self.retry_start, self.tray, self.next_detect = now, None, 0.0
                self._go("retry_wait", f"intento {self.attempts + 1}/{self.pick_attempts} en {self.retry_wait:g} s")
            elif ret["then"] == "give_up":
                self.orders.clear()  # the piece is still there: without this the order would pick it forever
                self._go("wait_order", "no he podido coger la pieza: revisa la bandeja y repite la orden")
            else:
                self._go("wait_piece" if self.orders.active else "wait_order")

    # -- party mode: rise slowly to the standard pose, dance, and walk back to rest on stop ------------------------

    def _start_party(self, state):
        home = self.dance.home
        self.pt = {"from": state.copy(), "start": self.clock(), "seconds": party_dance.rise_seconds(state, home),
                   "lag_since": None}
        self.last_gripper, self.result, self.period = None, "", 1 / 30
        self._go("party_up", f"{self.pt['seconds']:.0f} s hasta la posición de baile")

    def _party_send(self, obs, state, target, now) -> bool:
        """Send a party target unless the arm is not following it (blocked): then hold the pose and pause."""
        lag = float(np.abs(target[:5] - state[:5]).max())
        if lag > PARTY_LAG_DEG:
            if self.pt["lag_since"] is None:
                self.pt["lag_since"] = now
            if now - self.pt["lag_since"] > PARTY_LAG_S:
                self.robot.config.max_relative_target = step_limits(RETURN_STEP, self.gripper_step)
                self.robot.send_action(self._hold_action(obs))
                self._go("paused", f"fiesta cancelada: el brazo va {lag:.0f}° por detrás de la consigna (¿choca?)")
                return False
        else:
            self.pt["lag_since"] = None
        self._send({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)}, PARTY_STEP)
        return True

    def _tick_party_up(self, obs, state, now):
        pt = self.pt
        elapsed = now - pt["start"]
        target = party_dance.rise_target(pt["from"], self.dance.home, elapsed, pt["seconds"])
        if not self._party_send(obs, state, target, now):
            return
        self.live = f"Subiendo: {min(elapsed / pt['seconds'], 1.0):.0%}"
        if elapsed >= pt["seconds"] + PARTY_SETTLE_S:
            pt["dance_start"] = now
            self._go("party", "¡a bailar!")

    def _tick_party(self, obs, state, now):
        seconds = now - self.pt["dance_start"]
        if not self._party_send(obs, state, self.dance.target(seconds), now):
            return
        self.live = f"Bailando · compás {int(self.dance.beat(seconds)) + 1}"

    def _tick_paused(self, obs, state, now):
        self.live = "Parado: los motores mantienen la pose"
        if self.resume_requested:
            self.resume_requested, self.failures, self.attempts = False, 0, 0
            self.last_gripper = None
            self._begin_return(state, "reanudando: vuelvo a reposo")


class Window:
    """Both cameras, the order, what was heard and its probabilities, and the controls."""

    def __init__(self, post, pick_choices, initial_pick, white_choices, initial_motion):
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self.post = post
        self.root = tk.Tk()
        self.root.title("SO-101 · demo completa: voz → pieza → clasificación → caja")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.Label(self.root, font=("TkDefaultFont", 16, "bold"), fg="white", anchor="w")
        self.status.pack(fill="x")
        self.picture = tk.Label(self.root)
        self.picture.pack()
        self.orders = tk.Label(self.root, font=("TkDefaultFont", 15, "bold"), anchor="w")
        self.orders.pack(fill="x", padx=8)
        self.voice = tk.Label(self.root, font=("TkDefaultFont", 12), anchor="w", justify="left", wraplength=1100)
        self.voice.pack(fill="x", padx=8)
        self.probs = tk.Label(self.root, font=("Courier", 10), anchor="w", justify="left")
        self.probs.pack(fill="x", padx=8)
        self.live = tk.Label(self.root, font=("TkDefaultFont", 12), anchor="w")
        self.live.pack(fill="x", padx=8)
        self.result = tk.Label(self.root, font=("TkDefaultFont", 22, "bold"), anchor="w")
        self.result.pack(fill="x", padx=8)

        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=8, pady=4)
        for text, event in (("🎤 Hablar (V)", ("listen",)), ("Parar (S)", ("key", "s")), ("Reanudar (R)", ("key", "r")),
                            ("Forzar recogida (F)", ("key", "f")), ("Borrar orden (C)", ("key", "c")),
                            ("🎉 Fiesta (P)", ("key", "p")),
                            ("1 Buenas→blanca, malas→verde", ("preset", "sort")),
                            ("2 Todas→verde", ("preset", "all_green")), ("3 Todas→blanca", ("preset", "all_white"))):
            tk.Button(buttons, text=text, command=lambda event=event: post(event)).pack(side="left", padx=2)
        self.hands_free = tk.BooleanVar(value=False)
        tk.Checkbutton(buttons, text="Escucha continua", variable=self.hands_free,
                       command=lambda: post(("hands_free", self.hands_free.get()))).pack(side="left", padx=8)

        typed = tk.Frame(self.root)
        typed.pack(fill="x", padx=8, pady=2)
        tk.Label(typed, text="Orden escrita:").pack(side="left")
        self.entry = tk.Entry(typed, width=70)
        self.entry.pack(side="left", padx=4)
        self.entry.bind("<Return>", self._send_text)
        tk.Button(typed, text="Enviar", command=self._send_text).pack(side="left")

        self._selectors = []
        selectors = tk.Frame(self.root)
        selectors.pack(fill="x", padx=8, pady=2)
        self.pick_checkpoint = tk.StringVar(value=initial_pick)
        tk.Label(selectors, text="Recogida:").pack(side="left")
        self._selectors.append(ttk.Combobox(selectors, textvariable=self.pick_checkpoint, values=list(pick_choices),
                                           state="readonly", width=48))
        self._selectors[-1].pack(side="left", padx=4)
        self.white_checkpoint = tk.StringVar(value=white_choices[0] if white_choices else "")
        if white_choices:
            tk.Label(selectors, text="Caja blanca (BAD):").pack(side="left")
            self._selectors.append(ttk.Combobox(selectors, textvariable=self.white_checkpoint,
                                               values=list(white_choices), state="readonly", width=40))
            self._selectors[-1].pack(side="left", padx=4)

        self.settings_frame = tk.Frame(self.root)
        self.motion = motion_controls.MotionControls(self.settings_frame, initial_motion)
        self.settings_shown = False
        tk.Button(selectors, text="Ajustes ▾", command=self._toggle_settings).pack(side="right")

        self.root.bind("<Key>", self._key)
        self.open, self._running, self._image = True, None, None

    def _toggle_settings(self):
        self.settings_shown = not self.settings_shown
        if self.settings_shown:
            self.settings_frame.pack(fill="x")
        else:
            self.settings_frame.pack_forget()

    def _send_text(self, event=None):
        text = self.entry.get().strip()
        if text:
            self.entry.delete(0, "end")
            self.post(("text", text))

    def _key(self, event):
        widget = event.widget.winfo_class()
        if widget in ("Entry", "TEntry") or (widget == "TCombobox" and str(event.widget.cget("state")) == "normal"):
            return
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.post(("key", "q"))
        elif key == "v":
            self.post(("listen",))
        elif key in ("1", "2", "3"):
            self.post(("preset", {"1": "sort", "2": "all_green", "3": "all_white"}[key]))
        elif key in ("s", "r", "f", "c", "p"):
            self.post(("key", key))

    def motion_settings(self) -> dict:
        return self.motion.read()

    def set_running(self, running: bool):
        if not self.open or running == self._running:
            return
        self._running = running
        for selector in self._selectors:
            selector.configure(state="disabled" if running else "readonly")
        self.motion.set_enabled(not running)

    def show(self, frames, status, color, orders, voice, probs, live, result):
        if not self.open:
            return
        picture = np.ascontiguousarray(np.hstack([f[::2, ::2] for f in frames]))  # half size: faster to draw
        height, width = picture.shape[:2]
        try:
            self._image = self._tk.PhotoImage(data=b"P6 %d %d 255\n" % (width, height) + picture.tobytes())
            self.picture.configure(image=self._image)
            self.status.configure(text=status, bg=color)
            self.orders.configure(text="Orden: " + orders)
            self.voice.configure(text=voice)
            self.probs.configure(text=probs)
            self.live.configure(text=live)
            self.result.configure(text=result)
            self.root.update()
        except self._tk.TclError:  # window closed by the user
            self.open = False

    def close(self):
        self.open = False
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass


class Feedback:
    """What the voice side tells the operator: the line heard, the probabilities and the verdict."""

    def __init__(self, voice="Pulsa V y habla: «mueve las piezas buenas a la caja blanca y las malas a la verde»"):
        self.voice, self.probs, self.reply = voice, "", None

    def _answer(self, text):
        """Send the verdict back to whoever gave the order (a Telegram chat); the window and mic have no reply."""
        reply, self.reply = self.reply, None
        if reply:
            try:
                reply(text)
            except Exception:  # the order is already applied; a lost reply must not stop the loop
                traceback.print_exc()

    def party(self, demo, text=None, language=None):
        """Party mode asked for by the button, the key or a sentence: start it, or say why it cannot start now."""
        verdict = demo_orders.message("party" if demo.request_party() else "party_busy", language)
        self.voice = f"({language or 'texto'}) «{text}»  →  {verdict}" if text else verdict
        self._answer(f"«{text}» ({language or 'texto'})\n{verdict}" if text else verdict)

    def handle(self, kind, payload, orders, demo, min_confidence):
        if kind == "source":
            self.reply = payload
        elif kind == "party":
            self.party(demo, *payload)
        elif kind == "listening":
            self.voice = "🎤 Escuchando…"
        elif kind in ("loading", "error"):
            self.voice = payload
            if kind == "error":
                self._answer(payload)
        elif kind == "heard":
            text, language = payload
            self.voice = f"Oído ({language or 'texto'}): «{text}»"
        elif kind == "reading":
            text, mapped, language, reading = payload
            outcome = orders.apply(reading, language, min_confidence)
            self.probs = f"frase leída por el modelo: {mapped!r}\n" + format_probabilities(reading)
            self.voice = f"({language or 'texto'}) «{text}»  →  {outcome.message}"
            if outcome.kind == "stop":
                demo.request_stop()
            self._answer(f"«{text}» ({language or 'texto'})\n{outcome.message}\n{orders.describe()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pick-policy", default=DEFAULT_PICK,
                        help="initial pick checkpoint (Hugging Face repo id or pretrained_model folder)")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER, help="pin classifier model.pt or repo id")
    parser.add_argument("--intent-model", default=DEFAULT_INTENT, help="intent model repo id or local folder")
    parser.add_argument("--intent-device", default="auto", help="auto: the GPU when there is one (else cpu), cpu, cuda")
    parser.add_argument("--min-confidence", type=float, default=0.85,
                        help="every deciding part of an order must be at least this probable")
    parser.add_argument("--whisper-device", default="auto",
                        help="auto: the GPU when there is one (float16, falls back to the cpu if its libraries are "
                             "missing), cpu (int8) or cuda")
    parser.add_argument("--no-voice", action="store_true", help="orders only from the buttons and keys 1 / 2 / 3")
    parser.add_argument("--telegram", action="store_true",
                        help="also take orders (text or voice notes) from a Telegram bot; token in TELEGRAM_BOT_TOKEN")
    parser.add_argument("--telegram-chats", default=os.environ.get("TELEGRAM_CHAT_IDS"),
                        help="comma-separated chat ids that may give orders (default: TELEGRAM_CHAT_IDS); any other "
                             "chat is told its id and ignored")
    parser.add_argument("--bad-has-pin", action="store_true", help="swap the meaning: pieces with a pin are bad")
    parser.add_argument("--tray-roi", default=None, help="x0,y0,x1,y1 fractions of the front frame to watch for pieces")
    parser.add_argument("--policy-routes", type=Path, default=DEMO_DIR / "policy_routes.json")
    parser.add_argument("--follower-port", default=None, help="default: the one 23_run_act_so101.py uses")
    parser.add_argument("--camera-front", default="auto")
    parser.add_argument("--camera-wrist", default="auto")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pick-seconds", type=float, default=DEFAULT_PICK_SECONDS,
                        help="time the pick policy gets per attempt (97%% of the recorded picks need less)")
    parser.add_argument("--pick-attempts", type=int, default=PICK_ATTEMPTS,
                        help="tries per piece: a pick that times out returns to rest and starts again")
    parser.add_argument("--retry-wait", type=float, default=RETRY_WAIT_S,
                        help="seconds at rest between a failed attempt and the next one")
    parser.add_argument("--arm-step", type=float, default=30.0,
                        help="largest target offset from the measured pose, arm joints, degrees (apps used 3-8)")
    parser.add_argument("--gripper-step", type=float, default=45.0,
                        help="same for the gripper, percentage points: a firm grip needs a large gap")
    parser.add_argument("--p-gain", type=int, default=None,
                        help="optional: write this P_Coefficient to the arm motors (LeRobot sets 16; the STS3215 "
                             "default is 32). Torque is released for a moment: hold the arm")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--no-smooth", action="store_true",
                        help="send the policies' commands as they come (open loop chunks, no filter)")
    parser.add_argument("--blend-stride", type=int, default=smooth_motion.DEFAULT_STRIDE,
                        help="a new prediction every this many control steps; the overlapping chunks are averaged "
                             "(1 = ACT's temporal ensembling, every step; the Acciones por predicción setting overrides)")
    parser.add_argument("--blend-coeff", type=float, default=smooth_motion.DEFAULT_COEFF,
                        help="chunk weight exp(-coeff * rank), rank 0 = oldest: negative favours the newest prediction")
    parser.add_argument("--filter-hz", type=float, default=smooth_motion.DEFAULT_FILTER_HZ,
                        help="cut-off of the filter on the sent targets, arm joints (0 = no filter); lag ~ 1/(pi*hz)")
    return parser


def resolve_device(spec: str) -> str:
    """``auto`` is the GPU when torch sees one."""
    if spec != "auto":
        return spec
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def start_voice(args, holder):
    """Load Whisper and the intent model off the main thread; the window shows the result when it is there."""
    def load():
        try:
            intent = demo_voice.load_intent(args.intent_model, resolve_device(args.intent_device))
            transcriber = demo_voice.Transcriber(resolve_device(args.whisper_device))
            worker = demo_voice.VoiceWorker(transcriber, intent, demo_orders.cell_vocabulary,
                                            party=demo_orders.is_party)
            holder["telegram_status"] = start_telegram(args, worker, holder)
            holder["worker"] = worker
        except Exception as exc:
            traceback.print_exc()
            holder["status"] = f"Voz no disponible ({exc}). Usa las teclas 1 / 2 / 3."

    holder["status"] = "Cargando voz…"
    threading.Thread(target=load, daemon=True).start()


def start_telegram(args, worker, holder) -> str:
    """Connect the Telegram bot when asked; returns the line the window shows. Never raises: orders still work."""
    if not args.telegram:
        return ""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return "Telegram: falta la variable TELEGRAM_BOT_TOKEN"
    try:
        chats = demo_telegram.chat_ids(args.telegram_chats)
        bridge = demo_telegram.TelegramBridge(token, chats, worker.submit_text, worker.submit_audio)
        name = bridge.start()
    except Exception as exc:  # TelegramError text never carries the token
        return f"Telegram no disponible ({exc})"
    holder["telegram"] = bridge
    if not chats:
        return f"Telegram {name}: escríbele para ver tu id y arranca con --telegram-chats <id>"
    return f"Telegram {name}: escucha {len(chats)} chat(s)"


def set_p_gain(robot, value: int):
    with robot.bus.torque_disabled():
        for motor in JOINTS[:-1]:
            robot.bus.write("P_Coefficient", motor, value)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps <= 0 or args.pick_seconds <= 0 or args.arm_step <= 0 or args.gripper_step <= 0:
        raise SystemExit("--fps, --pick-seconds, --arm-step and --gripper-step must be positive")
    if args.pick_attempts < 1 or args.retry_wait < 0:
        raise SystemExit("--pick-attempts must be at least 1 and --retry-wait cannot be negative")
    if args.blend_stride < 1 or args.filter_hz < 0:
        raise SystemExit("--blend-stride must be at least 1 and --filter-hz cannot be negative")
    runner = load_script("23_run_act_so101.py")  # DEFAULT_FOLLOWER_PORT, find_camera

    import torch
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: F401
    from lerobot.utils.robot_utils import precise_sleep

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    registry = app24.load_policy_registry(args.policy_routes)
    for refresh in (registry.refresh_checkpoints, registry.refresh_pick_checkpoints):
        try:
            refresh()
        except Exception as exc:
            print(f"Could not refresh Hugging Face checkpoints; using the local catalog: {exc}", flush=True)
    pick_choices = registry.pick_checkpoint_options(args.pick_policy)
    initial_pick = next((label for label, source in pick_choices.items() if source == args.pick_policy),
                        next(iter(pick_choices)))
    white_choices = registry.checkpoint_options("white")
    roi = tray_detect.parse_roi(args.tray_roi) if args.tray_roi else tray_detect.DEFAULT_ROI

    keys = queue.Queue()
    reach_profile = reach_detect.load_profile()
    settings = motion_controls.default_settings(
        pick_seconds=args.pick_seconds, pick_step=args.arm_step, green_step=args.arm_step, white_step=args.arm_step,
        control_hz=args.fps,
        # the reach detector's limits (measured on the recorded picks) and a short, averaged look at the piece:
        # about 1.8 s of frames after the arm has been still, instead of waiting for the time limit
        pose_window=reach_profile["window"], arm_motion=reach_profile["motion"], arm_error=reach_profile["error"],
        gripper_motion=reach_profile["gripper_motion"], min_pick=0, settle_seconds=SETTLE_S, classify_seconds=CLASSIFY_S)
    window = Window(keys.put, pick_choices, initial_pick, white_choices, settings)
    orders = demo_orders.OrderBook(good_has_pin=not args.bad_has_pin)
    models = Models(args, device, registry, pick_choices[initial_pick], white_choices[0] if white_choices else None)
    models.start()
    voice = {}
    feedback = Feedback("Voz desactivada (--no-voice): usa los botones o las teclas 1 / 2 / 3" if args.no_voice
                        else "Cargando voz…")
    if not args.no_voice:
        start_voice(args, voice)

    cameras = {
        name: OpenCVCameraConfig(index_or_path=runner.find_camera(name, path), fps=args.fps,
                                 width=args.camera_width, height=args.camera_height, warmup_s=5)
        for name, path in (("front", args.camera_front), ("wrist", args.camera_wrist))
    }
    robot = make_robot_from_config(SOFollowerRobotConfig(
        port=args.follower_port or runner.DEFAULT_FOLLOWER_PORT, id="mi_follower", cameras=cameras,
        max_relative_target=step_limits(8.0, args.gripper_step)))

    rest = app24.load_script_path(registry.resolve_destination("white", None).runner)  # REST_POSE / REST_TOLERANCE
    demo = Demo(robot, models, orders, window.motion_settings,
                lambda: (pick_choices[window.pick_checkpoint.get()], window.white_checkpoint.get() or None),
                lambda frame: tray_detect.detect_piece(frame, roi), rest.REST_POSE, rest.REST_TOLERANCE, device,
                gripper_step=args.gripper_step, pick_attempts=args.pick_attempts, retry_wait=args.retry_wait,
                reach_profile=reach_profile,
                target_filter=(None if args.no_smooth or not args.filter_hz
                               else smooth_motion.TargetFilter(args.filter_hz)))
    next_show, announced, bus_errors = 0.0, False, 0
    robot.connect()
    try:
        if args.p_gain:
            set_p_gain(robot, args.p_gain)
        while window.open:
            cycle = time.perf_counter()
            worker = voice.get("worker")
            if worker and not announced:
                feedback.voice = " · ".join(filter(None, ("Voz lista: pulsa V y habla, o escribe la orden",
                                                          voice.get("telegram_status"))))
                announced = True
            while not keys.empty():
                event = keys.get()
                if event[0] == "key":
                    {"q": window.close, "s": demo.request_stop, "r": demo.request_resume,
                     "f": demo.request_force, "c": orders.clear, "p": lambda: feedback.party(demo)}[event[1]]()
                elif event[0] == "preset":
                    orders.preset(event[1])
                    feedback.voice = "Orden fijada con el botón"
                elif worker and event[0] == "listen":
                    worker.listen_once()
                elif worker and event[0] == "text":
                    worker.submit_text(event[1])
                elif worker and event[0] == "hands_free":
                    (worker.hands_free.set if event[1] else worker.hands_free.clear)()
                else:
                    feedback.voice = voice.get("status", "Voz desactivada")
            while worker and not worker.events.empty():
                feedback.handle(*worker.events.get(), orders, demo, args.min_confidence)
            if not window.open:
                break

            try:
                obs = robot.get_observation()
                demo.tick(obs)
                bus_errors = 0
            except OSError as exc:  # the motors did not answer: USB, or the supply sagging while the arm carries a piece
                bus_errors += 1
                note = f"el bus de motores no responde ({type(exc).__name__}: {exc})"
                print(f"[bus] {note} ({bus_errors} seguidos)", flush=True)
                if bus_errors >= BUS_ERRORS_BEFORE_PAUSE:
                    demo.bus_error(note)
                if bus_errors >= BUS_ERRORS_BEFORE_EXIT:
                    raise
                precise_sleep(0.1)
                continue
            if cycle >= next_show:
                front = obs["front"]
                if demo.phase in ("wait_order", "wait_piece") and demo.tray is not None:
                    front = tray_detect.draw_overlay(front, demo.tray)
                window.set_running(demo.phase in MOVING)
                window.show([front, obs["wrist"]], demo.status_text(), demo.color, orders.describe(),
                            feedback.voice if worker or args.no_voice else voice.get("status", feedback.voice),
                            feedback.probs, demo.live, demo.result)
                next_show = cycle + PREVIEW_INTERVAL
            precise_sleep(max(demo.period - (time.perf_counter() - cycle), 0.0))
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    finally:
        worker = voice.get("worker")
        if voice.get("telegram"):
            voice["telegram"].close()
        if worker:
            worker.close()
        try:
            robot.disconnect()
        except Exception as exc:  # releasing the torque needs the same bus: do not hide the error that got us here
            print(f"No se pudo soltar el torque de los motores ({type(exc).__name__}: {exc}); "
                  "el brazo puede seguir rígido: revisa la alimentación y el USB.", flush=True)
        window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
