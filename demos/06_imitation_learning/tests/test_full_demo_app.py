"""The demo cycle (order -> wait for a piece -> pick -> classify -> box -> repeat) on a fake clock, robot and policies."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_app():
    path = Path(__file__).resolve().parent.parent / "30_full_demo_app.py"
    spec = importlib.util.spec_from_file_location("full_demo_app_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = load_app()
orders_module = app.demo_orders
REST = np.array([6.2, -100.1, 28.9, 99.5, -5.2, 0.9], dtype=np.float32)
TOLERANCE = np.array([20, 15, 15, 10, 20, 8], dtype=np.float32)


class World:
    def __init__(self):
        self.now = 0.0
        self.state = REST.copy()
        self.piece = False
        self.sent = []  # (time, action, per-joint caps)
        self.classified = []


class FakeRobot:
    def __init__(self, world):
        self.world = world
        self.config = SimpleNamespace(max_relative_target=None)

    def get_observation(self):
        front = np.empty((120, 160, 3), dtype=np.uint8)
        front[:] = (205, 95, 55)  # the orange tray floor
        if self.world.piece:
            front[50:70, 70:90] = (170, 172, 176)  # a grey piece inside the watched rectangle
        wrist = np.zeros((120, 160, 3), dtype=np.uint8)
        return {**{f"{j}.pos": float(v) for j, v in zip(app.JOINTS, self.world.state)}, "front": front, "wrist": wrist}

    def send_action(self, action):
        self.world.sent.append((self.world.now, dict(action), dict(self.config.max_relative_target)))
        self.world.state = np.array([action[f"{j}.pos"] for j in app.JOINTS], dtype=np.float32)  # perfect tracking
        return dict(action)


class PickPolicy:
    def __init__(self, world, lead=5.0):
        self.world, self.lead = world, lead
        self.config = SimpleNamespace(n_action_steps=30, chunk_size=90)

    def reset(self):
        pass

    def select_action(self, batch):
        target = self.world.state[None] + self.lead
        target[0, -1] = self.world.state[-1]  # the arm leads; the gripper stays where it is
        return torch.tensor(target, dtype=torch.float32)


class RouteRunner:
    """Releases on the first step, goes home on the third and reports finished after that."""

    def __init__(self, world, finishes=True):
        self.world, self.finishes, self.steps = world, finishes, 0

    def step(self, policy, pre, post, features, observation, device):
        self.steps += 1
        action = {f"{j}.pos": float(v) for j, v in zip(app.JOINTS, REST)}
        action["gripper.pos"] = 30.0 if self.steps <= 2 else 0.9
        return action

    def finished(self, state, released):
        return self.finishes and self.steps >= 3


class FakeModels:
    ready, error, status = True, None, ""

    def __init__(self, world, probability=0.9, runner_finishes=True, route_seconds=15.0, lead=5.0, holds_piece=False):
        self.world, self.probability, self.route_seconds = world, probability, route_seconds
        self.runner_finishes, self.lead, self.holds_piece = runner_finishes, lead, holds_piece
        self.picked, self.routed = [], []

    def pick(self, source):
        self.picked.append(source)
        self.world.piece = False  # the piece leaves the tray with the gripper
        return PickPolicy(self.world, self.lead), (lambda b: b), (lambda a: a), 30

    def route(self, destination, label):
        self.routed.append((destination, label))
        runner = RouteRunner(self.world, self.runner_finishes)
        if self.holds_piece:  # the gripper never opens
            runner.step = lambda *a, **k: {**{f"{j}.pos": float(v) for j, v in zip(app.JOINTS, REST)},
                                           "gripper.pos": 5.0}
        route = SimpleNamespace(name=f"up-down-{destination}", max_seconds=self.route_seconds, max_relative_target=5.0)
        return {"route": route, "runner": runner, "policy": PickPolicy(self.world), "pre": (lambda b: b),
                "post": (lambda a: a), "features": {}, "default_actions": 25}

    def classify(self, front, wrist):
        self.world.classified.append(self.world.now)
        return self.probability


MODEL_OPTIONS = ("probability", "runner_finishes", "route_seconds", "lead", "holds_piece")


def make(**kwargs):
    world = World()
    models = FakeModels(world, **{k: kwargs.pop(k) for k in MODEL_OPTIONS if k in kwargs})
    settings = app.motion_controls.default_settings(
        pick_seconds=1.0, auto_stop=False, settle_seconds=0.0, classify_seconds=0.3, green_clearance=False,
        white_clearance=False, pick_step=30, green_step=30, white_step=30, control_hz=20, **kwargs)
    orders = orders_module.OrderBook()
    log = []
    detect = lambda frame: SimpleNamespace(present=world.piece, area=500, box=(0, 0, 1, 1))
    demo = app.Demo(FakeRobot(world), models, orders, lambda: settings, lambda: ("IkerAzurmendi/pick", "ckpt"),
                    detect, REST, TOLERANCE, "cpu", gripper_step=45.0, clock=lambda: world.now,
                    log=lambda text, **_: log.append(text))
    return world, models, orders, demo, log


def run(world, demo, seconds, dt=0.05):
    for _ in range(round(seconds / dt)):
        world.now += dt
        demo.tick(demo.robot.get_observation())


def phases(log):
    return [line.split("]")[0][1:] for line in log if line.startswith("[")]


class CycleTests(unittest.TestCase):
    def test_no_order_means_no_motion(self):
        world, _, _, demo, _ = make()
        world.piece = True
        run(world, demo, 5)
        self.assertEqual(demo.phase, "wait_order")
        self.assertEqual(world.sent, [])

    def test_empty_tray_means_waiting_not_moving(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        run(world, demo, 5)
        self.assertEqual(demo.phase, "wait_piece")
        self.assertIn("vacía", demo.live)
        self.assertEqual(world.sent, [])

    def test_a_piece_must_stay_in_view_before_the_pick_starts(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        run(world, demo, 0.5)
        world.piece = True
        run(world, demo, 0.6)
        self.assertEqual(demo.phase, "wait_piece")
        run(world, demo, 1.0)
        self.assertEqual(demo.phase, "picking")

    def test_a_flickering_piece_never_starts_the_pick(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        for _ in range(40):
            world.piece = not world.piece
            run(world, demo, 0.3)
        self.assertEqual(demo.phase, "wait_piece")
        self.assertEqual(world.sent, [])

    def test_full_cycle_to_the_green_box_and_back_to_waiting(self):
        world, models, orders, demo, log = make()
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 6)
        self.assertEqual(phases(log), ["wait_order", "wait_piece", "picking", "classifying", "routing", "wait_piece"])
        self.assertEqual(models.picked, ["IkerAzurmendi/pick"])
        self.assertEqual(models.routed, [("green", None)])
        self.assertEqual(demo.delivered, {"green": 1, "white": 0})
        self.assertEqual(demo.result, "RESULTADO: PIN (90%)")
        self.assertTrue(orders.active)  # "all pieces to green" has no end: it waits for the next piece

    def test_classification_starts_only_after_the_arm_is_held(self):
        world, _, orders, demo, log = make()
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 6)
        self.assertTrue(world.classified)
        hold_time = next(t for t, _, _ in world.sent if t >= 1.0 + 1.0)  # pick starts ~1 s in and lasts 1 s
        self.assertGreaterEqual(min(world.classified), hold_time)

    def test_white_box_uses_the_selected_white_checkpoint(self):
        world, models, orders, demo, _ = make(probability=0.1)  # NO PIN
        orders.preset("sort")  # bad pieces -> green; good -> white
        world.piece = True
        run(world, demo, 6)
        self.assertEqual(models.routed, [("green", None)])
        orders.preset("all_white")
        world.piece = True
        run(world, demo, 8)
        self.assertEqual(models.routed[-1], ("white", "ckpt"))

    def test_command_caps_are_per_joint_eased_in_and_gripper_is_stronger(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 2.5)
        caps = [c for _, _, c in world.sent]
        self.assertEqual(set(caps[0]), set(app.JOINTS))
        self.assertLessEqual(caps[0]["shoulder_pan"], 12.0)  # eased in from 8 degrees, not the full 30 at once
        self.assertEqual(max(c["shoulder_lift"] for c in caps), 30.0)
        self.assertTrue(all(c["gripper"] >= 45.0 for c in caps))

    def test_a_piece_with_no_rule_is_held_until_an_order_covers_it(self):
        world, models, orders, demo, _ = make(probability=0.9)  # PIN
        orders.routes = {orders_module.NO_PIN: orders_module.Route("green", None)}
        world.piece = True
        run(world, demo, 4)
        self.assertEqual(demo.phase, "holding")
        sent_before = len(world.sent)
        run(world, demo, 3)
        self.assertEqual(len(world.sent), sent_before)  # holding commands nothing new
        self.assertEqual(models.routed, [])
        orders.routes[orders_module.PIN] = orders_module.Route("white", None)
        run(world, demo, 0.2)
        self.assertEqual(demo.phase, "routing")
        self.assertEqual(models.routed, [("white", "ckpt")])

    def test_a_counted_order_ends_and_the_robot_waits_for_the_next_order(self):
        world, _, orders, demo, _ = make()
        orders.routes = {orders_module.PIN: orders_module.Route("green", 1)}
        world.piece = True
        run(world, demo, 6)
        self.assertEqual(demo.phase, "wait_order")
        self.assertFalse(orders.active)
        self.assertEqual(demo.delivered["green"], 1)

    def test_pick_only_order_stops_with_the_piece_in_the_gripper(self):
        world, models, orders, demo, _ = make()
        orders.routes = {orders_module.PIN: orders_module.Route(orders_module.HOLD, 1)}
        world.piece = True
        run(world, demo, 5)
        self.assertEqual(demo.phase, "paused")
        self.assertEqual(models.routed, [])
        self.assertIn("quítala", demo.note)

    def test_stop_holds_the_measured_pose_and_the_grip(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 2.0)
        self.assertEqual(demo.phase, "picking")
        measured, grip = world.state.copy(), demo.last_gripper
        demo.request_stop()
        run(world, demo, 0.1)
        self.assertEqual(demo.phase, "paused")
        _, action, _ = world.sent[-1]
        self.assertEqual([action[f"{j}.pos"] for j in app.JOINTS[:-1]], [float(v) for v in measured[:-1]])
        self.assertEqual(action["gripper.pos"], grip)
        count = len(world.sent)
        run(world, demo, 2)
        self.assertEqual(len(world.sent), count)

    def test_resume_walks_back_to_rest_slowly_then_waits_for_a_piece(self):
        world, _, orders, demo, _ = make(lead=20.0)
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 2.0)
        demo.request_stop()
        run(world, demo, 0.1)
        self.assertFalse(demo.near_rest(world.state))
        world.piece = False
        demo.request_resume()
        run(world, demo, 0.5)
        self.assertEqual(demo.phase, "returning")
        self.assertGreater(np.abs(world.state - REST).max(), 1.0)  # not teleported: it is part way
        run(world, demo, 8)
        self.assertEqual(demo.phase, "wait_piece")
        self.assertTrue(demo.near_rest(world.state, 1.0))

    def test_a_first_action_far_from_the_arm_is_refused_and_two_refusals_pause(self):
        world, _, orders, demo, _ = make(lead=100.0)
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 1.6)
        self.assertEqual(demo.phase, "wait_piece")
        self.assertIn("no me muevo", demo.note)
        self.assertEqual(world.sent, [])
        world.piece = True  # a refused pick does not take the piece out of the tray
        run(world, demo, 3.0)
        self.assertEqual(demo.phase, "paused")
        self.assertEqual(world.sent, [])

    def test_a_route_that_never_releases_pauses_for_the_operator(self):
        world, _, orders, demo, _ = make(runner_finishes=False, route_seconds=1.0, holds_piece=True)
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 6)
        self.assertEqual(demo.phase, "paused")
        self.assertIn("no se soltó", demo.note)
        self.assertEqual(demo.delivered["green"], 0)

    def test_a_route_that_times_out_after_releasing_returns_to_rest(self):
        world, _, orders, demo, log = make(runner_finishes=False, route_seconds=1.0)
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 12)
        self.assertIn("returning", phases(log))
        self.assertEqual(demo.phase, "wait_piece")
        self.assertEqual(demo.delivered["green"], 1)

    def test_force_pick_skips_the_detector(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        run(world, demo, 0.3)
        demo.request_force()
        run(world, demo, 0.2)
        self.assertEqual(demo.phase, "picking")

    def test_a_piece_in_view_with_the_arm_away_from_rest_waits(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        world.state = REST + 60
        world.piece = True
        run(world, demo, 3)
        self.assertEqual(demo.phase, "wait_piece")
        self.assertIn("reposo", demo.live)
        demo.request_force()
        run(world, demo, 0.2)
        self.assertEqual(demo.phase, "picking")

    def test_nothing_moves_while_the_models_load_and_a_failure_is_shown(self):
        world, models, orders, demo, _ = make()
        models.ready = False
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 3)
        self.assertEqual((demo.phase, world.sent), ("loading", []))
        models.error = "RuntimeError: no weights"
        run(world, demo, 0.2)
        self.assertEqual(demo.phase, "loading")
        self.assertIn("no weights", demo.note)

    def test_the_settings_of_the_cycle_come_from_the_window_when_it_starts(self):
        world, _, orders, demo, _ = make()
        orders.preset("all_green")
        world.piece = True
        run(world, demo, 2.0)
        self.assertEqual(demo.run["pick_seconds"], 1.0)


class FakeWorker:
    """Stands in for the voice thread: tests put events in ``events`` as if somebody had spoken."""

    def __init__(self):
        import queue
        import threading

        self.events, self.hands_free, self.listened, self.typed, self.closed = queue.Queue(), threading.Event(), 0, [], False

    def listen_once(self):
        self.listened += 1

    def submit_text(self, text):
        self.typed.append(text)

    def close(self):
        self.closed = True


class MainWiringTests(unittest.TestCase):
    """``main()`` end to end: arguments, registry, rest pose, window events, voice events, robot connect/disconnect."""

    def run_main(self, argv=(), spoken=None, presets=None):
        from unittest.mock import patch

        world, models, worker = World(), None, FakeWorker()
        models = FakeModels(world)
        models.start = lambda: None
        shown, robots = [], []

        class Bus:
            def __init__(self):
                self.writes = []

            def torque_disabled(self):
                from contextlib import nullcontext
                return nullcontext()

            def write(self, *args):
                self.writes.append(args)

        class Robot(FakeRobot):
            def __init__(self, config):
                super().__init__(world)
                self.config, self.bus, self.events = config, Bus(), []
                robots.append(self)

            def connect(self):
                self.events.append("connect")

            def disconnect(self):
                self.events.append("disconnect")

        class Registry:
            def refresh_checkpoints(self):
                pass

            def refresh_pick_checkpoints(self):
                pass

            def pick_checkpoint_options(self, current):
                return {"v2 30000": "IkerAzurmendi/act-pick-v2-30000", "v1": "IkerAzurmendi/act-pick-30000"}

            def checkpoint_options(self, destination):
                return ("25.000 pasos",)

            def resolve_destination(self, destination, label):
                return SimpleNamespace(runner=Path("run_policy.py"))

        class Window:
            def __init__(self, post, pick_choices, initial_pick, white_choices, initial_motion):
                self.open, self.post, self.initial_pick, self.initial_motion = True, post, initial_pick, initial_motion
                self.pick_checkpoint = SimpleNamespace(get=lambda: initial_pick)
                self.white_checkpoint = SimpleNamespace(get=lambda: white_choices[0])
                self.pick_choices, self.running = pick_choices, []

            def motion_settings(self):
                return app.motion_controls.parse_settings(self.initial_motion)

            def set_running(self, running):
                self.running.append(running)

            def show(self, frames, status, color, orders, voice, probs, live, result):
                shown.append(SimpleNamespace(status=status, orders=orders, voice=voice, probs=probs, result=result,
                                             frames=len(frames)))
                if len(shown) == 1:
                    if spoken:
                        worker.events.put(("reading", spoken))
                    if presets:
                        self.post(("preset", presets))
                    world.piece = True
                if "entregadas" in status:
                    self.post(("key", "q"))
                if world.now > 40:  # never hang the suite
                    self.open = False

            def close(self):
                self.open = False

        windows = []
        original_window = Window

        def make_window(*args):
            windows.append(original_window(*args))
            return windows[-1]

        cli = ["30_full_demo_app.py", "--device", "cpu", "--fps", "20", "--pick-seconds", "1", *argv]
        sleeps = lambda seconds: setattr(world, "now", world.now + max(seconds, 0.01))
        with patch.object(app, "load_script", return_value=SimpleNamespace(
                DEFAULT_FOLLOWER_PORT="/dev/fake", find_camera=lambda name, path: Path("/dev/fake_camera"))), \
                patch.object(app.app24, "load_policy_registry", return_value=Registry()), \
                patch.object(app.app24, "load_script_path",
                             return_value=SimpleNamespace(REST_POSE=REST, REST_TOLERANCE=TOLERANCE)), \
                patch.object(app.app24, "add_route_clearance", lambda action, *a, **k: action), \
                patch.object(app, "Models", lambda *a, **k: models), \
                patch.object(app, "Window", make_window), \
                patch.object(app, "start_voice", lambda args, holder: holder.update(worker=worker)), \
                patch.object(app.time, "perf_counter", lambda: world.now), \
                patch("lerobot.robots.make_robot_from_config", side_effect=Robot), \
                patch("lerobot.utils.robot_utils.precise_sleep", side_effect=sleeps), \
                patch("sys.argv", cli):
            code = app.main(cli[1:])
        return code, world, models, worker, shown, robots, windows

    def test_a_spoken_order_drives_a_full_delivery_and_everything_is_released(self):
        reading = {"status": "ok", "intent": "move", "confidence": 0.99,
                   "probabilities": {"intent": {"move": 0.99}, "box": {"big": 0.99, "white": 0.01}},
                   "rules": [{"feature": "any", "quality": "any", "count": "all", "box": "big", "p_feature": 0.99,
                              "p_quality": 0.99, "p_count": 0.99, "p_box": 0.99, "box_source": "said"}]}
        code, world, models, worker, shown, robots, windows = self.run_main(
            spoken=("mueve todo a la caja verde", "mueve todo a la caja grande", "es", reading))
        self.assertEqual(code, 0)
        self.assertEqual(models.picked, ["IkerAzurmendi/act-pick-v2-30000"])  # the default pick checkpoint
        self.assertEqual(models.routed, [("green", None)])
        self.assertEqual(robots[0].events, ["connect", "disconnect"])
        self.assertTrue(worker.closed)
        self.assertTrue(any("Orden entendida" in s.voice for s in shown))
        self.assertTrue(any("caja VERDE" in s.orders for s in shown))
        self.assertTrue(any("move 99%" in s.probs for s in shown))
        self.assertTrue(all(s.frames == 2 for s in shown))
        self.assertTrue(any("ESPERANDO PIEZA" in s.status for s in shown))
        self.assertTrue(any(s.result == "RESULTADO: PIN (90%)" for s in shown))
        self.assertTrue(all(isinstance(caps, dict) for _, _, caps in world.sent))

    def test_motion_defaults_use_the_large_caps(self):
        code, world, models, worker, shown, robots, windows = self.run_main(["--no-voice"], presets="all_green")
        settings = windows[0].initial_motion
        self.assertEqual((settings["pick_step"], settings["green_step"], settings["white_step"]), (30.0, 30.0, 30.0))
        self.assertEqual(max(c["gripper"] for _, _, c in world.sent), 45.0)
        self.assertEqual(max(c["wrist_flex"] for _, _, c in world.sent), 30.0)

    def test_cli_caps_are_honoured(self):
        _, world, *_ = self.run_main(["--no-voice", "--arm-step", "12", "--gripper-step", "20"], presets="all_green")
        self.assertEqual(max(c["wrist_flex"] for _, _, c in world.sent), 12.0)
        self.assertEqual(max(c["gripper"] for _, _, c in world.sent), 20.0)

    def test_without_voice_the_buttons_set_the_order(self):
        code, world, models, worker, shown, robots, windows = self.run_main(["--no-voice"], presets="sort")
        self.assertEqual(code, 0)
        self.assertTrue(any("Voz desactivada" in s.voice for s in shown))
        self.assertEqual(models.routed[0], ("white", "25.000 pasos"))  # the fake classifier says PIN: good -> white
        self.assertFalse(worker.listened)

    def test_p_gain_is_written_to_the_arm_motors_only_when_asked(self):
        _, _, _, _, _, robots, _ = self.run_main(["--no-voice", "--p-gain", "24"], presets="all_green")
        self.assertEqual(robots[0].bus.writes, [("P_Coefficient", joint, 24) for joint in app.JOINTS[:-1]])
        _, _, _, _, _, robots, _ = self.run_main(["--no-voice"], presets="all_green")
        self.assertEqual(robots[0].bus.writes, [])

    def test_bad_roi_is_rejected_before_anything_connects(self):
        with self.assertRaises(ValueError):
            self.run_main(["--no-voice", "--tray-roi", "0.9,0.1,0.2,0.5"])


class HelperTests(unittest.TestCase):
    def test_step_limits_cover_every_joint_and_the_gripper_may_be_larger(self):
        limits = app.step_limits(8.0, 45.0)
        self.assertEqual(set(limits), set(app.JOINTS))
        self.assertTrue(all(isinstance(v, float) for v in limits.values()))
        self.assertEqual((limits["elbow_flex"], limits["gripper"]), (8.0, 45.0))
        self.assertEqual(app.step_limits(60.0, 45.0)["gripper"], 60.0)

    def test_ramp(self):
        self.assertEqual(app.ramped(30.0, 0.0), app.RAMP_START_DEG)
        self.assertEqual(app.ramped(30.0, app.RAMP_S), 30.0)
        self.assertEqual(app.ramped(30.0, 10.0), 30.0)
        self.assertEqual(app.ramped(5.0, 0.0), 5.0)  # a cap below the start value is not raised

    def test_probabilities_are_listed_per_question_best_first(self):
        text = app.format_probabilities({"status": "ok", "confidence": 0.97, "probabilities": {
            "intent": {"move": 0.97, "sort": 0.02, "pick": 0.01}, "box": {"white": 0.9, "big": 0.1}}})
        self.assertIn("intent  move 97%   sort 2%   pick 1%", text)
        self.assertIn("box     white 90%   big 10%", text)
        self.assertIn("confianza 97%", text)


class FeedbackTests(unittest.TestCase):
    def reading(self, status="ok", box="big"):
        return {"status": status, "intent": "move", "confidence": 0.99,
                "probabilities": {"intent": {"move": 0.99}},
                "rules": [] if status == "stop" else [{"feature": "any", "quality": "any", "count": "all", "box": box,
                                                       "p_feature": 0.99, "p_quality": 0.99, "p_count": 0.99,
                                                       "p_box": 0.99, "box_source": "said"}]}

    def test_a_reading_updates_the_order_and_shows_the_probabilities(self):
        feedback, orders = app.Feedback(), orders_module.OrderBook()
        demo = SimpleNamespace(request_stop=lambda: None)
        feedback.handle("reading", ("mueve todo a la caja verde", "mueve todo a la caja grande", "es", self.reading()),
                        orders, demo, 0.85)
        self.assertEqual(orders.destination_for(orders_module.PIN), "green")
        self.assertIn("Orden entendida", feedback.voice)
        self.assertIn("move 99%", feedback.probs)

    def test_a_spoken_stop_clears_the_order_and_stops_the_arm(self):
        feedback, orders, stops = app.Feedback(), orders_module.OrderBook(), []
        orders.preset("all_green")
        demo = SimpleNamespace(request_stop=lambda: stops.append(1))
        feedback.handle("reading", ("stop", "stop", "en", self.reading("stop")), orders, demo, 0.85)
        self.assertFalse(orders.active)
        self.assertEqual(stops, [1])
        self.assertIn("Stopped", feedback.voice)

    def test_heard_listening_and_error_lines(self):
        feedback, orders, demo = app.Feedback(), orders_module.OrderBook(), SimpleNamespace()
        feedback.handle("listening", None, orders, demo, 0.85)
        self.assertIn("Escuchando", feedback.voice)
        feedback.handle("heard", ("kaixo", "eu"), orders, demo, 0.85)
        self.assertEqual(feedback.voice, "Oído (eu): «kaixo»")
        feedback.handle("error", "Voz: sin micrófono", orders, demo, 0.85)
        self.assertEqual(feedback.voice, "Voz: sin micrófono")

    def test_a_telegram_order_is_answered_in_its_chat_with_the_verdict_and_the_order(self):
        feedback, orders, replies = app.Feedback(), orders_module.OrderBook(), []
        demo = SimpleNamespace(request_stop=lambda: None)
        feedback.handle("source", replies.append, orders, demo, 0.85)
        feedback.handle("reading", ("todo a la caja verde", "todo a la caja grande", None, self.reading()),
                        orders, demo, 0.85)
        self.assertEqual(len(replies), 1)
        self.assertIn("Orden entendida", replies[0])
        self.assertIn("caja VERDE", replies[0])
        self.assertIsNone(feedback.reply)

    def test_the_reply_goes_once_and_only_to_the_order_that_asked(self):
        feedback, orders, replies = app.Feedback(), orders_module.OrderBook(), []
        demo = SimpleNamespace(request_stop=lambda: None)
        feedback.handle("source", replies.append, orders, demo, 0.85)
        feedback.handle("loading", "Transcribiendo nota de voz…", orders, demo, 0.85)
        self.assertEqual(replies, [])  # the progress line is not the answer
        feedback.handle("error", "No se ha entendido ninguna palabra.", orders, demo, 0.85)
        feedback.handle("reading", ("stop", "stop", "en", self.reading("stop")), orders, demo, 0.85)  # typed in the app
        self.assertEqual(replies, ["No se ha entendido ninguna palabra."])

    def test_a_broken_reply_does_not_stop_the_loop(self):
        feedback, orders = app.Feedback(), orders_module.OrderBook()
        demo = SimpleNamespace(request_stop=lambda: None)

        def broken(text):
            raise RuntimeError("network down")

        feedback.handle("source", broken, orders, demo, 0.85)
        feedback.handle("reading", ("todo a la caja verde", "todo a la caja grande", None, self.reading()),
                        orders, demo, 0.85)
        self.assertEqual(orders.destination_for(orders_module.PIN), "green")


class StartTelegramTests(unittest.TestCase):
    worker = SimpleNamespace(submit_text=print, submit_audio=print)

    def args(self, telegram=True, chats="7"):
        return SimpleNamespace(telegram=telegram, telegram_chats=chats)

    def test_off_by_default(self):
        from unittest.mock import patch

        with patch.dict(app.os.environ, {"TELEGRAM_BOT_TOKEN": "1:x"}):
            self.assertEqual(app.start_telegram(self.args(telegram=False), SimpleNamespace(), {}), "")

    def test_a_missing_token_is_reported_not_raised(self):
        from unittest.mock import patch

        with patch.dict(app.os.environ, {}, clear=True):
            self.assertIn("TELEGRAM_BOT_TOKEN", app.start_telegram(self.args(), SimpleNamespace(), {}))

    def test_the_bridge_is_wired_to_the_worker_and_kept_for_shutdown(self):
        from unittest.mock import patch

        made = []

        class Bridge:
            def __init__(self, token, chats, on_text, on_audio):
                made.append(SimpleNamespace(token=token, chats=chats, on_text=on_text, on_audio=on_audio))

            def start(self):
                return "@bin_bot"

        worker, holder = SimpleNamespace(submit_text=print, submit_audio=print), {}
        with patch.dict(app.os.environ, {"TELEGRAM_BOT_TOKEN": " 1:x "}), \
                patch.object(app.demo_telegram, "TelegramBridge", Bridge):
            line = app.start_telegram(self.args(chats="7, 8"), worker, holder)
        self.assertEqual((made[0].token, made[0].chats), ("1:x", {7, 8}))
        self.assertIs(made[0].on_text, print)
        self.assertIs(made[0].on_audio, print)
        self.assertIn("@bin_bot", line)
        self.assertIn("2 chat", line)
        self.assertIn("telegram", holder)

    def test_without_chats_the_line_says_how_to_find_the_id(self):
        from unittest.mock import patch

        class Bridge:
            def __init__(self, *args):
                pass

            def start(self):
                return "@bin_bot"

        with patch.dict(app.os.environ, {"TELEGRAM_BOT_TOKEN": "1:x"}), \
                patch.object(app.demo_telegram, "TelegramBridge", Bridge):
            self.assertIn("--telegram-chats", app.start_telegram(self.args(chats=None), self.worker, {}))

    def test_a_bad_token_or_chat_id_is_reported_not_raised(self):
        from unittest.mock import patch

        class Bridge:
            def __init__(self, *args):
                pass

            def start(self):
                raise app.demo_telegram.TelegramError("getMe: HTTP 401")

        with patch.dict(app.os.environ, {"TELEGRAM_BOT_TOKEN": "1:x"}), \
                patch.object(app.demo_telegram, "TelegramBridge", Bridge):
            line = app.start_telegram(self.args(), self.worker, {})
            self.assertIn("HTTP 401", line)
            self.assertIn("no disponible", app.start_telegram(self.args(chats="@iker"), self.worker, {}))


if __name__ == "__main__":
    unittest.main()
