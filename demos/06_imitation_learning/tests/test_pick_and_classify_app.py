"""Exercise the pick/hold/classify sequence without cameras, downloads or motors."""

import importlib.util
import unittest
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


def load_app():
    path = Path(__file__).resolve().parent.parent / "24_pick_and_classify_app.py"
    spec = importlib.util.spec_from_file_location("pick_and_classify_app", path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def run_app(mode="complete", cycles=1, pick_seconds=1, pick_selection=None, explicit_policy=None,
            route_load_seconds=0, motion=None, next_cycle_motion=None):
    app = load_app()
    clock = SimpleNamespace(now=0.0)
    events, results, configurations = [], [], []
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    measured = {f"{joint}.pos": 0.25 for joint in joints}
    measured["gripper.pos"] = 10.0  # Contact with the piece prevents reaching the closing target of 5.
    expected_hold = {**measured, "gripper.pos": 5.0}
    runner = SimpleNamespace(JOINTS=joints, DEFAULT_FOLLOWER_PORT="/dev/fake", MAX_FIRST_GAP=60,
                             find_camera=lambda name, path: Path("/dev/fake_camera"))

    class Robot:
        def connect(self):
            events.append(("connect", clock.now))

        def get_observation(self):
            frame = np.full((2, 2, 3), round(clock.now * 10) % 256, dtype=np.uint8)
            return {**measured, "front": frame, "wrist": frame.copy()}

        def send_action(self, action):
            events.append(("send", clock.now, action.copy(), self.config.max_relative_target))
            return action

        def disconnect(self):
            events.append(("disconnect", clock.now))

    class Policy:
        def __init__(self):
            self.config = SimpleNamespace(device="cpu", n_action_steps=30, chunk_size=90)

        def eval(self):
            return self

        def reset(self):
            events.append(("reset", clock.now, self.config.n_action_steps))

        def select_action(self, batch):
            events.append(("policy", clock.now))
            if mode == "settled":
                return torch.tensor([[0.25] * 5 + [5.0]])
            if mode == "noisy_settled":
                odd = len([event for event in events if event[0] == "policy"]) % 2
                return torch.tensor([[0.25 + (4.0 if odd else -4.0)] * 5 + [0.0 if odd else 20.0]])
            if mode == "stalled":
                return torch.full((1, len(joints)), 30.0)
            return torch.full((1, len(joints)), 200.0 if mode == "blocked" else 5.0)

    class RouteRunner:
        GRIPPER_RELEASED = 20.0
        steps = 0

        def load_policy(self, *args):
            self.steps = 0
            return RoutePolicy(), lambda value: value, lambda value: value

        def step(self, policy, pre, post, features, observation, device):
            self.steps += 1
            events.append(("route_step", clock.now, observation["front"][0, 0, 0]))
            return {f"{joint}.pos": 30.0 if joint == "gripper" else 1.0 for joint in joints}

        def finished(self, state, released):
            events.append(("route_finished", clock.now, released))
            return self.steps >= 2

    class RoutePolicy:
        def __init__(self):
            self.config = SimpleNamespace(n_action_steps=25, chunk_size=100)

        def reset(self):
            events.append(("route_reset", clock.now, self.config.n_action_steps))

    def classifier(front, wrist):
        events.append(("cnn", clock.now, int(front[0, 0, 0])))
        return 0.9 if sum(event[0] == "reset" for event in events) == 1 else 0.1

    class Window:
        def __init__(self, on_key, checkpoint_choices=(), pick_choices=(), initial_pick=None,
                     initial_motion=None):
            self.open, self.started, self.stopped, self.probed = True, False, False, False
            self.on_key = on_key
            self.pick_choices = pick_choices
            self.pick_choice = initial_pick
            self.settings = {**initial_motion, **(motion or {})}

        def motion_settings(self):
            return app.motion_controls.parse_settings(self.settings)

        def set_running(self, running):
            pass

        def pick_checkpoint_for(self):
            return self.pick_choice

        def show(self, frames, status, color, live, result):
            events.append(("status", clock.now, live))
            if next_cycle_motion and status.startswith("PICKING"):
                self.settings.update(next_cycle_motion)
            if not self.started and clock.now >= 0.4:
                self.started = True
                if pick_selection is not None:
                    self.pick_choice = self.pick_choices[pick_selection]
                self.on_key("space")
            if mode == "cancel" and not self.stopped and clock.now >= 0.8:
                self.stopped = True
                self.on_key("s")
            if status.startswith("CLASSIFYING") and not self.probed:
                self.probed = True
                self.on_key("space")  # A second Space must not interrupt classification.
            if status.startswith("DONE") and result:
                results.append(result)
                if len(results) < cycles:
                    self.on_key("space")
                else:
                    self.open = False
            if clock.now > route_load_seconds + (4 if mode in ("cancel", "blocked") else max(12, cycles * (pick_seconds + 4))):
                self.open = False

        def close(self):
            self.open = False

    def make_robot(config):
        configurations.append(config)
        config.max_relative_target = 8.0
        robot = Robot()
        robot.config = config
        return robot

    class Registry:
        def refresh_checkpoints(self):
            pass

        def refresh_pick_checkpoints(self):
            pass

        def checkpoint_options(self, destination):
            return ("bad",)

        def pick_checkpoint_options(self, current_policy=None):
            return {"v2": "IkerAzurmendi/act-pick-v2-12000", "v1": "IkerAzurmendi/act-pick-30000"}

        def pick_revision(self, source):
            return None

        def resolve(self, classifier_label):
            events.append(("route_selected", clock.now, classifier_label))
            return SimpleNamespace(name="up-down-bad", max_seconds=15.0, max_relative_target=5.0)

    route_runner = RouteRunner()

    def sleep(seconds):
        clock.now += max(seconds, 0.01)

    def load_pick(source, device, revision):
        events.append(("load_pick", clock.now, source))
        return Policy(), lambda batch: batch, lambda action: action

    def load_route(route, device):
        clock.now += route_load_seconds
        events.append(("load_route", clock.now))
        return (route_runner, *route_runner.load_policy()[:1],
                lambda value: value, lambda value: value, {})

    cli_args = ["24_pick_and_classify_app.py", "--device", "cpu", "--fps", "10",
                "--pick-seconds", str(pick_seconds)]
    if explicit_policy:
        cli_args.extend(["--policy", explicit_policy])

    with ExitStack() as stack:
        replacements = [
            patch.object(app, "load_script", return_value=runner),
            patch.object(app, "load_policy_registry", return_value=Registry()),
            patch.object(app, "load_pick_components", side_effect=load_pick),
            patch.object(app, "load_route_components", side_effect=load_route),
            patch.object(app, "PinClassifier", return_value=classifier),
            patch.object(app, "resolve_classifier", return_value=Path("unused.pt")),
            patch.object(app, "Window", Window),
            patch.object(app.time, "perf_counter", side_effect=lambda: clock.now),
            patch("sys.argv", cli_args),
            patch("lerobot.configs.policies.PreTrainedConfig.from_pretrained",
                  return_value=SimpleNamespace(device="cpu")),
            patch("lerobot.policies.act.modeling_act.ACTPolicy.from_pretrained", return_value=Policy()),
            patch("lerobot.policies.factory.make_pre_post_processors",
                  return_value=(lambda batch: batch, lambda action: action)),
            patch("lerobot.robots.make_robot_from_config", side_effect=make_robot),
            patch("lerobot.utils.robot_utils.precise_sleep", side_effect=sleep),
        ]
        for replacement in replacements:
            stack.enter_context(replacement)
        code = app.main()
    return app, code, events, results, configurations, expected_hold


class PickSequenceTests(unittest.TestCase):
    def test_pick_selector_loads_the_selected_model_before_any_policy_action(self):
        _, code, events, _, _, _ = run_app(mode="blocked", pick_selection=1)
        self.assertEqual(code, 0)
        loads = [event for event in events if event[0] == "load_pick"]
        self.assertEqual([event[2] for event in loads],
                         ["IkerAzurmendi/act-pick-v2-12000", "IkerAzurmendi/act-pick-30000"])
        self.assertLess(events.index(loads[-1]), next(i for i, event in enumerate(events) if event[0] == "policy"))

    def test_explicit_cli_policy_remains_the_initial_pick_selection(self):
        _, code, events, _, _, _ = run_app(mode="blocked", explicit_policy="IkerAzurmendi/act-pick-30000")
        self.assertEqual(code, 0)
        self.assertEqual([event[2] for event in events if event[0] == "load_pick"],
                         ["IkerAzurmendi/act-pick-30000"])

    def test_checkpoint_download_time_does_not_consume_route_movement_time(self):
        _, code, events, _, _, _ = run_app(route_load_seconds=30)
        self.assertEqual(code, 0)
        loaded_at = next(event[1] for event in events if event[0] == "load_route")
        steps = [event for event in events if event[0] == "route_step"]
        self.assertEqual(len(steps), 2)
        self.assertTrue(all(event[1] >= loaded_at for event in steps))

    def test_route_clearance_can_be_disabled_without_changing_the_policy_action(self):
        app = load_app()
        action = {"shoulder_lift.pos": 10.0, "elbow_flex.pos": 20.0,
                  "wrist_flex.pos": 30.0, "gripper.pos": 5.0}
        settings = app.motion_controls.default_settings(white_clearance=False)
        with patch.object(app, "load_route_kinematics") as kinematics:
            self.assertIs(app.add_route_clearance(action, False, [0] * 6, settings), action)
        kinematics.assert_not_called()

    def test_route_loader_rejects_unhydrated_git_lfs_weights(self):
        app = load_app()
        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "pretrained_model"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_text(
                "version https://git-lfs.github.com/spec/v1\n", encoding="utf-8"
            )
            route = SimpleNamespace(name="up-down-bad", checkpoint=checkpoint)

            with self.assertRaisesRegex(RuntimeError, "git lfs pull"):
                app.load_route_components(route, "cpu")

    def test_cnn_runs_only_after_hold_and_settling(self):
        app, code, events, results, configs, expected_hold = run_app()
        self.assertEqual(code, 0)
        self.assertEqual(len(results), 1)
        sends = [event for event in events if event[0] == "send"]
        holds = [event for event in sends if event[2] == expected_hold]
        self.assertEqual(len(holds), 1)
        hold_time = holds[0][1]
        cnn = [event for event in events if event[0] == "cnn"]
        self.assertGreater(len(cnn), 1)
        self.assertGreaterEqual(cnn[0][1], hold_time + app.SETTLE_S)
        self.assertGreaterEqual(cnn[-1][1] - cnn[0][1], app.RESULT_WINDOW_S)
        self.assertTrue(all(event[2] >= round((hold_time + app.SETTLE_S) * 10) for event in cnn))
        self.assertTrue(all(event[1] < hold_time for event in events if event[0] == "policy"))
        route = next(event for event in events if event[0] == "route_selected")
        route_steps = [event for event in events if event[0] == "route_step"]
        self.assertEqual(route[2], "PIN")
        self.assertGreaterEqual(route[1], cnn[-1][1])
        self.assertEqual(len(route_steps), 2)
        self.assertTrue(all(event[1] >= cnn[-1][1] for event in route_steps))
        self.assertTrue(any(event[0] == "send" and event[2]["gripper.pos"] == 30.0 for event in sends))
        route_commands = [event[2] for event in sends if event[2].get("gripper.pos") == 30.0]
        self.assertTrue(all(np.isfinite(list(command.values())).all() for command in route_commands))
        self.assertTrue(any(command["elbow_flex.pos"] == 0.25 and command["wrist_flex.pos"] == 0.25
                            for command in route_commands))  # final hold is the measured, unshifted pose
        self.assertEqual(sum(event[0] == "reset" for event in events), 1)
        self.assertEqual(configs[0].max_relative_target, 8.0)
        self.assertIsInstance(configs[0].max_relative_target, float)
        self.assertEqual(events[-1][0], "disconnect")

    def test_manual_stop_holds_without_classifying(self):
        _, _, events, results, _, expected_hold = run_app(mode="cancel")
        sends = [event for event in events if event[0] == "send"]
        self.assertTrue(sends)
        self.assertEqual(sends[-1][2], expected_hold)
        self.assertFalse(any(event[0] == "cnn" for event in events))
        self.assertFalse(results)
        self.assertTrue(all(event[1] < sends[-1][1] for event in events if event[0] == "policy"))

    def test_restart_uses_only_new_classification_samples(self):
        _, _, events, results, _, expected_hold = run_app(cycles=2)
        self.assertEqual(results, ["RESULT: PIN (90%)", "RESULT: NO PIN (90%)"])
        self.assertEqual(sum(event[0] == "reset" for event in events), 2)
        self.assertEqual(sum(event[0] == "send" and event[2] == expected_hold for event in events), 2)
        self.assertEqual([event[2] for event in events if event[0] == "route_selected"], ["PIN", "NO PIN"])

    def test_rejected_first_action_does_not_enable_cnn(self):
        _, _, events, results, _, _ = run_app(mode="blocked")
        self.assertFalse(any(event[0] in ("send", "cnn") for event in events))
        self.assertFalse(results)

    def test_settled_pose_finishes_before_maximum_and_resets_for_next_pick(self):
        app, _, events, results, _, _ = run_app(mode="settled", cycles=2, pick_seconds=20)
        resets = [event[1] for event in events if event[0] == "reset"]
        for index, reset in enumerate(resets):
            end = resets[index + 1] if index + 1 < len(resets) else float("inf")
            cnn = [event for event in events if event[0] == "cnn" and reset < event[1] < end]
            sends = [event for event in events if event[0] == "send" and reset <= event[1] < cnn[0][1]]
            hold_time = sends[-1][1]
            self.assertGreaterEqual(hold_time - reset + 1e-8, app.MIN_PICK_S)
            self.assertLess(hold_time - reset, 20)
            self.assertGreaterEqual(cnn[0][1] - hold_time + 1e-8, app.SETTLE_S)
            self.assertTrue(all(event[1] <= hold_time for event in events
                                if event[0] == "policy" and reset <= event[1] < end))
        self.assertEqual(len(resets), 2)
        self.assertEqual(results, ["RESULT: PIN (90%)", "RESULT: NO PIN (90%)"])

    def test_noisy_predictions_finish_early_and_report_stop_metrics(self):
        _, _, events, results, _, _ = run_app(mode="noisy_settled", pick_seconds=20)
        cnn = [event for event in events if event[0] == "cnn"]
        self.assertEqual(len(results), 1)
        self.assertLess(cnn[0][1], 10)
        self.assertTrue(any(event[0] == "status" and "Auto-stop" in event[2] for event in events))
        self.assertFalse(any(event[0] == "policy" and event[1] >= cnn[0][1] for event in events))

    def test_stalled_pose_reports_tracking_gap_and_uses_time_limit(self):
        _, _, events, results, _, _ = run_app(mode="stalled", pick_seconds=8)
        start = next(event[1] for event in events if event[0] == "reset")
        cnn = [event for event in events if event[0] == "cnn"]
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(cnn[0][1] - start, 8)
        self.assertTrue(any(event[0] == "status" and "target gap 29.8/6" in event[2] for event in events))


class PoseCompletionTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app()

    def settled(self, samples, elapsed):
        return self.app.pose_stop_status(samples, elapsed)[0]

    def history(self, measured_arm=10.0, target_arm=10.0):
        return deque((t, np.array([measured_arm] * 5 + [20.0]),
                      np.array([target_arm] * 5 + [0.0])) for t in np.linspace(3.0, 5.0, 21))

    def test_settled_pose_allows_gripper_contact_gap(self):
        self.assertTrue(self.settled(self.history(), 5.0))

    def test_stalled_arm_far_from_raw_target_is_not_finished(self):
        self.assertFalse(self.settled(self.history(target_arm=30.0), 5.0))

    def test_moving_pose_is_not_finished(self):
        measured = self.history()
        for index, (_, state, action) in enumerate(measured):
            state[:5] += index * 0.4
            action[:5] += index * 0.4
        self.assertFalse(self.settled(measured, 5.0))

    def test_stationary_arm_with_target_jitter_and_gripper_contact_finishes(self):
        samples = self.history()
        for index, (_, state, action) in enumerate(samples):
            state[:5] += 0.4 if index % 2 else -0.4
            action[:5] += 4.0 if index % 2 else -4.0
            action[-1] = 0.0 if index % 2 else 20.0
        self.assertTrue(self.settled(samples, 5.0))

    def test_isolated_old_encoder_spike_does_not_prevent_completion(self):
        samples = self.history()
        samples[-5][1][0] += 15.0
        self.assertTrue(self.settled(samples, 5.0))

    def test_latest_tracking_failure_prevents_completion(self):
        samples = self.history()
        samples[-1][2][0] += 20.0
        self.assertFalse(self.settled(samples, 5.0))

    def test_latest_gripper_motion_prevents_completion(self):
        samples = self.history()
        samples[-1][1][-1] += 10.0
        self.assertFalse(self.settled(samples, 5.0))

    def test_minimum_duration_and_full_window_are_required(self):
        self.assertFalse(self.settled(self.history(), 4.0))
        self.assertFalse(self.settled(deque([self.history()[-1]]), 5.0))

    def test_gripper_motion_delays_classification(self):
        samples = self.history()
        samples[-1][1][-1] += 10.0
        self.assertFalse(self.settled(samples, 5.0))

    def test_old_motion_expires_from_bounded_window(self):
        samples = self.history(target_arm=30.0)
        samples.extend((t, np.array([10.0] * 5 + [20.0]), np.array([10.0] * 5 + [0.0]))
                       for t in np.linspace(5.1, 8.0, 30))
        self.assertTrue(self.settled(samples, 8.0))
        self.assertLessEqual(len(samples), 22)


if __name__ == "__main__":
    unittest.main()
