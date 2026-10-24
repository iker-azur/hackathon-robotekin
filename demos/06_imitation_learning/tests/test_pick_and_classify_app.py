"""Exercise the pick/hold/classify sequence without cameras, downloads or motors."""

import importlib.util
import unittest
from collections import deque
from contextlib import ExitStack
from pathlib import Path
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


def run_app(mode="complete", cycles=1, pick_seconds=1):
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
            frame = np.full((2, 2, 3), round(clock.now * 10), dtype=np.uint8)
            return {**measured, "front": frame, "wrist": frame.copy()}

        def send_action(self, action):
            events.append(("send", clock.now, action.copy()))
            return action

        def disconnect(self):
            events.append(("disconnect", clock.now))

    class Policy:
        config = SimpleNamespace(device="cpu")

        def eval(self):
            return self

        def reset(self):
            events.append(("reset", clock.now))

        def select_action(self, batch):
            events.append(("policy", clock.now))
            if mode == "settled":
                return torch.tensor([[0.25] * 5 + [5.0]])
            return torch.full((1, len(joints)), 200.0 if mode == "blocked" else 5.0)

    def classifier(front, wrist):
        events.append(("cnn", clock.now, int(front[0, 0, 0])))
        return 0.9 if sum(event[0] == "reset" for event in events) == 1 else 0.1

    class Window:
        def __init__(self, on_key):
            self.open, self.started, self.stopped, self.probed = True, False, False, False
            self.on_key = on_key

        def show(self, frames, status, color, live, result):
            if not self.started and clock.now >= 0.4:
                self.started = True
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
            if clock.now > (4 if mode in ("cancel", "blocked") else max(12, cycles * (pick_seconds + 4))):
                self.open = False

        def close(self):
            self.open = False

    def make_robot(config):
        configurations.append(config)
        return Robot()

    def sleep(seconds):
        clock.now += max(seconds, 0.01)

    with ExitStack() as stack:
        replacements = [
            patch.object(app, "load_script", return_value=runner),
            patch.object(app, "PinClassifier", return_value=classifier),
            patch.object(app, "resolve_classifier", return_value=Path("unused.pt")),
            patch.object(app, "Window", Window),
            patch.object(app.time, "perf_counter", side_effect=lambda: clock.now),
            patch("sys.argv", ["24_pick_and_classify_app.py", "--device", "cpu", "--fps", "10",
                               "--pick-seconds", str(pick_seconds)]),
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
    def test_cnn_runs_only_after_hold_and_settling(self):
        app, code, events, results, configs, expected_hold = run_app()
        self.assertEqual(code, 0)
        self.assertEqual(len(results), 1)
        sends = [event for event in events if event[0] == "send"]
        holds = [event for event in sends if event[2] == expected_hold]
        self.assertEqual(len(holds), 1)
        self.assertEqual(sends[-1], holds[0])
        hold_time = holds[0][1]
        cnn = [event for event in events if event[0] == "cnn"]
        self.assertGreater(len(cnn), 1)
        self.assertGreaterEqual(cnn[0][1], hold_time + app.SETTLE_S)
        self.assertGreaterEqual(cnn[-1][1] - cnn[0][1], app.RESULT_WINDOW_S)
        self.assertTrue(all(event[2] >= round((hold_time + app.SETTLE_S) * 10) for event in cnn))
        self.assertTrue(all(event[1] < hold_time for event in events if event[0] == "policy"))
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


class PoseCompletionTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app()

    def history(self, measured_arm=10.0, target_arm=10.0):
        return deque((t, np.array([measured_arm] * 5 + [20.0]),
                      np.array([target_arm] * 5 + [0.0])) for t in np.linspace(3.0, 5.0, 21))

    def test_settled_pose_allows_gripper_contact_gap(self):
        self.assertTrue(self.app.pose_is_settled(self.history(), 5.0))

    def test_stalled_arm_far_from_raw_target_is_not_finished(self):
        self.assertFalse(self.app.pose_is_settled(self.history(target_arm=30.0), 5.0))

    def test_moving_pose_or_moving_target_is_not_finished(self):
        measured = self.history()
        targets = self.history()
        for index, (_, state, action) in enumerate(measured):
            state[:5] += index * 0.1
            action[:5] += index * 0.1
        for index, (_, _, action) in enumerate(targets):
            action[:5] += index * 0.15
        self.assertFalse(self.app.pose_is_settled(measured, 5.0))
        self.assertFalse(self.app.pose_is_settled(targets, 5.0))

    def test_minimum_duration_and_full_window_are_required(self):
        self.assertFalse(self.app.pose_is_settled(self.history(), 4.0))
        self.assertFalse(self.app.pose_is_settled(deque([self.history()[-1]]), 5.0))

    def test_gripper_motion_delays_classification(self):
        samples = self.history()
        samples[-1][1][-1] += 10.0
        self.assertFalse(self.app.pose_is_settled(samples, 5.0))

    def test_old_motion_expires_from_bounded_window(self):
        samples = self.history(target_arm=30.0)
        samples.extend((t, np.array([10.0] * 5 + [20.0]), np.array([10.0] * 5 + [0.0]))
                       for t in np.linspace(5.1, 8.0, 30))
        self.assertTrue(self.app.pose_is_settled(samples, 8.0))
        self.assertLessEqual(len(samples), 22)


if __name__ == "__main__":
    unittest.main()
