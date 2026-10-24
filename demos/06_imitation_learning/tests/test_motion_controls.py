"""Verify that GUI tuning changes execution, rather than just the displayed values."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from test_pick_and_classify_app import load_app, run_app


class MotionSettingsTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app()
        self.controls = self.app.motion_controls

    def test_defaults_keep_checkpoint_parameters_and_allow_custom_decimal_values(self):
        values = self.controls.parse_settings({"pick_step": "7,5", "green_actions": "Modelo",
                                               "white_actions": "13"})
        self.assertEqual(values["pick_step"], 7.5)
        self.assertIsNone(values["green_actions"])
        self.assertEqual(values["white_actions"], 13)
        self.assertTrue(values["auto_stop"])
        self.assertEqual(len(values), 32)

    def test_invalid_numbers_and_reversed_gripper_thresholds_are_rejected(self):
        for values in ({"pick_step": "nan"}, {"control_hz": "inf"}, {"pick_seconds": ""},
                       {"white_actions": "2.5"}, {"green_seconds": "-3"},
                       {"green_closed": 20, "green_released": 10}, {"auto_stop": "maybe"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.controls.parse_settings(values)

    def test_action_steps_rebuild_the_queue_and_model_choice_restores_its_default(self):
        policy = SimpleNamespace(config=SimpleNamespace(chunk_size=90, n_action_steps=30), reset=Mock())
        self.assertEqual(self.app.set_action_steps(policy, 5, 30), 5)
        self.assertEqual(policy.config.n_action_steps, 5)
        self.assertEqual(self.app.set_action_steps(policy, 100, 30), 90)
        self.assertEqual(self.app.set_action_steps(policy, None, 30), 30)
        self.assertEqual(policy.reset.call_count, 3)

    def test_stop_thresholds_and_disabled_auto_stop_change_the_completion_decision(self):
        def samples():
            return deque((t, np.zeros(6), np.full(6, 4.0)) for t in (5.0, 5.5, 6.0, 6.5))
        self.assertTrue(self.app.pose_stop_status(samples(), 6.5)[0])
        strict = self.controls.default_settings(arm_error=2)
        self.assertFalse(self.app.pose_stop_status(samples(), 6.5, strict)[0])
        disabled = self.controls.default_settings(auto_stop=False)
        self.assertFalse(self.app.pose_stop_status(samples(), 6.5, disabled)[0])

    def test_box_height_and_gripper_opening_control_the_cartesian_offset(self):
        arm = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        pose = np.eye(4)
        pose[:3, 3] = (0.1, 0.03, 0.2)
        goals = []

        def ik(goal, base, q, **kwargs):
            goals.append(goal.copy())
            return SimpleNamespace(ok=True, q=q)

        kinematics = SimpleNamespace(ARM=arm, fk=lambda q: {"gripper": pose}, ik=ik)
        action = {f"{joint}.pos": 0.0 for joint in arm}
        action["gripper.pos"] = 12.5
        values = self.controls.default_settings(green_height_cm=10, white_height_cm=2)
        with patch.object(self.app, "load_route_kinematics", return_value=kinematics):
            self.app.add_route_clearance(action, False, [0] * 6, values, "green")
            self.app.add_route_clearance(action, False, [0] * 6, values, "white")
        self.assertAlmostEqual(goals[0][2, 3], 0.25)  # half open: half of the 10 cm lift
        self.assertAlmostEqual(goals[1][2, 3], 0.21)
        np.testing.assert_array_equal(goals[0][:3, :3], pose[:3, :3])
        np.testing.assert_array_equal(goals[0][:2, 3], pose[:2, 3])

    def test_descent_control_uses_millimetres_and_unreachable_lift_keeps_policy_motion(self):
        arm = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        policy_pose, measured_pose = np.eye(4), np.eye(4)
        policy_pose[2, 3], measured_pose[2, 3] = 0.1, 0.2
        kinematics = Mock()
        kinematics.ARM = arm
        kinematics.fk.side_effect = [{"gripper": policy_pose}, {"gripper": measured_pose}]
        kinematics.ik.return_value = SimpleNamespace(ok=False)
        action = {f"{joint}.pos": 5.0 for joint in arm}
        action["gripper.pos"] = 5.0
        settings = self.controls.default_settings(white_height_cm=0, white_descent_mm=1)
        with patch.object(self.app, "load_route_kinematics", return_value=kinematics):
            output = self.app.add_route_clearance(action, False, [0] * 6, settings)
        self.assertIs(output, action)
        for call in kinematics.ik.call_args_list:
            self.assertAlmostEqual(call.args[0][2, 3], 0.199)


class MotionExecutionTests(unittest.TestCase):
    def test_controls_change_motor_caps_replanning_timing_and_classifier_threshold(self):
        settings = {"pick_step": 3, "white_step": 2, "pick_actions": 5, "white_actions": 7,
                    "control_hz": 5, "pick_seconds": 0.6, "auto_stop": False,
                    "settle_seconds": 1, "classify_seconds": 0.5, "pin_threshold": 95}
        _, code, events, results, configs, expected_hold = run_app(motion=settings)
        self.assertEqual(code, 0)
        self.assertEqual(results, ["RESULT: NO PIN (10%)"])
        self.assertEqual(next(e[2] for e in events if e[0] == "reset"), 5)
        self.assertEqual([e[2] for e in events if e[0] == "route_reset"][-1], 7)
        cnn = [e for e in events if e[0] == "cnn"]
        hold = next(e for e in events if e[0] == "send" and e[2] == expected_hold)
        self.assertGreaterEqual(cnn[0][1] - hold[1], 1.0)
        self.assertGreaterEqual(cnn[-1][1] - cnn[0][1], 0.5)
        self.assertTrue(all(e[3] == 3 for e in events if e[0] == "send" and e[1] < cnn[0][1]))
        self.assertTrue(all(e[3] == 2 for e in events if e[0] == "send" and e[1] > cnn[-1][1]))
        self.assertEqual(configs[0].max_relative_target, 3)
        predictions = [e[1] for e in events if e[0] == "policy"]
        self.assertTrue(all(abs(b - a - 0.2) < 1e-6 for a, b in zip(predictions, predictions[1:])))

    def test_settings_edits_are_applied_to_the_next_cycle(self):
        _, _, events, _, _, _ = run_app(cycles=2, motion={"pick_step": 2},
                                        next_cycle_motion={"pick_step": 4})
        starts = [e[1] for e in events if e[0] == "reset"]
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else float("inf")
            cnn_start = next(e[1] for e in events if e[0] == "cnn" and start < e[1] < end)
            pick_sends = [e for e in events if e[0] == "send" and start <= e[1] < cnn_start]
            self.assertTrue(pick_sends)
            self.assertTrue(all(e[3] == (2 if index == 0 else 4) for e in pick_sends))

    def test_invalid_tuning_prevents_movement_and_classification(self):
        _, _, events, results, _, _ = run_app(mode="blocked", motion={"pick_step": "NaN"})
        self.assertFalse(any(e[0] in ("send", "policy", "cnn") for e in events))
        self.assertFalse(results)

    def test_route_finish_can_use_time_limit_and_configured_release_threshold(self):
        _, _, events, _, _, _ = run_app(motion={"white_finish": False, "white_seconds": 0.5})
        self.assertFalse(any(e[0] == "route_finished" for e in events))
        steps = [e[1] for e in events if e[0] == "route_step"]
        self.assertGreaterEqual(steps[-1] - steps[0], 0.4 - 1e-6)
        _, _, events, _, _, _ = run_app(motion={"white_released": 8})
        self.assertTrue(all(e[2] for e in events if e[0] == "route_finished"))


if __name__ == "__main__":
    unittest.main()
