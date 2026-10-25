"""Reach detection and firm hold: what counts as "the pick arrived" and what the hold commands."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load():
    path = Path(__file__).resolve().parent.parent / "reach_detect.py"
    spec = importlib.util.spec_from_file_location("reach_detect_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reach = load()
PROFILE = reach.load_profile()
DT = 1 / 30
REST = np.array([6.0, -100.0, 28.0, 99.0, -5.0, 1.0], dtype=np.float32)


def detector(**kwargs):
    return reach.ReachDetector(PROFILE, **kwargs)


def feed(det, states, targets=None, start=0.0):
    """Run frames through the detector; returns (list of reached flags, last message)."""
    targets = states if targets is None else targets
    flags, message = [], ""
    for i, (state, target) in enumerate(zip(states, targets)):
        reached, message = det.update(start + i * DT, state, target)
        flags.append(reached)
    return flags, message


def pick(open_s=1.0, close_s=1.0, lift_s=2.0, lift=40.0, hold_s=2.0, gripper_open=40.0, gripper_closed=5.0):
    """Frames of a pick: the gripper opens, closes on the piece, the arm lifts and then holds still."""
    frames = []
    for i in range(int(open_s / DT)):
        frames.append(REST + [0, 0, 0, 0, 0, (gripper_open - 1.0) * (i + 1) / int(open_s / DT)])
    for i in range(int(close_s / DT)):
        frames.append(REST + [0, 0, 0, 0, 0, gripper_open - 1.0 - (gripper_open - gripper_closed) * (i + 1) / int(close_s / DT)])
    grasp = frames[-1].copy()
    for i in range(int(lift_s / DT)):
        u = (i + 1) / int(lift_s / DT)
        frames.append(grasp + [0, lift * u, lift * u, 0, 0, 0])
    top = frames[-1].copy()
    frames.extend(top.copy() for _ in range(int(hold_s / DT)))
    return np.array(frames, dtype=np.float32)


class ProfileTests(unittest.TestCase):
    def test_the_committed_profile_has_every_threshold(self):
        for key in ("open_min", "drop", "lift", "window", "motion", "error", "gripper_motion"):
            self.assertGreater(PROFILE[key], 0)
        self.assertEqual(len(PROFILE["hold_gap"]), 5)
        self.assertTrue(np.all(PROFILE["hold_gap"] > 0))

    def test_the_recorded_evaluation_shows_the_new_rule_arrives_more_often_and_not_more_prematurely(self):
        evaluation = json.loads(reach.PROFILE_PATH.read_text())["evaluation"]
        self.assertGreater(evaluation["new"]["fired_share"], 3 * evaluation["today"]["fired_share"])
        self.assertLess(evaluation["new"]["premature_share_of_fired"], 0.05)
        self.assertLess(evaluation["new"]["delay_p90_s"], 1.2)

    def test_a_profile_without_a_threshold_or_with_a_bad_hold_gap_is_refused(self):
        raw = json.loads(reach.PROFILE_PATH.read_text())
        broken = {"missing lift": lambda r: r["thresholds"].pop("lift"),
                  "short hold_gap": lambda r: r["thresholds"].__setitem__("hold_gap", [1, 2, 3]),
                  "zero hold_gap": lambda r: r["thresholds"].__setitem__("hold_gap", [1, 2, 0, 4, 5]),
                  "zero window": lambda r: r["thresholds"].__setitem__("window", 0)}
        for label, damage in broken.items():
            candidate = json.loads(json.dumps(raw))
            damage(candidate)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "profile.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaises((ValueError, KeyError), msg=label):
                    reach.load_profile(path)


class PatternTests(unittest.TestCase):
    def test_a_full_pick_is_called_arrived_a_window_after_the_arm_stops(self):
        frames = pick()
        flags, message = feed(detector(), frames)
        first = flags.index(True)
        stops = int((1.0 + 1.0 + 2.0) / DT)  # the arm reaches the top at 4 s
        self.assertGreaterEqual(first, stops)
        self.assertLessEqual((first - stops) * DT, PROFILE["window"] + 0.1)
        self.assertIn("Alcance", message)

    def test_an_arm_that_is_still_but_has_not_grasped_is_not_there(self):
        frames = np.array([REST] * 150, dtype=np.float32)  # hovering at rest for 5 s, gripper never opens
        flags, message = feed(detector(), frames)
        self.assertFalse(any(flags))
        self.assertIn("esperando el agarre", message)

    def test_a_grasp_without_a_lift_is_not_there(self):
        frames = pick(lift=5.0)
        flags, message = feed(detector(), frames)
        self.assertFalse(any(flags))
        self.assertIn("levante la pieza", message)

    def test_a_lift_that_keeps_moving_is_not_there(self):
        frames = pick(hold_s=0.0)
        wobble = frames[-1] + np.array([[0, 6 * np.sin(i / 3), 0, 0, 0, 0] for i in range(90)])
        flags, _ = feed(detector(), np.vstack([frames, wobble]).astype(np.float32))
        self.assertFalse(any(flags[len(frames):]))

    def test_the_lift_is_measured_from_the_grasp_not_from_rest(self):
        frames = pick(lift=20.0)  # the arm is 100+ degrees from rest but only 20 from where it grasped
        self.assertFalse(any(feed(detector(), frames)[0]))

    def test_a_gripper_that_opens_again_starts_the_question_over(self):
        frames = pick(hold_s=0.0)
        reopen = np.array([frames[-1] + [0, 0, 0, 0, 0, 30.0 * (i + 1) / 15] for i in range(15)], dtype=np.float32)
        det = detector()
        feed(det, np.vstack([frames, reopen]))
        self.assertIsNone(det.grasp_pose)
        self.assertFalse(det.lifted)

    def test_a_thick_piece_that_keeps_the_gripper_half_open_still_counts_as_a_grasp(self):
        frames = pick(gripper_open=60.0, gripper_closed=32.0)  # closes on a thick piece: only falls to 32
        self.assertTrue(any(feed(detector(), frames)[0]))

    def test_a_gripper_that_never_opened_wide_is_not_a_grasp(self):
        frames = pick(gripper_open=12.0, gripper_closed=1.0)  # below open_min
        self.assertFalse(any(feed(detector(), frames)[0]))

    def test_the_arm_not_being_where_the_policy_wants_is_not_there(self):
        frames = pick()
        flags, message = feed(detector(), frames, targets=frames + 12.0)
        self.assertFalse(any(flags))
        self.assertIn("distancia al objetivo", message)

    def test_a_small_steady_gap_to_the_target_is_a_firm_hold_not_a_failure(self):
        frames = pick()
        targets = frames.copy()
        targets[:, 1:4] -= 3.0  # the demonstrations hold with the target a few degrees past the pose
        self.assertTrue(any(feed(detector(), frames, targets=targets)[0]))

    def test_an_isolated_spike_in_the_measurement_does_not_hide_a_still_arm(self):
        frames = pick(hold_s=3.0)
        frames[int(5.0 / DT), 2] += 15.0  # one bad frame in the middle of the hold
        self.assertTrue(any(feed(detector(), frames)[0]))

    def test_the_earliest_second_can_be_set(self):
        flags, message = feed(detector(min_time=30.0), pick())
        self.assertFalse(any(flags))
        self.assertIn("se activa en", message)

    def test_the_stillness_limits_can_be_overridden_like_the_apps_settings(self):
        frames = pick(hold_s=1.0)
        short = feed(detector(window=0.3), frames)[0].index(True)
        long = feed(detector(window=0.9), frames)[0].index(True)
        self.assertLess(short, long)

    def test_an_invalid_measurement_is_reported_not_raised(self):
        det = detector()
        reached, message = det.update(0.0, [np.nan] * 6, [0.0] * 6)
        self.assertFalse(reached)
        self.assertIn("no válida", message)

    def test_reset_forgets_the_pick(self):
        det = detector()
        feed(det, pick())
        det.reset()
        self.assertIsNone(det.grasp_pose)
        self.assertEqual(len(det.samples), 0)
        self.assertFalse(any(feed(det, np.array([REST] * 60, dtype=np.float32))[0]))


class HoldTests(unittest.TestCase):
    def test_the_hold_is_the_policy_target_when_it_is_within_the_demonstrated_gap(self):
        det = detector()
        state = np.array([0, -10, -30, 15, 0, 5], dtype=np.float32)
        feed(det, [state] * 30, targets=[state + [0.5, -2.0, -3.0, -1.5, 0.0, 0.0]] * 30)
        np.testing.assert_allclose(det.hold_target(state), state[:5] + [0.5, -2.0, -3.0, -1.5, 0.0], atol=1e-4)

    def test_the_hold_never_commands_more_than_the_largest_recorded_gap(self):
        det = detector()
        state = np.zeros(6, dtype=np.float32)
        feed(det, [state] * 30, targets=[state + 50.0] * 30)
        np.testing.assert_allclose(det.hold_target(state), PROFILE["hold_gap"], atol=1e-4)

    def test_the_median_ignores_a_stray_target(self):
        det = detector()
        state = np.zeros(6, dtype=np.float32)
        targets = [state + [0, -2.0, 0, 0, 0, 0]] * 29 + [state + [0, 40.0, 0, 0, 0, 0]]
        feed(det, [state] * 30, targets=targets)
        self.assertAlmostEqual(float(det.hold_target(state)[1]), -2.0, places=3)

    def test_without_samples_it_holds_the_measured_pose(self):
        state = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        np.testing.assert_allclose(detector().hold_target(state), state[:5])

    def test_the_hold_returns_the_five_arm_joints_only(self):
        det = detector()
        feed(det, [REST] * 5)
        self.assertEqual(det.hold_target(REST).shape, (5,))


if __name__ == "__main__":
    unittest.main()
