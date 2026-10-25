"""Party mode motion: the numbers behind "safe and slow", checked on the committed profile."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load():
    path = Path(__file__).resolve().parent.parent / "party_dance.py"
    spec = importlib.util.spec_from_file_location("party_dance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dance = load()
REST = np.array([6.2, -100.1, 28.9, 99.5, -5.2, 0.9], dtype=np.float32)
PROFILE = dance.load_profile()
CHOREOGRAPHY = dance.Choreography(PROFILE)


def speeds(samples, dt):
    return np.abs(np.diff(samples, axis=0)) / dt


class ProfileTests(unittest.TestCase):
    def test_the_committed_profile_is_consistent(self):
        self.assertTrue(np.all(PROFILE["low"] <= PROFILE["home"]))
        self.assertTrue(np.all(PROFILE["home"] <= PROFILE["high"]))
        self.assertTrue(np.all(PROFILE["swing"] > 0))

    def test_the_swings_stay_inside_the_envelope_with_a_margin(self):
        self.assertTrue(np.all(PROFILE["home"] - PROFILE["swing"] >= PROFILE["low"] + 1.9))
        self.assertTrue(np.all(PROFILE["home"] + PROFILE["swing"] <= PROFILE["high"] - 1.9))

    def test_a_home_outside_the_envelope_is_refused(self):
        raw = json.loads(dance.PROFILE_PATH.read_text())
        raw["home"][0] = raw["high"][0] + 5
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                dance.load_profile(path)

    def test_a_profile_without_six_joints_is_refused(self):
        raw = json.loads(dance.PROFILE_PATH.read_text())
        raw["swing"] = raw["swing"][:5]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                dance.load_profile(path)


class RiseTests(unittest.TestCase):
    def test_the_rise_starts_where_the_arm_is_and_ends_on_the_standard_pose(self):
        seconds = dance.rise_seconds(REST, PROFILE["home"])
        np.testing.assert_allclose(dance.rise_target(REST, PROFILE["home"], 0.0, seconds), REST, atol=1e-4)
        np.testing.assert_allclose(dance.rise_target(REST, PROFILE["home"], seconds, seconds), PROFILE["home"], atol=1e-4)
        np.testing.assert_allclose(dance.rise_target(REST, PROFILE["home"], seconds * 3, seconds), PROFILE["home"], atol=1e-4)

    def test_the_rise_is_slow_no_joint_exceeds_the_rise_speed(self):
        seconds = dance.rise_seconds(REST, PROFILE["home"])
        times = np.arange(0, seconds, 0.01)
        path = np.array([dance.rise_target(REST, PROFILE["home"], t, seconds) for t in times])
        self.assertLessEqual(speeds(path, 0.01).max(), dance.RISE_SPEED * 1.01)
        self.assertGreaterEqual(seconds, dance.RISE_MIN_S)

    def test_a_short_rise_still_takes_the_minimum_time(self):
        near = PROFILE["home"] + 1.0
        self.assertEqual(dance.rise_seconds(near, PROFILE["home"]), dance.RISE_MIN_S)

    def test_the_rise_is_monotonic_per_joint(self):
        seconds = dance.rise_seconds(REST, PROFILE["home"])
        path = np.array([dance.rise_target(REST, PROFILE["home"], t, seconds) for t in np.arange(0, seconds, 0.05)])
        direction = np.sign(PROFILE["home"] - REST)
        self.assertTrue(np.all(np.diff(path, axis=0) * direction >= -1e-4))


class ChoreographyTests(unittest.TestCase):
    TIMES = np.arange(0, 120, 0.005)

    def targets(self):
        return np.array([CHOREOGRAPHY.target(t) for t in self.TIMES])

    def test_the_dance_starts_exactly_on_the_standard_pose(self):
        np.testing.assert_allclose(CHOREOGRAPHY.target(0.0), PROFILE["home"], atol=1e-4)

    def test_every_target_stays_inside_the_recorded_envelope(self):
        path = self.targets()
        self.assertTrue(np.all(path >= PROFILE["low"] - 1e-3))
        self.assertTrue(np.all(path <= PROFILE["high"] + 1e-3))

    def test_no_joint_ever_exceeds_the_speed_limit_including_the_fade_in(self):
        peak = speeds(self.targets(), 0.005).max(0)
        self.assertTrue(np.all(peak <= dance.MAX_SPEED), peak)

    def test_the_analytic_peak_speeds_are_under_the_limit_too(self):
        self.assertTrue(np.all(CHOREOGRAPHY.peak_speeds() <= dance.MAX_SPEED), CHOREOGRAPHY.peak_speeds())

    def test_every_joint_actually_moves(self):
        travelled = np.ptp(self.targets()[int(10 / 0.005):], axis=0)  # after the fade-in
        self.assertTrue(np.all(travelled > PROFILE["swing"] * 1.2), travelled)  # a swing each way, nearly

    def test_the_motion_repeats_every_sixteen_beats(self):
        period = 16 * 60.0 / dance.BPM
        np.testing.assert_allclose(CHOREOGRAPHY.target(20.0), CHOREOGRAPHY.target(20.0 + period), atol=1e-3)

    def test_the_fade_in_is_gradual(self):
        early = np.abs(CHOREOGRAPHY.target(0.5) - PROFILE["home"]).max()
        late = np.abs(CHOREOGRAPHY.target(15.0) - PROFILE["home"]).max()
        self.assertLess(early, late)

    def test_the_beat_follows_the_tempo(self):
        self.assertAlmostEqual(CHOREOGRAPHY.beat(60.0), dance.BPM)


if __name__ == "__main__":
    unittest.main()
