"""The demo comparison is geometric, explicitly synthetic and phase aligned."""
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comparison import PERIOD, demo_joints, tcp_position, comparison_sample
import server as hmi


class ComparisonTests(unittest.TestCase):
    def test_reference_loop_closes_and_stays_inside_model_joint_limits(self):
        from recoverykit.kinematics import limits
        bounds = limits()
        start = demo_joints(0, hmi.JOINTS, False)
        end = demo_joints(PERIOD, hmi.JOINTS, False)
        for name in hmi.JOINTS:
            self.assertAlmostEqual(start[name], end[name])
        for sample in range(181):
            for name, angle in demo_joints(sample * PERIOD / 180, hmi.JOINTS).items():
                self.assertGreaterEqual(angle, bounds[name][0])
                self.assertLessEqual(angle, bounds[name][1])

    def test_tcp_uses_fingertip_offset_and_quaternion_rotation(self):
        self.assertEqual(tcp_position({'gripper': [[1, 2, 3], [1, 0, 0, 0]]}), [1, 2, 2.896])
        self.assertAlmostEqual(tcp_position({'gripper': [[0, 0, 0], [0, 1, 0, 0]]})[2], .104)

    def test_identical_traces_have_zero_separation(self):
        links = hmi.joint_poses(demo_joints(1, hmi.JOINTS))
        sample = comparison_sample(1, links, links)
        self.assertEqual(sample['distance_mm'], 0)
        self.assertEqual(sample['kind'], 'illustrative')

    def test_separation_is_actual_synchronized_euclidean_distance(self):
        for time in [0, 1, 5, 11.9, 12]:
            reference = hmi.joint_poses(demo_joints(time, hmi.JOINTS, False))
            follower = hmi.joint_poses(demo_joints(time, hmi.JOINTS))
            sample = comparison_sample(time, reference, follower)
            expected = math.dist(tcp_position(reference), tcp_position(follower)) * 1000
            self.assertAlmostEqual(sample['distance_mm'], expected, places=2)
            self.assertGreaterEqual(sample['phase'], 0)
            self.assertLess(sample['phase'], 1)

    def test_reference_endpoint_is_labelled_and_has_a_complete_cycle(self):
        handler = object.__new__(hmi.make_handler(hmi.Telemetry('demo')))
        handler.path = '/api/demo-reference'
        handler.send = lambda status, body, content_type='application/json': setattr(handler, 'result', (status, body))
        handler.do_GET()
        import json
        status, body = handler.result
        path = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(path['kind'], 'illustrative')
        self.assertEqual(len(path['points']), 181)
        for x, y in zip(path['points'][0], path['points'][-1]):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
