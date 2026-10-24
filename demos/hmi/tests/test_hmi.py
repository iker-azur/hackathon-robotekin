"""Telemetry, demo animation and local asset checks without hardware or sockets."""
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hmi_server', ROOT / 'server.py')
hmi = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hmi
spec.loader.exec_module(hmi)
from telegram_control import RobotStartRequests, TelegramControl, control_mode_from_text


class HMIChecks(unittest.TestCase):
    def setUp(self):
        self.telemetry = hmi.Telemetry('test')

    def request(self, path, body=None):
        handler = object.__new__(hmi.make_handler(self.telemetry))
        handler.path = path
        handler.send = lambda status, payload, content_type='application/json': (setattr(handler, 'result', (status, payload, content_type)))
        if body is None:
            handler.do_GET()
        else:
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
            handler.headers = {'Content-Length': str(len(encoded))}
            handler.rfile = io.BytesIO(encoded)
            handler.do_POST()
        return handler.result

    def test_measured_joints_move_the_corresponding_chain(self):
        q = dict.fromkeys(hmi.JOINTS, 0.0)
        initial = hmi.joint_poses(q)
        q['elbow_flex'] = .5
        changed = hmi.joint_poses(q)
        self.assertEqual(initial['base'], changed['base'])
        self.assertEqual(initial['upper_arm'], changed['upper_arm'])
        self.assertNotEqual(initial['gripper'], changed['gripper'])
        for pos, quat in changed.values():
            self.assertTrue(all(math.isfinite(value) for value in pos))
            self.assertAlmostEqual(sum(value * value for value in quat), 1)

    def test_no_sample_is_never_live(self):
        state = self.telemetry.snapshot()
        self.assertFalse(state['connected'])
        self.assertIsNone(state['pose_age'])
        self.assertIsNone(state['frame_age'])

    def test_lost_signal_becomes_stale(self):
        with patch.object(hmi.time, 'monotonic', return_value=10):
            self.telemetry.publish({'joints': dict.fromkeys(hmi.JOINTS, 0)})
            self.assertTrue(self.telemetry.snapshot()['connected'])
        with patch.object(hmi.time, 'monotonic', return_value=13):
            self.assertFalse(self.telemetry.snapshot()['connected'])
            self.assertEqual(self.telemetry.snapshot()['pose_age'], 3)

    def test_camera_and_robot_have_independent_freshness(self):
        with patch.object(hmi.time, 'monotonic', return_value=10):
            self.telemetry.frame(b'\xff\xd8frame\xff\xd9')
        with patch.object(hmi.time, 'monotonic', return_value=13):
            self.telemetry.publish({'joints': dict.fromkeys(hmi.JOINTS, 0)})
            state = self.telemetry.snapshot()
            self.assertTrue(state['connected'])
            self.assertEqual(state['frame_age'], 3)

    def test_malformed_payloads_are_rejected(self):
        for body in [{}, {'joints': {}}, {'joints': dict.fromkeys(hmi.JOINTS, float('nan'))},
                     {'links': {'base': [[0, 0, 0], [0, 0, 0, 0]]}}]:
            with self.subTest(body=body):
                self.assertEqual(self.request('/api/telemetry', body)[0], 400)
        self.assertEqual(self.request('/api/frame.jpg', b'not JPEG')[0], 400)
        self.assertEqual(self.request('/api/frame.jpg')[0], 404)

    def test_telemetry_and_image_routes(self):
        self.assertEqual(self.request('/api/telemetry', {'joints': dict.fromkeys(hmi.JOINTS, 0)})[0], 200)
        self.assertTrue(json.loads(self.request('/api/state')[1])['connected'])
        frame = b'\xff\xd8frame\xff\xd9'
        self.assertEqual(self.request('/api/frame.jpg', frame)[0], 200)
        self.assertEqual(self.request('/api/frame.jpg')[1:], (frame, 'image/jpeg'))
        for name in ('front', 'wrist'):
            path = f'/api/camera/{name}.jpg'
            self.assertEqual(self.request(path, frame)[0], 200)
            self.assertEqual(self.request(path)[1:], (frame, 'image/jpeg'))
            self.assertEqual(json.loads(self.request('/api/state')[1])['camera_frames'][name]['frame_id'], 1)

    def test_telegram_recognizes_and_confirms_manual_mode_phrases(self):
        requests = RobotStartRequests()
        requests.poll('idle', {'control_mode': 'automatic', 'leader_available': True})
        bot = TelegramControl('token', 1, None, None, requests)
        messages = []
        bot._send = lambda chat_id, text: messages.append(text)

        self.assertEqual(control_mode_from_text('pon el modo manual'), 'manual')
        self.assertEqual(control_mode_from_text('controla con el maestro'), 'manual')
        self.assertEqual(control_mode_from_text('cambia a modo automático'), 'automatic')
        self.assertIsNone(control_mode_from_text('no pongas el modo manual'))

        bot._handle_text(1, 'pon el modo manual')
        self.assertIn('CONFIRMAR', messages[-1])
        self.assertEqual(requests.requested_control_mode, 'automatic')
        bot._handle_text(1, 'confirmar')
        self.assertEqual(requests.requested_control_mode, 'manual')

    def test_telegram_automatic_mode_is_requested_without_motion_confirmation(self):
        requests = RobotStartRequests()
        requests.poll('manual', {'control_mode': 'manual', 'leader_available': True})
        bot = TelegramControl('token', 1, None, None, requests)
        messages = []
        bot._send = lambda chat_id, text: messages.append(text)
        bot._handle_text(1, 'pon el modo automático')
        self.assertEqual(requests.requested_control_mode, 'automatic')
        self.assertIn('automático', messages[-1].casefold())

    def test_every_frontend_asset_and_reference_is_served(self):
        for path in ['/', '/style.css', '/app.js', '/robot-model.js', '/robot_meshes.json', '/api/reference',
                     '/vendor/three.module.min.js', '/vendor/three.core.min.js', '/vendor/OrbitControls.js']:
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 200)
        self.assertEqual(self.request('/vendor/../../server.py')[0], 404)
        self.assertEqual(self.request('/api/control', {})[0], 404)

    def test_demo_is_labelled_and_can_pause_without_losing_signal(self):
        class StopAfterSamples:
            def __init__(self, count): self.count = count
            def is_set(self): return self.count <= 0
            def wait(self, seconds): self.count -= 1
        hmi.demo_worker(self.telemetry, StopAfterSamples(2))
        state = self.telemetry.snapshot()
        self.assertEqual(state['mode'], 'demo')
        self.assertTrue(state['connected'])
        self.assertEqual(set(state['poses']['joints']), set(hmi.JOINTS))
        self.assertEqual(self.request('/api/demo/toggle', {})[0], 200)
        self.assertTrue(self.telemetry.snapshot()['paused'])
        samples = []
        publish = self.telemetry.publish
        def capture(data):
            samples.append(data['joints'])
            publish(data)
        with patch.object(self.telemetry, 'publish', side_effect=capture):
            hmi.demo_worker(self.telemetry, StopAfterSamples(3))
        self.assertEqual(samples[0], samples[-1])
        self.assertTrue(self.telemetry.snapshot()['connected'])


if __name__ == '__main__':
    unittest.main()
