"""Trained intent plans, isolated-process lifecycle and HTTP integration without downloads."""

import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from commands import make_plan
from intent_service import IntentService, IntentUnavailable
import server as hmi


def rule(feature='bolt', quality='any', box='big', count='all'):
    return {'feature': feature, 'quality': quality, 'box': box, 'count': count, 'confidence': .97}


def prediction(intent='move', status='ok', rules=None, confidence=.98):
    return {'intent': intent, 'status': status, 'rules': [rule()] if rules is None else rules,
            'confidence': confidence, 'probabilities': {'intent': {intent: confidence}}, 'note': ''}


def service_with(result):
    service = Mock(model_name='IkerAzurmendi/bin-intent-gliner2')
    service.predict.return_value = result
    return service


class IntentPlanTests(unittest.TestCase):
    def test_model_drives_features_destination_quantity_and_confidence(self):
        service = service_with(prediction(rules=[rule('no bolt', box='white', count='two')]))
        plan = make_plan('Mueve dos piezas sin perno a la caja blanca', 'voice', service)
        self.assertEqual(plan['status'], 'proposed')
        self.assertEqual(plan['source'], 'voice')
        self.assertEqual(plan['execution'], 'preview_only')
        self.assertEqual(plan['classifier'], 'GLiNER2 · IkerAzurmendi/bin-intent-gliner2')
        action = plan['actions'][0]
        self.assertEqual(action['parameters']['object_feature'], 'sin perno')
        self.assertEqual(action['parameters']['quantity'], 2)
        self.assertEqual(action['parameters']['destination_role'], 'blanca')
        self.assertIsNone(action['parameters']['object_quality'])
        self.assertEqual(action['confidence'], .97)
        self.assertTrue(any('caja blanca' in step['text'] for step in plan['steps']))
        service.predict.assert_called_once()

    def test_two_model_rules_keep_each_piece_group_linked_to_its_box(self):
        service = service_with(prediction('both', rules=[rule(), rule('no bolt', box='white')]))
        plan = make_plan('Las piezas con perno a la caja grande y las sin perno a la blanca', intent_service=service)
        self.assertEqual(plan['status'], 'proposed')
        self.assertEqual([(a['parameters']['object_feature'], a['parameters']['destination_role'])
                          for a in plan['actions']], [('con perno', 'grande'), ('sin perno', 'blanca')])

    def test_quality_is_separate_from_bolt_presence(self):
        service = service_with(prediction('sort', rules=[rule('any', quality='bad', box='white')]))
        plan = make_plan('Clasifica las piezas malas en la caja blanca', intent_service=service)
        self.assertEqual(plan['actions'][0]['intent'], 'SORT_OBJECTS')
        self.assertIsNone(plan['actions'][0]['parameters']['object_feature'])
        self.assertEqual(plan['actions'][0]['parameters']['object_quality'], 'defectuosa')

    def test_uncertain_missing_box_unsupported_box_and_refusals_produce_no_steps(self):
        for result in [prediction(confidence=.7), prediction(status='ask_box'),
                       prediction(status='not_a_cell_box'), prediction('refuse', 'no_action', [])]:
            with self.subTest(result=result):
                plan = make_plan('Mueve las piezas con perno', intent_service=service_with(result))
                self.assertEqual(plan['status'], 'needs_clarification')
                self.assertEqual(plan['steps'], [])
                self.assertTrue(plan['issues'])

    def test_pick_without_destination_and_english_stop_have_the_right_intents(self):
        pick = make_plan('Recoge una pieza con perno', intent_service=service_with(
            prediction('pick', rules=[rule(box='none', count='one')])))
        self.assertEqual(pick['actions'][0]['intent'], 'PICK_OBJECT')
        stop = make_plan('stop', intent_service=service_with(prediction('stop', 'stop', [])))
        self.assertEqual(stop['actions'][0]['intent'], 'STOP_PROCESS')
        self.assertEqual(stop['status'], 'proposed')

    def test_existing_controls_inspection_and_colour_commands_still_use_labelled_rules(self):
        service = service_with(prediction())
        for text in ('Pausa el proceso', 'Parada de emergencia', 'Inspecciona todas las piezas',
                     'Coloca dos piezas azules en la caja roja'):
            with self.subTest(text=text):
                plan = make_plan(text, intent_service=service)
                self.assertEqual(plan['status'], 'proposed')
                self.assertIn('Reglas de RoboQC', plan['classifier'])
        service.predict.assert_not_called()


class IntentServiceTests(unittest.TestCase):
    def test_missing_environment_and_busy_worker_fail_without_launching(self):
        service = IntentService(python='/missing/intent/python')
        self.assertFalse(service.status()['available'])
        with self.assertRaisesRegex(IntentUnavailable, 'entorno'):
            service.load()
        service.lock.acquire()
        try:
            with patch.object(service, 'load') as load, self.assertRaisesRegex(IntentUnavailable, 'otra orden'):
                service.predict('mueve las piezas')
            load.assert_not_called()
        finally:
            service.lock.release()

    def test_worker_is_reused_and_terminated_cleanly(self):
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.StringIO()
        process.stdout = io.StringIO('\n'.join(json.dumps(row) for row in
                                               [{'ready': True}, prediction(), prediction()]) + '\n')
        service = IntentService(python=sys.executable)
        with patch('intent_service.subprocess.Popen', return_value=process) as launch:
            self.assertEqual(service.predict('Primera orden')['status'], 'ok')
            self.assertEqual(service.predict('Segunda orden')['status'], 'ok')
            self.assertTrue(service.status()['loaded'])
            self.assertEqual(len(process.stdin.getvalue().splitlines()), 2)
            launch.assert_called_once()
        service.close()
        process.terminate.assert_called_once()
        self.assertTrue(process.stdin.closed and process.stdout.closed)
        self.assertFalse(service.status()['loaded'])

    def test_startup_failure_cleans_worker_and_releases_request_lock(self):
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.StringIO()
        process.stdout = io.StringIO('{"error":"Cannot load checkpoint"}\n')
        service = IntentService(python=sys.executable)
        with patch('intent_service.subprocess.Popen', return_value=process), self.assertRaises(IntentUnavailable):
            service.predict('mueve las piezas')
        self.assertFalse(service.lock.locked())
        self.assertIsNone(service.process)
        process.terminate.assert_called_once()


class IntentHTTPTests(unittest.TestCase):
    def request(self, service, path, body=None):
        handler = object.__new__(hmi.make_handler(hmi.Telemetry('test'), intent=service))
        handler.path = path
        handler.send = lambda status, payload, content_type='application/json': setattr(
            handler, 'result', (status, json.loads(payload)))
        if body is None:
            handler.do_GET()
        else:
            encoded = json.dumps(body).encode()
            handler.headers = {'Content-Length': str(len(encoded))}
            handler.rfile = io.BytesIO(encoded)
            handler.do_POST()
        return handler.result

    def test_text_and_voice_use_the_same_trained_model_and_expose_its_status(self):
        service = service_with(prediction())
        service.status.return_value = {'engine': 'GLiNER2 local', 'loaded': True}
        self.assertEqual(self.request(service, '/api/intent/status')[1]['engine'], 'GLiNER2 local')
        for source in ('text', 'voice'):
            status, plan = self.request(service, '/api/command',
                                        {'text': 'Mueve las piezas con perno a la caja grande', 'source': source})
            self.assertEqual(status, 200)
            self.assertEqual(plan['source'], source)
            self.assertEqual(plan['status'], 'proposed')
            self.assertEqual(plan['execution'], 'preview_only')
        self.assertEqual(service.predict.call_count, 2)

    def test_unavailable_model_returns_503_and_has_no_robot_control_endpoint(self):
        service = service_with(prediction())
        service.predict.side_effect = IntentUnavailable('Modelo no disponible')
        status, body = self.request(service, '/api/command', {'text': 'Mueve las piezas con perno'})
        self.assertEqual(status, 503)
        self.assertEqual(body['detail'], 'Modelo no disponible')
        self.assertEqual(self.request(service, '/api/control', {})[0], 404)


if __name__ == '__main__':
    unittest.main()
