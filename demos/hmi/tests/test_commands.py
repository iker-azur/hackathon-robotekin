import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from commands import make_plan
from speech import SpeechService, SpeechUnavailable
import server as hmi


class CommandTests(unittest.TestCase):
    def test_pick_place_preserves_quantity_object_and_destination(self):
        plan = make_plan('Coloca dos piezas azules en la caja roja')
        self.assertEqual(plan['status'], 'proposed')
        self.assertEqual(plan['actions'][0]['intent'], 'PLACE_OBJECT')
        self.assertEqual(plan['actions'][0]['parameters']['quantity'], 2)
        self.assertEqual(plan['actions'][0]['parameters']['object_color'], 'azul')
        self.assertEqual(plan['actions'][0]['parameters']['destination_color'], 'rojo')
        self.assertTrue(any('2 piezas de color azul' in step['text'] for step in plan['steps']))
        self.assertEqual(plan['steps'][-1]['text'], 'Verificar la colocación con la cámara')
        self.assertEqual(plan['execution'], 'preview_only')

    def test_inspection_and_pause_are_distinct_intents(self):
        inspect = make_plan('Inspecciona todas las piezas', 'voice')
        pause = make_plan('Pausa el proceso')
        self.assertEqual(inspect['actions'][0]['intent'], 'INSPECT_BATCH')
        self.assertEqual(inspect['source'], 'voice')
        self.assertEqual(pause['actions'][0]['intent'], 'PAUSE_PROCESS')
        self.assertNotEqual(inspect['steps'], pause['steps'])
        self.assertTrue(all(step['status'] == 'proposed' for step in pause['steps']))

    def test_pronoun_uses_previous_object_within_same_order(self):
        plan = make_plan('Coge la pieza roja y ponla en la caja azul')
        self.assertEqual(plan['status'], 'proposed')
        self.assertEqual(plan['actions'][1]['parameters']['object_color'], 'rojo')
        self.assertEqual(plan['actions'][1]['parameters']['destination_color'], 'azul')

    def test_negated_unknown_and_incomplete_commands_have_no_action_plan(self):
        for text in ['No cojas la pieza roja', 'Cuéntame un chiste', 'Coloca la pieza azul', 'Mueve a la caja roja']:
            with self.subTest(text=text):
                plan = make_plan(text)
                self.assertEqual(plan['status'], 'needs_clarification')
                self.assertEqual(plan['steps'], [])
                self.assertTrue(plan['issues'])

    def test_bad_input_rejected(self):
        for text in ['', ' ', None, 25, 'a' * 2001]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    make_plan(text)
        with self.assertRaises(ValueError):
            make_plan('Inspecciona las piezas', 'invalid-source')

    def test_new_routes_validate_requests_and_expose_missing_speech(self):
        speech = Mock()
        speech.status.return_value = {'available': False}
        speech.transcribe.side_effect = SpeechUnavailable('Falta Whisper')
        handler_class = hmi.make_handler(hmi.Telemetry('test'), speech)
        def request(path, payload=None):
            handler = object.__new__(handler_class)
            handler.path = path
            handler.send = lambda status, body, content_type='application/json': setattr(handler, 'result', (status, body))
            if payload is None:
                handler.do_GET()
            else:
                encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                handler.headers = {'Content-Length': str(len(encoded))}
                handler.rfile = io.BytesIO(encoded)
                handler.do_POST()
            return handler.result
        self.assertFalse(json.loads(request('/api/voice/status')[1])['available'])
        self.assertEqual(request('/voice.js')[0], 200)
        status, body = request('/api/command', {'text': 'Inspecciona las piezas', 'source': 'voice'})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['source'], 'voice')
        self.assertEqual(request('/api/command', [1, 2])[0], 400)
        self.assertEqual(request('/api/command', {'text': ''})[0], 400)
        self.assertEqual(request('/api/voice/transcribe', b'recorded-audio')[0], 503)
        speech.transcribe.side_effect = None
        speech.transcribe.return_value = {'text': 'Pausa el proceso'}
        self.assertEqual(json.loads(request('/api/voice/transcribe', b'audio')[1])['text'], 'Pausa el proceso')


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.speech = SpeechService()
        self.speech.status = lambda: {'available': True}
        self.model = Mock()
        self.speech.model = self.model
        self.audio_module = types.ModuleType('faster_whisper.audio')
        self.audio_module.decode_audio = Mock(return_value=np.full(16000, .1, dtype=np.float32))

    def test_decoded_audio_is_transcribed_locally_in_spanish(self):
        self.model.transcribe.return_value = (iter([types.SimpleNamespace(text=' Pausa el proceso ')]), None)
        with patch.dict(sys.modules, {'faster_whisper.audio': self.audio_module}):
            result = self.speech.transcribe(b'audio-bytes')
        self.assertEqual(result['text'], 'Pausa el proceso')
        self.assertEqual(result['duration'], 1)
        self.assertEqual(self.model.transcribe.call_args.kwargs['language'], 'es')
        self.assertTrue(self.model.transcribe.call_args.kwargs['vad_filter'])
        self.assertFalse(self.speech.lock.locked())

    def test_empty_silent_long_and_corrupt_audio_are_rejected(self):
        for samples in [np.array([]), np.zeros(16000), np.ones(32 * 16000)]:
            self.audio_module.decode_audio.return_value = samples
            with self.subTest(size=len(samples)), patch.dict(sys.modules, {'faster_whisper.audio': self.audio_module}):
                with self.assertRaises(ValueError): self.speech.transcribe(b'audio')
        self.audio_module.decode_audio.side_effect = RuntimeError('bad codec')
        with patch.dict(sys.modules, {'faster_whisper.audio': self.audio_module}):
            with self.assertRaises(ValueError): self.speech.transcribe(b'corrupt')
        self.model.transcribe.assert_not_called()

    def test_busy_or_failed_model_does_not_leave_lock_held(self):
        self.speech.lock.acquire()
        with patch.dict(sys.modules, {'faster_whisper.audio': self.audio_module}):
            with self.assertRaises(SpeechUnavailable): self.speech.transcribe(b'audio')
        self.speech.lock.release()
        self.model.transcribe.side_effect = RuntimeError('failed inference')
        with patch.dict(sys.modules, {'faster_whisper.audio': self.audio_module}):
            with self.assertRaises(SpeechUnavailable): self.speech.transcribe(b'audio')
        self.assertFalse(self.speech.lock.locked())


if __name__ == '__main__':
    unittest.main()
