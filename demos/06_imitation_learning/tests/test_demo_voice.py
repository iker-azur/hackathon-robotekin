"""Three-language transcription and the listen -> transcribe -> intent worker, without a microphone or models."""

import importlib.util
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load():
    path = Path(__file__).resolve().parent.parent / "demo_voice.py"
    spec = importlib.util.spec_from_file_location("demo_voice_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


voice = load()
AUDIO = np.zeros(16000, dtype=np.float32)


class FakeWhisper:
    def __init__(self, name, probabilities, text):
        self.name, self.probabilities, self.text, self.calls = name, probabilities, text, []

    def detect_language(self, audio):
        return self.probabilities[0][0], self.probabilities[0][1], self.probabilities

    def transcribe(self, audio, **options):
        self.calls.append(options)
        return iter([SimpleNamespace(text=f" {self.text} ")]), None


def transcriber(probabilities, texts=None):
    texts = texts or {}
    models = {}

    def factory(name):
        models[name] = FakeWhisper(name, probabilities, texts.get(name, "hola"))
        return models[name]

    return voice.Transcriber(model_factory=factory), models


class TranscriberTests(unittest.TestCase):
    def test_only_spanish_english_and_basque_are_candidates(self):
        t, _ = transcriber([("fr", 0.5), ("eu", 0.25), ("en", 0.2), ("es", 0.05)])
        self.assertEqual(t.detect_language(AUDIO), "eu")

    def test_no_candidate_falls_back_to_spanish(self):
        t, _ = transcriber([("fr", 0.6), ("de", 0.4)])
        self.assertEqual(t.detect_language(AUDIO), "es")

    def test_spanish_and_english_use_small_and_basque_uses_the_fine_tune(self):
        for language, model in (("es", "small"), ("en", "small"), ("eu", "xezpeleta/whisper-small-eu-ct2")):
            t, models = transcriber([(language, 0.9), ("fr", 0.1)], {model: f"texto {language}"})
            text, detected = t.transcribe(AUDIO)
            self.assertEqual((text, detected), (f"texto {language}", language))
            self.assertIn(model, models)

    def test_language_and_domain_prompt_are_passed_to_whisper(self):
        t, models = transcriber([("en", 0.9), ("es", 0.1)])
        t.transcribe(AUDIO)
        options = models["small"].calls[0]
        self.assertEqual(options["language"], "en")
        self.assertIn("white box", options["initial_prompt"])
        self.assertFalse(options["condition_on_previous_text"])

    def test_models_are_loaded_once(self):
        loaded = []
        t = voice.Transcriber(model_factory=lambda name: loaded.append(name) or FakeWhisper(
            name, [("es", 0.9)], "hola"))
        t.transcribe(AUDIO)
        t.transcribe(AUDIO)
        self.assertEqual(loaded, ["small"])


class FakeIntent:
    def __init__(self, fail=False):
        self.texts, self.fail, self.closed = [], fail, False

    def predict(self, text):
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("boom")
        return {"status": "ok", "intent": "move", "rules": []}

    def close(self):
        self.closed = True


class FakeTranscriber:
    def __init__(self, text="mueve todo a la caja verde", language="es"):
        self.result = (text, language)

    def transcribe(self, audio):
        return self.result


def collect(worker, count, timeout=3.0):
    return [worker.events.get(timeout=timeout) for _ in range(count)]


class VoiceWorkerTests(unittest.TestCase):
    def worker(self, intent=None, recorder=lambda abort: AUDIO, transcriber=None):
        self.intent = intent or FakeIntent()
        worker = voice.VoiceWorker(transcriber or FakeTranscriber(), self.intent,
                                   lambda text: text.replace("verde", "grande"), recorder)
        self.addCleanup(worker.close)
        return worker

    def test_listen_reports_each_step_and_reads_the_mapped_sentence(self):
        worker = self.worker()
        worker.listen_once()
        events = collect(worker, 4)
        self.assertEqual([kind for kind, _ in events], ["listening", "loading", "heard", "reading"])
        self.assertEqual(events[2][1], ("mueve todo a la caja verde", "es"))
        text, mapped, language, reading = events[3][1]
        self.assertEqual((text, mapped, language), ("mueve todo a la caja verde", "mueve todo a la caja grande", "es"))
        self.assertEqual(reading["status"], "ok")
        self.assertEqual(self.intent.texts, ["mueve todo a la caja grande"])

    def test_silence_produces_no_reading(self):
        worker = self.worker(recorder=lambda abort: None)
        worker.listen_once()
        self.assertEqual(collect(worker, 1)[0][0], "listening")
        with self.assertRaises(queue.Empty):
            worker.events.get(timeout=0.5)
        self.assertEqual(self.intent.texts, [])

    def test_empty_transcript_is_reported(self):
        worker = self.worker(transcriber=FakeTranscriber(text=""))
        worker.listen_once()
        kinds = [kind for kind, _ in collect(worker, 3)]
        self.assertEqual(kinds, ["listening", "loading", "error"])

    def test_typed_order_goes_through_the_same_model_without_a_language(self):
        worker = self.worker()
        worker.submit_text("pon todo en la caja verde")
        events = collect(worker, 2)
        self.assertEqual(events[0], ("heard", ("pon todo en la caja verde", None)))
        self.assertEqual(events[1][1][2], None)
        self.assertEqual(self.intent.texts, ["pon todo en la caja grande"])

    def test_a_failure_becomes_an_error_event_and_the_worker_keeps_running(self):
        worker = self.worker(intent=FakeIntent(fail=True))
        worker.submit_text("hola")
        self.assertEqual([kind for kind, _ in collect(worker, 2)], ["heard", "error"])
        self.intent.fail = False
        worker.submit_text("otra vez")
        self.assertEqual([kind for kind, _ in collect(worker, 2)], ["heard", "reading"])

    def test_hands_free_keeps_listening_until_switched_off(self):
        worker = self.worker()
        worker.hands_free.set()
        self.assertEqual(collect(worker, 4)[3][0], "reading")
        self.assertEqual(collect(worker, 1)[0][0], "listening")  # it started the next recording on its own
        worker.hands_free.clear()

    def test_a_voice_note_is_transcribed_and_read_with_its_reply_target(self):
        worker = self.worker()
        reply = lambda text: None
        worker.submit_audio(AUDIO, reply)
        events = collect(worker, 4)
        self.assertEqual([kind for kind, _ in events], ["source", "loading", "heard", "reading"])
        self.assertIs(events[0][1], reply)
        self.assertEqual(events[2][1], ("mueve todo a la caja verde", "es"))
        self.assertEqual(self.intent.texts, ["mueve todo a la caja grande"])

    def test_a_typed_order_with_a_reply_announces_its_source_first(self):
        worker = self.worker()
        reply = lambda text: None
        worker.submit_text("hola", reply)
        events = collect(worker, 3)
        self.assertEqual([kind for kind, _ in events], ["source", "heard", "reading"])
        self.assertIs(events[0][1], reply)

    def test_a_failing_voice_note_still_ends_in_an_error_event(self):
        worker = self.worker(intent=FakeIntent(fail=True))
        worker.submit_audio(AUDIO, lambda text: None)
        self.assertEqual([kind for kind, _ in collect(worker, 4)], ["source", "loading", "heard", "error"])

    def test_a_party_sentence_skips_the_intent_model(self):
        worker = self.worker(transcriber=FakeTranscriber(text="vamos de fiesta"))
        worker.party = lambda text: "fiesta" in text
        worker.listen_once()
        events = collect(worker, 4)
        self.assertEqual([kind for kind, _ in events], ["listening", "loading", "heard", "party"])
        self.assertEqual(events[3][1], ("vamos de fiesta", "es"))
        self.assertEqual(self.intent.texts, [])  # decided on the words, before any classification

    def test_a_typed_party_with_a_reply_target_announces_its_source_then_the_party(self):
        worker = self.worker()
        worker.party = lambda text: "festa" in text
        reply = lambda text: None
        worker.submit_text("festa egin dezagun", reply)
        events = collect(worker, 3)
        self.assertEqual([kind for kind, _ in events], ["source", "heard", "party"])
        self.assertEqual(self.intent.texts, [])

    def test_other_sentences_still_reach_the_intent_model_when_a_party_check_is_set(self):
        worker = self.worker()
        worker.party = lambda text: "fiesta" in text
        worker.submit_text("pon todo en la caja verde")
        self.assertEqual([kind for kind, _ in collect(worker, 2)], ["heard", "reading"])
        self.assertEqual(self.intent.texts, ["pon todo en la caja grande"])

    def test_close_stops_the_intent_model(self):
        worker = self.worker()
        worker.close()
        self.assertTrue(self.intent.closed)


if __name__ == "__main__":
    unittest.main()
