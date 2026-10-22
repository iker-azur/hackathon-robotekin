import os
import tempfile

from roboqc.intent.inference import IntentInferenceEngine
from roboqc.speech.recorder import is_available as rec_available
from roboqc.speech.transcriber import is_available as vosk_available


def test_missing_model_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        eng = IntentInferenceEngine(model_dir=d)
        try:
            eng.load()
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


def test_empty_audio_text_handled():
    from roboqc.commands.parser import parse_text
    cmd = parse_text("   ")
    assert cmd.status == "INVALID"


def test_unknown_not_crash():
    from roboqc.commands.parser import parse_text
    cmd = parse_text("Pon música")
    assert cmd.actions[0].intent == "UNKNOWN"


def test_recorder_available_returns_bool():
    assert isinstance(rec_available(), bool)


def test_vosk_available_returns_bool():
    assert isinstance(vosk_available(), bool)
