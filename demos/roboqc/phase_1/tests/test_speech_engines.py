import numpy as np
import pytest

from roboqc.speech.audio import prepare, to_int16_bytes, TARGET_SR
from roboqc.speech.engines import ENGINES, DEFAULT_ENGINE, get_engine


def _pcm(samples):
    return np.asarray(samples, dtype=np.int16).tobytes()


def test_prepare_resamples_to_target_rate():
    pcm = _pcm(np.zeros(44100, dtype=np.int16))
    assert len(prepare(pcm, 44100)) == TARGET_SR


def test_prepare_removes_dc_and_limits_peak():
    t = np.linspace(0, 1, TARGET_SR, endpoint=False)
    wave = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1) * 32767
    x = prepare(_pcm(wave), TARGET_SR)
    assert abs(float(x.mean())) < 1e-3
    assert float(np.max(np.abs(x))) == pytest.approx(0.9, abs=1e-3)


def test_prepare_does_not_amplify_silence_or_unbounded_noise():
    assert not prepare(_pcm([1, -1] * 100), TARGET_SR).any()
    x = prepare(_pcm([100, -100] * 100), TARGET_SR)
    assert float(np.max(np.abs(x))) <= 300 / 32768 + 1e-6


def test_prepare_empty_input_is_empty():
    assert prepare(b"", 44100).size == 0


def test_to_int16_bytes_clips_range():
    out = np.frombuffer(to_int16_bytes(np.array([2.0, -2.0, 0.0], dtype=np.float32)), dtype=np.int16)
    assert out.tolist() == [32767, -32767, 0]


def test_engine_registry_is_consistent():
    assert DEFAULT_ENGINE in ENGINES
    assert all(name == e.name for name, e in ENGINES.items())


def test_get_engine_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_engine("no-such-engine")
