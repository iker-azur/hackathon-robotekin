"""Vosk-based offline Spanish transcription helpers."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

try:
    import vosk
    _VOSK = True
except Exception:
    _VOSK = False
    vosk = None

from roboqc.speech.audio import prepare, to_int16_bytes, TARGET_SR

_models: Dict[str, object] = {}


def is_available() -> bool:
    return _VOSK


def model_exists(model_path: str) -> bool:
    return os.path.isdir(model_path) and (
        os.path.exists(os.path.join(model_path, "am", "final.mdl"))
        or os.path.exists(os.path.join(model_path, "conf"))
    )


def _load(model_path: str):
    if model_path not in _models:
        _models[model_path] = vosk.Model(model_path)
    return _models[model_path]


def transcribe_pcm(model_path: str, pcm_bytes: bytes, samplerate: int) -> Tuple[str, float]:
    """Return (text, latency_seconds). The model is loaded once and reused."""
    if not _VOSK:
        raise RuntimeError("vosk no disponible.")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Modelo Vosk no encontrado: {model_path}")
    t0 = time.perf_counter()
    samples = prepare(pcm_bytes, samplerate)
    if not samples.size or not samples.any():
        return "", time.perf_counter() - t0
    model = _load(model_path)
    rec = vosk.KaldiRecognizer(model, TARGET_SR)
    rec.AcceptWaveform(to_int16_bytes(samples))
    text = json.loads(rec.FinalResult()).get("text", "")
    return text, time.perf_counter() - t0
