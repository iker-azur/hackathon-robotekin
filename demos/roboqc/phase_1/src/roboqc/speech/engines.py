"""Selectable offline speech-to-text engines (Vosk and faster-whisper)."""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Dict, List, Tuple

from roboqc.speech import transcriber
from roboqc.speech.audio import prepare

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Vocabulary of the industrial commands; biases Whisper towards the domain terms.
DOMAIN_PROMPT = (
    "Instrucciones para un robot industrial: inspecciona, clasifica, coge, mueve, pon, "
    "rechaza, acepta, para, reintenta, informe. Colores rojo, verde, azul, amarillo. "
    "Piezas buenas, defectuosas tipo uno y tipo dos. Cajas roja, verde y azul. "
    "Tornillos, tuercas, arandelas y soportes. Deten y continua la clasificacion."
)


class VoskEngine:
    def __init__(self, name: str, label: str, model_dir: str):
        self.name = name
        self.label = label
        self.model_dir = model_dir

    def available(self) -> bool:
        return transcriber.is_available() and transcriber.model_exists(self.model_dir)

    def transcribe(self, pcm: bytes, samplerate: int) -> Tuple[str, float]:
        return transcriber.transcribe_pcm(self.model_dir, pcm, samplerate)


class WhisperEngine:
    def __init__(self, name: str, label: str, size: str):
        self.name = name
        self.label = label
        self.size = size
        self._model = None

    def available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def transcribe(self, pcm: bytes, samplerate: int) -> Tuple[str, float]:
        t0 = time.perf_counter()
        samples = prepare(pcm, samplerate)
        if not samples.size or not samples.any():
            return "", time.perf_counter() - t0
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.size, device="cpu", compute_type="int8",
                                       cpu_threads=int(os.environ.get("ROBOQC_CPU_THREADS", "2")))
        segments, _ = self._model.transcribe(
            samples,
            language="es",
            beam_size=1,
            initial_prompt=DOMAIN_PROMPT,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, time.perf_counter() - t0


ENGINES: Dict[str, object] = {
    e.name: e for e in (
        VoskEngine("vosk-small", "Vosk small", os.environ.get(
            "VOSK_MODEL_PATH", os.path.join(ROOT, "models", "vosk-model-small-es"))),
        VoskEngine("vosk-es", "Vosk grande", os.path.join(ROOT, "models", "vosk-model-es-0.42")),
        WhisperEngine("whisper-tiny", "Whisper tiny", "tiny"),
        WhisperEngine("whisper-base", "Whisper base", "base"),
        WhisperEngine("whisper-small", "Whisper small", "small"),
    )
}

DEFAULT_ENGINE = "whisper-base"


def available_engines() -> List[str]:
    return [name for name, e in ENGINES.items() if e.available()]


def get_engine(name: str):
    if name not in ENGINES:
        raise KeyError(f"Motor de voz desconocido: {name}. Disponibles: {', '.join(ENGINES)}")
    return ENGINES[name]
