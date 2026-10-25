"""Voice in, intent reading out, for ``30_full_demo_app.py``: microphone -> faster-whisper (es / en / eu) -> intent model.

* ``Transcriber`` detects which of Spanish, English and Basque was spoken, then transcribes with the model the
  voice demos use for that language (``small`` for es and en, the Basque fine-tune ``xezpeleta/whisper-small-eu-ct2``
  for eu; see ``demos/02_voice/06_voice_command.py``). The recording gate is the one of
  ``demos/05_planner_and_executor/voice.py`` (RMS above -40 dBFS starts it, silence ends it).
* ``load_intent`` loads ``IkerAzurmendi/bin-intent-gliner2`` in this process, or, when GLiNER2 cannot be imported next
  to LeRobot (their ``huggingface_hub`` versions conflict), in the isolated worker of ``demos/hmi``.
* ``VoiceWorker`` runs both off the control loop. The window polls ``events``; nothing here touches the robot.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

DEMOS = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000
LANGUAGES = ("es", "en", "eu")
STT_MODELS = {"es": "small", "en": "small", "eu": "xezpeleta/whisper-small-eu-ct2"}
PROMPTS = {  # the demo's own vocabulary, so Whisper spells "perno" and "caja grande" the way the model expects
    "es": "Mueve las piezas con perno a la caja grande y las piezas sin perno a la caja blanca.",
    "en": "Move the pieces with a pin to the big box and the pieces without a pin to the white box.",
    "eu": "Eraman perno duten piezak kutxa handira eta perno gabekoak kutxa zurira.",
}
THRESHOLD_DB = -40.0  # speech starts above this level (a room is about -55)
BLOCK_S = 0.1
MIN_SPEECH_S = 0.4  # shorter than this is a click or a cough, not an order


class Transcriber:
    def __init__(self, device: str = "cpu", model_factory=None) -> None:
        self.device = device
        self._factory = model_factory or self._whisper
        self._models: dict = {}
        self._lock = threading.Lock()

    def _whisper(self, name: str):
        from faster_whisper import WhisperModel

        compute = "float16" if self.device == "cuda" else "int8"
        return WhisperModel(name, device=self.device, compute_type=compute)

    def model(self, name: str):
        with self._lock:
            if name not in self._models:
                self._models[name] = self._factory(name)
            return self._models[name]

    def detect_language(self, audio: np.ndarray) -> str:
        """The most probable of es / en / eu: Whisper's other languages are not options in this demo."""
        _, _, probabilities = self.model(STT_MODELS["es"]).detect_language(audio)
        scores = {code: p for code, p in probabilities if code in LANGUAGES}
        return max(scores, key=scores.get) if scores else "es"

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """(text, language) of 16 kHz mono float32 audio; the text is empty when nothing intelligible was said."""
        language = self.detect_language(audio)
        segments, _ = self.model(STT_MODELS[language]).transcribe(
            audio, language=language, beam_size=3, vad_filter=False, condition_on_previous_text=False,
            initial_prompt=PROMPTS[language])
        return " ".join(segment.text.strip() for segment in segments).strip(), language


def record_utterance(abort: threading.Event, max_seconds: float = 10.0, silence_s: float = 1.0,
                     start_timeout_s: float = 8.0, device=None) -> np.ndarray | None:
    """Record from the microphone until the speaker stops. None if nobody spoke or ``abort`` was set."""
    import sounddevice as sd

    frames, speaking, quiet, started = [], False, 0.0, time.perf_counter()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device) as stream:
        while not abort.is_set():
            block, _ = stream.read(int(SAMPLE_RATE * BLOCK_S))
            level = float(np.sqrt(np.mean(np.square(block))))
            loud = 20.0 * math.log10(max(level, 1e-9)) > THRESHOLD_DB
            if loud:
                speaking, quiet = True, 0.0
            elif speaking:
                quiet += BLOCK_S
            if speaking:
                frames.append(block[:, 0].copy())
            waited = time.perf_counter() - started
            if (speaking and quiet >= silence_s) or waited >= (max_seconds if speaking else start_timeout_s):
                break
    audio = np.concatenate(frames).astype(np.float32) if frames else None
    return audio if audio is not None and len(audio) >= MIN_SPEECH_S * SAMPLE_RATE else None


class _InProcessIntent:
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, text: str) -> dict:
        plan = self.model.plan(text)
        return {"status": plan.status, "intent": plan.reading.intent, "confidence": plan.reading.confidence,
                "probabilities": plan.reading.probabilities,
                "rules": [{**asdict(rule), "confidence": rule.confidence} for rule in plan.rules], "note": plan.note}

    def close(self) -> None:
        pass


class _WorkerIntent:
    def __init__(self, service) -> None:
        self.service = service

    def predict(self, text: str) -> dict:
        return self.service.predict(text)

    def close(self) -> None:
        self.service.close()


def load_intent(model: str = "IkerAzurmendi/bin-intent-gliner2", device: str = "cpu"):
    """An object with ``predict(text) -> dict`` (the reply of ``demos/hmi/intent_worker.py``) and ``close()``."""
    errors = []
    try:
        sys.path.insert(0, str(DEMOS / "10_bin_intent_model"))
        from intent_model import BinIntentModel

        return _InProcessIntent(BinIntentModel(model, device=device))
    except Exception as exc:  # GLiNER2 missing, or its huggingface_hub clashes with LeRobot's
        errors.append(f"in this process: {exc!r}")
    try:
        sys.path.insert(0, str(DEMOS / "hmi"))
        from intent_service import IntentService

        service = IntentService(model)
        service.load()
        return _WorkerIntent(service)
    except Exception as exc:
        errors.append(f"isolated worker (demos/hmi/.venv-intent): {exc!r}")
    raise RuntimeError("Could not load the intent model. " + " | ".join(errors))


class VoiceWorker:
    """Listen -> transcribe -> read the intent, in a thread. ``events`` carries (kind, payload) tuples:

    ``("listening", None)``, ``("loading", text)``, ``("heard", (text, language))`` and ``("error", text)``;
    ``("reading", (text, mapped_text, language, reading_dict))`` for each sentence the intent model read.
    A request that came with a ``reply`` callable is preceded by ``("source", reply)``.
    """

    def __init__(self, transcriber: Transcriber, intent, vocabulary=lambda text: text, recorder=record_utterance) -> None:
        self.transcriber, self.intent, self.vocabulary, self.recorder = transcriber, intent, vocabulary, recorder
        self.events: queue.Queue = queue.Queue()
        self.requests: queue.Queue = queue.Queue()
        self.abort = threading.Event()
        self.hands_free = threading.Event()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def listen_once(self) -> None:
        self.requests.put("listen")

    def submit_text(self, text: str, reply=None) -> None:
        """A typed order goes through the same intent model; no language is detected, so replies are in Spanish.
        ``reply`` (Telegram) is handed to the window with the order so its answer goes back to the sender."""
        self.requests.put(("text", text, reply))

    def submit_audio(self, audio: np.ndarray, reply=None) -> None:
        """16 kHz mono float32 speech recorded elsewhere (a Telegram voice note)."""
        self.requests.put(("audio", audio, reply))

    def _run(self) -> None:
        while not self.closed.is_set():
            try:
                request = self.requests.get(timeout=0.2)
            except queue.Empty:
                request = "listen" if self.hands_free.is_set() else None
            if request is None:
                continue
            try:
                if request == "listen":
                    self.abort.clear()
                    self.events.put(("listening", None))
                    audio = self.recorder(self.abort)
                    if audio is None:
                        continue
                    self.events.put(("loading", "Transcribiendo…"))
                    text, language = self.transcriber.transcribe(audio)
                else:
                    kind, content, reply = request
                    if reply is not None:  # every request ends in an "error" or a "reading" event
                        self.events.put(("source", reply))
                    if kind == "audio":
                        self.events.put(("loading", "Transcribiendo nota de voz…"))
                        text, language = self.transcriber.transcribe(content)
                    else:
                        text, language = content, None
                if not text:
                    self.events.put(("error", "No se ha entendido ninguna palabra."))
                    continue
                self.events.put(("heard", (text, language)))
                mapped = self.vocabulary(text)
                self.events.put(("reading", (text, mapped, language, self.intent.predict(mapped))))
            except Exception as exc:
                self.events.put(("error", f"Voz: {exc}"))

    def close(self) -> None:
        self.closed.set()
        self.abort.set()
        self.thread.join(timeout=3)
        self.intent.close()
