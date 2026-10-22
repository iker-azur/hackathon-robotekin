"""Qt worker threads for RoboQC UI."""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from PySide6.QtCore import QThread, Signal
    _QT = True
except Exception:
    _QT = False
    QThread = object
    Signal = None

from roboqc.speech.recorder import record_until_stop


# Signals are named recorded/transcribed/classified, not finished: a Signal called
# "finished" would shadow QThread.finished, which Qt itself relies on.
class RecordingWorker(QThread if _QT else object):
    recorded = Signal(object, int) if _QT else None  # pcm_bytes, samplerate
    error = Signal(str) if _QT else None

    def __init__(self, device=None, samplerate=None, parent=None):
        if _QT:
            super().__init__(parent)
        self.device = device
        self.samplerate = samplerate
        self._stop_event = threading.Event()

    def run(self):
        try:
            # We can't easily stop via Enter in GUI; record fixed chunks until stop() called.
            # Use sounddevice InputStream with manual stop flag.
            import sounddevice as sd
            import numpy as np
            if self.samplerate is None:
                if self.device is None:
                    try:
                        sr = sd.default.samplerate
                    except Exception:
                        sr = None
                    if not sr:
                        try:
                            dev = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
                            info = sd.query_devices(dev)
                            sr = info.get("default_samplerate", 16000)
                        except Exception:
                            sr = 16000
                    self.samplerate = int(sr) if sr else 16000
                else:
                    info = sd.query_devices(self.device)
                    self.samplerate = int(info.get("default_samplerate", 16000))
            if not self.samplerate or self.samplerate <= 0:
                self.samplerate = 16000
            frames = []
            def cb(indata, frames_count, time_info, status):
                frames.append(indata.copy())
            stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype="int16",
                                    device=self.device, callback=cb)
            stream.start()
            while not self._stop_event.is_set():
                time.sleep(0.1)
            stream.stop()
            stream.close()
            if not frames:
                self.recorded.emit(b"", self.samplerate)
                return
            data = np.concatenate(frames, axis=0)
            self.recorded.emit(data.tobytes(), self.samplerate)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._stop_event.set()


class TranscriptionWorker(QThread if _QT else object):
    transcribed = Signal(str, float) if _QT else None
    error = Signal(str) if _QT else None

    def __init__(self, engine, pcm_bytes, samplerate, parent=None):
        if _QT:
            super().__init__(parent)
        self.engine = engine
        self.pcm_bytes = pcm_bytes
        self.samplerate = samplerate

    def run(self):
        try:
            text, latency = self.engine.transcribe(self.pcm_bytes, self.samplerate)
            self.transcribed.emit(text, latency)
        except Exception as e:
            self.error.emit(str(e))


class ClassificationWorker(QThread if _QT else object):
    classified = Signal(object, object) if _QT else None  # StructuredCommand, [(intent, probability)]
    error = Signal(str) if _QT else None

    def __init__(self, text, source, engine, parent=None, dialogue=None):
        if _QT:
            super().__init__(parent)
        self.text = text
        self.source = source
        self.engine = engine
        self.dialogue = dialogue

    def run(self):
        try:
            from roboqc.industrial.voice import handles, to_legacy
            from roboqc.commands.vision_intents import classify_vision_intent
            bundle = getattr(self.engine, "bundle", None)
            route_industrial = self.dialogue is not None and (self.dialogue.pending or handles(self.text))
            if self.dialogue is not None and not route_industrial and bundle is not None and classify_vision_intent(self.text) is None:
                route_industrial = bundle.predict(self.text)["intent"] == "sort"
            if route_industrial:
                response = self.dialogue.submit(self.text, bundle)
                self.classified.emit(to_legacy(response, self.text, self.source), [])
                return
            from roboqc.commands.pipeline import understand_text
            classifier = None
            conf_fn = None
            if self.engine is not None and getattr(self.engine,"kind",None) != "industrial":
                try:
                    classifier = self.engine.make_classifier_callable()
                    conf_fn = self.engine.make_confidence_callable()
                except Exception:
                    classifier = None
                    conf_fn = None
            cmd = understand_text(self.text, source=self.source, classifier=classifier, confidence_fn=conf_fn)
            topk = []
            if self.engine is not None:
                try:
                    topk = self.engine.predict_topk(self.text, 100)
                except Exception:
                    topk = []
            self.classified.emit(cmd, topk)
        except Exception as e:
            self.error.emit(str(e))


class ModelLoadWorker(QThread if _QT else object):
    loaded = Signal(object) if _QT else None  # loaded IntentInferenceEngine
    error = Signal(str) if _QT else None

    def __init__(self, model_dir, name, parent=None):
        if _QT:
            super().__init__(parent)
        self.model_dir = model_dir
        self.name = name

    def run(self):
        try:
            from roboqc.intent.inference import load_engine
            engine = load_engine(model_dir=self.model_dir, preferred=self.name)
            engine.load()
            self.loaded.emit(engine)
        except Exception as e:
            self.error.emit(str(e))
