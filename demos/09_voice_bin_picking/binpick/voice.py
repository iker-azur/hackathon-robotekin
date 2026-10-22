"""Voz: del micrófono a texto en español, reutilizando el ``Listener`` del demo 05 (faster-whisper).

No se reescribe el reconocimiento: se carga ``05_planner_and_executor/voice.py`` por ruta (la carpeta no
es un nombre de módulo válido) y se le fija el idioma a español, que es más fiable que detectarlo con
frases tan cortas. El modelo (``Systran/faster-whisper-small``) ya está en la caché de Hugging Face de los
demos anteriores, así que no se descarga nada.

**Hay que cargarlo antes de arrancar Isaac** (``VoiceService.load()``): dos librerías nativas pesadas
cargadas después de que Kit arranque rompen el proceso (demo 10). La transcripción en sí se hace luego en
un hilo del servidor, sin tocar el simulador.

Sin micrófono el panel sigue funcionando con texto: ``VoiceService.problem()`` dice por qué no hay voz.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path

from . import DEMO_ROOT

VOICE_FILE = DEMO_ROOT.parent / "05_planner_and_executor" / "voice.py"


def _load_demo05():
    if not VOICE_FILE.is_file():
        raise FileNotFoundError(f"no encuentro {VOICE_FILE}")
    spec = importlib.util.spec_from_file_location("demo05_voice", VOICE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VoiceService:
    def __init__(self, model: str = "small", language: str | None = "es", device: str = "cpu"):
        self.model, self.language, self.device = model, language, device
        self._mod = None
        self._listener = None
        self._lock = threading.Lock()
        self.busy = False

    def problem(self) -> str | None:
        """Por qué la voz no va a funcionar (o ``None`` si va a funcionar)."""
        try:
            mod = self._mod or _load_demo05()
        except Exception as exc:  # noqa: BLE001
            return f"no se pudo cargar el reconocimiento de voz: {exc}"
        return mod.check_microphone()

    def load(self) -> None:
        """Carga Whisper. Llamar ANTES de arrancar Isaac."""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._mod = _load_demo05()
        self._listener = self._mod.Listener(model=self.model, language=self.language, device=self.device)

    @property
    def loaded(self) -> bool:
        return self._listener is not None

    def listen(self) -> str:
        """Graba una frase y la transcribe. Bloquea hasta ~10 s; llamar desde un hilo."""
        if not self.loaded:
            raise RuntimeError("la voz no está cargada (VoiceService.load() antes de arrancar Isaac)")
        with self._lock:
            self.busy = True
            try:
                return self._listener.listen()
            finally:
                self.busy = False


class TextOnly:
    """Sustituto sin micrófono: ``problem()`` explica por qué y ``listen()`` falla con claridad."""

    loaded = False
    busy = False

    def __init__(self, why: str = "voz desactivada (--voice para activarla)"):
        self.why = why

    def problem(self) -> str | None:
        return self.why

    def listen(self) -> str:
        raise RuntimeError(self.why)
