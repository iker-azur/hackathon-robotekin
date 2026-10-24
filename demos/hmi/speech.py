"""Lazy local Whisper transcription for browser or Telegram audio."""
from __future__ import annotations

import importlib.util
import io
import threading
from pathlib import Path

CACHE = Path(__file__).resolve().parent / '.cache' / 'whisper'
INSTALL = 'uv pip install --python .venv/bin/python -r demos/hmi/requirements-voice.txt'


class SpeechUnavailable(RuntimeError):
    pass


class SpeechService:
    def __init__(self, model='tiny'):
        self.model_name = model
        self.model = None
        self.lock = threading.Lock()

    def status(self):
        available = importlib.util.find_spec('faster_whisper') is not None
        return {'available': available, 'loaded': self.model is not None,
                'engine': 'Whisper local', 'model': self.model_name,
                'detail': ('La primera transcripción carga el modelo; puede necesitar descargarlo.' if available
                           else 'Falta instalar el motor de transcripción local.'),
                'install_command': INSTALL}

    def load(self):
        if self.model is not None:
            return
        if not self.status()['available']:
            raise SpeechUnavailable('Falta Whisper. Ejecuta: ' + INSTALL)
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_name, device='cpu', compute_type='int8',
                                      cpu_threads=2, download_root=str(CACHE))
        except Exception as exc:
            raise SpeechUnavailable('No se pudo cargar Whisper. Comprueba la conexión para la primera descarga del modelo.') from exc

    def transcribe(self, audio):
        if not self.status()['available']:
            raise SpeechUnavailable('Falta Whisper. Ejecuta: ' + INSTALL)
        from faster_whisper.audio import decode_audio
        try:
            samples = decode_audio(io.BytesIO(audio), sampling_rate=16000)
        except Exception as exc:
            raise ValueError('El audio está vacío, incompleto o en un formato no compatible.') from exc
        if len(samples) < 1600:
            raise ValueError('La grabación es demasiado corta. Mantén el micrófono activo mientras hablas.')
        if len(samples) > 31 * 16000:
            raise ValueError('La grabación supera el límite de 30 segundos.')
        if max(abs(samples.min()), abs(samples.max())) < .002:
            raise ValueError('No se ha detectado señal de voz. Comprueba el micrófono y vuelve a grabar.')
        if not self.lock.acquire(blocking=False):
            raise SpeechUnavailable('Ya se está transcribiendo otra grabación. Espera e inténtalo de nuevo.')
        try:
            self.load()
            try:
                segments, _ = self.model.transcribe(samples, language='es', beam_size=5,
                                                    condition_on_previous_text=False, vad_filter=True)
                segments = list(segments)
                if not segments or any(segment.no_speech_prob > 0.5 or segment.avg_logprob < -1.0
                                       for segment in segments):
                    raise ValueError("No he reconocido el audio con suficiente seguridad. Repite la frase.")
                text = ' '.join(segment.text.strip() for segment in segments).strip()
            except ValueError:
                raise
            except Exception as exc:
                raise SpeechUnavailable('Whisper no pudo transcribir el audio. Comprueba la instalación del motor local.') from exc
            if not text:
                raise ValueError('No se ha reconocido voz. Revisa el nivel del micrófono y vuelve a grabar.')
            return {'text': text, 'engine': 'Whisper local', 'model': self.model_name,
                    'duration': round(len(samples) / 16000, 2)}
        finally:
            self.lock.release()
