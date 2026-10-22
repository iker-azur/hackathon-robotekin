"""Microphone recording helpers for RoboQC (sounddevice)."""

from __future__ import annotations

import os
import wave
from typing import List, Optional, Tuple

try:
    import sounddevice as sd
    import numpy as np
    _SD = True
except Exception:
    _SD = False
    sd = None
    np = None


def is_available() -> bool:
    return _SD


def list_devices() -> List[dict]:
    if not _SD:
        return []
    devs = sd.query_devices()
    out = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d.get("name"), "channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate")})
    return out


def record_until_stop(device: Optional[int] = None, samplerate: Optional[int] = None) -> Tuple[bytes, int]:
    """Record mono audio until user presses Enter. Returns (pcm bytes, samplerate)."""
    if not _SD:
        raise RuntimeError("sounddevice no disponible.")
    if samplerate is None:
        if device is None:
            try:
                samplerate = sd.default.samplerate
            except Exception:
                samplerate = None
            if not samplerate:
                try:
                    dev = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
                    info = sd.query_devices(dev)
                    samplerate = info.get("default_samplerate", 16000)
                except Exception:
                    samplerate = 16000
        else:
            info = sd.query_devices(device)
            samplerate = info.get("default_samplerate", 16000)
    try:
        samplerate = int(samplerate)
    except (TypeError, ValueError):
        samplerate = 16000
    # ensure supported
    if samplerate <= 0:
        samplerate = 16000

    frames: List = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="int16", device=device, callback=callback)
    print("Grabando... pulsa ENTER para detener.")
    stream.start()
    try:
        input()
    except EOFError:
        pass
    stream.stop()
    stream.close()

    if not frames:
        return b"", samplerate
    import numpy as np
    data = np.concatenate(frames, axis=0)
    return data.tobytes(), samplerate


def save_wav(path: str, pcm_bytes: bytes, samplerate: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm_bytes)


def load_wav(path: str) -> Tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    # convert to mono 16-bit if needed
    import numpy as np
    if sw != 2:
        raise ValueError("El WAV debe ser PCM de 16 bits. Convierte el archivo antes de transcribir.")
    if nch > 1:
        arr = np.frombuffer(frames, dtype=np.int16).reshape(-1, nch)
        arr = arr.mean(axis=1).astype(np.int16)
        frames = arr.tobytes()
    return frames, sr
