"""Audio preparation shared by the speech engines."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16000


def prepare(pcm_bytes: bytes, samplerate: int) -> np.ndarray:
    """Return 16 kHz mono audio with bounded gain; do not amplify near-silence."""
    if samplerate <= 0 or len(pcm_bytes) % 2:
        raise ValueError("Expected signed 16-bit PCM and a positive sample rate.")
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if samplerate != TARGET_SR and x.size:
        g = gcd(TARGET_SR, samplerate)
        x = resample_poly(x, TARGET_SR // g, samplerate // g).astype(np.float32)
    if not x.size:
        return x
    x = x - x.mean()
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 0.001:
        return np.zeros_like(x)
    if peak > 0:
        x = x * min(3.0, 0.9 / peak)
    return x.astype(np.float32)


def to_int16_bytes(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
