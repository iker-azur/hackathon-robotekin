"""Say it instead of typing it. Needs a microphone.

Both packages this needs are already in the environment - ``faster-whisper`` 1.2.1 and
``sounddevice`` 0.5.6 - and ``Systran/faster-whisper-small`` is already in the Hugging Face
cache, so the first run does not download anything and this works with the network off.

**Why the model is a good fit for this demo rather than a generic one.** Whisper returns
lower-case text with no punctuation, and that is not a degradation here: demo 03's command
model was trained with a ``train_asr.jsonl`` set of the same sentences *as speech recognition
returns them* - 25% of the generated sentences have their punctuation stripped. So the
parser has already seen this shape of text and the demo's three languages are all in
Whisper's language list, Basque included.

Nothing here is imported unless ``--voice`` is passed, so a run without voice never pays for
it and never fails because there is no microphone.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

# Block size for the input stream. 100 ms is short enough that the silence gate reacts while
# someone is still talking, and long enough not to be a busy loop.
BLOCK_SECONDS = 0.1

# Above this RMS the microphone is hearing speech. -40 dBFS is a room, not a person; it is
# deliberately low so a quiet voice still starts the recording, and the silence gate below is
# what actually decides when it has finished.
THRESHOLD_DB = -40.0

# Stop after this much silence once speech has started.
SILENCE_SECONDS = 1.2

# Give up after this long whether or not anything was said, so an idle microphone cannot leave
# the demo waiting forever.
MAX_SECONDS = 10.0


class Listener:
    """Records from the microphone until the speaker stops, then transcribes it.

    ``model`` is any faster-whisper model name. ``tiny``, ``base``, ``small`` and ``medium``
    are already cached locally. ``language`` may be None to let Whisper detect it, which is
    what you want with three languages in play; forcing one is more reliable if you know who
    is talking.
    """

    def __init__(self, model: str = "small", language: str | None = None,
                 device: str = "cpu", compute_type: str = "int8",
                 cache: str | Path | None = None):
        from faster_whisper import WhisperModel

        self.language = language
        # int8 on CPU is several times faster than float32 and, for a short command, just as
        # accurate. On CUDA, int8 needs extra cuBLAS libraries that are not always present, so
        # float16 there.
        if device == "cuda":
            compute_type = "float16"
        self.model = WhisperModel(model, device=device, compute_type=compute_type,
                                  download_root=str(cache) if cache else None)
        self.rate = 16000  # what Whisper resamples to anyway

    def listen(self, max_seconds: float = MAX_SECONDS,
               silence: float = SILENCE_SECONDS) -> str:
        """Record one utterance and return it as text, or "" if nothing was heard."""
        import sounddevice as sd

        frames: list[np.ndarray] = []
        speaking = False
        quiet = 0.0
        started = time.perf_counter()

        with sd.InputStream(samplerate=self.rate, channels=1, dtype="float32") as stream:
            while True:
                block, _overflow = stream.read(int(self.rate * BLOCK_SECONDS))
                level = float(np.sqrt(np.mean(np.square(block))))
                loud = 20.0 * math.log10(max(level, 1e-9)) > THRESHOLD_DB
                if loud:
                    speaking = True
                    quiet = 0.0
                elif speaking:
                    quiet += BLOCK_SECONDS
                if speaking:
                    frames.append(block.copy())

                waited = time.perf_counter() - started
                if speaking and quiet >= silence:
                    break
                if waited >= max_seconds:
                    break

        if not frames:
            return ""

        audio = np.concatenate(frames).astype(np.float32)
        segments, _info = self.model.transcribe(audio, language=self.language,
                                               beam_size=1, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def close(self) -> None:
        return None

    def __enter__(self) -> "Listener":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def check_microphone() -> str | None:
    """A message describing why voice will not work, or None if it will."""
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        return f"sounddevice is not usable: {exc}"

    try:
        devices = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception as exc:  # noqa: BLE001
        return f"no audio device could be listed: {exc}"
    if not devices:
        return "no input device: this machine has no microphone as far as sounddevice can see"
    return None