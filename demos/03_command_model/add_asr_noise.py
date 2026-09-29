"""Step 2 (optional): add the mistakes speech recognition really makes.

The robot never reads typed text: it reads what Whisper heard. This script
takes a sample of the generated training sentences, says each one with a Piper
voice, transcribes it with the same Whisper models demo 06 uses, and keeps the
transcripts that came out different ("en el cesto" -> "en el incesto",
lost accents, merged words...). Their answers are the original ones, so the
model learns to read through those mistakes.

A transcript is kept only if every labelled word (item phrase, name, colour...)
still appears in it; otherwise the labels would no longer point at real words.

Writes data/train_asr.jsonl; train.py uses it automatically when it exists.
About 10 minutes for the default 600 sentences per language on an RTX 5060 Ti.

Run:

    python .\\demos\\03_command_model\\add_asr_noise.py
    python .\\demos\\03_command_model\\add_asr_noise.py --per-language 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VOICE_DEMO = HERE.parent / "02_voice" / "06_voice_command.py"
# Two voices per language where Piper has them, so the noise is not one speaker's.
VOICES = {
    "es": ["es/es_ES/davefx/medium/es_ES-davefx-medium", "es/es_ES/sharvard/medium/es_ES-sharvard-medium"],
    "en": ["en/en_US/lessac/medium/en_US-lessac-medium", "en/en_US/amy/medium/en_US-amy-medium"],
    "eu": ["eu/eu_ES/maider/medium/eu_ES-maider-medium", "eu/eu_ES/antton/medium/eu_ES-antton-medium"],
}


def load_voice06():
    spec = importlib.util.spec_from_file_location("voice06", VOICE_DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def similarity(a: str, b: str) -> float:
    """0..1: how alike two sentences are, ignoring case and punctuation."""
    import difflib
    import re

    def clean(text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def labelled_words(record: dict) -> list[str]:
    words = []
    for structure in record["output"].get("json_structures", []):
        words += [v for v in structure["item"].values() if isinstance(v, str)]
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    import numpy as np
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass

    voice06 = load_voice06()
    records = [json.loads(line) for line in (DATA / "train_commands.jsonl").open(encoding="utf-8")]
    rng = random.Random(args.seed)
    by_lang = {lang: [r for r in records if r["meta"]["lang"] == lang] for lang in VOICES}
    models, device = voice06.load_models("auto", cpu_only=False)
    print(f"speech recognition on the {device.upper()}", flush=True)

    kept, tried, start = [], 0, time.perf_counter()
    for lang, paths in VOICES.items():
        voices = []
        for path in paths:
            hf_hub_download("rhasspy/piper-voices", path + ".onnx.json")
            voices.append((path.split("/")[-1], PiperVoice.load(hf_hub_download("rhasspy/piper-voices", path + ".onnx"))))
        model = models[voice06.STT_MODELS[lang]]
        for record in rng.sample(by_lang[lang], min(args.per_language, len(by_lang[lang]))):
            tried += 1
            name, voice = rng.choice(voices)
            chunks = list(voice.synthesize(record["input"]))
            audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
            n = int(len(audio) * voice06.SAMPLE_RATE / chunks[0].sample_rate)
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
            heard = voice06.transcribe(model, audio.astype(np.float32), lang).strip()
            if not heard or similarity(heard, record["input"]) == 1.0:
                continue  # same words (only case or punctuation changed): nothing new to learn
            if not all(word.lower() in heard.lower() for word in labelled_words(record)):
                continue  # a labelled word was misheard: the labels no longer fit
            if similarity(heard, record["input"]) < 0.75:
                continue  # too different ("chao" -> "¡Todo!"): the labels may no longer be true
            kept.append({"input": heard, "output": record["output"],
                         "meta": {**record["meta"], "asr": {"original": record["input"], "voice": name}}})
        print(f"  {lang}: {len([k for k in kept if k['meta']['lang'] == lang])} kept "
              f"({time.perf_counter() - start:.0f} s so far)", flush=True)

    with (DATA / "train_asr.jsonl").open("w", encoding="utf-8") as sink:
        for record in kept:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} of {tried} transcripts to {DATA / 'train_asr.jsonl'}")
    for record in kept[:8]:
        print(f"  {record['meta']['asr']['original']!r}\n    -> {record['input']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
