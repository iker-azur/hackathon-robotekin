"""Voice command with a "System One" model as the action classifier instead of rules.

Same pipeline as ``06_voice_command.py`` (speech -> command -> spoken reply, in
Spanish, English or Basque, detected automatically), but the step
"text -> command" is done by a model built like TypeSafe's Jev: it reads the
sentence together with the list of possible answers and returns a probability
for every answer in one pass. It never writes text, so it cannot invent an
answer that is not on the list, and every decision comes with a confidence.

Classifiers (``--classifier``):

* ``gliner`` (default, open source, local): Fastino's GLiNER2.5-multi
  (Apache-2.0, 0.3B, mDeBERTa-v3 encoder), a schema-driven
  classifier/extractor, fine-tuned for these commands by
  ``train_command_model.py`` (run it once first). One pass answers six
  questions at once (intent, object kind, object colour, place kind, place
  colour, one or several) and extracts the word naming the object, in about
  30 ms on the GPU.
* ``jev``: TypeSafe's Jev over its HTTP API. One request with six typed
  questions (intent, object, object colour, place, place colour, plural);
  unknown words of the sentence are offered as candidate object names, the
  pattern TypeSafe documents for extraction. Needs ``TYPESAFE_API_KEY``.

A command whose weakest decision is below ``CONFIDENCE_MIN`` is not executed:
the robot asks to repeat instead. The command format is the same as demo 06.

Run:

    python .\\demos\\02_voice\\07_action_classifier.py                    # microphone, gliner
    python .\\demos\\02_voice\\07_action_classifier.py --classifier jev   # needs TYPESAFE_API_KEY
    python .\\demos\\02_voice\\07_action_classifier.py --text "coge la esfera azul y déjala en la canasta"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

def _load(module: str, file: str):
    spec = importlib.util.spec_from_file_location(module, Path(__file__).with_name(file))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


# Speech, language detection and replies come from demo 06 unchanged; the
# labels of the trained model come from its training script.
voice06 = _load("voice06", "06_voice_command.py")
training = _load("train_command_model", "train_command_model.py")
TRAINED_MODEL = training.OUTPUT / "best"

CONFIDENCE_MIN = 0.5

# The options of Jev's questions (the trained GLiNER model learnt the same ones
# from examples). Descriptions carry the words for each option in the three
# languages, so "gorria" or "otzara" are recognised too.
INTENTS = {
    "pick_and_place": "pick something up and say where to put it",
    "pick": "pick something up, without saying where to put it",
    "greeting": "a greeting or small talk, like hola, hello, kaixo, how are you",
    "other": "a question or request that is not about moving objects, like the time or the weather",
}
OBJECT_KINDS = {
    "ball": "a ball or sphere (bola, pelota, esfera, pilota)",
    "cube": "a cube or block (cubo, bloque, dado, kubo)",
    "any": "only a colour or a generic word (objeto, cosa, pieza, object, thing, objektu, gauza)",
    "other": "a specific named thing that is not a ball or a cube, like a fruit or a tool (plátano, pera, apple, sagar)",
}
PLACE_KINDS = {
    "basket": "a basket (cesto, cesta, canasta, saski, otzara)",
    "box": "a box or crate (caja, kutxa)",
    "table": "the table (mesa, mahai)",
    "left": "the left side (izquierda, ezker, ezkerrean)",
    "right": "the right side (derecha, eskuin, eskuinean)",
    "other": "another place that is not in this list, like a shelf (estantería, apala)",
}
COLORS = {
    "red": "red (rojo, roja, gorri, gorria)",
    "green": "green (verde, berde, berdea)",
    "blue": "blue (azul, urdin, urdina)",
    "yellow": "yellow (amarillo, amarilla, hori, horia)",
    "none": "no colour is named",
}


def object_word(phrase: str) -> str | None:
    """The name in an object phrase without colours or articles: "manzanas verdes" -> "manzanas"."""
    kept = [said for plain, said in voice06.words(phrase)
            if not any(voice06.lookup(plain, voice06.COLORS[lang], lang) for lang in voice06.LANGS)
            and plain not in voice06.ARTICLES["es"] | voice06.ARTICLES["en"]]
    return " ".join(kept) or None


def finish(command: dict, confidences: list[float]) -> dict:
    """Attach the weakest confidence, or refuse the command if it is too low."""
    confidence = min(confidences)
    if confidence < CONFIDENCE_MIN:
        return {"error": "unsure", "confidence": round(confidence, 2)}
    return {**command, "confidence": round(confidence, 2)}


def reply(command: dict, lang: str) -> str:
    if command.get("error") == "unsure":
        return {"es": "No estoy seguro de haberte entendido. ¿Me lo repites?",
                "en": "I'm not sure I understood. Could you say it again?",
                "eu": "Ez nago ziur ulertu dudan. Errepikatuko duzu?"}[lang]
    return voice06.reply(command, lang)


# --- GLiNER2 (open source, local) -----------------------------------------------

class GLiNER:
    """GLiNER2.5-multi fine-tuned by ``train_command_model.py``: one pass answers every question."""

    def __init__(self, model: str = str(TRAINED_MODEL)):
        import torch
        from gliner2 import AutoExtractor

        if not Path(model).exists():
            raise RuntimeError(f"no trained model at {model}: run  python .\\demos\\02_voice\\train_command_model.py")
        self.model = AutoExtractor.from_pretrained(model)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        schema = self.model.create_schema()
        for task, labels in training.LABELS.items():
            schema = schema.classification(task, labels)
        self.schema = schema.entities({"object_name": "the word naming the thing to pick up"})
        self.command("coge la bola roja")  # the first GPU call is slow: warm up now

    def command(self, text: str) -> dict:
        found = self.model.extract(text, self.schema, include_confidence=True)
        label = {task: found[task]["label"] for task in training.LABELS}
        sure = {task: found[task]["confidence"] for task in training.LABELS}

        if label["intent"] in ("greeting", "other") or label["object_kind"] == "none":
            error = "greeting" if label["intent"] == "greeting" else "no_object"
            return finish({"error": error}, [sure["intent"]])
        names = sorted(found["entities"].get("object_name") or [], key=lambda n: -n["confidence"])
        kind = label["object_kind"]
        thing = {"type": kind,
                 "color": None if label["object_color"] == "none" else label["object_color"],
                 "word": object_word(names[0]["text"]) if kind == "other" and names else None}
        if label["number"] == "several":
            thing["plural"] = True
        confidences = [sure["intent"], sure["object_kind"], sure["object_color"], sure["place_kind"]]
        if kind == "other" and thing["word"] is None:
            return finish({"error": "no_object"}, confidences)
        if kind != "other" and thing["color"] is None:
            return finish({"error": "no_color", "object": thing}, confidences)
        place = label["place_kind"]
        if place == "none":
            return finish({"action": "pick", "object": thing, "target": None}, confidences)
        if place == "other":
            return finish({"error": "no_target", "object": thing}, confidences)
        target = {"type": place, "color": None if label["place_color"] == "none" else label["place_color"],
                  "word": None}
        return finish({"action": "pick_and_place", "object": thing, "target": target},
                      confidences + [sure["place_color"]])


# --- Jev (TypeSafe, hosted) ----------------------------------------------------

@dataclass
class Choice:
    instructions: str
    criteria: dict[str, str]  # option -> description

    def to_json(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass
class Noul:
    instructions: str

    def to_json(self) -> dict:
        return {"type": "noul", "instructions": self.instructions}


@dataclass
class Answer:
    choice: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    noul: float | None = None  # probability of "yes" for a Noul


def candidate_words(text: str, limit: int = 6) -> list[str]:
    """Words of the sentence the rules do not know, offered to Jev as object names ("plátano")."""
    seen = []
    for plain, said in voice06.words(text):
        if len(plain) < 3 or said in seen or plain in voice06.ARTICLES["es"] | voice06.ARTICLES["en"]:
            continue
        if any(voice06.known(plain, lang) or plain in voice06.COMMON[lang] for lang in voice06.LANGS):
            continue
        seen.append(said)
    return seen[:limit]


def jev_questions(text: str) -> dict:
    objects = {k: v for k, v in OBJECT_KINDS.items() if k != "other"}
    return {
        "intent": Choice("What does the person ask the robot arm to do?", INTENTS),
        "object": Choice(
            "Which object does the person ask the robot to pick up? If they correct themselves, use the final one.",
            {**objects, **{word: f'the "{word}"' for word in candidate_words(text)},
             "none": "no object is mentioned"}),
        "object_color": Choice(
            "Which colour does the person say for the object to pick up? Answer none if they say no colour for "
            "it: do not guess from what the object usually looks like, and ignore the colour of the place.",
            COLORS),
        "target": Choice(
            "Does the person say where to put the object? Answer none if they only ask to pick it up.",
            {**PLACE_KINDS, "none": "the person does not say where to put it"}),
        "target_color": Choice(
            "Which colour does the person say for the place where the object goes? Answer none if they say no "
            "colour for that place.",
            COLORS),
        "plural": Noul("Does the person ask for several objects at once (a plural, such as 'the pears')?"),
    }


def jev_command(answers: dict[str, Answer]) -> dict:
    """Turn Jev's answers into the same command format as demo 06."""
    def picked(name):
        choice = answers[name].choice
        return None if choice in (None, "none") else choice

    confidences = [answers["intent"].confidence]
    if picked("intent") == "greeting":
        return finish({"error": "greeting"}, confidences)
    if picked("intent") == "other" or picked("object") is None:
        return finish({"error": "no_object"}, confidences)
    kind = picked("object")
    thing = {"type": kind if kind in ("ball", "cube", "any") else "other", "color": picked("object_color"),
             "word": None if kind in ("ball", "cube", "any") else kind}
    if (answers["plural"].noul or 0) > 0.5:
        thing["plural"] = True
    confidences += [answers["object"].confidence, answers["target"].confidence]
    place = picked("target")
    if thing["type"] != "other" and thing["color"] is None:
        return finish({"error": "no_color", "object": thing}, confidences)
    if place == "other":
        return finish({"error": "no_target", "object": thing}, confidences)
    if place is None:
        return finish({"action": "pick", "object": thing, "target": None}, confidences)
    target = {"type": place, "color": picked("target_color"), "word": None}
    return finish({"action": "pick_and_place", "object": thing, "target": target}, confidences)


class Jev:
    """TypeSafe's Jev over its HTTP API (https://docs.typesafe.ai/api)."""

    URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model: str = "jev-latest"):
        import httpx

        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("set TYPESAFE_API_KEY (get one at https://console.typesafe.ai/keys)")
        self.model = model
        self.http = httpx.Client(timeout=15.0, headers={"Authorization": f"Bearer {key}"})

    def ask(self, state: str, questions: dict) -> dict[str, Answer]:
        payload = {"state": {"spoken_command": state}, "model": self.model,
                   "questions": {name: q.to_json() for name, q in questions.items()}}
        for attempt in range(4):  # back off on 429 (rate limit) and 529 (overloaded)
            response = self.http.post(self.URL, json=payload)
            if response.status_code not in (429, 529):
                break
            time.sleep(0.5 * 2 ** attempt)
        response.raise_for_status()
        return {name: Answer(choice=a.get("choice"), probabilities=a.get("probabilities") or {},
                             confidence=a.get("confidence", 1.0), noul=a.get("noul"))
                for name, a in response.json()["answers"].items()}

    def command(self, text: str) -> dict:
        return jev_command(self.ask(text, jev_questions(text)))


CLASSIFIERS = {"gliner": GLiNER, "jev": Jev}


# --- main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=sorted(CLASSIFIERS), default="gliner", help="Who classifies the action")
    parser.add_argument("--model", help="Hugging Face id of the GLiNER2 model, or the Jev model name")
    parser.add_argument("--lang", choices=["auto", *voice06.LANGS], default="auto",
                        help="Language spoken (default: detect it)")
    parser.add_argument("--text", help="Use this sentence instead of the microphone")
    parser.add_argument("--no-speak", action="store_true", help="Do not play the spoken reply")
    parser.add_argument("--cpu", action="store_true", help="Keep speech recognition off the GPU")
    args = parser.parse_args()

    import numpy as np

    try:
        import truststore  # Norton re-signs HTTPS; see demo 06

        truststore.inject_into_ssl()
    except ImportError:
        pass

    print(f"Loading the '{args.classifier}' classifier (the first run downloads it)...", flush=True)
    classifier = CLASSIFIERS[args.classifier](*([args.model] if args.model else []))
    models = {}
    if args.text is None:
        models, device = voice06.load_models(args.lang, args.cpu)
        print(f"  speech recognition on the {device.upper()}", flush=True)
    voices = {}

    while True:
        if args.text is not None:
            text, t_stt = args.text, 0.0
            lang = voice06.guess_language(text) if args.lang == "auto" else args.lang
        else:
            if input("\n[Enter] to talk, q + Enter to quit: ").strip().lower() == "q":
                return 0
            audio = voice06.record_until_enter()
            t0 = time.perf_counter()
            lang, text = voice06.listen(audio, args.lang, models)
            t_stt = time.perf_counter() - t0
            if not text.strip():
                print("  (no speech heard)")
                continue

        t0 = time.perf_counter()
        command = classifier.command(text)
        t_classify = time.perf_counter() - t0

        answer = reply(command, lang)
        t0 = time.perf_counter()
        if lang not in voices:
            voices[lang] = voice06.load_voice(lang)
        chunks = list(voices[lang].synthesize(answer))
        speech = np.concatenate([c.audio_int16_array for c in chunks])
        t_tts = time.perf_counter() - t0

        print(f"  language: {lang}{' (detected)' if args.lang == 'auto' else ''}")
        print(f"  heard   : {text!r}")
        print(f"  command : {json.dumps(command, ensure_ascii=False)}")
        print(f"  reply   : {answer!r}")
        print(f"  time    : stt {t_stt * 1000:.0f} ms, {args.classifier} {t_classify * 1000:.0f} ms, "
              f"tts {t_tts * 1000:.0f} ms", flush=True)

        if not args.no_speak:
            import sounddevice as sd

            sd.play(speech, chunks[0].sample_rate)
            sd.wait()
        if args.text is not None:
            return 0 if "error" not in command else 2


if __name__ == "__main__":
    raise SystemExit(main())
