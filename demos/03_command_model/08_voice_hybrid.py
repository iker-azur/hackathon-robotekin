"""Demo 08: voice commands understood by two fine-tuned "System One" models together.

    microphone -> speech recognition (Whisper, language detected: es / en / eu)
               -> Laya (fine-tuned)   : intent, place, place colour, sort criterion
               -> GLiNER2 (fine-tuned): the items: which objects, colour, quantity, position
               -> command (JSON for the robot) + spoken reply (Piper)

Why two models: on the hand-written test sentences the fine-tuned Laya is the
best at the sentence-level decisions (99% intent, 0 non-commands read as an
action), but it can only choose between options. GLiNER2 can also pull the
objects out of the sentence ("una bola roja y dos azules" -> two items). Both
are encoders that score options in one pass (no text generation), ~40 + ~60 ms.

Safety: "stop" is answered first; an action whose weakest decision is under
CONFIDENCE_MIN is not executed (the robot asks to repeat); a missing object,
colour or destination is asked for instead of guessed.

Needs the trained models:  models/command-model/best (train.py)  and
models/laya-commands (laya/laya_train.py).

Run:

    python .\\demos\\03_command_model\\08_voice_hybrid.py                 # microphone, any of the 3 languages
    python .\\demos\\03_command_model\\08_voice_hybrid.py --text "coge una bola roja y dos azules y mételas en la caja"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "laya"))
from command_model import VOCAB, CommandModel, plain  # noqa: E402
from laya_questions import QUESTIONS  # noqa: E402

LAYA_MODEL = HERE / "models" / "laya-commands"
CONFIDENCE_MIN = 0.5
ACTIONS = {"pick_and_place", "pick", "give", "sort", "clear", "sequence"}
NEEDS_ITEMS = {"pick_and_place", "pick", "give", "sequence"}


def language_words() -> dict[str, set[str]]:
    """Every word each language uses in the training data: the generator's phrase banks + vocabulary."""
    import re

    from generate_dataset import GRAMMARS

    def collect(value, into):
        if isinstance(value, str):
            into.update(plain(w) for w in re.findall(r"[^\W\d_]+", value))
        elif isinstance(value, (list, tuple)):
            for v in value:
                collect(v, into)
        elif isinstance(value, dict):
            for v in value.values():
                collect(v, into)

    words = {}
    for lang, grammar in GRAMMARS.items():
        found = set()
        collect([v for k, v in vars(type(grammar)).items() if k.isupper()], found)
        for section in ("colors", "numbers", "positions"):
            collect([entry.get(lang) for entry in VOCAB[section].values()], found)
        for section in ("objects", "places"):
            for split in VOCAB[section].values():
                collect([kinds.get(lang) for kinds in split.values()], found)
        words[lang] = found
    # a word shared by two languages ("bola", "pieza") says nothing about which one it is
    return {lang: found - set().union(*(w for other, w in words.items() if other != lang))
            for lang, found in words.items()}


def load_voice_demo():
    """Speech recognition, language detection and Piper voices come from demo 06.

    Its language scoring only knows demo 06's small word list, so it is widened here
    with every word of the command dataset: "sailkatu piezak koloreka" is then Basque.
    """
    spec = importlib.util.spec_from_file_location("voice06", HERE.parent / "02_voice" / "06_voice_command.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    known = language_words()
    old_score = module.language_score

    def language_score(text: str, lang: str) -> float:
        words = [plain(w) for w, _ in module.words(text)]
        share = sum(w in known[lang] for w in words) / len(words) if words else 0.0
        return old_score(text, lang) + 2 * share

    module.language_score = language_score  # listen() and guess_language() look it up at call time
    return module


# --- understanding --------------------------------------------------------------

class HybridModel:
    def __init__(self):
        import laya

        if not LAYA_MODEL.exists():
            raise FileNotFoundError(f"no Laya model at {LAYA_MODEL}: run laya/laya_prepare.py and laya/laya_train.py")
        self.laya = laya.load(str(LAYA_MODEL))
        self.gliner = CommandModel()
        self.understand("coge la bola roja")  # warm up both

    def understand(self, text: str) -> tuple[dict, dict]:
        start = time.perf_counter()
        answers = self.laya.predict(text, QUESTIONS)["answers"]
        middle = time.perf_counter()
        decisions = {q: (a["choice"], a.get("answer_confidence", a.get("confidence", 1.0))) for q, a in answers.items()}
        command = self.gliner.parse(text, decisions=decisions)
        end = time.perf_counter()
        return command, {"laya": (middle - start) * 1000, "gliner": (end - middle) * 1000}


def problem(command: dict) -> str | None:
    """What stops the robot from acting on this command, if anything."""
    intent = command["intent"]
    if intent not in ACTIONS:
        return None
    if command["confidence"] < CONFIDENCE_MIN:
        return "unsure"
    if intent in NEEDS_ITEMS and not command["items"]:
        return "no_object"
    place = command["place"]
    if (place and place["kind"] == "other") or (intent == "pick_and_place" and not place):
        return "no_target"
    for item in command["items"]:
        if item["kind"] in ("ball", "cube", "any") and not item["color"] and not item["position"] \
                and item["quantity"] != "all":
            return "no_color"  # which one? there are several balls
    return None


# --- saying it back ----------------------------------------------------------------

def several(item: dict) -> bool:
    return item["quantity"] == "all" or (isinstance(item["quantity"], int) and item["quantity"] > 1)


def color_word(color: str | None, lang: str, form: int | str = 0) -> str:
    if not color:
        return ""
    forms = VOCAB["colors"][color][lang]
    return forms[form] if lang == "eu" else forms[form if lang == "es" else 0]


def es_item(item: dict) -> str:
    name = item["name"] or {"ball": "bola", "cube": "cubo", "any": "pieza", "other": "objeto"}[item["kind"]]
    plural = several(item)
    if plural and not name.endswith("s"):
        name += "s" if name[-1] in "aeiouáéó" else "es"
    base = name[:-1] if plural and name.endswith("s") else name
    feminine = plain(base).endswith("a")
    color = color_word(item["color"], "es", (2 if plural else 0) + (1 if feminine else 0))
    if item["quantity"] == "all":
        head = "todas las" if feminine else "todos los"
    elif plural:
        head = VOCAB["numbers"][item["quantity"]]["es"][0]
    else:
        head = "la" if feminine else "el"
    text = " ".join(filter(None, [head, name, color]))
    position = item["position"]
    if position in ("left", "right"):
        text += " de la " + ("izquierda" if position == "left" else "derecha")
    elif position in ("nearest", "farthest"):
        text += " más " + ("cercan" if position == "nearest" else "lejan") + ("a" if feminine else "o")
    elif position == "front":
        text += " de delante"
    elif position == "back":
        text += " del fondo"
    return text


def en_item(item: dict) -> str:
    name = item["name"] or {"ball": "ball", "cube": "cube", "any": "piece", "other": "object"}[item["kind"]]
    if several(item) and not name.endswith("s"):
        name += "s"
    head = "all the" if item["quantity"] == "all" else (
        VOCAB["numbers"][item["quantity"]]["en"][0] if several(item) else "the")
    text = " ".join(filter(None, [head, item["color"] or "", name]))
    if item["position"] in ("left", "right", "front", "back"):
        text += f" {'at' if item['position'] in ('front', 'back') else 'on'} the {item['position']}"
    elif item["position"]:
        text = text.replace("the ", f"the {item['position']} ", 1)
    return text


def eu_item(item: dict) -> str:
    name = item["name"] or {"ball": "bola", "cube": "kubo", "any": "pieza", "other": "objektu"}[item["kind"]]
    quantity = item["quantity"]
    if quantity == "all":
        text = " ".join(filter(None, [name, color_word(item["color"], "eu", "bare"), "guztiak"]))
    elif several(item):
        text = " ".join(filter(None, [VOCAB["numbers"][quantity]["eu"][0], name, color_word(item["color"], "eu", "bare")]))
    else:
        text = " ".join(filter(None, [name, color_word(item["color"], "eu", "sg")]))
    if item["position"]:
        text = f"{VOCAB['positions'][item['position']]['eu'][0]} {text}"
    return text


def place_phrase(place: dict | None, lang: str) -> str:
    if not place:
        return ""
    kind, color = place["kind"], place["color"]
    if lang == "es":
        if kind in ("left", "right"):
            return "a la " + ("izquierda" if kind == "left" else "derecha")
        noun, feminine = {"basket": ("cesto", False), "box": ("caja", True), "tray": ("bandeja", True),
                          "table": ("mesa", True)}[kind]
        prep = "encima de" if kind == "table" else "en"
        return f"{prep} {'la' if feminine else 'el'} {noun}" + (f" {color_word(color, 'es', 1 if feminine else 0)}" if color else "")
    if lang == "en":
        if kind in ("left", "right"):
            return f"on the {kind}"
        return f"{'on' if kind == 'table' else 'in'} the " + " ".join(filter(None, [color, kind]))
    if kind in ("left", "right"):
        return "ezkerrean" if kind == "left" else "eskuinean"
    if kind == "table":
        return f"mahai {color_word(color, 'eu', 'of')} gainean" if color else "mahai gainean"
    stem, inside = {"basket": ("saski", "saskian"), "box": ("kutxa", "kutxan"), "tray": ("erretilu", "erretiluan")}[kind]
    return f"{stem} {color_word(color, 'eu', 'in')}" if color else inside


def listing(items: list, lang: str) -> str:
    phrases = [{"es": es_item, "en": en_item, "eu": eu_item}[lang](item) for item in items]
    if len(phrases) < 2:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + {"es": " y ", "en": " and ", "eu": " eta "}[lang] + phrases[-1]


ASK = {
    "unsure": {"es": "No estoy seguro de haberte entendido. ¿Me lo repites?",
               "en": "I'm not sure I understood. Could you say it again?",
               "eu": "Ez nago ziur ulertu dudan. Errepikatuko duzu?"},
    "no_object": {"es": "¿Qué quieres que coja?", "en": "What should I pick up?", "eu": "Zer hartu behar dut?"},
    "no_target": {"es": "¿Dónde lo dejo? Puedo usar el cesto, la caja, la bandeja o la mesa.",
                  "en": "Where should I put it? I can use the basket, the box, the tray or the table.",
                  "eu": "Non utziko dut? Saskia, kutxa, erretilua edo mahaia erabil ditzaket."},
}
FIXED = {
    "stop": {"es": "Me paro.", "en": "Stopping.", "eu": "Gelditzen naiz."},
    "repeat": {"es": "Vale, lo repito.", "en": "OK, doing it again.", "eu": "Ados, berriro egingo dut."},
    "cancel": {"es": "Vale, lo cancelo.", "en": "OK, cancelled.", "eu": "Ados, bertan behera utzi dut."},
    "answer_yes": {"es": "Perfecto.", "en": "Great.", "eu": "Primeran."},
    "answer_no": {"es": "Vale, entonces no.", "en": "OK, not that then.", "eu": "Ados, orduan ez."},
    "question": {"es": "Todavía no puedo ver la mesa: la cámara llegará en la siguiente demo.",
                 "en": "I can't see the table yet: the camera comes in the next demo.",
                 "eu": "Oraindik ezin dut mahaia ikusi: kamera hurrengo demoan etorriko da."},
    "greeting": {"es": "¡Hola! Dime qué hago, por ejemplo: coge la bola roja y métela en la caja azul.",
                 "en": "Hi! Tell me what to do, for example: put the red ball in the blue box.",
                 "eu": "Kaixo! Esan zer egin, adibidez: hartu bola gorria eta sartu kutxa urdinean."},
    "off_topic": {"es": "Eso no lo sé hacer. Puedo coger, darte, ordenar y colocar objetos.",
                  "en": "I can't help with that. I can pick up, hand over, sort and place objects.",
                  "eu": "Hori ez dakit egiten. Objektuak hartu, eman, sailkatu eta jar ditzaket."},
    "not_doable": {"es": "Eso no lo puedo hacer.", "en": "I can't do that.", "eu": "Hori ezin dut egin."},
}


def reply(command: dict, issue: str | None, lang: str) -> str:
    intent = command["intent"]
    if intent in FIXED:
        return FIXED[intent][lang]
    if issue == "no_color":
        item = next(i for i in command["items"] if not i["color"])
        phrase = listing([item], lang)
        return {"es": f"¿De qué color {'son' if several(item) else 'es'} {phrase}?",
                "en": f"Which colour {'are' if several(item) else 'is'} {phrase}?",
                "eu": f"Zein koloretakoa da {phrase}?"}[lang]
    if issue:
        return ASK[issue][lang]
    things, where = listing(command["items"], lang), place_phrase(command["place"], lang)
    plural = len(command["items"]) > 1 or any(several(i) for i in command["items"])
    if intent == "answer_value":
        return {"es": "Entendido: ", "en": "Got it: ", "eu": "Ulertuta: "}[lang] + " ".join(filter(None, [things, where])) + "."
    if intent == "sort":
        by = command["sort_by"] if command["sort_by"] != "none" else "color"
        text = {"es": f"Vale, separo las piezas por {'colores' if by == 'color' else 'tipo'}",
                "en": f"OK, sorting the pieces by {'colour' if by == 'color' else 'type'}",
                "eu": f"Ados, piezak {'koloreka' if by == 'color' else 'motaka'} sailkatuko ditut"}[lang]
        return text + (f" ({where})" if where else "") + "."
    if intent == "clear":
        text = {"es": "Vale, recojo la mesa", "en": "OK, clearing the table", "eu": "Ados, mahaia jasoko dut"}[lang]
        return text + (f" y lo dejo todo {where}" if where and lang == "es" else
                       f", everything goes {where}" if where and lang == "en" else
                       f", dena {where}" if where else "") + "."
    if intent == "give":
        return {"es": f"Ahora te paso {things}.", "en": f"Bringing you {things}.",
                "eu": f"Ados, {things} ekarriko {'dizkizut' if plural else 'dizut'}."}[lang]
    if where:  # pick_and_place, or a sequence with a destination
        return {"es": f"Vale, pongo {things} {where}.", "en": f"OK, putting {things} {where}.",
                "eu": f"Ados, {things} {where} jarriko {'ditut' if plural else 'dut'}."}[lang]
    return {"es": f"Vale, cojo {things}.", "en": f"OK, picking up {things}.",
            "eu": f"Ados, {things} hartuko {'ditut' if plural else 'dut'}."}[lang]


# --- main ------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["auto", "es", "en", "eu"], default="auto",
                        help="Language spoken (default: detect it)")
    parser.add_argument("--text", action="append", help="Use this sentence instead of the microphone (repeatable)")
    parser.add_argument("--no-speak", action="store_true", help="Do not play the spoken reply")
    parser.add_argument("--cpu", action="store_true", help="Keep speech recognition off the GPU")
    args = parser.parse_args()

    import numpy as np

    try:
        import truststore  # Norton re-signs HTTPS; see demos/02_voice

        truststore.inject_into_ssl()
    except ImportError:
        pass

    voice06 = load_voice_demo()
    print("Loading Laya + GLiNER2...", flush=True)
    model = HybridModel()
    models = {}
    if not args.text:
        models, device = voice06.load_models(args.lang, args.cpu)
        print(f"  speech recognition on the {device.upper()}", flush=True)
    voices = {}
    queue = list(args.text or [])

    while True:
        if args.text:
            if not queue:
                return 0
            text, t_stt = queue.pop(0), 0.0
            lang = voice06.guess_language(text) if args.lang == "auto" else args.lang
        else:
            if input("\n[Enter] to talk, q + Enter to quit: ").strip().lower() == "q":
                return 0
            audio = voice06.record_until_enter()
            t0 = time.perf_counter()
            lang, text = voice06.listen(audio, args.lang, models)
            t_stt = (time.perf_counter() - t0) * 1000
            if not text.strip():
                print("  (no speech heard)")
                continue

        command, times = model.understand(text)
        issue = problem(command)
        command["status"] = issue or ("ok" if command["intent"] in ACTIONS else "no_action")
        answer = reply(command, issue, lang)

        t0 = time.perf_counter()
        if lang not in voices:
            voices[lang] = voice06.load_voice(lang)
        chunks = list(voices[lang].synthesize(answer))
        speech = np.concatenate([c.audio_int16_array for c in chunks])
        t_tts = (time.perf_counter() - t0) * 1000

        print(f"\n  language: {lang}{' (detected)' if args.lang == 'auto' else ''}")
        print(f"  heard   : {text!r}")
        print(f"  command : {json.dumps(command, ensure_ascii=False)}")
        print(f"  reply   : {answer!r}")
        print(f"  time    : stt {t_stt:.0f} ms, laya {times['laya']:.0f} ms, gliner {times['gliner']:.0f} ms, "
              f"tts {t_tts:.0f} ms", flush=True)
        if not args.no_speak:
            import sounddevice as sd

            sd.play(speech, chunks[0].sample_rate)
            sd.wait()


if __name__ == "__main__":
    raise SystemExit(main())
