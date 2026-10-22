"""Voice command: speech -> text -> robot command -> spoken reply.

Say "Coge la bola roja y métela en el cesto verde" and get back

    {"action": "pick_and_place", "object": {"type": "ball", "color": "red", "word": "bola"},
     "target": {"type": "basket", "color": "green", "word": "cesto"}}

plus a spoken confirmation. Spanish, English and Basque, detected automatically.

* speech to text: faster-whisper ``small`` (Spanish, English) or a Basque
  fine-tune of it (``xezpeleta/whisper-small-eu-ct2``), on the GPU when it
  works (about 0.3 s per command), else on the CPU (about 2 s);
* text to command: a word list per language and a few rules, no AI model;
* text to speech: a Piper voice per language, on the CPU.

Models download from the Hugging Face Hub on first use (about 250 MB per
Whisper model and 60 MB per voice). The robot is not involved yet: the command
printed at the end is what a later demo hands to the arm.

Run:

    python .\\demos\\02_voice\\06_voice_command.py               # any of the three languages
    python .\\demos\\02_voice\\06_voice_command.py --lang eu     # only Basque (a bit faster)
    python .\\demos\\02_voice\\06_voice_command.py --text "put the red ball in the green basket"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata

SAMPLE_RATE = 16000  # what Whisper expects

STT_MODELS = {"es": "small", "en": "small", "eu": "xezpeleta/whisper-small-eu-ct2"}
VOICES = {
    "es": "es/es_ES/davefx/medium/es_ES-davefx-medium",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
    "eu": "eu/eu_ES/maider/medium/eu_ES-maider-medium",
}

# Words the parser understands. Spanish and Basque match on the start of a word,
# so "roj" catches rojo/roja and "saski" catches saskia/saskian/saskira.
# English matches whole words (a trailing plural "s" is dropped).
# "any" is a generic word ("el objeto rojo", "the red thing"): the colour alone
# picks the object. Any other noun ("el plátano") becomes type "other".
OBJECTS = {
    "es": {"bola": "ball", "pelota": "ball", "cubo": "cube", "dado": "cube",
           "objet": "any", "pieza": "any", "cosa": "any", "elemento": "any", "figura": "any"},
    "en": {"ball": "ball", "cube": "cube", "block": "cube",
           "object": "any", "thing": "any", "item": "any", "piece": "any", "one": "any"},
    "eu": {"bola": "ball", "pilota": "ball", "kubo": "cube",
           "objektu": "any", "gauza": "any", "pieza": "any"},
}
# Where the object goes. Spanish and Basque match these anywhere inside a word,
# because Whisper sometimes glues words together ("en el cesto" -> "el incesto").
PLACES = {
    "es": {"cesto": "basket", "cesta": "basket", "caja": "box", "mesa": "table",
           "izquierd": "left", "derech": "right"},
    "en": {"basket": "basket", "box": "box", "bin": "basket", "table": "table",
           "left": "left", "right": "right"},
    "eu": {"saski": "basket", "otarre": "basket", "kutxa": "box", "mahai": "table",
           "ezker": "left", "eskuin": "right"},
}
COLORS = {
    "es": {"roj": "red", "verd": "green", "azul": "blue", "amarill": "yellow"},
    "en": {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow"},
    # "hori" alone also means "that", so yellow only matches its inflected forms.
    "eu": {"gorri": "red", "berde": "green", "urdin": "blue",
           "horia": "yellow", "horian": "yellow", "horira": "yellow"},
}
GREETINGS = {
    "es": {"hola", "buenas", "buenos"},
    "en": {"hello", "hi", "hey"},
    "eu": {"kaixo", "egun", "arratsalde", "gabon"},
}
# Words that say "put it somewhere". If one appears but no known place does,
# the robot asks where instead of silently dropping the destination.
PLACE_WORDS = {
    "es": {"en", "dentro", "encima", "sobre"},
    "en": {"in", "into", "on", "onto", "inside"},
    "eu": {"sartu", "jarri", "utzi", "eraman", "gainean"},
}
# A new object name is the noun after an article ("coge el plátano"); Basque has
# no separate articles, so there it is the word next to the verb ("hartu platanoa").
ARTICLES = {"es": {"el", "la", "un", "una", "los", "las", "unos", "unas"}, "en": {"the", "a", "an", "some"}}
ES_PLURAL_ARTICLES = {"los", "las", "unos", "unas"}
EU_VERBS = {"hartu", "jaso", "har", "ekarri"}
# Small words that are never an object name.
FILLER = {
    "es": {"que", "por", "favor", "otro", "otra", "mismo", "misma"},
    "en": {"and", "please", "other", "same"},
    "eu": {"eta", "ere", "mesedez", "bat", "hau", "hori", "hura", "beste"},
}
# Frequent words, only used to guess the language of a sentence.
COMMON = {
    "es": {"coge", "coger", "cogeme", "pon", "poner", "ponlo", "ponla", "mete", "meter", "metela", "metelo",
           "deja", "dejar", "dejalo", "dejala", "y", "el", "la", "los", "las", "de", "del", "que", "me", "puedes",
           "por", "favor", "es", "esta", "estas", "tal", "como", "hora", "gracias"},
    "en": {"pick", "up", "put", "grab", "take", "drop", "leave", "move", "it", "them", "the", "and", "to",
           "please", "can", "you", "what", "how", "are", "is", "thanks", "there"},
    "eu": {"hartu", "sartu", "jarri", "utzi", "eraman", "eman", "eta", "zer", "moduz", "mesedez", "da", "dut",
           "duzu", "nahi", "bat", "eskerrik", "asko", "on", "gero", "arte"},
}
LANGS = ("es", "en", "eu")  # ties go to the first one


def words(text: str) -> list[tuple[str, str]]:
    """Split into (plain, as_said) pairs: plain drops accents ("métela" -> "metela")."""
    pairs = []
    for said in re.findall(r"[^\W\d_]+", text.lower()):
        plain = unicodedata.normalize("NFD", said)
        pairs.append(("".join(c for c in plain if unicodedata.category(c) != "Mn"), said))
    return pairs


def lookup(word: str, table: dict[str, str], lang: str) -> str | None:
    if lang == "en":
        return table.get(word) or table.get(word.rstrip("s"))
    if table is COLORS["eu"] and word.startswith("hori"):
        return table.get(word)
    if table is PLACES[lang]:
        return next((value for stem, value in table.items() if stem in word), None)
    return next((value for stem, value in table.items() if word.startswith(stem)), None)


def known(word: str, lang: str) -> bool:
    return any(lookup(word, table[lang], lang) for table in (OBJECTS, PLACES, COLORS)) or \
        word in GREETINGS[lang] or word in PLACE_WORDS[lang] or word in EU_VERBS or word in FILLER[lang]


def parse(text: str, lang: str) -> dict:
    """Turn a sentence into {action, object, target}, or {error: ...}."""
    obj, target = None, None
    last = None  # the object or place the next colour may belong to
    pending_color = None  # English puts the colour before the noun
    sentence = words(text)
    plain = [word for word, _ in sentence]

    def new(kind: str, said: str | None, plural: bool = False) -> dict:
        nonlocal pending_color
        thing = {"type": kind, "color": pending_color, "word": said}
        if plural:
            thing["plural"] = True  # "las peras", "the pears": all of them
        pending_color = None
        return thing

    for i, (word, said) in enumerate(sentence):
        before = plain[i - 1] if i > 0 else ""
        before2 = plain[i - 2] if i > 1 else ""
        kind = lookup(word, OBJECTS[lang], lang)
        place = lookup(word, PLACES[lang], lang)
        color = lookup(word, COLORS[lang], lang)
        plural = (before in ES_PLURAL_ARTICLES if lang == "es"
                  else lang == "en" and word.endswith("s") and not word.endswith("ss"))
        if kind and obj is None:
            obj = last = new(kind, said, plural)
        elif place and target is None:
            target = last = new(place, said)
        elif color:
            if lang != "en" and last is not None and last["color"] is None:
                last["color"] = color  # "la bola roja", "bola gorria"
            elif lang != "en" and obj is None:
                obj = last = new("any", None, plural)  # "coge el rojo", "coge los rojos"
                obj["color"] = color
            else:
                if pending_color and obj is None:
                    obj = last = new("any", None)  # "the red and ..."
                pending_color = color  # "the red ball"
        elif obj is None and target is None and len(word) > 2 and not known(word, lang):
            # "coge el plátano", but not "en la estantería" (that is a place)
            after_article = before in ARTICLES.get(lang, ()) and before2 not in PLACE_WORDS[lang]
            if after_article or (lang == "eu" and before in EU_VERBS):
                obj = last = new("other", said, plural)
    if obj is None and lang == "eu":  # "platanoa hartu": the word before the verb
        for i, word in enumerate(plain[1:], start=1):
            if word in EU_VERBS and len(plain[i - 1]) > 2 and not known(plain[i - 1], lang):
                obj = {"type": "other", "color": None, "word": sentence[i - 1][1]}
                break
    if obj is None and pending_color:
        obj = new("any", None)  # "pick up the red"
    if obj is None:
        greeted = any(word in GREETINGS[lang] for word in plain)
        return {"error": "greeting" if greeted else "no_object"}
    if obj["color"] is None and obj["type"] != "other":
        return {"error": "no_color", "object": obj}  # which ball? (there are several)
    if target is None and any(word in PLACE_WORDS[lang] for word in plain):
        return {"error": "no_target", "object": obj}  # "déjalo en la estantería"
    action = "pick_and_place" if target else "pick"
    return {"action": action, "object": obj, "target": target}


def language_score(text: str, lang: str) -> float:
    """How much a sentence looks like `lang`: share of familiar words, +1 if it makes a command."""
    plain = [word for word, _ in words(text)]
    if not plain:
        return 0.0
    familiar = sum(word in COMMON[lang] or known(word, lang) for word in plain) / len(plain)
    return familiar + (parse(text, lang).get("error") in (None, "greeting"))


def guess_language(text: str) -> str:
    return max(LANGS, key=lambda lang: language_score(text, lang))


# --- spoken replies ---------------------------------------------------------

ES_NOUN = {"ball": "bola", "cube": "cubo", "any": "objeto", "basket": "cesto", "box": "caja", "table": "mesa"}
ES_COLOR = {"red": ("rojo", "roja"), "green": ("verde", "verde"),
            "blue": ("azul", "azul"), "yellow": ("amarillo", "amarilla")}
EN_NOUN = {"ball": "ball", "cube": "cube", "any": "object", "basket": "basket", "box": "box", "table": "table"}
EU_NOUN = {"ball": "bola", "cube": "kubo", "any": "objektu", "basket": "saski", "box": "kutxa", "table": "mahai"}
# Basque colour as subject ("bola gorria") and as "in the ..." ("saski berdean").
EU_COLOR = {"red": ("gorria", "gorrian"), "green": ("berdea", "berdean"),
            "blue": ("urdina", "urdinean"), "yellow": ("horia", "horian")}
EU_BARE = {"ball": ("bola", None), "cube": ("kuboa", None), "any": ("objektua", None),
           "basket": ("saskia", "saskian"), "box": ("kutxa", "kutxan"),
           "table": ("mahaia", "mahai gainean")}
SIDES = {"es": {"left": "a la izquierda", "right": "a la derecha"},
         "en": {"left": "on the left", "right": "on the right"},
         "eu": {"left": "ezkerrean", "right": "eskuinean"}}

# Given to Whisper as context, so it expects our words ("en el cesto", not "el incesto").
STT_HINT = {
    "es": "Coge la bola roja y métela en el cesto verde. Pon el cubo azul en la caja amarilla. "
          "Deja el objeto en la mesa. Déjalo a la izquierda.",
    "en": "Pick up the red ball and put it in the green basket. Put the blue cube in the yellow box. "
          "Leave the object on the table. Put it on the left.",
    "eu": "Hartu bola gorria eta sartu saski berdean. Jarri kubo urdina kutxa horian. "
          "Utzi objektua mahai gainean. Utzi ezkerrean.",
}

EXAMPLE = {"es": "coge la bola roja y métela en el cesto verde",
           "en": "put the red ball in the green basket",
           "eu": "hartu bola gorria eta sartu saski berdean"}


def es_plural(word: str) -> str:
    return word + ("s" if word[-1] in "aeiouáéó" else "es")  # objetos, azules


def es_noun(thing: dict) -> tuple[str, bool]:
    """The noun as the user said it ("cesta", "peras"), and whether it is feminine."""
    said, plural = thing["word"], thing.get("plural", False)
    if said and (plural or not said.endswith("s")):
        noun = said
    else:
        noun = ES_NOUN.get(thing["type"], said or "objeto")
        noun = es_plural(noun) if plural else noun
    singular = noun[:-1] if plural and noun.endswith("s") else noun
    return noun, singular.endswith("a")  # la bola, las peras, el cesto, el plátano


def es_phrase(thing: dict) -> str:
    noun, feminine = es_noun(thing)
    plural = thing.get("plural", False)
    article = {(False, False): "el", (True, False): "la", (False, True): "los", (True, True): "las"}
    text = f"{article[feminine, plural]} {noun}"
    if not thing["color"]:
        return text
    color = ES_COLOR[thing["color"]][feminine]
    return f"{text} {es_plural(color) if plural else color}"


def es_pronoun(thing: dict) -> str:
    feminine = es_noun(thing)[1]
    return ("la" if feminine else "lo") + ("s" if thing.get("plural") else "")


def en_phrase(thing: dict) -> str:
    if thing["type"] == "other":
        noun = thing["word"]
    else:
        noun = EN_NOUN[thing["type"]] + ("s" if thing.get("plural") else "")
    return " ".join(filter(None, ["the", thing["color"], noun]))


def eu_phrase(thing: dict, inside: bool = False) -> str:
    if thing["type"] == "other":
        return f"{thing['word']} {EU_COLOR[thing['color']][0]}" if thing["color"] else thing["word"]
    if not thing["color"]:
        return EU_BARE[thing["type"]][inside]
    if thing["type"] == "table" and inside:
        return f"mahai {EU_COLOR[thing['color']][0]}ren gainean"  # "mahai gorriaren gainean"
    return f"{EU_NOUN[thing['type']]} {EU_COLOR[thing['color']][inside]}"


def reply(command: dict, lang: str) -> str:
    error = command.get("error")
    if error == "greeting":
        return {"es": f"¡Hola! Dime qué quieres que coja. Por ejemplo: {EXAMPLE['es']}.",
                "en": f"Hi! Tell me what to pick up. For example: {EXAMPLE['en']}.",
                "eu": f"Kaixo! Esan zer hartu behar dudan. Adibidez: {EXAMPLE['eu']}."}[lang]
    if error == "no_object":
        return {"es": f"Perdona, no he entendido qué quieres que coja. Prueba: {EXAMPLE['es']}.",
                "en": f"Sorry, I didn't catch what to pick up. Try: {EXAMPLE['en']}.",
                "eu": f"Barkatu, ez dut ulertu zer hartu behar dudan. Saiatu: {EXAMPLE['eu']}."}[lang]
    obj = command["object"]
    plural = obj.get("plural", False)
    if error == "no_color":
        return {"es": f"¿De qué color {'son' if plural else 'es'} {es_phrase(obj)}?",
                "en": f"Which colour {'are' if plural else 'is'} {en_phrase(obj)}?",
                "eu": f"Zein koloretakoa da {eu_phrase(obj)}?"}[lang]
    if error == "no_target":
        return {"es": f"¿Dónde dejo {es_phrase(obj)}? Por ejemplo: en el cesto, en la mesa o a la izquierda.",
                "en": f"Where should I put {en_phrase(obj)}? For example: in the basket, on the table or on the left.",
                "eu": f"Non utziko dut {eu_phrase(obj)}? Adibidez: saskian, mahai gainean edo ezkerrean."}[lang]

    target = command["target"]
    kind = target["type"] if target else None
    if lang == "es":
        text = f"Vale, cojo {es_phrase(obj)}"
        if target:
            pronoun = es_pronoun(obj)
            if kind in SIDES["es"]:
                text += f" y {pronoun} dejo {SIDES['es'][kind]}"
            else:
                text += f" y {pronoun} {'dejo' if kind == 'table' else 'meto'} en {es_phrase(target)}"
        return text + "."
    if lang == "en":
        text = f"OK, picking up {en_phrase(obj)}"
        if target:
            where = SIDES["en"].get(kind) or f"{'on' if kind == 'table' else 'in'} {en_phrase(target)}"
            text += f" and putting {'them' if plural else 'it'} {where}"
        return text + "."
    if target:
        where = SIDES["eu"].get(kind) or eu_phrase(target, inside=True)
        verb = "sartuko" if kind in ("basket", "box") else "utziko"
        return f"Ados, {eu_phrase(obj)} hartu eta {where} {verb} dut."
    return f"Ados, {eu_phrase(obj)} hartuko dut."


# --- audio -----------------------------------------------------------------

def record_until_enter():
    """Record from the default microphone until Enter is pressed."""
    import numpy as np
    import sounddevice as sd

    chunks = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("  recording... press Enter to stop ")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype="float32")


def transcribe(model, audio, lang: str) -> str:
    segments, _ = model.transcribe(audio, language=lang, beam_size=1, vad_filter=True,
                                   initial_prompt=STT_HINT[lang])
    text = " ".join(s.text.strip() for s in segments)
    # On short or unclear audio Whisper sometimes repeats a sentence of the hint as if it had been
    # said ("Eh, recoge todo" -> "Deja el objeto en el cesto verde. Recoge todo."): drop those.
    for sentence in re.split(r"(?<=[.!?])\s+", STT_HINT[lang]):
        text = text.replace(sentence, "")
    return " ".join(text.split())


def listen(audio, lang: str, models: dict) -> tuple[str | None, str]:
    """Speech to (language, text). With lang "auto" the language is detected.

    Whisper's own language detection tells English from Spanish reliably but
    mistakes Basque for Spanish most of the time. So three jobs run at once:
    Whisper's language detection, a Spanish transcript and a Basque transcript.
    If Whisper heard English, the audio is transcribed as English; otherwise the
    transcript that reads more like its language (Spanish or Basque) wins.
    """
    from concurrent.futures import ThreadPoolExecutor

    if len(audio) < SAMPLE_RATE // 4:  # under 0.25 s: nothing was said
        return None, ""
    if lang != "auto":
        return lang, transcribe(models[STT_MODELS[lang]], audio, lang)
    small, basque_model = models[STT_MODELS["es"]], models[STT_MODELS["eu"]]
    with ThreadPoolExecutor(3) as pool:
        detected = pool.submit(small.detect_language, audio, vad_filter=True)
        spanish = pool.submit(transcribe, small, audio, "es")
        basque = pool.submit(transcribe, basque_model, audio, "eu")
        probs = dict(detected.result()[2])
        if probs.get("en", 0) > max(probs.get("es", 0), probs.get("eu", 0)):
            return "en", transcribe(small, audio, "en")
        spanish, basque = spanish.result(), basque.result()
    if language_score(basque, "eu") > language_score(spanish, "es"):
        return "eu", basque
    return "es", spanish


def load_models(lang: str, cpu_only: bool) -> tuple[dict, str]:
    """Load the Whisper models for `lang` ("auto" loads both) on the GPU if it works, else the CPU."""
    import numpy as np
    from faster_whisper import WhisperModel

    names = set(STT_MODELS.values()) if lang == "auto" else {STT_MODELS[lang]}
    options = [] if cpu_only else [("cuda", "int8_float16")]
    for device, compute in options + [("cpu", "int8")]:
        try:
            models = {}
            for name in names:
                extra = {"num_workers": 2} if lang == "auto" and name == STT_MODELS["es"] else {}
                if device == "cpu" and lang == "auto":
                    # three jobs run side by side (see listen): a third of the cores each
                    extra["cpu_threads"] = max(1, (os.cpu_count() or 3) // 3)
                models[name] = WhisperModel(name, device=device, compute_type=compute, **extra)
            # The first GPU call compiles kernels and takes seconds: do it now, not on the first command.
            # This also proves CUDA really works (missing DLLs only fail on first use).
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            for model in models.values() if device == "cuda" else ():
                model.detect_language(silence)
                list(model.transcribe(silence, language="es", vad_filter=False)[0])
            return models, device
        except Exception as error:  # no CUDA, missing cuDNN/cuBLAS DLLs, ...
            print(f"  {device} not usable ({type(error).__name__}: {error}), trying the next option")
    raise RuntimeError("could not load Whisper")


def load_voice(lang: str):
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    path = hf_hub_download("rhasspy/piper-voices", VOICES[lang] + ".onnx")
    hf_hub_download("rhasspy/piper-voices", VOICES[lang] + ".onnx.json")
    return PiperVoice.load(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["auto", *LANGS], default="auto",
                        help="Language spoken (default: detect it)")
    parser.add_argument("--text", help="Use this sentence instead of the microphone")
    parser.add_argument("--no-speak", action="store_true", help="Do not play the spoken reply")
    parser.add_argument("--cpu", action="store_true",
                        help="Keep speech recognition off the GPU (slower: about 1-2 s per command)")
    args = parser.parse_args()

    import numpy as np

    try:
        # Norton's Web Shield re-signs HTTPS with its own root certificate, which
        # Windows trusts but Python's certifi list does not. truststore makes
        # Python use the Windows store, so the model downloads work.
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    print(f"Loading models for '{args.lang}' (the first run downloads them)...", flush=True)
    voices = {}  # loaded on first use, one per language
    models = {}
    if args.text is None:
        models, device = load_models(args.lang, args.cpu)
        print(f"  speech recognition on the {device.upper()}", flush=True)

    while True:
        if args.text is not None:
            t0 = time.perf_counter()
            text = args.text
            lang = guess_language(text) if args.lang == "auto" else args.lang
            t_stt = time.perf_counter() - t0
        else:
            if input("\n[Enter] to talk, q + Enter to quit: ").strip().lower() == "q":
                return 0
            audio = record_until_enter()
            t0 = time.perf_counter()
            lang, text = listen(audio, args.lang, models)
            t_stt = time.perf_counter() - t0
            if not text.strip():
                print("  (no speech heard)")
                continue

        t0 = time.perf_counter()
        command = parse(text, lang)
        t_parse = time.perf_counter() - t0

        answer = reply(command, lang)
        t0 = time.perf_counter()
        if lang not in voices:
            voices[lang] = load_voice(lang)
        chunks = list(voices[lang].synthesize(answer))
        speech = np.concatenate([c.audio_int16_array for c in chunks])
        t_tts = time.perf_counter() - t0

        print(f"  language: {lang}{' (detected)' if args.lang == 'auto' else ''}")
        print(f"  heard   : {text!r}")
        print(f"  command : {json.dumps(command, ensure_ascii=False)}")
        print(f"  reply   : {answer!r}")
        print(f"  time    : stt {t_stt * 1000:.0f} ms, parse {t_parse * 1000:.2f} ms, "
              f"tts {t_tts * 1000:.0f} ms", flush=True)

        if not args.no_speak:
            import sounddevice as sd

            sd.play(speech, chunks[0].sample_rate)
            sd.wait()
        if args.text is not None:
            return 0 if "error" not in command else 2


if __name__ == "__main__":
    raise SystemExit(main())
