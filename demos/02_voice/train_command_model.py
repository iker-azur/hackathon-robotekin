"""Train the command classifier used by ``07_action_classifier.py``.

Jev-style models are trained for the decisions they make, so this script does
the same for ours: it writes a few thousand synthetic commands in Spanish,
English and Basque from templates, each labelled with the answers the
classifier must give, and fine-tunes Fastino's GLiNER2.5-multi on them.

One pass of the trained model answers, for a whole sentence at once:

    intent        pick_and_place / pick / greeting / other
    object_kind   ball / cube / any / other / none
    object_color  red / green / blue / yellow / none
    place_kind    basket / box / table / left / right / other / none
    place_color   red / green / blue / yellow / none
    number        one / several
    object_name   (extracted) the word naming the object, as said

The words and patterns of the held-out test sentences (esfera, sphere,
canasta, otzara, crate, bloque, weather/time questions, self-corrections) are
deliberately not in the training data, so that test measures generalisation.

Run (about 27 minutes on an RTX 5060 Ti; the model is 1.1 GB):

    python .\\demos\\02_voice\\train_command_model.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

BASE_MODEL = "fastino/gliner2.5-multi-v1"
OUTPUT = Path(__file__).with_name("models") / "command-gliner"

LABELS = {
    "intent": ["pick_and_place", "pick", "greeting", "other"],
    "object_kind": ["ball", "cube", "any", "other", "none"],
    "object_color": ["red", "green", "blue", "yellow", "none"],
    "place_kind": ["basket", "box", "table", "left", "right", "other", "none"],
    "place_color": ["red", "green", "blue", "yellow", "none"],
    "number": ["one", "several"],
}
COLOR_NAMES = ["red", "green", "blue", "yellow"]

# --- Spanish -----------------------------------------------------------------

ES_OBJECTS = [  # (singular, plural, feminine, kind)
    ("bola", "bolas", True, "ball"), ("pelota", "pelotas", True, "ball"),
    ("cubo", "cubos", False, "cube"), ("dado", "dados", False, "cube"),
    ("objeto", "objetos", False, "any"), ("pieza", "piezas", True, "any"), ("cosa", "cosas", True, "any"),
    ("elemento", "elementos", False, "any"), ("figura", "figuras", True, "any"),
    ("plátano", "plátanos", False, "other"), ("pera", "peras", True, "other"),
    ("manzana", "manzanas", True, "other"), ("naranja", "naranjas", True, "other"),
    ("destornillador", "destornilladores", False, "other"), ("vaso", "vasos", False, "other"),
    ("taza", "tazas", True, "other"), ("llave", "llaves", True, "other"), ("lápiz", "lápices", False, "other"),
    ("botella", "botellas", True, "other"), ("martillo", "martillos", False, "other"),
    ("cuchara", "cucharas", True, "other"), ("libro", "libros", False, "other"), ("juguete", "juguetes", False, "other"),
]
ES_COLORS = {"red": ("rojo", "roja"), "green": ("verde", "verde"), "blue": ("azul", "azul"),
             "yellow": ("amarillo", "amarilla")}
ES_PLACES = [  # (noun, feminine, kind)
    ("cesto", False, "basket"), ("cesta", True, "basket"), ("caja", True, "box"), ("mesa", True, "table"),
    ("estantería", True, "other"), ("silla", True, "other"), ("suelo", False, "other"),
    ("cajón", False, "other"), ("armario", False, "other"),
]
ES_PICK = ["coge", "agarra", "toma", "recoge", "levanta", "puedes coger", "me puedes coger", "podrías coger",
           "quiero que cojas", "pásame", "dame", "por favor, coge", "coge por favor"]
ES_PUT = ["déja", "pon", "méte", "lléva", "colóca", "suélta"]  # + lo/la/los/las
ES_PUT_ALONE = ["pon", "mete", "deja", "coloca"]
ES_GREETINGS = ["hola", "hola, ¿qué tal?", "buenos días", "buenas tardes", "buenas noches", "hola robot",
                "¿cómo estás?", "hola, buenas", "qué tal estás", "hola, ¿cómo estás?", "adiós", "hasta luego"]
ES_OTHER = ["cuéntame un chiste", "¿cómo te llamas?", "¿cuántos años tienes?", "apágate", "¿qué puedes hacer?",
            "pon música", "¿dónde estamos?", "gracias", "me gusta el fútbol", "¿quién te ha programado?",
            "estoy cansado", "abre la puerta"]


def es_plural(word: str) -> str:
    return word + ("s" if word[-1] in "aeiouáéó" else "es")


def es_sentence(rng: random.Random) -> tuple[str, dict, list[str]]:
    kind = rng.random()
    if kind < 0.1:
        return rng.choice(ES_GREETINGS), {"intent": "greeting"}, []
    if kind < 0.2:
        return rng.choice(ES_OTHER), {"intent": "other"}, []
    plural = rng.random() < 0.2
    color = rng.choice(COLOR_NAMES + [None])
    names = []
    if rng.random() < 0.12:  # colour only: "coge el rojo"
        color = color or "red"
        noun, feminine, obj_kind = None, rng.random() < 0.5, "any"
    else:
        singular, plural_word, feminine, obj_kind = rng.choice(ES_OBJECTS)
        noun = plural_word if plural else singular
        names = [noun]
    article = {(False, False): "el", (True, False): "la", (False, True): "los", (True, True): "las"}[feminine, plural]
    if rng.random() < 0.15:
        article = {"el": "un", "la": "una", "los": "unos", "las": "unas"}[article]
    color_word = ""
    if color:
        color_word = ES_COLORS[color][feminine]
        color_word = es_plural(color_word) if plural else color_word
    obj_phrase = " ".join(filter(None, [article, noun, color_word]))
    labels = {"intent": "pick", "object_kind": obj_kind, "object_color": color or "none",
              "number": "several" if plural else "one"}

    place_text = ""
    if rng.random() < 0.7:
        side = rng.random() < 0.15
        if side:
            place_kind = rng.choice(["left", "right"])
            place_text = "a la izquierda" if place_kind == "left" else "a la derecha"
            place_color = None
        else:
            noun_p, fem_p, place_kind = rng.choice(ES_PLACES)
            place_color = rng.choice(COLOR_NAMES + [None, None]) if place_kind != "other" else None
            art = "la" if fem_p else "el"
            prep = rng.choice(["en", "en", "dentro de", "encima de", "sobre"] if place_kind == "table"
                              else ["en", "en", "dentro de", "a"])
            if prep == "a" and art == "el":
                head = f"al {noun_p}"
            elif prep in ("dentro de", "encima de") and art == "el":
                head = f"{prep[:-3]} del {noun_p}"
            else:
                head = f"{prep} {art} {noun_p}"
            place_text = head + (f" {ES_COLORS[place_color][fem_p]}" if place_color else "")
        labels.update(intent="pick_and_place", place_kind=place_kind, place_color=place_color or "none")
        pronoun = ("la" if feminine else "lo") + ("s" if plural else "")
        if rng.random() < 0.2:  # "pon en la caja roja el cubo azul"
            text = f"{rng.choice(ES_PUT_ALONE)} {place_text} {obj_phrase}"
        elif rng.random() < 0.25:
            text = f"{rng.choice(ES_PUT_ALONE)} {obj_phrase} {place_text}"
        else:
            text = f"{rng.choice(ES_PICK)} {obj_phrase} y {rng.choice(ES_PUT)}{pronoun} {place_text}"
    else:
        text = f"{rng.choice(ES_PICK)} {obj_phrase}"
    return text, labels, names


# --- English -----------------------------------------------------------------

EN_OBJECTS = [  # (singular, plural, kind)
    ("ball", "balls", "ball"), ("cube", "cubes", "cube"), ("block", "blocks", "cube"),
    ("object", "objects", "any"), ("thing", "things", "any"), ("item", "items", "any"), ("piece", "pieces", "any"),
    ("one", "ones", "any"),
    ("banana", "bananas", "other"), ("pear", "pears", "other"), ("apple", "apples", "other"),
    ("orange", "oranges", "other"), ("screwdriver", "screwdrivers", "other"), ("cup", "cups", "other"),
    ("glass", "glasses", "other"), ("bottle", "bottles", "other"), ("key", "keys", "other"),
    ("pencil", "pencils", "other"), ("hammer", "hammers", "other"), ("spoon", "spoons", "other"),
    ("book", "books", "other"), ("toy", "toys", "other"),
]
EN_PLACES = [("basket", "basket"), ("bin", "basket"), ("box", "box"), ("table", "table"),
             ("shelf", "other"), ("chair", "other"), ("floor", "other"), ("drawer", "other"), ("cupboard", "other")]
EN_PICK = ["pick up", "grab", "take", "get", "lift", "fetch", "could you pick up", "please grab",
           "I want you to take", "hand me", "can you grab"]
EN_PUT = ["put", "place", "drop", "leave", "move", "set"]
EN_GREETINGS = ["hello", "hi", "hey", "good morning", "hi there", "how are you", "hey robot", "good evening",
                "hello, how are you?", "goodbye", "see you later"]
EN_OTHER = ["tell me a joke", "what's your name?", "how old are you?", "what can you do?", "play some music",
            "thank you", "where are we?", "I like football", "who programmed you?", "I'm tired", "open the door"]


def en_sentence(rng: random.Random) -> tuple[str, dict, list[str]]:
    kind = rng.random()
    if kind < 0.1:
        return rng.choice(EN_GREETINGS), {"intent": "greeting"}, []
    if kind < 0.2:
        return rng.choice(EN_OTHER), {"intent": "other"}, []
    plural = rng.random() < 0.2
    color = rng.choice(COLOR_NAMES + [None])
    names = []
    if rng.random() < 0.08:  # "pick up the red"
        color = color or "red"
        noun, obj_kind = None, "any"
    else:
        singular, plural_word, obj_kind = rng.choice(EN_OBJECTS)
        noun = plural_word if plural else singular
        if obj_kind == "any" and singular == "one" and not color:
            color = "red"
        names = [noun]
    obj_phrase = " ".join(filter(None, ["the" if (plural or rng.random() < 0.85) else "a", color, noun]))
    labels = {"intent": "pick", "object_kind": obj_kind, "object_color": color or "none",
              "number": "several" if plural else "one"}
    if rng.random() < 0.7:
        if rng.random() < 0.15:
            place_kind = rng.choice(["left", "right"])
            place_text = rng.choice(["on the", "to the"]) + " " + place_kind
            place_color = None
        else:
            noun_p, place_kind = rng.choice(EN_PLACES)
            place_color = rng.choice(COLOR_NAMES + [None, None]) if place_kind != "other" else None
            prep = rng.choice(["on", "onto"]) if noun_p in ("table", "shelf", "chair", "floor") else rng.choice(
                ["in", "into", "inside"])
            place_text = " ".join(filter(None, [prep, "the", place_color, noun_p]))
        labels.update(intent="pick_and_place", place_kind=place_kind, place_color=place_color or "none")
        it = "them" if plural else "it"
        if rng.random() < 0.4:  # "put the red ball in the green basket"
            text = f"{rng.choice(EN_PUT)} {obj_phrase} {place_text}"
        else:
            text = f"{rng.choice(EN_PICK)} {obj_phrase} and {rng.choice(EN_PUT)} {it} {place_text}"
    else:
        text = f"{rng.choice(EN_PICK)} {obj_phrase}"
    return text, labels, names


# --- Basque ------------------------------------------------------------------

EU_OBJECTS = [  # (stem, bare with article, plural, kind)
    ("bola", "bola", "bolak", "ball"), ("pilota", "pilota", "pilotak", "ball"), ("kubo", "kuboa", "kuboak", "cube"),
    ("objektu", "objektua", "objektuak", "any"), ("gauza", "gauza", "gauzak", "any"), ("pieza", "pieza", "piezak", "any"),
    ("platano", "platanoa", "platanoak", "other"), ("sagar", "sagarra", "sagarrak", "other"),
    ("udare", "udarea", "udareak", "other"), ("laranja", "laranja", "laranjak", "other"),
    ("bihurkin", "bihurkina", "bihurkinak", "other"), ("edalontzi", "edalontzia", "edalontziak", "other"),
    ("kikara", "kikara", "kikarak", "other"), ("giltza", "giltza", "giltzak", "other"),
    ("arkatz", "arkatza", "arkatzak", "other"), ("botila", "botila", "botilak", "other"),
    ("mailu", "mailua", "mailuak", "other"), ("koilara", "koilara", "koilarak", "other"),
    ("liburu", "liburua", "liburuak", "other"), ("jostailu", "jostailua", "jostailuak", "other"),
]
# colour: (with article, plural, "in the ...", "to the ...", "of the ...")
EU_COLORS = {"red": ("gorria", "gorriak", "gorrian", "gorrira", "gorriaren"),
             "green": ("berdea", "berdeak", "berdean", "berdera", "berdearen"),
             "blue": ("urdina", "urdinak", "urdinean", "urdinera", "urdinaren"),
             "yellow": ("horia", "horiak", "horian", "horira", "horiaren")}
EU_CONTAINERS = [("saski", "saskian", "saskira", "basket"), ("kutxa", "kutxan", "kutxara", "box")]
EU_OTHER_PLACES = ["apalean", "aulkian", "lurrean", "tiraderan", "armairuan"]
EU_PICK = ["hartu", "jaso", "ekarri", "hartu mesedez", "mesedez, hartu", "har ezazu", "eman"]
EU_PUT = ["sartu", "jarri", "utzi", "eraman"]
EU_GREETINGS = ["kaixo", "kaixo, zer moduz?", "egun on", "arratsalde on", "gabon", "kaixo robot",
                "zer moduz zaude?", "aupa", "agur", "gero arte"]
EU_OTHER = ["kontatu txiste bat", "nola deitzen zara?", "zenbat urte dituzu?", "zer egin dezakezu?",
            "jarri musika", "eskerrik asko", "non gaude?", "futbola gustatzen zait", "nekatuta nago", "ireki atea"]


def eu_sentence(rng: random.Random) -> tuple[str, dict, list[str]]:
    kind = rng.random()
    if kind < 0.1:
        return rng.choice(EU_GREETINGS), {"intent": "greeting"}, []
    if kind < 0.2:
        return rng.choice(EU_OTHER), {"intent": "other"}, []
    plural = rng.random() < 0.2
    color = rng.choice(COLOR_NAMES + [None])
    names = []
    if rng.random() < 0.12:  # "hartu gorria"
        color = color or "red"
        obj_phrase, obj_kind = EU_COLORS[color][1 if plural else 0], "any"
    else:
        stem, bare, plural_word, obj_kind = rng.choice(EU_OBJECTS)
        if color:
            obj_phrase = f"{stem} {EU_COLORS[color][1 if plural else 0]}"
            names = [stem]
        else:
            obj_phrase = plural_word if plural else bare
            names = [obj_phrase]
    labels = {"intent": "pick", "object_kind": obj_kind, "object_color": color or "none",
              "number": "several" if plural else "one"}
    if rng.random() < 0.7:
        roll = rng.random()
        place_color = None
        if roll < 0.15:
            place_kind = rng.choice(["left", "right"])
            place_text = {"left": rng.choice(["ezkerrean", "ezkerraldean"]),
                          "right": rng.choice(["eskuinean", "eskuinaldean"])}[place_kind]
        elif roll < 0.3:
            place_kind, place_text = "other", rng.choice(EU_OTHER_PLACES)
        elif roll < 0.5:
            place_kind = "table"
            place_color = rng.choice(COLOR_NAMES + [None, None])
            place_text = f"mahai {EU_COLORS[place_color][4]} gainean" if place_color else "mahai gainean"
        else:
            stem_p, inessive, allative, place_kind = rng.choice(EU_CONTAINERS)
            place_color = rng.choice(COLOR_NAMES + [None, None])
            to = rng.random() < 0.3
            if place_color:
                place_text = f"{stem_p} {EU_COLORS[place_color][3 if to else 2]}"
            else:
                place_text = allative if to else inessive
        labels.update(intent="pick_and_place", place_kind=place_kind, place_color=place_color or "none")
        put = "sartu" if place_kind in ("basket", "box") and rng.random() < 0.5 else rng.choice(EU_PUT)
        if rng.random() < 0.25:  # object first: "platanoa hartu eta ..."
            text = f"{obj_phrase} hartu eta {put} {place_text}"
        elif rng.random() < 0.2:
            text = f"{put} {obj_phrase} {place_text}"
        else:
            text = f"{rng.choice(EU_PICK)} {obj_phrase} eta {put} {place_text}"
    else:
        text = f"{rng.choice(EU_PICK)} {obj_phrase}" if rng.random() < 0.75 else f"{obj_phrase} hartu"
    return text, labels, names


# --- dataset -----------------------------------------------------------------

def example(text: str, labels: dict, names: list[str]):
    from gliner2.training.data import Classification, InputExample

    full = {"object_kind": "none", "object_color": "none", "place_kind": "none", "place_color": "none",
            "number": "one", **labels}
    names = [n for n in names if n.lower() in text.lower()]
    return InputExample(
        text=text,
        # greetings and colour-only commands name nothing: no entity task for them
        entities={"object_name": names} if names else {},
        entity_descriptions={"object_name": "the word naming the thing to pick up"} if names else None,
        classifications=[Classification(task=task, labels=options, true_label=full[task])
                         for task, options in LABELS.items()],
    )


def build(n_per_language: int, seed: int) -> list:
    rng = random.Random(seed)
    data = []
    for make in (es_sentence, en_sentence, eu_sentence):
        for _ in range(n_per_language):
            text, labels, names = make(rng)
            if rng.random() < 0.5:
                text = text[0].upper() + text[1:]
            if rng.random() < 0.3 and text[-1] not in ".?!":
                text += "."
            data.append(example(text, labels, names))
    rng.shuffle(data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=2000, help="Training sentences per language")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop early (for a quick check)")
    parser.add_argument("--show", type=int, default=0, help="Only print this many generated examples")
    args = parser.parse_args()

    try:
        import truststore  # Norton re-signs HTTPS; see demo 06

        truststore.inject_into_ssl()
    except ImportError:
        pass

    train = build(args.per_language, seed=1)
    evaluation = build(max(50, args.per_language // 10), seed=2)
    if args.show:
        for item in train[:args.show]:
            answers = {c.task: c.true_label for c in item.classifications}
            print(f"{item.text!r:70} {item.entities.get('object_name', [])} {answers}")
        return 0

    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    model = AutoExtractor.from_pretrained(BASE_MODEL)
    config = TrainingConfig(
        output_dir=str(OUTPUT), experiment_name="command-gliner", num_epochs=args.epochs, max_steps=args.max_steps,
        batch_size=16, eval_batch_size=32, encoder_lr=2e-5, task_lr=5e-4, warmup_ratio=0.1,
        eval_strategy="epoch", save_best=True, logging_steps=50, num_workers=0, seed=1,
        fp16=False, bf16=False,  # full precision: bf16 gave non-finite losses on this task
    )
    ExtractorTrainer(model, config).train(train_data=train, eval_data=evaluation)
    # Keep only the best checkpoint (the one the demo loads); each other one is 1.1 GB.
    for checkpoint in OUTPUT.iterdir():
        if checkpoint.is_dir() and (checkpoint.name.startswith("checkpoint-") or checkpoint.name == "final"):
            shutil.rmtree(checkpoint)
    print(f"saved under {OUTPUT / 'best'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
