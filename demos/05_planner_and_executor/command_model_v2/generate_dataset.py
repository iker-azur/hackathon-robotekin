"""Step 1: generate the training data for the command model.

Every sample is a sentence someone could say to the robot at the hackathon, in
Spanish, English or Basque, together with the answers the model must learn:

    intent        what the person wants (16 options, see INTENTS)
    items         the objects mentioned, one record each:
                    mention        the whole phrase           "dos bolas rojas"
                    name           the word naming the thing  "bolas"
                    color_word     the colour word, as said   "rojas"
                    quantity_word  the number word            "dos"
                    position_word  which one of several       "izquierda"
    place_kind    where it goes (basket, box, tray, table, left, right, other, none)
    place_color   the colour of that place
    sort_by       for "sort" commands: color, kind or none

Colours, numbers and positions are extracted as the words that were said
("rojas", "bi"); command_model.py maps them to values with vocabulary.yaml.
The kind of each item (ball, cube, any, other) and whether it is one or several
are learnt from a second, smaller set of examples: the item phrases alone.

Outputs, in ./data (JSON Lines, one sample per line):

    train_commands.jsonl     sentences for training (lang x N)
    train_item_kinds.jsonl   item phrases with their kind and count, for training
    eval_commands.jsonl      same distribution, other random seed, to watch training
    eval_item_kinds.jsonl
    test_unseen.jsonl        built only from the vocabulary's `test_only` words
    stats.md                 counts per language and intent, and examples

Each line has "input" and "output" in the format GLiNER2's trainer reads, plus a
"meta" block with the language, the template used and the clean command.

Run:

    python .\\demos\\03_command_model\\generate_dataset.py               # 10000 per language
    python .\\demos\\03_command_model\\generate_dataset.py --show 40     # print samples, write nothing
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VOCAB = yaml.safe_load((HERE / "vocabulary.yaml").read_text(encoding="utf-8"))

LANGS = ("es", "en", "eu")
INTENTS = [
    "pick_and_place", "pick", "give", "sort", "clear", "sequence",  # things to do
    "stop", "repeat", "cancel",                                     # control
    "answer_yes", "answer_no", "answer_value",                      # replies to a robot question
    "question", "greeting", "off_topic", "not_doable",              # no action
]
# Share of each intent in the data (sums to 1).
MIX = {
    "pick_and_place": 0.18, "pick": 0.08, "give": 0.06, "sort": 0.05, "clear": 0.04, "sequence": 0.09,
    "stop": 0.04, "repeat": 0.025, "cancel": 0.025,
    "answer_yes": 0.025, "answer_no": 0.025, "answer_value": 0.04,
    "question": 0.05, "greeting": 0.06, "off_topic": 0.09, "not_doable": 0.12,
}
COLORS = list(VOCAB["colors"])
PLACE_KINDS = ["basket", "box", "tray", "table", "left", "right", "other", "none"]
SORT_BY = ["color", "kind", "none"]
ITEM_KINDS = ["ball", "cube", "cylinder", "triangular_prism", "hexagonal_prism", "any", "other"]
ITEM_COUNTS = ["one", "several"]
ITEM_FIELDS = ["mention", "name", "color_word", "quantity_word", "position_word",
               "destination_kind_word", "destination_color_word"]

# ADDED IN THIS COPY. The colours a multi-destination sort is allowed to talk about.
#
# Restricted to what perception can actually distinguish. vocabulary.yaml has nine, and the
# command model can name all nine, but this repository's HSV segmentation separates only
# red, green and blue by default, and the nine-colour measurement found orange 5 hue units
# from yellow and pink 11 from red with saturation separating neither. A sort sentence about
# orange would train the model toward a sort the robot cannot carry out.
SORTABLE_COLORS = ["red", "green", "blue"]


# --- what a sample is made of ---------------------------------------------------

@dataclass
class Item:
    kind: str                     # supported geometry / any / other
    noun: list | None             # vocabulary entry, None for "la roja" (colour only)
    color: str | None             # canonical colour
    quantity: int | str | None    # None = "the" (one), 1 = "a/una/bat", 2..5, "all", "plural" (the X-s)
    position: str | None = None   # left / right / nearest ...

    # ADDED IN THIS COPY. Where this particular item goes, for a command that sends
    # different things to different places: "pon la bola roja en la caja azul y el cubo
    # verde en la bandeja roja".
    #
    # Demo 03's schema had exactly one place per command and said so deliberately - its
    # README: *"One destination per command. 'Put the red ones in the red box and the blue
    # ones in the blue box' is handled by `sort` (by colour), not by per-item
    # destinations."* But `sort` had no way to express a coloured destination either:
    # generate_dataset forced `place.color = None` for every sort sample, so all 1500 of
    # them were trained with an uncoloured destination. This field is what makes the
    # sentence above expressible at all.
    dest: "Place | None" = None

    @property
    def several(self) -> bool:
        return self.quantity in ("all", "plural") or (isinstance(self.quantity, int) and self.quantity > 1)


@dataclass
class Place:
    kind: str                     # basket / box / tray / table / left / right / other
    noun: list | None
    color: str | None = None


@dataclass
class Phrase:
    text: str
    spans: dict = field(default_factory=dict)  # name / color_word / quantity_word / position_word -> words
    # ADDED IN THIS COPY: destination_kind_word / destination_color_word. Two flat fields
    # rather than a nested structure, because GLiNER2.5 cannot nest: calling .structure()
    # on a record builder emits a *sibling* record, not a child (verified against the
    # installed gliner2 2.0.0). They must not use choices=, either: demo 03's README records
    # that multiple-choice fields inside a repeated record do not depend on which record
    # they belong to, and the second item copied the first one's colour. Extracting the words
    # and looking them up in a table is what the existing five fields already do.
    #
    # Two and not three: a third "destination_mention" holding the whole noun phrase was
    # tried and dropped. to_records asserts every labelled word occurs verbatim in the
    # sentence, and a composed phrase is not a literal substring - Basque inflects the
    # colour differently inside place() than place_spans() would, since place() picks
    # between the "in" and "to" forms at random. Both words that remain do occur verbatim,
    # and they are all the lookup needs.


@dataclass
class Sample:
    lang: str
    intent: str
    text: str
    template: str
    items: list = field(default_factory=list)   # [(Item, Phrase)]
    place: Place | None = None
    sort_by: str = "none"


# --- sampling the content (shared by all languages) -------------------------------

def pick_color(rng, chance=0.8):
    if rng.random() > chance:
        return None
    # the first four colours are the most likely ones in a robot demo
    return rng.choice(COLORS[:4] * 3 + COLORS[4:])


def sample_item(rng, words, lang, *, allow_color_only=True, singular=False, need_color=False):
    # Give every geometry the robot can manipulate enough examples to learn its own class.
    # Previously cylinders and prisms were sampled as ``other`` (or omitted), making the
    # trained classifier structurally unable to name them even though the executor supports
    # them.
    kinds = (["ball"] * 18 + ["cube"] * 18 + ["cylinder"] * 14
             + ["triangular_prism"] * 14 + ["hexagonal_prism"] * 14
             + ["any"] * 12 + ["other"] * 10)
    kind = rng.choice(kinds)
    entries = words["objects"][kind].get(lang) or words["objects"]["train"][kind][lang]
    color_only = kind == "any" and allow_color_only and rng.random() < 0.4
    noun = None if color_only else rng.choice(entries)
    color = pick_color(rng, 1.0 if (color_only or need_color) else (0.5 if kind == "other" else 0.8))
    roll = rng.random()
    if singular or roll < 0.62:
        quantity = None
    elif roll < 0.72:
        quantity = 1
    elif roll < 0.84:
        quantity = rng.choice([2, 2, 3, 3, 4, 5])
    elif roll < 0.92:
        quantity = "plural"
    else:
        quantity = "all"
    position = None
    if quantity is None and not color_only and rng.random() < 0.08:
        position = rng.choice(list(VOCAB["positions"]))
    return Item(kind, noun, color, quantity, position)


def sample_place(rng, words, lang, *, usable_only=False):
    kinds = ["basket"] * 25 + ["box"] * 25 + ["tray"] * 12 + ["table"] * 15 + ["left"] * 7 + ["right"] * 7
    if not usable_only:
        kinds += ["other"] * 9
    kind = rng.choice(kinds)
    if kind in ("left", "right"):
        return Place(kind, None)
    entries = _place_entries(words, lang, kind)
    color = pick_color(rng, 0.4) if kind != "other" else None
    return Place(kind, rng.choice(entries), color)


def vocab_split(split: str) -> dict:
    """Word lists for a split: train words, or test-only words where they exist (else train)."""
    if split == "train":
        return {"objects": VOCAB["objects"]["train"], "places": VOCAB["places"]["train"]}
    objects = {k: VOCAB["objects"]["test_only"].get(k) or {} for k in VOCAB["objects"]["train"]}
    places = {k: VOCAB["places"]["test_only"].get(k) or {} for k in VOCAB["places"]["train"]}
    return {"objects": {**objects, "train": VOCAB["objects"]["train"]},
            "places": {**places, "train": VOCAB["places"]["train"]}}


# --- Spanish ------------------------------------------------------------------

class Spanish:
    lang = "es"
    PICK = ["coge", "agarra", "toma", "recoge", "levanta", "pilla"]
    PICK_ASK = ["¿puedes coger {x}?", "¿me puedes coger {x}?", "¿podrías coger {x}?", "quiero que cojas {x}",
                "necesito que cojas {x}", "por favor, coge {x}", "coge {x}, por favor", "a ver, coge {x}",
                "¿coges {x}?", "venga, coge {x}"]
    PUT_PRONOUN = ["déja", "pon", "méte", "lléva", "colóca", "suélta", "guárda"]  # + lo/la/los/las
    PUT = ["pon", "mete", "deja", "coloca", "lleva", "guarda"]
    PUT_ASK = ["¿puedes poner {x} {p}?", "¿puedes meter {x} {p}?", "quiero que pongas {x} {p}",
               "necesito que dejes {x} {p}", "¿me pones {x} {p}?"]
    GIVE = ["pásame {x}", "dame {x}", "tráeme {x}", "acércame {x}", "¿me pasas {x}?", "¿me das {x}?",
            "necesito {x}", "quiero {x}", "¿me acercas {x}?", "pásame {x}, porfa", "¿me traes {x}?"]
    SORT_COLOR = ["separa {g} por colores", "clasifica {g} por color", "ordena {g} por colores", "agrupa {g} por color",
                  "separa {g} según su color", "clasifica {g} por colores"]
    SORT_KIND = ["separa {g} por tipo", "clasifica {g} por forma", "ordena {g} por tipo", "separa las bolas de los cubos",
                 "separa los cubos de las bolas", "agrupa {g} por forma"]
    GROUPS = ["las piezas", "los objetos", "todo", "las cosas", "todas las piezas", "los elementos"]
        # ADDED IN THIS COPY. Two groups, each with its own matching receptacle, which is
    # the sentence demo 03 could not represent: "mete las rojas en la caja roja y las
    # azules en la bandeja azul". {g1}/{g2} is the group as a noun phrase, {p1}/{p2} is
    # the receptacle with its preposition.
    SORT_INTO = ["mete {g1} {p1} y {g2} {p2}",
                 "pon {g1} {p1} y {g2} {p2}",
                 "separa por colores: {g1} {p1} y {g2} {p2}",
                 "cada color en su caja: {g1} {p1} y {g2} {p2}",
                 "ordena {g1} {p1} y {g2} {p2}"]
    # One item each to its own place. {x} is the item, {p} its destination.
    EACH_OWN_PLACE = ["mete {x} {p} y {x2} {p2}", "pon {x} {p} y {x2} {p2}",
                      "pon {x} {p} y luego {x2} {p2}"]
    CLEAR = ["recoge la mesa", "limpia la mesa", "despeja la mesa", "recógelo todo", "guárdalo todo",
             "quita todo de la mesa", "deja la mesa limpia", "recoge todo", "ordena la mesa"]
    CLEAR_PLACE = ["mete todo {p}", "pon todo {p}", "recoge todo y mételo {p}", "guarda todo {p}", "lleva todo {p}"]
    STOP = ["para", "¡para!", "detente", "stop", "quieto", "alto", "para ya", "no, para", "espera, para", "frena",
            "¡quieto!", "para, para", "detente ahora", "¡alto!", "para inmediatamente", "no te muevas"]
    REPEAT = ["otra vez", "repítelo", "hazlo otra vez", "vuelve a intentarlo", "inténtalo de nuevo", "una vez más",
              "repite", "otra vez, por favor", "hazlo de nuevo", "vuelve a hacerlo"]
    CANCEL = ["cancela", "déjalo", "olvídalo", "no importa", "cancela eso", "mejor no", "déjalo estar",
              "cancela la orden", "nada, déjalo", "olvida lo que te he dicho"]
    YES = ["sí", "vale", "de acuerdo", "claro", "correcto", "eso es", "exacto", "sí, por favor", "venga", "sí, esa",
           "perfecto", "afirmativo", "sí, eso"]
    NO = ["no", "no, gracias", "incorrecto", "no es eso", "esa no", "negativo", "no, esa no", "para nada", "no, no"]
    QUESTIONS = ["¿qué ves?", "¿qué hay en la mesa?", "¿qué estás haciendo?", "¿has terminado?", "¿estás listo?",
                 "¿qué puedes hacer?", "¿qué objetos hay?", "¿cuántas piezas hay?", "¿qué tienes en la pinza?",
                 "¿estás funcionando?", "¿qué colores ves?", "¿dónde está la caja?"]
    QUESTION_ITEM = ["¿dónde está {x}?", "¿ves {x}?", "¿hay {x} en la mesa?", "¿de qué color es {x}?",
                     "¿puedes ver {x}?", "¿cuántas {plural} hay?"]
    GREETINGS = ["hola", "hola, ¿qué tal?", "buenos días", "buenas tardes", "buenas noches", "hola robot",
                 "¿cómo estás?", "hola, buenas", "qué tal estás", "hola, ¿cómo estás?", "adiós", "hasta luego",
                 "encantado de conocerte", "hasta mañana", "buenas", "hola, ¿cómo te va?", "nos vemos", "chao"]
    OFF_TOPIC = ["cuéntame un chiste", "¿cómo te llamas?", "¿cuántos años tienes?", "pon música",
                 "¿dónde estamos?", "gracias", "me gusta el fútbol", "¿quién te ha programado?", "estoy cansado",
                 "abre la puerta", "¿qué tiempo hace hoy?", "¿qué hora es?", "¿quién ganó ayer?", "tengo hambre",
                 "¿cuánto es dos más dos?", "llama a mi madre", "apaga la luz", "¿me quieres?", "qué calor hace",
                 "¿de dónde eres?", "canta una canción", "hoy es mi cumpleaños", "¿sabes hablar euskera?",
                 "el examen fue difícil", "vamos a comer", "¿cuál es la capital de Francia?", "estoy aburrido",
                 "mañana tengo clase", "¿te gusta la pizza?", "qué bonito día"]
    IMPOSSIBLE = ["el coche", "la casa", "la mesa", "la pared", "el sol", "la lámpara del techo", "el ordenador",
                  "la silla", "el suelo", "la puerta", "el edificio", "la luna", "el árbol", "el camión"]
    PEOPLE = ["coge a Juan", "coge al perro", "coge al gato", "dame la mano", "tócame", "coge a mi hermano",
              "levanta al niño", "abrázame", "coge mi brazo"]
    DESTROY = ["tira {x} al suelo", "lanza {x}", "rompe {x}", "golpea {x}", "aplasta {x}", "tira {x} por la ventana",
               "empuja {x} fuera de la mesa", "lanza {x} contra la pared", "destroza {x}"]
    NOT_NOW = ["ayer cogí {x}", "mi hermano cogió {x}", "antes había {x} en la mesa", "he puesto {x} en la caja",
               "si cogieras {x} sería genial", "no cojas {x}", "no toques {x}", "no muevas {x}"]
    NOTHING = ["no cojas nada", "no toques nada", "no hagas nada", "no muevas nada", "no hace falta que cojas nada"]
    FILLERS = ["eh, ", "venga, ", "oye, ", "a ver, ", "robot, ", "vale, ", "bueno, ", "mmm, "]
    POLITE_END = [", por favor", " por favor", ", porfa", ", gracias"]
    CORRECTION = ["{a}, no, mejor {b}", "{a}... no, espera, {b}", "{a}, perdón, {b}", "{a}, no, {b}"]

    def item(self, item: Item) -> Phrase:
        spans = {}
        if item.noun is None:  # colour only: "la roja", "dos rojas", "todas las rojas"
            feminine = random.random() < 0.6
            name_sg = name_pl = None
        else:
            name_sg, name_pl, gender = item.noun
            feminine = gender == "f"
        plural = item.several
        name = name_pl if plural else name_sg
        color = ""
        if item.color:
            forms = VOCAB["colors"][item.color]["es"]
            color = forms[(2 if plural else 0) + (1 if feminine else 0)]
            spans["color_word"] = color
        if item.quantity is None:
            head = "la" if feminine else "el"
        elif item.quantity == 1:
            head = "una" if feminine else "un"
            spans["quantity_word"] = head
        elif item.quantity == "plural":
            head = "las" if feminine else "los"
        elif item.quantity == "all":
            word = "todas" if feminine else "todos"
            head = f"{word} {'las' if feminine else 'los'}"
            spans["quantity_word"] = word
        else:
            head = VOCAB["numbers"][item.quantity]["es"][0]
            spans["quantity_word"] = head
        parts = [head]
        if name:
            parts.append(name)
            spans["name"] = name
        if color:
            parts.append(color)
        text = " ".join(parts)
        if item.position:
            word = rng_choice(VOCAB["positions"][item.position]["es"])
            if item.position in ("nearest", "farthest"):
                word = word[:-1] + ("a" if feminine else "o")
                text += f" más {word}"
            elif item.position in ("left", "right"):
                text += f" de la {word}"
            elif word == "fondo":
                text += " del fondo"
            else:
                text += f" de {word}"
            spans["position_word"] = word
        return Phrase(text, spans)

    def place(self, place: Place) -> str:
        if place.kind == "left":
            return random.choice(["a la izquierda", "en la izquierda", "a tu izquierda", "en el lado izquierdo"])
        if place.kind == "right":
            return random.choice(["a la derecha", "en la derecha", "a tu derecha", "en el lado derecho"])
        noun, gender = place.noun
        article = "la" if gender == "f" else "el"
        color = ""
        if place.color:
            color = " " + VOCAB["colors"][place.color]["es"][1 if gender == "f" else 0]
        if place.kind == "table" or noun in ("estantería", "armario", "silla", "sofá", "nevera"):
            prep = random.choice(["en", "encima de", "sobre"])
        else:
            prep = random.choice(["en", "en", "dentro de", "a"])
        if prep in ("a", "dentro de", "encima de") and article == "el":
            head = {"a": "al", "dentro de": "dentro del", "encima de": "encima del"}[prep]
            return f"{head} {noun}{color}"
        return f"{prep} {article} {noun}{color}"

    def pronoun(self, item: Item | None) -> str:
        if item is None:
            return "lo"
        feminine = item.noun is not None and item.noun[2] == "f"
        return ("la" if feminine else "lo") + ("s" if item.several else "")

    def place_spans(self, place: Place, rendered: str) -> dict:
        """The two destination words, read back out of the rendered destination.

        ADDED IN THIS COPY; see Spanish.place_spans and _longest_form_in.
        """
        if place.kind in ("left", "right"):
            return {"destination_kind_word": place.kind, "destination_color_word": ""}
        noun, _gender = place.noun
        forms = VOCAB["colors"][place.color]["es"] if place.color else []
        return {"destination_kind_word": noun,
                "destination_color_word": _longest_form_in(rendered, forms)}

    def plural_noun(self, item: Item) -> str:
        return item.noun[1] if item.noun else "piezas"


# --- English ------------------------------------------------------------------

class English:
    lang = "en"
    PICK = ["pick up", "grab", "take", "get", "lift", "fetch", "pick"]
    PICK_ASK = ["could you pick up {x}?", "can you grab {x}?", "please take {x}", "I want you to pick up {x}",
                "I need you to grab {x}", "pick up {x}, please", "can you get {x}?", "would you pick up {x}?",
                "go ahead and grab {x}", "okay, take {x}"]
    PUT = ["put", "place", "drop", "leave", "move", "set", "stick"]
    PUT_ASK = ["can you put {x} {p}?", "could you place {x} {p}?", "I want you to put {x} {p}",
               "please move {x} {p}", "would you drop {x} {p}?"]
    GIVE = ["hand me {x}", "give me {x}", "pass me {x}", "bring me {x}", "can you hand me {x}?",
            "could you pass me {x}?", "I need {x}", "I want {x}", "can I have {x}?", "get me {x}"]
    SORT_COLOR = ["sort {g} by colour", "sort {g} by color", "group {g} by colour", "separate {g} by color",
                  "organise {g} by colour", "sort {g} by their colour"]
    SORT_KIND = ["sort {g} by type", "sort {g} by shape", "separate the balls from the cubes",
                 "separate the cubes from the balls", "group {g} by shape", "organise {g} by type"]
    GROUPS = ["the pieces", "the objects", "everything", "the things", "all the pieces", "the items"]
        # ADDED IN THIS COPY; see Spanish.SORT_INTO.
    SORT_INTO = ["put {g1} {p1} and {g2} {p2}",
                 "sort {g1} {p1} and {g2} {p2}",
                 "put each colour in its own box: {g1} {p1} and {g2} {p2}",
                 "the {g1} go {p1}, the {g2} {p2}"]
    EACH_OWN_PLACE = ["put {x} {p} and {x2} {p2}", "place {x} {p} and {x2} {p2}",
                      "first {x} {p}, then {x2} {p2}"]
    CLEAR = ["clear the table", "clean up the table", "tidy up", "tidy the table", "pick everything up",
             "remove everything from the table", "clean up", "put everything away", "clear everything"]
    CLEAR_PLACE = ["put everything {p}", "move everything {p}", "pick everything up and put it {p}",
                   "put it all {p}", "drop everything {p}"]
    STOP = ["stop", "stop!", "halt", "freeze", "wait, stop", "stop right now", "don't move", "hold on, stop",
            "stop it", "stop, stop", "emergency stop", "stop moving", "stop now"]
    REPEAT = ["again", "one more time", "do it again", "try again", "repeat that", "once more", "again, please",
              "do that again", "try it again", "repeat"]
    CANCEL = ["cancel", "forget it", "never mind", "cancel that", "leave it", "don't bother", "actually, no",
              "scratch that", "cancel the order", "forget what I said"]
    YES = ["yes", "yeah", "yep", "okay", "sure", "correct", "that's right", "exactly", "yes, please", "right",
           "that one", "affirmative", "sounds good"]
    NO = ["no", "nope", "no, thanks", "wrong", "that's not it", "not that one", "negative", "no, no", "incorrect"]
    QUESTIONS = ["what do you see?", "what's on the table?", "what are you doing?", "are you done?", "are you ready?",
                 "what can you do?", "what objects are there?", "how many pieces are there?",
                 "what are you holding?", "are you working?", "what colours do you see?", "where is the box?"]
    QUESTION_ITEM = ["where is {x}?", "do you see {x}?", "is there {x} on the table?", "what colour is {x}?",
                     "can you see {x}?", "how many {plural} are there?"]
    GREETINGS = ["hello", "hi", "hey", "good morning", "hi there", "how are you", "hey robot", "good evening",
                 "hello, how are you?", "goodbye", "see you later", "nice to meet you", "good night", "bye",
                 "hey there", "how's it going?", "morning", "see you"]
    OFF_TOPIC = ["tell me a joke", "what's your name?", "how old are you?", "play some music", "thank you",
                 "where are we?", "I like football", "who programmed you?", "I'm tired", "open the door",
                 "what's the weather like today?", "what time is it?", "who won yesterday?", "I'm hungry",
                 "what's two plus two?", "call my mom", "turn off the lights", "do you love me?", "it's so hot",
                 "where are you from?", "sing a song", "it's my birthday today", "do you speak Basque?",
                 "the exam was hard", "let's go eat", "what's the capital of France?", "I'm bored",
                 "I have class tomorrow", "do you like pizza?", "what a nice day"]
    IMPOSSIBLE = ["the car", "the house", "the table", "the wall", "the sun", "the ceiling lamp", "the computer",
                  "the chair", "the floor", "the door", "the building", "the moon", "the tree", "the truck"]
    PEOPLE = ["pick up John", "grab the dog", "pick up the cat", "give me your hand", "touch me",
              "pick up my brother", "lift the child", "hug me", "grab my arm"]
    DESTROY = ["throw {x} on the floor", "throw {x}", "break {x}", "hit {x}", "smash {x}", "throw {x} out the window",
               "push {x} off the table", "throw {x} at the wall", "destroy {x}"]
    NOT_NOW = ["yesterday I picked up {x}", "my brother took {x}", "there was {x} on the table before",
               "I put {x} in the box", "it would be great if you picked up {x}", "don't pick up {x}",
               "don't touch {x}", "don't move {x}"]
    NOTHING = ["don't pick up anything", "don't touch anything", "don't do anything", "don't move anything",
               "no need to pick anything up"]
    FILLERS = ["uh, ", "okay, ", "hey robot, ", "so, ", "um, ", "right, ", "alright, ", "robot, "]
    POLITE_END = [", please", " please", ", thanks", ", thank you"]
    CORRECTION = ["{a}, no, actually {b}", "{a}... no, wait, {b}", "{a}, sorry, {b}", "{a}, no, {b}"]

    def item(self, item: Item) -> Phrase:
        spans = {}
        plural = item.several
        if item.noun is None:  # colour only: "the red one", "two red ones"
            name = "ones" if plural else "one"
        else:
            name = item.noun[1] if plural else item.noun[0]
        color = item.color or ""
        if color:
            spans["color_word"] = color
        if item.quantity is None:
            head = "the"
        elif item.quantity == 1:
            head = "an" if (color or name)[0] in "aeiou" else "a"
            spans["quantity_word"] = head
        elif item.quantity == "plural":
            head = "the"
        elif item.quantity == "all":
            head = random.choice(["all the", "all"])
            spans["quantity_word"] = "all"
        else:
            head = VOCAB["numbers"][item.quantity]["en"][0]
            spans["quantity_word"] = head
        spans["name"] = name
        text = " ".join(filter(None, [head, color, name]))
        if item.position:
            word = rng_choice(VOCAB["positions"][item.position]["en"])
            if item.position in ("left", "right"):
                text = random.choice([f"{text} on the {word}", f"the {word} {' '.join(filter(None, [color, name]))}"])
            elif item.position in ("nearest", "farthest"):
                text = f"the {word} {' '.join(filter(None, [color, name]))}"
            else:
                text = f"{text} at the {word}" if word == "back" else f"{text} in {word}"
            spans["position_word"] = word
        return Phrase(text, spans)

    def place(self, place: Place) -> str:
        if place.kind in ("left", "right"):
            side = place.kind
            return random.choice([f"on the {side}", f"to the {side}", f"on your {side}", f"on the {side} side"])
        noun = place.noun[0]
        color = f"{place.color} " if place.color else ""
        if place.kind == "table" or noun in ("shelf", "chair", "floor", "sofa"):
            prep = random.choice(["on", "onto", "on top of"])
        else:
            prep = random.choice(["in", "into", "inside"])
        return f"{prep} the {color}{noun}"

    def pronoun(self, item: Item | None) -> str:
        return "them" if item is not None and item.several else "it"

    def place_spans(self, place: Place, rendered: str) -> dict:
        """The two destination words. ADDED IN THIS COPY; see Spanish.place_spans."""
        if place.kind in ("left", "right"):
            return {"destination_kind_word": place.kind, "destination_color_word": ""}
        forms = VOCAB["colors"][place.color]["en"] if place.color else []
        return {"destination_kind_word": place.noun[0],
                "destination_color_word": _longest_form_in(rendered, forms)}

    def plural_noun(self, item: Item) -> str:
        return item.noun[1] if item.noun else "pieces"


# --- Basque -------------------------------------------------------------------

class Basque:
    lang = "eu"
    PICK = ["hartu", "jaso", "heldu", "har ezazu", "hartu mesedez", "mesedez, hartu"]
    PICK_ASK = ["hartuko zenuke {x}?", "{x} hartu, mesedez", "{x} hartzea nahi dut", "hartu {x}, mesedez",
                "ea, hartu {x}", "tira, hartu {x}", "{x} har dezakezu?"]
    PUT = ["jarri", "utzi", "eraman", "jar ezazu", "sartu"]
    PUT_CONTAINER = ["sartu", "sar ezazu"]
    PUT_ASK = ["jarriko zenuke {x} {p}?", "{x} {p} jartzea nahi dut", "{x} {p} utzi, mesedez"]
    GIVE = ["emadazu {x}", "eman {x}", "ekarri {x}", "pasa iezadazu {x}", "ekarridazu {x}", "{x} behar dut",
            "{x} nahi dut", "emango didazu {x}?", "ekarriko didazu {x}?"]
    SORT_COLOR = ["sailkatu {g} koloreka", "banandu {g} kolorearen arabera", "ordenatu {g} koloreka",
                  "taldekatu {g} kolorez", "bereizi {g} koloreka", "sailkatu {g} kolorearen arabera"]
    SORT_KIND = ["sailkatu {g} motaren arabera", "banandu bolak eta kuboak", "bereizi kuboak eta bolak",
                 "ordenatu {g} formaren arabera", "taldekatu {g} motaka"]
    GROUPS = ["piezak", "objektuak", "dena", "gauzak", "pieza guztiak", "gauza guztiak"]
        # ADDED IN THIS COPY; see Spanish.SORT_INTO. Basque puts the destination last, so
    # the pattern is reversed but the placeholders mean the same.
    SORT_INTO = ["jarri {g1} {p1} eta {g2} {p2}", "sartu {g1} {p1} eta {g2} {p2}",
                 "kolorez bereizten: {g1} {p1} eta {g2} {p2}",
                 "kolore bakoitza bere kutxan: {g1} {p1} eta {g2} {p2}"]
    EACH_OWN_PLACE = ["jarri {x} {p} eta {x2} {p2}", "sartu {x} {p} eta {x2} {p2}",
                      "lehenengo {x} {p}, gero {x2} {p2}"]
    CLEAR = ["garbitu mahaia", "jaso dena", "kendu dena mahaitik", "jaso gauza guztiak", "utzi mahaia garbi",
             "txukundu mahaia", "gorde dena", "jaso mahaian dagoen guztia"]
    CLEAR_PLACE = ["sartu dena {p}", "jarri dena {p}", "jaso dena eta sartu {p}", "gorde dena {p}", "eraman dena {p}"]
    STOP = ["gelditu", "geldi", "stop", "gelditu orain", "nahikoa", "itxaron, gelditu", "ez mugitu", "gelditu!",
            "geldi, geldi", "gelditu berehala", "ez, gelditu", "geldirik"]
    REPEAT = ["berriro", "egin berriro", "saiatu berriro", "beste behin", "errepikatu", "berriz", "egin ezazu berriro",
              "beste behin, mesedez", "saiatu berriz"]
    CANCEL = ["utzi", "ahaztu", "bertan behera utzi", "ez du axola", "ahaztu hori", "ez, utzi", "utzi bakean",
              "ahaztu esan dizudana", "hobe ez"]
    YES = ["bai", "ados", "noski", "bai, mesedez", "hori da", "zuzena", "primeran", "bai, hori", "ondo", "baietz",
           "hori bera", "jakina"]
    NO = ["ez", "ez, eskerrik asko", "ez da hori", "okerra", "hori ez", "ezetz", "ez, ez", "ezta pentsatu ere"]
    QUESTIONS = ["zer ikusten duzu?", "zer dago mahai gainean?", "zer egiten ari zara?", "amaitu duzu?",
                 "prest zaude?", "zer egin dezakezu?", "zer objektu daude?", "zenbat pieza daude?",
                 "zer duzu pintzan?", "martxan zaude?", "zer kolore ikusten dituzu?", "non dago kutxa?"]
    QUESTION_ITEM = ["non dago {x}?", "ikusten duzu {x}?", "badago {x} mahai gainean?", "zein koloretakoa da {x}?",
                     "ikus dezakezu {x}?", "zenbat {plural} daude?"]
    GREETINGS = ["kaixo", "kaixo, zer moduz?", "egun on", "arratsalde on", "gabon", "kaixo robot", "zer moduz zaude?",
                 "aupa", "agur", "gero arte", "pozten naiz zu ezagutzeaz", "bihar arte", "kaixo, zer berri?",
                 "aupa, zer moduz?", "ikusi arte", "egun on, robot"]
    OFF_TOPIC = ["kontatu txiste bat", "nola deitzen zara?", "zenbat urte dituzu?", "jarri musika", "eskerrik asko",
                 "non gaude?", "futbola gustatzen zait", "nekatuta nago", "ireki atea", "zer eguraldi dago gaur?",
                 "zer ordu da?", "nork irabazi zuen atzo?", "gose naiz", "zenbat da bi gehi bi?", "deitu nire amari",
                 "itzali argia", "maite nauzu?", "zer bero egiten duen", "nongoa zara?", "abestu abesti bat",
                 "gaur nire urtebetetzea da", "gaztelaniaz badakizu?", "azterketa zaila izan zen", "goazen jatera",
                 "zein da Frantziako hiriburua?", "aspertuta nago", "bihar klasea daukat", "pizza gustatzen zaizu?",
                 "zein egun polita"]
    IMPOSSIBLE = ["autoa", "etxea", "mahaia", "horma", "eguzkia", "sabaiko lanpara", "ordenagailua", "aulkia",
                  "lurra", "atea", "eraikina", "ilargia", "zuhaitza", "kamioia"]
    PEOPLE = ["hartu Jon", "hartu txakurra", "hartu katua", "eman eskua", "ukitu nazazu", "hartu nire anaia",
              "altxatu umea", "besarkatu nazazu", "heldu nire besoari"]
    DESTROY = ["bota {x} lurrera", "jaurti {x}", "apurtu {x}", "jo {x}", "zapaldu {x}", "bota {x} leihotik",
               "bultzatu {x} mahaitik kanpora", "jaurti {x} hormaren kontra", "suntsitu {x}"]
    NOT_NOW = ["atzo {x} hartu nuen", "nire anaiak {x} hartu zuen", "lehen {x} zegoen mahai gainean",
               "{x} kutxan jarri dut", "{x} hartuko bazenu, primeran", "ez hartu {x}", "ez ukitu {x}", "ez mugitu {x}"]
    NOTHING = ["ez hartu ezer", "ez ukitu ezer", "ez egin ezer", "ez mugitu ezer", "ez duzu ezer hartu behar"]
    FILLERS = ["tira, ", "aizu, ", "ea, ", "robot, ", "beno, ", "ados, ", "mmm, ", "begira, "]
    POLITE_END = [", mesedez", " mesedez", ", eskerrik asko"]
    CORRECTION = ["{a}, ez, hobeto {b}", "{a}... ez, itxaron, {b}", "{a}, barkatu, {b}", "{a}, ez, {b}"]

    def item(self, item: Item) -> Phrase:
        spans = {}
        forms = VOCAB["colors"][item.color]["eu"] if item.color else None
        stem = article = plural_form = None
        if item.noun:
            stem, article, plural_form = item.noun
        quantity = item.quantity
        if quantity is None or quantity == "plural":
            plural = quantity == "plural"
            if stem and forms:
                color = forms["pl" if plural else "sg"]
                text, spans["name"], spans["color_word"] = f"{stem} {color}", stem, color
            elif stem:
                name = plural_form if plural else article
                text, spans["name"] = name, name
            else:
                color = forms["pl" if plural else "sg"]
                text, spans["color_word"] = color, color
        else:
            if quantity == 1:
                word, number_first = "bat", False
            elif quantity == "all":
                word, number_first = random.choice(VOCAB["numbers"]["all"]["eu"][:1] * 3 + VOCAB["numbers"]["all"]["eu"][1:2]), False
            else:
                word, number_first = VOCAB["numbers"][quantity]["eu"][0], True
            parts = [stem] if stem else []
            if forms:
                parts.append(forms["bare"])
                spans["color_word"] = forms["bare"]
            parts = [word] + parts if number_first else parts + [word]
            text = " ".join(parts)
            spans["quantity_word"] = word
            if stem:
                spans["name"] = stem
        if item.position:
            word = rng_choice(VOCAB["positions"][item.position]["eu"])
            text = f"{word} dagoen {text}" if item.position in ("nearest", "farthest") else f"{word} {text}"
            spans["position_word"] = word
        return Phrase(text, spans)

    def place(self, place: Place) -> str:
        towards = random.random() < 0.3
        if place.kind == "left":
            return random.choice(["ezkerrera"] if towards else ["ezkerrean", "ezkerraldean", "zure ezkerrean"])
        if place.kind == "right":
            return random.choice(["eskuinera"] if towards else ["eskuinean", "eskuinaldean", "zure eskuinean"])
        stem, inside, to = place.noun
        if place.kind == "table":
            if place.color:
                return f"mahai {VOCAB['colors'][place.color]['eu']['of']} {'gainera' if towards else 'gainean'}"
            return to if towards else inside
        if place.color:
            return f"{stem} {VOCAB['colors'][place.color]['eu']['to' if towards else 'in']}"
        return to if towards else inside

    def pronoun(self, item: Item | None) -> str:
        return ""  # Basque needs no object pronoun here: "hartu bola eta sartu kutxan"

    def place_spans(self, place: Place, rendered: str) -> dict:
        """The two destination words. ADDED IN THIS COPY; see Spanish.place_spans.

        Basque has six case forms per colour and place() picks between the "in" and "to"
        ones at random, so this reads the form back out of the rendered sentence rather
        than computing it. A first version computed it, recorded "berdean" for a sentence
        that said "kutxa berdera", and the verbatim-word assertion stopped the generator.
        """
        if place.kind in ("left", "right"):
            return {"destination_kind_word": place.kind, "destination_color_word": ""}
        if place.kind == "table":
            return {"destination_kind_word": "mahai", "destination_color_word": ""}
        forms = list(VOCAB["colors"][place.color]["eu"].values()) if place.color else []
        return {"destination_kind_word": place.noun[0],
                "destination_color_word": _longest_form_in(rendered, forms)}

    def plural_noun(self, item: Item) -> str:
        return item.noun[0] if item.noun else "pieza"


GRAMMARS = {"es": Spanish(), "en": English(), "eu": Basque()}


def rng_choice(options):
    return random.choice(options)


# --- one sample per intent ---------------------------------------------------------

def _place_entries(words, lang, kind) -> list:
    """The noun entries for one place kind in one language, train words as a fallback.

    ADDED IN THIS COPY. ``sample_place`` does this fallback inline; the templates added here
    need it too, and the ``test`` split has no ``test_only`` entry for every kind - asking
    for a tray there raises KeyError, which is how this was found.
    """
    return (words["places"].get(kind) or {}).get(lang) or words["places"]["train"][kind][lang]


def _longest_form_in(rendered: str, forms: list) -> str:
    """Which of ``forms`` occurs in ``rendered``, longest first, accents ignored.

    ADDED IN THIS COPY. The destination colour word has to be a literal substring of the
    sentence - ``to_records`` asserts it, which is the right thing to assert - and Basque
    inflects a colour six ways, with ``place`` choosing between them at random. Computing
    the word independently of the rendered sentence is therefore a coin flip: the first
    version recorded "berdean" for a sentence that said "kutxa berdera" and the assertion
    stopped the generator. Reading the word back out of the text that was actually written
    cannot disagree with it.

    Longest first so "gorrian" does not match as the bare stem "gorri". A shorter form can
    still match when it is a prefix of a longer one ("roja" inside "rojas"); the value is
    only ever used for a table lookup and both resolve to the same colour.
    """
    haystack = strip_accents(rendered.lower())
    for form in sorted((str(f) for f in forms), key=len, reverse=True):
        if strip_accents(form.lower()) in haystack:
            return form
    return ""


def _coloured_place(rng, words, lang, color, kinds=("box", "tray", "basket")):
    """A receptacle of a named colour. ADDED IN THIS COPY, for the multi-destination sorts."""
    kind = rng.choice(list(kinds))
    entries = _place_entries(words, lang, kind)
    return Place(kind, rng.choice(entries), color)


def _executable_place(rng, words, lang, color=None):
    """A destination this demo's robot can actually put something into. ADDED IN THIS COPY.

    ``sample_place`` also returns table, left, right and "other" (a shelf, a sofa, the
    floor). Those make fine sentences and are right for a command model in general, but
    this demo's arm can only drop into a receptacle, so training on them would teach the
    model to ask for placements the executor must then refuse. Restricted to what hollow.py
    can build.
    """
    return _coloured_place(rng, words, lang, color, kinds=("box", "tray", "basket"))


def _sort_into_matching(g, rng, words, lang) -> Sample:
    """"Put the red ones in the red box and the blue ones in the blue tray", as a Sample.

    ADDED IN THIS COPY. This is the sentence demo 03's schema could not express. Each group
    becomes an item whose ``dest`` is a receptacle of the SAME colour, which is what lets a
    downstream executor sort by colour without the model having to know what "matching"
    means.

    Only two groups, and only with colours the demo can actually see. Perception in this
    repository segments red, green and blue by default; orange and pink were measured as
    not separable from yellow and red respectively, so a sort sentence about them would
    train the model toward a sort the robot cannot perform.
    """
    seen = rng.sample(SORTABLE_COLORS, 2)
    items, dests = [], []
    for color in seen:
        it = sample_item(rng, words, lang, allow_color_only=False)
        it.quantity, it.position, it.color = "plural", None, color
        dest = _executable_place(rng, words, lang, color)
        it.dest = dest
        dests.append(dest)
        items.append((it, g.item(it)))
    # Render the destinations ONCE, then read the span words back out of what was written.
    # place() uses the module-level random for some of its choices, so calling it twice for
    # the same Place can return two different strings.
    rendered = [g.place(dest) for dest in dests]
    for (_it, ph), dest, text in zip(items, dests, rendered):
        ph.spans.update(g.place_spans(dest, text))
    text = rng.choice(g.SORT_INTO).format(
        g1=items[0][1].text, p1=rendered[0],
        g2=items[1][1].text, p2=rendered[1],
        c1=items[0][0].color, c2=items[1][0].color,
    )
    return Sample(lang, "sort", text, "sort/into_matching", items, None, "color")


def _each_own_place(g, rng, words, lang) -> Sample:
    """"Put the red ball in the blue box and the green cube in the red tray", as a Sample.

    ADDED IN THIS COPY. Same mechanism as ``_sort_into_matching`` but two individual items
    rather than two colour groups, so the multi-destination form is trained on
    ``sequence`` too and not only on ``sort``.
    """
    items, dests = [], []
    for _ in range(2):
        it = sample_item(rng, words, lang)
        dest = _executable_place(rng, words, lang)
        it.dest = dest
        dests.append(dest)
        items.append((it, g.item(it)))
    rendered = [g.place(dest) for dest in dests]
    for (_it, ph), dest, text in zip(items, dests, rendered):
        ph.spans.update(g.place_spans(dest, text))
    text = rng.choice(g.EACH_OWN_PLACE).format(
        x=items[0][1].text, p=rendered[0],
        x2=items[1][1].text, p2=rendered[1],
    )
    return Sample(lang, "sequence", text, "sequence/each_own_place", items, None)


def build(intent: str, lang: str, rng: random.Random, words: dict) -> Sample:
    g = GRAMMARS[lang]
    r = rng.random
    joiner = {"es": " y ", "en": " and ", "eu": " eta "}[lang]

    def item(**kwargs):
        it = sample_item(rng, words, lang, **kwargs)
        return it, g.item(it)

    if intent == "pick":
        it, ph = item()
        if r() < 0.06:  # self-correction: only the final item counts
            _, wrong = item(singular=True)
            text = rng.choice(g.CORRECTION).format(a=f"{rng.choice(g.PICK)} {wrong.text}", b=ph.text)
            return Sample(lang, intent, text, "pick/correction", [(it, ph)])
        if r() < 0.45:
            return Sample(lang, intent, rng.choice(g.PICK_ASK).format(x=ph.text), "pick/ask", [(it, ph)])
        return Sample(lang, intent, f"{rng.choice(g.PICK)} {ph.text}", "pick/imperative", [(it, ph)])

    if intent == "pick_and_place":
        it, ph = item()
        place = sample_place(rng, words, lang)
        p = g.place(place)
        roll = r()
        if lang == "eu":
            put = rng.choice(g.PUT_CONTAINER if place.kind in ("basket", "box", "tray") and r() < 0.5 else g.PUT)
            if roll < 0.35:
                text, template = f"{rng.choice(g.PICK)} {ph.text} eta {put} {p}", "pick_place/pick_and_put"
            elif roll < 0.55:
                text, template = f"{ph.text} hartu eta {put} {p}", "pick_place/object_first"
            elif roll < 0.8:
                text, template = f"{put} {ph.text} {p}", "pick_place/put"
            elif roll < 0.9:
                text, template = rng.choice(g.PUT_ASK).format(x=ph.text, p=p), "pick_place/ask"
            else:
                text, template = f"{ph.text} {p} {put}", "pick_place/verb_last"
        else:
            put = rng.choice(g.PUT)
            if roll < 0.4:
                then = f"{rng.choice(g.PUT_PRONOUN)}{g.pronoun(it)}" if lang == "es" else f"{put} {g.pronoun(it)}"
                text, template = f"{rng.choice(g.PICK)} {ph.text}{joiner}{then} {p}", "pick_place/pick_and_put"
            elif roll < 0.75:
                text, template = f"{put} {ph.text} {p}", "pick_place/put"
            elif roll < 0.88:
                text, template = rng.choice(g.PUT_ASK).format(x=ph.text, p=p), "pick_place/ask"
            else:
                text, template = f"{put} {p} {ph.text}", "pick_place/place_first"
        if r() < 0.05:  # self-correction of the object: only the final one counts
            _, wrong = item(singular=True)
            text = rng.choice(g.CORRECTION).format(a=f"{rng.choice(g.PICK)} {wrong.text}", b=text)
            template += "+correction"
        return Sample(lang, intent, text, template, [(it, ph)], place)

    if intent == "give":
        it, ph = item()
        text = rng.choice(g.GIVE).format(x=ph.text)
        # ADDED IN THIS COPY. Demo 03 forced place=None for every `give`, so "give it to me
        # and leave it in the tray" was inexpressible. The design guide actually asks for
        # this: docs/guia-so-101-autonomous-agent.md:1686 - "La entrega directa en la mano
        # debe evitarse salvo que se hayan definido medidas de seguridad. Para el hackathon
        # es mas seguro dejarlo en una bandeja." So most givens now name a tray, and the
        # bare form is kept so the model still recognises it.
        place = None
        if r() < 0.7:
            place = Place("tray", rng.choice(_place_entries(words, lang, "tray")), None)
            then = {"es": " y déjalo ", "en": " and leave it ", "eu": " eta utzi "}[lang]
            # Only from the non-question forms. Some GIVE entries end in "?" ("¿me pasas el
            # naranja?"), and appending " y déjalo en la bandeja" after a question mark
            # reads as two unrelated utterances, which is exactly the kind of noise that
            # teaches a model to ignore half a sentence.
            bank = [g for g in g.GIVE if not g.rstrip().endswith("?")] or list(g.GIVE)
            text = rng.choice(bank).format(x=ph.text) + then + g.place(place)
            return Sample(lang, intent, text, "give/place", [(it, ph)], place)
        return Sample(lang, intent, text, "give", [(it, ph)])

    if intent == "sort":
        # ADDED IN THIS COPY: the two-colour, two-matching-receptacle form. Demo 03's sort
        # forced `place.color = None` a few lines below, so every one of its 1500 sort
        # sentences was trained with an uncoloured destination and "the red ones in the red
        # box" was not in the data at all. About a third of sorts are now this form.
        if r() < 0.34:
            return _sort_into_matching(g, rng, words, lang)
        by_color = r() < 0.65
        templates = g.SORT_COLOR if by_color else g.SORT_KIND
        template = rng.choice(templates)
        items = []
        if "{g}" in template and r() < 0.4:  # a named group: "separa las bolas por colores"
            it = sample_item(rng, words, lang, allow_color_only=False)
            it.quantity, it.color, it.position = "plural", None, None
            ph = g.item(it)
            group, items = ph.text, [(it, ph)]
        else:
            group = rng.choice(g.GROUPS)
        text = template.format(g=group)
        place = None
        if r() < 0.25:
            place = sample_place(rng, words, lang, usable_only=True)
            place.color = None
            then = {"es": " y ponlas ", "en": " and put them ", "eu": " eta jarri "}[lang]
            text += then + g.place(place)
        return Sample(lang, intent, text, f"sort/{'color' if by_color else 'kind'}", items, place,
                      "color" if by_color else "kind")

    if intent == "clear":
        # ADDED IN THIS COPY: an item filter, so "clear only the red things into the box"
        # is expressible. Demo 03's clear took no items at all.
        if r() < 0.25:
            it, ph = item()
            it.quantity = "plural"
            ph = g.item(it)
            place = _executable_place(rng, words, lang)
            text = {"es": f"quita {ph.text} de la mesa y ponlos {g.place(place)}",
                    "en": f"take {ph.text} off the table and put them {g.place(place)}",
                    "eu": f"kendu {ph.text} mahaitik eta jarri {g.place(place)}"}[lang]
            return Sample(lang, intent, text, "clear/filter", [(it, ph)], place)
        if r() < 0.45:
            place = sample_place(rng, words, lang, usable_only=True)
            return Sample(lang, intent, rng.choice(g.CLEAR_PLACE).format(p=g.place(place)), "clear/place", [], place)
        return Sample(lang, intent, rng.choice(g.CLEAR), "clear")

    if intent == "sequence":
        # ADDED IN THIS COPY: two items, each to its own destination.
        if r() < 0.22:
            return _each_own_place(g, rng, words, lang)
        if r() < 0.2:  # colour sequence: "rojo, azul, rojo"
            items = []
            for _ in range(rng.choice([2, 3, 3, 4])):
                c = pick_color(rng, 1.0)
                word = {"es": VOCAB["colors"][c]["es"][0], "en": c, "eu": VOCAB["colors"][c]["eu"]["sg"]}[lang]
                items.append((Item("any", None, c, None), Phrase(word, {"color_word": word})))
            words_ = [ph.text for _, ph in items]
            lead = {"es": ["", "coge ", "pon ", "quiero "], "en": ["", "pick ", "take ", "get "],
                    "eu": ["", "hartu ", "jarri "]}[lang]
            if r() < 0.6:
                text = rng.choice(lead) + ", ".join(words_[:-1]) + joiner + words_[-1]
            else:
                text = ", ".join(words_)
            return Sample(lang, intent, text, "sequence/colors", items)
        items = [item() for _ in range(rng.choice([2, 2, 2, 3]))]
        phrases = [ph.text for _, ph in items]
        body = ", ".join(phrases[:-1]) + joiner + phrases[-1]
        roll = r()
        if roll < 0.5:
            text, template = f"{rng.choice(g.PICK)} {body}", "sequence/list"
        elif roll < 0.8:
            first, then = {"es": ("primero", "luego"), "en": ("first", "then"), "eu": ("lehenengo", "gero")}[lang]
            text = f"{first} {rng.choice(g.PICK)} {phrases[0]}" + "".join(f"{joiner}{then} {p}" for p in phrases[1:])
            template = "sequence/first_then"
        else:
            text, template = rng.choice(g.GIVE).format(x=body), "sequence/give"
        place = None
        if r() < 0.4:
            place = sample_place(rng, words, lang, usable_only=True)
            text += {"es": " y mételos ", "en": " and put them ", "eu": " eta sartu "}[lang] + g.place(place)
        return Sample(lang, intent, text, template, items, place)

    if intent in ("stop", "repeat", "cancel", "answer_yes", "answer_no"):
        bank = {"stop": g.STOP, "repeat": g.REPEAT, "cancel": g.CANCEL, "answer_yes": g.YES, "answer_no": g.NO}[intent]
        return Sample(lang, intent, rng.choice(bank), intent)

    if intent == "answer_value":  # a reply to "which one?" or "where?"
        roll = r()
        if roll < 0.5:
            it, ph = item(singular=r() < 0.8)
            return Sample(lang, intent, ph.text, "answer/item", [(it, ph)])
        place = sample_place(rng, words, lang)
        if roll < 0.8:
            return Sample(lang, intent, g.place(place), "answer/place", [], place)
        it, ph = item(singular=True)
        return Sample(lang, intent, f"{ph.text} {g.place(place)}", "answer/item_place", [(it, ph)], place)

    if intent == "question":
        if r() < 0.5:
            return Sample(lang, intent, rng.choice(g.QUESTIONS), "question/general")
        template = rng.choice(g.QUESTION_ITEM)
        it = sample_item(rng, words, lang, allow_color_only=False, singular=True)
        if "{plural}" in template:
            it.quantity, it.color, it.position = "plural", None, None
            name = g.plural_noun(it)
            return Sample(lang, intent, template.format(plural=name), "question/count", [(it, Phrase(name, {"name": name}))])
        ph = g.item(it)
        return Sample(lang, intent, template.format(x=ph.text), "question/item", [(it, ph)])

    if intent == "greeting":
        return Sample(lang, intent, rng.choice(g.GREETINGS), "greeting")

    if intent == "off_topic":
        return Sample(lang, intent, rng.choice(g.OFF_TOPIC), "off_topic")

    if intent == "not_doable":
        roll = r()
        if roll < 0.25:  # something a robot arm cannot or must not pick up
            return Sample(lang, intent, f"{rng.choice(g.PICK)} {rng.choice(g.IMPOSSIBLE)}", "not_doable/impossible")
        if roll < 0.4:
            return Sample(lang, intent, rng.choice(g.PEOPLE), "not_doable/people")
        if roll < 0.55:
            return Sample(lang, intent, rng.choice(g.NOTHING), "not_doable/nothing")
        it, ph = item(singular=True)
        bank, template = (g.DESTROY, "not_doable/destroy") if roll < 0.78 else (g.NOT_NOW, "not_doable/not_now")
        return Sample(lang, intent, rng.choice(bank).format(x=ph.text), template, [(it, ph)])

    raise ValueError(intent)


# --- surface variation: what speech recognition and people do to text ---------------

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def vary(sample: Sample, rng: random.Random) -> Sample:
    g = GRAMMARS[sample.lang]
    text = sample.text
    if sample.intent not in ("answer_yes", "answer_no", "stop") and rng.random() < 0.12:
        text = rng.choice(g.FILLERS) + text
    if sample.intent in ("pick", "pick_and_place", "give", "sequence", "sort", "clear") and rng.random() < 0.1:
        text = text.rstrip("?!.") + rng.choice(g.POLITE_END)
    roll = rng.random()
    if roll < 0.35:
        text = text[0].upper() + text[1:]
    elif roll < 0.45:
        text = text.lower()
    if rng.random() < 0.3 and text[-1] not in "?!.":
        text += rng.choice([".", "!", "."])
    if rng.random() < 0.25:  # speech recognition often drops punctuation entirely
        text = text.strip("¿¡").rstrip("?!.")
    sample.text = text
    if sample.lang == "es" and rng.random() < 0.15:  # and accents
        sample.text = strip_accents(sample.text)
        for _, phrase in sample.items:
            phrase.text = strip_accents(phrase.text)
            phrase.spans = {k: strip_accents(v) for k, v in phrase.spans.items()}
    return sample


# --- records -------------------------------------------------------------------------

def classification(task, labels, value):
    return {"task": task, "labels": labels, "true_label": [value]}


def command_of(sample: Sample) -> dict:
    """The clean command this sentence means: what the robot should receive."""
    def number(item: Item):
        return 1 if item.quantity in (None, 1) else "all" if item.quantity in ("all", "plural") else item.quantity

    return {
        "intent": sample.intent,
        "items": [{"kind": it.kind, "name": ph.spans.get("name"), "color": it.color, "quantity": number(it),
                   "position": it.position,
                   # ADDED IN THIS COPY. The per-item destination, when the sentence gave
                   # one. A consumer falls back to the command-level ``place`` when this is
                   # null, which is every item of every command that has one shared
                   # destination.
                   "place": ({"kind": it.dest.kind, "color": it.dest.color} if it.dest else None)}
                  for it, ph in sample.items],
        "place": {"kind": sample.place.kind, "color": sample.place.color} if sample.place else None,
        "sort_by": sample.sort_by,
    }


def to_records(sample: Sample) -> tuple[dict, list[dict]]:
    text_lower = sample.text.lower()
    structures = []
    for it, ph in sample.items:
        record = {"mention": ph.text, **{k: v for k, v in ph.spans.items() if v}}
        for key, value in record.items():  # every labelled word must really be in the sentence
            assert value.lower() in text_lower, (sample.template, key, value, sample.text)
        structures.append({"item": record})
    place = sample.place
    command_record = {
        "input": sample.text,
        "output": {
            "classifications": [
                classification("intent", INTENTS, sample.intent),
                classification("place_kind", PLACE_KINDS, place.kind if place else "none"),
                classification("place_color", COLORS + ["none"], place.color if place and place.color else "none"),
                classification("sort_by", SORT_BY, sample.sort_by),
            ],
            **({"json_structures": structures} if structures else {}),
        },
        "meta": {"lang": sample.lang, "template": sample.template, "command": command_of(sample)},
    }
    kind_records = [{
        "input": ph.text,
        "output": {"classifications": [classification("item_kind", ITEM_KINDS, it.kind),
                                       classification("item_count", ITEM_COUNTS, "several" if it.several else "one")]},
        "meta": {"lang": sample.lang, "template": "item_phrase"},
    } for it, ph in sample.items]
    return command_record, kind_records


def generate(per_language: int, seed: int, split: str = "train", intents=None) -> tuple[list, list]:
    rng = random.Random(seed)
    random.seed(seed)  # the grammar classes use the module-level generator for small choices
    words = vocab_split(split)
    mix = {k: v for k, v in MIX.items() if intents is None or k in intents}
    total = sum(mix.values())
    commands, kinds = [], []
    for lang in LANGS:
        for intent, share in mix.items():
            for _ in range(round(per_language * share / total)):
                sample = vary(build(intent, lang, rng, words), rng)
                command_record, kind_records = to_records(sample)
                commands.append(command_record)
                kinds.extend(kind_records)
    rng.shuffle(commands)
    # item phrases repeat a lot ("la bola roja"): keep each (phrase, answers) pair once
    unique = {(k["input"].lower(), json.dumps(k["output"])): k for k in kinds}
    kinds = list(unique.values())
    rng.shuffle(kinds)
    return commands, kinds


def write_jsonl(path: Path, records: list) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(path: Path, commands: list, kinds: list, unseen: list) -> None:
    counts = collections.Counter((r["meta"]["lang"], r["meta"]["command"]["intent"]) for r in commands)
    unique_texts = len({r["input"].lower() for r in commands})
    lines = ["# Dataset statistics", "", f"- training sentences: {len(commands)} ({unique_texts} different)",
             f"- training item phrases: {len(kinds)}", f"- unseen-word test sentences: {len(unseen)}", "",
             "| intent | " + " | ".join(LANGS) + " |", "| --- |" + " --- |" * len(LANGS)]
    for intent in INTENTS:
        lines.append(f"| {intent} | " + " | ".join(str(counts[(lang, intent)]) for lang in LANGS) + " |")
    lines += ["", "## Examples", ""]
    shown = collections.defaultdict(int)
    for record in commands:
        key = (record["meta"]["lang"], record["meta"]["command"]["intent"])
        if shown[key] < 2:
            shown[key] += 1
            command = json.dumps(record["meta"]["command"], ensure_ascii=False)
            lines.append(f"- `{key[0]}` **{key[1]}**: {record['input']}  \n  `{command}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=10000, help="Training sentences per language")
    parser.add_argument("--show", type=int, default=0, help="Print this many samples and write nothing")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DATA,
                        help="where to write the generated train/eval sets")
    args = parser.parse_args()

    if args.show:
        commands, _ = generate(max(args.show, 200), args.seed)
        for record in commands[:args.show]:
            items = [s["item"] for s in record["output"].get("json_structures", [])]
            labels = {c["task"]: c["true_label"][0] for c in record["output"]["classifications"]}
            print(f"[{record['meta']['lang']}] {record['input']!r}\n      {labels}\n      items={items}")
        return 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    commands, kinds = generate(args.per_language, args.seed)
    eval_commands, eval_kinds = generate(max(100, args.per_language // 20), args.seed + 1)
    # unseen words, only for intents that name objects or places
    unseen, _ = generate(max(60, args.per_language // 20), args.seed + 2, split="test",
                         intents={"pick", "pick_and_place", "give", "sequence", "answer_value", "question"})
    write_jsonl(output_dir / "train_commands.jsonl", commands)
    write_jsonl(output_dir / "train_item_kinds.jsonl", kinds)
    write_jsonl(output_dir / "eval_commands.jsonl", eval_commands)
    write_jsonl(output_dir / "eval_item_kinds.jsonl", eval_kinds)
    write_jsonl(output_dir / "test_unseen.jsonl", unseen)
    write_stats(output_dir / "stats.md", commands, kinds, unseen)
    print(f"wrote {len(commands)} sentences, {len(kinds)} item phrases, {len(eval_commands)} eval sentences, "
          f"{len(unseen)} unseen-word sentences to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
