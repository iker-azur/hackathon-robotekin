"""Say what is on the table, in the language the person spoke.

Demo 03 ends with a fixed answer for the ``question`` intent ("todavía no puedo ver
la mesa"), and its own README names the camera demo as the thing that replaces it.
This module is that replacement: "¿qué ves?" becomes "Veo dos cubos azules y una
bola roja", built from the scene JSON and nothing else.

Doing it properly means agreeing with the grammar of each language rather than
translating a Spanish list:

* Spanish adjectives agree in gender and number with the noun: *dos cubos azules*,
  not *dos cubos azul*. Gender is fixed per noun, so the noun carries it.
* English puts the adjective first: *two blue cubes*.
* Basque puts the number first and then uses the bare form of everything after it,
  because a numeral already means plural: *bi kubo urdin*, never *bi kubo urdinez* or
  *bi kubo urdinak*. With one, the sentence-final ``bat`` and the noun go together.

The counts come from grouping the scene by kind and colour, which is also what tells
the listener how many things there are without counting them.
"""

from __future__ import annotations

from collections import Counter

from scene import Scene, Thing

# kind -> language -> (singular, plural). Spanish gender is a fourth field, which is
# what the colour adjective has to agree with.
NOUNS: dict[str, dict[str, tuple]] = {
    "ball": {"es": ("bola", "bolas", "f"), "en": ("ball", "balls"), "eu": ("bola", "bolak")},
    "cube": {"es": ("cubo", "cubos", "m"), "en": ("cube", "cubes"), "eu": ("kubo", "kuboak")},
    # The receptacle. Perception calls a container "box" (perceive.py: aspect ratio decides), so
    # without this a destination came out as "objeto rojo" - "a red object" - which is both vague
    # and wrong about gender, since the tray is what things go *into*. "cesta" is the word a
    # person uses for it at a bench; feminine, so the adjective agrees with it: "cesta roja".
    "box": {"es": ("cesta", "cestas", "f"), "en": ("box", "boxes"), "eu": ("kasa", "kasak")},
    "any": {"es": ("objeto", "objetos", "m"), "en": ("object", "objects"), "eu": ("objektu", "objektuak")},
    "other": {"es": ("objeto", "objetos", "m"), "en": ("object", "objects"), "eu": ("objektu", "objektuak")},
}

# colour -> language -> forms. Spanish carries four because the adjective agrees with
# the noun in gender and number: (masc sg, fem sg, masc pl, fem pl). Basque carries two
# (bare, with article) because a numeral already means plural and takes the bare form.
COLORS: dict[str, dict[str, tuple]] = {
    "red": {"es": ("rojo", "roja", "rojos", "rojas"), "en": ("red",), "eu": ("gorri", "gorria")},
    "green": {"es": ("verde", "verde", "verdes", "verdes"), "en": ("green",), "eu": ("berde", "berdea")},
    "blue": {"es": ("azul", "azul", "azules", "azules"), "en": ("blue",), "eu": ("urdin", "urdina")},
    "yellow": {"es": ("amarillo", "amarilla", "amarillos", "amarillas"), "en": ("yellow",), "eu": ("hori", "horia")},
    "white": {"es": ("blanco", "blanca", "blancos", "blancas"), "en": ("white",), "eu": ("zuri", "zuria")},
    "black": {"es": ("negro", "negra", "negros", "negras"), "en": ("black",), "eu": ("beltz", "beltza")},
    "purple": {"es": ("morado", "morada", "morados", "moradas"), "en": ("purple",), "eu": ("more", "morea")},
}

NUMBERS = {
    "es": {1: "un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis"},
    "en": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"},
    "eu": {1: "bat", 2: "bi", 3: "hiru", 4: "lau", 5: "bost", 6: "sei"},
}

JOINER = {"es": " y ", "en": " and ", "eu": " eta "}
SEVERAL = {"es": "varios", "en": "several", "eu": "batzu"}


def groups(scene: Scene) -> list[tuple[str, str, int]]:
    """The scene's objects as (kind, colour, count), most numerous first.

    ``Counter.items()`` yields ``((kind, colour), count)``, so the key is unpacked as a
    pair and flattened here. Sorted by count then by name so that the sentence is the
    same every time for the same scene, which is what makes it usable as an expected
    value in a test.
    """
    counter: Counter = Counter((thing.kind, thing.color) for thing in scene.objects)
    counts = [(kind, color, count) for (kind, color), count in counter.items()]
    return sorted(counts, key=lambda row: (-row[2], row[0], row[1]))


def phrase(kind: str, color: str | None, count: int, lang: str, bare: bool = False) -> str:
    """One noun phrase with the right agreement for the language and the count.

    ``bare`` drops the determiner, for use straight after a preposition that already
    carries one: "I cannot see any yellow cube" needs "yellow cube", not "a yellow cube".

    Also the entry point ``ground.py`` uses for its questions, so that "there are two
    red balls" never comes out as "there are two ball reds".
    """
    nouns = NOUNS.get(kind, NOUNS["any"])[lang]
    many = count != 1
    if not color:
        colour = ()
    else:
        colour = COLORS.get(color, {}).get(lang, (color,) * 4)

    if lang == "es":
        feminine = nouns[2] == "f"
        if many:
            word = (colour[3] if feminine else colour[2]) if colour else ""
            return " ".join(filter(None, [NUMBERS["es"].get(count, SEVERAL["es"]), nouns[1], word]))
        word = (colour[1] if feminine else colour[0]) if colour else ""
        return " ".join(filter(None, ["" if bare else ("una" if feminine else "un"), nouns[0], word]))
    if lang == "en":
        number = "" if bare else NUMBERS["en"].get(count, SEVERAL["en"])
        noun = nouns[1] if many else nouns[0]
        return " ".join(filter(None, [number, colour[0] if colour else "", noun]))
    # Basque: a numeral already carries the plural, so everything after it is bare and
    # the numeral itself goes last, after the noun it counts.
    if many:
        return " ".join(filter(None, [NUMBERS["eu"].get(count, SEVERAL["eu"]), nouns[0],
                                       colour[0] if colour else ""]))
    return " ".join(filter(None, [nouns[0], colour[1] if colour else "",
                                  "" if bare else NUMBERS["eu"][1]]))


def _join(parts: list[str], lang: str) -> str:
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + JOINER[lang] + parts[-1]


def what_i_see(scene: Scene, lang: str = "es") -> str:
    """The answer to "¿qué ves?" / "what do you see?" / "zer ikusten duzu?"."""
    counted = groups(scene)
    if not counted:
        return {
            "es": "No veo nada en la mesa.",
            "en": "I cannot see anything on the table.",
            "eu": "Ez ikusten dut ezer mahaian.",
        }[lang]
    parts = [phrase(kind, color, count, lang) for kind, color, count in counted]
    return {
        "es": f"Veo {_join(parts, 'es')}.",
        "en": f"I see {_join(parts, 'en')}.",
        "eu": f"Ikusten dut {_join(parts, 'eu')}.",
    }[lang]


def how_many(scene: Scene, lang: str = "es") -> str:
    """The answer to "¿cuántos hay?" without naming what they are."""
    parts = [phrase(kind, color, count, lang) for kind, color, count in groups(scene)]
    if not parts:
        return {"es": "Ninguno.", "en": "None.", "eu": "Bat ere ez."}[lang]
    return {"es": f"Hay {_join(parts, 'es')}.", "en": f"There are {_join(parts, 'en')}.",
            "eu": f"{_join(parts, 'eu')} daude."}[lang]


def where_is(scene: Scene, thing: Thing, lang: str = "es") -> str:
    """The answer to "¿dónde está el cubo amarillo?", from the scene's own frame.

    Only says left, right, near or far, never a coordinate: the person asked where it
    is, not how many metres from the world origin, and these are the four words they
    would have used to ask.
    """
    matches = [o for o in scene.objects if o.kind == thing.kind and o.color == thing.color]
    if len(matches) < 2:
        return {"es": "Está sobre la mesa.", "en": "It is on the table.",
                "eu": "Mahaian dago."}[lang]
    best = {
        "left": scene.rank(matches, "left")[0].id,
        "right": scene.rank(matches, "right")[0].id,
        "nearest": scene.rank(matches, "nearest")[0].id,
        "farthest": scene.rank(matches, "farthest")[0].id,
    }
    words = {
        "left": {"es": "a la izquierda", "en": "on the left", "eu": "ezkerrean"},
        "right": {"es": "a la derecha", "en": "on the right", "eu": "eskuinean"},
        "nearest": {"es": "más cerca de mí", "en": "closest to me", "eu": "niretik hurbilen"},
        "farthest": {"es": "más lejos de mí", "en": "furthest from me", "eu": "niretik urrunen"},
    }
    where = next((words[k][lang] for k, id_ in best.items() if id_ == thing.id), None)
    if where is None:
        where = {"es": "en el medio", "en": "in the middle", "eu": "erdian"}[lang]
    return {"es": f"Está {where}.", "en": f"It is {where}.", "eu": f"{where.capitalize()} dago."}[lang]


def answer(scene: Scene, command: dict, lang: str = "es") -> str:
    """The reply for a ``question`` intent, choosing the question that was asked.

    Demo 03's ``question`` intent covers three things that this demo can now answer
    differently: what is on the table, how many things there are, and where one of
    them is. Which one it was is read from the item the model extracted, because the
    command model already puts the object there for "¿dónde está el cubo amarillo?".
    """
    intent_text = (command.get("items") or [{}])[0].get("name") or ""
    words = set(intent_text.lower().split())
    if words & {"cuantos", "cuantas", "quantos", "quantas", "zenbat", "how", "many"}:
        return how_many(scene, lang)
    if words & {"donde", "where", "non", "nonko"}:
        named = _find(scene, command)
        if named is not None:
            return where_is(scene, named, lang)
    return what_i_see(scene, lang)


def _find(scene: Scene, command: dict) -> Thing | None:
    for item in command.get("items", []):
        found = scene.candidates(item.get("kind"), item.get("color"))
        if len(found) == 1:
            return found[0]
    return None
