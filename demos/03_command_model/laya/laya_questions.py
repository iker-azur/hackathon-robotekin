"""The questions Laya answers about a spoken command, and their right answers.

Laya (https://github.com/NandhaKishorM/laya) is an open-source "System One"
decision model with the same question format as TypeSafe's Jev: a `state` (our
sentence) and typed questions (`choice`, `score`, `noul`), each answered with
probabilities in one forward pass. It cannot extract words from the text, so it
answers the sentence-level part of a command:

    intent        which of the 16 intents
    place_kind    where the object goes (basket / box / tray / table / left / right / other / none)
    place_color   the colour of that place
    sort_by       for sort commands: color / kind / none

The items (which objects, their colours and quantities) still come from the
GLiNER2 command model: see ../command_model.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import COLORS, INTENTS, PLACE_KINDS, SORT_BY, VOCAB  # noqa: E402

INTENT_CRITERIA = {
    "pick_and_place": "pick something up and put it somewhere (coge y mete / pick up and put / hartu eta sartu)",
    "pick": "pick something up, without saying where to put it",
    "give": "hand something to the person (pásame / hand me / emadazu)",
    "sort": "sort or separate the objects by colour or type (separa, clasifica / sort / sailkatu)",
    "clear": "clear the table or put everything away (recoge la mesa / clear the table / garbitu mahaia)",
    "sequence": "several objects one after another (una roja y dos azules / first ... then / lehenengo ... gero)",
    "stop": "stop moving immediately (para / stop / gelditu)",
    "repeat": "do the last action again (otra vez / again / berriro)",
    "cancel": "cancel or forget the order (cancela, olvídalo / never mind / ahaztu)",
    "answer_yes": "yes, confirming a question (sí / yes / bai)",
    "answer_no": "no, rejecting a question (no / no / ez)",
    "answer_value": "only an object or a place, as a reply to a question (la roja / the one on the left / kutxan)",
    "question": "a question about the robot or the scene (¿qué ves? / what do you see? / zer ikusten duzu?)",
    "greeting": "a greeting or small talk (hola / hello / kaixo)",
    "off_topic": "something unrelated to the robot (the weather, a joke, the time)",
    "not_doable": "something the robot must not or cannot do: people, huge things, breaking things, "
                  "negations, things in the past (coge el coche / don't touch anything / ez hartu ezer)",
}
PLACE_CRITERIA = {
    "basket": "a basket (cesto, cesta, bin, saski)",
    "box": "a box (caja, box, kutxa)",
    "tray": "a tray (bandeja, tray, erretilu)",
    "table": "the table (mesa, table, mahai)",
    "left": "the left side (izquierda, left, ezker)",
    "right": "the right side (derecha, right, eskuin)",
    "other": "another place the robot does not use, such as a shelf or the floor",
    "none": "no destination is said",
}
COLOR_CRITERIA = {c: f"{c} ({', '.join(sorted(set(VOCAB['colors'][c]['es'])))}, {VOCAB['colors'][c]['eu']['bare']})"
                  for c in COLORS}
COLOR_CRITERIA["none"] = "no colour is said for the destination"
SORT_CRITERIA = {"color": "sort by colour", "kind": "sort by type or shape", "none": "not a sorting order"}

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What does the person ask the robot arm to do?",
               "criteria": INTENT_CRITERIA},
    "place_kind": {"type": "choice", "instructions": "Where should the object be put?",
                   "criteria": PLACE_CRITERIA},
    "place_color": {"type": "choice", "instructions": "Which colour is the place where the object should go?",
                    "criteria": COLOR_CRITERIA},
    "sort_by": {"type": "choice", "instructions": "If this is an order to sort objects, by what?",
                "criteria": SORT_CRITERIA},
}
assert list(INTENT_CRITERIA) == INTENTS and list(PLACE_CRITERIA) == PLACE_KINDS and list(SORT_CRITERIA) == SORT_BY


def answers_of(command: dict) -> dict[str, str]:
    """The right option of each question for a command (the "command" in a data line's meta)."""
    place = command.get("place") or {}
    return {
        "intent": command["intent"],
        "place_kind": place.get("kind") or "none",
        "place_color": place.get("color") or "none",
        "sort_by": command.get("sort_by") or "none",
    }
