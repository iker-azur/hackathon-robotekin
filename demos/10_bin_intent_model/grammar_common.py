"""Shared pieces of the bin-sorting intent dataset: the label sets and the sampling context.

The model answers five questions about a sentence, each with a probability per option (Jev style):

    intent   what the person wants       move sort pick both reply ask stop cancel repeat yes no hello chat refuse
    feature  which pieces: bolt or not   bolt | no bolt | any
    quality  which pieces: good or bad   good | bad | any
    box      where they go               big | white | other | none
    count    how many                    one | two | three | all | none

``box`` is the physical pair of boxes of the cell: the big one and the white one. ``other`` is a box
the person described some other way (small, black, left, "the second one"): the robot does not have
it and must say so. ``none`` is no destination at all.

Every language module (``grammar_es.py``, ``grammar_en.py``, ``grammar_eu.py``) builds sentences from
word lists split in three tiers: ``train`` words, ``calib`` words (used to fit the probability
calibration) and ``test`` words (never seen in training, used to measure generalisation).
"""

from __future__ import annotations

import random
import re
import unicodedata

INTENTS = ["move", "sort", "pick", "both", "reply", "ask", "stop", "cancel", "repeat", "yes", "no",
           "hello", "chat", "refuse"]
FEATURES = ["bolt", "no bolt", "any"]
QUALITIES = ["good", "bad", "any"]
BOXES = ["big", "white", "other", "none"]
COUNTS = ["one", "two", "three", "all", "none"]
TASKS = {"intent": INTENTS, "feature": FEATURES, "quality": QUALITIES, "box": BOXES, "count": COUNTS}
ACTION_INTENTS = ("move", "sort", "pick")
LANGUAGES = ("es", "en", "eu")


def plain(text: str) -> str:
    """Lower case, no accents, no punctuation: the key used to tell two sentences apart."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text))


class Ctx:
    """Random source plus the word tier a sentence is being built from."""

    def __init__(self, rng: random.Random, tier: str = "train") -> None:
        self.rng, self.tier, self.novel = rng, tier, False

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def pick(self, slot):
        """One entry of a plain list, or of a ``{"train": [...], "calib": [...], "test": [...]}`` slot.

        Outside the train tier an entry of that tier is chosen half of the time and ``novel`` is set,
        so the generator can insist that an evaluation sentence holds at least one unseen word."""
        if isinstance(slot, (list, tuple)):
            return self.rng.choice(slot)
        extra = slot.get(self.tier) if self.tier != "train" else None
        if extra and self.chance(0.5):
            self.novel = True
            return self.rng.choice(extra)
        return self.rng.choice(slot["train"])

    def weighted(self, options: dict):
        return self.rng.choices(list(options), weights=list(options.values()))[0]


def upper_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text
