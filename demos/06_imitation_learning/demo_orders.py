"""Spoken orders -> which piece goes to which box, for the pick-classify-sort demo (``30_full_demo_app.py``).

The intent model of ``demos/10_bin_intent_model`` (``IkerAzurmendi/bin-intent-gliner2``) reads one sentence and says,
with a probability per option, which pieces (``feature`` bolt / no bolt, ``quality`` good / bad), which box (big /
white) and how many (``count``). This module keeps the standing orders those readings produce and answers one
question for the robot: the piece in the gripper is PIN or NO PIN, where does it go?

The cell calls its boxes green (big) and white (small); the model only knows "big" and "white" and reads "green" or
"small" as a box this cell does not have, so ``cell_vocabulary`` rewrites those words before the sentence is read.
In this cell a piece with a pin is a good piece and a piece without one is a bad piece (``good_has_pin``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

PIN, NO_PIN = "PIN", "NO PIN"  # the labels of ``24_pick_and_classify_app.label``
BOXES = {"big": "green", "white": "white"}  # the model's boxes -> the destinations of policy_routes.json
HOLD = "hold"  # "pick the piece": no destination, keep it in the gripper
QUANTITIES = {"one": 1, "two": 2, "three": 3}

_SPANISH_SIZE = {"pequeño": "blanco", "pequeña": "blanca", "pequeños": "blancos", "pequeñas": "blancas"}
_WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)


def cell_vocabulary(text: str) -> str:
    """Say "big" for green and "white" for small, in Spanish, English and Basque, as the model knows the boxes."""

    def word(match):
        said = match.group(0)
        low = said.lower()
        if low == "green":
            return "big"
        if low in ("small", "little"):
            return "white"
        if low in ("verde", "verdes"):
            return "grande" + low[5:]
        if low in _SPANISH_SIZE:
            return _SPANISH_SIZE[low]
        if low.startswith("berde"):
            return "handi" + low[5:]
        if low.startswith("txiki"):
            return "zuri" + low[5:]
        return said

    return _WORDS.sub(word, text)


# Party mode: the words that start it, whatever else the sentence says. Spanish and English say fiesta / party; Basque
# says festa (fiesta, party and parranda are also used), jai (a celebration) and farra, which take case endings
# (festan, festara, jaia, jaietan), so those are matched by stem. Whole words only: "festival" is not a party order.
_PARTY = re.compile(r"\b(?:party|parties|partys|fiestas?|festa\w*|jai|jaia\w*|jaiet\w*|parranda\w*|farra(?:k|n|ra|z|rekin)?)\b")


def is_party(text: str) -> bool:
    """Does the sentence ask for the party mode? Case and accents do not matter."""
    plain = unicodedata.normalize("NFKD", text.lower())
    return bool(_PARTY.search("".join(c for c in plain if not unicodedata.combining(c))))


MESSAGES = {
    "set": {"es": "Orden entendida.", "en": "Order understood.", "eu": "Agindua ulertu dut."},
    "stop": {"es": "Parado: no hay ninguna orden activa.", "en": "Stopped: no order is active.",
             "eu": "Geldituta: ez dago agindu aktiborik."},
    "ask_box": {"es": "¿A qué caja? Di la caja grande (verde) o la blanca.",
                "en": "Which box? Say the big (green) box or the white one.",
                "eu": "Zein kutxara? Esan kutxa handia (berdea) edo zuria."},
    "unsure": {"es": "No estoy seguro de la orden; repítela, por favor.",
               "en": "I am not sure about the order; please say it again.",
               "eu": "Ez nago ziur aginduaz; errepikatu, mesedez."},
    "no_action": {"es": "No he entendido una orden de manipulación.", "en": "I did not hear a handling order.",
                  "eu": "Ez dut manipulazio agindurik ulertu."},
    "not_a_cell_box": {"es": "Esa caja no está en esta célula: solo hay la grande (verde) y la blanca.",
                       "en": "That box is not in this cell: there is only the big (green) and the white one.",
                       "eu": "Kutxa hori ez dago zelula honetan: handia (berdea) eta zuria bakarrik daude."},
    "party": {"es": "¡Fiesta! Subo despacio a la posición de baile. Di «para» (o pulsa S) para terminar.",
              "en": "Party time! I am rising slowly to the dance pose. Say \"stop\" (or press S) to finish.",
              "eu": "Festa! Mantso igoko naiz dantza-posiziora. Esan «gelditu» (edo sakatu S) amaitzeko."},
    "party_busy": {"es": "Ahora no puedo bailar: termina la pieza en curso (S para parar, R para reanudar) y pídelo otra vez.",
                   "en": "I cannot dance right now: finish the current piece (S to stop, R to resume) and ask again.",
                   "eu": "Orain ezin dut dantzan egin: amaitu uneko pieza (S gelditzeko, R jarraitzeko) eta eskatu berriro."},
    "contradiction": {"es": "La orden se contradice (por ejemplo, buenas y sin perno); repítela.",
                      "en": "The order contradicts itself (for example good and without a pin); say it again.",
                      "eu": "Aginduak kontraesana dauka; errepikatu, mesedez."},
}


def message(kind: str, language: str | None) -> str:
    return MESSAGES[kind].get(language or "es", MESSAGES[kind]["es"])


@dataclass
class Route:
    destination: str  # "green" | "white" | HOLD
    remaining: int | None  # pieces still to move; None = until told to stop
    group: int | None = None  # routes made by the same spoken rule share their quantity


@dataclass
class Outcome:
    kind: str  # set | stop | ask_box | unsure | no_action | not_a_cell_box | contradiction
    message: str
    routes: dict = field(default_factory=dict)


class OrderBook:
    """The standing orders. ``destination_for`` is what the robot consults after it classifies a piece."""

    def __init__(self, good_has_pin: bool = True) -> None:
        self.routes: dict[str, Route] = {}
        self.good_has_pin = good_has_pin
        self._groups = 0

    @property
    def active(self) -> bool:
        return bool(self.routes)

    def classes(self, feature: str, quality: str):
        """The piece classes a rule names, or None when feature and quality contradict each other."""
        by_feature = {"bolt": {PIN}, "no bolt": {NO_PIN}}.get(feature)
        good, bad = ({PIN}, {NO_PIN}) if self.good_has_pin else ({NO_PIN}, {PIN})
        by_quality = {"good": good, "bad": bad}.get(quality)
        named = [group for group in (by_feature, by_quality) if group]
        if not named:
            return {PIN, NO_PIN}
        merged = set.intersection(*named)
        return merged or None

    def apply(self, reading: dict, language: str | None = None, min_confidence: float = 0.85) -> Outcome:
        """Fold one intent reading (the dict ``hmi/intent_worker.py`` replies with) into the standing orders."""
        status = reading["status"]
        if status in ("stop", "cancel"):
            self.routes.clear()
            return Outcome("stop", message("stop", language))
        if status in ("ask_box", "not_a_cell_box", "no_action", "answer", "repeat"):
            kind = "no_action" if status in ("answer", "repeat") else status
            return Outcome(kind, message(kind, language))
        if status not in ("ok", "unsure") or not reading["rules"]:
            return Outcome("unsure", message("unsure", language))

        new: dict[str, Route] = {}
        for rule in reading["rules"]:
            # How sure the sentence was about the parts that decide the routing. The count only limits how many
            # pieces move; when none was said its probability is not part of the decision.
            parts = [rule["p_feature"], rule["p_quality"]]
            if rule["box_source"] == "said":
                parts.append(rule["p_box"])
            if rule["count"] != "none":
                parts.append(rule["p_count"])
            if min(parts) < min_confidence or reading["probabilities"]["intent"][reading["intent"]] < min_confidence:
                return Outcome("unsure", message("unsure", language))
            classes = self.classes(rule["feature"], rule["quality"])
            if classes is None:
                return Outcome("contradiction", message("contradiction", language))
            if rule["box"] in BOXES:
                destination = BOXES[rule["box"]]
            elif rule["box"] == "none" and reading["intent"] == "pick":
                destination = HOLD
            elif rule["box"] == "other":
                return Outcome("not_a_cell_box", message("not_a_cell_box", language))
            else:
                return Outcome("ask_box", message("ask_box", language))
            self._groups += 1
            for piece_class in classes:
                new[piece_class] = Route(destination, QUANTITIES.get(rule["count"]), self._groups)
        self.routes.update(new)
        return Outcome("set", message("set", language), dict(self.routes))

    def preset(self, name: str) -> None:
        """The usual orders without speaking: ``sort`` (good -> white, bad -> green), ``all_green`` or ``all_white``."""
        good, bad = (PIN, NO_PIN) if self.good_has_pin else (NO_PIN, PIN)
        routes = {"sort": {good: "white", bad: "green"},
                  "all_green": {PIN: "green", NO_PIN: "green"},
                  "all_white": {PIN: "white", NO_PIN: "white"}}[name]
        self.routes = {piece_class: Route(destination, None) for piece_class, destination in routes.items()}

    def destination_for(self, piece_class: str):
        route = self.routes.get(piece_class)
        return route.destination if route else None

    def completed(self, piece_class: str) -> None:
        """One piece of this class was delivered: count it against its rule and drop the rule when its quantity is
        done ("move two pieces to the green box" is two pieces in all, whichever class they turn out to be)."""
        route = self.routes.get(piece_class)
        if route is None or route.remaining is None:
            return
        siblings = [name for name, other in self.routes.items()
                    if other is route or (route.group is not None and other.group == route.group)]
        for name in siblings:
            self.routes[name].remaining -= 1
        for name in siblings:
            if self.routes[name].remaining <= 0:
                del self.routes[name]

    def clear(self) -> None:
        self.routes.clear()

    def describe(self) -> str:
        if not self.routes:
            return "Sin orden activa"
        names = {PIN: "CON perno (buena)" if self.good_has_pin else "CON perno (mala)",
                 NO_PIN: "SIN perno (mala)" if self.good_has_pin else "SIN perno (buena)"}
        boxes = {"green": "caja VERDE (grande)", "white": "caja BLANCA (pequeña)", HOLD: "solo recoger"}
        parts = []
        for piece_class in (PIN, NO_PIN):
            if piece_class in self.routes:
                route = self.routes[piece_class]
                left = "" if route.remaining is None else f" · quedan {route.remaining}"
                parts.append(f"{names[piece_class]} → {boxes[route.destination]}{left}")
        return "   |   ".join(parts)
