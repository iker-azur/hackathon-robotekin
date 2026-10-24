"""Interpret the persistent pin/no-pin sorting rule spoken to the HMI."""

from __future__ import annotations

import re
import unicodedata
import json
import threading
from pathlib import Path


PART = r"(?:perno|pasador|tornillo|bulon|pin)"
GREEN = r"(?:verde|verdes)"
WHITE = r"(?:blanco|blanca|blancos|blancas)"
GOOD = r"(?:buen[oa]s?|correct[oa]s?|valid[oa]s?|apt[oa]s?|aceptables?|aprobadas?|sirven)"
BAD = r"(?:mal[oa]s?|defectuos[oa]s?|incorrect[oa]s?|invalid[oa]s?|rechazad[oa]s?|inservibles?)"
WITH_PIN = re.compile(rf"\b(?:con|tiene|tienen|tenga|tengan|lleva|llevan|incluye|incluyen|porta|portan)\s+(?:(?:un|el)\s+)?{PART}\b")
WITHOUT_PIN = re.compile(rf"\b(?:sin|no\s+(?:tiene|tienen|tenga|tengan|lleva|llevan|incluye|incluyan)|carece\s+de|falta)\s+(?:(?:un|el)\s+)?{PART}\b")
IMPLICIT_WITHOUT_PIN = re.compile(r"\b(?:las?|aquellas)\s+que\s+no\s+(?:lo\s+)?(?:tenga|tengan|tiene|tienen|lleve|lleven|lleva|llevan)\b")
GOOD_WORD = re.compile(rf"\b{GOOD}\b")
BAD_WORD = re.compile(rf"\b{BAD}\b")
GREEN_WORD = re.compile(rf"\b{GREEN}\b")
WHITE_WORD = re.compile(rf"\b{WHITE}\b")
STOP = re.compile(
    r"^(?:(?:por favor)\s+)?(?:para(?:\s+ya)?|que\s+pare|parar|deten(?:te|er|ga)?|"
    r"cancela(?:r)?|anula(?:r)?|desactiva(?:r)?|termina(?:r)?|finaliza(?:r)?|stop|"
    r"deja\s+de\s+(?:clasificar|separar|ordenar))"
    r"(?:\s+(?:la|el|de|la regla|la clasificacion|el proceso|el sistema|la orden))*"
    r"(?:\s+por favor)?$"
)


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFD", text.casefold())
    plain = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def _nearest_kind(span, candidates, maximum=100):
    if not candidates:
        return None
    start, end = span
    distance, kind = min(
        (max(0, max(start - right, left - end)), label)
        for label, left, right in candidates
    )
    return kind if distance <= maximum else None


def _spans(pattern, label, text):
    return [(label, match.start(), match.end()) for match in pattern.finditer(text)]


def interpret_sorting_rule(text):
    """Return a stop action, a complete sorting rule, or a repeat-audio prompt."""
    if not isinstance(text, str) or len(text) > 2000:
        return {"status": "needs_repeat", "message": "No he entendido la orden. Repite el audio con la regla de perno y cajas."}

    normalized = normalize(text)
    if not normalized:
        return {"status": "needs_repeat", "message": "No he entendido el audio. Repite la orden."}
    if STOP.fullmatch(normalized):
        return {"status": "stopped", "message": "Regla de clasificación desactivada."}

    with_pin = _spans(WITH_PIN, "with_pin", normalized)
    without_pin = _spans(WITHOUT_PIN, "without_pin", normalized)
    if with_pin and not without_pin:
        without_pin = _spans(IMPLICIT_WITHOUT_PIN, "without_pin", normalized)
    conditions = with_pin + without_pin
    if not conditions:
        return {"status": "needs_repeat", "message": "No he entendido una regla de clasificación por perno. Repite el audio."}

    quality_words = _spans(GOOD_WORD, "good", normalized) + _spans(BAD_WORD, "bad", normalized)
    condition_quality = {}
    for quality, start, end in quality_words:
        condition = _nearest_kind((start, end), conditions, maximum=85)
        if condition:
            previous = condition_quality.setdefault(condition, quality)
            if previous != quality:
                return {"status": "needs_repeat", "message": "La regla mezcla buenas y malas para la misma pieza. Repite el audio."}

    color_words = _spans(GREEN_WORD, "green", normalized) + _spans(WHITE_WORD, "white", normalized)
    class_destinations = {}
    for destination, start, end in color_words:
        related_quality = _nearest_kind((start, end), quality_words, maximum=135)
        if related_quality:
            previous = class_destinations.setdefault(related_quality, destination)
            if previous != destination:
                return {"status": "needs_repeat", "message": "La regla asigna dos cajas distintas a la misma clase. Repite el audio."}
        else:
            condition = _nearest_kind((start, end), conditions, maximum=90)
            if condition:
                previous = class_destinations.setdefault(condition, destination)
                if previous != destination:
                    return {"status": "needs_repeat", "message": "La regla asigna dos cajas distintas a la misma pieza. Repite el audio."}

    # Convert destinations stated for a pin condition into good/bad classes.
    good_has_pin = None
    for condition, quality in condition_quality.items():
        is_good = quality == "good"
        condition_has_pin = condition == "with_pin"
        candidate = condition_has_pin if is_good else not condition_has_pin
        if good_has_pin is not None and candidate != good_has_pin:
            return {"status": "needs_repeat", "message": "La regla se contradice sobre qué piezas son buenas. Repite el audio."}
        good_has_pin = candidate

    direct_destinations = {
        key: value for key, value in class_destinations.items()
        if key in ("with_pin", "without_pin")
    }
    if good_has_pin is None:
        if "with_pin" in direct_destinations:
            good_has_pin = direct_destinations["with_pin"] == "green"
        elif "without_pin" in direct_destinations:
            good_has_pin = direct_destinations["without_pin"] == "white"
        else:
            return {"status": "needs_repeat", "message": "Indica si las piezas con perno o sin perno son buenas. Repite el audio."}

    good_destination = class_destinations.get("good")
    bad_destination = class_destinations.get("bad")
    with_destination = direct_destinations.get("with_pin")
    without_destination = direct_destinations.get("without_pin")
    if with_destination:
        target = "good" if good_has_pin else "bad"
        if target in class_destinations and class_destinations[target] != with_destination:
            return {"status": "needs_repeat", "message": "La regla se contradice sobre el destino de las piezas con perno."}
        class_destinations[target] = with_destination
    if without_destination:
        target = "bad" if good_has_pin else "good"
        if target in class_destinations and class_destinations[target] != without_destination:
            return {"status": "needs_repeat", "message": "La regla se contradice sobre el destino de las piezas sin perno."}
        class_destinations[target] = without_destination

    good_destination = class_destinations.get("good", good_destination)
    bad_destination = class_destinations.get("bad", bad_destination)
    if good_destination and not bad_destination:
        bad_destination = "white" if good_destination == "green" else "green"
    elif bad_destination and not good_destination:
        good_destination = "white" if bad_destination == "green" else "green"
    elif not good_destination and not bad_destination:
        return {"status": "needs_repeat", "message": "Indica la caja verde o blanca para las piezas buenas o malas. Repite el audio."}
    if good_destination == bad_destination:
        return {"status": "needs_repeat", "message": "Las piezas buenas y malas no pueden ir a la misma caja. Repite el audio."}

    with_pin_class = "good" if good_has_pin else "bad"
    without_pin_class = "bad" if good_has_pin else "good"
    rule = {
        "active": True,
        "transcript": text.strip(),
        "criterion": "perno",
        "good_has_pin": good_has_pin,
        "good_destination": good_destination,
        "bad_destination": bad_destination,
        "routes": {
            "with_pin": {"quality": with_pin_class, "destination": good_destination if with_pin_class == "good" else bad_destination},
            "without_pin": {"quality": without_pin_class, "destination": good_destination if without_pin_class == "good" else bad_destination},
        },
    }
    return {"status": "active", "message": "Regla guardada y activa hasta que digas «para».", "rule": rule}


class SortingRuleStore:
    """Keep the last accepted voice rule until a stop phrase clears it."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path(__file__).resolve().parent / ".cache" / "sorting-rule.json"
        self.lock = threading.Lock()
        self.rule = None
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("active") and saved.get("routes"):
                self.rule = saved
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    def snapshot(self):
        with self.lock:
            return {"active": self.rule is not None, "rule": self.rule}

    def apply(self, text):
        result = interpret_sorting_rule(text)
        with self.lock:
            if result["status"] == "stopped":
                self.rule = None
                self.path.unlink(missing_ok=True)
                return {**result, "active": False, "rule": None}
            if result["status"] == "active":
                self.rule = result["rule"]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.rule, ensure_ascii=False), encoding="utf-8")
                temporary.replace(self.path)
                return {**result, "active": True}
            return {**result, "active": self.rule is not None, "rule": self.rule}
