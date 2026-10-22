"""Conservative Spanish sorting parser and explicit clarification dialogue.

No model can bypass relation validation. No robot or GUI dependencies.
"""

from __future__ import annotations

import copy
import re
import unicodedata

COLORS = ("red", "green", "blue")
CATEGORIES = {"quality": ("good", "defect_1", "defect_2"), "color": COLORS}
COLOR_WORDS = {"red": "roja", "green": "verde", "blue": "azul"}
QUALITY = {
    "good": r"\b(?:buen[oa]s?|correct[oa]s?|aceptables?|sin defectos?)\b",
    "defect_1": r"\b(?:defect(?:uos[oa]s?|os?)\s*(?:de\s+)?(?:tipo\s*)?(?:uno|1)|tipo\s+(?:uno|1))\b",
    "defect_2": r"\b(?:defect(?:uos[oa]s?|os?)\s*(?:de\s+)?(?:tipo\s*)?(?:dos|2)|tipo\s+(?:dos|2))\b",
}
COLOR_PATTERNS = {"red": r"\broj[oa]s?\b", "green": r"\bverdes?\b", "blue": r"\bazul(?:es)?\b"}
DEST = re.compile(r"\b(?:en|a|al|hacia|dentro de)\s+(?:(?:la|el|las|los|una?|del)\s+)?(?:(?:cajas?|contenedor(?:es)?|recipientes?|destinos?)\s+)?(roj[oa]s?|verdes?|azul(?:es)?|amarill[oa]s?|negr[oa]s?|blanc[oa]s?)\b")
SORT = r"\b(?:clasifica\w*|ordena\w*|separa\w*|distribu\w*|reparte\w*|pon|poner|coloca\w*|deja\w*|lleva\w*|envia\w*|manda\w*|recoge\w*|recoger|coge\w*|recoger|organiza\w*)\b"


def normalize(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c)).split())


def handles(text):
    t = normalize(text)
    if re.match(r"^(?:cuant|donde|muestra|selecciona|inspecciona|hay |que )", t.lstrip("¿")):
        return False
    return bool(re.search(r"\b(?:clasifica\w*|continua|reanuda|deten|cancela|anula|para|pausa|interrumpe|sigue)\b", t)
                or (re.search(SORT, t) and re.search(r"\b(?:las|piezas|tornillos|tuercas|arandelas|soportes)\b", t)))


def mentions(text, patterns):
    return [(m.start(), m.end(), key) for key, pat in patterns.items() for m in re.finditer(pat, text)]


def result(command=None, status="valid", issues=None, clarification=None):
    return {"schema_version": "industrial-1.0", "status": status, "command": command,
            "issues": issues or [], "clarification": clarification}


def canonical(command):
    """Ordering-independent semantic comparison; excludes latency/confidence."""
    value = copy.deepcopy(command)
    if value and "rules" in value:
        value["rules"] = sorted(value["rules"], key=lambda r: (r["category"], r.get("destination") or ""))
    return value


def understand(text, model=None):
    t = normalize(text)
    if not t:
        return result(status="invalid", issues=["empty_text"])
    if len(t) > 2000:
        return result(status="unsupported", issues=["command_too_long"])
    if re.search(r"(?<!\w)-\d+\b", t):
        return result(status="invalid", issues=["negative_quantity"])
    if re.search(r"\b(?:suelda\w*|pinta\w*|taladra\w*|pule|atornilla\w*|empaqueta\w*|desmonta\w*)\b", t):
        return result(status="unsupported", issues=["unsupported_action"])
    # Match full control phrases: a negated stop must never become a stop.
    clean = re.sub(r"^(?:por favor[, ]+|robot[, ]+)", "", t).strip(" .!?¿¡")
    for intent, pattern in {
        "stop": r"(?:deten|para|interrumpe|pausa)(?:\s+(?:la\s+)?(?:clasificacion|proceso|robot))?(?:\s+por favor)?",
        "resume": r"(?:continua|reanuda|sigue)(?:\s+(?:con\s+)?(?:la\s+)?(?:clasificacion|clasificando|proceso))?(?:\s+por favor)?",
        "cancel": r"(?:cancela|anula)(?:\s+(?:la\s+)?(?:orden|peticion|clasificacion))?",
    }.items():
        if re.fullmatch(pattern, clean):
            return result({"intent": intent})
    if re.search(r"\b(?:no|nunca)\b", t) and re.search(r"\b(?:detengas|pares|continues|reanudes)\b", t):
        return result(status="unsupported", issues=["negated_control"])
    if re.search(r"\b(?:deten|pausa|continua|reanuda|cancela)\b|\bpara\s+(?:el\s+(?:proceso|robot)|la\s+clasificacion)\b", t) and re.search(SORT, t):
        return result(status="contradictory", issues=["mixed_control_and_sort"])
    q = mentions(t, QUALITY)
    predicted = model.predict(t) if model is not None else None
    # A learned action can extend verb coverage, but never supply missing destinations.
    is_sort = bool(re.search(SORT, t)) or bool(predicted and predicted.get("intent") == "sort")
    if not is_sort:
        return result(status="unsupported", issues=["unrecognized_action"])
    if re.search(r"\b(?:excepto|menos|salvo|no|nunca)\b", t):
        return result(status="needs_clarification", issues=["negation_scope"],
                      clarification="Indica las categorias que quieres clasificar y sus destinos de forma positiva.")
    destinations = list(DEST.finditer(t))
    source_text = DEST.sub(" ", t)
    c = mentions(source_text, COLOR_PATTERNS)
    if q and c:
        return result(status="needs_clarification", issues=["mixed_modes"],
                      clarification="¿Quieres clasificar por calidad o por color?")
    mode = "quality" if q or re.search(r"\b(?:defectuosas?|defectuosos?|calidad)\b", t) else "color"
    patterns = QUALITY if mode == "quality" else COLOR_PATTERNS
    rules = []
    cursor = 0
    for dest in destinations:
        clause = t[cursor:dest.start()]
        categories = [item[2] for item in sorted(mentions(clause, patterns))]
        consumed_end = dest.end()
        if not categories:
            after = re.split(r"\s+y\s+|[,;]", t[dest.end():], maxsplit=1)[0]
            categories = [item[2] for item in sorted(mentions(after, patterns))]
            if categories:
                consumed_end += len(after)
        if mode == "quality" and not categories and re.search(r"\bdefectuos[oa]s?\b", clause):
            categories = ["defect_1", "defect_2"]
        color = next((key for key, pat in COLOR_PATTERNS.items() if re.search(pat, dest.group(1))), None)
        if color is None:
            return result(status="unsupported", issues=["unsupported_destination"])
        if not categories:
            return result(status="needs_clarification", issues=["missing_category"],
                          clarification="¿Que categoria va en la caja indicada?")
        rules.extend({"category": category, "destination": color} for category in categories)
        cursor = consumed_end
    trailing = [item[2] for item in mentions(t[cursor:], patterns)]
    if destinations and re.search(r"\b(?:en|hacia|caja|contenedor)\b", t[cursor:]):
        return result(status="needs_clarification", issues=["unresolved_trailing_clause"],
                      clarification="Repite cada categoria con un unico destino completo.")
    if not destinations:
        trailing = [item[2] for item in mentions(t, patterns)]
        if mode == "quality" and not trailing and re.search(r"\bdefectuos[oa]s?\b", t):
            trailing = ["defect_1", "defect_2"]
    rules.extend({"category": category, "destination": None} for category in trailing)
    command = {"intent": "sort", "mode": mode, "rules": rules}
    if re.search(r"\b(?:solamente|solo|unicamente)\b", t):
        command["filter"] = list(dict.fromkeys(r["category"] for r in rules))
    return validate(command)


def validate(command):
    command = copy.deepcopy(command)
    mode = command.get("mode")
    if mode not in CATEGORIES:
        return result(status="invalid", issues=["invalid_mode"])
    unique = {}
    for rule in command.get("rules", []):
        cat, dest = rule.get("category"), rule.get("destination")
        if cat not in CATEGORIES[mode] or (dest is not None and dest not in COLORS):
            return result(status="unsupported", issues=["unsupported_rule"])
        if cat in unique and unique[cat] != dest:
            return result(command, "contradictory", ["conflicting_destinations"])
        unique[cat] = dest
    command["rules"] = [{"category": c, "destination": d} for c, d in unique.items()]
    if not unique:
        return result(command, "needs_clarification", ["missing_category"], "¿Que categorias y destinos quieres usar?")
    if any(d is None for d in unique.values()):
        missing = ", ".join(c for c, d in unique.items() if d is None)
        return result(command, "needs_clarification", ["missing_destination"],
                      f"¿A que caja (roja, verde o azul) envio {missing}?")
    return result(command)


class Dialogue:
    """A missing destination always needs an explicit answer, even with active rules."""

    def __init__(self):
        self.pending = None
        self.active = None
        self.running = False

    def submit(self, text, model=None):
        t = normalize(text)
        if self.pending and not re.search(SORT, t) and not re.search(r"\b(?:deten|para|pausa|cancela|anula|continua|reanuda)\b", t):
            pending_command = self.pending.get("command")
            if pending_command and "missing_destination" in self.pending["issues"]:
                candidate = copy.deepcopy(pending_command)
                explicit = understand("pon " + text)
                if explicit["status"] == "valid" and explicit.get("command", {}).get("rules"):
                    updates = {r["category"]: r["destination"] for r in explicit["command"]["rules"]}
                    for rule in candidate["rules"]:
                        if rule["destination"] is None and rule["category"] in updates:
                            rule["destination"] = updates[rule["category"]]
                else:
                    colors = {m[2] for m in mentions(t, COLOR_PATTERNS)}
                    # Only a bare destination answer can apply one color to all missing rules.
                    if len(colors) == 1 and re.fullmatch(r"(?:(?:a|en)\s+)?(?:(?:la|el)\s+)?(?:(?:caja|contenedor)\s+)?(?:roja?|rojo|verde|azul)[.! ]*", t):
                        for rule in candidate["rules"]:
                            if rule["destination"] is None:
                                rule["destination"] = next(iter(colors))
                response = validate(candidate)
            else:
                response = result(status="needs_clarification", issues=["repeat_complete_command"],
                                  clarification="Reformula la instruccion completa con categorias y destinos.")
        else:
            response = understand(text, model)
        if response["status"] == "needs_clarification":
            self.pending = response
            self.running = False
        elif response["status"] == "valid":
            intent = response["command"]["intent"]
            if intent == "sort":
                self.active = response["command"]
                self.pending = None
                self.running = True
            elif intent == "cancel":
                self.pending = None
                self.active = None
                self.running = False
            elif intent == "stop":
                self.running = False
            elif intent == "resume":
                if self.active is None or self.pending:
                    return result(status="needs_clarification", issues=["no_resumable_plan"],
                                  clarification="Completa una orden de clasificacion antes de continuar.")
                self.running = True
        else:
            self.running = False
        return response


def to_legacy(envelope, text, source="text"):
    from roboqc.commands.schema import Action, StructuredCommand
    status = {"valid": "VALID", "needs_clarification": "AMBIGUOUS", "unsupported": "UNKNOWN"}.get(envelope["status"], "INVALID")
    cmd = StructuredCommand(source=source, transcription=text, normalized_text=normalize(text), status=status)
    cmd.actions = [Action("INDUSTRIAL_COMMAND", {"industrial": envelope})]
    cmd.validation_errors = list(envelope["issues"])
    if envelope["clarification"]:
        cmd.unresolved_references = [envelope["clarification"]]
    return cmd
