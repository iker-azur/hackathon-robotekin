"""Conservative complete-phrase rules plus the existing multilingual extractor."""
from __future__ import annotations

import re
import time

from command_model_v2.command_model import COLOR_WORDS, NUMBER_WORDS, POSITION_WORDS, PLACE_WORDS, plain, lookup

KINDS = {
    "cube": "cubo cubos dado dados bloque bloques cube cubes block blocks kubo kuboak bloke blokeak".split(),
    "ball": "bola bolas pelota pelotas esfera esferas ball balls sphere spheres orb orbs bolak pilota pilotak baloi baloiak".split(),
    "any": "pieza piezas objeto objetos thing things piece pieces object objects element elements pieza piezak objektu objektuak".split(),
}
KIND_WORDS = {word: kind for kind, words in KINDS.items() for word in words}
PICK = set("coge coger coje agarra agarrar toma recoge take grab fetch pick hartu jaso".split())
PUT = set("pon poner coloca colocar mete meter introduce put place insert sartu jarri".split())
SORT = set("separa separad ordena ordenar clasifica clasificar sort separate arrange sailkatu bereizi".split())
CLEAR = set("limpia despeja quita retira clear clean remove garbitu kendu".split())
FILLER = set("el la los las un una unos unas de del a al en y o por para con lo le laslos primero luego despues todos todas the a an of to in into and it them on by please first then all eta bat mesedez ezazu itzazu mahai mahaia mesa table color colors colour colours colores kolore koloreak kolorearen kolorez".split())
VERBS = PICK | PUT | SORT | CLEAR


def words(text):
    return re.findall(r"[a-z0-9_]+", plain(text))


def value(token, table):
    if token in table:
        return table[token]
    # Restrict morphological fallback to known Basque endings; arbitrary words such
    # as redwood must not become red simply because they start with a colour.
    for stem in sorted(table, key=len, reverse=True):
        if len(stem) >= 3 and token.startswith(stem) and token[len(stem):] in {"a", "ak", "an", "ra", "aren", "arekin", "ko", "ren"}:
            return table[stem]
    return None


def item_group(text, inherited=None):
    tokens = words(text)
    kinds = {value(t, KIND_WORDS) for t in tokens} - {None}
    shades = {value(t, COLOR_WORDS) for t in tokens} - {None}
    positions = {value(t, POSITION_WORDS) for t in tokens} - {None}
    quantities = [int(t) if t.isdigit() else NUMBER_WORDS[t] for t in tokens if t.isdigit() or t in NUMBER_WORDS]
    if len(kinds) > 1 or len(shades) > 1 or len(positions) > 1 or len(set(quantities)) > 1:
        return None
    for token in tokens:
        if token in FILLER | VERBS or token.isdigit() or value(token, KIND_WORDS) or value(token, COLOR_WORDS) or value(token, POSITION_WORDS) or token in NUMBER_WORDS:
            continue
        return None
    if not kinds and not shades:
        return None
    kind = next(iter(kinds), inherited or "any")
    noun = next((t for t in tokens if value(t, KIND_WORDS)), None)
    plural = noun in {"bolas", "pelotas", "esferas", "cubos", "dados", "bloques", "balls", "spheres", "cubes", "blocks", "bolak", "kuboak", "pilotak", "piezas", "pieces"}
    quantity = quantities[0] if quantities else "all" if plural else 1
    return {"kind": kind, "name": noun, "color": next(iter(shades), None),
            "quantity": quantity, "position": next(iter(positions), None), "place": None}


class HybridParser:
    def __init__(self, fallback=None, catalog=None):
        self.fallback = fallback
        self.catalog = list(catalog or [])
        self.last_backend = ""

    def destination(self, text):
        normalized = plain(text)
        named = []
        for entry in self.catalog:
            if not entry["container"]:
                continue
            for label in [entry["id"], entry["name"], *entry.get("aliases", [])]:
                label = plain(label)
                if re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", normalized):
                    remainder = re.sub(r"(?<!\w)" + re.escape(label) + r"(?!\w)", "", normalized)
                    if all(w in FILLER | PUT or w in PLACE_WORDS for w in words(remainder)):
                        named.append(entry)
        if len({e["id"] for e in named}) == 1:
            entry = named[0]
            return {"kind": "box", "color": entry["color"], "entity_id": entry["id"]}
        tokens = words(text)
        kinds = {value(w, PLACE_WORDS) for w in tokens} - {None}
        shades = {value(w, COLOR_WORDS) for w in tokens} - {None}
        if len(kinds) != 1 or len(shades) > 1:
            return None
        if any(w not in FILLER | PUT and not value(w, PLACE_WORDS) and not value(w, COLOR_WORDS) for w in tokens):
            return None
        return {"kind": next(iter(kinds)), "color": next(iter(shades), None)}

    def clause(self, text):
        place, body = None, text
        # A separator is a destination only if its entire tail is recognised.
        for match in re.finditer(r"\b(?:en|in|into|to|a|sartu)\b", plain(text)):
            destination = self.destination(plain(text)[match.end():])
            if destination:
                place, body = destination, plain(text)[:match.start()]
                break
        if place is None:
            # Basque case endings encode the preposition: kutxa gorrian.
            for match in re.finditer(r"\b\w+\b", plain(text)):
                if value(match.group(), PLACE_WORDS):
                    destination = self.destination(plain(text)[match.start():])
                    if destination:
                        place, body = destination, plain(text)[:match.start()]
                        break
        pieces = re.split(r"\b(?:y|and|eta)\b", body)
        items = []
        for piece in pieces:
            if not words(piece):
                continue
            item = item_group(piece, items[-1]["kind"] if items else None)
            if not item:
                return None
            items.append(item)
        return items, place

    def rules(self, text):
        normalized = plain(text)
        tokens = words(text)
        if re.search(r"\b(?:no|not|never|nunca|ez|dont)\b|don't|n't", normalized):
            return None
        if normalized.strip(" .!?") in {"para", "parar", "parada", "stop", "detente", "gelditu", "cancel", "cancela", "cancelar", "bertan behera"}:
            return self.command("stop", [], None)
        verbs = set(tokens) & VERBS
        if not verbs:
            return None
        if verbs & SORT and not any(value(w, PLACE_WORDS) for w in tokens):
            group = item_group(normalized)
            meaningful = [w for w in tokens if w not in SORT | FILLER]
            if not group and meaningful:
                return None
            command = self.command("sort", [], None, "color")
            command["sort_filter"] = [group] if group and (group["kind"] != "any" or group["color"]) else []
            return command
        parsed = self.clause(normalized)
        if parsed:
            items, place = parsed
            intent = "clear" if verbs & CLEAR else "pick_and_place" if place else "pick"
            if verbs & PUT and place is None:
                return None
            return self.command(intent, items, place)
        # Each clause must independently name its object and its own destination.
        segments = re.split(r"\b(?:y|and|eta)\b", normalized)
        items = []
        for segment in segments:
            parsed = self.clause(segment)
            if not parsed or len(parsed[0]) != 1 or parsed[1] is None:
                return None
            item = parsed[0][0]
            item["place"] = parsed[1]
            items.append(item)
        return self.command("sequence", items, None) if len(items) > 1 else None

    @staticmethod
    def command(intent, items, place, sort_by="none"):
        # Rule matching is provenance, not a statistically calibrated confidence.
        return {"intent": intent, "items": items, "place": place, "sort_by": sort_by,
                "sort_filter": [], "confidence": None, "confidences": {}, "parser": "rules"}

    def parse(self, text, *args, **kwargs):
        started = time.perf_counter()
        command = self.rules(text)
        if self.fallback is not None and command is not None and command["intent"] != "stop":
            # Keep the trained interpretation for coordination, clear/sort and implicit
            # references. The rule fast path requires one fully named physical object.
            items = command.get("items", [])
            if (command["intent"] not in {"pick", "pick_and_place"} or len(items) != 1
                    or items[0]["kind"] not in {"ball", "cube"} or not items[0]["color"]):
                command = None
        if command is None:
            if self.fallback is None:
                command = self.command("not_doable", [], None)
                command["errors"] = ["No se reconoce toda la orden. Reformúlala usando piezas y recipientes de la escena."]
            else:
                command = self.fallback.parse(text, *args, **kwargs)
                command["parser"] = "model"
        errors = list(command.get("errors", []))
        if re.search(r"\b(?:no|not|never|nunca|ez|dont)\b|don't|n't", plain(text)) and command["intent"] in {"pick", "pick_and_place", "sort", "clear", "sequence", "give"}:
            errors.append("La orden contiene una negación o corrección. Reformúlala como una instrucción positiva.")
        if self.catalog and command["intent"] not in {"not_doable", "off_topic", "question", "greeting", "stop", "cancel"}:
            active_colors = {entry["color"] for entry in self.catalog}
            mentioned_colors = {value(w, COLOR_WORDS) for w in words(text)} - {None}
            if mentioned_colors - active_colors:
                errors.append("Hay colores solicitados que no pertenecen a la escena activa.")
            # Unknown destinations must not be replaced by a model's guessed box.
            for segment in re.split(r"\b(?:y|and|eta)\b", plain(text)):
                match = re.search(r"\b(?:en|into|in|to)\b(.*)$", segment)
                if match and self.destination(match.group(1)) is None:
                    errors.append("No reconozco el destino solicitado; usa su nombre o su tipo y color.")
            for item in command.get("items", []):
                quantity = item.get("quantity")
                if quantity != "all" and (type(quantity) is not int or quantity < 1):
                    errors.append("La cantidad solicitada no es válida.")
                if item.get("kind") not in {"ball", "cube", "any"}:
                    errors.append("La escena solo admite cubos y bolas.")
            known = FILLER | VERBS | set(KIND_WORDS) | set(COLOR_WORDS) | set(POSITION_WORDS) | set(PLACE_WORDS) | set(NUMBER_WORDS)
            known |= set("could would can you me quiero necesito favor please dame move mueve lleva itzazu ezazu".split())
            for entry in self.catalog:
                for label in [entry["id"], entry["name"], *entry.get("aliases", [])]:
                    known.update(words(label))
            unknown = [w for w in words(text) if w not in known and not w.isdigit()
                       and not any(value(w, table) for table in (KIND_WORDS, COLOR_WORDS, POSITION_WORDS, PLACE_WORDS))]
            if unknown and command["intent"] in {"pick", "pick_and_place", "sort", "clear", "sequence", "give"}:
                errors.append("Hay palabras o referencias no reconocidas: " + ", ".join(dict.fromkeys(unknown)) + ". Reformula la orden.")
        if errors:
            command["errors"] = list(dict.fromkeys(errors))
        self.last_backend = command["parser"]
        command["parse_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return command
