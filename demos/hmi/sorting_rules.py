"""Interpret the persistent pin/no-pin sorting rule spoken to the HMI."""

from __future__ import annotations

import re
import unicodedata
import json
import threading
import sys
from pathlib import Path


PART = r"(?:pernos?|pasadores?|tornillos?|bulones?|pin(?:es|s)?)"
GREEN = r"(?:verde|verdes)"
WHITE = r"(?:blanco|blanca|blancos|blancas)"
GOOD = r"(?:buen[oa]s?|correct[oa]s?|valid[oa]s?|apt[oa]s?|aceptables?|aprobadas?|sirven)"
BAD = r"(?:mal[oa]s?|defectuos[oa]s?|incorrect[oa]s?|invalid[oa]s?|rechazad[oa]s?|inservibles?)"
WITH_PIN = re.compile(rf"\b(?:con|tiene|tienen|tenga|tengan|lleva|llevan|incluye|incluyen|porta|portan)\s+(?:(?:un|unos|el|los)\s+)?{PART}\b")
WITHOUT_PIN = re.compile(rf"\b(?:sin|no\s+(?:tiene|tienen|tenga|tengan|lleva|llevan|incluye|incluyan)|carece\s+de|falta)\s+(?:(?:un|unos|el|los)\s+)?{PART}\b")
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
PICK_TIME_WORDS = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3,
    "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
    "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21,
    "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29, "treinta": 30, "cuarenta": 40,
    "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
    "noventa": 90, "cien": 100, "ciento": 100, "doscientos": 200,
    "trescientos": 300, "cuatrocientos": 400, "quinientos": 500,
    "seiscientos": 600, "setecientos": 700, "ochocientos": 800,
    "novecientos": 900, "mil": 1000,
}


VOICE_FIELD_ALIASES = {
    "pick_seconds": ("segundos de recogida", "duracion de recogida", "duracion maxima de recogida", "duracion maxima de la recogida", "duracion maxima recogida", "tiempo de recogida", "recogida a", "recogida en"),
    "pick_step": ("paso de recogida", "paso maximo de recogida", "paso maximo de la recogida"),
    "pick_actions": ("acciones de recogida", "acciones por prediccion de recogida", "acciones por prediccion de la recogida", "acciones por prediccion del ciclo"),
    "first_gap": ("diferencia inicial", "salto inicial", "distancia inicial"),
    "min_pick": ("tiempo minimo de recogida", "tiempo minimo antes de parar"),
    "pose_window": ("ventana estable", "tiempo de pose estable", "tiempo estable"),
    "arm_motion": ("movimiento tolerado del brazo", "movimiento maximo del brazo"),
    "arm_error": ("distancia tolerada al objetivo", "error maximo del brazo", "distancia al objetivo"),
    "gripper_motion": ("movimiento tolerado de pinza", "movimiento maximo de pinza"),
    "control_hz": ("frecuencia de control", "frecuencia de la aplicacion", "frecuencia"),
    "settle_seconds": ("pausa de estabilizacion", "tiempo de estabilizacion"),
    "classify_seconds": ("tiempo de clasificacion", "duracion de clasificacion"),
    "pin_threshold": ("umbral de perno", "umbral para con perno", "porcentaje de perno", "umbral perno"),
    "auto_stop": ("parada automatica", "auto stop", "parar al alcanzar la pose", "parar al llegar a la pose"),
}
for _box, _color in (("green", "verde"), ("white", "blanca")):
    VOICE_FIELD_ALIASES.update({
        f"{_box}_seconds": (
            f"segundos de la caja {_color}", f"duracion de la caja {_color}",
            f"duracion maxima de la caja {_color}", f"duracion maxima caja {_color}",
            f"tiempo de ruta {_color}"),
        f"{_box}_step": (
            f"paso de la caja {_color}", f"paso maximo {_color}",
            f"paso maximo de la caja {_color}"),
        f"{_box}_actions": (f"acciones de la caja {_color}", f"acciones por prediccion {_color}", f"acciones por prediccion de la caja {_color}"),
        f"{_box}_height_cm": (f"elevacion de la caja {_color}", f"elevacion adicional de la caja {_color}", f"elevacion adicional caja {_color}", f"altura adicional {_color}"),
        f"{_box}_descent_mm": (f"descenso de la caja {_color}", f"descenso caja {_color}", f"descenso en correccion de la caja {_color}", f"correccion de altura {_color}"),
        f"{_box}_closed": (f"apertura inicial de la pinza {_color}", f"apertura inicial pinza {_color}", f"apertura inicial de la pinza de la caja {_color}", f"inicio de apertura {_color}", f"inicio de apertura de la pinza de la caja {_color}"),
        f"{_box}_released": (f"umbral para soltar en la caja {_color}", f"umbral de liberacion {_color}", f"umbral de liberacion de la caja {_color}", f"umbral de liberacion caja {_color}", f"liberacion caja {_color}", f"pieza soltada caja {_color}"),
        f"{_box}_clearance": (f"compensacion de altura {_color}", f"compensacion altura caja {_color}", f"compensacion de altura de la caja {_color}"),
        f"{_box}_finish": (f"parar al volver a reposo {_color}", f"parar al volver a reposo de la caja {_color}"),
    })


def _motion_controls():
    controls_dir = Path(__file__).resolve().parents[1] / "06_imitation_learning"
    if str(controls_dir) not in sys.path:
        sys.path.insert(0, str(controls_dir))
    import motion_controls
    return motion_controls


def _voice_field_specs():
    controls = _motion_controls()
    specs = {}
    labels = [normalize(field[2]) for field in controls.NUMERIC_FIELDS + controls.BOOL_FIELDS]
    for key, tab, label, *_ in controls.NUMERIC_FIELDS + controls.BOOL_FIELDS:
        normalized_label = normalize(label)
        aliases = [normalize(f"{tab} {label}"), normalize(f"{label} {tab}")]
        if labels.count(normalized_label) == 1:
            aliases.append(normalized_label)
        aliases.extend(VOICE_FIELD_ALIASES.get(key, ()))
        aliases.append(normalize(key.replace("_", " ")))
        specs[key] = (tuple(sorted(set(aliases), key=len, reverse=True)), key in {field[0] for field in controls.BOOL_FIELDS})
    return specs


def _spoken_number(tokens, index):
    token = tokens[index]
    if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return float(token), 1
    if token not in PICK_TIME_WORDS:
        return None, 0
    value = PICK_TIME_WORDS[token]
    consumed = 1
    if value >= 100:
        next_index = index + 1
        if next_index < len(tokens) and tokens[next_index] in PICK_TIME_WORDS and 20 <= PICK_TIME_WORDS[tokens[next_index]] < 100:
            value += PICK_TIME_WORDS[tokens[next_index]]
            next_index += 1
            consumed += 1
            if next_index < len(tokens) and tokens[next_index] == "y":
                next_index += 1
                consumed += 1
            if next_index < len(tokens) and tokens[next_index] in PICK_TIME_WORDS and 0 < PICK_TIME_WORDS[tokens[next_index]] < 10:
                value += PICK_TIME_WORDS[tokens[next_index]]
                consumed += 1
        return float(value), consumed
    if value >= 20 and value % 10 == 0:
        next_index = index + 1
        if next_index < len(tokens) and tokens[next_index] == "y":
            next_index += 1
            consumed += 1
        if next_index < len(tokens) and tokens[next_index] in PICK_TIME_WORDS and 0 < PICK_TIME_WORDS[tokens[next_index]] < 10:
            value += PICK_TIME_WORDS[tokens[next_index]]
            consumed += 1
    return float(value), consumed


def _voice_settings(text, current):
    """Parse explicit spoken field assignments and validate them as one edit."""
    if not isinstance(text, str):
        return {}, None
    normalized = normalize(text)
    matches = []
    for key, (aliases, is_bool) in _voice_field_specs().items():
        for alias in aliases:
            if alias and alias in normalized:
                start = normalized.index(alias)
                matches.append((start, start + len(alias), key, is_bool))
                break
    if not matches:
        return {}, None
    matches.sort()
    tokens = normalized.split()
    changes = {}
    decimals = re.findall(r"(?<!\d)-?\d+[,.]\d+(?!\d)", text)
    if decimals and (len(decimals) != 1 or len(matches) != 1 or matches[0][3]):
        return {}, "Indica los valores decimales de uno en uno para evitar confusiones."
    for start, end, key, is_bool in matches:
        token_end = len(normalized[:end].split())
        if is_bool:
            segment_end = min((item[0] for item in matches if item[0] > start), default=len(normalized))
            segment_start = max(0, len(normalized[:start].split()) - 4)
            after = tokens[token_end:len(normalized[:segment_end].split())][:6]
            value = next((word for word in after if word in (
                "si", "no", "activar", "activa", "activado", "activa", "habilitar",
                "habilita", "habilitado", "desactivar", "desactiva", "desactivado",
                "deshabilitar", "deshabilita", "deshabilitado", "quitar", "eliminar")), None)
            if value is None:
                before = tokens[segment_start:token_end]
                value = next((word for word in reversed(before)
                              if word in ("activar", "activa", "activado", "habilitar", "habilita",
                                          "habilitado", "desactivar", "desactiva", "desactivado",
                                          "deshabilitar", "deshabilita", "deshabilitado")), None)
            if value is None:
                return {}, f"Indica sí/no para {key}."
            changes[key] = value in ("si", "activar", "activa", "activado", "habilitar", "habilita", "habilitado")
            continue
        next_start = min((item[0] for item in matches if item[0] > start), default=len(normalized))
        segment_end = len(normalized[:next_start].split())
        candidates = []
        optional = {field[0] for field in _motion_controls().NUMERIC_FIELDS if field[3] is None}
        model_value = next((word for word in tokens[token_end:segment_end][:6]
                            if word in ("modelo", "checkpoint", "politica")), None)
        if key in optional and model_value:
            changes[key] = None
            continue
        if decimals:
            changes[key] = float(decimals[0].replace(",", "."))
            continue
        for index in range(token_end, min(len(tokens), segment_end)):
            value, consumed = _spoken_number(tokens, index)
            if consumed:
                candidates.append((index, value))
        if not candidates:
            for index in range(max(0, token_end - 4), token_end):
                value, consumed = _spoken_number(tokens, index)
                if consumed:
                    candidates.append((index, value))
        if not candidates:
            return {}, f"No entendí el valor numérico de {key}."
        changes[key] = min(candidates, key=lambda item: abs(item[0] - token_end))[1]
    try:
        controls = _motion_controls()
        validated = controls.parse_settings({**current, **changes})
    except (ValueError, KeyError, TypeError) as exc:
        return {}, str(exc)
    return {key: validated[key] for key in changes}, None


def describe_settings(changes):
    fields = {key: (tab, label) for key, tab, label, *_ in
              _motion_controls().NUMERIC_FIELDS + _motion_controls().BOOL_FIELDS}
    return "; ".join(f"{fields[key][0]} · {fields[key][1]}: "
                      f"{'Sí' if value is True else 'No' if value is False else value}"
                      for key, value in changes.items() if key in fields)


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
    """Keep the active voice configuration and one-step rollback state."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path(__file__).resolve().parent / ".cache" / "sorting-rule.json"
        self.lock = threading.Lock()
        self.rule = None
        self.settings = None
        self.defaults = None
        self.previous = None
        self.revision = 0
        self.controller_seen = False
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if "rule" in saved:
                self.rule = saved.get("rule")
                self.settings = saved.get("settings")
                self.defaults = saved.get("defaults")
                self.previous = saved.get("previous")
                self.revision = int(saved.get("revision", 0))
                if self.settings is None and saved.get("pick_seconds") is not None:
                    self.settings = _motion_controls().default_settings(pick_seconds=saved["pick_seconds"])
                if self.revision == 0 and (self.rule is not None or self.settings is not None):
                    self.revision = 1
            elif saved.get("active") and saved.get("routes"):
                self.rule = saved
                self.revision = 1
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass

    @staticmethod
    def _defaults():
        return _motion_controls().default_settings()

    def snapshot(self):
        with self.lock:
            return {"active": self.rule is not None, "rule": self.rule,
                    "settings": dict(self.settings) if self.settings is not None else None,
                    "defaults": dict(self.defaults) if self.defaults is not None else None,
                    "revision": self.revision}

    def observe_controller(self, report):
        if not isinstance(report, dict):
            return
        controls = _motion_controls()
        try:
            reported_defaults = controls.parse_settings(report["defaults"])
            reported_settings = controls.parse_settings(report["settings"])
            reported_revision = int(report.get("config_revision", 0))
        except (KeyError, ValueError, TypeError):
            return
        with self.lock:
            generic_defaults = self._defaults()
            first_report = not self.controller_seen
            if self.defaults is None:
                self.defaults = reported_defaults
            if not self.controller_seen:
                if self.settings is None:
                    self.settings = reported_settings
                elif self.revision:
                    self.settings = {**reported_defaults, **self._differences(self.settings, generic_defaults)}
                if self.previous and self.previous.get("settings"):
                    self.previous["settings"] = {
                        **reported_defaults,
                        **self._differences(self.previous["settings"], generic_defaults),
                    }
                self.controller_seen = True
            changed = first_report
            if reported_revision == self.revision:
                if self.settings != reported_settings:
                    if not first_report and self.settings is not None:
                        self.previous = self._current()
                    self.settings = reported_settings
                    if not first_report:
                        self.revision += 1
                    changed = True
            if changed:
                self._save()

    @staticmethod
    def _differences(values, baseline):
        return {key: value for key, value in values.items()
                if key in baseline and value != baseline[key]}

    def _current(self):
        return {"rule": self.rule, "settings": self.settings}

    def _restore(self, config):
        self.rule = config.get("rule")
        self.settings = config.get("settings")

    def apply(self, text):
        if not isinstance(text, str):
            return {"status": "needs_repeat", "message": "No entendí la configuración. Repítela."}
        normalized = normalize(text)
        if any(phrase in normalized for phrase in (
            "vuelve a la configuracion anterior", "volver a la configuracion anterior",
            "restaura la configuracion anterior", "recupera la configuracion anterior",
            "vuelve al valor anterior", "deshaz el ultimo cambio", "deshacer el ultimo cambio",
            "revierte el ultimo cambio", "vuelve a lo anterior", "dejalo como estaba",
            "deja la configuracion como antes", "vuelve a la configuracion que habia antes",
            "restaura el estado anterior", "ponlo como estaba",
        )):
            with self.lock:
                if self.previous is None:
                    return {"status": "needs_repeat", "message": "Aún no hay una configuración anterior que restaurar."}
                current = self._current()
                previous = self.previous
                self.previous = current
                self._restore(previous)
                self.revision += 1
                self._save()
                return {"status": "restored", "message": "Configuración anterior restaurada para el siguiente ciclo.",
                        "active": self.rule is not None, **self._current(), "revision": self.revision}

        if any(phrase in normalized for phrase in (
            "restaura los valores predeterminados", "vuelve a los valores predeterminados",
            "configuracion por defecto", "configuracion predeterminada",
        )):
            with self.lock:
                current = self._current()
                self.previous = current
                self.rule = None
                self.settings = dict(self.defaults or self._defaults())
                self.revision += 1
                self._save()
                return {"status": "restored", "message": "Valores predeterminados restaurados.",
                        "active": False, **self._current(), "revision": self.revision}

        with self.lock:
            base_revision = self.revision
            current_settings = dict(self.settings or self.defaults or self._defaults())
        setting_changes, setting_error = _voice_settings(text, current_settings)
        if setting_error:
            return {"status": "needs_repeat", "message": setting_error}
        result = interpret_sorting_rule(text)
        sorting_context = bool(GOOD_WORD.search(normalized) or BAD_WORD.search(normalized)
                               or WITH_PIN.search(normalized) or WITHOUT_PIN.search(normalized))
        if sorting_context and result["status"] not in ("active", "stopped"):
            return {"status": "needs_repeat", "message": result["message"]}
        with self.lock:
            if self.revision != base_revision:
                return {"status": "needs_repeat", "message": "La configuración cambió mientras procesaba la orden. Repítela."}
            if not setting_changes and result["status"] not in ("active", "stopped"):
                return {**result, "active": self.rule is not None, **self._current(),
                        "revision": self.revision}
            if self.settings is None:
                self.settings = dict(current_settings)
            self.previous = self._current()
            if result["status"] == "active":
                self.rule = result["rule"]
            elif result["status"] == "stopped":
                self.rule = None
            self.settings.update(setting_changes)
            self.revision += 1
            self._save()
            if result["status"] not in ("active", "stopped"):
                result = {"status": "settings_updated",
                          "message": "Configuración actualizada para el siguiente ciclo."}
            return {**result, "active": self.rule is not None, **self._current(),
                    "revision": self.revision, "changes": setting_changes}

    def _save(self):
        if self.rule is None and self.settings is None and self.previous is None:
            self.path.unlink(missing_ok=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"rule": self.rule, "settings": self.settings,
                                         "defaults": self.defaults, "previous": self.previous,
                                         "revision": self.revision}, ensure_ascii=False),
                             encoding="utf-8")
        temporary.replace(self.path)
