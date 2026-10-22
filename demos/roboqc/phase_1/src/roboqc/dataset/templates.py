"""Compositional Spanish template generators for RoboQC dataset."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

OBJECTS: List[Tuple[str, str]] = [
    ("pieza", "f"), ("cubo", "m"), ("cilindro", "m"), ("esfera", "f"),
    ("prisma", "m"), ("componente", "m"), ("producto", "m"), ("objeto", "m"),
]

COLORS: List[str] = [
    "rojo", "azul", "verde", "amarillo", "blanco", "negro",
    "naranja", "morado", "rosa", "gris", "marron",
]

DESTS: List[Tuple[str, str]] = [
    ("caja", "f"), ("contenedor", "m"), ("bandeja", "f"),
]

# quantity: (value, singular_word, plural_word) ; value can be int/str
QUANTITIES: List[Tuple[Any, str, str]] = [
    ("UNSPECIFIED", "", ""),
    (1, "una", "un"),
    (2, "dos", "dos"),
    (3, "tres", "tres"),
    (4, "cuatro", "cuatro"),
    (5, "cinco", "cinco"),
    ("ALL", "toda", "todas"),
    ("SOME", "alguna", "algunas"),
    ("REMAINING", "la restante", "las restantes"),
]

QUALITIES: List[str] = ["bueno", "defectuoso", "desconocido"]

VERBS_PICK = ["Coge", "Coge", "Agarra", "Toma", "Recoge", "Coge"]
VERBS_PLACE = ["Deja", "Pon", "Coloca", "Sitúa"]
VERBS_MOVE = ["Mueve", "Transfiere", "Transporta"]
VERBS_INSPECT = ["Inspecciona", "Revisa"]
VERBS_CLASSIFY = ["Clasifica", "Evalúa"]

# color agreement
_COLOR_AGREE: Dict[str, Dict[Tuple[str, str], str]] = {
    "rojo": {("m", "sg"): "rojo", ("f", "sg"): "roja", ("m", "pl"): "rojos", ("f", "pl"): "rojas"},
    "azul": {("m", "sg"): "azul", ("f", "sg"): "azul", ("m", "pl"): "azules", ("f", "pl"): "azules"},
    "verde": {("m", "sg"): "verde", ("f", "sg"): "verde", ("m", "pl"): "verdes", ("f", "pl"): "verdes"},
    "amarillo": {("m", "sg"): "amarillo", ("f", "sg"): "amarilla", ("m", "pl"): "amarillos", ("f", "pl"): "amarillas"},
    "blanco": {("m", "sg"): "blanco", ("f", "sg"): "blanca", ("m", "pl"): "blancos", ("f", "pl"): "blancas"},
    "negro": {("m", "sg"): "negro", ("f", "sg"): "negra", ("m", "pl"): "negros", ("f", "pl"): "negras"},
    "naranja": {("m", "sg"): "naranja", ("f", "sg"): "naranja", ("m", "pl"): "naranjas", ("f", "pl"): "naranjas"},
    "morado": {("m", "sg"): "morado", ("f", "sg"): "morada", ("m", "pl"): "morados", ("f", "pl"): "moradas"},
    "rosa": {("m", "sg"): "rosa", ("f", "sg"): "rosa", ("m", "pl"): "rosas", ("f", "pl"): "rosas"},
    "gris": {("m", "sg"): "gris", ("f", "sg"): "gris", ("m", "pl"): "grises", ("f", "pl"): "grises"},
    "marron": {("m", "sg"): "marron", ("f", "sg"): "marron", ("m", "pl"): "marrones", ("f", "pl"): "marrones"},
}


def color_word(color: str, gender: str, number: str) -> str:
    key = (gender, number)
    return _COLOR_AGREE.get(color, {}).get(key, color)


def obj_plural(obj: str) -> str:
    return obj + ("es" if obj.endswith(("z", "s", "x")) else "s") if False else None or (obj + "s")


def pluralize(noun: str) -> str:
    if noun.endswith("z"):
        return noun[:-1] + "ces"
    if noun.endswith("s") or noun.endswith("x"):
        return noun
    return noun + "s"


def qty_is_plural(qval: Any) -> bool:
    if isinstance(qval, int):
        return qval != 1
    if qval in ("ALL", "SOME", "REMAINING"):
        return True
    return False


def qty_word(q: Tuple[Any, str, str], gender: str, plural: bool) -> str:
    val, sw, pw = q
    if val == "UNSPECIFIED":
        return ""
    if val == 1:
        return sw if sw else "un"
    if plural:
        return pw
    return sw


def make_object_phrase(obj: str, gender: str, color: str, q: Tuple[Any, str, str]) -> str:
    plural = qty_is_plural(q[0])
    noun = pluralize(obj) if plural else obj
    qw = qty_word(q, gender, plural)
    cw = color_word(color, gender, "pl" if plural else "sg")
    parts = []
    if qw:
        parts.append(qw)
    parts.append(noun)
    parts.append(cw)
    return " ".join(parts)


def make_dest_phrase(dest: str, gender: str, color: Optional[str]) -> str:
    art = "la" if gender == "f" else "el"
    if color:
        cw = color_word(color, gender, "sg")
        return f"{art} {dest} {cw}"
    return f"{art} {dest}"


def rchoice(rng, seq):
    return rng.choice(seq)


def rcolor(rng):
    return rng.choice(COLORS)


def robject(rng):
    return rng.choice(OBJECTS)


def rdest(rng):
    return rng.choice(DESTS)


def rqty(rng):
    return rng.choice(QUANTITIES)


def rquality(rng):
    return rng.choice(QUALITIES)


# ---------------------------------------------------------------------------
# Template generators: each returns (text, actions_list, family)
# ---------------------------------------------------------------------------

def _action(intent, params, negated=False):
    return {"intent": intent, "parameters": params, "negated": negated}


def gen_pick_simple(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    q = rqty(rng)
    phrase = make_object_phrase(obj, og, color, q)
    verb = rchoice(rng, VERBS_PICK)
    templates = [
        f"{verb} {phrase}.",
        f"{verb} la {obj} {color_word(color, og, 'sg')}.",
        f"Por favor {verb.lower()} {phrase}.",
        f"Necesito que {verb.lower()}s {phrase}.",  # rough
    ]
    text = rng.choice(templates[:3])
    params = {"object_type": obj, "object_color": color, "quantity": q[0]}
    return text, [_action("PICK_OBJECT", params)], "pick_simple"


def gen_place_simple(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    q = rqty(rng)
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    phrase = make_object_phrase(obj, og, color, q)
    destp = make_dest_phrase(dest, dg, dcolor)
    verb = rchoice(rng, VERBS_PLACE)
    structures = [
        f"{verb} {phrase} en {destp}.",
        f"{verb} {phrase} a {destp}.",
        f"En {destp} {verb.lower()} {phrase}.",
        f"{phrase.capitalize()} {verb.lower()}las en {destp}." if False else f"{verb} {phrase} en {destp}.",
    ]
    text = rng.choice(structures[:3])
    params = {
        "object_type": obj, "object_color": color, "quantity": q[0],
        "destination_type": dest, "destination_color": dcolor,
    }
    return text, [_action("PLACE_OBJECT", params)], "place_simple"


def gen_move(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    q = rqty(rng)
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    phrase = make_object_phrase(obj, og, color, q)
    destp = make_dest_phrase(dest, dg, dcolor)
    verb = rchoice(rng, VERBS_MOVE)
    text = rng.choice([
        f"{verb} {phrase} a {destp}.",
        f"{verb} {phrase} hacia {destp}.",
        f"Transfiere {phrase} al {dest} {color_word(dcolor, dg, 'sg')}.",
    ])
    params = {
        "object_type": obj, "object_color": color, "quantity": q[0],
        "destination_type": dest, "destination_color": dcolor,
    }
    return text, [_action("MOVE_OBJECT", params)], "move"


def gen_inspect_single(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    phrase = f"la {obj} {color_word(color, og, 'sg')}" if og == "f" else f"el {obj} {color_word(color, og, 'sg')}"
    verb = rchoice(rng, VERBS_INSPECT)
    text = rng.choice([
        f"{verb} {phrase}.",
        f"{verb} la pieza {color_word(color, 'f', 'sg')}." if obj != "pieza" else f"{verb} {phrase}.",
    ])
    params = {"object_type": obj, "object_color": color, "quantity": 1}
    return text, [_action("INSPECT_OBJECT", params)], "inspect_single"


def gen_inspect_batch(rng):
    obj, og = robject(rng)
    plural = pluralize(obj)
    verb = rchoice(rng, VERBS_INSPECT)
    structures = [
        f"{verb} todas las {plural}.",
        f"{verb} todos los {plural}.",
        f"Revisa todas las piezas.",
        f"Inspecciona el lote de {plural}.",
        f"{verb} todas las piezas del contenedor.",
    ]
    text = rng.choice(structures)
    params = {"object_type": obj, "quantity": "ALL"}
    return text, [_action("INSPECT_BATCH", params)], "inspect_batch"


def gen_classify(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    text = rng.choice([
        f"Clasifica esta {obj}.",
        f"Clasifica la {obj} {color_word(color, og, 'sg')}.",
        f"Determina la calidad de la {obj}.",
    ])
    params = {"object_type": obj, "object_color": color if "color_word" in text else None}
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("quantity", 1)
    return text, [_action("CLASSIFY_OBJECT", params)], "classify"


def gen_sort(rng):
    text = rng.choice([
        "Separa las piezas buenas de las defectuosas.",
        "Clasifica las buenas y las defectuosas.",
        "Ordena las piezas por calidad.",
        "Separa los componentes buenos de los malos.",
    ])
    return text, [_action("SORT_OBJECTS", {"quantity": "ALL"})], "sort"


def gen_change_quality_rule(rng):
    color = rcolor(rng)
    gender = rng.choice(["f", "m"])
    # phrase like "las piezas rojas ahora son defectuosas"
    quality = rng.choice(["defectuosas", "buenas", "malas"])
    qcanon = "defectuoso" if quality in ("defectuosas", "malas") else "bueno"
    obj = rng.choice(["piezas", "cubos", "componentes"])
    og = "f" if obj.endswith("as") else "m"
    colorf = color_word(color, og[0] if False else ("f" if obj.endswith("as") else "m"), "pl")
    # build
    text = rng.choice([
        f"Las {obj} {colorf} ahora son {quality}.",
        f"A partir de ahora las {obj} {colorf} son {quality}.",
        f"Considera las {obj} {colorf} como {quality}.",
        f"Las piezas de color {color} son {quality}.",
    ])
    params = {"object_type": "pieza", "object_color": color, "object_quality": qcanon}
    return text, [_action("CHANGE_QUALITY_RULE", params)], "change_quality_rule"


def gen_change_destination(rng):
    quality = rng.choice(["malas", "defectuosas", "buenas"])
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    destp = make_dest_phrase(dest, dg, dcolor)
    text = rng.choice([
        f"Las piezas {quality} deben ir a {destp}.",
        f"Los componentes {quality} van a {destp}.",
        f"Las piezas defectuosas deben ir a la caja {color_word(dcolor, dg, 'sg')}.",
    ])
    params = {"destination_type": dest, "destination_color": dcolor, "object_quality": "defectuoso" if quality in ("malas", "defectuosas") else "bueno"}
    return text, [_action("CHANGE_DESTINATION", params)], "change_destination"


def gen_mark_good(rng):
    text = rng.choice([
        "Marca esta pieza como buena.",
        "Esta pieza es aceptable.",
        "Marca este componente como válido.",
    ])
    return text, [_action("MARK_GOOD", {"object_quality": "bueno"})], "mark_good"


def gen_mark_defective(rng):
    text = rng.choice([
        "Esta pieza está defectuosa.",
        "Marca esta pieza como defectuosa.",
        "Este componente está dañado.",
    ])
    return text, [_action("MARK_DEFECTIVE", {"object_quality": "defectuoso"})], "mark_defective"


def gen_quarantine(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    text = rng.choice([
        f"Deja esta {obj} en revisión.",
        f"Pon la {obj} {color_word(color, og, 'sg')} en revisión.",
        f"Envía esta pieza a revisión.",
    ])
    params = {"object_type": obj, "object_color": color, "destination_role": "REVIEW_BIN"}
    return text, [_action("QUARANTINE_OBJECT", params)], "quarantine"


def gen_emergency(rng):
    text = rng.choice([
        "Parada de emergencia.",
        "Parada de emergencia inmediata.",
        "Emergencia, detén todo.",
        "Para de emergencia.",
    ])
    return text, [_action("EMERGENCY_STOP", {})], "emergency"


def gen_stop(rng):
    text = rng.choice([
        "Detén el proceso.",
        "Para el robot.",
        "Detener la operación.",
        "Para la máquina.",
    ])
    return text, [_action("STOP_PROCESS", {})], "stop"


def gen_pause(rng):
    text = rng.choice([
        "Pausa la operación.",
        "Pausa el proceso.",
        "Espera un momento.",
        "Detente un momento.",
    ])
    return text, [_action("PAUSE_PROCESS", {})], "pause"


def gen_resume(rng):
    text = rng.choice([
        "Continúa trabajando.",
        "Reanuda el proceso.",
        "Sigue con la operación.",
        "Continúa.",
    ])
    return text, [_action("RESUME_PROCESS", {})], "resume"


def gen_reset(rng):
    text = rng.choice([
        "Reinicia el sistema.",
        "Reinicia el proceso.",
        "Reset del sistema.",
    ])
    return text, [_action("RESET_SYSTEM", {})], "reset"


def gen_start(rng):
    text = rng.choice([
        "Empieza la clasificación.",
        "Inicia el proceso.",
        "Arranca la operación.",
        "Comienza la inspección.",
    ])
    return text, [_action("START_PROCESS", {})], "start"


def gen_status(rng):
    text = rng.choice([
        "¿Cuántas piezas quedan?",
        "¿Cuántos cubos quedan?",
        "Estado del sistema.",
        "¿Cuántas piezas hay?",
    ])
    return text, [_action("REQUEST_STATUS", {})], "status"


def gen_report(rng):
    text = rng.choice([
        "Muéstrame el informe.",
        "Dame el reporte de calidad.",
        "Informe de calidad.",
    ])
    return text, [_action("REQUEST_REPORT", {})], "report"


def gen_object_info(rng):
    text = rng.choice([
        "¿De qué color es esta pieza?",
        "¿Qué es este objeto?",
        "¿De qué color es el cubo?",
    ])
    return text, [_action("REQUEST_OBJECT_INFO", {})], "object_info"


def gen_calibrate(rng):
    text = rng.choice([
        "Calibra la cámara.",
        "Calibrar el sistema.",
        "Calibra el robot.",
    ])
    return text, [_action("CALIBRATE_SYSTEM", {})], "calibrate"


def gen_speed(rng):
    text = rng.choice([
        "Reduce la velocidad.",
        "Aumenta la velocidad.",
        "Más rápido.",
        "Más lento.",
    ])
    return text, [_action("SET_SPEED", {})], "speed"


def gen_retry(rng):
    text = rng.choice([
        "Intenta cogerla otra vez.",
        "Reintenta la acción.",
        "Intenta de nuevo.",
    ])
    return text, [_action("RETRY_ACTION", {})], "retry"


def gen_skip(rng):
    text = rng.choice([
        "Salta esta pieza.",
        "Omite este objeto.",
        "Pasa a la siguiente pieza.",
    ])
    return text, [_action("SKIP_OBJECT", {})], "skip"


def gen_replan(rng):
    text = rng.choice([
        "Busca otra forma de hacerlo.",
        "Recalcula el plan.",
        "Replanifica la tarea.",
    ])
    return text, [_action("REPLAN_TASK", {})], "replan"


def gen_cancel(rng):
    text = rng.choice([
        "Cancela el último movimiento.",
        "Cancela la acción.",
        "Anula el movimiento.",
    ])
    return text, [_action("CANCEL_ACTION", {})], "cancel"


def gen_confirm(rng):
    text = rng.choice([
        "Sí, confirma la operación.",
        "Confirma la acción.",
        "De acuerdo, procede.",
    ])
    return text, [_action("CONFIRM_ACTION", {})], "confirm"


def gen_release(rng):
    text = rng.choice([
        "Suelta la pieza.",
        "Libera el objeto.",
        "Suelta.",
    ])
    return text, [_action("RELEASE_OBJECT", {})], "release"


def gen_return(rng):
    text = rng.choice([
        "Devuelve la pieza a su posición inicial.",
        "Retorna el objeto.",
        "Devuelve la pieza.",
    ])
    return text, [_action("RETURN_OBJECT", {})], "return"


def gen_unknown(rng):
    text = rng.choice([
        "¿Qué tiempo hace mañana?",
        "Pon música.",
        "Cuéntame un chiste.",
        "Haz algo con las piezas.",
        "Abre la ventana.",
        "Baila.",
    ])
    return text, [_action("UNKNOWN", {})], "oos_generic"


def gen_negation_pick(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    cw = color_word(color, og, "pl")
    plural = pluralize(obj)
    text = rng.choice([
        f"No cojas las {plural} {cw}.",
        f"No cojas la {obj} {color_word(color, og, 'sg')}.",
        f"No agarres los {plural} {cw}.",
    ])
    params = {"object_type": obj, "object_color": color, "quantity": "ALL"}
    return text, [_action("PICK_OBJECT", params, negated=True)], "neg_pick"


def gen_negation_place(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    text = rng.choice([
        f"No dejes la pieza roja en la caja verde.".replace("roja", color_word(color, "f", "sg")).replace("verde", color_word(dcolor, dg, "sg")),
        f"No pongas las piezas {color_word(color, og, 'pl')} en {make_dest_phrase(dest, dg, dcolor)}.",
    ])
    params = {"object_type": obj, "object_color": color, "quantity": "ALL",
              "destination_type": dest, "destination_color": dcolor}
    return text, [_action("PLACE_OBJECT", params, negated=True)], "neg_place"


def gen_except(rng):
    color = rcolor(rng)
    text = rng.choice([
        f"Inspecciona todas excepto las {color_word(color, 'f', 'pl')}.",
        f"Coge todas menos la {color_word(color, 'f', 'sg')}.",
    ])
    return text, [_action("INSPECT_BATCH", {"quantity": "ALL"}, negated=False)], "except"


def gen_compound_two(rng):
    # pick + inspect + place
    obj, og = robject(rng)
    color = rcolor(rng)
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    cw = color_word(color, og, "sg")
    destp = make_dest_phrase(dest, dg, dcolor)
    variants = [
        f"Coge la {obj} {cw}, inspecciónala y déjala en {destp}.",
        f"Coge la {obj} {cw} e inspecciónala.",
        f"Inspecciona la {obj} {cw} y déjala en {destp}.",
    ]
    text = rng.choice(variants)
    actions = []
    if "Coge" in text:
        actions.append(_action("PICK_OBJECT", {"object_type": obj, "object_color": color, "quantity": 1}))
    if "inspecc" in text.lower() or "inspección" in text.lower():
        actions.append(_action("INSPECT_OBJECT", {"object_type": obj, "object_color": color, "quantity": 1}))
    if "déjala" in text or "deja" in text.lower():
        actions.append(_action("PLACE_OBJECT", {"object_type": obj, "object_color": color, "quantity": 1,
                                                "destination_type": dest, "destination_color": dcolor}))
    if not actions:
        actions = [_action("PICK_OBJECT", {"object_type": obj, "object_color": color, "quantity": 1})]
    return text, actions, "compound_two"


def gen_compound_colors(rng):
    c1 = rcolor(rng)
    c2 = rcolor(rng)
    while c2 == c1:
        c2 = rcolor(rng)
    dest1, dg1 = rdest(rng)
    dest2, dg2 = rdest(rng)
    d1 = rcolor(rng)
    d2 = rcolor(rng)
    text = rng.choice([
        f"Pon las {color_word(c1, 'f', 'pl')} en {make_dest_phrase(dest1, dg1, d1)} y las {color_word(c2, 'f', 'pl')} en {make_dest_phrase(dest2, dg2, d2)}.",
        f"Deja las rojas en la verde y las azules en la amarilla.".replace("rojas", color_word(c1, 'f', 'pl')).replace("verde", color_word(d1, dg1, 'sg')).replace("azules", color_word(c2, 'f', 'pl')).replace("amarilla", color_word(d2, dg2, 'sg')),
    ])
    a1 = _action("PLACE_OBJECT", {"object_type": "pieza", "object_color": c1, "quantity": "ALL",
                                  "destination_type": dest1, "destination_color": d1})
    a2 = _action("PLACE_OBJECT", {"object_type": "pieza", "object_color": c2, "quantity": "ALL",
                                  "destination_type": dest2, "destination_color": d2})
    return text, [a1, a2], "compound_colors"


def gen_correction(rng):
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    text = rng.choice([
        "No, mejor en la caja azul.",
        f"No, mejor en {make_dest_phrase(dest, dg, dcolor)}.",
        "Espera, ponlas en la azul.",
    ])
    params = {"destination_type": dest, "destination_color": dcolor}
    return text, [_action("PLACE_OBJECT", params)], "correction"


def gen_move_qty(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    q = rng.choice([q for q in QUANTITIES if isinstance(q[0], int) and q[0] >= 2])
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    phrase = make_object_phrase(obj, og, color, q)
    destp = make_dest_phrase(dest, dg, dcolor)
    text = f"Mueve {phrase} a {destp}."
    params = {"object_type": obj, "object_color": color, "quantity": q[0],
              "destination_type": dest, "destination_color": dcolor}
    return text, [_action("MOVE_OBJECT", params)], "move_qty"


def gen_place_all(rng):
    obj, og = robject(rng)
    color = rcolor(rng)
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    text = f"Deja todas las {pluralize(obj)} {color_word(color, og, 'pl')} en {make_dest_phrase(dest, dg, dcolor)}."
    params = {"object_type": obj, "object_color": color, "quantity": "ALL",
              "destination_type": dest, "destination_color": dcolor}
    return text, [_action("PLACE_OBJECT", params)], "place_all"


def gen_incomplete(rng):
    text = rng.choice([
        "Coge las piezas.",
        "Deja las piezas.",
        "Mueve las piezas.",
    ])
    intent = "PICK_OBJECT" if text.startswith("Coge") else ("PLACE_OBJECT" if text.startswith("Deja") else "MOVE_OBJECT")
    params = {"object_type": "pieza", "quantity": "UNSPECIFIED"}
    return text, [_action(intent, params)], "incomplete"


def gen_speech_noise(rng):
    # take a base sentence and add noise
    base, actions, fam = gen_place_simple(rng)
    noise_ops = [
        lambda s: s.replace("las", "la").replace("los", "lo"),
        lambda s: s + " eh",
        lambda s: s.replace("caja", "cajas"),
    ]
    text = rng.choice(noise_ops)(base)
    return text, actions, "speech_noise"


# ---------------------------------------------------------------------------
# Generic pattern-based templates (scalable)
# ---------------------------------------------------------------------------

def _build_context(rng: random.Random) -> Dict[str, Any]:
    obj, og = robject(rng)
    color = rcolor(rng)
    dest, dg = rdest(rng)
    dcolor = rcolor(rng)
    q = rqty(rng)
    quality = rquality(rng)
    return {
        "obj": obj,
        "og": og,
        "color": color,
        "dest": dest,
        "dg": dg,
        "dcolor": dcolor,
        "q": q,
        "quality": quality,
        "obj_phrase": make_object_phrase(obj, og, color, q),
        "dest_phrase": make_dest_phrase(dest, dg, dcolor),
        "color_obj_sg": color_word(color, og, "sg"),
        "color_obj_pl": color_word(color, og, "pl"),
        "color_dest_sg": color_word(dcolor, dg, "sg"),
        "color_dest_pl": color_word(dcolor, dg, "pl"),
        "qty_val": q[0],
        "verb_pick": rng.choice(VERBS_PICK),
        "verb_place": rng.choice(VERBS_PLACE),
        "verb_move": rng.choice(VERBS_MOVE),
        "verb_inspect": rng.choice(VERBS_INSPECT),
    }


def _run_pattern(rng: random.Random, spec: Dict[str, Any], idx: int):
    ctx = _build_context(rng)
    text = spec["text"].format(**ctx)
    text = " ".join(text.split())
    params = {}
    for param, key in spec.get("params", {}).items():
        # keys name context values; anything else is a literal such as "ALL" or "bueno"
        params[param] = ctx[key] if key in ctx else key
    if "quantity" not in params:
        # the quantity is only in the text when the object phrase carries it
        params["quantity"] = ctx["qty_val"] if "{obj_phrase}" in spec["text"] else "UNSPECIFIED"
    action = {"intent": spec["intent"], "parameters": params, "negated": spec.get("negated", False)}
    family = spec.get("family", f"pattern_{idx:03d}")
    return text, [action], family


# A large catalog of distinct Spanish phrasings. Each entry is one template.
PATTERN_SPECS: List[Dict[str, Any]] = [
    {"text": "{verb_pick} {obj_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_001"},
    {"text": "{verb_pick} la {obj} {color_obj_sg}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_002"},
    {"text": "Por favor {verb_pick} {obj_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_003"},
    {"text": "Necesito que cojas {obj_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_004"},
    {"text": "Agarra {obj_phrase} ahora.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_005"},
    {"text": "Toma {obj_phrase} de la mesa.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_006"},
    {"text": "Recoge {obj_phrase} del contenedor.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_007"},
    {"text": "Coge {obj_phrase} con cuidado.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_008"},
    {"text": "Sujeta {obj_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_009"},
    {"text": "Levanta {obj_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_pick_010"},

    {"text": "{verb_place} {obj_phrase} en {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_001"},
    {"text": "{verb_place} {obj_phrase} a {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_002"},
    {"text": "En {dest_phrase} {verb_place} {obj_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_003"},
    {"text": "Coloca {obj_phrase} en {dest_phrase} por favor.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_004"},
    {"text": "Deja {obj_phrase} sobre {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_005"},
    {"text": "Pon {obj_phrase} dentro de {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_006"},
    {"text": "Situa {obj_phrase} en {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_007"},
    {"text": "Deja {obj_phrase} en {dest_phrase} ahora.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_008"},
    {"text": "Coloca {obj_phrase} a {dest_phrase} con cuidado.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_009"},
    {"text": "Pon {obj_phrase} en {dest_phrase} lentamente.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_place_010"},

    {"text": "{verb_move} {obj_phrase} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_001"},
    {"text": "{verb_move} {obj_phrase} hacia {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_002"},
    {"text": "Transfiere {obj_phrase} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_003"},
    {"text": "Transporta {obj_phrase} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_004"},
    {"text": "Mueve {obj_phrase} hasta {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_005"},
    {"text": "Lleva {obj_phrase} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_006"},
    {"text": "Desplaza {obj_phrase} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_007"},
    {"text": "Mueve {obj_phrase} hacia {dest_phrase} ahora.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_008"},
    {"text": "Transfiere {obj_phrase} hasta {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_009"},
    {"text": "Transporta {obj_phrase} hasta {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_move_010"},

    {"text": "{verb_inspect} {obj_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_001"},
    {"text": "{verb_inspect} la {obj} {color_obj_sg}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_002"},
    {"text": "Revisa {obj_phrase} por favor.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_003"},
    {"text": "Inspecciona {obj_phrase} ahora.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_004"},
    {"text": "Comprueba {obj_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_005"},
    {"text": "Analiza {obj_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_006"},
    {"text": "Examina {obj_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_007"},
    {"text": "Verifica {obj_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_inspect_008"},

    {"text": "Inspecciona todas las {obj}s.", "intent": "INSPECT_BATCH", "params": {"object_type": "obj", "quantity": "ALL"}, "family": "pat_batch_001"},
    {"text": "Revisa todos los {obj}s.", "intent": "INSPECT_BATCH", "params": {"object_type": "obj", "quantity": "ALL"}, "family": "pat_batch_002"},
    {"text": "Inspecciona todas las piezas.", "intent": "INSPECT_BATCH", "params": {"object_type": "pieza", "quantity": "ALL"}, "family": "pat_batch_003"},
    {"text": "Revisa el lote completo.", "intent": "INSPECT_BATCH", "params": {"object_type": "pieza", "quantity": "ALL"}, "family": "pat_batch_004"},
    {"text": "Inspecciona todas las piezas del contenedor.", "intent": "INSPECT_BATCH", "params": {"object_type": "pieza", "quantity": "ALL"}, "family": "pat_batch_005"},
    {"text": "Revisa todas las piezas de la cinta.", "intent": "INSPECT_BATCH", "params": {"object_type": "pieza", "quantity": "ALL"}, "family": "pat_batch_006"},

    {"text": "Clasifica {obj_phrase}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_classify_001"},
    {"text": "Clasifica esta {obj}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj"}, "family": "pat_classify_002"},
    {"text": "Determina la calidad de {obj_phrase}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_classify_003"},
    {"text": "Evalua {obj_phrase}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_classify_004"},
    {"text": "Clasifica la pieza {color_obj_sg}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "pieza", "object_color": "color"}, "family": "pat_classify_005"},

    {"text": "Separa las piezas buenas de las defectuosas.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat_sort_001"},
    {"text": "Clasifica las buenas y las defectuosas.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat_sort_002"},
    {"text": "Ordena las piezas por calidad.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat_sort_003"},
    {"text": "Separa los componentes buenos de los malos.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat_sort_004"},
    {"text": "Separa las piezas por color.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat_sort_005"},
    {"text": "Ordena los cubos por tamaño.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat_sort_006"},

    {"text": "Las {obj}s {color_obj_pl} ahora son defectuosas.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "obj", "object_color": "color", "object_quality": "defectuoso"}, "family": "pat_cqr_001"},
    {"text": "A partir de ahora las {obj}s {color_obj_pl} son buenas.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "obj", "object_color": "color", "object_quality": "bueno"}, "family": "pat_cqr_002"},
    {"text": "Considera las {obj}s {color_obj_pl} como defectuosas.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "obj", "object_color": "color", "object_quality": "defectuoso"}, "family": "pat_cqr_003"},
    {"text": "Las piezas de color {color} son malas.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "pieza", "object_color": "color", "object_quality": "defectuoso"}, "family": "pat_cqr_004"},
    {"text": "Desde ahora las piezas {color_obj_pl} son aceptables.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "pieza", "object_color": "color", "object_quality": "bueno"}, "family": "pat_cqr_005"},

    {"text": "Las piezas malas deben ir a {dest_phrase}.", "intent": "CHANGE_DESTINATION", "params": {"destination_type": "dest", "destination_color": "dcolor", "object_quality": "defectuoso"}, "family": "pat_cd_001"},
    {"text": "Los componentes defectuosos van a {dest_phrase}.", "intent": "CHANGE_DESTINATION", "params": {"destination_type": "dest", "destination_color": "dcolor", "object_quality": "defectuoso"}, "family": "pat_cd_002"},
    {"text": "Las piezas buenas deben ir a {dest_phrase}.", "intent": "CHANGE_DESTINATION", "params": {"destination_type": "dest", "destination_color": "dcolor", "object_quality": "bueno"}, "family": "pat_cd_003"},
    {"text": "Las piezas defectuosas deben ir a la caja {color_dest_sg}.", "intent": "CHANGE_DESTINATION", "params": {"destination_type": "caja", "destination_color": "dcolor", "object_quality": "defectuoso"}, "family": "pat_cd_004"},
    {"text": "Los objetos malos van a {dest_phrase}.", "intent": "CHANGE_DESTINATION", "params": {"destination_type": "dest", "destination_color": "dcolor", "object_quality": "defectuoso"}, "family": "pat_cd_005"},

    {"text": "Marca esta pieza como buena.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat_markg_001"},
    {"text": "Marca este componente como válido.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat_markg_002"},
    {"text": "Esta pieza es aceptable.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat_markg_003"},
    {"text": "Marca la pieza como correcta.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat_markg_004"},
    {"text": "Esta pieza es buena.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat_markg_005"},

    {"text": "Esta pieza está defectuosa.", "intent": "MARK_DEFECTIVE", "params": {"object_quality": "defectuoso"}, "family": "pat_markd_001"},
    {"text": "Marca esta pieza como defectuosa.", "intent": "MARK_DEFECTIVE", "params": {"object_quality": "defectuoso"}, "family": "pat_markd_002"},
    {"text": "Este componente está dañado.", "intent": "MARK_DEFECTIVE", "params": {"object_quality": "defectuoso"}, "family": "pat_markd_003"},
    {"text": "Esta pieza está rota.", "intent": "MARK_DEFECTIVE", "params": {"object_quality": "defectuoso"}, "family": "pat_markd_004"},
    {"text": "Marca la pieza como mala.", "intent": "MARK_DEFECTIVE", "params": {"object_quality": "defectuoso"}, "family": "pat_markd_005"},

    {"text": "Deja esta {obj} en revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "obj", "destination_role": "REVIEW_BIN"}, "family": "pat_quar_001"},
    {"text": "Pon la {obj} {color_obj_sg} en revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_role": "REVIEW_BIN"}, "family": "pat_quar_002"},
    {"text": "Envía esta pieza a revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "pieza", "destination_role": "REVIEW_BIN"}, "family": "pat_quar_003"},
    {"text": "Deja esta pieza en la zona de revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "pieza", "destination_role": "REVIEW_BIN"}, "family": "pat_quar_004"},
    {"text": "Envía el componente a revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "componente", "destination_role": "REVIEW_BIN"}, "family": "pat_quar_005"},

    {"text": "Parada de emergencia.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat_emerg_001"},
    {"text": "Parada de emergencia inmediata.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat_emerg_002"},
    {"text": "Emergencia, detén todo.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat_emerg_003"},
    {"text": "Para de emergencia.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat_emerg_004"},
    {"text": "Parada de emergencia ahora.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat_emerg_005"},

    {"text": "Detén el proceso.", "intent": "STOP_PROCESS", "params": {}, "family": "pat_stop_001"},
    {"text": "Para el robot.", "intent": "STOP_PROCESS", "params": {}, "family": "pat_stop_002"},
    {"text": "Detener la operación.", "intent": "STOP_PROCESS", "params": {}, "family": "pat_stop_003"},
    {"text": "Para la máquina.", "intent": "STOP_PROCESS", "params": {}, "family": "pat_stop_004"},
    {"text": "Detén la cinta.", "intent": "STOP_PROCESS", "params": {}, "family": "pat_stop_005"},

    {"text": "Pausa la operación.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat_pause_001"},
    {"text": "Pausa el proceso.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat_pause_002"},
    {"text": "Espera un momento.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat_pause_003"},
    {"text": "Detente un momento.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat_pause_004"},
    {"text": "Pausa la cinta.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat_pause_005"},

    {"text": "Continúa trabajando.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat_resume_001"},
    {"text": "Reanuda el proceso.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat_resume_002"},
    {"text": "Sigue con la operación.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat_resume_003"},
    {"text": "Continúa.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat_resume_004"},
    {"text": "Reanuda la cinta.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat_resume_005"},

    {"text": "Reinicia el sistema.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat_reset_001"},
    {"text": "Reinicia el proceso.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat_reset_002"},
    {"text": "Reset del sistema.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat_reset_003"},
    {"text": "Reinicia la máquina.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat_reset_004"},
    {"text": "Reinicia la cinta.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat_reset_005"},

    {"text": "Empieza la clasificación.", "intent": "START_PROCESS", "params": {}, "family": "pat_start_001"},
    {"text": "Inicia el proceso.", "intent": "START_PROCESS", "params": {}, "family": "pat_start_002"},
    {"text": "Arranca la operación.", "intent": "START_PROCESS", "params": {}, "family": "pat_start_003"},
    {"text": "Comienza la inspección.", "intent": "START_PROCESS", "params": {}, "family": "pat_start_004"},
    {"text": "Inicia la cinta.", "intent": "START_PROCESS", "params": {}, "family": "pat_start_005"},

    {"text": "¿Cuántas piezas quedan?", "intent": "REQUEST_STATUS", "params": {}, "family": "pat_status_001"},
    {"text": "¿Cuántos cubos quedan?", "intent": "REQUEST_STATUS", "params": {}, "family": "pat_status_002"},
    {"text": "Estado del sistema.", "intent": "REQUEST_STATUS", "params": {}, "family": "pat_status_003"},
    {"text": "¿Cuántas piezas hay?", "intent": "REQUEST_STATUS", "params": {}, "family": "pat_status_004"},
    {"text": "¿Qué queda por procesar?", "intent": "REQUEST_STATUS", "params": {}, "family": "pat_status_005"},

    {"text": "Muéstrame el informe.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat_report_001"},
    {"text": "Dame el reporte de calidad.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat_report_002"},
    {"text": "Informe de calidad.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat_report_003"},
    {"text": "Muestra el reporte.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat_report_004"},
    {"text": "Dame el informe final.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat_report_005"},

    {"text": "¿De qué color es esta pieza?", "intent": "REQUEST_OBJECT_INFO", "params": {}, "family": "pat_info_001"},
    {"text": "¿Qué es este objeto?", "intent": "REQUEST_OBJECT_INFO", "params": {}, "family": "pat_info_002"},
    {"text": "¿De qué color es el cubo?", "intent": "REQUEST_OBJECT_INFO", "params": {}, "family": "pat_info_003"},
    {"text": "¿Qué pieza es esta?", "intent": "REQUEST_OBJECT_INFO", "params": {}, "family": "pat_info_004"},
    {"text": "Dime el color de esta pieza.", "intent": "REQUEST_OBJECT_INFO", "params": {}, "family": "pat_info_005"},

    {"text": "Calibra la cámara.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat_calib_001"},
    {"text": "Calibrar el sistema.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat_calib_002"},
    {"text": "Calibra el robot.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat_calib_003"},
    {"text": "Calibra la cinta.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat_calib_004"},
    {"text": "Calibración del sistema.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat_calib_005"},

    {"text": "Reduce la velocidad.", "intent": "SET_SPEED", "params": {}, "family": "pat_speed_001"},
    {"text": "Aumenta la velocidad.", "intent": "SET_SPEED", "params": {}, "family": "pat_speed_002"},
    {"text": "Más rápido.", "intent": "SET_SPEED", "params": {}, "family": "pat_speed_003"},
    {"text": "Más lento.", "intent": "SET_SPEED", "params": {}, "family": "pat_speed_004"},
    {"text": "Cambia la velocidad.", "intent": "SET_SPEED", "params": {}, "family": "pat_speed_005"},

    {"text": "Intenta cogerla otra vez.", "intent": "RETRY_ACTION", "params": {}, "family": "pat_retry_001"},
    {"text": "Reintenta la acción.", "intent": "RETRY_ACTION", "params": {}, "family": "pat_retry_002"},
    {"text": "Intenta de nuevo.", "intent": "RETRY_ACTION", "params": {}, "family": "pat_retry_003"},
    {"text": "Repite el movimiento.", "intent": "RETRY_ACTION", "params": {}, "family": "pat_retry_004"},
    {"text": "Vuelve a intentarlo.", "intent": "RETRY_ACTION", "params": {}, "family": "pat_retry_005"},

    {"text": "Salta esta pieza.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat_skip_001"},
    {"text": "Omite este objeto.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat_skip_002"},
    {"text": "Pasa a la siguiente pieza.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat_skip_003"},
    {"text": "Salta este componente.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat_skip_004"},
    {"text": "Omite esta pieza.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat_skip_005"},

    {"text": "Busca otra forma de hacerlo.", "intent": "REPLAN_TASK", "params": {}, "family": "pat_replan_001"},
    {"text": "Recalcula el plan.", "intent": "REPLAN_TASK", "params": {}, "family": "pat_replan_002"},
    {"text": "Replanifica la tarea.", "intent": "REPLAN_TASK", "params": {}, "family": "pat_replan_003"},
    {"text": "Busca un plan alternativo.", "intent": "REPLAN_TASK", "params": {}, "family": "pat_replan_004"},
    {"text": "Recalcula la ruta.", "intent": "REPLAN_TASK", "params": {}, "family": "pat_replan_005"},

    {"text": "Cancela el último movimiento.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat_cancel_001"},
    {"text": "Cancela la acción.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat_cancel_002"},
    {"text": "Anula el movimiento.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat_cancel_003"},
    {"text": "Cancela la operación.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat_cancel_004"},
    {"text": "Anula la acción.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat_cancel_005"},

    {"text": "Sí, confirma la operación.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat_confirm_001"},
    {"text": "Confirma la acción.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat_confirm_002"},
    {"text": "De acuerdo, procede.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat_confirm_003"},
    {"text": "Confirma el movimiento.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat_confirm_004"},
    {"text": "Acepta la acción.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat_confirm_005"},

    {"text": "Suelta la pieza.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat_release_001"},
    {"text": "Libera el objeto.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat_release_002"},
    {"text": "Suelta.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat_release_003"},
    {"text": "Libera la pieza.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat_release_004"},
    {"text": "Suelta el componente.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat_release_005"},

    {"text": "Devuelve la pieza a su posición inicial.", "intent": "RETURN_OBJECT", "params": {}, "family": "pat_return_001"},
    {"text": "Retorna el objeto.", "intent": "RETURN_OBJECT", "params": {}, "family": "pat_return_002"},
    {"text": "Devuelve la pieza.", "intent": "RETURN_OBJECT", "params": {}, "family": "pat_return_003"},
    {"text": "Devuelve el componente.", "intent": "RETURN_OBJECT", "params": {}, "family": "pat_return_004"},
    {"text": "Retorna la pieza a su sitio.", "intent": "RETURN_OBJECT", "params": {}, "family": "pat_return_005"},

    {"text": "¿Qué tiempo hace mañana?", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_001"},
    {"text": "Pon música.", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_002"},
    {"text": "Cuéntame un chiste.", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_003"},
    {"text": "Haz algo con las piezas.", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_004"},
    {"text": "Abre la ventana.", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_005"},
    {"text": "Baila.", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_006"},
    {"text": "Canta una canción.", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_007"},
    {"text": "¿Qué hora es?", "intent": "UNKNOWN", "params": {}, "family": "pat_oos_008"},

    {"text": "No cojas {obj_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "negated": True, "family": "pat_neg_001"},
    {"text": "No dejes {obj_phrase} en {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "negated": True, "family": "pat_neg_002"},
    {"text": "No muevas {obj_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "negated": True, "family": "pat_neg_003"},
    {"text": "No inspecciones {obj_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "negated": True, "family": "pat_neg_004"},
    {"text": "No clasifiques {obj_phrase}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "negated": True, "family": "pat_neg_005"},

    {"text": "Coge {obj_phrase}, inspecciónala y déjala en {dest_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_comp_001"},
    {"text": "Inspecciona {obj_phrase} y déjala en {dest_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_comp_002"},
    {"text": "Coge {obj_phrase} y ponla en {dest_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_comp_003"},
    {"text": "Mueve {obj_phrase} a {dest_phrase} y clasifícala.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_comp_004"},
    {"text": "Inspecciona {obj_phrase} y clasifícala.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_comp_005"},

    {"text": "Coge las {obj}s {color_obj_pl}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": "ALL"}, "family": "pat_all_001"},
    {"text": "Deja todas las {obj}s {color_obj_pl} en {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": "ALL", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_all_002"},
    {"text": "Mueve todos los {obj}s {color_obj_pl} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": "ALL", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_all_003"},
    {"text": "Inspecciona todas las {obj}s {color_obj_pl}.", "intent": "INSPECT_BATCH", "params": {"object_type": "obj", "object_color": "color", "quantity": "ALL"}, "family": "pat_all_004"},
    {"text": "Clasifica todas las {obj}s {color_obj_pl}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": "ALL"}, "family": "pat_all_005"},

    {"text": "Coge tres {obj}s {color_obj_pl}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": 3}, "family": "pat_qty_001"},
    {"text": "Deja dos {obj}s {color_obj_pl} en {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": 2, "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_qty_002"},
    {"text": "Mueve cinco {obj}s {color_obj_pl} a {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": 5, "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_qty_003"},
    {"text": "Inspecciona cuatro {obj}s {color_obj_pl}.", "intent": "INSPECT_BATCH", "params": {"object_type": "obj", "object_color": "color", "quantity": 4}, "family": "pat_qty_004"},
    {"text": "Clasifica seis {obj}s {color_obj_pl}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color", "quantity": 6}, "family": "pat_qty_005"},

    {"text": "Coge la {obj} {color_obj_sg} y déjala en {dest_phrase}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_mix_001"},
    {"text": "Deja la {obj} {color_obj_sg} en {dest_phrase} por favor.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_mix_002"},
    {"text": "Mueve la {obj} {color_obj_sg} a {dest_phrase} ahora.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_mix_003"},
    {"text": "Inspecciona la {obj} {color_obj_sg} y déjala en {dest_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_mix_004"},
    {"text": "Clasifica la {obj} {color_obj_sg} y ponla en {dest_phrase}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_mix_005"},

    {"text": "Coge {obj_phrase} de la caja {color_dest_sg}.", "intent": "PICK_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat_src_001"},
    {"text": "Deja {obj_phrase} junto a {dest_phrase}.", "intent": "PLACE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_src_002"},
    {"text": "Mueve {obj_phrase} desde {dest_phrase}.", "intent": "MOVE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_src_003"},
    {"text": "Inspecciona {obj_phrase} en {dest_phrase}.", "intent": "INSPECT_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_src_004"},
    {"text": "Clasifica {obj_phrase} sobre {dest_phrase}.", "intent": "CLASSIFY_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_type": "dest", "destination_color": "dcolor"}, "family": "pat_src_005"},

    # Second catalogue: control, status, recovery and workflow-rule phrasings (families pat2_*).
    {"text": "Inicia el proceso.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_001"},
    {"text": "Empieza el proceso.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_002"},
    {"text": "Arranca la celda.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_003"},
    {"text": "Comienza el flujo de trabajo.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_004"},
    {"text": "Empieza ya.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_005"},
    {"text": "Comienza la tarea del lote.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_006"},
    {"text": "Inicia el turno de trabajo.", "intent": "START_PROCESS", "params": {}, "family": "pat2_start_007"},

    {"text": "Detén la cinta.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_001"},
    {"text": "Para la cinta transportadora.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_002"},
    {"text": "Detén el proceso ahora.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_003"},
    {"text": "Para el lote.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_004"},
    {"text": "Detén la operación actual.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_005"},
    {"text": "Detén el robot, por favor.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_006"},
    {"text": "Detente ahora.", "intent": "STOP_PROCESS", "params": {}, "family": "pat2_stop_007"},

    {"text": "¡Emergencia! Para todo.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat2_emerg_001"},
    {"text": "Emergencia, para el robot ya.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat2_emerg_002"},
    {"text": "Activa la parada de emergencia.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat2_emerg_003"},
    {"text": "Parada de emergencia del robot.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat2_emerg_004"},
    {"text": "Emergencia, para todo inmediatamente.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat2_emerg_005"},
    {"text": "Para todo, es una emergencia.", "intent": "EMERGENCY_STOP", "params": {}, "family": "pat2_emerg_006"},

    {"text": "Pausa el proceso.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat2_pause_001"},
    {"text": "Pausa un momento.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat2_pause_002"},
    {"text": "Haz una pausa en el proceso.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat2_pause_003"},
    {"text": "Pausa la cinta mientras reviso.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat2_pause_004"},
    {"text": "Pausa el lote hasta nuevo aviso.", "intent": "PAUSE_PROCESS", "params": {}, "family": "pat2_pause_005"},

    {"text": "Reanuda el proceso.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat2_resume_001"},
    {"text": "Continúa con el proceso.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat2_resume_002"},
    {"text": "Sigue con el lote.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat2_resume_003"},
    {"text": "Reanuda la operación.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat2_resume_004"},
    {"text": "Continúa el trabajo.", "intent": "RESUME_PROCESS", "params": {}, "family": "pat2_resume_005"},

    {"text": "Reinicia el sistema.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat2_reset_001"},
    {"text": "Reinicia el flujo de trabajo.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat2_reset_002"},
    {"text": "Resetea la celda.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat2_reset_003"},
    {"text": "Vuelve al estado inicial.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat2_reset_004"},
    {"text": "Restablece el sistema.", "intent": "RESET_SYSTEM", "params": {}, "family": "pat2_reset_005"},

    {"text": "Suelta {obj_phrase}.", "intent": "RELEASE_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_release_001"},
    {"text": "Libera {obj_phrase}.", "intent": "RELEASE_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_release_002"},
    {"text": "Suelta lo que tienes en la pinza.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat2_release_003"},
    {"text": "Suelta el objeto ahora.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat2_release_004"},
    {"text": "Libera la pinza.", "intent": "RELEASE_OBJECT", "params": {}, "family": "pat2_release_005"},

    {"text": "Devuelve {obj_phrase} a su sitio.", "intent": "RETURN_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_return_001"},
    {"text": "Devuelve {obj_phrase} a la posición inicial.", "intent": "RETURN_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_return_002"},
    {"text": "Devuelve el objeto a su posición original.", "intent": "RETURN_OBJECT", "params": {}, "family": "pat2_return_003"},
    {"text": "Retorna {obj_phrase} a su posición.", "intent": "RETURN_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_return_004"},

    {"text": "Reintenta la acción.", "intent": "RETRY_ACTION", "params": {}, "family": "pat2_retry_001"},
    {"text": "Vuelve a intentarlo.", "intent": "RETRY_ACTION", "params": {}, "family": "pat2_retry_002"},
    {"text": "Inténtalo otra vez.", "intent": "RETRY_ACTION", "params": {}, "family": "pat2_retry_003"},
    {"text": "Reintenta la última acción.", "intent": "RETRY_ACTION", "params": {}, "family": "pat2_retry_004"},
    {"text": "Prueba otra vez.", "intent": "RETRY_ACTION", "params": {}, "family": "pat2_retry_005"},
    {"text": "Repite el intento.", "intent": "RETRY_ACTION", "params": {}, "family": "pat2_retry_006"},

    {"text": "Salta {obj_phrase}.", "intent": "SKIP_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_skip_001"},
    {"text": "Omite {obj_phrase}.", "intent": "SKIP_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_skip_002"},
    {"text": "Pasa de esta pieza.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat2_skip_003"},
    {"text": "Salta la pieza actual.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat2_skip_004"},
    {"text": "Omite el objeto y sigue.", "intent": "SKIP_OBJECT", "params": {}, "family": "pat2_skip_005"},
    {"text": "Ignora {obj_phrase} por ahora.", "intent": "SKIP_OBJECT", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_skip_006"},

    {"text": "Recalcula el plan.", "intent": "REPLAN_TASK", "params": {}, "family": "pat2_replan_001"},
    {"text": "Replanifica las tareas pendientes.", "intent": "REPLAN_TASK", "params": {}, "family": "pat2_replan_002"},
    {"text": "Vuelve a planificar el lote.", "intent": "REPLAN_TASK", "params": {}, "family": "pat2_replan_003"},
    {"text": "Actualiza el plan de trabajo.", "intent": "REPLAN_TASK", "params": {}, "family": "pat2_replan_004"},
    {"text": "Recalcula la secuencia restante.", "intent": "REPLAN_TASK", "params": {}, "family": "pat2_replan_005"},

    {"text": "Cancela la acción pendiente.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat2_cancel_001"},
    {"text": "Cancela eso.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat2_cancel_002"},
    {"text": "Anula la última orden.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat2_cancel_003"},
    {"text": "Cancela la operación en curso.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat2_cancel_004"},
    {"text": "Olvida la última instrucción.", "intent": "CANCEL_ACTION", "params": {}, "family": "pat2_cancel_005"},

    {"text": "Confirma la acción.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat2_confirm_001"},
    {"text": "Sí, confirma.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat2_confirm_002"},
    {"text": "Confirmado, adelante.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat2_confirm_003"},
    {"text": "Acepta la propuesta.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat2_confirm_004"},
    {"text": "Adelante, confirma el movimiento.", "intent": "CONFIRM_ACTION", "params": {}, "family": "pat2_confirm_005"},

    {"text": "Dime el estado del robot.", "intent": "REQUEST_STATUS", "params": {}, "family": "pat2_status_001"},
    {"text": "¿Cuál es el estado actual?", "intent": "REQUEST_STATUS", "params": {}, "family": "pat2_status_002"},
    {"text": "¿Cómo va el proceso?", "intent": "REQUEST_STATUS", "params": {}, "family": "pat2_status_003"},
    {"text": "Estado del sistema, por favor.", "intent": "REQUEST_STATUS", "params": {}, "family": "pat2_status_004"},

    {"text": "Genera el informe del lote.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat2_report_001"},
    {"text": "Dame el informe de calidad.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat2_report_002"},
    {"text": "Quiero el resumen del lote.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat2_report_003"},
    {"text": "Muéstrame el informe final.", "intent": "REQUEST_REPORT", "params": {}, "family": "pat2_report_004"},

    {"text": "Dime la información de {obj_phrase}.", "intent": "REQUEST_OBJECT_INFO", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_info_001"},
    {"text": "¿Qué sabes de {obj_phrase}?", "intent": "REQUEST_OBJECT_INFO", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_info_002"},
    {"text": "Dame los datos de {obj_phrase}.", "intent": "REQUEST_OBJECT_INFO", "params": {"object_type": "obj", "object_color": "color"}, "family": "pat2_info_003"},

    {"text": "Calibra el sistema.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat2_calib_001"},
    {"text": "Calibra la cámara.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat2_calib_002"},
    {"text": "Haz una calibración.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat2_calib_003"},
    {"text": "Recalibra el robot.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat2_calib_004"},
    {"text": "Calibra el espacio de trabajo.", "intent": "CALIBRATE_SYSTEM", "params": {}, "family": "pat2_calib_005"},

    {"text": "Cambia la velocidad a lenta.", "intent": "SET_SPEED", "params": {}, "family": "pat2_speed_001"},
    {"text": "Ve más despacio.", "intent": "SET_SPEED", "params": {}, "family": "pat2_speed_002"},
    {"text": "Velocidad media, por favor.", "intent": "SET_SPEED", "params": {}, "family": "pat2_speed_003"},
    {"text": "Reduce la velocidad.", "intent": "SET_SPEED", "params": {}, "family": "pat2_speed_004"},
    {"text": "Acelera el proceso.", "intent": "SET_SPEED", "params": {}, "family": "pat2_speed_005"},

    {"text": "A partir de ahora rechaza también las {obj}s {color_obj_pl}.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "obj", "object_color": "color", "object_quality": "defectuoso"}, "family": "pat2_cqr_001"},
    {"text": "Desde ahora rechaza las piezas {color_obj_pl} también.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "pieza", "object_color": "color", "object_quality": "defectuoso"}, "family": "pat2_cqr_002"},
    {"text": "Acepta también las {obj}s {color_obj_pl} a partir de ahora.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "obj", "object_color": "color", "object_quality": "bueno"}, "family": "pat2_cqr_003"},
    {"text": "Desde ahora considera buenas las {obj}s {color_obj_pl}.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "obj", "object_color": "color", "object_quality": "bueno"}, "family": "pat2_cqr_004"},
    {"text": "Rechaza además las piezas {color_obj_pl}.", "intent": "CHANGE_QUALITY_RULE", "params": {"object_type": "pieza", "object_color": "color", "object_quality": "defectuoso"}, "family": "pat2_cqr_005"},

    {"text": "Las piezas defectuosas van al rechazo desde ahora.", "intent": "CHANGE_DESTINATION", "params": {"destination_role": "REJECT_BIN", "object_quality": "defectuoso"}, "family": "pat2_cd_001"},
    {"text": "Las buenas van a aceptación.", "intent": "CHANGE_DESTINATION", "params": {"destination_role": "GOOD_BIN", "object_quality": "bueno"}, "family": "pat2_cd_002"},
    {"text": "Manda las defectuosas al rechazo.", "intent": "CHANGE_DESTINATION", "params": {"destination_role": "REJECT_BIN", "object_quality": "defectuoso"}, "family": "pat2_cd_003"},
    {"text": "Desde ahora las piezas {color_obj_pl} van a revisión.", "intent": "CHANGE_DESTINATION", "params": {"object_type": "pieza", "object_color": "color", "destination_role": "REVIEW_BIN"}, "family": "pat2_cd_004"},

    {"text": "Separa las piezas buenas en aceptación y las defectuosas en rechazo.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat2_sort_001"},
    {"text": "Envía las buenas a aceptación y las defectuosas a rechazo.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat2_sort_002"},
    {"text": "Clasifica el lote: buenas a aceptación, defectuosas a rechazo.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat2_sort_003"},
    {"text": "Reparte las piezas entre aceptación y rechazo.", "intent": "SORT_OBJECTS", "params": {"quantity": "ALL"}, "family": "pat2_sort_004"},

    {"text": "Envía {obj_phrase} a revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_role": "REVIEW_BIN"}, "family": "pat2_quar_001"},
    {"text": "Aparta {obj_phrase} para revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_role": "REVIEW_BIN"}, "family": "pat2_quar_002"},
    {"text": "Lleva {obj_phrase} a revisión.", "intent": "QUARANTINE_OBJECT", "params": {"object_type": "obj", "object_color": "color", "destination_role": "REVIEW_BIN"}, "family": "pat2_quar_003"},

    {"text": "Inspecciona todos los componentes del lote.", "intent": "INSPECT_BATCH", "params": {"object_type": "componente", "quantity": "ALL"}, "family": "pat2_batch_001"},
    {"text": "Inspecciona todos los productos.", "intent": "INSPECT_BATCH", "params": {"object_type": "producto", "quantity": "ALL"}, "family": "pat2_batch_002"},
    {"text": "Revisa todas las piezas de la bandeja.", "intent": "INSPECT_BATCH", "params": {"object_type": "pieza", "quantity": "ALL"}, "family": "pat2_batch_003"},
    {"text": "Inspecciona el lote entero.", "intent": "INSPECT_BATCH", "params": {"object_type": "pieza", "quantity": "ALL"}, "family": "pat2_batch_004"},

    # Third catalogue: synonyms for accepted parts, retry phrasings and out-of-domain requests.
    {"text": "Marca {obj_phrase} como válida.", "intent": "MARK_GOOD", "params": {"object_type": "obj", "object_color": "color", "object_quality": "bueno"}, "family": "pat3_markg_001"},
    {"text": "Esta pieza es correcta.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_002"},
    {"text": "Marca esta pieza como aprobada.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_003"},
    {"text": "Esta pieza está conforme.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_004"},
    {"text": "Esta pieza no tiene defectos.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_005"},
    {"text": "Esta pieza está en buen estado.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_006"},
    {"text": "Marca la pieza como OK.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_007"},
    {"text": "Aprueba {obj_phrase}.", "intent": "MARK_GOOD", "params": {"object_type": "obj", "object_color": "color", "object_quality": "bueno"}, "family": "pat3_markg_008"},
    {"text": "Da por buena esta pieza.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_009"},
    {"text": "Marca este componente como correcto.", "intent": "MARK_GOOD", "params": {"object_quality": "bueno"}, "family": "pat3_markg_010"},

    {"text": "Vuelve a intentarlo, por favor.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_001"},
    {"text": "Intenta de nuevo.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_002"},
    {"text": "Hazlo otra vez.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_003"},
    {"text": "Inténtalo de nuevo, por favor.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_004"},
    {"text": "Vuelve a probar.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_005"},
    {"text": "Repite la operación.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_006"},
    {"text": "Reintenta la operación fallida.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_007"},
    {"text": "Otra vez, por favor.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_008"},
    {"text": "Ese intento ha fallado, vuelve a intentarlo.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_009"},
    {"text": "Vuelve a intentar la acción.", "intent": "RETRY_ACTION", "params": {}, "family": "pat3_retry_010"},

    {"text": "Pon una canción.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_001"},
    {"text": "Reproduce mi lista de música.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_002"},
    {"text": "Pon la televisión.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_003"},
    {"text": "Cuéntame una historia.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_004"},
    {"text": "¿Cuál es la capital de Francia?", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_005"},
    {"text": "Escribe un poema sobre el mar.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_006"},
    {"text": "Hazme una receta de tortilla.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_007"},
    {"text": "Llama a mi madre.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_008"},
    {"text": "Enciende la luz del salón.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_009"},
    {"text": "Apaga el ordenador de casa.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_010"},
    {"text": "Busca vuelos a Madrid.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_011"},
    {"text": "¿Va a llover esta tarde?", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_012"},
    {"text": "Recuérdame comprar leche.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_013"},
    {"text": "Pon un temporizador de diez minutos.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_014"},
    {"text": "Traduce esta frase al inglés.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_015"},
    {"text": "¿Quién ganó el partido de ayer?", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_016"},
    {"text": "Dime un dato curioso.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_017"},
    {"text": "Abre el navegador.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_018"},
    {"text": "Sube el volumen.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_019"},
    {"text": "¿Qué opinas de la película?", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_020"},
    {"text": "Hola, ¿cómo estás?", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_021"},
    {"text": "Canta algo.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_022"},
    {"text": "Sácame una foto.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_023"},
    {"text": "Pide una pizza.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_024"},
    {"text": "Ponme una alarma a las siete.", "intent": "UNKNOWN", "params": {}, "family": "pat3_oos_025"},
]

# Register one generator per pattern spec.
PATTERN_GENS: List[Callable[[random.Random], Tuple[str, List[Dict[str, Any]], str]]] = []
for _i, _spec in enumerate(PATTERN_SPECS):
    def _make(spec, i):
        def _gen(rng):
            return _run_pattern(rng, spec, i)
        _gen.__name__ = f"gen_pattern_{i:03d}"
        return _gen
    PATTERN_GENS.append(_make(_spec, _i))

# registry
TEMPLATE_GENERATORS: List[Callable[[random.Random], Tuple[str, List[Dict[str, Any]], str]]] = [
    gen_pick_simple, gen_place_simple, gen_move, gen_inspect_single, gen_inspect_batch,
    gen_classify, gen_sort, gen_change_quality_rule, gen_change_destination, gen_mark_good,
    gen_mark_defective, gen_quarantine, gen_emergency, gen_stop, gen_pause, gen_resume,
    gen_reset, gen_start, gen_status, gen_report, gen_object_info, gen_calibrate, gen_speed,
    gen_retry, gen_skip, gen_replan, gen_cancel, gen_confirm, gen_release, gen_return,
    gen_unknown, gen_negation_pick, gen_negation_place, gen_except, gen_compound_two,
    gen_compound_colors, gen_correction, gen_move_qty, gen_place_all, gen_incomplete,
    gen_speech_noise,
] + PATTERN_GENS


def generate_one(rng: random.Random):
    gen = rng.choice(TEMPLATE_GENERATORS)
    text, actions, family = gen(rng)
    # clean text
    text = " ".join(text.split())
    return text, actions, family
