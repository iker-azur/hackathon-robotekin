"""Answer structured Phase 1 commands from the visual world state.

Rule-based and Qt-free. The resolver reads a WorldSnapshot, may change the
quality rules and the on-screen selection, and never claims that a physical
action happened. Ambiguous references return CLARIFY instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..commands.schema import Action, StructuredCommand
from ..commands.taxonomy import INTENT_DESCRIPTIONS_ES
from ..commands.vision_intents import fold_text
from .inspection import QUALITY_LABEL_ES
from .quality import DEFECTIVE, GOOD, UNCERTAIN, QualityRule
from .world_state import ObjectRecord, WorldSnapshot, WorldState

OK = "OK"
INFO = "INFO"
CLARIFY = "CLARIFY"
NOT_FOUND = "NOT_FOUND"
UNSUPPORTED = "UNSUPPORTED"
_PRIORITY = {CLARIFY: 0, NOT_FOUND: 1, UNSUPPORTED: 2, INFO: 3, OK: 4}

NO_PHYSICAL_ACTION = "No se ha ejecutado ninguna acción física."

QUALITY_FROM_PARAM = {"bueno": GOOD, "defectuoso": DEFECTIVE, "desconocido": UNCERTAIN}
# Answers that name objects; their matches are highlighted on the camera view and in the table.
HIGHLIGHT_INTENTS = {"COUNT_OBJECTS", "INSPECT_OBJECT", "INSPECT_BATCH", "CLASSIFY_OBJECT", "SORT_OBJECTS"}
# Parameter -> object types the filter accepts. Cylinders and spheres are both "circular" from above.
TYPE_FILTER: Dict[str, Tuple[str, ...]] = {
    "cubo": ("cubo",), "prisma": ("prisma",), "cilindro": ("circular",), "esfera": ("circular",),
}
RULE_TYPE = {"cubo": "cubo", "prisma": "prisma", "cilindro": "circular", "esfera": "circular"}
# noun singular, noun plural, gender
NOUNS: Dict[str, Tuple[str, str, str]] = {
    "cubo": ("cubo", "cubos", "m"), "prisma": ("prisma", "prismas", "m"),
    "cilindro": ("cilindro", "cilindros", "m"), "esfera": ("esfera", "esferas", "f"),
}
# canonical -> (masc sing, fem sing, masc plural, fem plural)
COLOR_FORMS: Dict[str, Tuple[str, str, str, str]] = {
    "rojo": ("rojo", "roja", "rojos", "rojas"),
    "naranja": ("naranja", "naranja", "naranjas", "naranjas"),
    "amarillo": ("amarillo", "amarilla", "amarillos", "amarillas"),
    "verde": ("verde", "verde", "verdes", "verdes"),
    "azul": ("azul", "azul", "azules", "azules"),
    "morado": ("morado", "morada", "morados", "moradas"),
    "blanco": ("blanco", "blanca", "blancos", "blancas"),
    "gris": ("gris", "gris", "grises", "grises"),
    "negro": ("negro", "negra", "negros", "negras"),
}
QUALITY_FORMS: Dict[str, Tuple[str, str, str, str]] = {
    GOOD: ("bueno", "buena", "buenos", "buenas"),
    DEFECTIVE: ("defectuoso", "defectuosa", "defectuosos", "defectuosas"),
    UNCERTAIN: ("incierto", "incierta", "inciertos", "inciertas"),
}
TYPE_ES = {
    "cubo": "cubo", "prisma": "prisma",
    "circular": "forma circular (cilindro o esfera, no distinguible desde arriba)",
    "desconocido": "forma no reconocida",
}
_SINGULAR_REFERENCE = re.compile(
    r"\b(la|el|una|un)\s+(pieza|cubo|cilindro|esfera|prisma|componente|producto|objeto)\b"
)
_NOUN_WORDS = ("pieza", "piezas", "objeto", "objetos", "componente", "componentes", "producto", "productos")


@dataclass
class VisionResult:
    status: str
    messages: List[str]
    matched_ids: List[str] = field(default_factory=list)
    selected_ids: Optional[List[str]] = None
    rule: Optional[QualityRule] = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.messages)


def _ids(objects: List[ObjectRecord]) -> str:
    return ", ".join(o.object_id for o in objects)


def _form(table: Dict[str, Tuple[str, str, str, str]], key: str, feminine: bool, plural: bool) -> str:
    masc_sg, fem_sg, masc_pl, fem_pl = table[key]
    if plural:
        return fem_pl if feminine else masc_pl
    return fem_sg if feminine else masc_sg


def _phrase(params: Dict, plural: bool, quality: Optional[str] = None) -> str:
    """Spanish noun phrase such as "piezas azules defectuosas" built from parameters."""
    otype = params.get("object_type")
    noun_sg, noun_pl, gender = NOUNS.get(otype, ("pieza", "piezas", "f"))
    feminine = gender == "f"
    words = [noun_pl if plural else noun_sg]
    color = params.get("object_color")
    if color in COLOR_FORMS:
        words.append(_form(COLOR_FORMS, color, feminine, plural))
    if quality is None:
        quality = QUALITY_FROM_PARAM.get(params.get("object_quality"))
    if quality in QUALITY_FORMS:
        words.append(_form(QUALITY_FORMS, quality, feminine, plural))
    return " ".join(words)


def _match(obj: ObjectRecord, params: Dict, ignore_quality: bool = False) -> bool:
    color = params.get("object_color")
    if color and obj.color != color:
        return False
    otype = params.get("object_type")
    if otype in TYPE_FILTER and obj.object_type not in TYPE_FILTER[otype]:
        return False
    if not ignore_quality:
        quality = QUALITY_FROM_PARAM.get(params.get("object_quality"))
        if quality and obj.quality != quality:
            return False
    return True


def _is_singular(action: Action, text: str) -> bool:
    quantity = action.parameters.get("quantity")
    if quantity == 1:
        return True
    if isinstance(quantity, int) or quantity not in (None, "UNSPECIFIED"):
        return False
    return bool(_SINGULAR_REFERENCE.search(fold_text(text)))


def _describe(obj: ObjectRecord) -> str:
    x, y = obj.position_px
    color = obj.color or "color no concluyente"
    text = (
        f"{obj.object_id}: {TYPE_ES.get(obj.object_type, obj.object_type)}, {color}, "
        f"{QUALITY_LABEL_ES.get(obj.quality, obj.quality).lower()}, en px ({x:.0f}, {y:.0f})"
    )
    if obj.detection_status == "PARTIAL":
        text += ", parcialmente visible"
    if obj.world_position:
        xy = obj.world_position["xy"]
        text += f", mundo {xy[0]}, {xy[1]} {obj.world_position['units']}"
    if obj.quality_reasons:
        text += ". Motivo: " + "; ".join(obj.quality_reasons)
    return text


def _count(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    params = action.parameters
    matches = [o for o in snap.visible() if _match(o, params)]
    phrase = _phrase(params, plural=True)
    if not matches:
        return VisionResult(NOT_FOUND, [f"No hay {phrase} visibles."])
    return VisionResult(INFO, [f"Hay {len(matches)} {phrase} visibles: {_ids(matches)}."],
                        matched_ids=[o.object_id for o in matches])


def _locate(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    params = action.parameters
    singular = _is_singular(action, text)
    matches = [o for o in snap.visible() if _match(o, params)]
    if not matches:
        return VisionResult(NOT_FOUND, [f"No veo {_phrase(params, plural=not singular)} en el área de trabajo."])
    if singular and len(matches) > 1:
        return VisionResult(CLARIFY, [
            f"Hay {len(matches)} {_phrase(params, plural=True)}: {_ids(matches)}. "
            "¿Cuál quieres localizar? Indica su ID, por ejemplo OBJ_001."
        ], matched_ids=[o.object_id for o in matches])
    lines = [_describe(o) for o in matches]
    ids = [o.object_id for o in matches]
    return VisionResult(OK, lines, matched_ids=ids, selected_ids=ids)


def _show(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    params = action.parameters
    matches = [o for o in snap.visible() if _match(o, params)]
    if not matches:
        return VisionResult(NOT_FOUND, [f"No veo {_phrase(params, plural=True)} para resaltar."])
    ids = [o.object_id for o in matches]
    return VisionResult(OK, [f"Resalto {len(ids)} {_phrase(params, plural=len(ids) != 1)}: {_ids(matches)}."],
                        matched_ids=ids, selected_ids=ids)


def _select(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    params = action.parameters
    singular = _is_singular(action, text)
    matches = [o for o in snap.visible() if _match(o, params)]
    if not matches:
        return VisionResult(NOT_FOUND, [f"No veo {_phrase(params, plural=True)} para seleccionar."])
    quantity = params.get("quantity")
    note = ""
    if isinstance(quantity, int) and quantity > 0:
        chosen = matches[:quantity]
        if len(matches) < quantity:
            note = f" Solo hay {len(matches)}, así que selecciono todas."
    elif singular and len(matches) > 1:
        return VisionResult(CLARIFY, [
            f"Hay {len(matches)} {_phrase(params, plural=True)}: {_ids(matches)}. "
            "¿Cuál seleccionas? Indica su ID o una cantidad."
        ], matched_ids=[o.object_id for o in matches])
    else:
        chosen = matches
    ids = [o.object_id for o in chosen]
    return VisionResult(OK, [f"Selecciono {len(ids)} {_phrase(params, plural=len(ids) != 1)}: {_ids(chosen)}.{note}"],
                        matched_ids=ids, selected_ids=ids)


def _inspect(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    params = action.parameters
    singular = _is_singular(action, text)
    matches = [o for o in snap.visible() if _match(o, params)]
    if not matches:
        return VisionResult(NOT_FOUND, [f"No veo {_phrase(params, plural=not singular)} para inspeccionar."])
    if singular and len(matches) > 1:
        return VisionResult(CLARIFY, [
            f"Hay {len(matches)} {_phrase(params, plural=True)}: {_ids(matches)}. "
            "¿Cuál inspecciono? Indica su ID."
        ], matched_ids=[o.object_id for o in matches])
    lines = [f"Inspección de {len(matches)} {_phrase(params, plural=len(matches) != 1)}:"]
    lines += [_describe(o) for o in matches]
    return VisionResult(OK, ["\n".join(lines)], matched_ids=[o.object_id for o in matches])


def _classify(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    # Grouping shows every verdict, so "buenas y malas" becomes one answer, not two.
    matches = [o for o in snap.visible() if _match(o, action.parameters, ignore_quality=True)]
    if not matches:
        return VisionResult(NOT_FOUND, [f"No veo {_phrase(action.parameters, plural=True)} para clasificar."])
    lines = []
    for verdict in (GOOD, DEFECTIVE, UNCERTAIN):
        group = [o for o in matches if o.quality == verdict]
        label = QUALITY_FORMS[verdict][2].capitalize()
        lines.append(f"{label} ({len(group)}): {_ids(group) or 'ninguna'}")
    return VisionResult(OK, ["\n".join(lines)], matched_ids=[o.object_id for o in matches])


def _change_rule(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    params = action.parameters
    verdict = QUALITY_FROM_PARAM.get(params.get("object_quality"))
    if verdict not in (GOOD, DEFECTIVE):
        return VisionResult(UNSUPPORTED, [
            "No entiendo si las piezas deben quedar buenas o defectuosas. "
            "Ejemplo: 'Ahora las piezas azules son defectuosas'."
        ])
    color = params.get("object_color")
    object_type = RULE_TYPE.get(params.get("object_type"))
    if not color and not object_type:
        return VisionResult(CLARIFY, ["¿A qué piezas se aplica el criterio? Indica un color o un tipo de pieza."])
    name = _phrase({"object_color": color, "object_type": params.get("object_type")}, plural=True)
    rule = QualityRule(name=name, verdict=verdict, color=color, object_type=object_type)
    world.add_rule(rule)
    affected = [o for o in world.snapshot().visible() if rule.matches(o.color, o.object_type)]
    target = "defectuosas" if verdict == DEFECTIVE else "buenas"
    return VisionResult(
        OK,
        [f"Criterio aplicado: {name} se consideran {target}. Afecta ahora a {len(affected)} pieza(s) visible(s): "
         f"{_ids(affected) or 'ninguna'}. Solo cambia la clasificación visual. {NO_PHYSICAL_ACTION}"],
        matched_ids=[o.object_id for o in affected],
        rule=rule,
    )


def _status(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    counts = snap.counts()
    if counts["visible"] == 0:
        return VisionResult(INFO, ["No veo piezas visibles en el área de trabajo."])
    lines = [
        f"Veo {counts['visible']} pieza(s) visibles: {counts['GOOD']} buenas, "
        f"{counts['DEFECTIVE']} defectuosas, {counts['UNCERTAIN']} inciertas."
    ]
    if snap.rules:
        lines.append("Criterios activos: " + "; ".join(
            f"{r.name} -> {QUALITY_LABEL_ES[r.verdict].lower()}" for r in snap.rules))
    return VisionResult(INFO, ["\n".join(lines)])


def _manual_mark(action: Action, snap: WorldSnapshot, world: WorldState, text: str) -> VisionResult:
    return VisionResult(UNSUPPORTED, [
        "Marcar piezas concretas a mano no está implementado en esta fase. "
        "Usa un criterio por color o tipo, por ejemplo: 'Ahora las piezas azules son defectuosas'."
    ])


_HANDLERS: Dict[str, Callable[[Action, WorldSnapshot, WorldState, str], VisionResult]] = {
    "COUNT_OBJECTS": _count,
    "LOCATE_OBJECT": _locate,
    "SHOW_OBJECTS": _show,
    "SELECT_OBJECTS": _select,
    "INSPECT_OBJECT": _inspect,
    "INSPECT_BATCH": _inspect,
    "CLASSIFY_OBJECT": _classify,
    "SORT_OBJECTS": _classify,
    "CHANGE_QUALITY_RULE": _change_rule,
    "REQUEST_STATUS": _status,
    "REQUEST_REPORT": _status,
    "MARK_GOOD": _manual_mark,
    "MARK_DEFECTIVE": _manual_mark,
}


def _unsupported(action: Action) -> VisionResult:
    if action.intent == "UNKNOWN":
        return VisionResult(UNSUPPORTED, [
            "No he entendido la instrucción. Ejemplos: '¿Cuántas piezas rojas hay?', "
            "'Inspecciona las piezas azules', 'Ahora las piezas amarillas son defectuosas'."
        ])
    description = INTENT_DESCRIPTIONS_ES.get(action.intent, action.intent)
    return VisionResult(UNSUPPORTED, [
        f"'{description}' se reconoce, pero esta fase no controla el robot. {NO_PHYSICAL_ACTION}"
    ])


def resolve_command(cmd: StructuredCommand, world: WorldState) -> VisionResult:
    """Resolve every action of a command against the current world state."""
    snap = world.snapshot()
    results: List[VisionResult] = []
    seen = set()
    for action in cmd.actions:
        params = dict(action.parameters)
        if action.intent in ("CLASSIFY_OBJECT", "SORT_OBJECTS"):
            params.pop("object_quality", None)  # grouping shows every verdict anyway
            if params.get("object_type") in ("pieza", "objeto"):
                params.pop("object_type", None)
        key = (action.intent, tuple(sorted((k, str(v)) for k, v in params.items() if v not in (None,"UNSPECIFIED"))))
        if key in seen:  # "buenas y malas" yields two CLASSIFY actions that must answer once
            continue
        seen.add(key)
        handler = _HANDLERS.get(action.intent)
        results.append(handler(action, snap, world, cmd.transcription) if handler else _unsupported(action))
    if not results:
        return VisionResult(UNSUPPORTED, [_unsupported(Action(intent="UNKNOWN")).text])

    status = min((r.status for r in results), key=_PRIORITY.get)
    matched: List[str] = []
    for r in results:
        matched += [i for i in r.matched_ids if i not in matched]
    selected = next((r.selected_ids for r in reversed(results) if r.selected_ids is not None), None)
    if selected is None and matched and any(a.intent in HIGHLIGHT_INTENTS for a in cmd.actions):
        selected = matched  # query answers highlight the objects they talk about
    if selected is not None:
        world.set_selection(selected)
    return VisionResult(
        status=status,
        messages=[m for r in results for m in r.messages],
        matched_ids=matched,
        selected_ids=selected,
        rule=next((r.rule for r in results if r.rule is not None), None),
    )
