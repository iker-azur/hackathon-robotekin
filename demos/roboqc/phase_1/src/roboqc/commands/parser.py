"""Command parsing: segmentation, intent classification (rule-based fallback),
entity extraction, and structured command construction.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Callable, List, Optional


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

from .entities import extract_entities, entities_to_parameters, ExtractedEntities
from .schema import Action, StructuredCommand
from .taxonomy import ALL_INTENTS, Intent
from .validator import validate_command
from .vision_intents import classify_vision_intent


# ---------------------------------------------------------------------------
# Action verbs and rule-based intent classification
# ---------------------------------------------------------------------------

ACTION_VERBS = [
    "parada de emergencia", "parada", "emergencia",
    "detén", "deten", "detener", "para el", "parar", "para la", "para los",
    "pausa", "pausar", "espera",
    "continúa", "continua", "continuar", "reanuda", "reanudar", "sigue",
    "reinicia", "reiniciar", "reset",
    "empieza", "empezar", "inicia", "iniciar", "comienza", "comenzar", "arranca", "arrancar",
    "inspecciona", "inspeccionar", "revisa", "revisar",
    "clasifica", "clasificar",
    "separa", "separar", "ordena", "ordenar",
    "marca", "marcar",
    "coge", "coger", "cojas", "agarra", "agarrar", "agarres", "toma", "tomar", "tomes", "recoge", "recoger", "recojas",
    "deja", "dejar", "dejes", "pon", "poner", "coloca", "colocar", "sitúa", "situa", "situá",
    "mueve", "mover", "transfiere", "transferir", "transporta", "transportar",
    "suelta", "soltar", "libera", "liberar",
    "devuelve", "devolver", "retorna", "retornar",
    "intenta", "intentar", "reintenta", "reintentar",
    "salta", "saltar", "omite", "omitir",
    "cancela", "cancelar",
    "confirma", "confirmar",
    "calibra", "calibrar",
]


def _norm(s: str) -> str:
    return " ".join(_strip_accents(s).lower().split())


def classify_intent_rule(text: str) -> str:
    """Deterministic high-priority + keyword intent classifier (Spanish)."""
    t = _norm(text)
    # Emergency first
    if "parada de emergencia" in t or "emergencia" in t:
        return Intent.EMERGENCY_STOP.value
    # Safety stops
    if any(k in t for k in ["detén", "deten", "detener", "parada", "para el robot", "para la cinta"]):
        # avoid matching "para la caja" as stop
        if "para el" in t or "detén" in t or "deten" in t or "parada" in t or "para el robot" in t or "para la máquina" in t:
            pass
    # Pause
    if any(k in t for k in ["pausa", "pausar", "espera un momento", "detente un momento"]):
        if "parada" not in t:
            return Intent.PAUSE_PROCESS.value
    # Resume
    if any(k in t for k in ["continúa", "continua", "continuar", "reanuda", "reanudar", "sigue trabajando", "sigue"]):
        return Intent.RESUME_PROCESS.value
    # Reset
    if any(k in t for k in ["reinicia", "reiniciar", "reset", "reiniciar sistema"]):
        return Intent.RESET_SYSTEM.value
    # Start
    if any(k in t for k in ["empieza", "empezar", "inicia", "iniciar", "comienza", "comenzar", "arranca", "arrancar"]):
        return Intent.START_PROCESS.value
    # Stop (process) - only if stop verbs and not pause
    if any(k in t for k in ["detén", "deten", "detener", "para el proceso", "para el robot", "para la máquina", "parar"]):
        if "parada de emergencia" not in t:
            return Intent.STOP_PROCESS.value
    # emergency already handled above

    # Unknown detection: common non-industrial
    unknown_markers = ["música", "musica", "tiempo", "chiste", "película", "pelicula", "broma", "algo con"]
    if any(m in t for m in unknown_markers):
        # but ensure not a real command mentioning music? default unknown
        if not any(v in t for v in ["pieza", "cubo", "caja", "inspecciona", "coge", "deja", "mueve"]):
            return Intent.UNKNOWN.value
    if "haz algo" in t:
        return Intent.UNKNOWN.value

    # Change quality rule: quality state assigned to a group
    quality_words = ["buenas", "buenos", "defectuosas", "defectuosos", "malas", "malos", "dañadas", "danadas"]
    if ("son" in t or "ahora" in t or "considera" in t or "serán" in t or "seran" in t) and any(q in t for q in quality_words):
        if "deben ir" not in t and "debe ir" not in t:
            return Intent.CHANGE_QUALITY_RULE.value
    # Change destination
    if "deben ir" in t or "debe ir" in t or "deberán ir" in t or "deberan ir" in t:
        return Intent.CHANGE_DESTINATION.value

    # Mark good/defective
    if "marca" in t and ("buena" in t or "bueno" in t or "aceptable" in t):
        return Intent.MARK_GOOD.value
    if ("está defectuosa" in t or "esta defectuosa" in t or "está dañada" in t or "marcar" in t and "defectuosa" in t):
        if "marca" in t:
            return Intent.MARK_DEFECTIVE.value
    if "defectuosa" in t and t.startswith("esta ") :
        return Intent.MARK_DEFECTIVE.value

    # Quarantine
    if any(k in t for k in ["revisión", "revision", "rechazo"]) and any(v in t for v in ["deja", "pon", "coloca", "mueve"]):
        return Intent.QUARANTINE_OBJECT.value

    # Inspection
    if any(k in t for k in ["inspecciona", "inspeccionar", "revisa", "revisar"]):
        if "todas" in t or "todos" in t or "lotes" in t or "lote" in t or re.search(r"\d+", t):
            # batch if plural/all/number>1
            if "todas" in t or "todos" in t or "todas las" in t:
                return Intent.INSPECT_BATCH.value
            # if "todas las piezas" handled; if number present maybe batch
            if re.search(r"\b[2-9]\d*\b", t):
                return Intent.INSPECT_BATCH.value
        return Intent.INSPECT_OBJECT.value

    # Classify
    if any(k in t for k in ["clasifica", "clasificar"]):
        return Intent.CLASSIFY_OBJECT.value

    # Sort
    if any(k in t for k in ["separa", "separar", "ordena", "ordenar"]) or ("buenas" in t and "defectuosas" in t):
        return Intent.SORT_OBJECTS.value

    # Return
    if any(k in t for k in ["devuelve", "devolver", "retorna", "retornar"]):
        return Intent.RETURN_OBJECT.value
    # Release
    if any(k in t for k in ["suelta", "soltar", "libera", "liberar"]):
        return Intent.RELEASE_OBJECT.value
    # Move
    if any(k in t for k in ["mueve", "muevas", "mover", "transfiere", "transferir", "transporta", "transportar"]):
        return Intent.MOVE_OBJECT.value
    # Place
    if any(k in t for k in ["deja", "dejar", "dejes", "pon", "poner", "coloca", "colocar", "sitúa", "situa", "situá"]):
        # if also cogemos? place takes priority for deja/pon/coloca
        return Intent.PLACE_OBJECT.value
    # Pick
    if any(k in t for k in ["coge", "coger", "cojas", "agarra", "agarrar", "agarres", "toma", "tomar", "tomes", "recoge", "recoger", "recojas"]):
        return Intent.PICK_OBJECT.value

    # Info / status
    if "cuánt" in t or "cuant" in t or "quedan" in t or "estado" in t or "status" in t:
        return Intent.REQUEST_STATUS.value
    if "informe" in t or "reporte" in t or "report" in t:
        return Intent.REQUEST_REPORT.value
    if "de qué color" in t or "de que color" in t or "color es" in t or "qué es" in t or "que es" in t:
        return Intent.REQUEST_OBJECT_INFO.value
    if "calibra" in t or "calibrar" in t:
        return Intent.CALIBRATE_SYSTEM.value
    if "velocidad" in t or "rápido" in t or "rapido" in t or "lento" in t or "reduce" in t or "aumenta" in t:
        return Intent.SET_SPEED.value
    # Retry / skip / replan / cancel / confirm
    if any(k in t for k in ["intenta", "intentar", "reintenta", "reintentar"]):
        return Intent.RETRY_ACTION.value
    if any(k in t for k in ["salta", "saltar", "omite", "omitir"]):
        return Intent.SKIP_OBJECT.value
    if any(k in t for k in ["busca otra", "recalcula", "replanifica", "replanificar", "otra forma"]):
        return Intent.REPLAN_TASK.value
    if any(k in t for k in ["cancela", "cancelar"]):
        return Intent.CANCEL_ACTION.value
    # Word match only: a bare substring "si" fired on words such as "visible" or "posible".
    if any(k in t for k in ["confirma", "confirmar"]) or re.search(r"\bsi\b", t):
        return Intent.CONFIRM_ACTION.value

    # Incomplete / generic
    if any(k in t for k in ["las piezas", "los cubos", "piezas"]) and len(t.split()) <= 4:
        return Intent.UNKNOWN.value
    # Default unknown for unsupported
    return Intent.UNKNOWN.value


def _has_verb(text: str) -> bool:
    t = _norm(text)
    return any(v in t for v in ACTION_VERBS)


def segment_command(text: str) -> List[str]:
    """Split a compound instruction into ordered segments."""
    t = text.strip()
    if not t:
        return []
    # normalize separators
    # split on commas/semicolons first
    parts = re.split(r"[,;]", t)
    segments: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # split on conjunctions that likely separate clauses
        # Use regex to split on ' y ' / ' e ' / ' luego ' / ' después ' / ' entonces '
        sub = re.split(r"\s+(?:y|e|luego|después|despues|entonces)\s+", p, flags=re.IGNORECASE)
        for s in sub:
            s = s.strip()
            if s:
                segments.append(s)
    # Merge segments that are just trailing destination phrases? keep.
    # If a segment has no verb, it might be a correction or trailing clause; keep as-is.
    # Collapse if only one
    if not segments:
        segments = [t]
    # Remove exact duplicates adjacent
    out = []
    for s in segments:
        if not out or s != out[-1]:
            out.append(s)
    return out


def _is_negated(segment: str) -> bool:
    s = _norm(segment)
    return s.startswith("no ") or " no " in s or s.startswith("nunca ")


def _inherit_intent(prev: Optional[str], seg: str) -> str:
    if _has_verb(seg):
        return classify_intent_rule(seg)
    if prev is not None:
        return prev
    return classify_intent_rule(seg)


def parse_text(
    text: str,
    source: str = "text",
    classifier: Optional[Callable[[str], str]] = None,
    confidence_fn: Optional[Callable[[str], float]] = None,
) -> StructuredCommand:
    """Parse free-form Spanish instruction into a StructuredCommand."""
    start = time.perf_counter()
    raw = text.strip()
    cmd = StructuredCommand(source=source, transcription=raw, normalized_text=raw)

    if not raw:
        cmd.status = "INVALID"
        cmd.validation_errors.append("Texto vacio.")
        cmd.processing_time_ms = (time.perf_counter() - start) * 1000
        return cmd

    # Safety deterministic pre-check
    safety_intent = None
    low = raw.lower()
    if "parada de emergencia" in low or "emergencia" in low:
        safety_intent = Intent.EMERGENCY_STOP.value

    segments = segment_command(raw)

    # Correction detection
    unresolved: List[str] = []
    if re.search(r"\bmejor\b|\bno,\b|\bcorrijo\b", low):
        unresolved.append("correccion_contextual")

    prev_intent: Optional[str] = None
    actions: List[Action] = []
    for seg in segments:
        vision_intent = classify_vision_intent(seg)
        if vision_intent is not None:
            intent = vision_intent
        elif classifier is not None:
            try:
                intent = classifier(seg)
            except Exception:
                intent = classify_intent_rule(seg)
        else:
            intent = classify_intent_rule(seg)
        # safety override
        if safety_intent and intent in (Intent.STOP_PROCESS.value, Intent.PAUSE_PROCESS.value):
            intent = safety_intent
        if safety_intent and "emergencia" in seg.lower():
            intent = safety_intent

        # Inherit intent for verbless trailing clauses (e.g. "las azules en la amarilla").
        # Visual queries never inherit: "¿Dónde está la roja?" is not a continuation.
        if vision_intent is None and not _has_verb(seg) and prev_intent is not None:
            intent = prev_intent

        prev_intent = intent

        # Emergency safety layer: if any segment contains emergency, force
        if "emergencia" in seg.lower():
            intent = Intent.EMERGENCY_STOP.value

        ents = extract_entities(seg)
        # For correction segments like "No, mejor en la caja azul", object may be missing -> unresolved
        params = entities_to_parameters(ents)
        # Ensure a stable parameter schema (nulls preserved as in the spec example)
        for key in ("object_type", "object_color", "quantity", "object_quality",
                    "destination_type", "destination_color", "destination_role"):
            params.setdefault(key, None)
        if params.get("quantity") is None:
            params["quantity"] = ents.quantity

        negated = ents.negated
        # negation scope: if segment starts with no
        if _is_negated(seg):
            negated = True

        conf = None
        if confidence_fn is not None:
            try:
                conf = confidence_fn(seg)
            except Exception:
                conf = None

        from .taxonomy import get_intent_spec
        spec = get_intent_spec(intent)
        action = Action(
            intent=intent,
            parameters=params,
            negated=negated,
            confidence=conf,
            requires_confirmation=spec.requires_confirmation,
        )
        actions.append(action)

    # If safety_intent and no actions? ensure
    if safety_intent and not any(a.intent == safety_intent for a in actions):
        actions.insert(0, Action(intent=safety_intent, parameters={}, negated=False, requires_confirmation=False))

    # Unknown handling: if all unknown and raw not empty
    if all(a.intent == Intent.UNKNOWN.value for a in actions):
        cmd.status = "UNKNOWN"

    cmd.actions = actions
    cmd.unresolved_references = unresolved

    # Validation
    validate_command(cmd)

    # If single action and unknown status already set, keep
    cmd.processing_time_ms = (time.perf_counter() - start) * 1000
    return cmd
