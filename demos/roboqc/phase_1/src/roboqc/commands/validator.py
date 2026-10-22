"""Command validation rules for RoboQC.

Validates structured commands for completeness, contradictions, ambiguity and
safety. Produces validation errors and status rather than raising.
"""

from __future__ import annotations

from typing import List

from .schema import StructuredCommand, Action
from .taxonomy import SAFETY_CRITICAL, Intent


def _is_empty(v) -> bool:
    return v is None or v == "" or v == "UNSPECIFIED"


def validate_action(action: Action) -> List[str]:
    errors: List[str] = []
    intent = action.intent
    p = action.parameters or {}

    # Safety critical intents should not carry object params unexpectedly; allow.
    if intent in ("PICK_OBJECT", "PLACE_OBJECT", "MOVE_OBJECT"):
        if "object_type" not in p and "object_color" not in p:
            # could be incomplete but not invalid; mark as possibly incomplete
            pass
    if intent in ("PLACE_OBJECT", "MOVE_OBJECT"):
        if p.get("destination_color") is None and p.get("destination_type") is None and p.get("destination_role") is None:
            errors.append("Falta el destino para la accion de colocacion/transferencia.")
    # Negation on safety stop is contradictory
    if intent in SAFETY_CRITICAL and action.negated:
        errors.append("Negacion sobre una orden de parada critica.")
    # Unknown with params
    if intent == "UNKNOWN" and (p and any(v is not None for v in p.values())):
        # strip params for unknown
        pass
    # quantity validation
    q = p.get("quantity", "UNSPECIFIED")
    if isinstance(q, int) and q < 0:
        errors.append("Cantidad negativa no permitida.")
    return errors


def validate_command(cmd: StructuredCommand) -> StructuredCommand:
    errors: List[str] = list(cmd.validation_errors)
    # empty actions
    if not cmd.actions:
        if cmd.status not in ("UNKNOWN",):
            cmd.status = "INVALID"
        errors.append("No se ha reconocido ninguna accion.")
    # per action
    for a in cmd.actions:
        errs = validate_action(a)
        for e in errs:
            if e not in errors:
                errors.append(e)
    # ambiguity detection: unresolved references
    if cmd.unresolved_references:
        if cmd.status == "VALID":
            cmd.status = "AMBIGUOUS"
    # if any validation errors, mark invalid unless ambiguous
    if errors and cmd.status == "VALID":
        cmd.status = "INVALID"
    cmd.validation_errors = errors
    # high-priority safety: force requires_confirmation false for emergency? keep true
    for a in cmd.actions:
        if a.intent in SAFETY_CRITICAL:
            a.requires_confirmation = False
    return cmd
