"""Industrial command taxonomy for RoboQC Phase 1.

This module is the single source of truth for supported intents. Keep it
extensible: new intents should be added here and documented in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set


class Intent(str, Enum):
    # Operation management
    START_PROCESS = "START_PROCESS"
    STOP_PROCESS = "STOP_PROCESS"
    PAUSE_PROCESS = "PAUSE_PROCESS"
    RESUME_PROCESS = "RESUME_PROCESS"
    RESET_SYSTEM = "RESET_SYSTEM"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    # Object manipulation
    PICK_OBJECT = "PICK_OBJECT"
    PLACE_OBJECT = "PLACE_OBJECT"
    MOVE_OBJECT = "MOVE_OBJECT"
    RELEASE_OBJECT = "RELEASE_OBJECT"
    RETURN_OBJECT = "RETURN_OBJECT"

    # Industrial inspection and sorting
    INSPECT_OBJECT = "INSPECT_OBJECT"
    INSPECT_BATCH = "INSPECT_BATCH"
    SORT_OBJECTS = "SORT_OBJECTS"
    CLASSIFY_OBJECT = "CLASSIFY_OBJECT"
    CHANGE_QUALITY_RULE = "CHANGE_QUALITY_RULE"
    CHANGE_DESTINATION = "CHANGE_DESTINATION"
    MARK_GOOD = "MARK_GOOD"
    MARK_DEFECTIVE = "MARK_DEFECTIVE"
    QUARANTINE_OBJECT = "QUARANTINE_OBJECT"

    # Recovery and replanning
    RETRY_ACTION = "RETRY_ACTION"
    SKIP_OBJECT = "SKIP_OBJECT"
    REPLAN_TASK = "REPLAN_TASK"
    CANCEL_ACTION = "CANCEL_ACTION"
    CONFIRM_ACTION = "CONFIRM_ACTION"

    # Information and configuration
    REQUEST_STATUS = "REQUEST_STATUS"
    REQUEST_REPORT = "REQUEST_REPORT"
    REQUEST_OBJECT_INFO = "REQUEST_OBJECT_INFO"
    CALIBRATE_SYSTEM = "CALIBRATE_SYSTEM"
    SET_SPEED = "SET_SPEED"

    # Unknown / unsupported
    UNKNOWN = "UNKNOWN"


INTENT_DESCRIPTIONS_ES: Dict[str, str] = {
    Intent.START_PROCESS.value: "Iniciar un proceso",
    Intent.STOP_PROCESS.value: "Detener el proceso actual",
    Intent.PAUSE_PROCESS.value: "Pausar temporalmente",
    Intent.RESUME_PROCESS.value: "Reanudar el procesamiento",
    Intent.RESET_SYSTEM.value: "Reiniciar el flujo de trabajo",
    Intent.EMERGENCY_STOP.value: "Parada de emergencia",
    Intent.PICK_OBJECT.value: "Coger o sujetar objetos",
    Intent.PLACE_OBJECT.value: "Colocar objetos en un destino",
    Intent.MOVE_OBJECT.value: "Transferir objetos",
    Intent.RELEASE_OBJECT.value: "Soltar objeto sujetado",
    Intent.RETURN_OBJECT.value: "Devolver un objeto",
    Intent.INSPECT_OBJECT.value: "Inspeccionar un objeto",
    Intent.INSPECT_BATCH.value: "Inspeccionar varios objetos",
    Intent.SORT_OBJECTS.value: "Clasificar u ordenar objetos",
    Intent.CLASSIFY_OBJECT.value: "Determinar la calidad",
    Intent.CHANGE_QUALITY_RULE.value: "Modificar criterio de calidad",
    Intent.CHANGE_DESTINATION.value: "Cambiar destino de clasificacion",
    Intent.MARK_GOOD.value: "Marcar como aceptable",
    Intent.MARK_DEFECTIVE.value: "Marcar como defectuoso",
    Intent.QUARANTINE_OBJECT.value: "Enviar a revision",
    Intent.RETRY_ACTION.value: "Reintentar una accion",
    Intent.SKIP_OBJECT.value: "Saltar un objeto",
    Intent.REPLAN_TASK.value: "Recalcular el plan",
    Intent.CANCEL_ACTION.value: "Cancelar una accion pendiente",
    Intent.CONFIRM_ACTION.value: "Confirmar una accion propuesta",
    Intent.REQUEST_STATUS.value: "Consultar estado actual",
    Intent.REQUEST_REPORT.value: "Solicitar informe de calidad",
    Intent.REQUEST_OBJECT_INFO.value: "Consultar informacion de un objeto",
    Intent.CALIBRATE_SYSTEM.value: "Solicitar calibracion",
    Intent.SET_SPEED.value: "Cambiar velocidad de operacion",
    Intent.UNKNOWN.value: "Instruccion no soportada",
}

# Phase 2 vision queries. They are resolved by deterministic rules
# (commands/vision_intents.py) before the trained classifier runs. They are kept
# out of Intent/ALL_INTENTS on purpose: the Phase 1 dataset and models cover
# exactly ALL_INTENTS, so adding them there would require regenerating data.
VISION_INTENTS: Dict[str, str] = {
    "COUNT_OBJECTS": "Contar piezas visibles",
    "LOCATE_OBJECT": "Localizar una pieza visible",
    "SHOW_OBJECTS": "Resaltar piezas visibles",
    "SELECT_OBJECTS": "Seleccionar piezas visibles",
}
INTENT_DESCRIPTIONS_ES.update(VISION_INTENTS)


# Safety-critical intents that must also be caught by a deterministic layer.
SAFETY_CRITICAL: Set[str] = {
    Intent.EMERGENCY_STOP.value,
    Intent.STOP_PROCESS.value,
}

# Intents that should require confirmation before execution.
REQUIRES_CONFIRMATION: Set[str] = {
    Intent.RESET_SYSTEM.value,
    Intent.CHANGE_QUALITY_RULE.value,
    Intent.CHANGE_DESTINATION.value,
    Intent.EMERGENCY_STOP.value,
}

ALL_INTENTS: List[str] = [i.value for i in Intent]


@dataclass
class IntentSpec:
    name: str
    description_es: str
    safety_critical: bool = False
    requires_confirmation: bool = False


def get_intent_spec(name: str) -> IntentSpec:
    n = name.upper()
    return IntentSpec(
        name=n,
        description_es=INTENT_DESCRIPTIONS_ES.get(n, "Desconocido"),
        safety_critical=n in SAFETY_CRITICAL,
        requires_confirmation=n in REQUIRES_CONFIRMATION,
    )
