"""Structured inspection report built from a WorldSnapshot. No robot involved."""

from __future__ import annotations

from typing import Any, Dict, List

from .world_state import WorldSnapshot

QUALITY_LABEL_ES = {"GOOD": "BUENA", "DEFECTIVE": "DEFECTUOSA", "UNCERTAIN": "INCIERTA"}


def build_inspection_report(snapshot: WorldSnapshot) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "timestamp": snapshot.timestamp,
        "summary": snapshot.counts(),
        "active_rules": [r.to_dict() for r in snapshot.rules],
        "events": {
            "entered": list(snapshot.events.entered),
            "moved": list(snapshot.events.moved),
            "disappeared": list(snapshot.events.disappeared),
        },
        "objects": [o.to_dict() for o in snapshot.objects],
    }


def format_report(report: Dict[str, Any]) -> str:
    """Plain-text table for the terminal."""
    s = report["summary"]
    lines: List[str] = [
        f"Visibles: {s['visible']}  (seguidas: {s['tracked']})   "
        f"Buenas: {s['GOOD']}  Defectuosas: {s['DEFECTIVE']}  Inciertas: {s['UNCERTAIN']}",
    ]
    if report["active_rules"]:
        lines.append("Reglas activas: " + "; ".join(
            f"{r['name']} -> {QUALITY_LABEL_ES.get(r['verdict'], r['verdict'])}" for r in report["active_rules"]
        ))
    header = f"{'ID':8}{'TIPO':11}{'COLOR':10}{'ESTADO':11}{'VISTA':10}{'POS(px)':16}MOTIVO"
    lines.append(header)
    lines.append("-" * len(header))
    for o in report["objects"]:
        pos = f"({o['position_px'][0]:.0f}, {o['position_px'][1]:.0f})"
        reasons = "; ".join(o["quality_reasons"])
        lines.append(
            f"{o['object_id']:8}{o['type']:11}{(o['color'] or '?'):10}"
            f"{QUALITY_LABEL_ES.get(o['quality'], o['quality']):11}{o['detection_status']:10}{pos:16}{reasons}"
        )
    return "\n".join(lines)
