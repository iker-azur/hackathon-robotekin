"""Versioned structured command schema for RoboQC.

Uses plain dataclasses (no mandatory third-party dependency) with explicit
serialization helpers. Future robot planners can consume these objects directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


@dataclass
class Action:
    intent: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    negated: bool = False
    confidence: Optional[float] = None
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "parameters": self.parameters,
            "negated": self.negated,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(
            intent=str(data.get("intent", "UNKNOWN")),
            parameters=dict(data.get("parameters") or {}),
            negated=bool(data.get("negated", False)),
            confidence=data.get("confidence"),
            requires_confirmation=bool(data.get("requires_confirmation", False)),
        )


@dataclass
class StructuredCommand:
    schema_version: str = SCHEMA_VERSION
    source: str = "text"  # voice | text | manual
    transcription: str = ""
    normalized_text: str = ""
    status: str = "VALID"  # VALID | AMBIGUOUS | UNKNOWN | INVALID
    actions: List[Action] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    processing_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "transcription": self.transcription,
            "normalized_text": self.normalized_text,
            "status": self.status,
            "actions": [a.to_dict() for a in self.actions],
            "unresolved_references": list(self.unresolved_references),
            "validation_errors": list(self.validation_errors),
            "processing_time_ms": self.processing_time_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuredCommand":
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            source=str(data.get("source", "text")),
            transcription=str(data.get("transcription", "")),
            normalized_text=str(data.get("normalized_text", "")),
            status=str(data.get("status", "VALID")),
            actions=[Action.from_dict(a) for a in (data.get("actions") or [])],
            unresolved_references=list(data.get("unresolved_references") or []),
            validation_errors=list(data.get("validation_errors") or []),
            processing_time_ms=data.get("processing_time_ms"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "StructuredCommand":
        return cls.from_dict(json.loads(raw))
