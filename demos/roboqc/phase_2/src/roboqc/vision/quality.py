"""Quality classification from observable criteria and runtime rules.

Verdicts:
- DEFECTIVE: an active rule rejects the object, or a measured anomaly is visible
  (too few solid edges, area outside the expected range, color or type not allowed).
- UNCERTAIN: the evidence is insufficient (partially visible, inconclusive color,
  unrecognized outline). Uncertainty never turns into GOOD.
- GOOD: no anomaly is observed and the object is fully evaluable.

Runtime rules come from voice or text commands such as "Ahora las piezas azules
son defectuosas" and change the verdict without retraining anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

GOOD = "GOOD"
DEFECTIVE = "DEFECTIVE"
UNCERTAIN = "UNCERTAIN"
VERDICTS = (GOOD, DEFECTIVE, UNCERTAIN)


@dataclass(frozen=True)
class QualityRule:
    name: str  # human-readable label shown in the UI, e.g. "piezas azules"
    verdict: str  # GOOD or DEFECTIVE
    color: Optional[str] = None
    object_type: Optional[str] = None

    def matches(self, color: Optional[str], object_type: str) -> bool:
        if self.color is None and self.object_type is None:
            return False
        if self.color is not None and self.color != color:
            return False
        if self.object_type is not None and self.object_type != object_type:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "verdict": self.verdict, "color": self.color, "object_type": self.object_type}


@dataclass(frozen=True)
class QualityEvaluation:
    quality: str
    reasons: List[str]


class QualityRuleSet:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._rules: List[QualityRule] = []  # newest first

    @property
    def rules(self) -> List[QualityRule]:
        return list(self._rules)

    def add_rule(self, rule: QualityRule) -> None:
        """Insert a rule with priority. A newer rule for the same target replaces the older one."""
        self._rules = [r for r in self._rules if not (r.color == rule.color and r.object_type == rule.object_type)]
        self._rules.insert(0, rule)

    def clear(self) -> None:
        self._rules = []

    def evaluate(
        self,
        color: Optional[str],
        object_type: str,
        solidity: float,
        area_px: float,
        color_confidence: float,
        partial: bool,
    ) -> QualityEvaluation:
        reasons: List[str] = []
        uncertain: List[str] = []

        rule = next((r for r in self._rules if r.matches(color, object_type)), None)
        if rule is not None and rule.verdict == DEFECTIVE:
            reasons.append(f"regla activa: {rule.name} defectuosas")

        expected_colors = self.cfg.get("expected_colors")
        if expected_colors and color is not None and color not in expected_colors:
            reasons.append(f"color {color} no admitido")

        expected_types = self.cfg.get("expected_types")
        if expected_types and object_type != "desconocido":
            if object_type == "circular":
                if not {"cilindro", "esfera"} & set(expected_types):
                    reasons.append("forma circular no admitida")
            elif object_type not in expected_types:
                reasons.append(f"tipo {object_type} no esperado")

        if partial:
            uncertain.append("pieza parcialmente visible: geometria no evaluada")
        else:
            if solidity < self.cfg["min_solidity"]:
                reasons.append(f"bordes o esquinas anomalos (solidez {solidity:.2f})")
            expected_area = self.cfg.get("expected_area_px")
            if expected_area:
                tol = float(self.cfg["expected_area_tolerance"])
                low, high = float(expected_area[0]) * (1 - tol), float(expected_area[1]) * (1 + tol)
                if not low <= area_px <= high:
                    reasons.append(f"tamano fuera de rango ({area_px:.0f} px)")

        if color is None:
            uncertain.append("color no concluyente")
        elif color_confidence < self.cfg["min_color_confidence"]:
            uncertain.append(f"color poco concluyente ({color_confidence:.0%})")
        if object_type == "desconocido":
            uncertain.append("forma no reconocida")

        if reasons:
            return QualityEvaluation(DEFECTIVE, reasons)
        if uncertain:
            return QualityEvaluation(UNCERTAIN, uncertain)
        return QualityEvaluation(GOOD, [])
