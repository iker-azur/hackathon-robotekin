"""Deterministic intent rules for Phase 2 visual queries (Spanish).

These rules run before the trained classifier. The Phase 1 models were trained
without visual-query examples, so they cannot be trusted for these commands.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


def fold_text(text: str) -> str:
    """Lowercase and strip accents so patterns match "¿Dónde?" and "donde"."""
    nfd = unicodedata.normalize("NFD", text.lower())
    return " ".join("".join(c for c in nfd if unicodedata.category(c) != "Mn").split())


_LOCATE = re.compile(r"\b(donde|ubica|ubicar|localiza|localizar)\b")
_SHOW = re.compile(r"\b(muestra|muestrame|muestrales|ensena|ensename|resalta|resaltar)\b")
_SELECT = re.compile(r"\b(selecciona|selecciones|seleccionar|elige|elija|escoge|escoja)\b")
_COUNT = re.compile(r"\b(cuantas?|cuantos?)\b")
_EXISTS = re.compile(r"\bhay (alguna|algun|ninguna|ningun|piezas?|cubos?|cilindros?|esferas?|prismas?|objetos?)\b")
_STATUS_WORDS = re.compile(r"\bquedan\b")


def classify_vision_intent(text: str) -> Optional[str]:
    """Return a Phase 2 vision intent name, or None if the text is not a visual query."""
    t = fold_text(text)
    if _LOCATE.search(t):
        return "LOCATE_OBJECT"
    if _SELECT.search(t):
        return "SELECT_OBJECTS"
    if _SHOW.search(t):
        return "SHOW_OBJECTS"
    # "¿Cuántas piezas quedan?" is the Phase 1 status request and keeps its label.
    if _STATUS_WORDS.search(t):
        return None
    if _COUNT.search(t) or _EXISTS.search(t):
        return "COUNT_OBJECTS"
    return None
