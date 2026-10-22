"""Spanish entity extraction and normalization for RoboQC commands.

This module is deterministic and dependency-free. It normalizes morphological
variants to canonical labels and extracts structured parameters from a single
command segment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLOR_FORMS: Dict[str, str] = {}
_COLOR_CANONICAL = [
    "rojo", "azul", "verde", "amarillo", "blanco", "negro",
    "naranja", "morado", "rosa", "gris", "marron",
]
# Build morphological variants -> canonical
_COLOR_VARIANTS: Dict[str, List[str]] = {
    "rojo": ["rojo", "roja", "rojos", "rojas"],
    "azul": ["azul", "azules"],
    "verde": ["verde", "verdes"],
    "amarillo": ["amarillo", "amarilla", "amarillos", "amarillas"],
    "blanco": ["blanco", "blanca", "blancos", "blancas"],
    "negro": ["negro", "negra", "negros", "negras"],
    "naranja": ["naranja", "naranjas"],
    "morado": ["morado", "morada", "morados", "moradas"],
    "rosa": ["rosa", "rosas"],
    "gris": ["gris", "grises"],
    "marron": ["marron", "marrones", "marrón", "marrones"],
}
for canon, forms in _COLOR_VARIANTS.items():
    for f in forms:
        COLOR_FORMS[f.lower()] = canon
# also accept accented input
COLOR_FORMS["marrón"] = "marron"

# Object types (canonical singular)
OBJECT_TYPES = [
    "pieza", "cubo", "cilindro", "esfera", "prisma",
    "componente", "producto", "objeto",
]
_OBJECT_PLURALS: Dict[str, str] = {
    "piezas": "pieza", "cubos": "cubo", "cilindros": "cilindro",
    "esferas": "esfera", "prismas": "prisma", "componentes": "componente",
    "productos": "producto", "objetos": "objeto",
}
for o in OBJECT_TYPES:
    _OBJECT_PLURALS.setdefault(o, o)

OBJECT_GENDER: Dict[str, str] = {
    "pieza": "f", "cubo": "m", "cilindro": "m", "esfera": "f",
    "prisma": "m", "componente": "m", "producto": "m", "objeto": "m",
}

# Destination types
DESTINATION_TYPES = [
    "caja", "contenedor", "bandeja", "zona", "area", "área",
]
DESTINATION_GENDER: Dict[str, str] = {
    "caja": "f", "contenedor": "m", "bandeja": "f", "zona": "f",
    "area": "f", "área": "f",
}

# Quality normalization
QUALITY_FORMS: Dict[str, str] = {
    "bueno": "bueno", "buena": "bueno", "buenos": "bueno", "buenas": "bueno",
    "correcto": "bueno", "correcta": "bueno", "correctos": "bueno", "correctas": "bueno",
    "valido": "bueno", "valida": "bueno", "válido": "bueno", "válida": "bueno",
    "aceptable": "bueno", "aceptables": "bueno",
    "defectuoso": "defectuoso", "defectuosa": "defectuoso",
    "defectuosos": "defectuoso", "defectuosas": "defectuoso",
    "danado": "defectuoso", "dañado": "defectuoso", "dañada": "defectuoso",
    "roto": "defectuoso", "rota": "defectuoso", "rotos": "defectuoso", "rotas": "defectuoso",
    "incorrecto": "defectuoso", "incorrecta": "defectuoso",
    "desconocido": "desconocido", "desconocida": "desconocido",
    "malo": "defectuoso", "mala": "defectuoso", "malos": "defectuoso", "malas": "defectuoso",
}
for k in list(QUALITY_FORMS.keys()):
    QUALITY_FORMS[k.lower()] = QUALITY_FORMS[k]

# Semantic destination roles
DESTINATION_ROLE_KEYWORDS: Dict[str, str] = {
    "revision": "REVIEW_BIN", "revisión": "REVIEW_BIN",
    "rechazo": "REJECT_BIN", "rechazados": "REJECT_BIN",
    "aceptacion": "GOOD_BIN", "aceptación": "GOOD_BIN",
    "inspeccion": "INSPECTION_ZONE", "inspección": "INSPECTION_ZONE",
}

# Quantity words
SPANISH_NUMBERS: Dict[str, int] = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
}
QUANTITY_ALL_WORDS = {"todos", "todas", "todo", "toda"}
QUANTITY_SOME_WORDS = {"algunos", "algunas", "algun", "varios", "varias", "unas", "unos"}
QUANTITY_REMAINING_WORDS = {"restantes", "resto", "sobrantes", "remaining"}


def normalize_color(word: str) -> Optional[str]:
    w = word.strip().lower().rstrip(".,;:!?")
    return COLOR_FORMS.get(w)


def find_colors(text: str) -> List[str]:
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    out = []
    for t in toks:
        c = COLOR_FORMS.get(t)
        if c and c not in out:
            out.append(c)
    return out


def normalize_object_type(word: str) -> Optional[str]:
    w = word.strip().lower().rstrip(".,;:!?")
    if w in _OBJECT_PLURALS:
        return _OBJECT_PLURALS[w]
    if w in OBJECT_TYPES:
        return w
    return None


def detect_object_type(text: str) -> Optional[str]:
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    for t in toks:
        ot = normalize_object_type(t)
        if ot:
            return ot
    return None


def parse_quantity(text: str) -> Any:
    """Return int, 'ALL', 'SOME', 'REMAINING', or 'UNSPECIFIED'."""
    low = text.lower()
    # explicit digits
    m = re.search(r"\b(\d+)\b", low)
    if m:
        return int(m.group(1))
    toks = re.findall(r"\w+", low, flags=re.UNICODE)
    for t in toks:
        if t in QUANTITY_ALL_WORDS:
            return "ALL"
        if t in QUANTITY_SOME_WORDS:
            return "SOME"
        if t in QUANTITY_REMAINING_WORDS:
            return "REMAINING"
        if t in SPANISH_NUMBERS:
            return SPANISH_NUMBERS[t]
    # "un par" -> 2
    if "par" in toks:
        return 2
    return "UNSPECIFIED"


def normalize_quality(word: str) -> Optional[str]:
    w = word.strip().lower().rstrip(".,;:!?")
    return QUALITY_FORMS.get(w)


def detect_quality(text: str) -> Optional[str]:
    toks = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    for t in toks:
        q = QUALITY_FORMS.get(t)
        if q:
            return q
    return None


def detect_destination_type(text: str) -> Optional[str]:
    low = text.lower()
    # multi-word destinations
    if "zona de inspeccion" in low or "zona de inspección" in low:
        return "zona de inspeccion"
    if "zona de revision" in low or "zona de revisión" in low:
        return "zona de revision"
    if "zona de rechazo" in low:
        return "zona de rechazo"
    toks = re.findall(r"\w+", low, flags=re.UNICODE)
    for t in toks:
        if t in ("caja", "contenedor", "bandeja"):
            return t
        if t == "zona":
            return "zona"
    return None


def detect_destination_role(text: str) -> Optional[str]:
    # Whole words only: "inspeccion" is a substring of "inspecciona" and must not match it.
    tokens = set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
    for kw, role in DESTINATION_ROLE_KEYWORDS.items():
        if kw in tokens:
            return role
    return None


@dataclass
class ExtractedEntities:
    object_type: Optional[str] = None
    object_color: Optional[str] = None
    quantity: Any = "UNSPECIFIED"
    object_quality: Optional[str] = None
    destination_type: Optional[str] = None
    destination_color: Optional[str] = None
    destination_role: Optional[str] = None
    negated: bool = False


# Preposition patterns for destination attribution
_DEST_PREPS = ["en la", "en el", "en las", "en los", "a la", "a el", "al", "para la", "para el", "hacia la", "hacia el"]


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def extract_color_roles(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (object_color, destination_color) using syntactic cues.

    Strategy:
    - Find destination noun phrases introduced by prepositions (en/a/para/hacia + article + dest noun).
      The color immediately following the destination noun is the destination color.
    - The object color is the color that modifies the object noun (usually before destination prep).
    """
    low = text.lower()
    obj_color = None
    dest_color = None

    # Destination phrases: prep + article? + dest noun + (color)?
    dest_nouns = ["caja", "contenedor", "bandeja", "zona", "area", "área"]
    # regex: (en|a|para|hacia)\s+(la|el|las|los)?\s*(caja|contenedor|bandeja|zona|area|área)\s+(\w+)
    for m in re.finditer(
        r"(?:en|a|para|hacia)\s+(?:la|el|las|los)?\s*(caja|contenedor|bandeja|zona|area|área)\s+(\w+)",
        low,
    ):
        cand = COLOR_FORMS.get(m.group(2))
        if cand:
            dest_color = cand
    # Also destination color before noun: "caja verde" already captured (color after noun). 
    # Handle "verde caja"? uncommon; skip.

    # Object color: color adjective near object noun, excluding destination phrase.
    # Remove destination phrases to avoid picking dest color.
    cleaned = low
    for m in re.finditer(
        r"(?:en|a|para|hacia)\s+(?:la|el|las|los)?\s*(?:caja|contenedor|bandeja|zona|area|área)\s+\w+",
        low,
    ):
        cleaned = cleaned.replace(m.group(0), " ")
    # Now find first color in cleaned
    for t in _tokens(cleaned):
        c = COLOR_FORMS.get(t)
        if c:
            obj_color = c
            break

    # Fallback: if only one color total, assign to object unless destination phrase present
    colors = find_colors(low)
    if obj_color is None and dest_color is None:
        if len(colors) == 1:
            obj_color = colors[0]
        elif len(colors) >= 2:
            obj_color = colors[0]
            dest_color = colors[1]
    if obj_color is None and len(colors) >= 1 and dest_color is not None:
        # object color is the one not used as dest
        for c in colors:
            if c != dest_color:
                obj_color = c
                break
    if dest_color is None and len(colors) >= 2:
        for c in colors:
            if c != obj_color:
                dest_color = c
                break

    # Special: same color object and destination ("piezas verdes en caja verde")
    # If text explicitly has color twice, our regex should have captured dest_color.
    # If only one occurrence but both roles same color, duplicate not present; check count.
    # Handle case "piezas verdes en la caja verde": regex captures verde dest, cleaned removes phrase -> obj from remaining "verdes"? cleaned removed whole phrase including color, so obj_color None. Fix: count occurrences.
    # Recompute obj_color considering occurrence counts.
    color_counts: Dict[str, int] = {}
    for t in _tokens(low):
        c = COLOR_FORMS.get(t)
        if c:
            color_counts[c] = color_counts.get(c, 0) + 1
    # If a color appears at least twice, it can be both object and destination.
    for c, n in color_counts.items():
        if n >= 2:
            if obj_color is None:
                obj_color = c
            if dest_color is None:
                dest_color = c
    # If dest_color set and obj_color None and color count 1, maybe same color intended? Keep None to avoid hallucination? 
    # But "piezas verdes en caja verde" handled above.

    return obj_color, dest_color


def extract_entities(text: str) -> ExtractedEntities:
    e = ExtractedEntities()
    low = text.lower()
    e.object_type = detect_object_type(text)
    e.quantity = parse_quantity(text)
    q = detect_quality(text)
    e.object_quality = q
    e.destination_type = detect_destination_type(text)
    e.destination_role = detect_destination_role(text)
    oc, dc = extract_color_roles(text)
    e.object_color = oc
    e.destination_color = dc
    # negation
    if re.search(r"\bno\b", low):
        e.negated = True
    # destination role implies destination_type maybe
    if e.destination_type is None and e.destination_role is not None:
        e.destination_type = "zona"
    # normalize destination_type canonical
    if e.destination_type in ("area", "área"):
        e.destination_type = "zona"
    return e


def entities_to_parameters(e: ExtractedEntities) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if e.object_type is not None:
        params["object_type"] = e.object_type
    if e.object_color is not None:
        params["object_color"] = e.object_color
    params["quantity"] = e.quantity
    params["object_quality"] = e.object_quality
    if e.destination_type is not None:
        params["destination_type"] = e.destination_type
    if e.destination_color is not None:
        params["destination_color"] = e.destination_color
    if e.destination_role is not None:
        params["destination_role"] = e.destination_role
    return params
