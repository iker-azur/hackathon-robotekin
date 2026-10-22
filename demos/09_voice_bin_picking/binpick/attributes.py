"""Atributos de una pieza por los que se puede pedir y clasificar: color, forma, ….

Añadir un criterio nuevo (p. ej. «defectuosa») es registrar un ``Attribute``: su
vocabulario en español (lo usa ``commands.py``) y, cuando exista, la función que
lo mide sobre la máscara de la pieza (la usa la percepción). Nada más cambia: ni
el parser, ni la selección, ni la máquina de estados saben qué atributos hay.

Hoy hay dos clasificadores reales (color y forma). ``defect`` está declarado pero
**sin clasificador**: pedir «las defectuosas» devuelve un error claro en vez de
adivinar. Ver ``docs/02_arquitectura.md`` (apartado «defectos») para conectar uno.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

# tono OpenCV (0–180) de cada color del taller; medido en 08_recovery_kit/tools/capture_perception.py
HUES = {"red": 0.0, "green": 67.0, "blue": 107.0}
HUE_TOL = {"red": 10.0, "green": 18.0, "blue": 16.0}   # el marrón de las cajas (tono ~15) no debe pasar por rojo
MIN_SAT = 150                                           # la del demo 08; el marrón llega a ~125
MIN_VAL = 40


@dataclass
class Attribute:
    name: str                                   # clave en ``Instance.attributes``
    label_es: str                               # cómo lo llama la interfaz
    vocabulary: dict[str, str]                  # palabra normalizada → valor
    classify: Callable[[np.ndarray, np.ndarray], tuple[str | None, float]] | None = None
    values_es: dict[str, str] = field(default_factory=dict)

    @property
    def implemented(self) -> bool:
        return self.classify is not None


REGISTRY: dict[str, Attribute] = {}


def register(attr: Attribute) -> Attribute:
    REGISTRY[attr.name] = attr
    return attr


def get(name: str) -> Attribute:
    return REGISTRY[name]


def word_table() -> dict[str, tuple[str, str]]:
    """palabra normalizada → (atributo, valor), de todos los atributos registrados."""
    out: dict[str, tuple[str, str]] = {}
    for attr in REGISTRY.values():
        for word, value in attr.vocabulary.items():
            out[word] = (attr.name, value)
    return out


# ------------------------------------------------------------------- color
def classify_color(rgb: np.ndarray, mask: np.ndarray) -> tuple[str | None, float]:
    """Color de una pieza por el tono mediano de su máscara (erosionada para quitar el borde).

    Devuelve ``(color, confianza)``; ``(None, 0)`` si la máscara no es de un color
    saturado de los del taller (p. ej. el brazo amarillo o la caja marrón).
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return None, 0.0
    core = cv2.erode(m, np.ones((3, 3), np.uint8))
    if core.sum() < 6:
        core = m
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    px = hsv[core.astype(bool)]
    px = px[(px[:, 1] >= MIN_SAT) & (px[:, 2] >= MIN_VAL)]
    if len(px) < 4:
        return None, 0.0
    # tono circular: se vota por color, no se promedia (el rojo está en 0 y en 180)
    votes: dict[str, int] = {}
    for name, hue in HUES.items():
        d = np.abs(px[:, 0].astype(float) - hue)
        d = np.minimum(d, 180 - d)
        votes[name] = int((d <= HUE_TOL[name]).sum())
    best = max(votes, key=votes.get)
    conf = votes[best] / len(px)
    return (best, conf) if conf >= 0.5 else (None, conf)


# ------------------------------------------------------------------- forma
def classify_shape_from_fill(fill: float) -> str:
    """Relleno del rectángulo mínimo → forma (umbrales medidos en el demo 08: cubos 0.97–0.99, cilindros 0.78–0.82)."""
    if fill > 0.90:
        return "cube"
    if fill > 0.70:
        return "cylinder"
    return "?"


register(Attribute(
    "color", "color",
    {"rojo": "red", "roja": "red", "rojos": "red", "rojas": "red",
     "verde": "green", "verdes": "green",
     "azul": "blue", "azules": "blue"},
    classify=classify_color,
    values_es={"red": "rojo", "green": "verde", "blue": "azul"},
))

register(Attribute(
    "shape", "forma",
    {"cubo": "cube", "cubos": "cube", "dado": "cube", "dados": "cube",
     "cilindro": "cylinder", "cilindros": "cylinder"},
    # la forma sale de la geometría de la máscara (ver segmentation.finalize), no de los píxeles
    classify=lambda rgb, mask: (None, 0.0),
    values_es={"cube": "cubo", "cylinder": "cilindro"},
))

# Punto de extensión: clasificación por defectos u otras características industriales.
# Sin ``classify``: el vocabulario se reconoce para poder decir «todavía no sé hacerlo».
register(Attribute(
    "defect", "defecto",
    {"defectuosa": "defective", "defectuosas": "defective", "defectuoso": "defective", "defectuosos": "defective",
     "danada": "defective", "danadas": "defective", "danado": "defective", "danados": "defective",
     "buena": "ok", "buenas": "ok", "bueno": "ok", "buenos": "ok",
     "correcta": "ok", "correctas": "ok", "correcto": "ok", "correctos": "ok"},
    classify=None,
    values_es={"defective": "defectuosa", "ok": "correcta"},
))

# palabras que nombran «una pieza» sin decir cuál
GENERIC_WORDS = {"pieza", "piezas", "objeto", "objetos", "bloque", "bloques", "elemento", "elementos"}
