"""Del texto del usuario a un objetivo de kit: qué piezas y cuántas.

Reglas, no un modelo: el vocabulario de un kit es pequeño y cerrado (números,
colores, formas) y una regla que falla lo dice, en lugar de adivinar.

    >>> parse("Prepara un kit con una pieza roja y dos azules").goal
    {('red', None): 1, ('blue', None): 2}

Una forma que no se repite se hereda del elemento anterior ("un cubo rojo y dos
azules" son tres cubos). ``None`` como forma significa "cualquier pieza".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

NUMBERS = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}
ALL = {"todas", "todos", "all", "every"}
COLORS = {
    "rojo": "red", "roja": "red", "rojos": "red", "rojas": "red", "red": "red",
    "azul": "blue", "azules": "blue", "blue": "blue",
    "verde": "green", "verdes": "green", "green": "green",
    "amarillo": "yellow", "amarilla": "yellow", "amarillos": "yellow", "amarillas": "yellow", "yellow": "yellow",
}
UNKNOWN_COLORS = {"naranja", "naranjas", "orange", "rosa", "rosas", "pink", "morado", "morada", "purple",
                  "negro", "negra", "black", "blanco", "blanca", "white"}
SHAPES = {
    "cubo": "cube", "cubos": "cube", "cube": "cube", "cubes": "cube",
    "cilindro": "cylinder", "cilindros": "cylinder", "cylinder": "cylinder", "cylinders": "cylinder",
    "prisma": "prism", "prismas": "prism", "prism": "prism", "prisms": "prism",
    "pieza": None, "piezas": None, "bloque": None, "bloques": None,
    "piece": None, "pieces": None, "block": None, "blocks": None,
}
NEGATIONS = {"sin", "without", "except", "excepto", "menos"}
STOP = {"stop", "para", "parar", "detente", "alto", "cancela", "cancelar"}
CONTAINER_WORDS = {"kit", "conjunto", "set", "lote"}

COLOR_ES_F = {"red": ("roja", "rojas"), "blue": ("azul", "azules"), "green": ("verde", "verdes"),
              "yellow": ("amarilla", "amarillas")}
COLOR_ES_M = {"red": ("rojo", "rojos"), "blue": ("azul", "azules"), "green": ("verde", "verdes"),
              "yellow": ("amarillo", "amarillos")}
SHAPE_ES = {None: ("pieza", "piezas", "f"), "cube": ("cubo", "cubos", "m"),
            "cylinder": ("cilindro", "cilindros", "m"), "prism": ("prisma", "prismas", "m")}

Goal = dict[tuple[str, str | None], int | str]


@dataclass
class KitRequest:
    text: str
    goal: Goal = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    stop: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.goal) and not self.errors and not self.stop

    def summary(self) -> str:
        return describe(self.goal)


def _norm(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z]+|\d+", t)


def _new(count: int | str = 1) -> dict:
    return {"count": count, "color": None, "shape": None, "shape_set": False}


def parse(text: str) -> KitRequest:
    req = KitRequest(text=text)
    words = _norm(text)
    if not words:
        req.errors.append("no he entendido nada: escribe qué piezas quieres")
        return req
    if words[0] in STOP or (len(words) <= 2 and set(words) & STOP):
        req.stop = True
        return req

    items: list[dict] = []
    current: dict | None = None
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w in NEGATIONS or (w == "no" and (nxt in COLORS or nxt in SHAPES)):
            req.errors.append("no sé preparar un kit con exclusiones; di sólo qué piezas quieres")
            return req
        count = int(w) if w.isdigit() else NUMBERS.get(w)
        if count is not None and nxt in CONTAINER_WORDS:
            continue   # "un kit", "a set": no es una cantidad
        if count is not None or w in ALL:
            if current and (current["color"] or current["shape_set"]):
                items.append(current)
            current = _new("all" if w in ALL else count)
        elif w in COLORS:
            if current is None or current["color"] is not None:
                if current:
                    items.append(current)
                current = _new()
            current["color"] = COLORS[w]
        elif w in SHAPES:
            if current is None:
                current = _new()
            elif current["shape_set"] and current["color"]:
                items.append(current)
                current = _new()
            current["shape"] = SHAPES[w]
            current["shape_set"] = True
        elif w in UNKNOWN_COLORS:
            req.errors.append(f"no hay piezas de color «{w}»: los colores son rojo, azul, verde y amarillo")
    if current and (current["color"] or current["shape_set"]):
        items.append(current)

    last_shape: str | None = None
    shape_seen = False
    for it in items:
        if it["shape_set"]:
            last_shape, shape_seen = it["shape"], True
        elif shape_seen:
            it["shape"] = last_shape
        if it["color"] is None:
            req.errors.append("falta el color de alguna pieza")
            continue
        if it["count"] == 0:
            req.errors.append("una cantidad de cero no tiene sentido en un kit")
            continue
        key = (it["color"], it["shape"])
        prev = req.goal.get(key, 0)
        req.goal[key] = "all" if "all" in (it["count"], prev) else prev + it["count"]
    if not req.goal and not req.errors:
        req.errors.append("no he encontrado ninguna pieza en el pedido (p. ej. «una roja y dos azules»)")
    return req


def describe(goal: Goal) -> str:
    """'1 pieza roja y 2 piezas azules' — con concordancia de género y número."""
    parts = []
    for (color, shape), n in goal.items():
        sing, plur, gender = SHAPE_ES[shape]
        adj = COLOR_ES_F if gender == "f" else COLOR_ES_M
        if n == "all":
            art = "todas las" if gender == "f" else "todos los"
            parts.append(f"{art} {plur} {adj[color][1]}")
        else:
            parts.append(f"{n} {sing if n == 1 else plur} {adj[color][0] if n == 1 else adj[color][1]}")
    if not parts:
        return "nada"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " y " + parts[-1]
