"""De una frase en español a una orden de clasificación (sin Isaac, sin modelos).

Reglas y no un modelo: el vocabulario de la tarea es pequeño y cerrado (colores,
formas, cantidades, izquierda/derecha) y una regla que no entiende lo dice en
lugar de adivinar. Por qué no el modelo de órdenes del demo 05
(``command_model_v2``, GLiNER): sus destinos son bandejas de colores («la caja
roja»), no «izquierda/derecha»; pesa cientos de MB y, cargado después de arrancar
Isaac, rompe el proceso (ver ``08_recovery_kit`` y ``05_planner_and_executor``).
La entrada es texto, venga del teclado o de Whisper (``voice.py``).

    >>> parse("Clasifica las piezas rojas a la izquierda y las verdes a la derecha").summary()
    'rojas → izquierda; verdes → derecha'

«Izquierda» y «derecha» son las **del robot** (el lado +y del mundo es su
izquierda). El panel gira la imagen cenital para que coincida con esa vista.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from . import attributes as A
from .config import COLOR_ES, COLOR_ES_M, DESTINATIONS

NUMBERS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6}
ALL_WORDS = {"todas", "todos", "toda", "todo"}
REST_WORDS = {"resto", "demas", "restantes", "otras", "otros"}
DEST_WORDS = {
    "izquierda": "izquierda", "izquierdo": "izquierda", "izquierdas": "izquierda", "izq": "izquierda", "izda": "izquierda",
    "derecha": "derecha", "derecho": "derecha", "derechas": "derecha", "der": "derecha", "dcha": "derecha",
}
STOP_WORDS = {"para", "parar", "detente", "detener", "alto", "stop", "cancela", "cancelar", "basta"}
NEGATION_WORDS = {"sin", "excepto", "menos", "salvo"}
UNKNOWN_COLORS = {"naranja", "naranjas", "rosa", "rosas", "morado", "morada", "morados", "moradas", "negro", "negra",
                  "negros", "negras", "blanco", "blanca", "blancos", "blancas", "amarillo", "amarilla", "amarillos",
                  "amarillas", "gris", "grises", "marron", "marrones"}
FILLER = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a", "al", "en", "y", "e", "o", "por", "para",
          "con", "lo", "hacia", "dentro", "caja", "cajas", "lado", "mano", "contenedor", "sitio", "que", "se", "mi", "me",
          "esta", "esa", "ese", "este", "siguiente", "tambien", "luego", "despues", "primero", "ahora", "favor", "por"}
VERBS = {"coloca", "colocar", "pon", "poner", "mete", "meter", "lleva", "llevar", "pasa", "pasar", "recoge", "recoger",
         "coge", "coger", "agarra", "agarrar", "toma", "tomar", "clasifica", "clasificar", "separa", "separar",
         "ordena", "ordenar", "mueve", "mover", "deja", "dejar", "guarda", "guardar", "quiero", "puedes", "robot"}

_KNOWN = (set(A.word_table()) | set(DEST_WORDS) | set(NUMBERS) | ALL_WORDS | REST_WORDS | FILLER | VERBS
          | STOP_WORDS | NEGATION_WORDS | UNKNOWN_COLORS | A.GENERIC_WORDS)


def normalize(text: str) -> list[str]:
    """Minúsculas, sin acentos ni puntuación; ``«ñ»`` → ``n``. Whisper devuelve texto así de limpio o peor."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z]+|\d+", t)


def _repair(token: str) -> str:
    """Corrige una errata de reconocimiento de voz («isquierda», «derrecha») hacia el vocabulario conocido."""
    if token in _KNOWN or token.isdigit() or len(token) < 5:
        return token
    near = difflib.get_close_matches(token, _KNOWN, n=1, cutoff=0.84)
    return near[0] if near else token


@dataclass
class Selector:
    """Qué piezas: ``color``/``shape`` ``None`` = cualquiera; ``count`` entero o ``"all"``; ``rest`` = «las demás»."""

    color: str | None = None
    shape: str | None = None
    count: int | str = "all"
    rest: bool = False
    extra: dict = field(default_factory=dict)       # otros atributos registrados (p. ej. defect)
    plural: bool = False

    def matches(self, color: str | None, shape: str | None, attrs: dict | None = None) -> bool:
        if self.rest:
            return True
        if self.color is not None and color != self.color:
            return False
        if self.shape is not None and shape != self.shape:
            return False
        for name, value in self.extra.items():
            if (attrs or {}).get(name) != value:
                return False
        return True

    def describe(self) -> str:
        if self.rest:
            return "las demás"
        shape_es = {"cube": ("cubo", "cubos"), "cylinder": ("cilindro", "cilindros")}.get(self.shape, ("pieza", "piezas"))
        masc = self.shape is not None
        n = self.count
        plural = n == "all" or (isinstance(n, int) and n > 1)
        noun = shape_es[1 if plural else 0] if self.shape else ("piezas" if plural else "pieza")
        colour = ""
        if self.color:
            table = COLOR_ES_M if masc else COLOR_ES
            colour = " " + table[self.color][1 if plural else 0]
        extra = "".join(" " + A.get(k).values_es.get(v, v) for k, v in self.extra.items())
        if n == "all":
            return f"todos los {noun}{colour}{extra}" if masc else f"todas las {noun}{colour}{extra}"
        return f"{n} {noun}{colour}{extra}"


@dataclass
class Rule:
    selector: Selector
    dest: str
    dest_defaulted: bool = False

    def describe(self) -> str:
        s = self.selector
        if s.rest:
            who = "las demás"
        else:
            who = ""
            if s.color:
                who = (COLOR_ES_M if s.shape else COLOR_ES)[s.color][1 if s.count != 1 else 0]
            if s.shape:
                who = (f"{'cubos' if s.shape == 'cube' else 'cilindros'} {who}").strip()
            if not who:
                who = "piezas"
            if isinstance(s.count, int):
                who = f"{s.count} {who}"
        return f"{who} → {self.dest}"


@dataclass
class SortOrder:
    text: str
    intent: str = "sort"                    # "sort" | "stop" | "none"
    rules: list[Rule] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.intent == "sort" and bool(self.rules) and not self.errors

    def summary(self) -> str:
        if self.intent == "stop":
            return "parar"
        return "; ".join(r.describe() for r in self.rules) if self.rules else "nada"

    def as_dict(self) -> dict:
        return {"text": self.text, "intent": self.intent, "ok": self.ok, "summary": self.summary(),
                "rules": [{"who": r.selector.describe(), "dest": r.dest, "count": r.selector.count,
                           "color": r.selector.color, "shape": r.selector.shape, "rest": r.selector.rest,
                           "dest_defaulted": r.dest_defaulted} for r in self.rules],
                "errors": self.errors, "warnings": self.warnings}


def parse(text: str, default_dest: str = "derecha") -> SortOrder:
    """Interpreta una orden. Nunca lanza: los problemas van en ``errors`` y ``warnings`` (en español)."""
    order = SortOrder(text=text)
    words = [_repair(w) for w in normalize(text)]
    if not words:
        order.intent = "none"
        order.errors.append("No he entendido nada: di o escribe qué piezas mover y a qué caja.")
        return order
    if words[0] in STOP_WORDS or (len(words) <= 3 and set(words) & STOP_WORDS and not set(words) & set(A.word_table())):
        order.intent = "stop"
        return order

    table = A.word_table()
    selectors: list[tuple[int, Selector]] = []      # (posición, selector)
    dests: list[tuple[int, str]] = []
    current: Selector | None = None
    pending_count: int | str | None = None
    has_attr = False

    def close(pos: int) -> None:
        nonlocal current, pending_count, has_attr
        if current is not None:
            selectors.append((pos, current))
        current, pending_count, has_attr = None, None, False

    for pos, w in enumerate(words):
        nxt = words[pos + 1] if pos + 1 < len(words) else ""
        if w in NEGATION_WORDS:
            order.errors.append("No sé hacer exclusiones («todas menos…»); di qué piezas quieres mover.")
            return order
        if w in DEST_WORDS:
            close(pos)
            dests.append((pos, DEST_WORDS[w]))
            continue
        if w in UNKNOWN_COLORS:
            order.errors.append(f"No hay piezas de color «{w}»: los colores son rojo, verde y azul.")
            continue
        count = int(w) if w.isdigit() else NUMBERS.get(w)
        if count is not None and nxt in ("de", "del"):
            continue
        if count is not None or w in ALL_WORDS:
            if current is not None and has_attr:
                close(pos)
            current = current or Selector()
            pending_count = "all" if w in ALL_WORDS else count
            current.count = pending_count
            current.plural = w in ALL_WORDS or (isinstance(count, int) and count > 1)
            continue
        if w in REST_WORDS:
            if current is not None and has_attr:
                close(pos)
            current = Selector(rest=True)
            has_attr = True
            continue
        if w in A.GENERIC_WORDS:
            if current is None:
                current = Selector()
            current.plural = current.plural or w.endswith("s")
            has_attr = True
            continue
        if w in table:
            name, value = table[w]
            attr = A.get(name)
            if not attr.implemented:
                order.errors.append(f"Todavía no sé clasificar por «{attr.label_es}» (no hay clasificador). "
                                    "Pide piezas por color o por forma.")
                continue
            if current is not None and has_attr and _taken(current, name):
                close(pos)
            if current is None:
                current = Selector()
            if name == "color":
                current.color = value
            elif name == "shape":
                current.shape = value
            else:
                current.extra[name] = value
            current.plural = current.plural or w.endswith("s")
            has_attr = True
            continue
        if w in FILLER or w in VERBS:
            continue
        if len(w) > 2:
            order.warnings.append(f"No conozco la palabra «{w}»; la ignoro.")
    close(len(words))

    # ya hay selectores y destinos; falta unirlos
    rules, leading = _bind(selectors, dests)
    for sel, dest in rules:
        if dest is None:
            dest = default_dest
            order.warnings.append(f"No has dicho a qué caja: uso la caja {default_dest}.")
            defaulted = True
        else:
            defaulted = False
        if sel.count is not None and sel.count != "all" and sel.count == 0:
            order.errors.append("Una cantidad de cero no tiene sentido.")
            continue
        if sel.count == "all" and not sel.plural and (sel.color or sel.shape) and not _all_explicit(words):
            sel.count = 1                      # «la pieza roja»: una
        order.rules.append(Rule(sel, dest, defaulted))
    if not order.rules and not order.errors:
        order.errors.append("No he encontrado qué piezas mover (por ejemplo: «las rojas a la izquierda»).")
    if dests and not selectors:
        order.errors.append("Has dicho una caja pero no qué piezas llevar a ella.")
    _check_conflicts(order)
    return order


def _all_explicit(words: list[str]) -> bool:
    return bool(set(words) & ALL_WORDS)


def _taken(sel: Selector, attr_name: str) -> bool:
    return {"color": sel.color is not None, "shape": sel.shape is not None}.get(attr_name, attr_name in sel.extra)


def _bind(selectors: list[tuple[int, Selector]], dests: list[tuple[int, str]]):
    """Une cada destino con los selectores que lo preceden; un destino inicial toma los que le siguen."""
    out: list[tuple[Selector, str | None]] = []
    bound: set[int] = set()
    prev_pos = -1
    empty_dests: list[tuple[int, str]] = []
    for dpos, dest in dests:
        mine = [i for i, (spos, _s) in enumerate(selectors) if prev_pos < spos <= dpos and i not in bound]
        if not mine:
            empty_dests.append((dpos, dest))
        for i in mine:
            bound.add(i)
            out.append((selectors[i][1], dest))
        prev_pos = dpos
    for dpos, dest in empty_dests:
        later = [i for i, (spos, _s) in enumerate(selectors) if spos > dpos and i not in bound]
        nxt_d = min((p for p, _d in dests if p > dpos), default=10 ** 9)
        for i in later:
            if selectors[i][0] <= nxt_d or nxt_d == 10 ** 9:
                bound.add(i)
                out.append((selectors[i][1], dest))
    for i, (_p, sel) in enumerate(selectors):
        if i not in bound:
            out.append((sel, None))
    # conserva el orden en que se dijo (los «demás» al final: se calculan con lo que quede)
    order_index = {id(s): i for i, (_p, s) in enumerate(selectors)}
    out.sort(key=lambda r: (r[0].rest, order_index[id(r[0])]))
    return out, bool(empty_dests)


def _check_conflicts(order: SortOrder) -> None:
    """Dos reglas que mandan las mismas piezas a cajas distintas son una orden contradictoria."""
    seen: dict[tuple, str] = {}
    for r in order.rules:
        s = r.selector
        if s.rest:
            continue
        key = (s.color, s.shape, tuple(sorted(s.extra.items())))
        if key in seen and seen[key] != r.dest and s.count == "all":
            order.errors.append("Has mandado las mismas piezas a las dos cajas; elige una.")
        seen.setdefault(key, r.dest)


def to_assignments(order: SortOrder, pieces: list[tuple[str | None, str | None, dict]]) -> list[tuple[int, int]]:
    """Reparto *previsto* de piezas ``(color, forma, atributos)`` a reglas: ``[(índice de pieza, índice de regla)]``.

    Sólo estima cuántas van a cada caja (para avisar de que no caben); qué pieza concreta se coge
    en cada momento lo decide ``selection.py`` con lo que ve la cámara.
    """
    taken: set[int] = set()
    out: list[tuple[int, int]] = []
    for ri, r in enumerate(order.rules):
        if r.selector.rest:
            continue
        quota = r.selector.count
        for pi, (c, s, a) in enumerate(pieces):
            if pi in taken or not r.selector.matches(c, s, a):
                continue
            if quota != "all" and quota <= 0:
                break
            taken.add(pi)
            out.append((pi, ri))
            if quota != "all":
                quota -= 1
    for ri, r in enumerate(order.rules):
        if r.selector.rest:
            for pi in range(len(pieces)):
                if pi not in taken:
                    taken.add(pi)
                    out.append((pi, ri))
    return out


def capacity_warnings(order: SortOrder, pieces: list[tuple[str | None, str | None, dict]], capacity: int,
                      already: dict[str, int] | None = None) -> list[str]:
    """Avisos si una caja recibiría más piezas de las que caben."""
    load = dict(already or {})
    for _pi, ri in to_assignments(order, pieces):
        d = order.rules[ri].dest
        load[d] = load.get(d, 0) + 1
    return [f"En la caja {d} caben {capacity} piezas y esta orden manda {n}; las que sobren se quedan en el contenedor."
            for d, n in load.items() if d in DESTINATIONS and n > capacity]
