"""Validated, versioned scene catalog shared by UI, perception and command parsing."""
from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
from dataclasses import replace
from pathlib import Path

import colors
import real_world
import scene_spec

# Additional colours remain unavailable until a recorded validation earns them a place.
AVAILABLE_COLORS = ("red", "green", "blue")
COLOR_LABELS = {"red": "rojo", "green": "verde", "blue": "azul", "yellow": "amarillo", "purple": "morado"}


def default_config():
    objects, containers = [], []
    for item in scene_spec.DEFAULT_SCENE:
        entry = {"id": item.prim, "color": item.color, "x": item.x, "y": item.y}
        if item.container:
            entry.update(name=f"{COLOR_LABELS[item.color].capitalize()} A", aliases=[],
                         size_mm=[round(x * 1000, 3) for x in item.size],
                         cavity_mm=[round(x * 1000, 3) for x in item.inner],
                         wall_mm=item.wall * 1000, floor_mm=item.floor * 1000)
            containers.append(entry)
        else:
            entry.update(kind=item.kind, count=1, size_mm=item.size[0] * 1000)
            objects.append(entry)
    return {"version": 1, "objects": objects, "containers": containers}


def simple_config():
    cfg = default_config()
    cfg["objects"] = [o for o in cfg["objects"] if o["id"] in ("cube", "cube_blue")]
    # Spawn both cubes with the same calibrated mass and contact material.
    for obj in cfg["objects"]:
        if obj["id"] == "cube":
            obj["id"] = "cube_red"
            obj["x"] = .38
    for tray in cfg["containers"]:
        tray["size_mm"][2] = 30
    return cfg


def number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}: debe estar entre {low} y {high}.")
    return float(value)


def build(config):
    """Return canonical JSON and physical items; never silently drop an entry."""
    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("La configuración debe tener version: 1.")
    if set(config) - {"version", "objects", "containers"}:
        raise ValueError("Hay campos desconocidos en la configuración.")
    cfg = copy.deepcopy(config)
    identifiers, names, items, pending = set(), set(), [], []
    for container, group in ((True, "containers"), (False, "objects")):
        entries = cfg.get(group)
        if not isinstance(entries, list) or len(entries) > 64:
            raise ValueError(f"{group}: debe ser una lista de hasta 64 entradas.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Cada entrada debe ser un objeto JSON.")
            allowed = ({"id", "color", "name", "aliases", "size_mm", "cavity_mm", "wall_mm", "floor_mm", "x", "y"}
                       if container else {"id", "color", "kind", "count", "size_mm", "x", "y"})
            if set(entry) - allowed:
                raise ValueError(f"Campos desconocidos en {entry.get('id', 'una entrada')}.")
            identifier, color = entry.get("id"), entry.get("color")
            if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", identifier):
                raise ValueError("El identificador debe usar letras minúsculas, cifras o guion bajo.")
            if color not in AVAILABLE_COLORS:
                raise ValueError(f"Color no validado para esta escena: {color}.")
            count = 1 if container else entry.get("count", 1)
            if type(count) is not int or not 1 <= count <= 64:
                raise ValueError("La cantidad debe ser un entero entre 1 y 64.")
            if container:
                name = entry.get("name", identifier)
                aliases = entry.get("aliases", [])
                if not isinstance(name, str) or not name.strip() or not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
                    raise ValueError("El recipiente necesita un nombre y una lista de alias de texto.")
                for label in [identifier, name, *aliases]:
                    label = " ".join("".join(c for c in unicodedata.normalize("NFD", label.casefold())
                                             if unicodedata.category(c) != "Mn").split())
                    if label in names:
                        raise ValueError(f"Nombre o alias de recipiente repetido: {label}.")
                    names.add(label)
                entry["name"], entry["aliases"] = name.strip(), aliases
                size = entry.get("size_mm", [70, 70, 30])
                cavity = entry.get("cavity_mm", [56, 56])
                if not isinstance(size, list) or len(size) != 3 or not isinstance(cavity, list) or len(cavity) != 2:
                    raise ValueError("size_mm necesita tres medidas; cavity_mm, dos.")
                size = tuple(number(x, "Tamaño del recipiente", 10, 50 if index == 2 else 250) / 1000 for index, x in enumerate(size))
                cavity = tuple(number(x, "Cavidad", 1, 240) / 1000 for x in cavity)
                wall = number(entry.get("wall_mm", 7), "Pared", 1, 20) / 1000
                floor = number(entry.get("floor_mm", 6), "Suelo", 1, 20) / 1000
                if any(abs(size[i] - cavity[i] - 2 * wall) > 1e-6 for i in (0, 1)) or floor >= size[2]:
                    raise ValueError("La cavidad, paredes y altura no coinciden con la geometría exterior.")
            else:
                if entry.get("kind") not in ("cube", "ball"):
                    raise ValueError("Las piezas deben ser cubos o bolas.")
                edge = number(entry.get("size_mm", 30), "Tamaño de pieza", 6, 36) / 1000
                if identifier == scene_spec.SHIPPED_CUBE and (entry["kind"] != "cube" or color != "red" or edge != .03 or count != 1):
                    raise ValueError("El cubo de la escena original debe seguir siendo rojo de 30 mm; usa otro identificador para una pieza distinta.")
            for index in range(count):
                prim = identifier if count == 1 else f"{identifier}_{index + 1}"
                if prim in identifiers:
                    raise ValueError(f"Identificador repetido: {prim}.")
                identifiers.add(prim)
                if container:
                    item = scene_spec.Item(prim=prim, kind="box", color=color, x=0, y=0,
                                           shape="cuboid", size=size, container=True, inner=cavity, wall=wall, floor=floor)
                else:
                    factory = scene_spec._cube if entry["kind"] == "cube" else scene_spec._ball
                    item = factory(prim, color, 0, 0, edge)
                    if not real_world.grasp(item).feasible:
                        raise ValueError(f"No se puede agarrar {prim}: {real_world.grasp(item).reason}")
                positioned = "x" in entry or "y" in entry
                if positioned:
                    if count != 1 or "x" not in entry or "y" not in entry:
                        raise ValueError("Las posiciones explícitas requieren x, y y cantidad uno.")
                    item = replace(item, x=number(entry["x"], "x", *scene_spec.WORK_X),
                                   y=number(entry["y"], "y", *scene_spec.WORK_Y))
                    items.append(item)
                else:
                    pending.append(item)
    if len(items) + len(pending) > 64:
        raise ValueError("La escena supera el límite de piezas y recipientes.")
    for item in items:
        _check_position(item, [other for other in items if other is not item])
    for item in pending:
        placed = None
        # Containers from the back; loose objects from the front. Fixed order is reproducible.
        xs = _range(scene_spec.WORK_X[0] + item.size[0] / 2, scene_spec.WORK_X[1] - item.size[0] / 2)
        ys = _range(scene_spec.WORK_Y[0] + item.size[1] / 2, scene_spec.WORK_Y[1] - item.size[1] / 2)
        if not item.container:
            ys.reverse()
        for y in ys:
            for x in xs:
                candidate = replace(item, x=x, y=y)
                try:
                    _check_position(candidate, items)
                except ValueError:
                    continue
                placed = candidate
                break
            if placed:
                break
        if not placed:
            raise ValueError(f"No queda espacio seguro para {item.prim}. Reduce la cantidad o el tamaño.")
        items.append(placed)
    if not any(not i.container for i in items) or not any(i.container for i in items):
        raise ValueError("La escena necesita al menos una pieza y un recipiente.")
    return cfg, items


def _range(low, high):
    return [round(low + i * .005, 6) for i in range(max(0, int((high - low) / .005) + 1))]


def _check_position(item, others):
    for value, extent, bounds in ((item.x, item.size[0], scene_spec.WORK_X), (item.y, item.size[1], scene_spec.WORK_Y)):
        if value - extent / 2 < bounds[0] - 1e-8 or value + extent / 2 > bounds[1] + 1e-8:
            raise ValueError(f"{item.prim} sobresale del área de trabajo.")
    if scene_spec.why_unreachable(item.x, item.y):
        raise ValueError(f"{item.prim} está fuera del alcance del brazo.")
    for other in others:
        if abs(item.x - other.x) < (item.size[0] + other.size[0]) / 2 + .003 and abs(item.y - other.y) < (item.size[1] + other.size[1]) / 2 + .003:
            raise ValueError(f"{item.prim} está demasiado cerca de {other.prim}.")


def load(path=None):
    return build(json.loads(Path(path).read_text(encoding="utf-8")) if path else default_config())


def catalog(config, items):
    names = {entry["id"]: entry for entry in config["containers"]}
    return [{"id": item.prim, "kind": item.kind, "color": item.color, "container": item.container,
             "name": names.get(item.prim, {}).get("name", item.prim),
             "aliases": names.get(item.prim, {}).get("aliases", []),
             "size": list(item.size), "inner": list(item.inner), "height": item.height}
            for item in items]
