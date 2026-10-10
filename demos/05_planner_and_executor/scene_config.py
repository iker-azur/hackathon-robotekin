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

# The colours a scene may be built from.
#
# WIDENED from three to five, and the widening is the interesting part: yellow and
# purple are the two colours ``colors.py`` measured as separable from every other
# ``WIDER_DETECT`` band (yellow H 29, purple H 153, both >= 20 hue units away from
# red, green and blue), so a part of either colour can be SPAWNED and SEEN instead of
# being accepted here and then never detected.
#
# What has NOT been earned yet, recorded so it is not forgotten: neither has been
# through the 20-trial, full-scene check that red, green and blue had (117/120 kinds
# correct). Yellow is the thin one - it needs S >= 130 to clear the wood table, which
# its measured S 167-180 supports, but the table's saturation peaks at 95. White and
# black stay out because the background beyond the table measures as white and the
# robot's own links measure as black, so a white part would turn the whole frame into
# detections. Orange and pink stay out because they were measured as NOT separable at
# all (a hue gap of 5 from yellow, and of 11 from red).
AVAILABLE_COLORS = ("red", "green", "blue", "yellow", "purple")
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


def demo_config():
    """The scene for a live demo: two cubes, two trays, positions that were run.

    Balls were the problem this replaces. Measured over the recorded sessions
    (``results/demo11_metrics.json``): a command to move a red ball ended FAILED,
    0/1 verified, in 18.9 s, and the seeds that did finish needed 2 refused grasps
    each. A ball meets the pads at a point and a cube at a face, and once a part is
    tipped onto its edge it presents its diagonal - 42 mm to jaws that open 36.2 mm
    - which no force can hold. The seed that finished 2/2 with **zero** refused
    grasps was the one with two cubes of the same material and no spread.

    So this preset is that scene, written down: two 30 mm cubes, the two trays of
    matching colour, and fixed positions chosen inside the measured grasp envelope
    (``results/grip_and_release.md``: the tool-centre servo holds a part anywhere in
    the declared working area, and the servo is what makes a wider table usable).

    ``--spread 0`` and the seed in the session settings keep the layout exactly
    where this put it, because a randomised layout is the one thing a live demo
    cannot afford to discover on stage.
    """
    return {
        "version": 1,
        "objects": [
            # x 0.38 and 0.43: inside 0.27..0.52, and the pair is 50 mm apart so the
            # segmentation keeps them separate at 640x480 (two reds within 8 mm is
            # what ``dynamics.randomize`` rejects for exactly this reason).
            {"id": "cube_red", "kind": "cube", "color": "red", "count": 1,
             "size_mm": 30, "x": .38, "y": -.435},
            {"id": "cube_blue", "kind": "cube", "color": "blue", "count": 1,
             "size_mm": 30, "x": .43, "y": -.435},
        ],
        "containers": [
            # Behind the parts, at y -0.50, so the arm never reaches over a tray to
            # get to a cube. Same 70 mm outer / 56 mm cavity as the recommended
            # tray in the setup sheet, which is what gives a 30 mm part its margin.
            {"id": "tray_blue", "name": "Azul A", "aliases": [], "color": "blue",
             "size_mm": [70, 70, 30], "cavity_mm": [56, 56], "wall_mm": 7,
             "floor_mm": 6, "x": .32, "y": -.50},
            {"id": "box_red", "name": "Roja A", "aliases": [], "color": "red",
             "size_mm": [70, 70, 30], "cavity_mm": [56, 56], "wall_mm": 7,
             "floor_mm": 6, "x": .44, "y": -.50},
        ],
    }



# The named scenes the dashboard offers, in the order it should show them. Each
# entry builds its own identical JSON every time, so a preset can never be half
# edited by a previous save.
PRESETS = {
    "demo": demo_config,
    "simple": simple_config,
    "default": default_config,
}
PRESET_NAMES = tuple(PRESETS)


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


# Where the dashboard saves a scene, and where a script that was not told anything else
# looks. Both sides of the demo read the same file, so a scene saved in the panel is the
# scene a plain ``python 11_real_sorting_with_monitor.py`` runs - which is what a
# reproducible session needs. ``SAVED_SCENE`` is resolved at import time, not at call
# time, because the demo's own module lives one directory up.
SAVED_SCENE = (Path(__file__).resolve().parent / "out11" / "scene_config.json")


def load(path=None):
    """Build the scene to run: ``path`` if given, the saved one if it exists, else default.

    ``path=None`` used to mean ``default_config()``, which meant a script run with no
    ``--scene-config`` silently ignored the scene the dashboard had saved and put four
    parts - two of them balls - on the table. For a live demo that is the difference
    between the scene that was measured and one that was not.
    """
    if path is None and SAVED_SCENE.exists():
        path = SAVED_SCENE
    return build(json.loads(Path(path).read_text(encoding="utf-8")) if path else default_config())


def catalog(config, items):
    names = {entry["id"]: entry for entry in config["containers"]}
    return [{"id": item.prim, "kind": item.kind, "color": item.color, "container": item.container,
             "name": names.get(item.prim, {}).get("name", item.prim),
             "aliases": names.get(item.prim, {}).get("aliases", []),
             "size": list(item.size), "inner": list(item.inner), "height": item.height,
             # Where the builder put it. The dashboard's table map draws from these, so a scene
             # can be seen instead of read: a part that would not fit is a position, and a
             # position is the one thing a JSON form does not show.
             "x": item.x, "y": item.y}
            for item in items]


def layout():
    """The two numbers a table map needs and must not invent: the working area, and where the arm is.

    The dashboard draws the working area as a rectangle and the robot behind it. Hard-coding either
    in JavaScript would make the map a picture of a table that is not the one on the bench.
    """
    return {"work_x": list(scene_spec.WORK_X), "work_y": list(scene_spec.WORK_Y),
            "robot": list(scene_spec.ROBOT_XY), "reach": scene_spec.MAX_REACH_M}
