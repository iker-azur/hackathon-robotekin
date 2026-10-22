"""Interpretación de pedidos de kit (español e inglés)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recoverykit import kit  # noqa: E402

CASES = [
    ("Prepara un kit con una pieza roja y dos azules", {("red", None): 1, ("blue", None): 2}),
    ("Prepare a kit containing one red piece and two blue pieces", {("red", None): 1, ("blue", None): 2}),
    ("Place one red cube and two blue cubes inside the box", {("red", "cube"): 1, ("blue", "cube"): 2}),
    ("un cubo rojo y dos azules", {("red", "cube"): 1, ("blue", "cube"): 2}),
    ("mete 2 rojas, 1 verde", {("red", None): 2, ("green", None): 1}),
    ("tres piezas azules", {("blue", None): 3}),
    ("un cilindro verde y un cubo amarillo", {("green", "cylinder"): 1, ("yellow", "cube"): 1}),
    ("una roja", {("red", None): 1}),
    ("todas las piezas rojas", {("red", None): "all"}),
    ("Quiero un kit: 1 azul y 1 azul", {("blue", None): 2}),
]


@pytest.mark.parametrize("text,goal", CASES)
def test_goals(text, goal):
    req = kit.parse(text)
    assert req.errors == []
    assert req.goal == goal
    assert req.ok


@pytest.mark.parametrize("text,fragment", [
    ("dos naranjas", "naranja"),
    ("todas menos las rojas", "exclusiones"),
    ("no rojas", "exclusiones"),
    ("dos cubos", "color"),
    ("hola", "ninguna pieza"),
    ("", "nada"),
])
def test_errors(text, fragment):
    req = kit.parse(text)
    assert not req.ok
    assert any(fragment in e for e in req.errors), req.errors


def test_stop():
    assert kit.parse("para").stop
    assert kit.parse("Stop").stop
    assert not kit.parse("una roja para la caja").stop


def test_describe_agreement():
    assert kit.describe({("red", None): 1, ("blue", None): 2}) == "1 pieza roja y 2 piezas azules"
    assert kit.describe({("red", "cube"): 1, ("blue", "cube"): 2}) == "1 cubo rojo y 2 cubos azules"
    assert kit.describe({("green", "cylinder"): "all"}) == "todos los cilindros verdes"
