"""Interpretación de órdenes en español (texto del teclado o de Whisper)."""

import pytest

from binpick import attributes as A
from binpick import commands as K


def rules(text, **kw):
    o = K.parse(text, **kw)
    return o, [(r.selector.color, r.selector.shape, r.selector.count, r.selector.rest, r.dest) for r in o.rules]


# ---- los cuatro ejemplos del enunciado
def test_place_red_left():
    o, r = rules("Coloca las piezas rojas en la caja izquierda.")
    assert o.ok and r == [("red", None, "all", False, "izquierda")]


def test_put_blue_right():
    o, r = rules("Pon las piezas azules en la caja derecha.")
    assert o.ok and r == [("blue", None, "all", False, "derecha")]


def test_classify_two_colors_two_boxes():
    o, r = rules("Clasifica las piezas rojas a la izquierda y las verdes a la derecha.")
    assert o.ok and r == [("red", None, "all", False, "izquierda"), ("green", None, "all", False, "derecha")]
    assert o.summary() == "rojas → izquierda; verdes → derecha"


def test_pick_all_blue_without_box_asks_and_does_not_move():
    o, r = rules("Recoge todas las piezas azules.")
    assert not o.ok and r == []                    # sin caja no se adivina: el robot no se mueve
    assert any("a qué caja" in e and "izquierda" in e and "derecha" in e for e in o.errors)


def test_box_default_only_when_asked_for_and_it_warns():
    o, r = rules("Recoge todas las piezas azules.", default_dest="izquierda")
    assert o.ok and r == [("blue", None, "all", False, "izquierda")]
    assert o.rules[0].dest_defaulted and any("caja" in w for w in o.warnings)


def test_one_rule_without_box_blocks_the_whole_order():
    o = K.parse("las rojas a la izquierda y las azules")
    assert not o.ok and any("a qué caja" in e for e in o.errors)


# ---- variantes
@pytest.mark.parametrize("text,expected", [
    ("pon las rojas a la izquierda", [("red", None, "all", False, "izquierda")]),
    ("mete las azules en la caja de la derecha", [("blue", None, "all", False, "derecha")]),
    ("lleva las piezas verdes a la caja derecha", [("green", None, "all", False, "derecha")]),
    ("pon en la caja izquierda las piezas rojas", [("red", None, "all", False, "izquierda")]),
    ("las rojas y las azules a la izquierda", [("red", None, "all", False, "izquierda"), ("blue", None, "all", False, "izquierda")]),
    ("pon dos piezas rojas en la caja izquierda", [("red", None, 2, False, "izquierda")]),
    ("coloca una pieza azul a la derecha", [("blue", None, 1, False, "derecha")]),
    ("pon la pieza roja en la caja izquierda", [("red", None, 1, False, "izquierda")]),
    ("los cubos rojos a la izquierda", [("red", "cube", "all", False, "izquierda")]),
    ("pon los cilindros azules a la derecha", [("blue", "cylinder", "all", False, "derecha")]),
    ("las rojas a la izquierda y las demás a la derecha",
     [("red", None, "all", False, "izquierda"), (None, None, "all", True, "derecha")]),
    ("las rojas a la izquierda y el resto a la derecha",
     [("red", None, "all", False, "izquierda"), (None, None, "all", True, "derecha")]),
])
def test_variants(text, expected):
    o, r = rules(text)
    assert o.ok and r == expected, o.as_dict()


def test_accents_case_and_punctuation_do_not_matter():
    assert rules("¡CLASIFICA las piezas ROJAS a la IZQUIERDA, y las VERDES a la derecha!")[1] == \
        rules("clasifica las piezas rojas a la izquierda y las verdes a la derecha")[1]


@pytest.mark.parametrize("heard", ["pon las piezas rojas en la caja isquierda", "pon las piezas rojas en la caja izquirda",
                                   "pon las piezas rojas en la caja ezquierda"])
def test_asr_typos_are_repaired(heard):
    o = K.parse(heard)
    assert o.ok and o.rules[0].dest == "izquierda"


# ---- lo que NO se entiende se dice
def test_stop_words():
    for t in ("para", "detente", "alto", "cancela", "stop"):
        assert K.parse(t).intent == "stop"
    assert K.parse("para").summary() == "parar"


def test_empty_and_nonsense():
    assert not K.parse("").ok and K.parse("").errors
    o = K.parse("hola qué tal")
    assert not o.ok and o.errors


def test_unknown_color_is_reported_not_guessed():
    o = K.parse("pon las piezas naranjas a la izquierda")
    assert not o.ok and any("naranjas" in e for e in o.errors)


def test_exclusions_are_refused():
    o = K.parse("pon todas menos las rojas a la izquierda")
    assert not o.ok and any("exclusiones" in e for e in o.errors)


def test_same_pieces_to_both_boxes_is_contradictory():
    o = K.parse("pon las rojas a la izquierda y las rojas a la derecha")
    assert not o.ok and any("las dos cajas" in e for e in o.errors)


def test_box_without_pieces():
    o = K.parse("a la izquierda")
    assert not o.ok


def test_defect_attribute_is_recognised_but_not_implemented():
    o = K.parse("pon las piezas defectuosas a la izquierda")
    assert not o.ok and any("defecto" in e and "clasificador" in e for e in o.errors)


def test_registering_a_classifier_enables_the_attribute():
    attr = A.get("defect")
    old = attr.classify
    try:
        attr.classify = lambda rgb, mask: ("defective", 1.0)
        o = K.parse("pon las piezas defectuosas a la izquierda")
        assert o.ok
        s = o.rules[0].selector
        assert s.extra == {"defect": "defective"}
        assert s.matches("red", "cube", {"defect": "defective"}) and not s.matches("red", "cube", {"defect": "ok"})
    finally:
        attr.classify = old


# ---- reparto previsto y capacidad
PIECES = [("red", "cube", {}), ("red", "cylinder", {}), ("blue", "cube", {}), ("green", "cube", {}), ("blue", "cube", {})]


def test_assignments_and_rest():
    o = K.parse("las rojas a la izquierda y el resto a la derecha")
    got = {pi: ri for pi, ri in K.to_assignments(o, PIECES)}
    assert got == {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}


def test_count_limits_how_many_are_assigned():
    o = K.parse("pon dos piezas azules en la caja derecha")
    assert len(K.to_assignments(o, PIECES)) == 2


def test_capacity_warning():
    o = K.parse("las rojas y las azules a la izquierda")
    assert K.capacity_warnings(o, PIECES, capacity=3) and not K.capacity_warnings(o, PIECES, capacity=4)


def test_selector_matches():
    s = K.Selector(color="red", shape="cube")
    assert s.matches("red", "cube") and not s.matches("red", "cylinder") and not s.matches("blue", "cube")
    assert K.Selector(rest=True).matches("anything", None)


def test_describe_agreement():
    assert K.Selector(color="red", count="all").describe() == "todas las piezas rojas"
    assert K.Selector(color="blue", shape="cube", count=2).describe() == "2 cubos azules"
    assert K.Selector(color="green", count=1).describe() == "1 pieza verde"
