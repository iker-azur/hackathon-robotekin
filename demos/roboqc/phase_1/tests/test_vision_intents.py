"""Phase 2 visual-query intents and the Phase 1 fixes they depend on."""

from roboqc.commands.entities import extract_entities
from roboqc.commands.parser import classify_intent_rule, parse_text
from roboqc.commands.taxonomy import ALL_INTENTS, VISION_INTENTS
from roboqc.commands.vision_intents import classify_vision_intent


def test_vision_intents_are_not_in_trained_label_set():
    # The Phase 1 dataset covers exactly ALL_INTENTS; vision intents must stay outside it.
    for name in VISION_INTENTS:
        assert name not in ALL_INTENTS


def test_visual_queries_are_routed():
    assert classify_vision_intent("¿Cuántas piezas rojas hay?") == "COUNT_OBJECTS"
    assert classify_vision_intent("¿Hay alguna pieza defectuosa?") == "COUNT_OBJECTS"
    assert classify_vision_intent("¿Dónde está la pieza roja?") == "LOCATE_OBJECT"
    assert classify_vision_intent("Muéstrame las piezas verdes") == "SHOW_OBJECTS"
    assert classify_vision_intent("Selecciona las tres piezas azules") == "SELECT_OBJECTS"


def test_phase1_status_question_keeps_its_label():
    assert classify_vision_intent("¿Cuántas piezas quedan?") is None
    assert parse_text("¿Cuántas piezas quedan?").actions[0].intent == "REQUEST_STATUS"


def test_non_visual_commands_are_not_hijacked():
    assert classify_vision_intent("Inspecciona todos los cubos azules.") is None
    assert classify_vision_intent("Ahora las piezas amarillas son defectuosas.") is None


def test_parse_visual_query_uses_vision_intent_and_color():
    cmd = parse_text("¿Cuántas piezas rojas hay?")
    assert cmd.actions[0].intent == "COUNT_OBJECTS"
    assert cmd.actions[0].parameters.get("object_color") == "rojo"


def test_visual_query_does_not_inherit_previous_intent():
    cmd = parse_text("Inspecciona las piezas azules y ¿dónde está la roja?")
    assert [a.intent for a in cmd.actions] == ["INSPECT_OBJECT", "LOCATE_OBJECT"]


def test_confirm_requires_whole_word_si():
    assert classify_intent_rule("La pieza es visible") != "CONFIRM_ACTION"
    assert classify_intent_rule("Sí, confirma") == "CONFIRM_ACTION"


def test_plural_malas_is_defective():
    assert extract_entities("Pon las piezas malas").object_quality == "defectuoso"
    assert extract_entities("Clasifica las piezas buenas y malas").object_quality == "bueno"


def test_inspecciona_is_not_an_inspection_zone_destination():
    e = extract_entities("Inspecciona todos los cilindros azules")
    assert e.destination_role is None
    assert e.destination_type is None
