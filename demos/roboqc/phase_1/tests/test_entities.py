from roboqc.commands.entities import (
    normalize_color, parse_quantity, extract_entities, entities_to_parameters,
    COLOR_FORMS,
)


def test_color_morphology():
    for form in ["rojo", "roja", "rojos", "rojas", "amarillo", "amarilla", "amarillos", "amarillas",
                 "blanco", "blanca", "blancos", "blancas"]:
        assert normalize_color(form) is not None


def test_quantity_extraction():
    assert parse_quantity("coge tres piezas") == 3
    assert parse_quantity("coge una pieza") == 1
    assert parse_quantity("coge todas las piezas") == "ALL"
    assert parse_quantity("coge algunas piezas") == "SOME"
    assert parse_quantity("coge las restantes") == "REMAINING"
    assert parse_quantity("coge piezas") == "UNSPECIFIED"


def test_color_attribution_basic():
    e = extract_entities("Deja la pieza roja en la caja verde.")
    assert e.object_color == "rojo"
    assert e.destination_color == "verde"


def test_color_attribution_reversed_order():
    e = extract_entities("En la caja azul coloca las piezas amarillas.")
    assert e.object_color == "amarillo"
    assert e.destination_color == "azul"


def test_same_color_object_and_destination():
    e = extract_entities("Pon las piezas verdes en la caja verde.")
    assert e.object_color == "verde"
    assert e.destination_color == "verde"


def test_two_colors_roles():
    # single segment; color roles
    e = extract_entities("Pon las rojas en la verde y las azules en la amarilla.")
    # object color should be rojo (first) at least
    assert e.object_color in ("rojo", "azul")
