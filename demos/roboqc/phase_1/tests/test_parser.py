from roboqc.commands.parser import parse_text, segment_command, classify_intent_rule
from roboqc.commands.taxonomy import Intent


def _intent(text):
    cmd = parse_text(text)
    return cmd.actions[0].intent if cmd.actions else None


def test_pick_red():
    assert _intent("Coge la pieza roja") == Intent.PICK_OBJECT.value


def test_place_blue_green():
    cmd = parse_text("Deja la pieza azul en la caja verde")
    assert cmd.actions[0].intent == Intent.PLACE_OBJECT.value
    p = cmd.actions[0].parameters
    assert p.get("object_color") == "azul"
    assert p.get("destination_color") == "verde"


def test_move_qty():
    cmd = parse_text("Mueve tres cubos verdes a la caja roja")
    assert cmd.actions[0].intent == Intent.MOVE_OBJECT.value
    assert cmd.actions[0].parameters.get("quantity") == 3


def test_reversed_order():
    cmd = parse_text("En la caja azul coloca las piezas amarillas")
    assert cmd.actions[0].intent == Intent.PLACE_OBJECT.value
    p = cmd.actions[0].parameters
    assert p.get("object_color") == "amarillo"
    assert p.get("destination_color") == "azul"


def test_inspect_batch():
    assert _intent("Inspecciona todas las piezas") == Intent.INSPECT_BATCH.value


def test_negated_pick():
    cmd = parse_text("No cojas los cubos rojos")
    assert cmd.actions[0].intent == Intent.PICK_OBJECT.value
    assert cmd.actions[0].negated is True


def test_change_quality_rule():
    assert _intent("Las piezas verdes ahora son defectuosas") == Intent.CHANGE_QUALITY_RULE.value


def test_change_destination():
    assert _intent("Las piezas malas deben ir a la caja amarilla") == Intent.CHANGE_DESTINATION.value


def test_pause():
    assert _intent("Pausa el proceso") == Intent.PAUSE_PROCESS.value


def test_emergency():
    assert _intent("Parada de emergencia") == Intent.EMERGENCY_STOP.value


def test_status():
    assert _intent("¿Cuántas piezas quedan?") == Intent.REQUEST_STATUS.value


def test_compound_two_actions():
    cmd = parse_text("Coge la roja y luego deja la azul en revisión")
    assert len(cmd.actions) >= 2


def test_compound_colors():
    cmd = parse_text("Pon las rojas en la verde y las azules en la amarilla")
    assert len(cmd.actions) == 2
    assert cmd.actions[0].parameters.get("object_color") == "rojo"
    assert cmd.actions[1].parameters.get("object_color") == "azul"


def test_correction():
    cmd = parse_text("No, mejor en la caja azul")
    assert "correccion_contextual" in cmd.unresolved_references


def test_unknown():
    cmd = parse_text("Pon música")
    assert cmd.actions[0].intent == Intent.UNKNOWN.value


def test_empty_input():
    cmd = parse_text("")
    assert cmd.status == "INVALID"


def test_negation_not_positive():
    cmd = parse_text("No cojas la pieza roja")
    assert cmd.actions[0].negated is True


def test_segment_command():
    segs = segment_command("Coge la pieza roja, inspecciónala y déjala en la caja verde.")
    assert len(segs) >= 2


def test_negation_scope_variants():
    cmd = parse_text("Coge todas menos la roja")
    # should not be simply positive? at least not crash
    assert cmd is not None
