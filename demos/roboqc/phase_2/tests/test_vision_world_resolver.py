"""World state updates and command resolution against a known scene (no camera, no Qt)."""

import pytest

from roboqc.commands.pipeline import understand_text
from roboqc.vision.config import DEFAULTS, load_config
from roboqc.vision.detector import Detection
from roboqc.vision.inspection import build_inspection_report, format_report
from roboqc.vision.quality import QualityRuleSet
from roboqc.vision.resolver import CLARIFY, INFO, NOT_FOUND, OK, UNSUPPORTED, resolve_command
from roboqc.vision.tracker import Tracker
from roboqc.vision.world_state import WorldState


def _det(color, x, y, obj_type="cubo", touches=False, solidity=0.95):
    return Detection(
        color=color, color_confidence=0.9, shape="cuadrado" if obj_type == "cubo" else "rectangulo",
        object_type=obj_type, shape_confidence=0.9, bbox=(int(x) - 25, int(y) - 25, 50, 50),
        centroid=(float(x), float(y)), area_px=2500.0, orientation_deg=0.0, solidity=solidity,
        vertices=4, touches_border=touches,
    )


def _world(frames):
    cfg = load_config("__no_such_file__.json")
    world = WorldState(Tracker(cfg["tracking"]), QualityRuleSet(cfg["quality"]))
    for i, dets in enumerate(frames):
        world.update(dets, i * 0.1)
    return world


def _scene():
    # Detection order fixes the IDs: A=OBJ_001 red, B=OBJ_002 red, C=OBJ_003 blue cube, D=OBJ_004 green prism.
    dets = [_det("rojo", 100, 100), _det("rojo", 300, 100), _det("azul", 500, 100), _det("verde", 100, 300, "prisma")]
    return _world([dets, dets])


def _resolve(text, world):
    return resolve_command(understand_text(text, source="text"), world)


def test_count_red_pieces():
    result = _resolve("¿Cuántas piezas rojas hay?", _scene())
    assert result.status == INFO
    assert "Hay 2 piezas rojas visibles" in result.text


def test_inspect_all_blue_cubes_reports_quality_and_highlights_matches():
    world = _scene()
    result = _resolve("Inspecciona todos los cubos azules.", world)
    assert result.status == OK
    assert "OBJ_003" in result.text
    assert "OBJ_001" not in result.text
    assert world.selected_ids() == ["OBJ_003"]


def test_ambiguous_singular_reference_requests_clarification():
    result = _resolve("¿Dónde está la pieza roja?", _scene())
    assert result.status == CLARIFY
    assert "OBJ_001" in result.text and "OBJ_002" in result.text


def test_unique_reference_is_located_with_pixel_position():
    result = _resolve("¿Dónde está el cubo azul?", _scene())
    assert result.status == OK
    assert "OBJ_003" in result.text
    assert "en px (500, 100)" in result.text


def test_missing_object_is_reported_not_invented():
    result = _resolve("Muéstrame las piezas amarillas.", _scene())
    assert result.status == NOT_FOUND
    assert "No veo" in result.text


def test_selection_with_quantity_larger_than_available():
    world = _scene()
    result = _resolve("Selecciona las tres piezas azules.", world)
    assert result.status == OK
    assert result.selected_ids == ["OBJ_003"]
    assert "Solo hay 1" in result.text
    assert world.selected_ids() == ["OBJ_003"]


def test_quality_rule_changes_classification_without_retraining():
    world = _scene()
    result = _resolve("Ahora las piezas azules son defectuosas.", world)
    assert result.status == OK
    assert result.rule is not None and result.rule.color == "azul"
    assert "No se ha ejecutado ninguna acción física" in result.text
    blue = world.snapshot().by_id("OBJ_003")
    assert blue.quality == "DEFECTIVE"
    assert any("regla activa" in r for r in blue.quality_reasons)
    assert world.snapshot().by_id("OBJ_001").quality == "GOOD"


def test_defective_query_after_rule():
    world = _scene()
    _resolve("Ahora las piezas azules son defectuosas.", world)
    result = _resolve("¿Hay alguna pieza defectuosa?", world)
    assert result.status == INFO
    assert "OBJ_003" in result.text


def test_classify_good_and_bad_answers_once_with_both_groups():
    result = _resolve("Clasifica las piezas buenas y malas.", _scene())
    assert result.status == OK
    assert result.text.count("Buenos (") == 1
    assert "Defectuosos (" in result.text


def test_robot_commands_are_refused_without_claiming_action():
    result = _resolve("Coge la pieza roja.", _scene())
    assert result.status == UNSUPPORTED
    assert "no controla el robot" in result.text
    assert "No se ha ejecutado ninguna acción física" in result.text


def test_unknown_instruction_is_not_guessed():
    result = _resolve("Pon música.", _scene())
    assert result.status == UNSUPPORTED
    assert "No he entendido" in result.text


def test_partial_object_is_uncertain_and_reported_as_partial():
    world = _world([[_det("rojo", 10, 100, touches=True)], [_det("rojo", 10, 100, touches=True)]])
    obj = world.snapshot().objects[0]
    assert obj.detection_status == "PARTIAL"
    assert obj.quality == "UNCERTAIN"
    assert world.snapshot().counts()["UNCERTAIN"] == 1


def test_world_counts_and_inspection_report():
    world = _scene()
    snap = world.snapshot()
    assert snap.counts() == {"visible": 4, "tracked": 4, "GOOD": 4, "DEFECTIVE": 0, "UNCERTAIN": 0}
    report = build_inspection_report(snap)
    assert report["summary"]["visible"] == 4
    assert len(report["objects"]) == 4
    assert "OBJ_001" in format_report(report)


def test_snapshot_is_immutable_copy():
    world = _scene()
    snap = world.snapshot()
    world.update([], 1.0)
    assert len(snap.objects) == 4  # earlier snapshot is unaffected by later frames


def test_world_position_only_with_calibration():
    world = _scene()
    assert world.snapshot().objects[0].world_position is None
