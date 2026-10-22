from roboqc.industrial.voice import Dialogue, understand, handles


def test_quality_multiple_rules():
    response = understand("Pon las piezas buenas en la caja verde, las defectuosas tipo uno en la roja y las tipo dos en la azul.")
    assert response["status"] == "valid"
    assert response["command"]["rules"] == [
        {"category": "good", "destination": "green"},
        {"category": "defect_1", "destination": "red"},
        {"category": "defect_2", "destination": "blue"}]


def test_color_multiple_rules_and_grouped_categories():
    response = understand("Clasifica las piezas rojas en la caja azul, las verdes en la roja y las azules en la verde.")
    assert response["status"] == "valid"
    assert response["command"]["mode"] == "color"
    assert response["command"]["rules"][0] == {"category": "red", "destination": "blue"}
    assert len(understand("Pon las rojas y verdes en la azul")["command"]["rules"]) == 2


def test_contradiction_and_unsupported():
    assert understand("Pon las rojas en la azul y las rojas en la verde")["status"] == "contradictory"
    assert understand("Pon las buenas en la amarilla")["status"] == "unsupported"
    assert understand("Suelda las tuercas")["status"] == "unsupported"
    assert understand("No pares la clasificacion")["status"] == "unsupported"
    assert understand("Pon las rojas excepto las buenas en la azul")["status"] == "needs_clarification"


def test_explicit_destination_even_with_active_rules():
    dialogue = Dialogue()
    assert dialogue.submit("Pon las defectuosas tipo uno y tipo dos en la roja")["status"] == "valid"
    assert dialogue.submit("Recoge solamente las piezas defectuosas")["status"] == "needs_clarification"
    assert dialogue.pending is not None
    assert dialogue.submit("la azul")["status"] == "valid"
    assert all(r["destination"] == "blue" for r in dialogue.active["rules"])


def test_partial_answers_wait_and_controls():
    dialogue = Dialogue()
    assert dialogue.submit("Continua clasificando")["status"] == "needs_clarification"
    dialogue.submit("Pon las rojas y las verdes")
    assert dialogue.submit("las rojas en la azul")["status"] == "needs_clarification"
    assert dialogue.submit("las verdes en la roja")["status"] == "valid"
    assert dialogue.running
    dialogue.submit("Deten la clasificacion")
    assert not dialogue.running
    dialogue.submit("Continua clasificando")
    assert dialogue.running
    dialogue.submit("Cancela la orden")
    assert dialogue.pending is None and not dialogue.running


def test_polite_prefix_and_legacy_quality_rule_are_not_control_conflicts():
    assert understand("Para esta tanda, pon las buenas en la verde")["status"] == "valid"
    assert not handles("Ahora las piezas azules son defectuosas")
    assert not handles("Inspecciona las piezas defectuosas")
    assert handles("Recoge solamente las defectuosas")
    assert handles("Para el proceso") and handles("Pausa")


def test_learned_verb_cannot_supply_a_missing_destination():
    class Model:
        def predict(self,text):
            return {"intent":"sort","mode":"quality","confidence":.99}
    assert understand("Destina las buenas en la verde",Model())["status"] == "valid"
    assert understand("Destina las buenas",Model())["status"] == "needs_clarification"
    assert understand("Pon las buenas en la roja y taladra las tuercas",Model())["status"] == "unsupported"


def test_unknown_answer_and_stopped_request_do_not_resume():
    dialogue = Dialogue()
    dialogue.submit("Pon las rojas en la verde")
    dialogue.submit("Recoge solamente las defectuosas")
    assert not dialogue.running
    assert dialogue.submit("no se")["status"] == "needs_clarification"
    assert dialogue.submit("Continua clasificando")["status"] == "needs_clarification"
    assert not dialogue.running
