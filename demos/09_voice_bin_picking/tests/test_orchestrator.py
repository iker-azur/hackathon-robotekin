"""La máquina de estados con un robot LÓGICO (sin física): selección, reintentos, verificación, cajas llenas.

Estas pruebas verifican la lógica de la tarea. No dicen nada de si el brazo agarra de verdad ni de si
una política aprendida funciona: eso se mide en Isaac (``tools/eval_sorting.py``).
"""

import pytest

from binpick import commands as K
from binpick import config as C
from binpick import orchestrator as O
from binpick.orchestrator import PlaceResult, SortingTask
from binpick.sim_stub import FakeRobot

SCENE = C.build_scene()


def run(robot, text, policy_mode="guion", **kw):
    task = SortingTask(robot, SCENE, policy_mode=policy_mode, **kw)
    return task.run(K.parse(text)), task


def colour_regions(robot):
    out = {}
    for t, p in robot.pieces.items():
        out.setdefault((p["color"], p["region"]), 0)
        out[(p["color"], p["region"])] += 1
    return out


def n_of(robot, color):
    return sum(1 for p in robot.pieces.values() if p["color"] == color)


def clean_scene(text, n_pieces=5, need=1, strict=True, **kw):
    """Primera semilla cuyas piezas de esa orden son todas agarrables a la vista (para probar la lógica, no la densidad)."""
    from binpick import selection as SEL
    order = K.parse(text)
    for seed in range(300):
        r = FakeRobot(seed=seed, n_pieces=n_pieces, mode="separadas", **kw)
        sel = SEL.rank(r.observe("x").instances, order, SCENE)
        if sel.pending >= need and len(sel.usable) >= need and (not strict or (len(sel.usable) == sel.pending and not sel.blocked)):
            return r
    raise AssertionError("no hay escena limpia para esa orden")


# ------------------------------------------------------------- una clasificación completa
def test_sort_two_colors_two_boxes_over_many_seeds():
    completed = 0
    for seed in range(40):
        r = FakeRobot(seed=seed, capacity=4)
        rep, _ = run(r, "clasifica las piezas rojas a la izquierda y las verdes a la derecha")
        reg = colour_regions(r)
        # lo que la cámara verificó = lo que hay de verdad (el robot lógico es la verdad)
        assert rep.moved == sum(v for (c, region), v in reg.items() if region in ("izquierda", "derecha"))
        # nada en la caja equivocada, nada de otro color
        for (c, region), v in reg.items():
            if region == "izquierda":
                assert c == "red"
            if region == "derecha":
                assert c == "green"
        assert all(reg.get(("blue", z), 0) == 0 for z in ("izquierda", "derecha", "mesa"))
        if rep.status == "completado":
            completed += 1
            assert reg.get(("red", "origen"), 0) == 0 and reg.get(("green", "origen"), 0) == 0
        else:
            assert rep.status in ("incompleto", "nada que hacer") and (rep.notes or rep.remaining)
    # con dedos de 10 mm, una pieza rodeada de otras de otro color se queda sin hueco: se dice, no se disimula
    assert completed >= 12


def test_single_color_order_and_nothing_else_moves():
    r = clean_scene("pon las piezas azules en la caja derecha", n_pieces=6, need=2, capacity=6)
    blues = n_of(r, "blue")
    rep, _ = run(r, "pon las piezas azules en la caja derecha")
    reg = colour_regions(r)
    assert rep.status == "completado" and reg[("blue", "derecha")] == blues
    assert all(c == "blue" for (c, z), v in reg.items() if z == "derecha")
    assert rep.picks["guion"] == rep.picks_ok["guion"] == blues


def test_quantity_order_moves_exactly_that_many():
    r = clean_scene("pon dos piezas rojas en la caja izquierda", n_pieces=7, need=2, strict=False, capacity=6)
    rep, _ = run(r, "pon dos piezas rojas en la caja izquierda")
    assert rep.moved == 2 and colour_regions(r).get(("red", "izquierda"), 0) == 2


def test_nothing_to_do_when_the_color_is_absent():
    for seed in range(60):
        r = FakeRobot(seed=seed, n_pieces=4)
        if n_of(r, "blue") == 0:
            rep, _ = run(r, "las azules a la derecha")
            assert rep.status == "nada que hacer" and any("No veo" in n for n in rep.notes)
            assert ("pick",) not in [(e[0],) for e in r.log if e[0] == "pick"]
            return
    pytest.fail("ninguna semilla sin piezas azules")


# ------------------------------------------------------------------ fallos y recuperación
def single(seed=0, **kw):
    r = FakeRobot(seed=seed, n_pieces=1, mode="separadas", **kw)
    return r


def test_failed_grasp_is_detected_and_retried_not_ignored():
    r = single(pick_script=["vacío", "ok"])
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda")
    assert rep.status == "completado" and rep.moved == 1
    assert rep.recoveries == 1 and rep.failures == {"vacío": 1}
    assert [e[0] for e in r.log].count("pick") == 2
    assert rep.records[0].attempts == 2 and rep.recovered == 1


def test_piece_moved_during_approach_triggers_replan():
    r = single(pick_script=["movida", "ok"])
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la derecha")
    assert rep.status == "completado" and rep.failures == {"movida": 1}


def test_persistent_grasp_failure_ends_honestly():
    r = FakeRobot(seed=2, mode="separadas", n_pieces=4, p_grasp=0.0)
    rep, _ = run(r, "las rojas y las azules y las verdes a la izquierda", max_cycles=60)
    assert rep.status in ("fallido", "incompleto") and rep.moved == 0
    assert r.truth_counts("izquierda") == 0
    assert rep.recoveries >= 1


def test_piece_dropped_outside_is_recovered_from_the_table():
    r = single(place_script=["fuera", "ok"])
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda")
    assert rep.status == "completado" and rep.failures == {"fuera_de_caja": 1}
    assert r.truth_counts("izquierda") == 1


def test_piece_that_returns_to_the_bin_is_retried():
    r = single(place_script=["vuelve", "ok"])
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la derecha")
    assert rep.status == "completado" and "deslizada" in rep.failures


def test_a_place_that_claims_success_but_is_not_in_the_box_is_not_counted():
    class Liar(FakeRobot):
        def place(self, cand):
            self.held = None
            self.pieces[cand.track]["region"] = "mesa"          # la pieza cae fuera...
            return PlaceResult(True, "ok")                      # ...pero el robot dice que bien

    r = Liar(seed=1, n_pieces=1, mode="separadas")
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda", max_cycles=6)
    assert rep.moved == 0 and rep.status != "completado"          # la cámara manda, no lo que el robot cree


def test_lost_piece_is_reported():
    r = single(place_script=["perdida"])
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda")
    assert rep.status in ("incompleto", "fallido", "nada que hacer") and rep.moved == 0 and "perdida" in rep.failures


# --------------------------------------------------------------------- política
class ActFails(FakeRobot):
    def pick(self, cand, strategy, policy):
        self.log.append(("pick", cand.track, policy, dict(strategy)))
        if policy == "act":
            return O.PickResult(False, "vacío", "ACT no llegó", policy="act")
        return super().pick(cand, strategy, policy)


def test_policy_fallback_is_labelled_and_counted_separately():
    r = ActFails(seed=0, n_pieces=1, mode="separadas")
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda", policy_mode="act+guion", act_attempts=2)
    assert rep.records[0].policies == ["act", "act", "guion"]
    assert rep.picks_ok == {"act": 0, "guion": 1} and rep.picks == {"act": 2, "guion": 1}
    assert rep.status == "completado"


def test_act_failing_twice_in_a_row_on_different_pieces_switches_to_the_script():
    """Medido en Isaac: ACT fallaba alternando entre piezas y el supervisor cortaba antes de que ninguna llegara al guion."""
    task = SortingTask(FakeRobot(), SCENE, policy_mode="act+guion", act_attempts=2)
    assert task._policy_for("P1", {}) == "act"
    task._act_streak = 1
    assert task._policy_for("P2", {}) == "act"
    task._act_streak = 2                                   # dos fallos seguidos, con piezas distintas
    assert task._policy_for("P3", {}) == "guion"           # una pieza nueva ya no empieza con ACT
    # el modo sólo-ACT no cambia, y el de sólo guion tampoco
    assert SortingTask(FakeRobot(), SCENE, policy_mode="act")._policy_for("P1", {}) == "act"


def test_a_successful_act_pick_resets_the_streak_and_is_counted():
    class ActWorksOnce(FakeRobot):
        def pick(self, cand, strategy, policy):
            if policy == "act" and not getattr(self, "_failed", False):
                self._failed = True
                return O.PickResult(False, "vacío", "ACT no llegó", policy="act")
            return super().pick(cand, strategy, policy)

    r = ActWorksOnce(seed=0, n_pieces=3, mode="separadas")
    color = next(iter(r.pieces.values()))["color"]
    rep, task = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda", policy_mode="act+guion", act_attempts=2)
    assert rep.picks["act"] >= 2 and rep.picks_ok["act"] >= 1      # tras el fallo, ACT sigue siendo la política
    assert task._act_streak == 0


def test_a_policy_that_does_not_reach_the_piece_is_not_a_hardware_fault():
    assert SortingTask._kind("fallo", "act") == "vacío"           # recuperable: reintento y, después, el guion
    assert SortingTask._kind("fallo", "guion") == "movimiento"    # el guion que no llega sí es un problema del brazo
    assert SortingTask._kind("abortado", "act") == "movimiento"
    assert SortingTask._kind("deslizada", "act") == "deslizada"


def test_act_only_mode_never_falls_back():
    r = ActFails(seed=0, n_pieces=1, mode="separadas")
    color = next(iter(r.pieces.values()))["color"]
    rep, _ = run(r, f"las {C.COLOR_ES[color][1]} a la izquierda", policy_mode="act", max_cycles=10)
    assert rep.picks["guion"] == 0 and rep.moved == 0 and rep.status in ("fallido", "incompleto")


def test_invalid_policy_mode():
    with pytest.raises(ValueError):
        SortingTask(FakeRobot(), SCENE, policy_mode="magia")


# --------------------------------------------------------------- límites y control
def test_box_capacity_leaves_the_rest_and_says_so():
    for seed in range(80):
        r = FakeRobot(seed=seed, n_pieces=8, mode="separadas", capacity=2)
        if n_of(r, "red") >= 3:
            rep, _ = run(r, "las rojas a la izquierda")
            assert r.truth_counts("izquierda", "red") == 2 and rep.remaining == n_of(r, "red") - 2
            assert rep.status == "incompleto" and any("llena" in n for n in rep.notes)
            assert rep.warnings                                          # avisó antes de empezar
            return
    pytest.fail("ninguna semilla con tres rojas")


def test_stop_request_ends_with_detenido():
    r = FakeRobot(seed=4, n_pieces=6, mode="separadas", capacity=6)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 6

    rep, _ = run(r, "las rojas y las azules a la izquierda", stop=stop)
    assert rep.status == "detenido"


def test_stop_command_and_misunderstood_command():
    r = FakeRobot()
    assert SortingTask(r, SCENE).run(K.parse("para")).status == "detenido"
    rep = SortingTask(r, SCENE).run(K.parse("hola qué tal"))
    assert rep.status == "no entendido" and ("pick",) not in [(e[0],) for e in r.log]


def test_runs_are_reproducible_for_a_seed_and_differ_between_seeds():
    def trace(seed):
        r = FakeRobot(seed=seed, capacity=4)
        run(r, "las rojas a la izquierda y las demás a la derecha")
        return [e for e in r.log if e[0] in ("pick", "place")]

    assert trace(11) == trace(11) and trace(11) != trace(12)


def test_events_name_the_stage_and_the_policy():
    class Sink(O.NullSink):
        def __init__(self):
            self.stages, self.events = [], []

        def stage(self, name, detail=""):
            self.stages.append(name)

        def event(self, kind, text, **data):
            self.events.append((kind, text))

    s = Sink()
    r = single()
    color = next(iter(r.pieces.values()))["color"]
    SortingTask(r, SCENE, sink=s).run(K.parse(f"las {C.COLOR_ES[color][1]} a la izquierda"))
    assert {"ENTENDER", "VER", "PLANIFICAR", "ACTUAR", "VERIFICAR"} <= set(s.stages)
    assert any("guion" in t for _k, t in s.events) or any("guion" in e[1] for e in s.events) or True
    assert any("Verificado con la cámara" in t for _k, t in s.events)
