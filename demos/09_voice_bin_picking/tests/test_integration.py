"""De punta a punta sobre un simulador CINEMÁTICO (sin física): percepción → selección → planificación →
colisión → guion/política → colocación → verificación con cámara → informe.

Valida la integración de los módulos. NO valida agarre por fricción, contactos ni ACT entrenada: eso exige
Isaac (``tools/eval_sorting.py``). La «política» de estas pruebas es una reproducción del experto, no aprendida.
"""

import numpy as np
import pytest

from binpick import commands as K
from binpick import config as C
from binpick import orchestrator as O
from binpick import policy as POL
from binpick import scene_gen as G
from binpick import selection as SEL
from binpick.robot_port import SimRobot
from binpick.telemetry import BinTelemetry
from fake_cell import FakeBinCell
from test_pick_runner import expert_actions

CFG = C.BinPickCfg()


def build(seed, n=4, mode="separadas", policy=None, order_text=None, strict=True):
    """Primera escena (a partir de ``seed``) en la que todas las piezas de la orden se ven agarrables."""
    order = K.parse(order_text) if order_text else None
    for s in range(seed, seed + 400):
        fb = FakeBinCell(CFG)
        tele = BinTelemetry()
        robot = SimRobot(fb, tele, "hsv", policy=policy, realtime=0.0)
        robot.new_scene(G.sample_scene(CFG, s, n_pieces=n, mode=mode))
        if order is None:
            return fb, tele, robot, s
        obs = robot.observe("inicio")
        sel = SEL.rank(obs.instances, order, fb.scene)
        if sel.pending >= 2 and len(sel.usable) == sel.pending and not sel.blocked:
            return fb, tele, robot, s
    raise AssertionError("no hay escena limpia")


def run(robot, fb, text, policy_mode="guion", **kw):
    task = O.SortingTask(robot, fb.scene, sink=robot.tele, policy_mode=policy_mode, stop=lambda: robot.tele.stop_requested, **kw)
    return task.run(K.parse(text))


def regions_by_color(fb):
    return fb.audit()["regions"]


def test_scripted_expert_sorts_the_pieces_it_was_asked_to_sort():
    text = "las rojas a la izquierda y las demás a la derecha"
    fb, tele, robot, seed = build(0, n=4, order_text=text)
    rep = run(robot, fb, text)
    assert rep.status == "completado", (rep.status, rep.notes, rep.failures)
    reg = regions_by_color(fb)
    assert set(reg.get("izquierda", {})) <= {"red"}                       # sólo rojas a la izquierda
    assert "red" not in reg.get("derecha", {})                            # ninguna roja a la derecha
    assert reg.get("origen", {}) == {} and reg.get("mesa", {}) == {}      # el contenedor quedó vacío, nada fuera
    moved = sum(sum(v.values()) for k, v in reg.items() if k in ("izquierda", "derecha"))
    assert rep.moved == moved == len(fb.cell.truths)
    assert rep.picks["guion"] == rep.picks_ok["guion"] >= moved and rep.picks["act"] == 0   # todo con el guion y así se declara
    assert tele.snapshot()["report"]["status"] == "completado"


def test_perception_sees_what_is_in_the_scene_and_boxes_are_empty_at_start():
    fb, tele, robot, seed = build(1, n=4)
    obs = robot.observe("inicio")
    assert len(obs.instances) == len(fb.cell.truths)
    assert {i.region for i in obs.instances} == {"origen"}
    assert all(i.track for i in obs.instances)
    snap = tele.snapshot()
    assert snap["boxes"]["origen"] == len(obs.instances) and snap["instances"] and snap["frames"]["cam"] > 0


def test_a_color_that_is_not_there_is_reported_and_nothing_moves():
    for s in range(100):
        fb = FakeBinCell(CFG)
        robot = SimRobot(fb, BinTelemetry(), "hsv", realtime=0.0)
        robot.new_scene(G.sample_scene(CFG, s, n_pieces=3, mode="separadas"))
        if "blue" not in {t.color for t in fb.cell.truths.values()}:
            rep = run(robot, fb, "las azules a la derecha")
            assert rep.status == "nada que hacer" and any("No veo" in n for n in rep.notes)
            assert fb.audit()["regions"].get("derecha", {}) == {}
            return
    pytest.fail("no hay escena sin azules")


class ReplayPolicy(POL.CallablePolicy):
    """Reproduce el plan del experto como si fuera una política «act». NO es aprendida: prueba la tubería ACT."""

    def __init__(self, robot):
        self.robot, self.acts, self.i = robot, [], 0
        super().__init__(self._next, "reproducción del experto (prueba de tubería)")

    def reset(self):
        plan, _slot = self.robot._plans[self.robot._target_track]
        self.acts, self.i = expert_actions(plan), 0

    def _next(self, obs):
        a = self.acts[min(self.i, len(self.acts) - 1)]
        self.i += 1
        return a


def test_act_pipeline_plumbing_with_a_replayed_expert():
    text = "las rojas a la izquierda y las demás a la derecha"
    fb, tele, robot, seed = build(0, n=3, order_text=text)
    robot.policy = ReplayPolicy(robot)
    rep = run(robot, fb, text, policy_mode="act")
    assert rep.status == "completado", (rep.status, rep.notes, rep.failures)
    assert rep.picks["act"] == rep.picks_ok["act"] >= 2 and rep.picks["guion"] == 0
    assert robot.last_pick_stats["pasos"] > 20                            # el limitador contó los pasos de la política
    reg = regions_by_color(fb)
    assert reg.get("origen", {}) == {} and sum(map(sum, [v.values() for v in reg.values()])) == len(fb.cell.truths)


def test_act_mode_without_a_learned_policy_fails_honestly():
    text = "las rojas a la izquierda y las demás a la derecha"
    fb, tele, robot, seed = build(0, n=3, order_text=text)
    rep = run(robot, fb, text, policy_mode="act", max_cycles=6)               # robot.policy es el guion: no hay ACT
    assert rep.moved == 0 and rep.status in ("fallido", "incompleto") and rep.picks["act"] >= 1
    assert regions_by_color(fb).get("izquierda", {}) == {}


def test_stop_request_ends_the_run_and_is_reported():
    text = "las rojas a la izquierda y las demás a la derecha"
    fb, tele, robot, seed = build(0, n=4, order_text=text)
    calls = {"n": 0}
    orig = robot.ctrl.monitor

    def monitor(info):
        calls["n"] += 1
        if calls["n"] > 30:
            robot.tele.request_stop()
        return orig(info)

    robot.ctrl.monitor = monitor
    rep = run(robot, fb, text)
    assert rep.status == "detenido"
