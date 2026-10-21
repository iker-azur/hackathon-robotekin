"""Modelo del mundo, planificador y supervisor con observaciones construidas a mano."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recoverykit import kit, planner, supervisor  # noqa: E402
from recoverykit.perception import BoxDetection, Detection, Observation  # noqa: E402
from recoverykit.world_model import WorldModel  # noqa: E402

TABLE = 0.76
SLOTS = [(0.115, 0.102), (0.166, 0.102), (0.115, 0.152), (0.166, 0.152)]
BOX = BoxDetection("caja", np.array([0.141, 0.127, TABLE + 0.004]), 0.0, (0.11, 0.11), (0.102, 0.102),
                   TABLE + 0.045, TABLE + 0.004)


def det(color, x, y, in_box=False, shape="cube", h=0.025):
    z = (TABLE + 0.004 if in_box else TABLE) + h / 2
    return Detection(color, shape, np.array([x, y, z]), 0.0, 0.025, h, 1500, (0, 0), in_box=in_box,
                     box_id="caja" if in_box else None)


def obs(*dets):
    return Observation(pieces=list(dets), boxes=[BOX])


def reach_all(_track):
    return None


def test_tracking_and_events():
    w = WorldModel()
    ev = w.update(obs(det("red", 0.17, -0.07), det("blue", 0.19, -0.13)), 0.0)
    assert {e.kind for e in ev} == {"nueva"}
    red = next(t for t in w.tracks.values() if t.color == "red")
    # ruido pequeño: misma pieza, sin evento
    assert w.update(obs(det("red", 0.172, -0.07), det("blue", 0.19, -0.13)), 1.0) == []
    # empujón de 4 cm: "movida", misma identidad
    ev = w.update(obs(det("red", 0.20, -0.04), det("blue", 0.19, -0.13)), 2.0)
    assert [e.kind for e in ev] == ["movida"] and ev[0].track == red.id
    assert abs(ev[0].displacement_m - 0.0424) < 0.002
    # la coge el robot y la deja en la caja
    w.grasped(red.id)
    assert w.update(obs(det("blue", 0.19, -0.13)), 3.0) == []
    w.released()
    ev = w.update(obs(det("red", 0.115, 0.102, in_box=True), det("blue", 0.19, -0.13)), 4.0)
    assert [e.kind for e in ev] == ["en_caja"]
    # se cae fuera
    ev = w.update(obs(det("red", 0.24, 0.05), det("blue", 0.19, -0.13)), 5.0)
    assert [e.kind for e in ev] == ["fuera_de_caja"]


def test_lost_only_when_visible():
    w = WorldModel()
    w.update(obs(det("red", 0.17, -0.07)), 0.0)
    for t in range(5):
        assert w.update(obs(), t + 1.0, visible=lambda p: False) == []
    kinds = []
    for t in range(3):
        kinds += [e.kind for e in w.update(obs(), t + 10.0, visible=lambda p: True)]
    assert kinds == ["perdida"]


def test_plan_counts_box_contents_and_orders_tasks():
    w = WorldModel()
    w.update(obs(det("red", 0.17, -0.07), det("red", 0.16, -0.14), det("blue", 0.19, -0.13),
                 det("blue", 0.20, -0.02), det("blue", 0.115, 0.102, in_box=True)), 0.0)
    goal = kit.parse("una pieza roja y dos azules").goal
    kp = planner.plan(goal, w, SLOTS, reach_all, set())
    states = {t.label: t.state for t in kp.tasks}
    assert states["pieza azul 1/2"] == "completada"
    pending = [t for t in kp.tasks if t.state == "pendiente"]
    assert len(pending) == 2
    assert all(t.slot not in (None, 0) for t in pending)       # el hueco 0 ya está ocupado
    assert len({t.track for t in pending}) == 2
    assert not kp.complete


def test_plan_complete_and_extras():
    w = WorldModel()
    w.update(obs(det("red", 0.115, 0.102, in_box=True), det("green", 0.166, 0.152, in_box=True, shape="cylinder")), 0)
    kp = planner.plan(kit.parse("una roja").goal, w, SLOTS, reach_all, set())
    assert kp.covered and kp.extras
    assert not kp.complete
    assert any(t.kind == "retirar" for t in kp.tasks)


def test_plan_blocks_when_not_enough_or_unreachable():
    w = WorldModel()
    w.update(obs(det("red", 0.17, -0.07)), 0.0)
    kp = planner.plan(kit.parse("dos rojas").goal, w, SLOTS, reach_all, set())
    assert [t.state for t in kp.tasks].count("bloqueada") == 1
    kp = planner.plan(kit.parse("una roja").goal, w, SLOTS, lambda t: "demasiado lejos", set())
    assert kp.tasks[0].state == "bloqueada" and "demasiado lejos" in kp.tasks[0].note


def test_blacklisted_piece_is_replaced():
    w = WorldModel()
    w.update(obs(det("blue", 0.19, -0.13), det("blue", 0.20, -0.02)), 0.0)
    ids = list(w.tracks)
    kp = planner.plan(kit.parse("una azul").goal, w, SLOTS, reach_all, {ids[0]})
    assert kp.tasks[0].track == ids[1]


def test_supervisor_escalates_then_switches():
    s = supervisor.Supervisor()
    f = supervisor.Failure("vacío", "t", "azul_1")
    actions = [s.decide(f, alternatives=1) for _ in range(4)]
    assert [a.action for a in actions] == ["reintentar"] * 3 + ["otra_pieza"]
    assert actions[0].strategy == {}
    assert "yaw_offset" in actions[1].strategy
    assert actions[2].strategy.get("lift_speed") == 0.04
    assert actions[3].blacklist == "azul_1"


def test_supervisor_moved_and_misplaced():
    s = supervisor.Supervisor()
    assert s.decide(supervisor.Failure("movida", "t", "r", "se ha movido 40 mm"), 0).action == "replanificar"
    f = supervisor.Failure("fuera_de_caja", "t", "r", "fuera")
    assert [s.decide(f, 0).action for _ in range(3)] == ["replanificar", "replanificar", "abandonar"]
