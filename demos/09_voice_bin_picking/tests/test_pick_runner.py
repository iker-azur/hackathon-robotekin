"""El ejecutor de la política a 30 Hz, con un robot falso que sólo sigue las consignas (sin física).

Comprueba la lógica de decisión (cuándo hay éxito y por qué se falla), no que el brazo agarre de verdad.
"""

import numpy as np
import pytest

from binpick import config as C
from binpick import observation as O
from binpick import picking
from binpick import policy as POL
from binpick.pick_runner import PickRunner
from recoverykit import skills
from recoverykit.workcell import base_transform

SCENE = C.build_scene()
BASE = base_transform(SCENE)
TOP = SCENE.table.top_z_m
BIN = C.container(SCENE, C.BIN_ID)
Q_HOME = np.array(SCENE.robot.start_joints[:5])
SPEC = O.make_spec(__import__("binpick.segmentation", fromlist=["x"]).overhead_camera(SCENE.cameras[0]), TOP, C.BinPickCfg())


def make_plan():
    x, y = BIN.center_xy_m
    z = TOP + BIN.floor_m + C.PIECE_SIZE_M / 2
    t = skills.Target(x, y, 0.0, C.PIECE_SIZE_M, C.PIECE_SIZE_M, "cube")
    plan = picking.plan_pick_ex(BASE, TOP, t, z, TOP + BIN.outer_m[2], Q_HOME, floor_z=TOP + BIN.floor_m)
    assert not isinstance(plan, str), plan
    return plan


class FakeCell:
    """Las articulaciones llegan a la consigna; la pinza se detiene en el contacto si hay pieza entre los dedos."""

    def __init__(self, plan, piece="present"):
        self.plan, self.piece = plan, piece
        self.q = Q_HOME.copy()
        self.g = 0.0
        self.q_t, self.g_t = Q_HOME.copy(), 0.0
        self.sim_time = 0.0
        self.targets = []

    def proprio(self):
        return {"arm_q": self.q.copy(), "gripper_q": self.g}

    def set_arm_targets(self, q):
        self.q_t = np.asarray(q, float)
        self.targets.append(self.q_t.copy())

    def set_gripper_target(self, g):
        self.g_t = float(g)

    def tick(self, n):
        self.q = self.q_t.copy()
        h = picking.tcp_height(BASE, self.q, self.plan.local)
        low = h < picking.tcp_height(BASE, self.plan.descent[-1], self.plan.local) + 0.01
        lifted = h > picking.tcp_height(BASE, self.plan.descent[-1], self.plan.local) + 0.02
        has = self.piece == "present" or (self.piece == "slips" and not lifted)
        if has and self.g_t < self.plan.q_contact and (low or self.piece == "present" or self.g > self.plan.q_contact):
            self.g = max(self.g_t, self.plan.q_contact)           # bloqueada por la pieza
        else:
            self.g = self.g_t
        self.sim_time += 1 / 240
        return None


def expert_actions(plan, descend=True, close=True, lift=True):
    """Lo que haría un experto, un paso de control a la vez: abrir, aproximar, bajar, cerrar, subir."""
    acts = []
    q = Q_HOME.copy()
    for _ in range(8):
        acts.append(np.r_[q, plan.q_open])
    for i in range(1, 31):
        acts.append(np.r_[Q_HOME + (plan.q_pre - Q_HOME) * i / 30, plan.q_open])
    if descend:
        for dq in plan.descent[::2]:
            acts.append(np.r_[dq, plan.q_open])
    last = plan.descent[-1] if descend else plan.q_pre
    if close:
        for g in np.linspace(plan.q_open, plan.q_squeeze, 8):
            acts.append(np.r_[last, g])
    if lift:
        for dq in plan.lift[::2]:
            acts.append(np.r_[dq, plan.q_squeeze])
        for _ in range(HOLD := 12):
            acts.append(np.r_[plan.lift[-1], plan.q_squeeze])
    return acts


class Replay(POL.CallablePolicy):
    def __init__(self, acts):
        self.acts, self.i = acts, 0
        super().__init__(self._next, "replay")

    def _next(self, obs):
        a = self.acts[min(self.i, len(self.acts) - 1)]
        self.i += 1
        return a

    def reset(self):
        self.i = 0


def runner(cell, **kw):
    img = np.zeros((SPEC.roi[3] + 1, SPEC.roi[2] + 1, 3), np.uint8)
    return PickRunner(cell, cell.tick, lambda: (img, None), SPEC, BASE, TOP, **kw)


def run(plan, piece="present", **kw):
    cell = FakeCell(plan, piece)
    seg = np.zeros_like(np.zeros((SPEC.roi[3] + 1, SPEC.roi[2] + 1, 3), np.uint8))
    out = runner(cell).run(Replay(expert_actions(plan, **kw)), plan, seg)
    return out, cell


def test_expert_replay_succeeds_and_reports_limiter_stats():
    plan = make_plan()
    out, cell = run(plan)
    assert out.ok and out.status == "ok" and out.steps < 270
    assert out.limiter["pasos"] == out.steps and out.max_height_m > 0
    assert out.limiter["rechazos_por_mesa"] == 0


def test_never_descending_is_reported_as_not_arriving():
    out, _ = run(make_plan(), descend=False, close=False, lift=False)
    assert not out.ok and out.status == "fallo" and "no llegó" in out.detail


def test_closing_on_nothing_is_reported_as_empty():
    out, _ = run(make_plan(), piece="absent")
    assert not out.ok and out.status == "vacío"


def test_piece_that_slips_while_lifting_is_reported_as_slipped():
    out, _ = run(make_plan(), piece="slips")
    assert not out.ok and out.status in ("deslizada", "vacío") and "soltó" in out.detail or out.status == "vacío"


def test_holding_but_not_lifting_times_out_with_a_reason():
    out, _ = run(make_plan(), lift=False)
    assert not out.ok and out.status == "fallo" and "no la levantó" in out.detail


def test_a_wild_policy_is_clipped_and_cannot_hit_the_table():
    plan = make_plan()
    cell = FakeCell(plan)
    seg = np.zeros((SPEC.roi[3] + 1, SPEC.roi[2] + 1, 3), np.uint8)
    wild = POL.CallablePolicy(lambda obs: np.array([3.0, 1.7, -1.7, 1.6, 0.0, 3.0]), "loca")
    out = runner(cell, max_steps=60).run(wild, plan, seg)
    assert not out.ok
    assert out.limiter["recortes_de_limite"] > 0 and out.limiter["recortes_de_velocidad"] > 0
    q = np.array(cell.targets)
    lim = POL.ActionLimiter()
    assert (q >= lim.lo[:5] - 1e-9).all() and (q <= lim.hi[:5] + 1e-9).all()
    assert (np.abs(np.diff(q, axis=0)) <= lim.max_arm_step + 1e-9).all()


def test_stop_request_during_motion_aborts_cleanly():
    plan = make_plan()
    cell = FakeCell(plan)
    seg = np.zeros((SPEC.roi[3] + 1, SPEC.roi[2] + 1, 3), np.uint8)

    def tick(n):
        cell.tick(n)
        return "detenido por el usuario" if n > 40 else None

    img = np.zeros_like(seg)
    r = PickRunner(cell, tick, lambda: (img, None), SPEC, BASE, TOP)
    out = r.run(Replay(expert_actions(plan)), plan, seg)
    assert out.status == "abortado" and out.stopped


def test_policy_receives_the_documented_observation():
    plan = make_plan()
    seen = []

    def fn(obs):
        seen.append(obs)
        return np.r_[Q_HOME, plan.q_open]

    cell = FakeCell(plan)
    seg = np.zeros((SPEC.roi[3] + 1, SPEC.roi[2] + 1, 3), np.uint8)
    seg[10:20, 10:20, 0] = 255
    runner(cell, max_steps=3).run(POL.CallablePolicy(fn), plan, seg)
    o = seen[0]
    assert o.overhead.shape == (*SPEC.out_hw, 3) and o.target.shape == (*SPEC.out_hw, 3) and o.state.shape == (6,)
    assert seen[0].state.dtype == np.float32
