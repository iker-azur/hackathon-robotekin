"""Seguimiento entre fotogramas y elección de la pieza a coger."""

import math

import numpy as np
import pytest

from binpick import commands as K
from binpick import config as C
from binpick import selection as SEL
from binpick import segmentation as S
from binpick.tracker import Tracker
from synth import piece, render, scene_and_ctx

SCENE, CTX = scene_and_ctx()
SEG = S.HsvSegmenter(CTX)
SZ = C.PIECE_SIZE_M


def see(pieces):
    rgb, depth = render(SCENE, CTX, pieces)
    return SEG.segment(rgb, depth)


# ------------------------------------------------------------------ tracker
def test_tracker_keeps_ids_and_reports_new():
    tr = Tracker()
    a = see([piece(SCENE, "r", "red", 0.19, -0.02), piece(SCENE, "b", "blue", 0.22, 0.025)])
    ev = tr.update(a)
    assert sorted(e.kind for e in ev) == ["nueva", "nueva"]
    ids = {i.color: i.track for i in a}
    b = see([piece(SCENE, "r", "red", 0.1905, -0.0195), piece(SCENE, "b", "blue", 0.22, 0.025)])
    assert tr.update(b) == []
    assert {i.color: i.track for i in b} == ids


def test_tracker_detects_moved_piece_and_region_change():
    tr = Tracker()
    tr.update(see([piece(SCENE, "r", "red", 0.19, -0.02)]))
    ev = tr.update(see([piece(SCENE, "r", "red", 0.215, -0.02)]))
    assert [e.kind for e in ev] == ["movida"]
    # la pieza aparece en la caja izquierda: misma pista, cambio de caja
    inst = see([piece(SCENE, "r", "red", 0.215, -0.02)])[0]
    inst.region, inst.pos = "izquierda", np.array([0.095, 0.15, 0.78])
    ev = tr.update([inst])
    assert [e.kind for e in ev] == ["cambio_de_caja"] and "izquierda" in ev[0].detail


def test_tracker_color_must_match():
    tr = Tracker()
    tr.update(see([piece(SCENE, "r", "red", 0.2, 0.0)]))
    ev = tr.update(see([piece(SCENE, "b", "blue", 0.2, 0.0)]))
    assert "nueva" in [e.kind for e in ev]


def test_tracker_lost_only_when_visible():
    tr = Tracker()
    tr.update(see([piece(SCENE, "r", "red", 0.2, 0.0)]))
    for _ in range(5):
        tr.update([], visible=lambda p: False)       # el brazo la tapa
    assert not any(t.lost for t in tr.tracks.values())
    kinds = [e.kind for _ in range(3) for e in tr.update([], visible=lambda p: True)]
    assert "perdida" in kinds and all(t.lost for t in tr.tracks.values())


def test_tracker_counts():
    tr = Tracker()
    tr.update(see([piece(SCENE, "r", "red", 0.19, -0.02), piece(SCENE, "r2", "red", 0.22, 0.025)]))
    assert tr.counts("origen") == 2 and tr.counts("origen", "red") == 2 and tr.counts("izquierda") == 0


# ----------------------------------------------------------------- selection
def order(text):
    o = K.parse(text)
    assert o.ok, o.errors
    return o


def with_tracks(insts):
    Tracker().update(insts)
    return insts


def test_only_matching_colors_are_candidates():
    insts = with_tracks(see([piece(SCENE, "r", "red", 0.19, -0.02), piece(SCENE, "b", "blue", 0.22, 0.025)]))
    sel = SEL.rank(insts, order("pon las rojas a la izquierda"), SCENE)
    assert [c.inst.color for c in sel.usable] == ["red"] and sel.usable[0].dest == "izquierda"
    assert sel.pending == 1


def test_top_of_pile_first_and_covered_piece_is_blocked():
    insts = with_tracks(see([piece(SCENE, "lo", "red", 0.2, 0.0), piece(SCENE, "up", "red", 0.208, 0.004, layer=1),
                             piece(SCENE, "x", "red", 0.23, -0.02)]))
    sel = SEL.rank(insts, order("las rojas a la derecha"), SCENE)
    assert sel.usable[0].inst.top_z > 0.80 or sel.usable[0].inst.top_z == max(i.top_z for i in insts)
    assert len(sel.usable) == 2 and len(sel.blocked) == 1
    assert "encima" in sel.blocked[0].blocked or "tapada" in sel.blocked[0].blocked


def test_quota_limits_candidates():
    insts = with_tracks(see([piece(SCENE, "a", "red", 0.19, -0.04), piece(SCENE, "b", "red", 0.22, 0.04),
                             piece(SCENE, "c", "red", 0.205, 0.0)]))
    sel = SEL.rank(insts, order("pon dos piezas rojas en la caja izquierda"), SCENE)
    assert len(sel.usable) == 2
    sel = SEL.rank(insts, order("pon dos piezas rojas en la caja izquierda"), SCENE, done={0: 1})
    assert len(sel.usable) == 1
    assert SEL.rank(insts, order("pon dos piezas rojas en la caja izquierda"), SCENE, done={0: 2}).usable == []


def test_rest_rule_takes_what_other_rules_do_not_claim():
    insts = with_tracks(see([piece(SCENE, "r", "red", 0.19, -0.02), piece(SCENE, "b", "blue", 0.22, 0.025),
                             piece(SCENE, "g", "green", 0.2, 0.0)]))
    sel = SEL.rank(insts, order("las rojas a la izquierda y las demás a la derecha"), SCENE)
    dest = {c.inst.color: c.dest for c in sel.usable}
    assert dest == {"red": "izquierda", "blue": "derecha", "green": "derecha"}


def test_blacklist_attempts_and_full_box():
    insts = with_tracks(see([piece(SCENE, "a", "red", 0.19, -0.02), piece(SCENE, "b", "red", 0.22, 0.025)]))
    o = order("las rojas a la izquierda")
    first = SEL.rank(insts, o, SCENE).usable
    bad = first[0].track
    sel = SEL.rank(insts, o, SCENE, blacklist={bad})
    assert [c.track for c in sel.usable] == [first[1].track] and "descartada" in sel.blocked[0].blocked
    sel = SEL.rank(insts, o, SCENE, attempts={first[0].track: 2})
    assert sel.usable[0].track == first[1].track                       # el que falló dos veces baja
    sel = SEL.rank(insts, o, SCENE, full={"izquierda"})
    assert sel.usable == [] and any("llena" in n for n in sel.notes)


def test_pieces_outside_the_bin_are_not_candidates():
    insts = with_tracks(see([piece(SCENE, "a", "red", 0.2, 0.0)]))
    insts[0].region = "izquierda"
    assert SEL.rank(insts, order("las rojas a la derecha"), SCENE).usable == []


def test_piece_hard_against_the_wall_has_fewer_gripper_orientations():
    b = C.container(SCENE, C.BIN_ID)
    wall_x = b.center_xy_m[0] + b.inner_m[0] / 2
    near_wall = with_tracks(see([piece(SCENE, "w", "red", wall_x - SZ / 2 - 0.001, b.center_xy_m[1], absolute=True)]))[0]
    middle = with_tracks(see([piece(SCENE, "m", "red", *C.container(SCENE, C.BIN_ID).center_xy_m, absolute=True)]))[0]
    rect = C.inner_margin_box(SCENE, C.BIN_ID, 0.0)
    gw = SEL.assess_grasp(near_wall, [near_wall], rect)
    gm = SEL.assess_grasp(middle, [middle], rect)
    assert len(gw.yaws_ok) < len(gm.yaws_ok) == 4


def test_piece_surrounded_by_four_neighbours_is_blocked():
    cx, cy = 0.205, 0.0
    ps = [piece(SCENE, "c", "red", cx, cy)] + [piece(SCENE, f"n{k}", "blue", cx + dx, cy + dy)
                                               for k, (dx, dy) in enumerate(((SZ + .001, 0), (-SZ - .001, 0), (0, SZ + .001), (0, -SZ - .001)))]
    insts = with_tracks(see(ps))
    sel = SEL.rank(insts, order("las rojas a la izquierda"), SCENE)
    assert sel.usable == [] and "dedos" in sel.blocked[0].blocked


def test_first_reachable_skips_unreachable_and_explains():
    insts = with_tracks(see([piece(SCENE, "a", "red", 0.19, -0.02), piece(SCENE, "b", "red", 0.22, 0.025)]))
    sel = SEL.rank(insts, order("las rojas a la izquierda"), SCENE)
    first = sel.usable[0].track
    cand, why = SEL.first_reachable(sel, lambda c: "fuera de alcance" if c.track == first else None)
    assert cand.track != first and len(why) == 1 and "fuera de alcance" in why[0]
    cand, why = SEL.first_reachable(SEL.rank(insts, order("las rojas a la izquierda"), SCENE), lambda c: "no")
    assert cand is None and len(why) == 2


def test_first_reachable_relaxed_is_last_resort_and_strict_candidates_win():
    insts = with_tracks(see([piece(SCENE, "a", "red", 0.19, -0.02), piece(SCENE, "b", "red", 0.22, 0.025)]))
    sel = SEL.rank(insts, order("las rojas a la izquierda"), SCENE)
    first, second = sel.usable[0].track, sel.usable[1].track
    relaxed_calls = []

    def relaxed(c):
        relaxed_calls.append(c.track)
        return None

    # una estricta disponible: gana aunque sea la segunda, y no se llega a probar la permisiva
    cand, _ = SEL.first_reachable(sel, lambda c: "roza la pared" if c.track == first else None, relaxed)
    assert cand.track == second and relaxed_calls == []
    # ninguna estricta: se usa la permisiva, en orden de puntuación, y ya no figura como bloqueada
    sel = SEL.rank(insts, order("las rojas a la izquierda"), SCENE)
    cand, why = SEL.first_reachable(sel, lambda c: "roza la pared", relaxed)
    assert cand.track == first and why == [] and cand.blocked is None and relaxed_calls == [first]
    # sin permisiva, el comportamiento de siempre
    sel = SEL.rank(insts, order("las rojas a la izquierda"), SCENE)
    assert SEL.first_reachable(sel, lambda c: "no")[0] is None


def test_first_reachable_relaxed_also_tries_finger_blocked_pieces_last():
    cx, cy = 0.205, 0.0
    ps = [piece(SCENE, "c", "red", cx, cy)] + [piece(SCENE, f"n{k}", "blue", cx + dx, cy + dy)
                                               for k, (dx, dy) in enumerate(((SZ + .001, 0), (-SZ - .001, 0), (0, SZ + .001), (0, -SZ - .001)))]
    insts = with_tracks(see(ps))
    sel = SEL.rank(insts, order("las rojas a la izquierda"), SCENE)
    assert sel.usable == [] and "dedos" in sel.blocked[0].blocked
    assert sel.blocked[0].grasp.yaws_all and not sel.blocked[0].grasp.yaws_ok
    assert SEL.first_reachable(sel, lambda c: "no")[0] is None                      # estricto: no hay nada que hacer
    tried = []
    cand, why = SEL.first_reachable(sel, lambda c: "no", lambda c: tried.append(c.track))
    assert cand is not None and cand.track == sel.blocked[0].track and why == [] and cand.blocked is None
    assert cand.grasp.yaws_ok == cand.grasp.yaws_all[:2]                            # se le da el giro de más holgura


def test_jaw_slots_geometry():
    fixed, moving = SEL.jaw_slots(np.array([0.2, 0.0]), 0.0, SZ)
    assert fixed[:, 0].max() == pytest.approx(0.2 - SZ / 2) and moving[:, 0].min() == pytest.approx(0.2 + SZ / 2 + SEL.picking.OPEN_MARGIN_M)


def test_real_inverse_kinematics_reaches_pieces_in_the_bin():
    from recoverykit import kinematics as Kn, skills
    from recoverykit.workcell import base_transform
    from binpick import picking
    base = base_transform(SCENE)
    top = SCENE.table.top_z_m
    b = C.container(SCENE, C.BIN_ID)
    inst = see([piece(SCENE, "a", "red", 0.205, 0.0)])[0]
    t = skills.Target(float(inst.pos[0]), float(inst.pos[1]), inst.yaw, SZ, SZ, "cube")
    plan = picking.plan_pick_ex(base, top, t, float(inst.pos[2]), top + b.outer_m[2],
                                np.array(SCENE.robot.start_joints[:5]), floor_z=top + b.floor_m)
    assert not isinstance(plan, str), plan
    assert len(plan.descent) > 5 and len(plan.lift) >= 5
    lift_m = abs(Kn.fk(dict(zip(Kn.ARM, plan.lift[-1])), base)["gripper"][2, 3] - Kn.fk(dict(zip(Kn.ARM, plan.descent[-1])), base)["gripper"][2, 3])
    assert 0.02 < lift_m < 0.07
