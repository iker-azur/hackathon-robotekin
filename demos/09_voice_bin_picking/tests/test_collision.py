"""Colisión del brazo con las paredes en los trayectos de agarre (numpy)."""

import numpy as np

from binpick import collision as CO
from binpick import config as C
from binpick import picking
from recoverykit import skills
from recoverykit.workcell import base_transform

SCENE = C.build_scene()
BASE = base_transform(SCENE)
TOP = SCENE.table.top_z_m
BIN = C.container(SCENE, C.BIN_ID)
Q_HOME = np.array(SCENE.robot.start_joints[:5])


def plan_at(x, y, layer=0):
    t = skills.Target(x, y, 0.0, C.PIECE_SIZE_M, C.PIECE_SIZE_M, "cube")
    z = TOP + BIN.floor_m + layer * C.PIECE_SIZE_M + C.PIECE_SIZE_M / 2
    return picking.plan_pick_ex(BASE, TOP, t, z, TOP + BIN.outer_m[2], Q_HOME, floor_z=TOP + BIN.floor_m)


def test_wall_boxes_cover_every_container_with_four_walls():
    boxes = CO.wall_boxes(SCENE)
    assert len(boxes) == 4 * len(SCENE.containers)
    for _n, lo, hi in boxes:
        assert (hi > lo).all()


def test_home_pose_is_clear_of_all_walls():
    assert CO.path_hit(BASE, SCENE, [Q_HOME]) is None


def test_pick_in_the_middle_of_the_bin_has_a_clear_path():
    plan = plan_at(*BIN.center_xy_m)
    assert not isinstance(plan, str), plan
    assert CO.check_pick(BASE, SCENE, Q_HOME, plan) is None


def test_a_piece_touching_the_wall_can_still_be_grasped_along_the_free_axis():
    """Pegada a la pared +x se agarra cerrando a lo largo de y; sólo hay plan con ese giro."""
    from binpick import selection as SEL
    from binpick.instances import Instance
    x = BIN.center_xy_m[0] + BIN.inner_m[0] / 2 - C.PIECE_SIZE_M / 2 - 0.0005
    y = BIN.center_xy_m[1]
    z = TOP + BIN.floor_m + C.PIECE_SIZE_M / 2
    inst = Instance(0, np.zeros((2, 2), bool), color="red", shape="cube", pos=np.array([x, y, z]),
                    top_z=z + C.PIECE_SIZE_M / 2, region="origen")
    g = SEL.assess_grasp(inst, [inst], C.inner_margin_box(SCENE, C.BIN_ID, 0.0))
    t = skills.Target(x, y, 0.0, C.PIECE_SIZE_M, C.PIECE_SIZE_M, "cube")
    assert g.yaws_ok and all(abs(np.cos(y)) < 1e-6 for y in g.yaws_ok)      # el eje x (hacia la pared) está bloqueado
    plan, why = picking.find_plan(BASE, SCENE, t, z, TOP + BIN.floor_m, Q_HOME, g.yaws_ok)
    assert plan is not None, why


def test_grasping_a_piece_inside_a_wall_is_flagged():
    wall_x = BIN.center_xy_m[0] + BIN.outer_m[0] / 2 - BIN.wall_m / 2        # centro de la pared +x
    plan = plan_at(wall_x, BIN.center_xy_m[1])
    if isinstance(plan, str):                                                 # fuera de alcance: tampoco se intentaría
        return
    problem = CO.check_pick(BASE, SCENE, Q_HOME, plan)
    assert problem is not None and "pared" in problem


def test_joint_path_endpoints():
    qs = CO.joint_path(np.zeros(5), np.ones(5), n=10)
    assert len(qs) == 11 and np.allclose(qs[0], 0) and np.allclose(qs[-1], 1)


def test_with_finger_aware_orientations_most_paths_are_clear():
    """Como en el flujo real: el plan sólo recibe los giros con hueco para los dedos (selection.assess_grasp)."""
    from binpick import selection as SEL
    from binpick.instances import Instance
    rect = C.inner_margin_box(SCENE, C.BIN_ID, 0.0)
    free = hit = 0
    # rango en el que caen las piezas (scene_gen: a ≥ 22 mm de la pared): ahí el cuerpo de la pinza cabe
    for dx in np.linspace(-0.015, 0.015, 4):
        for dy in np.linspace(-0.06, 0.08, 6):
            x, y = BIN.center_xy_m[0] + dx, BIN.center_xy_m[1] + dy
            z = TOP + BIN.floor_m + C.PIECE_SIZE_M / 2
            inst = Instance(0, np.zeros((2, 2), bool), color="red", shape="cube", pos=np.array([x, y, z]),
                            top_z=z + C.PIECE_SIZE_M / 2, region="origen")
            g = SEL.assess_grasp(inst, [inst], rect)
            if not g.yaws_ok:
                continue
            t = skills.Target(x, y, 0.0, C.PIECE_SIZE_M, C.PIECE_SIZE_M, "cube")
            plan, why = picking.find_plan(BASE, SCENE, t, z, TOP + BIN.floor_m, Q_HOME, g.yaws_ok)
            free += plan is not None
            hit += plan is None
            if plan is not None:                                   # lo que devuelve está libre de choques
                assert CO.check_pick(BASE, SCENE, Q_HOME, plan) is None
            else:
                assert why
    print(f"con plan libre {free}, sin plan {hit}")
    assert free >= 18 and hit <= 0.15 * (free + hit)
