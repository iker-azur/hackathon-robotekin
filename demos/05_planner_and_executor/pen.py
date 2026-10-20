"""A kerb around the working area, so a disturbed object cannot leave it.

**The problem this solves, measured.** The arm knocks whatever it reaches for: a ball ends up
57-62 mm away on the way in and then rolls 100-370 mm more, sometimes leaving the table
entirely - measured at x = -0.22 on a table whose near edge is x = 0.003, and once at x = -2.1.
Outside the working area the shape classifier stops calling it a ball (the thresholds in
``colors.py`` were fitted near the centre of the frame), so an object that escapes cannot be
re-planned for even in principle.

**How it is built.** One kinematic rigid body whose collision shapes are four walls, and no
floor - the table is the floor. That is ``hollow.py``'s tray mechanism with the floor part left
off, so this reuses its spawner rather than writing a second one.

**Why kinematic.** The same reason the trays are: a wall the arm can knock over is not a wall.
It also means the pen can be repositioned between rounds through ``write_root_pose_to_sim``.

**Where it goes, and the tension that creates.** The inner face of the kerb sits exactly on the
working-area boundary, so containment is precise rather than approximate: an object is either
inside the area, where the robot can reach it, or stopped at the edge. But the arm's base is at
y = -0.64, beyond the area's near edge at y = -0.53, so **every move crosses the near kerb**.
Whether the gripper and forearm clear a 2.5 cm wall on the way in is not something to assume:
``tools/probe_pen.py`` measures it, and ``PEN_HEIGHT`` below is the number that measurement is
expected to justify or contradict.
"""

from __future__ import annotations

# The kerb's inner face is on the working-area boundary and it grows outward from there, so the
# contained region is exactly WORK_X by WORK_Y.
PEN_WALL = 0.008

# Tall enough to stop a ball rather than let it roll over: a 4 cm ball's centre is 2 cm above
# the table, so a kerb that reaches past the centre catches it instead of being climbed. This
# is the value tools/probe_pen.py is expected to confirm or overturn.
PEN_HEIGHT = 0.025

# The contained region, in world metres. Read from scene_spec at spawn time so there is one
# definition of the working area rather than two.
PEN_INNER = None  # set by build, from scene_spec.WORK_X / WORK_Y

# The pen's own centre, and the prim it is spawned under.
PEN_PRIM = "pen"
PEN_COLOR = "black"


def pen_parts(inner_w: float, inner_d: float, wall: float, height: float
              ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """The four cuboids of the kerb: name, per-axis size, and offset from the pen's centre.

    Same shape as ``hollow.tray_parts`` and deliberately so - identical return type, identical
    offset convention, no floor part. The table is the floor; adding a slab here would raise the
    whole working area by its thickness and put a step under everything.
    """
    outer_w = inner_w + 2.0 * wall
    outer_d = inner_d + 2.0 * wall
    return [
        ("wall_neg_x", (wall, outer_d, height), (-(inner_w / 2.0 + wall / 2.0), 0.0, 0.0)),
        ("wall_pos_x", (wall, outer_d, height), ((inner_w / 2.0 + wall / 2.0), 0.0, 0.0)),
        ("wall_neg_y", (outer_w, wall, height), (0.0, -(inner_d / 2.0 + wall / 2.0), 0.0)),
        ("wall_pos_y", (outer_w, wall, height), (0.0, (inner_d / 2.0 + wall / 2.0), 0.0)),
    ]


# How close to the robot's base axis the kerb's near (robot-side) face may be. The arm folds low
# beside its base, and a wall nearer than this stops it: strict-physical pick-and-place of the
# same cube passed with the near face 85 mm from the base axis (y = -0.555) and failed every
# time at 75, 65, 55 and 45 mm ("pre-grasp hover did not converge ... la pose no es alcanzable"),
# on the same part, in the open variant it passed. A saved scene whose working area reaches
# closer to the base than this (the dashboard allows it) therefore got a kerb the arm could not
# work beside, and nothing was ever picked up.
PEN_NEAR_LIMIT_FROM_BASE_M = 0.090

# Room the open jaws need between a part and the kerb on the far and side faces. With the
# quarter-turn wrist (+90 degrees, which the controller picks for 30 mm cubes beyond 0.23 m) the
# jaws span the y direction and reach about 33 mm from the part's centre. The same cube passed
# in the contained variant with the far wall 45 mm from its centre and failed ("aperture ... the
# jaws cannot reach it from here") with it 22 mm away, passing in the open variant in the same
# scene. A part inside the working area is at least 15 mm from its edge, so 30 mm more keeps
# every part at least 45 mm from the wall.
PEN_JAW_CLEARANCE_M = 0.030


def contained_region() -> tuple[tuple[float, float], tuple[float, float]]:
    """The area the pen encloses: the working area plus jaw room, but never nearer the robot than the arm allows.

    The far and side walls stand ``PEN_JAW_CLEARANCE_M`` outside the working area so the jaws
    never touch them; the near (robot-side) wall is only ever moved away from the robot. Objects
    and trays nearer the base than the near wall simply stand outside the kerb, on open table.
    """
    import scene_spec

    work_x, work_y = scene_spec.WORK_X, scene_spec.WORK_Y
    near_limit = scene_spec.ROBOT_XY[1] + PEN_NEAR_LIMIT_FROM_BASE_M
    near = min(max(work_y[0], near_limit), work_y[1] - 0.05)
    return ((work_x[0] - PEN_JAW_CLEARANCE_M, work_x[1] + PEN_JAW_CLEARANCE_M),
            (near, work_y[1] + PEN_JAW_CLEARANCE_M))


def make_pen_spawner_cfg():
    """The spawner config: a tray-shaped kinematic body with no floor, sized to the work area."""
    import isaaclab.sim as sim_utils
    import colors
    import hollow

    work_x, work_y = contained_region()
    inner_w = work_x[1] - work_x[0]
    inner_d = work_y[1] - work_y[0]

    # floor=0.0 tells hollow's part builder to emit no floor slab.
    return hollow.tray_spawner_cls()(
        inner_w=inner_w,
        inner_d=inner_d,
        wall=PEN_WALL,
        height=PEN_HEIGHT,
        floor=0.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=colors.COLORS[PEN_COLOR].albedo, roughness=0.9
        ),
    )


def make_pen_asset_cfg():
    """A ``RigidObjectCfg`` for the kerb, centred on the working area."""
    from isaaclab.assets import RigidObjectCfg

    import scene_spec

    work_x, work_y = contained_region()
    centre = ((work_x[0] + work_x[1]) / 2.0, (work_y[0] + work_y[1]) / 2.0)
    # Sits on the table: the kerb's local origin is its mid-height, so the centre is half the
    # height above the table surface.
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/" + PEN_PRIM,
        spawn=make_pen_spawner_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(centre[0], centre[1], scene_spec.TABLE_TOP_Z + PEN_HEIGHT / 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )