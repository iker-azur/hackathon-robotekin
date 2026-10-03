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


def contained_region() -> tuple[tuple[float, float], tuple[float, float]]:
    """The area the pen encloses, which is the working area exactly."""
    import scene_spec

    return (scene_spec.WORK_X, scene_spec.WORK_Y)


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