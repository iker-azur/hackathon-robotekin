"""Hollow receptacles: one rigid body whose collision shapes are a floor and four walls.

Why this is not five objects
----------------------------
The obvious way to make a tray is five cuboids. That does not work here, and the reason is
worth recording. IsaacLab's ``RigidObject`` refuses a prim that contains more than one
rigid body outright - ``isaaclab/assets/rigid_object/rigid_object.py:474-479`` raises

    Failed to find a single rigid body when resolving '<path>'.
    Found multiple [...] under '<path>'.

because it builds one PhysX tensor view per asset and a view cannot span bodies. So the
five parts cannot be five sibling assets.

``RigidObjectCollection`` does give one handle over several bodies, but then the walls and
the floor collide *with each other*, and IsaacLab exposes no pairwise collision filter
(there is no ``filter_collision_pairs`` API in this version; only a per-asset
``collision_group``). A floor pushing up against its own walls is not a tray.

The way through is USD's own model: a rigid body prim with collision geometry as
*descendants*. ``spawn_cuboid`` already does exactly that for one box - the root Xform
gets mass and rigid body properties, and ``{prim_path}/geometry/mesh`` gets collision
(``isaaclab/sim/spawners/shapes/shapes.py:276-311``). So this spawner calls it five times
with the child paths, passing ``rigid_props=None`` and ``mass_props=None`` each time so
the children are geometry only, and puts a single rigid body on the parent. The parent is
kinematic, so the tray cannot be knocked over, but objects still collide with it and
``write_root_pose_to_sim`` can still move it between rounds.

Geometry
--------
A tray is ``inner`` x ``inner`` x ``height`` of usable cavity inside an ``outer`` footprint
of ``inner + 2 * wall``. ``Item.size`` stays the OUTER footprint on purpose: the layout
maths - ``half_width``, ``half_depth``, ``clearance``, the reach and working-area checks -
all read ``size``, so keeping it at the outer footprint means the working area that was
already fitted for demo 09 still holds.

The declared outer footprint is not the same as the *detected* one: a tray seen from 67
degrees shows its rim, the inside of its far wall and part of its floor, and its blob is
not a filled rectangle. ``perceive.inner_rect`` therefore estimates the cavity by
insetting the detected bounding box, and ``WALL_INSET_FRACTION`` is the constant that has
to be measured against whatever this actually renders. That measurement is the gate on
this file.
"""

from __future__ import annotations


def tray_parts(inner_w: float, inner_d: float, wall: float, height: float,
               floor: float) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """The five cuboids of a tray: name, per-axis size, and centre offset from the tray's centre.

    Offsets are local to the tray's own Xform, whose origin is its centre, so this is
    pure geometry with no knowledge of where the tray sits on the table.
    """
    outer_w = inner_w + 2.0 * wall
    outer_d = inner_d + 2.0 * wall
    bottom = -height / 2.0 + floor / 2.0
    return [
        ("floor", (outer_w, outer_d, floor), (0.0, 0.0, bottom)),
        ("wall_neg_x", (wall, outer_d, height), (-(inner_w / 2.0 + wall / 2.0), 0.0, 0.0)),
        ("wall_pos_x", (wall, outer_d, height), ((inner_w / 2.0 + wall / 2.0), 0.0, 0.0)),
        ("wall_neg_y", (outer_w, wall, height), (0.0, -(inner_d / 2.0 + wall / 2.0), 0.0)),
        ("wall_pos_y", (outer_w, wall, height), (0.0, (inner_d / 2.0 + wall / 2.0), 0.0)),
    ]


def cavity_ratio(inner_w: float, inner_d: float, wall: float) -> float:
    """The cavity as a fraction of the outer footprint, averaged over both axes.

    This is the number ``perceive.WALL_INSET_FRACTION`` has to reproduce from the *detected*
    footprint, and it is what the placement target is actually derived from.
    """
    return ((inner_w / (inner_w + 2.0 * wall)) + (inner_d / (inner_d + 2.0 * wall))) / 2.0


def inset_fraction_for(inner_w: float, inner_d: float, wall: float) -> float:
    """The ``WALL_INSET_FRACTION`` value that makes an inset reproduce this cavity."""
    return (1.0 - cavity_ratio(inner_w, inner_d, wall)) / 2.0


def spawn_hollow_tray(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a hollow tray. Resolves ``{ENV_REGEX_NS}``, then builds one tray per env.

    Called by IsaacLab's ``AssetBase.spawn`` as
    ``cfg.func(prim_path, cfg, translation=..., orientation=...)``.

    The path arrives with ``{ENV_REGEX_NS}`` still in it, as the literal string
    ``/World/envs/env_.*/Scene/tray_blue``, and USD rejects it: *"Path must be an absolute
    path"* is the unhelpful way it says ``.*`` is not a legal path component. IsaacLab's
    own spawners never hit this because every one of them is wrapped in the ``@clone``
    decorator (``isaaclab/sim/spawners/shapes/shapes.py:107``), which splits the path into
    a possibly-regex root and a plain leaf and resolves the root against the stage
    (``isaaclab/sim/utils.py:246-264``).

    This does the same resolution by hand rather than using the decorator, because
    ``@clone`` also arranges visibility flags, semantic tags and instance cloning, none of
    which apply: a tray's five parts are spawned as real children at the resolved path
    instead of being cloned from a prototype.
    """
    import re

    from isaaclab.sim.utils import find_matching_prim_paths

    path = str(prim_path)
    if not path.startswith("/"):
        raise ValueError(f"Prim path '{path}' is not global. It must start with '/'.")
    root, leaf = path.rsplit("/", 1)
    is_regex = re.match(r"^[a-zA-Z0-9/_]+$", root) is None
    if is_regex and root:
        roots = find_matching_prim_paths(root)
        if not roots:
            raise RuntimeError(
                f"Unable to find source prim path: '{root}'. Please create the prim before spawning."
            )
    else:
        roots = [root]
    prims = [_spawn_tray_at(f"{r}/{leaf}", cfg, translation, orientation, **kwargs) for r in roots]
    return prims[0]


def _spawn_tray_at(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Build one tray at one already-resolved prim path."""
    import isaacsim.core.utils.prims as prim_utils
    import isaaclab.sim as sim_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners.shapes import shapes as shape_spawners

    # The parent. Created here rather than by spawn_cuboid, because spawn_cuboid would put
    # collision on the parent's own mesh and we want none of that - the parent is the body,
    # the children are the shapes.
    if prim_utils.is_prim_path_valid(prim_path):
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation,
                           orientation=orientation)

    rigid = cfg.rigid_props or sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    if getattr(rigid, "kinematic_enabled", False) is not True:
        raise ValueError(
            f"tray {prim_path} would be dynamic. A receptacle the robot can knock over is "
            f"not a receptacle; pass kinematic_enabled=True."
        )
    schemas.define_rigid_body_properties(prim_path, rigid)
    schemas.define_mass_properties(prim_path, cfg.mass_props or sim_utils.MassPropertiesCfg(mass=0.2))

    # The five shapes. rigid_props=None and mass_props=None are the whole point: they make
    # each child geometry-only, so the parent stays the single rigid body.
    for name, size, offset in tray_parts(cfg.inner_w, cfg.inner_d, cfg.wall, cfg.height, cfg.floor):
        child_cfg = sim_utils.CuboidCfg(
            size=size,
            rigid_props=None,
            mass_props=None,
            collision_props=cfg.collision_props or sim_utils.CollisionPropertiesCfg(),
            visual_material=cfg.visual_material,
        )
        shape_spawners.spawn_cuboid(
            f"{prim_path}/{name}", child_cfg, translation=offset, orientation=(1.0, 0.0, 0.0, 0.0)
        )
    return prim_utils.get_prim_at_path(prim_path)


_TRAY_SPAWNER_CLS = None


def tray_spawner_cls():
    """``RigidObjectSpawnerCfg`` plus the five numbers ``spawn_hollow_tray`` needs.

    Built on first use because ``configclass`` and the IsaacLab imports are not available
    at module import time. It closes over nothing: every tray's numbers are passed to the
    constructor, so all trays share one class.
    """
    global _TRAY_SPAWNER_CLS
    if _TRAY_SPAWNER_CLS is not None:
        return _TRAY_SPAWNER_CLS

    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.utils import configclass

    @configclass
    class HollowTraySpawnerCfg(RigidObjectSpawnerCfg):
        """The tray geometry, carried on the spawner config the way rigid props are."""

        func: object = spawn_hollow_tray
        inner_w: float = 0.03
        inner_d: float = 0.03
        wall: float = 0.008
        height: float = 0.05
        floor: float = 0.006
        # RigidObjectSpawnerCfg carries mass, rigid and collision properties but not a
        # material - those live on the shape configs, and this spawner builds its own five
        # shapes, so it has to bring its own.
        visual_material: object = None

    _TRAY_SPAWNER_CLS = HollowTraySpawnerCfg
    return _TRAY_SPAWNER_CLS


def make_tray_spawner_cfg(item):
    """The spawner config for one tray ``Item``, from its declared inner size and wall.

    Needs Isaac, so it is a function rather than a module-level constant.
    """
    import isaaclab.sim as sim_utils
    import colors

    if not item.container:
        raise ValueError(f"{item.prim} is not a container")
    if item.inner == (0.0, 0.0) or item.wall <= 0.0:
        raise ValueError(f"{item.prim} has no declared cavity: inner={item.inner} wall={item.wall}")

    return tray_spawner_cls()(
        inner_w=item.inner[0],
        inner_d=item.inner[1],
        wall=item.wall,
        height=item.height,
        floor=item.floor,
        # Kinematic, so the tray stays put when a dropped object lands on its rim, and so
        # it can still be repositioned between rounds through write_root_pose_to_sim.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=colors.COLORS[item.color].albedo, roughness=0.7
        ),
    )


def make_tray_asset_cfg(item):
    """A ``RigidObjectCfg`` for a hollow tray, spawned by ``spawn_hollow_tray``."""
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/" + item.prim,
        spawn=make_tray_spawner_cfg(item),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(item.x, item.y, item.z), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )