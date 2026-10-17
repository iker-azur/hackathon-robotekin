"""Demo 12 — the Isaac Lab scene. This module is the physics authority.

Design rules, in order of importance:

1. **Nothing here is allowed to fake physics.** There is no pose writing after the scene is
   built, no object attachment, no "held" state that suppresses gravity. A part is in a
   tray because PhysX put it there.

2. **Every mass, friction and restitution is derived from a real number.** Mass comes from
   geometry and material density (:mod:`demo12.materials`); friction comes from the
   material. A part cannot weigh something different from its size.

3. **Materials are bound to collision shapes, not assumed.** PhysX combines friction from
   the two surfaces in a contact, so declaring mu=0.9 on the gripper pads and then leaving
   the USD's default on the prim would quietly measure nothing. :func:`_bind_material`
   puts a real ``UsdPhysics.MaterialAPI`` on every collider that matters.

4. **The actuator gains are stated as tuning, not measurement.** See
   :mod:`demo12.materials` for which numbers are which.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from . import materials as M
from .config import BinConfig, PartConfig, RobotConfig, SceneConfig

# Isaac Lab prim paths. The cell lives under /World/Demo12 so it cannot collide with the
# default ground plane's bookkeeping or anything else a launch leaves behind.
ROOT = "/World/Demo12"

# Rigid bodies the contact sensors are attached to.
#
# ``jaw`` alone is not enough, and finding that out was the point of measuring: a descent
# onto a cube pushed the cube 31 mm while the jaw reported *zero* contact the whole time.
# Whatever the fingers were not doing, some other link was doing the pushing. Monitoring
# the wrist chain as well is what makes that visible instead of leaving the grasp geometry
# to guesswork.
MONITORED_BODIES = ("gripper", "jaw", "wrist", "lower_arm", "upper_arm")

# Links whose *shipped* collision is kept.
#
# The gripper assembly -- the wrist housing and the jaw -- is **not** in this list, because
# its collision in this asset is unusable and is replaced by the authored pads in
# ``_author_jaw_pads``. Keeping the jaw's own collider enabled while also adding pads was
# measured and is wrong twice over: the solid 22.3 mm block sits inside the 30 mm part the
# pads are trying to close on, so it shoves the part away *and* holds the jaw open at
# 1.488 rad -- past the 1.40 rad open command -- so the pads never close at all.
#
# Whether the *arm* links keep theirs was re-tested in Demo 12 Step 2, and the reason they
# were originally switched off no longer holds. That build's note says the instability came
# from the pads being "kinematic and teleported into the wrist every step" -- the pads are
# now bodies on the arm's own articulation driven by prismatic joints, so there is no
# teleport and no kinematic-versus-dynamic resolution to go wrong. See docs/PHYSICS.md for
# the measured outcome either way.
#
# Overridable from the environment so the experiment is reproducible without editing code.
# Set it to the empty string to switch arm collision off and re-run
# ``demo12.tools.probe_grasp`` to reproduce the comparison.
#
#     DEMO12_ARM_COLLISION=wrist,lower_arm,upper_arm,base,shoulder
#
def _arm_bodies_with_collision() -> tuple[str, ...]:
    import os

    raw = os.environ.get("DEMO12_ARM_COLLISION")
    if raw is None:
        # Measured, Demo 12 Step 2: this is stable. The grasp probe passes identically with
        # it on and off -- tracking 0.53 mm, 60.9 N of grip, a 98.9 mm rise, the part in the
        # bin and at rest in both cases -- because a pick-and-place never touches the bench
        # with the arm links. Verified from the stage rather than inferred: 5 arm colliders
        # enabled here against 0 with it off.
        return ("wrist", "lower_arm", "upper_arm", "base", "shoulder")
    return tuple(s.strip() for s in raw.split(",") if s.strip())


ARM_BODIES_WITH_COLLISION: tuple[str, ...] = _arm_bodies_with_collision()

# Measured, and worth recording rather than hiding. Keeping the arm's own links collidable
# while the pads are kinematic and teleported into the wrist every step is not stable:
# the pads see the arm, PhysX resolves a kinematic-versus-dynamic overlap by flinging one
# of them, and the arm goes with it. Measured on a clean build: 84,975 N on the upper arm
# and descents landing between 0.09 m and 0.46 m for commands spanning 32 mm.
#
# With the arm's collision off -- the pads are then the only thing on the robot that can
# touch anything -- the same descent converges to 0.3 mm. What is given up is that the arm
# can no longer be stopped by the bench or a tray. That is a real cost and it is recorded
# in docs/PHYSICS.md; restoring it needs a PhysX collision-group filter, which is declared
# as a token-array *relationship* that ``pxr.Usd.Relationship`` cannot write (it only
# accepts path targets).

# Physics contact-report force threshold, in newtons. Low on purpose: a grasp is made of
# light grazing contacts, and PhysX's default threshold hides exactly those.
CONTACT_REPORT_THRESHOLD_N = 1e-4

# PhysX collision groups.
#
# ``0``  everything that is supposed to collide with everything (parts, trays, bench).
# ``1``  the robot's own arm links.
# ``2``  the authored finger pads.
#
# Pads are excluded from ``1`` because they are kinematic: a kinematic body has infinite
# mass, so a pad that intersects its own wrist is resolved by the solver flinging the pad,
# and the arm goes with it. With the pads left colliding with the arm, every descent
# failed at a different height -- measured jaw heights of 0.09 to 0.42 m for nine commands
# spanning 32 mm -- which is what an unstable solver does, not what a reachable arm does.
PAD_COLLISION_GROUP = "2"
ARM_COLLISION_GROUP = "1"

# A finger contact below this is treated as "not touching". PhysX reports numerical
# residue on resting contacts, and a threshold that admits it would make every arm look
# like it is holding something.
CONTACT_FORCE_EPSILON_N = 1e-3


# ---------------------------------------------------------------------------
# Physics materials
# ---------------------------------------------------------------------------


def _self_collision_enabled() -> bool:
    import os

    return os.environ.get("DEMO12_SELF_COLLISION", "1") != "0"


def _set_collision_group(prim, group: str) -> None:
    """Put a prim into a PhysX collision group."""
    from pxr import Sdf

    if prim is None or not prim.IsValid():
        return
    prim.CreateAttribute("physxCollision:collisionGroups",
                         Sdf.ValueTypeNames.Token).Set(group)


def _filter_collision_pair(stage, group_a: str, group_b: str) -> None:
    """Declare that two collision groups must not collide.

    Not implemented. PhysX declares ``physxScene:collisionGroups`` as a *token-array
    relationship*, and ``pxr.Usd.Relationship`` only supports path targets -- writing it
    raises "Cannot map <> to layer". The pads do not need this: they spawn clear of the
    cell and ride at the end of the chain, so they never reach the arm's own links. Kept as
    the place to add it if a future layout does put them in contact.
    """
    return None


def _keeps_collision(path: str) -> bool:
    """Whether this collider's shipped geometry is kept, per ARM_BODIES_WITH_COLLISION."""
    segments = {s.lower() for s in path.replace("-", "_").split("/")}
    if any(s.startswith("pad_") for s in segments):
        # Authored by _author_jaw_pads. These are the gripper's real collision and must
        # stay on; the asset's own gripper geometry is what gets switched off.
        return True
    return bool(segments & set(ARM_BODIES_WITH_COLLISION))


def _enable_contact_reporting(prim) -> None:
    """Turn on PhysX contact reporting for a prim.

    Needed on *both* sides of a pair, not just the side being measured. A sensor on the
    robot's links and a part that carries no reporter can produce real contact and still
    report zero force, which is worse than no instrument at all: it says "not touching"
    about a part that is visibly being pushed.
    """
    from pxr import PhysxSchema

    if prim is None or not prim.IsValid():
        return
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(
        CONTACT_REPORT_THRESHOLD_N)


def _bind_material(stage, prim_path: str, material: M.Material, name: str) -> None:
    """Bind a real PhysX material to a collision prim.

    PhysX resolves contact friction from *both* surfaces, so a friction value that is only
    written in a Python dataclass changes nothing at all. This writes a
    ``UsdPhysics.MaterialAPI`` onto a scope and points the collider at it.
    """
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    mat_path = Sdf.Path(f"/World/PhysicsMaterials/{name}")
    if not stage.GetPrimAtPath(mat_path).IsValid():
        UsdGeom.Scope.Define(stage, mat_path.GetParentPath())
        prim = UsdShade.Material.Define(stage, mat_path).GetPrim()
        api = UsdPhysics.MaterialAPI.Apply(prim)
        api.CreateStaticFrictionAttr().Set(float(material.static_friction))
        api.CreateDynamicFrictionAttr().Set(float(material.dynamic_friction))
        api.CreateRestitutionAttr().Set(float(material.restitution))
        PhysxSchema.PhysxMaterialAPI.Apply(prim)

    from omni.physx.scripts.physicsUtils import add_physics_material_to_prim

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    add_physics_material_to_prim(stage, prim, mat_path)


def _prepare_finger_colliders(stage, prim_root: str, material: M.Material) -> tuple[int, int]:
    """Prepare the SO-101 colliders: pad material, contact reporting, collision filtering.

    Three separate things that a USD asset does not give you for free.

    **Material.** The shipped USD leaves its finger colliders on PhysX's default
    (mu 0.5), and at mu 0.5 the friction cone for a 28 g cube needs 0.55 N of normal force
    across two pads, which this servo does not produce at the shipped gains. At the pad's
    real mu 0.9 it needs 0.15 N.

    **Contact reporting.** ``activate_contact_sensors`` is a field on the *rigid-object*
    and *articulation* spawners, but ``UsdFileCfg`` calls ``spawn_from_usd``, which never
    reads it. A USD asset therefore arrives with no contact reporters, and a
    ``ContactSensor`` pointed at it fails with "could not find any bodies with contact
    reporter API".

    **Collision filtering.** See :data:`ARM_BODIES_WITH_COLLISION`.

    Takes the USD root *path* rather than the ``Articulation``: the PhysX tensor views do
    not exist yet at the point colliders are prepared (that is the whole reason this runs
    before the reset), and reading ``articulation.root_physx_view`` here returns None.

    Returns ``(finger_colliders, disabled_colliders)``.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(prim_root)
    if not root or not root.IsValid():
        raise RuntimeError(f"robot USD root is unavailable: {prim_root}")

    # The reporter goes on the *rigid body*, not on its child colliders. PhysX reports per
    # actor, and a ContactSensor pointed at the body finds nothing if the reporter was
    # applied one level down: the colliders under `jaw/collisions` are an Xform that is not
    # itself an actor.
    bodies: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        # Every link, not just the fingers: the grasp geometry is only debuggable if a
        # contact anywhere in the arm is visible. Restricting the reporter to the jaw is
        # how a descent can push a cube 31 mm while reporting zero contact.
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(
            CONTACT_REPORT_THRESHOLD_N)
        bodies.append(prim.GetPath().pathString)

    colliders: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = prim.GetPath().pathString
        if not _keeps_collision(path):
            # Everything from the wrist inwards. Its shipped collision is replaced by the
            # authored pads. See ARM_BODIES_WITH_COLLISION.
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)
        colliders.append(path)

    _filter_collision_pair(stage, PAD_COLLISION_GROUP, ARM_COLLISION_GROUP)

    if not bodies:
        raise RuntimeError(
            f"no gripper finger rigid body found under {prim_root}; without contact "
            f"reporting the demo cannot measure whether a grasp is real"
        )
    kept = [c for c in colliders if _keeps_collision(c)]
    if not colliders:
        raise RuntimeError(
            f"no collision prims found under {prim_root}; the arm would fall through "
            f"the bench"
        )
    print(f"    collision: {len(kept)} arm collider(s) kept, "
          f"{len(colliders) - len(kept)} gripper collider(s) replaced by pads",
          flush=True)
    return len(kept), len(colliders) - len(kept)


def _yaw_only_quat(quat):
    """The yaw of a wxyz quaternion as a pure-yaw quaternion, pitch and roll removed."""
    import math

    import torch

    w, x, y, z = (float(v) for v in quat[0])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw / 2.0
    out = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]],
                       dtype=torch.float32, device=quat.device)
    return out


def _segment_matches(path: str, name: str) -> bool:
    return name.lower() in {s.lower() for s in path.replace("-", "_").split("/")}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _tray_shapes(cfg: BinConfig) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """A tray as one floor plus four walls, in the parent's local frame.

    A receptacle has to be hollow. A solid coloured slab is a zone painted on the table
    and nothing can be put into it, which is exactly what Demo 09 shipped and what Demo 10
    then had to fix.
    """
    w, d = cfg.inner_m
    wall = cfg.wall_m
    floor = cfg.floor_m
    height = cfg.height_m
    outer_w = w + 2 * wall
    outer_d = d + 2 * wall

    shapes = [
        # floor slab, spanning the whole footprint
        ("Floor", (outer_w, outer_d, floor), (0.0, 0.0, floor / 2.0)),
    ]
    # Walls, centred so each one sits on the floor and stands to the rim.
    wall_h = height - floor
    z = floor + wall_h / 2.0
    shapes += [
        ("WallPosX", (wall, outer_d, wall_h), (outer_w / 2.0 - wall / 2.0, 0.0, z)),
        ("WallNegX", (wall, outer_d, wall_h), (-(outer_w / 2.0 - wall / 2.0), 0.0, z)),
        ("WallPosY", (w, wall, wall_h), (0.0, outer_d / 2.0 - wall / 2.0, z)),
        ("WallNegY", (w, wall, wall_h), (0.0, -(outer_d / 2.0 - wall / 2.0), z)),
    ]
    return shapes


# ---------------------------------------------------------------------------
# State read out of the simulation
# ---------------------------------------------------------------------------


@dataclass
class PartState:
    """What the simulation currently says about one part."""

    id: str
    x: float
    y: float
    z: float
    yaw_deg: float
    lin_speed: float
    ang_speed: float
    held_by: str | None = None

    @property
    def resting(self) -> bool:
        """Settled: slow enough to be measuring a resting contact, not catching it mid-air."""
        return self.lin_speed < 0.02 and self.ang_speed < 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "z": round(self.z, 5),
            "yawDeg": round(self.yaw_deg, 1),
            "linSpeed": round(self.lin_speed, 4),
            "resting": self.resting,
            "heldBy": self.held_by,
        }


@dataclass
class RobotState:
    """What the simulation currently says about one arm."""

    id: str
    base_x: float
    base_y: float
    tcp_x: float
    tcp_y: float
    tcp_z: float
    joint_pos_rad: list[float]
    joint_target_rad: list[float]
    jaw_rad: float
    gripping: bool
    holding: str | None
    status: str = "idle"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "baseXy": [round(self.base_x, 4), round(self.base_y, 4)],
            "tcp": [round(self.tcp_x, 5), round(self.tcp_y, 5), round(self.tcp_z, 5)],
            "jointPosRad": [round(v, 4) for v in self.joint_pos_rad],
            "jointTargetRad": [round(v, 4) for v in self.joint_target_rad],
            "jawRad": round(self.jaw_rad, 4),
            "gripping": self.gripping,
            "holding": self.holding,
            "status": self.status,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------


class IsaacCell:
    """A built Isaac Lab scene plus the handles needed to drive and read it.

    Construction order matters and is deliberate:

    1. Create the ``SimulationContext`` (this starts nothing; the worker already ran Kit).
    2. Spawn floor, table, trays, parts and robots.
    3. Bind physics materials to every collider that matters.
    4. ``reset()`` once, so PhysX owns everything from the very first step.
    5. Let the scene settle, so nothing starts mid-air.

    After step 5 the only way to change a body is to send a robot a joint target. There is
    no ``write_root_pose_to_sim`` anywhere in this class, on purpose.
    """

    def __init__(self, scene: SceneConfig, device: str = "cuda:0", headless: bool = True):
        import isaaclab.sim as sim_utils

        self.scene = scene
        self.device = device
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=M.PHYSICS_DT_S,
                device=device,
                render_interval=1,
                # The exact SI value, not PhysX's rounded default. A 0.003 m/s^2 error is
                # small but it is an error, and this demo quotes masses and forces.
                gravity=(0.0, 0.0, -M.GRAVITY_M_S2),
            )
        )
        self.sim_dt = self.sim.get_physics_dt()

        # PhysxSceneAPI settings have to be in place *before* PhysX initialises, i.e.
        # before any asset is spawned. Writing them afterwards re-creates the simulation
        # view and every tensor view an asset already captured goes stale, which shows up
        # much later as "Failed to get root link transforms from backend".
        self._tune_physics_scene()

        self.robots: dict[str, Any] = {}          # id -> Articulation
        self.robot_cfgs: dict[str, RobotConfig] = {}
        self.parts: dict[str, Any] = {}           # id -> RigidObject
        self.part_cfgs: dict[str, PartConfig] = {}
        self.bins: dict[str, BinConfig] = {}

        self._grip_body: dict[str, int] = {}
        self._jaw_body: dict[str, int] = {}
        self._articulation_cfgs: dict[str, Any] = {}
        self._pad_joint_ids: dict[str, list[int]] = {}
        self._pad_body_ids: dict[str, list[int]] = {}
        self._arm_joint_ids: dict[str, list[int]] = {}
        self._grip_joint_id: dict[str, int] = {}
        self._root_quat_inv: dict[str, Any] = {}
        self._root_pos: dict[str, Any] = {}
        self._home_joints: dict[str, Any] = {}

        self.sim_time_s = 0.0
        self.step_count = 0
        self.control_step_count = 0
        self.pad_material_bound = 0
        self._ccd_enabled = False

        self._build()

    # -- construction ----------------------------------------------------

    def _build(self) -> None:
        self._spawn_floor()
        self._spawn_table()
        for b in self.scene.bins:
            self._spawn_bin(b)
        for p in self.scene.parts:
            self._spawn_part(p)
        for r in self.scene.robots:
            self._spawn_robot(r)

        # Before the reset, with everything else: a RigidObject captures its PhysX tensor
        # views at construction, and ``sim.reset()`` is what creates them. Creating the pads
        # afterwards leaves them without views and the first pose write raises
        # "'RigidObject' object has no attribute '_ALL_INDICES'".
        self._author_jaw_pads()
        self._rebuild_with_linkage_actuators()

        # Materials are bound *before* the reset, and that ordering is load-bearing.
        # Writing a physics material onto a prim under an articulation fires Isaac Lab's
        # stage-change callback, which drops that asset's PhysX tensor views. The reset
        # that follows replays the timeline and rebuilds every view cleanly. Bind them
        # afterwards and the views stay stale, and the first read of any articulation
        # raises "Failed to get root link transforms from backend".
        self._bind_all_materials()
        self._build_contact_sensors()

        # Also load-bearing: ``reset()`` is what makes the USD's collision prims real and
        # is what runs Isaac Lab's initialise callbacks. The contact sensors must already
        # exist by the time it does, or they are left without buffers
        # ("ContactSensor object has no attribute '_is_outdated'").
        self.sim.reset()
        self.sim.forward()

        self._apply_actuator_gains()
        self._capture_robot_frames()
        self._build_arms()

    def _build_contact_sensors(self) -> None:
        """Report what each arm's fingers are actually touching, and how hard.

        This is the demo's measurement of a grasp. Without it, "the jaws are closed" and
        "the jaws are gripping something" are indistinguishable, and the second is the
        only one that matters.

        Two constraints found the hard way, both from the sensor refusing to initialise:

        * The path has to resolve to a *rigid body*. The finger colliders live under a
          ``jaw/collisions`` Xform that is not itself a body, so pointing the sensor at it
          fails with "did not match any rigid bodies". The body is ``jaw``.
        * ``filter_prim_paths_expr`` has to match exactly one body. A ``Parts/*`` filter
          in a four-part cell is rejected outright ("expected 1, found 4"), so no filter is
          used; the sensor reports total force on the fingers, and the orchestrator pairs
          that with the part it is actually holding.
        """
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        # Isaac Lab disables PhysX contact processing during SimulationContext setup
        # ("/physics/disableContactProcessing") to save overhead, with the note that it
        # "will be enabled when a contact sensor is created" -- but that note describes
        # Isaac Lab's own InteractiveScene, which re-applies it. Here the sensors are
        # built by hand, so without this the contacts are real and unreported: a descent
        # that shoves a cube 88 mm reads as zero force on every link.
        self.sim.set_setting("/physics/disableContactProcessing", False)

        self.contact_sensors: dict[tuple[str, str], Any] = {}
        for robot_id in self.robots:
            for body in MONITORED_BODIES:
                self.contact_sensors[(robot_id, body)] = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{ROOT}/Robots/{robot_id}/{body}",
                        track_pose=False,
                        track_contact_points=False,
                        track_air_time=False,
                        max_contact_data_count_per_prim=32,
                        filter_prim_paths_expr=[],
                    )
                )

        # One sensor per pad. The pads are separate rigid bodies joined to the jaw by
        # prismatic joints, so unlike the earlier rigidly-parented pair each can be sensed
        # directly -- and that is what makes "the jaws closed" and "the jaws are gripping"
        # distinguishable.
        #
        # No filter: a Parts/* filter has to match exactly one body and this cell has
        # several, so each sensor reports the total force on one pad.
        self.pad_sensors: dict[tuple[str, str], Any] = {}
        for robot_id in self.robots:
            for tag in ("L", "R"):
                self.pad_sensors[(robot_id, tag)] = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"{ROOT}/Robots/{robot_id}/pad_{tag}",
                        track_pose=False,
                        track_contact_points=False,
                        track_air_time=False,
                        max_contact_data_count_per_prim=32,
                        filter_prim_paths_expr=[],
                    )
                )

    def _sensor(self, robot_id: str, body: str):
        return self.contact_sensors[(robot_id, body)]

    def _author_jaw_pads(self) -> None:
        """Author the finger pads as a **two-prismatic-joint linkage** hanging off the jaw.

        This is the design that works, and getting here took three others to failure. All
        three failures are recorded at the bottom of this docstring because each one rules
        out an approach somebody will otherwise try.

        **The gripper is a linkage: one pad rigidly on each side is not enough, because a
        rigid body cannot change shape.** The pads here are two separate rigid bodies, each
        connected to the jaw by a prismatic joint along the jaw's local Y, so they slide in
        and out under a real joint drive. That means:

        * PhysX simulates them as part of the arm's own articulation, so they are never a
          separate body that can fall asleep and stop responding;
        * the closing force comes from the joint drive, which is a real force between two
          real bodies -- there is no force model here and no attachment of any kind;
        * contact is ordinary contact, so a part that is too heavy or too slippery simply
          slips, and the demo reports that it did.

        The ``gripper`` joint is left doing what the asset does with it. It was measured,
        not assumed, to rotate about the jaw's local Z: sweeping the joint and differencing
        the jaw quaternion gives ``(-0.0000, -0.0000, +1.0000)`` at every sample.

        Geometry, in the jaw's own frame. **Measured** at the grasp pose
        (``out12/pose_probe.json``, 28 poses): the jaw's world directions of its local axes
        are local X -> (0.571, 0.821, -0.032), local Y -> (0.018, 0.027, +0.9995), local Z
        -> (0.821, -0.571, -0.000005). So the jaw frame is world-aligned with **local Y
        pointing up**, and local X and Z are horizontal.

        The pads therefore slide along local Z (:data:`materials.JAW_PAD_AXIS`), whose
        world Z component is -5e-06 at every pose, and a pair mirrored about it closes
        *sideways* while staying level.

        An earlier build slid them along local **Y**, on the strength of a prose claim in
        DEMO12_HANDOFF.md that "local Y is world -Y (horizontal)". That claim was wrong --
        local Y is vertical -- so the pads closed straight up and down, pressed on top of
        the part, and produced no grip. It is recorded here because it is the single most
        expensive error in the demo's history and it was invisible until it was measured.

        -- three that were built and measured first --

        * **Pads rigidly parented to the jaw.** Impossible, and the proof is short: a
          revolute preserves every point's distance from its axis and turns every point by
          the same angle, so the distance between any two points on a rigid body is
          invariant. Measured 42.74 mm at every angle from 0.2 to 1.4 rad, constant to the
          millimetre. (An earlier version of this file derived a separation of ``2 R
          cos(theta)`` and treated it as a closing span. That describes the direction the
          pads point, not how far apart they are, and it was wrong.)
        * **Pads as standalone bodies, kinematic.** Closing them across a part means 6 mm
          of interference and PhysX fires the part out of the grip at 0.62 m/s. Position
          control on an infinitely massive body produces no normal force, so there is
          nothing for friction to work with.
        * **Pads as standalone bodies, dynamic.** Driven by velocity they track to 0.5 mm
          against a 10 mm target, but a velocity command every tick drives them through the
          contact they are gripping, so it never develops and the sensor reads 0.00 N.
          Driven by force instead they track to 1.70 mm and then stop being simulated
          altogether -- not by the force term, not by a pose write, and not by
          ``sleepThreshold``, which was swept from -1 to 1000 and changed the frozen state
          by exactly zero. Only a velocity write revives a pad, and that is the very thing
          that breaks the contact.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        half = M.JAW_PAD_CLOSED_HALF_M
        axis = M.JAW_PAD_AXIS
        drop_axis = M.JAW_PAD_DROP_AXIS
        for robot_id in self.robots:
            for side, tag in ((-1.0, "L"), (1.0, "R")):
                # The pad is its own rigid body, a sibling of the arm's own bodies so it
                # joins the articulation.
                path = f"{ROOT}/Robots/{robot_id}/pad_{tag}"
                cube = UsdGeom.Cube.Define(self.sim.stage, path)
                cube.CreateSizeAttr(1.0)
                # Scale is (depth along X, height along Y, thickness along Z).
                #
                # The thin dimension must lie on JAW_PAD_AXIS, because that is the
                # separation axis and it is the faces the part is squeezed between. The
                # 10 mm height goes on Y because that axis is vertical at the grasp pose
                # (measured, out12/pose_probe.json), so the pads stay level as they close;
                # the long 18 mm depth goes on X, which is horizontal.
                #
                # The previous build scaled this (height, thickness, depth), which was
                # correct for an X separation axis and is wrong for a Z one.
                scale = {
                    "X": Gf.Vec3f(M.FINGER_PAD_THICK_M, M.FINGER_PAD_HEIGHT_M,
                                  M.FINGER_PAD_DEPTH_M),
                    "Y": Gf.Vec3f(M.FINGER_PAD_DEPTH_M, M.FINGER_PAD_THICK_M,
                                  M.FINGER_PAD_HEIGHT_M),
                    "Z": Gf.Vec3f(M.FINGER_PAD_DEPTH_M, M.FINGER_PAD_HEIGHT_M,
                                  M.FINGER_PAD_THICK_M),
                }[axis]
                cube.AddScaleOp().Set(scale)
                cube.CreateExtentAttr([
                    Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(
                    M.FINGER_PAD_MASS_KG)

                # The slide. Axis is JAW_PAD_AXIS, signed so that positive travel always
                # *opens* the gripper: the two pads are mirror images.
                jpath = f"{ROOT}/Robots/{robot_id}/joints/pad_{tag}_slide"
                joint = UsdPhysics.PrismaticJoint.Define(self.sim.stage, jpath)
                joint.CreateBody0Rel().SetTargets(
                    [f"{ROOT}/Robots/{robot_id}/jaw"])
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateAxisAttr().Set(axis)
                # The anchor on the jaw: the pads hang ``JAW_PAD_DROP_M`` down the drop
                # axis and are mirrored either side of it along the separation axis. The
                # mirror offset must sit on the separation axis, or the two pads are offset
                # from each other along a direction they cannot slide in and the span is
                # wrong.
                local0 = Gf.Vec3f(0.0, 0.0, 0.0)
                if drop_axis == "Y":
                    local0[1] = M.JAW_PAD_DROP_M
                elif drop_axis == "X":
                    local0[0] = M.JAW_PAD_DROP_M
                if axis == "Z":
                    local0[2] = side * half
                elif axis == "Y":
                    local0[1] = side * half
                else:
                    local0[0] = side * half
                joint.CreateLocalPos0Attr().Set(local0)
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                pj = UsdPhysics.PrismaticJoint(joint)
                if side > 0:
                    pj.CreateLowerLimitAttr().Set(0.0)
                    pj.CreateUpperLimitAttr().Set(M.JAW_PAD_TRAVEL_M)
                else:
                    pj.CreateLowerLimitAttr().Set(-M.JAW_PAD_TRAVEL_M)
                    pj.CreateUpperLimitAttr().Set(0.0)

        print(f"  finger linkage: {2 * len(self.robots)} pads on prismatic joints "
              f"along jaw local {axis} (inner-face span "
              f"{2000 * M.JAW_PAD_CLOSED_HALF_M - 1000 * M.FINGER_PAD_THICK_M:.1f} mm shut, "
              f"{2000 * (M.JAW_PAD_CLOSED_HALF_M + M.JAW_PAD_TRAVEL_M) - 1000 * M.FINGER_PAD_THICK_M:.1f} mm "
              f"open, travel {1000 * M.JAW_PAD_TRAVEL_M:.0f} mm, "
              f"drop {1000 * M.JAW_PAD_DROP_M:.0f} mm on local {drop_axis})",
              flush=True)

    def _rebuild_with_linkage_actuators(self) -> None:
        """Rebuild every articulation so the authored linkage joints have actuators.

        Isaac Lab processes ``ArticulationCfg.actuators`` when the ``Articulation`` object
        is constructed, and the linkage joints do not exist until after that. So the
        actuators have to be registered on a second pass.

        This is not a shortcut. Measured: authoring a prismatic joint into the articulation
        and then driving it with ``set_joint_position_target`` left it at exactly
        -0.00000 for 180 ticks. Writing the gains straight to the PhysX view afterwards
        changed nothing either, because ``apply_actuator_model`` only writes forces for
        joints that have a registered actuator. The joint was there all along -- Isaac Lab
        listed it as the arm's 7th joint -- it just had no actuator behind it.
        """
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation

        for robot_id, act in self._articulation_cfgs.items():
            actuators = dict(act.actuators or {})
            pad_cfg = ImplicitActuatorCfg(
                joint_names_expr=["pad_L_slide", "pad_R_slide"],
                effort_limit_sim=M.JAW_PAD_EFFORT_LIMIT_N,
                velocity_limit_sim=M.JAW_PAD_VELOCITY_LIMIT_M_S,
                stiffness=M.JAW_PAD_ACTUATOR_STIFFNESS,
                damping=M.JAW_PAD_ACTUATOR_DAMPING,
            )
            # Front of the dict: Isaac Lab takes the first matching entry.
            act.actuators = {"demo12-finger-linkage": pad_cfg, **actuators}
            self.robots[robot_id] = Articulation(act)

        print(f"  finger linkage actuated: {len(self.robots)} arm(s) x 2 joints, "
              f"kp={M.JAW_PAD_ACTUATOR_STIFFNESS} "
              f"kd={M.JAW_PAD_ACTUATOR_DAMPING}", flush=True)
        # Joint and body ids are resolved in _capture_robot_frames, not here: a fresh
        # Articulation has no PhysX views until sim.reset(), and reading them before that
        # raises "'Articulation' object has no attribute '_root_physx_view'".

    def set_fingers(self, robot_id: str, open_: bool) -> None:
        """Open or shut this arm's finger pads by commanding the linkage joints.

        This is the whole gripper interface, and it is what a real controller would do:
        a position target on a joint. No pose is written, no force is applied by hand, and
        nothing about the part is touched.
        """
        import torch

        travel = M.JAW_PAD_TRAVEL_M if open_ else 0.0
        ids = self._pad_joint_ids[robot_id]
        # Right pad opens in +Y, left pad opens in -Y (see _author_jaw_pads).
        target = torch.tensor([[-travel, travel]], dtype=torch.float32,
                              device=self.device)
        self.robots[robot_id].set_joint_position_target(
            target, joint_ids=list(ids))

    def jaw_pad_positions(self, robot_id: str) -> tuple[Any, Any]:
        """World positions of this arm's two pads, read from the articulation."""
        art = self.robots[robot_id]
        ids = self._pad_body_ids[robot_id]
        return (art.data.body_pos_w[0, ids[0]].clone(),
                art.data.body_pos_w[0, ids[1]].clone())

    def jaw_pad_span_m(self, robot_id: str) -> float:
        """Measured inner-face separation of the rigid pads, in metres.

        Projected onto the jaw's own separation axis (:data:`materials.JAW_PAD_AXIS`), not
        on a world axis: the jaw yaws with ``shoulder_pan``, so the span has to be read in
        the tool's frame or it is measured across the wrong direction.
        """
        import torch
        from isaaclab.utils.math import quat_apply

        a, b = self.jaw_pad_positions(robot_id)
        d = b - a
        art = self.robots[robot_id]
        quat = art.data.body_quat_w[0, self._jaw_body[robot_id]]
        unit = {"X": [1.0, 0.0, 0.0],
                "Y": [0.0, 1.0, 0.0],
                "Z": [0.0, 0.0, 1.0]}[M.JAW_PAD_AXIS]
        axis = quat_apply(quat, torch.tensor([unit],
                                             dtype=torch.float32,
                                             device=d.device))[0]
        return float((d * axis).abs().sum()) - M.FINGER_PAD_THICK_M

    def _jaw_open_fraction(self, robot_id: str) -> float:
        """0 when the jaws are commanded shut, 1 when fully open, from the real joint."""
        art = self.robots[robot_id]
        joint_id = self._grip_joint_id[robot_id]
        angle = float(art.data.joint_pos[0, joint_id].item())
        return (angle - M.GRIPPER_CLOSE_RAD) / max(
            1e-6, M.GRIPPER_OPEN_RAD - M.GRIPPER_CLOSE_RAD)

    def _pad_span_for(self, robot_id: str) -> float:
        """Separation of this arm's pads right now.

        Now a *measurement* of the rigid pads rather than a schedule: the pads are on the
        jaw body and the jaw joint moves them, so the span is whatever the geometry says.
        Nothing here decides where the pads are.
        """
        return self.jaw_pad_span_m(robot_id)

    def pad_span_m(self, robot_id: str) -> float:
        """Inner-face separation of this arm's pads right now, in metres."""
        return self.jaw_pad_span_m(robot_id)

    def finger_contact_forces(self, robot_id: str, body: str = "jaw") -> list[dict[str, Any]]:
        """Live contact force on the colliders of one of this arm's links.

        Empty when the link touches nothing, which is the common case and is the honest
        answer. ``net_forces_w`` is (bodies, envs, 3) for the current step; the per-step
        history lives in ``net_forces_w_history``, and indexing the history as if it were
        the current step is off by two axes.
        """
        forces = self._sensor(robot_id, body).data.net_forces_w
        out: list[dict[str, Any]] = []
        for i in range(forces.shape[0]):
            f = forces[i, 0]
            mag = float((f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5)
            if mag > CONTACT_FORCE_EPSILON_N:
                out.append({
                    "body": body,
                    "forceN": round(mag, 4),
                    "fx": round(float(f[0]), 4),
                    "fy": round(float(f[1]), 4),
                    "fz": round(float(f[2]), 4),
                })
        return out

    def pad_contact_forces(self, robot_id: str) -> list[dict[str, Any]]:
        """Contact force on each finger pad, via that pad's own sensor.

        The pads are separate rigid bodies joined to the jaw by prismatic joints, so unlike
        the earlier rigidly-parented pair they can each be sensed directly. This is the
        demo's measurement of a grasp: without it, "the jaws closed" and "the jaws are
        gripping" are indistinguishable, and only the second one matters.
        """
        out: list[dict[str, Any]] = []
        for tag in ("L", "R"):
            sensor = self.pad_sensors.get((robot_id, tag))
            if sensor is None:
                continue
            forces = sensor.data.net_forces_w
            total = 0.0
            for i in range(forces.shape[0]):
                f = forces[i, 0]
                total += float((f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5)
            if total > CONTACT_FORCE_EPSILON_N:
                out.append({"body": f"finger_{tag}", "forceN": round(total, 4)})
        return out

    def grip_force_n(self, robot_id: str) -> float:
        """Total force the two pads are exerting right now, in newtons.

        Zero means the pads are touching nothing, whatever the linkage is commanded to do; a
        value in newtons is a real squeeze, reported by PhysX rather than invented here.
        """
        return round(sum(c["forceN"] for c in self.pad_contact_forces(robot_id)), 4)

    def link_contacts(self, robot_id: str) -> dict[str, float]:
        """Total contact force on every monitored link, plus the two pads.

        This is the grasp diagnostic. Everything except the pads is zero while the arm is
        moving, because the asset's gripper collision is off; the pad entries are the ones
        that say whether anything is actually being held.
        """
        contacts = {
            body: round(sum(c["forceN"] for c in
                            self.finger_contact_forces(robot_id, body)), 4)
            for body in MONITORED_BODIES
        }
        by_body = {c["body"]: c["forceN"] for c in self.pad_contact_forces(robot_id)}
        contacts["finger_L"] = by_body.get("finger_L", 0.0)
        contacts["finger_R"] = by_body.get("finger_R", 0.0)
        return contacts

    def _build_arms(self) -> None:
        from .robot import ArmController

        self.arms: dict[str, ArmController] = {
            robot_id: ArmController(self, robot_id) for robot_id in self.robots
        }

    def arm(self, robot_id: str):
        try:
            return self.arms[robot_id]
        except KeyError:
            raise KeyError(f"no robot with id {robot_id!r} in this cell") from None

    def _apply_actuator_gains(self) -> None:
        """Re-apply the actuator gains *after* the reset.

        ``Articulation._process_actuators_cfg`` writes each implicit actuator's stiffness,
        damping, effort limit and velocity limit into PhysX while the asset is being
        constructed -- i.e. before the reset. The reset re-initialises PhysX, which puts the
        joint drives back to zero, so every joint target set afterwards is silently ignored
        and the arm holds its spawn pose forever. That looks like a robot that refuses to
        move rather than like a config that was thrown away, so it is worth stating.

        Re-applying is idempotent and cheap: six joints per arm.
        """
        for robot_id, art in self.robots.items():
            for actuator in art.actuators.values():
                ids = actuator.joint_indices
                art.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=ids)
                art.write_joint_damping_to_sim(actuator.damping, joint_ids=ids)
                art.write_joint_effort_limit_to_sim(actuator.effort_limit_sim, joint_ids=ids)
                art.write_joint_velocity_limit_to_sim(actuator.velocity_limit_sim, joint_ids=ids)
        print(f"  actuator gains re-applied after reset: arm kp={M.ARM_ACTUATOR_STIFFNESS} "
              f"kd={M.ARM_ACTUATOR_DAMPING}, gripper kp={M.GRIPPER_ACTUATOR_STIFFNESS} "
              f"kd={M.GRIPPER_ACTUATOR_DAMPING}", flush=True)

    def _tune_physics_scene(self) -> None:
        """Global solver settings, applied to ``/physicsScene`` after the reset.

        Set here rather than per-asset because the contact that decides whether this demo
        works is between two *different* assets: a finger pad and a part. Tuning only the
        robot leaves the table and the trays on PhysX's defaults, which is where the
        interpenetration comes from.

        Written as plain attributes on the prim rather than through the typed schema
        helpers: ``UsdPhysics.Scene``/``PhysxSceneAPI`` do not agree on which of ``Apply``
        and ``Get`` they expose across USD builds, and an attribute write is the same
        thing either way.
        """
        from pxr import Sdf, UsdPhysics

        prim = self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)
        if not prim or not prim.IsValid():
            return

        def put(name: str, type_name, value) -> None:
            # SimulationContext has already applied PhysxSceneAPI, so these
            # attributes exist with their authored types (broadphaseType is an
            # enum, not an int). Write through the existing attribute when it
            # is there and only create one when it is not.
            attr = prim.GetAttribute(name)
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(name, type_name)
            attr.Set(value)

        vec3 = Sdf.ValueTypeNames.Float3
        put("physxScene:gravityDirection", vec3, (0.0, 0.0, -1.0))

        # A fresh stage has no GPU broadphase until the first step, so ask for it here.
        # ``physxScene:broadphaseType`` is a token enum, not an int: GPU / SAP / MBP.
        put("physxScene:broadphaseType", Sdf.ValueTypeNames.Token, "GPU")
        put("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, True)

        # Contact offsets. PhysX defaults these to 5% of a shape's extent, which does not
        # resolve a millimetre-thick finger pad at all. Absolute and sized for this cell:
        # 1 mm contact offset against 0.5 mm rest offset means a pad touches within a
        # millimetre and is not resting on anything until it overlaps by half a millimetre.
        put("physxScene:contactOffset", Sdf.ValueTypeNames.Float, 0.001)
        put("physxScene:restOffset", Sdf.ValueTypeNames.Float, 0.0005)

        # Global solver iterations, raised from the default 4. This is the second half of
        # the answer to contact creep; the first is the 240 Hz step in materials.py.
        put("physxScene:solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32)
        put("physxScene:solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 1)

        # Continuous collision detection. A part that slips out of a grasp is the one body
        # here that moves fast relative to a thin pad, and discrete detection tunnels.
        put("physxScene:enableCCD", Sdf.ValueTypeNames.Bool, True)

        self._ccd_enabled = bool(prim.GetAttribute("physxScene:enableCCD").Get())
        print(f"  physics scene: dt={self.sim_dt * 1000:.2f} ms  contact=1.0 mm  "
              f"rest=0.5 mm  solver=32/1  ccd={self._ccd_enabled}", flush=True)

    def _spawn_floor(self) -> None:
        """A static ground slab, so a part that leaves the table lands somewhere observable.

        Without it a dropped part free-falls forever and every subsequent read of its
        position is meaningless. It is kinematic: the robots are bolted to the bench, and
        so is the bench to the floor.

        Deliberately *not* Isaac Lab's ``GroundPlaneCfg``, which fetches a USD from the
        NVIDIA Nucleus server on first use. A demo that needs the network to build its own
        floor is a demo that fails on a plane, and it is a 100 m plane we do not want.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        thickness = 0.02
        cfg = RigidObjectCfg(
            prim_path=f"{ROOT}/Floor",
            spawn=sim_utils.CuboidCfg(
                size=(6.0, 6.0, thickness),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.16, 0.17, 0.19), roughness=0.9),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, -0.35, -thickness / 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.floor = RigidObject(cfg)

    def _spawn_table(self) -> None:
        """The workbench: a static slab with the measured melamine top height."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        t = self.scene.table
        thickness = M.TABLE_TOP_Z_M
        cfg = RigidObjectCfg(
            prim_path=f"{ROOT}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(t.size_m[0], t.size_m[1], thickness),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.72, 0.70, 0.66), roughness=0.6),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(t.center_xy[0], t.center_xy[1], thickness / 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.table = RigidObject(cfg)
        self.table_surface_z = M.TABLE_TOP_Z_M

    def _spawn_bin(self, cfg: BinConfig) -> None:
        """One hollow tray: a static compound collider of a floor and four walls.

        Static rather than a dynamic body on purpose. A receptacle bolted to the bench is
        what this cell is, and a tray the robots can knock over is not a destination. A
        dynamic tray would also add a mass the orchestration then has to compensate for.
        Documented as an assumption, not a measurement.
        """
        import isaacsim.core.utils.prims as prim_utils
        import isaaclab.sim as sim_utils
        from isaaclab.sim import schemas
        from isaaclab.sim.spawners.shapes import shapes as shape_spawners

        path = f"{ROOT}/Bins/{cfg.id}"
        prim_utils.create_prim(
            path, prim_type="Xform",
            translation=(cfg.x, cfg.y, M.TABLE_TOP_Z_M),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        schemas.define_rigid_body_properties(
            path, sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True))
        schemas.define_mass_properties(path, sim_utils.MassPropertiesCfg(mass=0.18))

        for name, size, offset in _tray_shapes(cfg):
            child = sim_utils.CuboidCfg(
                size=size,
                rigid_props=None,
                mass_props=None,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=_rgb(cfg.color), roughness=0.55),
            )
            shape_spawners.spawn_cuboid(
                f"{path}/{name}", child, translation=offset, orientation=(1.0, 0.0, 0.0, 0.0))

        self.bins[cfg.id] = cfg

    def _spawn_part(self, cfg: PartConfig) -> None:
        """One part, with the mass its geometry and material actually give."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        mat = cfg.material_spec
        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                linear_damping=mat.linear_damping,
                angular_damping=mat.angular_damping,
                # A rigid body with no explicit mass is given one by PhysX from its shape,
                # which would be a solid block of the default density. The real mass is
                # declared instead.
                disable_gravity=False,
            ),
            "mass_props": sim_utils.MassPropertiesCfg(mass=cfg.mass_kg),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(
                diffuse_color=_rgb(cfg.color), roughness=0.45),
        }
        sx, sy, sz = cfg.size_m
        if cfg.kind == "ball":
            spawn = sim_utils.SphereCfg(radius=sx / 2.0, **common)
        elif cfg.kind == "cylinder":
            spawn = sim_utils.CylinderCfg(radius=sx / 2.0, height=sz, **common)
        else:
            spawn = sim_utils.CuboidCfg(size=cfg.size_m, **common)

        yaw = math.radians(cfg.yaw_deg)
        quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        obj_cfg = RigidObjectCfg(
            prim_path=f"{ROOT}/Parts/{cfg.id}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(cfg.x, cfg.y, self.table_surface_z + sz / 2.0),
                rot=quat),
        )
        self.parts[cfg.id] = RigidObject(obj_cfg)
        self.part_cfgs[cfg.id] = cfg

    def _spawn_robot(self, cfg: RobotConfig) -> None:
        """One SO-101 follower with tuned actuators and a fixed base.

        The articulation itself is the unmodified LeIsaac ``so101_follower.usd``: real
        joint limits, real body masses, real inertia tensors. Only the implicit actuator
        gains are changed, because the shipped 17.8 N.m/rad droops under the arm's own
        weight by more than the grasp tolerance.
        """
        from isaaclab.assets import Articulation
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sim.schemas import JointDrivePropertiesCfg
        from leisaac.assets.robots.lerobot import (SO101_FOLLOWER_ASSET_PATH,
                                                   SO101_FOLLOWER_CFG)

        arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

        act = SO101_FOLLOWER_CFG.replace(prim_path=f"{ROOT}/Robots/{cfg.id}")
        act.init_state.pos = (cfg.base_xy[0], cfg.base_xy[1], self.table_surface_z)
        act.init_state.rot = _yaw_quat(cfg.base_yaw_deg)

        # The SO-101 USD authors NO joint drives at all -- reading it with UsdPhysics finds
        # zero PhysicsDrive prims. Without a drive, a joint position target is written to a
        # drive that does not exist, PhysX ignores it, and the arm holds its spawn pose
        # forever while every layer above reports that it is commanding motion. Declaring
        # the drives here is what makes the articulation actually controllable.
        #
        # These are the arm's gains; :meth:`_apply_actuator_gains` re-writes them per joint
        # group after the reset, which is where the gripper gets its stiffer servo.
        act.spawn.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            stiffness=M.ARM_ACTUATOR_STIFFNESS,
            damping=M.ARM_ACTUATOR_DAMPING,
            max_effort=M.STS3215_EFFORT_LIMIT_NM,
            max_velocity=M.STS3215_VELOCITY_LIMIT_RAD_S,
        )

        act.actuators = {
            "sts3215-arm": ImplicitActuatorCfg(
                joint_names_expr=arm_joints,
                effort_limit_sim=M.STS3215_EFFORT_LIMIT_NM,
                velocity_limit_sim=M.STS3215_VELOCITY_LIMIT_RAD_S,
                stiffness=M.ARM_ACTUATOR_STIFFNESS,
                damping=M.ARM_ACTUATOR_DAMPING,
            ),
            "sts3215-gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=M.STS3215_EFFORT_LIMIT_NM,
                velocity_limit_sim=M.STS3215_VELOCITY_LIMIT_RAD_S,
                stiffness=M.GRIPPER_ACTUATOR_STIFFNESS,
                damping=M.GRIPPER_ACTUATOR_DAMPING,
            ),
        }

        # Which USD to build the arm from. The shipped `so101_follower.usd` models the
        # jaw's collision as one solid shape, so the fingers cannot get around a 30 mm
        # part -- the jaw bottoms out ~40 mm above the bench and presses down on top of it.
        # This project already ships re-authored pad variants next to the original
        # (`so101_follower_demo11_pads_*.usd`, differing in pad thickness and vee angle);
        # DEMO12_ROBOT_USD selects one, and `tests/test_grasp_geometry.py` reports which
        # of them can actually straddle a part.
        override = os.environ.get("DEMO12_ROBOT_USD")
        if override:
            act.spawn.usd_path = override
        # Self-collision on: the USD default. It is what stops the arm folding through its
        # own links when the IK asks for a pose the mechanism cannot reach.
        #
        # Overridable because this asset's wrist and gripper housings interpenetrate at
        # some tool-down poses, and a self-collision there both blocks the descent and
        # loads the arm with a force that has nothing to do with the part. Set
        # DEMO12_SELF_COLLISION=0 to measure which poses are reachable without it.
        act.spawn.articulation_props.enabled_self_collisions = _self_collision_enabled()
        # Tighter solver iterations than the shipped 4/4. A narrow two-pad contact against a
        # flat face is the hardest contact in this scene and 4 iterations lets the part
        # creep while it is being held.
        act.spawn.articulation_props.solver_position_iteration_count = 16
        act.spawn.articulation_props.solver_velocity_iteration_count = 4

        self.robots[cfg.id] = Articulation(act)
        self.robot_cfgs[cfg.id] = cfg
        # Kept so the articulation can be rebuilt once the finger linkage has been authored.
        # Isaac Lab processes ``act.actuators`` when the Articulation is constructed, and the
        # linkage joints do not exist until :meth:`_author_jaw_pads` has run -- so the
        # actuators for them can only be registered on a second pass.
        self._articulation_cfgs[cfg.id] = act

    def _bind_all_materials(self) -> None:
        """Bind physics materials to every collider that will be in contact.

        Done after ``sim.reset()`` because the articulation's collision prims only exist
        once the USD has been instantiated.
        """
        stage = self.sim.stage
        _bind_material(stage, f"{ROOT}/Floor", M.Material(
            "concrete", "Suelo", "Floor", 2400.0, 0.7, 0.6, 0.05), "Demo12Floor")
        _bind_material(stage, f"{ROOT}/Table", M.TABLE_MATERIAL, "Demo12Melamine")

        # The far side of every pair the arms can touch.
        for path in (f"{ROOT}/Floor", f"{ROOT}/Table"):
            _enable_contact_reporting(stage.GetPrimAtPath(path))
        for bin_id, cfg in self.bins.items():
            for name, _size, _off in _tray_shapes(cfg):
                p = f"{ROOT}/Bins/{bin_id}/{name}"
                _bind_material(stage, p, M.ABS, "Demo12Tray")
                _enable_contact_reporting(stage.GetPrimAtPath(p))

        for part_id, cfg in self.part_cfgs.items():
            _bind_material(stage, f"{ROOT}/Parts/{part_id}",
                           cfg.material_spec, f"Demo12Part_{cfg.material}")
            _enable_contact_reporting(stage.GetPrimAtPath(f"{ROOT}/Parts/{part_id}"))

        fingers = disabled = 0
        for robot_id in self.robots:
            f, d = _prepare_finger_colliders(
                stage, f"{ROOT}/Robots/{robot_id}", M.RUBBER_PAD)
            fingers += f
            disabled += d
            # The authored pads carry the same rubber as the shipped fingers did.
            for tag in ("L", "R"):
                prim = stage.GetPrimAtPath(f"{ROOT}/Robots/{robot_id}/pad_{tag}")
                if prim and prim.IsValid():
                    _bind_material(stage, prim.GetPath().pathString, M.RUBBER_PAD,
                                   "Demo12RubberPad")
                    _enable_contact_reporting(prim)
        self.pad_material_bound = fingers
        self.disabled_colliders = disabled

    def _capture_robot_frames(self) -> None:
        """Record each base pose once, so local control frames never drift."""
        from isaaclab.utils.math import quat_inv

        for robot_id, art in self.robots.items():
            self._root_pos[robot_id] = art.data.root_pos_w.clone()
            self._root_quat_inv[robot_id] = quat_inv(art.data.root_quat_w.clone())
            ids, _ = art.find_joints(
                ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
            self._arm_joint_ids[robot_id] = list(ids)
            self._grip_joint_id[robot_id] = art.find_joints("gripper")[0][0]
            self._grip_body[robot_id] = art.find_bodies("gripper")[0][0]
            self._jaw_body[robot_id] = art.find_bodies("jaw")[0][0]
            self._pad_joint_ids[robot_id] = list(art.find_joints(
                ["pad_L_slide", "pad_R_slide"])[0])
            self._pad_body_ids[robot_id] = list(art.find_bodies(
                ["pad_L", "pad_R"])[0])
            # A home pose that folds the arm up and clear of the table, so no arm starts
            # inside a part or a tray.
            self._home_joints[robot_id] = torch_tensor(
                [[0.0, 0.60, 1.30, 0.70, 0.0, M.GRIPPER_OPEN_RAD,
                  -M.JAW_PAD_TRAVEL_M, M.JAW_PAD_TRAVEL_M]],
                art.device if hasattr(art, "device") else self.device)

    # -- settling --------------------------------------------------------

    def settle(self, steps: int = 90) -> None:
        """Let everything come to rest before the first commanded motion.

        Each arm is first walked to a retract pose above and behind its own base through
        the normal IK path, so the cell never starts with an arm folded into the table or
        into a part. Parts are spawned resting by construction, but a freshly reset PhysX
        scene needs a few frames before contacts are established; skipping that is how a
        part ends up 2 mm inside the table at t=0 and then jumps.
        """
        for robot_id in self.arms:
            arm = self.arms[robot_id]
            bx, by = self.robot_cfgs[robot_id].base_xy
            arm.move_to(self, (bx, by - 0.10, self.table_surface_z + 0.20),
                        open_=True, max_ticks=240, tol_m=0.004)
        for _ in range(steps):
            for arm in self.arms.values():
                arm.set_jaws(open_=True)
            self.step()

    # -- the one and only write path -------------------------------------

    def step(self, substeps: int = M.CELL_STEPS_PER_CONTROL_STEP) -> None:
        """Advance one control tick (four 240 Hz physics steps by default).

        No poses are written. This is the only way time moves in this demo.

        ``write_data_to_sim`` is required because there is no ``ManagerBasedEnv`` here to
        do it for us: ``set_joint_position_target`` only fills a buffer, and the implicit
        actuators read that buffer during this call. Without it the arm holds whatever
        pose it was spawned in no matter what is commanded, which looks exactly like a
        robot that is refusing to move.
        """
        # The finger pads are joints on the arm's own articulation, so ``write_data_to_sim``
        # on the articulation carries their targets with it. Nothing else has to be driven.
        for art in self.robots.values():
            art.write_data_to_sim()
        for i in range(substeps):
            self.sim.step(render=(i == substeps - 1) and self.sim.has_gui())

        # Refresh Isaac Lab's cached tensors. Without this ``asset.data`` keeps returning
        # whatever was there when the last ``update`` ran, so the orchestration reads a
        # world that has not moved and concludes the robot is not following its commands.
        control_dt = self.sim_dt * substeps
        self.sim.forward()
        for art in self.robots.values():
            art.update(control_dt)
        for obj in self.parts.values():
            obj.update(control_dt)
        for sensor in getattr(self, "contact_sensors", {}).values():
            sensor.update(control_dt)
        for sensor in getattr(self, "pad_sensors", {}).values():
            sensor.update(control_dt)

        self.sim_time_s += control_dt
        self.control_step_count += 1
        self.step_count += substeps

    def forward(self) -> None:
        """Refresh the cached tensor views without advancing time."""
        self.sim.forward()

    def render_once(self) -> None:
        """Ask PhysX to render the scene once, even without a window.

        ``step`` only renders when Kit has a GUI::

            self.sim.step(render=(i == substeps - 1) and self.sim.has_gui())

        which is right for simulation and wrong for the server's frame endpoint: in
        headless there is no window, so the render never happens and a camera pointed at
        the cell returns a perfectly valid JPEG of a black rectangle. That is the worst
        failure mode available, because the interface shows it as a live render.

        This does not advance time. It only makes the next ``sim.step`` -- or the camera's
        own update -- produce pixels.
        """
        try:
            self.sim.render()
        except Exception as exc:                   # noqa: BLE001
            # Reported by the caller; a headless Kit without a viewport is a configuration
            # fact, not a crash.
            print(f"  render requested but unavailable: {exc}", flush=True)

    # -- control ---------------------------------------------------------

    def home(self, robot_id: str) -> None:
        self.robots[robot_id].set_joint_position_target(self._home_joints[robot_id])

    def set_gripper(self, robot_id: str, open_: bool) -> None:
        """Command the jaws.

        The command goes to the *joint*, through the same actuator model as the arm. The
        0.20 rad close target is not arbitrary: at LeIsaac's shipped 0.40 rad the jaws close
        so far in that they never squeeze a 30 mm part, and a 28 mm cube follows a lift 0%
        of the time.
        """
        import torch

        art = self.robots[robot_id]
        target = art.data.joint_pos_target.clone()
        value = M.GRIPPER_OPEN_RAD if open_ else M.GRIPPER_CLOSE_RAD
        target[:, self._grip_joint_id[robot_id]] = value
        art.set_joint_position_target(target, joint_ids=[self._grip_joint_id[robot_id]])

    def gripper_command(self, robot_id: str) -> float:
        import torch

        art = self.robots[robot_id]
        return float(art.data.joint_pos_target[0, self._grip_joint_id[robot_id]].item())

    def gripper_is_closing(self, robot_id: str) -> bool:
        """Whether the jaws have actually reached their closed target.

        Read from the simulated joint, not from the command, so "closed" means the fingers
        are where they were told to go.
        """
        art = self.robots[robot_id]
        actual = float(art.data.joint_pos[0, self._grip_joint_id[robot_id]].item())
        return abs(actual - M.GRIPPER_CLOSE_RAD) < 0.06

    # -- reading the world ----------------------------------------------

    def tcp_pos(self, robot_id: str):
        """World position of the gripper body, which is the tool frame this demo aims."""
        art = self.robots[robot_id]
        return art.data.body_pos_w[0, self._grip_body[robot_id]]

    def jaw_pos(self, robot_id: str):
        art = self.robots[robot_id]
        return art.data.body_pos_w[0, self._jaw_body[robot_id]]

    def part_pos(self, part_id: str):
        return self.parts[part_id].data.root_pos_w[0]

    def part_vel(self, part_id: str):
        return self.parts[part_id].data.root_lin_vel_w[0]

    def part_ang_vel(self, part_id: str):
        return self.parts[part_id].data.root_ang_vel_w[0]

    def part_state(self, part_id: str) -> PartState:
        import math as _m

        obj = self.parts[part_id]
        p = obj.data.root_pos_w[0]
        lin = float(obj.data.root_lin_vel_w[0].norm().item())
        ang = float(obj.data.root_ang_vel_w[0].norm().item())
        quat = obj.data.root_quat_w[0]
        yaw = _m.degrees(_quat_to_yaw_rad(quat))
        return PartState(
            id=part_id, x=float(p[0]), y=float(p[1]), z=float(p[2]),
            yaw_deg=yaw, lin_speed=lin, ang_speed=ang,
        )

    def robot_state(self, robot_id: str, status: str = "idle", holding: str | None = None,
                    note: str = "") -> RobotState:
        art = self.robots[robot_id]
        cfg = self.robot_cfgs[robot_id]
        tcp = self.tcp_pos(robot_id)
        joints = art.data.joint_pos[0]
        targets = art.data.joint_pos_target[0]
        return RobotState(
            id=robot_id,
            base_x=cfg.base_xy[0], base_y=cfg.base_xy[1],
            tcp_x=float(tcp[0]), tcp_y=float(tcp[1]), tcp_z=float(tcp[2]),
            joint_pos_rad=[float(v) for v in joints],
            joint_target_rad=[float(v) for v in targets],
            jaw_rad=float(joints[self._grip_joint_id[robot_id]].item()),
            gripping=self.gripper_is_closing(robot_id),
            holding=holding,
            status=status,
            note=note,
        )

    def snapshot(self, statuses: dict[str, tuple[str, str | None, str]] | None = None
                 ) -> dict[str, Any]:
        """Everything the interface needs, read out of the live simulation."""
        statuses = statuses or {}
        holding_by_part: dict[str, str] = {}
        robot_states: list[RobotState] = []
        for robot_id in self.robots:
            status, holding, note = statuses.get(robot_id, ("idle", None, ""))
            st = self.robot_state(robot_id, status, holding, note)
            robot_states.append(st)
            if holding:
                holding_by_part[holding] = robot_id

        part_states = []
        for part_id in self.parts:
            st = self.part_state(part_id)
            st.held_by = holding_by_part.get(part_id)
            part_states.append(st)

        return {
            "simTimeS": round(self.sim_time_s, 3),
            "stepCount": self.step_count,
            "controlStepCount": self.control_step_count,
            "dt": self.sim_dt,
            "controlDt": self.sim_dt * M.CELL_STEPS_PER_CONTROL_STEP,
            "tableTopZ": self.table_surface_z,
            "padMaterialBound": self.pad_material_bound,
            "robots": [r.to_dict() for r in robot_states],
            "parts": [p.to_dict() for p in part_states],
            "bins": [
                {
                    "id": b.id, "label": b.label, "color": b.color,
                    "x": b.x, "y": b.y,
                    "innerM": list(b.inner_m), "outerM": list(b.outer_size_m),
                    "rimZ": round(b.rim_z_m, 5), "floorZ": round(b.cavity_floor_z_m, 5),
                }
                for b in self.bins.values()
            ],
        }

    # -- teardown --------------------------------------------------------

    def close(self) -> None:
        try:
            self.sim.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def torch_tensor(rows, device: str):
    import torch

    return torch.tensor(rows, dtype=torch.float32, device=device)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _quat_to_yaw_rad(quat) -> float:
    """Yaw of a wxyz quaternion."""
    import math as _m

    w, x, y, z = (float(v) for v in quat)
    return _m.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))











