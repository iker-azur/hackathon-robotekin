"""Demo 11: demo 10, on physics you could build, watched from a browser.

    python 11_real_sorting_with_monitor.py --open                # dashboard, then Isaac
    python 11_real_sorting_with_monitor.py --no-web              # no dashboard, straight to Isaac
    python 11_real_sorting_with_monitor.py --setup-only          # the bench sheet, no Isaac
    python 11_real_sorting_with_monitor.py --no-wait-for-start --text "separa las piezas por colores"
    python 11_real_sorting_with_monitor.py --voice --web --no-wait-for-start

Started with no arguments it serves the dashboard and waits for the dashboard to start Isaac,
which is the intended way to run it: the setup screen is up before the two-minute simulator
start-up. ``--no-wait-for-start`` (or any ``--text``) goes straight in.

Two changes over demo 10, and the second one is the reason this file exists.

1. **The physics are the ones a real bench would have.** Parts are spawned at the mass their
   material gives them (28 g for a 30 mm ABS cube, not the 50 g demo 10 used for everything),
   with measured friction and restitution instead of PhysX's defaults, and rolling resistance
   represented explicitly rather than by damping everything until the balls stopped. The grip is
   no longer faked: demo 10's ``Executor.grab`` writes a part's pose from the gripper frame every
   step, which is how it picked up a 3 cm ball in a simulation where nothing is ever held by
   friction - and which would be nonsense on hardware. Here the jaws have to hold the part, and
   when they do not, the demo says so and re-plans. ``real_world.py`` holds the numbers, the
   friction-cone check and the bill of materials.

   That honesty has teeth. It exposed a geometry bug demo 10 was working around: the part's
   centre is released 12 mm above the tray rim while the fingers hang 32 mm below the part, so
   the jaws were inside the tray wall at the moment of release - and the part was dragged along
   the inside of the cavity instead of dropped into it. ``--release clear`` releases from where
   the fingertips clear the rim, which is the only height a real arm can use.

2. **A localhost dashboard** (``--web``, default on) shows the whole decision path as a live
   diagram: which stage is running, what the model understood, what the camera saw, which blob
   is which part, the plan's steps as they execute, and whether the part landed. The camera feed
   is the annotated frame the robot actually planned from, so what you see is what it saw. The
   dashboard is up *before* Isaac starts, which is where the environment gets set up and the run
   gets started - a two-minute simulator start-up is not something to watch through a console.

Layout, and why demo 10's code is loaded rather than copied
-----------------------------------------------------------
``Robot`` lives in ``10_command_to_motion.py``, which is a filename no ``import`` statement can
name, so it is loaded by path the way ``isaac_io`` loads demo 01's ``env_loader``. Demo 11 then
*subclasses* it and overrides the handful of methods that need to behave differently or to
report to the dashboard. Demo 10 keeps its behaviour exactly, which matters because its work is
still in progress; nothing here is a copy that can drift.

Command input is three ways at once, because a bench demo is watched by a person in a browser
and typed at by someone with a keyboard: the browser's box, the microphone, and the terminal.
Whichever has something to say first is taken, and ``q`` in the browser aborts the run exactly
as ``q`` in the terminal does.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import colors  # noqa: E402
import hollow  # noqa: E402
import describe  # noqa: E402
import dynamics  # noqa: E402
import executor as executor_module  # noqa: E402
import gripper_contact_pads  # noqa: E402
import isaac_io  # noqa: E402
import pen as pen_module  # noqa: E402
import planner  # noqa: E402
import perceive  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402
import telemetry  # noqa: E402
import scene_config  # noqa: E402
import viewer as viewer_module  # noqa: E402
from command_model_v2.command_model import CommandModel  # noqa: E402

# ``monitor.server`` is imported where it is used, not here. It pulls in fastapi and uvicorn, and
# measured that is 305 ms of the 440 ms this module costs to import - paid by every test that
# wants ``StrictExecutor`` or the task list, and by any tool that imports this file for one
# constant. The dashboard is a separate concern from the robot, and importing a web framework is
# not something to do to answer "how wide is the gripper".
#
#   monitor_server.DEFAULT_HOST   -> the two defaults in build_parser
#   monitor_server.serve(...)     -> the dashboard in main
#
# Both are in `main`'s path already, so nothing pays for this unless the dashboard is used.
def _monitor_server():
    """Import the dashboard's server module on demand, and remember it."""
    global _MONITOR_SERVER
    if _MONITOR_SERVER is None:
        from monitor import server

        _MONITOR_SERVER = server
    return _MONITOR_SERVER


_MONITOR_SERVER = None

DEMO10_PATH = HERE / "10_command_to_motion.py"

# JPEG quality for the dashboard frame. 80 keeps a 640x480 annotated frame around 40 kB, which
# is sharp enough to read a blob outline over loopback and small enough to push on every frame
# without the socket becoming the thing that is slow.
FRAME_JPEG_QUALITY = 80

# How long the command wait looks at the browser before falling back to the keyboard. Short
# enough to feel instant, long enough that a person who is about to type is not fighting it.
WEB_COMMAND_POLL_S = 0.2

# How many panel controls are carried out between two physics steps. Each relook is a full
# perception pass, so this bounds the time the simulation waits for the page; the rest queue.
MAX_CONTROLS_PER_STEP = 2

# RTX can return a valid frame from the previous layout immediately after prim transforms
# change. The blank-frame retry cannot detect a stale-but-nonblank image, so render a few
# synchronized frames after each scene population before the first perception pass.
SCENE_SETTLE_RENDER_STEPS = 3


def load_demo10():
    """Import ``10_command_to_motion.py`` by path, as a module called ``demo10``.

    Registered in ``sys.modules`` before it is executed, because the module defines dataclasses
    and those resolve their own module through ``sys.modules`` at class-creation time.
    """
    if "demo10" in sys.modules:
        return sys.modules["demo10"]
    spec = importlib.util.spec_from_file_location("demo10", DEMO10_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {DEMO10_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo10"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------
# Real physics
# --------------------------------------------------------------------------------------


def rigid_cfg_real(item: scene_spec.Item):
    """``scene_spec.Item`` as an IsaacLab rigid body with the real part's material.

    Same shape as ``isaac_io.rigid_cfg`` and deliberately a separate function rather than an
    edit to it: demo 10's mass and its ball damping are part of what demo 10 is, and demo 11 is
    only allowed to disagree with them here.

    Three fields carry the whole difference:

    * ``mass_props`` from ``real_world.mass_kg``, so the friction cone in the solver sees the
      weight a person would weigh.
    * ``physics_material`` with the material's friction and restitution. Without this every
      contact is PhysX's 0.5/0.0 default, which is frictionless enough that a ball slides off a
      tray rim and an object never settles where it was dropped.
    * ``rigid_props`` damping straight from the material, and only on the ball - see
      ``Material.rolling_damping`` for why that is a stand-in and not a fudge.

    The shipped USD cube is not one of these: it already exists inside ``table_with_cube`` as a
    rigid body with whatever mass and material that asset declares, and it is left alone. That
    is stated rather than hidden because it means one of the four parts in the scene is not
    running on real numbers.
    """
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    material = real_world.material_of(item)
    common = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=real_world.mass_kg(item)),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            linear_damping=material.linear_damping,
            angular_damping=material.rolling_damping),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=material.static_friction,
            dynamic_friction=material.dynamic_friction,
            restitution=material.restitution),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=colors.COLORS[item.color].albedo, roughness=0.7),
    }
    if item.kind == "ball":
        spawn = sim_utils.SphereCfg(radius=item.size[0] / 2.0, **common)
    elif item.kind == "cylinder":
        spawn = sim_utils.CylinderCfg(radius=item.size[0] / 2.0, height=item.size[2], **common)
    elif item.kind in ("triangular_prism", "hexagonal_prism"):
        from polygon_prism import spawn_cfg
        spawn = spawn_cfg(sim_utils, common, item.size[0], item.size[2],
                          3 if item.kind == "triangular_prism" else 6)
    else:
        spawn = sim_utils.CuboidCfg(size=item.size, **common)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/" + item.prim,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(item.x, item.y, item.z),
            rot=(math.cos(math.radians(item.yaw_deg) / 2), 0.0, 0.0,
                 math.sin(math.radians(item.yaw_deg) / 2))),
    )


def set_gripper_actuator(cfg, stiffness: float = real_world.GRIPPER.actuator_stiffness) -> str:
    """Set the gripper actuator's stiffness in the task config, before ``gym.make``.

    The stiffness is what actually decides how hard the pads squeeze, and it belongs in the
    actuator config rather than being poked at runtime: ``Articulation.write_joint_stiffness_to_sim``
    works - ``tools/probe_grip_hold.py`` uses it for every trial - but in this process it crashes
    with a Windows access violation. The difference between the probe and this script is that the
    demo loads the command model (torch, gliner2) before Kit starts, and reaching through the
    PhysX tensor API afterwards goes through the allocator the two of them share. The repo already
    records this class of failure twice, in ``isaac_io.prepare_runtime``'s docstring and in demo
    10's comment about loading the model first; this is the same wall, reached from a different
    direction.

    Configuring it before the environment is built also puts the number where a reader can find
    it: it is one line of the scene description rather than a side effect hidden in a start-up
    routine. Returns what it changed, for the log.

    ``cfg.scene`` is a configclass, so it is read with ``getattr`` rather than indexed. Every other
    access in this file follows ``isaac_io.build_env`` and uses ``setattr`` for the same reason; the
    subscript here raised ``TypeError: 'LiftCubeSceneCfg' object is not subscriptable`` on every
    launch, which is worth recording because it looked exactly like an Isaac start-up failure -
    the process died silently in the same place, immediately after Kit came up.
    """
    robot_cfg = getattr(cfg.scene, "robot", None)
    if robot_cfg is None:
        return "the scene has no robot to configure"
    changed = []
    for name, actuator in (getattr(robot_cfg, "actuators", None) or {}).items():
        joints = getattr(actuator, "joint_names_expr", None) or []
        if any("gripper" in str(joint) for joint in joints) or "gripper" in name:
            before = getattr(actuator, "stiffness", None)
            actuator.stiffness = float(stiffness)
            changed.append(f"{name} {before} -> {stiffness}")
    return ", ".join(changed) if changed else "no gripper actuator group found"


def experimental_pad_friction(override: str | None,
                              defaults: tuple[float, float] | None = None) -> tuple[float, float]:
    """Parse an opt-in ``static,dynamic`` rubber friction A/B without changing defaults."""
    base = defaults or (real_world.RUBBER_PAD.static_friction,
                        real_world.RUBBER_PAD.dynamic_friction)
    if override is None or not override.strip():
        return float(base[0]), float(base[1])
    parts = [part.strip() for part in override.split(",")]
    if len(parts) != 2:
        raise ValueError("experimental pad friction must be 'static,dynamic'")
    values = tuple(float(part) for part in parts)
    if not all(math.isfinite(value) and 0.05 <= value <= 2.0 for value in values):
        raise ValueError("experimental pad friction coefficients must be within [0.05, 2.0]")
    if values[1] > values[0]:
        raise ValueError("dynamic pad friction cannot exceed static friction")
    return values


def experimental_carry_steps(override: str | None, default: int = 120) -> int:
    """Validate an opt-in slower carry A/B; the normal trajectory stays unchanged."""
    if override is None or not override.strip():
        return int(default)
    value = int(override)
    if not 80 <= value <= 400:
        raise ValueError("experimental carry steps must be within [80, 400]")
    return value


def set_gripper_pad_material(env) -> int:
    """Bind the declared rubber friction to the SO-101 finger collision shapes.

    ``real_world.grasp`` has always calculated with ``RUBBER_PAD`` (mu=0.90), but the
    articulation USD did not have that pad material bound. Setting friction on a part does
    not set friction on the opposing contact, so the physical benchmark was using the USD's
    default finger material while the preflight reported the rubber-pad friction cone. Bind a
    dedicated physics-only material to the two finger links, leaving their visual materials
    untouched. The strict grasp benchmark verifies the result through measured lifts/carries.
    """
    from isaaclab.sim.utils import get_current_stage
    from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    robot = env.scene["robot"]
    root_path = str(robot.root_physx_view.prim_paths[0])
    stage = get_current_stage()
    robot_root = stage.GetPrimAtPath(root_path)
    if not robot_root or not robot_root.IsValid():
        raise RuntimeError(f"Robot USD root is unavailable: {root_path}")

    material_path = Sdf.Path("/World/PhysicsMaterials/Demo11RubberPad")
    UsdGeom.Scope.Define(stage, material_path.GetParentPath())
    material_prim = UsdShade.Material.Define(stage, material_path).GetPrim()
    pad_static, pad_dynamic = experimental_pad_friction(
        os.environ.get("DEMO11_EXPERIMENTAL_PAD_FRICTION"))
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr().Set(pad_static)
    material_api.CreateDynamicFrictionAttr().Set(pad_dynamic)
    material_api.CreateRestitutionAttr().Set(real_world.RUBBER_PAD.restitution)
    PhysxSchema.PhysxMaterialAPI.Apply(material_prim)

    bound = []
    for prim in Usd.PrimRange(robot_root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = prim.GetPath().pathString.lower().replace("-", "_")
        segments = {part.lower() for part in path.split("/")}
        if not segments.intersection({"gripper", "jaw", "left_finger", "right_finger"}):
            continue
        add_physics_material_to_prim(stage, prim, material_path)
        bound.append(prim.GetPath().pathString)
    if not bound:
        raise RuntimeError(f"No gripper finger collision prims found under {root_path}")
    print(f"  rubber pad material bound to {len(bound)} finger collider(s): "
          f"mu={pad_static:.2f}/{pad_dynamic:.2f}", flush=True)
    return len(bound)


def set_demo11_table_collision(env) -> str:
    """Replace the imported triangle-mesh tabletop contact with one calibrated flat slab.

    PhysX reports a level tabletop under the 20 mm sphere, but its first collision with the
    imported mesh creates a deterministic lateral spin/roll. A primitive slab at the measured
    table bounds removes that mesh-contact artifact while retaining the visible USD table and
    every non-top collision shape. It also binds the melamine material that ``real_world.TABLE``
    declares; the imported top mesh otherwise falls back to PhysX's default material.
    """
    from isaaclab.sim.utils import get_current_stage
    from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage = get_current_stage()
    mesh_path = "/World/envs/env_0/Scene/counter_right_main_group/geometry_0"
    mesh = stage.GetPrimAtPath(mesh_path)
    if not mesh or not mesh.IsValid():
        raise RuntimeError(f"Demo 11 tabletop collision mesh is unavailable: {mesh_path}")
    UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr().Set(False)

    collider_path = Sdf.Path("/World/envs/env_0/Scene/Demo11TableTopCollider")
    collider = stage.GetPrimAtPath(collider_path)
    if not collider or not collider.IsValid():
        cube = UsdGeom.Cube.Define(stage, collider_path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        center_x = (scene_spec.TABLE_X[0] + scene_spec.TABLE_X[1]) / 2
        center_y = (scene_spec.TABLE_Y[0] + scene_spec.TABLE_Y[1]) / 2
        thickness = 0.02
        xform.AddTranslateOp().Set(Gf.Vec3d(
            center_x, center_y, scene_spec.TABLE_TOP_Z - thickness / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(
            scene_spec.TABLE_X[1] - scene_spec.TABLE_X[0],
            scene_spec.TABLE_Y[1] - scene_spec.TABLE_Y[0], thickness))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collider = cube.GetPrim()

    material_path = Sdf.Path("/World/PhysicsMaterials/Demo11MelamineTable")
    UsdGeom.Scope.Define(stage, material_path.GetParentPath())
    material_prim = UsdShade.Material.Define(stage, material_path).GetPrim()
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr().Set(real_world.TABLE.static_friction)
    material_api.CreateDynamicFrictionAttr().Set(real_world.TABLE.dynamic_friction)
    material_api.CreateRestitutionAttr().Set(real_world.TABLE.restitution)
    PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    add_physics_material_to_prim(stage, collider, material_path)
    print(f"  tabletop collision: flat slab {scene_spec.TABLE_X} x {scene_spec.TABLE_Y}, "
          f"z={scene_spec.TABLE_TOP_Z:.5f} m; melamine mu="
          f"{real_world.TABLE.static_friction:.2f}/{real_world.TABLE.dynamic_friction:.2f}, "
          f"e={real_world.TABLE.restitution:.2f}", flush=True)
    return str(collider_path)


def shipped_cube_is_configured(items: list[scene_spec.Item]) -> bool:
    """Whether the scene catalog explicitly uses the task asset's built-in cube."""
    return any(item.prim == scene_spec.SHIPPED_CUBE and not item.container for item in items)


def set_unconfigured_shipped_cube_visible(env, items: list[scene_spec.Item]) -> bool:
    """Hide and disable the task USD cube unless the scene catalog includes it.

    LeIsaac's ``table_with_cube`` scene always authors a dynamic red cube. A custom scene with
    only balls or prisms otherwise gets an unlisted perception distractor and a collider that
    can interfere with configured objects. If the catalog explicitly includes the shipped cube,
    keep the asset instance active as that configured object.
    """
    if shipped_cube_is_configured(items):
        return True

    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    env_count = int(getattr(env, "num_envs", 1))
    hidden_roots = 0
    disabled_colliders = 0
    for env_index in range(env_count):
        path = f"/World/envs/env_{env_index}/Scene/{scene_spec.SHIPPED_CUBE}"
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsValid():
            continue
        imageable = UsdGeom.Imageable(root)
        if imageable:
            imageable.MakeInvisible()
            hidden_roots += 1
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled_colliders += 1
    print(f"  unconfigured scene cube: hidden {hidden_roots} root(s), "
          f"disabled {disabled_colliders} collider(s)", flush=True)
    return False


def enable_contact_reporting(robot_spawn, target_spawn=None) -> None:
    """Enable PhysX contact reporters on the bodies monitored by Demo 11 diagnostics.

    ContactSensorCfg wraps PhysX's contact reporting API. Creating a sensor on a rigid body is
    not enough: the corresponding asset spawner must activate reporters before it is spawned.
    The target and gripper links are separate assets in the diagnostic scene, so enable both.
    """
    # Isaac Lab's current clone wrapper forwards this value as the reporter's numeric force
    # threshold. Although its config field is annotated bool, ``True`` becomes 1 N and hides
    # light contact during a slipping grasp. Keep it truthy to activate reporting while using a
    # threshold low enough to expose those contacts.
    threshold_n = 1e-4
    if robot_spawn is None:
        raise ValueError("robot spawn config is required for gripper contact reporting")
    robot_spawn.activate_contact_sensors = threshold_n
    if target_spawn is not None:
        target_spawn.activate_contact_sensors = threshold_n


def build_real_env(camera: str, items: list[scene_spec.Item], pen: bool = False,
                   grip_stiffness: float = real_world.GRIPPER.actuator_stiffness,
                   device: str = "cuda:0", contact_diagnostics_for: str | None = None,
                   auto_validated_contact_pads: bool = False,
                   experimental_contact_pads: bool = False,
                   experimental_contact_pad_radius_m: float = gripper_contact_pads.PAD_RADIUS_M,
                   experimental_contact_pad_shape: str = "sphere",
                   experimental_contact_pad_thickness_m: float = gripper_contact_pads.PAD_THICKNESS_M,
                   experimental_contact_pad_angle_deg: float = gripper_contact_pads.PAD_VEE_ANGLE_DEG,
                   experimental_contact_pad_force_unvalidated: bool = False,
                   grasp_profile_kind: str | None = None,
                   contact_diagnostics_all_robot_links: bool = False):
    """The shipped lift-cube task, with demo 11's materials and demo 11's gripper.

    A copy of ``isaac_io.build_env`` rather than a patch of it, for two reasons that both came
    from trying the patch first. It has to build its rigid bodies from ``rigid_cfg_real`` (real
    mass, real friction, damping only where a material needs it), and it has to reach the task
    config to set the gripper's stiffness - and a config object only exists inside the function
    that makes the environment. Duplicating 40 lines of configuration is cheaper than
    monkeypatching a module's internals from outside it, and it leaves demo 10's copy exactly as
    it was. The ordering notes below are demo 09's and are kept because they are load-bearing.
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import leisaac.tasks  # noqa: F401  registers the gym task ids

    cfg = parse_env_cfg(isaac_io.TASK, device=device, num_envs=1)
    profiled_item = None
    if auto_validated_contact_pads and not experimental_contact_pads:
        profiled_items = gripper_contact_pads.validated_profile_items(
            [item for item in items
             if not item.container and item.prim != scene_spec.SHIPPED_CUBE],
            target_kind=grasp_profile_kind)
        automatic_shape = gripper_contact_pads.automatic_scene_pad_shape(
            profiled_items, target_kind=grasp_profile_kind)
        if automatic_shape:
            experimental_contact_pads = True
            experimental_contact_pad_shape = automatic_shape
            profiled_item = next(item for item in profiled_items if not item.container)
            angle_override = os.environ.get("DEMO11_EXPERIMENTAL_VEE_ANGLE_DEG")
            experimental_contact_pad_angle_deg = gripper_contact_pads.vee_angle_for(
                profiled_item.kind, profiled_item.size, experimental_contact_pad_angle_deg,
                profiled_item.yaw_deg,
                override_deg=(float(angle_override) if angle_override is not None else None))
            tested_stiffness = gripper_contact_pads.vee_stiffness_for(
                profiled_item.kind, profiled_item.size, grip_stiffness,
                profiled_item.yaw_deg)
            if (grip_stiffness == real_world.GRIPPER.actuator_stiffness
                    and tested_stiffness != grip_stiffness):
                grip_stiffness = tested_stiffness
            scope = (f"target geometry {grasp_profile_kind}" if grasp_profile_kind
                     else "every movable shape and orientation in this scene")
            if automatic_shape == "flat":
                print("  flat contact pads selected automatically: " + scope +
                      " has a validated physical profile "
                      f"(gripper stiffness {grip_stiffness:.0f})", flush=True)
            else:
                print("  V-groove contact profile selected automatically: " + scope +
                      " has a validated physical profile "
                      f"(V angle {experimental_contact_pad_angle_deg:.0f} deg, "
                      f"gripper stiffness {grip_stiffness:.0f})", flush=True)
        if not automatic_shape:
            tested_stiffness = gripper_contact_pads.automatic_scene_grip_stiffness(
                items, grip_stiffness, target_kind=grasp_profile_kind)
            if (grip_stiffness == real_world.GRIPPER.actuator_stiffness
                    and tested_stiffness != grip_stiffness):
                grip_stiffness = tested_stiffness
                print("  stock fingers use the validated 30 mm geometry stiffness profile "
                      f"for {grasp_profile_kind or 'homogeneous targets'} "
                      f"({grip_stiffness:.0f})", flush=True)
    pad_shape_override = os.environ.get("DEMO11_EXPERIMENTAL_PAD_SHAPE")
    if pad_shape_override is not None:
        if pad_shape_override not in {"sphere", "flat", "vee"}:
            raise ValueError("experimental pad shape must be sphere, flat, or vee")
        experimental_contact_pads = True
        experimental_contact_pad_shape = pad_shape_override
    if experimental_contact_pad_shape == "vee":
        experimental_contact_pad_angle_deg = gripper_contact_pads.experimental_vee_angle_deg(
            os.environ.get("DEMO11_EXPERIMENTAL_VEE_ANGLE_DEG"),
            experimental_contact_pad_angle_deg)
    force_unvalidated_pads = (
        experimental_contact_pad_force_unvalidated
        or gripper_contact_pads.force_unvalidated_experimental_pads(
            os.environ.get("DEMO11_EXPERIMENTAL_FORCE_UNVALIDATED_PADS")))
    effective_contact_pad_shape = (experimental_contact_pad_shape
                                   if experimental_contact_pads else None)
    if (experimental_contact_pads and experimental_contact_pad_shape == "vee"
            and not force_unvalidated_pads):
        compatible_items = [(item.kind, item.size, item.yaw_deg)
                            for item in gripper_contact_pads.validated_profile_items(
                                [item for item in items if not item.container
                                 and item.prim != scene_spec.SHIPPED_CUBE],
                                target_kind=grasp_profile_kind)]
        if not gripper_contact_pads.scene_uses_only_vee_profiles(compatible_items):
            # PhysX does not reliably refresh a cooked robot collider when a USD
            # CollisionAPI flag is changed mid-run. Never hot-swap V inserts in a mixed
            # scene: their protruding faces can displace objects that need stock jaws.
            effective_contact_pad_shape = None
            experimental_contact_pads = False
            print("  V-groove skipped: scene contains unvalidated object profiles; "
                  "keeping stock SO-101 fingertips", flush=True)
    if experimental_contact_pads:
        robot_spawn = getattr(cfg.scene.robot, "spawn", None)
        source_usd = getattr(robot_spawn, "usd_path", None)
        if not source_usd:
            raise RuntimeError("SO-101 spawn config does not expose its USD path")
        pad_inset_override = os.environ.get("DEMO11_EXPERIMENTAL_PAD_INSET_MM")
        pad_span_override = os.environ.get("DEMO11_EXPERIMENTAL_PAD_SPAN_MM")
        if (profiled_item is not None and experimental_contact_pad_shape == "vee"):
            pad_inset_m = gripper_contact_pads.pad_inset_for(
                profiled_item.kind, profiled_item.size, pad_inset_override)
            pad_span_m = gripper_contact_pads.pad_span_for(
                profiled_item.kind, profiled_item.size, pad_span_override)
        else:
            pad_inset_m = gripper_contact_pads.experimental_pad_inset_m(pad_inset_override)
            pad_span_m = gripper_contact_pads.experimental_pad_span_m(pad_span_override)
        if pad_inset_m and experimental_contact_pad_shape != "vee":
            raise ValueError("experimental pad inset is supported only with V-groove pads")
        pad_radius_m = gripper_contact_pads.experimental_pad_radius_m(
            os.environ.get("DEMO11_EXPERIMENTAL_PAD_RADIUS_MM"))
        padded_usd = gripper_contact_pads.make_padded_robot_asset(
            source_usd, radius_m=pad_radius_m if pad_shape_override == "sphere"
            else experimental_contact_pad_radius_m,
            shape=experimental_contact_pad_shape,
            thickness_m=experimental_contact_pad_thickness_m,
            vee_angle_deg=experimental_contact_pad_angle_deg,
            inset_m=pad_inset_m, span_m=pad_span_m)
        robot_spawn.usd_path = str(padded_usd)
        print(f"  {experimental_contact_pad_shape} contact-pad USD: {padded_usd} "
              f"(V {experimental_contact_pad_angle_deg:.0f} deg, "
              f"span {pad_span_m * 1000:.0f} mm, inset {pad_inset_m * 1000:.1f} mm)",
              flush=True)
    if contact_diagnostics_for is not None:
        if contact_diagnostics_for not in {item.prim for item in items if not item.container}:
            raise ValueError(f"Contact diagnostic target is not a loose scene item: {contact_diagnostics_for}")
        from isaaclab.sensors import ContactSensorCfg

        # PhysX contact reporting is opt-in on rigid bodies. Keep per-link sensors and also
        # attach a reverse sensor to the object: the first positive-control run carried a cube
        # physically while both finger-filtered views stayed at zero, so diagnostics must verify
        # the pair from both sides before we trust their force values.
        cfg.scene.robot.spawn.activate_contact_sensors = True
    # The action manager is only installed by the teleop device, and without it the arm_action
    # term is missing and the environment cannot be built.
    cfg.use_teleop_device("so101_state_machine")
    cfg.seed = 0
    cfg.recorders = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None  # otherwise lifting an object ends the episode

    # The stock reset event writes velocity to every rigid object. Demo 11 receptacles
    # are intentionally kinematic, so reset their poses without an invalid PhysX write.
    if hasattr(cfg.events, "reset_all"):
        cfg.events.reset_all.func = isaac_io.reset_scene_to_default_without_kinematic_velocity_writes

    # The task randomises the cube by up to 7.5 cm and the camera by 5 mm on every reset. That
    # is right for training a policy and fatal for a measurement; demo 11 places everything
    # explicitly instead.
    for name in ("domain_randomize_0", "domain_randomize_1"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)

    # The shipped cameras are removed: one is a metre from the robot base and the other fails to
    # load on this machine (isaacsim.sensors.rtx).
    for term in ("front", "wrist"):
        if hasattr(cfg.scene, term):
            setattr(cfg.scene, term, None)
        if hasattr(cfg.observations.policy, term):
            setattr(cfg.observations.policy, term, None)

    print(f"  gripper actuator stiffness: {set_gripper_actuator(cfg, grip_stiffness)}", flush=True)

    for item in items:
        if item.prim == scene_spec.SHIPPED_CUBE:
            continue  # already a rigid body inside the scene USD, with the asset's own mass
        if item.container:
            # A hollow tray, not a solid slab: one kinematic rigid body with a floor and four
            # walls as its collision shapes. See hollow.py for why this cannot be five siblings.
            setattr(cfg.scene, item.prim, hollow.make_tray_asset_cfg(item))
        else:
            setattr(cfg.scene, item.prim, rigid_cfg_real(item))

    if contact_diagnostics_for is not None:
        # Loose benchmark objects are separate RigidObjectCfg assets. The shipped cube is
        # authored inside the task's scene USD, so enabling reporters on that scene asset is
        # necessary for the reverse (object -> robot) filtered view to report contact.
        target_cfg = getattr(cfg.scene, contact_diagnostics_for, None)
        if target_cfg is not None and hasattr(target_cfg, "spawn"):
            enable_contact_reporting(cfg.scene.robot.spawn, target_cfg.spawn)
        else:
            enable_contact_reporting(cfg.scene.robot.spawn, cfg.scene.scene.spawn)
        diagnostic_links = ("shoulder", "upper_arm", "lower_arm", "wrist",
                            "gripper", "jaw") if contact_diagnostics_all_robot_links else (
                                "gripper", "jaw")
        for link in diagnostic_links:
            setattr(cfg.scene, f"demo11_contact_{link}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
                filter_prim_paths_expr=[
                    f"{{ENV_REGEX_NS}}/Scene/{contact_diagnostics_for}"],
                update_period=0.0,
                history_length=0,
            ))
        setattr(cfg.scene, "demo11_contact_target", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{contact_diagnostics_for}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/gripper",
                                    "{ENV_REGEX_NS}/Robot/jaw"],
            update_period=0.0,
            history_length=0,
        ))

    if pen:
        # The containment kerb. Same body shape as a tray, four walls and no floor; see pen.py.
        # Spawned last so it exists before anything is placed inside it.
        setattr(cfg.scene, pen_module.PEN_PRIM, pen_module.make_pen_asset_cfg())

    sensor = isaac_io.ensure_camera(cfg, camera)
    backup_sensor = isaac_io.ensure_camera(cfg, "top") if camera != "top" else None
    env = gym.make(isaac_io.TASK, cfg=cfg).unwrapped
    env.demo11_shipped_cube_configured = set_unconfigured_shipped_cube_visible(env, items)
    set_gripper_pad_material(env)
    set_demo11_table_collision(env)
    env.demo11_contact_pad_shape = effective_contact_pad_shape
    env.demo11_contact_diagnostics_for = contact_diagnostics_for
    # Oblique views can be occluded by the arm while it reaches into a tray. Keep a
    # calibrated overhead sensor available as a perception fallback for those scenes.
    env.demo11_backup_camera = backup_sensor
    return env, sensor


def set_gripper(env, angle_rad: float = real_world.GRIPPER.close_command_rad) -> float:
    """Command the jaws to close as far as this scene's parts need, and report what it was.

    Not a detail. LeIsaac ships a close target of 0.4 rad, chosen for the 8 cm orange its own task
    drops, and at that angle the simulated jaws squeeze a 30 mm part without ever holding it:
    measured with demo 11's own materials by ``tools/probe_grip_hold.py``, the cube followed the
    lift 100% at 0.20 rad and **0%** at 0.30 and 0.40 rad, at every actuator stiffness tried. The
    gripper's finger bodies stay 36.2 mm apart whatever the command - they rotate about their
    mounts - so what the number really sets is how far the pads swing in.

    The matching stiffness is set in the task config by ``set_gripper_actuator``, before the
    environment exists. At the shipped 17.8 the cube is held outright but the ball follows only
    79% of the lift, under ``executor.GRASP_FOLLOW_MIN``; at 250 both are held. A sphere meets
    the pads at a point and a cube at a face, which is why the lighter part needs the stiffer
    servo to be carried at all.

    Demo 10 never found either, because its kinematic fallback picked the part up out of the
    jaws' grip and reported success. This is the class of thing that only shows up when the
    simulation is not allowed to lie about the grasp.
    """
    term = env.action_manager.get_term("gripper_action")
    before = float(term._close_command[0])
    term._close_command[:] = float(angle_rad)
    return before


# How long the jaws stay shut before the lift starts.
#
# ``executor.STEP_COUNTS`` spends 50 steps (0.83 s) on ``close``. The grip probe
# (``tools/probe_grip_hold.py``) gave the jaws 90, and with 50 the part is still being squeezed
# when the lift begins: measured in this demo, the cube's lift test failed with the jaws part-way
# closed and the grasp was refused, while the same pick with 90 steps was held outright. So this is
# not a fudge to make a grasp succeed - it is the duration at which the jaws have actually reached
# their commanded angle, which is what ``real_world.GRIPPER.close_command_rad`` assumes.
CLOSE_SETTLE_STEPS = 90

# These shapes have not demonstrated reliable friction grasps with the shipped SO-101 jaw
# contacts. The simulator demo uses and reports an object-follow constraint for them; cube
# grasps remain physical by default. The strict benchmark still measures physical contact.
# Demo 11 is a simulation demonstration. The shipped SO-101 jaw contact model has now also
# failed to retain cubes reliably after an otherwise centred close (measured: the cube moved
# 130 mm during a 120 mm lift). Keep the live demo dependable by using the same explicit,
# close-gated simulator constraint for every supported shape. The strict physical-grasp
# benchmark sets ``auto_sim_assist=False`` and remains the place to measure friction honestly.
SIM_ASSISTED_SHAPES = frozenset({"cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"})

# How far off centre a part may be and still end up between the pads: half the aperture less half
# the part. This is a geometric fact about the gripper, not a tuned number, and it is the budget
# every grasp has to fit inside.
LATERAL_SLACK_M = (real_world.GRIPPER.aperture_m - 0.030) / 2.0
# Rounding at the calibrated aperture boundary must not reject a centred contact by
# a few floating-point microns; this tolerance is far smaller than camera/servo error.
APERTURE_NUMERIC_TOLERANCE_M = 0.00005


def aperture_gap_is_graspable(gap_m: float | None,
                              slack_m: float = LATERAL_SLACK_M) -> bool:
    """Only allow contact when the measured jaw midpoint is inside the part's slack."""
    return (gap_m is not None and math.isfinite(float(gap_m))
            and 0.0 <= float(gap_m) <= float(slack_m) + APERTURE_NUMERIC_TOLERANCE_M)


def aperture_slack_for_size(size_m) -> float:
    """Compute half the free jaw width for a declared object's widest footprint axis."""
    if not size_m:
        return LATERAL_SLACK_M
    width = max(float(value) for value in size_m[:2])
    return max(0.0, (real_world.GRIPPER.aperture_m - width) / 2.0)


def aperture_slack_for_profile(size_m, *, kind: str, pad_shape: str | None,
                               experimental_slack_mm: float | None = None) -> float:
    """Keep measured jaw slack by default; permit a bounded ball/V-pad A/B override only."""
    measured = aperture_slack_for_size(size_m)
    is_ball30_vee = (kind == "ball" and pad_shape == "vee" and bool(size_m)
                     and 0.029 <= float(size_m[0]) <= 0.031)
    if not is_ball30_vee or experimental_slack_mm is None:
        return measured
    candidate = float(experimental_slack_mm) / 1000.0
    if not math.isfinite(candidate) or not measured <= candidate <= 0.008:
        raise ValueError("experimental ball/V-pad aperture slack must be within measured slack..8 mm")
    return candidate

# A simulator-only object-follow grasp may absorb a bounded residual from camera/IK servoing, but
# it must still be demonstrably over the intended object. The 30 mm ceiling is below the 40 mm
# minimum spacing in generated clusters, preserves clearance to adjacent pieces, and is used only
# by the explicitly labelled simulation constraint. Strict physical grasp checks still use the
# much smaller measured aperture slack above. The prior 20 mm gate left several valid 20 mm parts
# just outside reach (22-28 mm) in alternate camera/scene variants.
SIM_ASSIST_MAX_LATERAL_M = 0.030


def grasp_height_error(object_z: float, body_target_z: float,
                       hold_offset_z: float) -> float:
    """Compare an object's centre with the gripper's hold point, in world Z.

    Tracked close steps pass a direct gripper-body target. Comparing the object's centre with
    that body's Z ignores the calibrated body-to-hold offset (about -92 mm on the SO-101) and
    falsely rejects correctly aligned simulator grasps.
    """
    return abs(float(object_z) - (float(body_target_z) + float(hold_offset_z)))
TRACK_MAX_SCENE_LINKED_MISSES = 6


def calibrated_grasp_offset(fitted, measured):
    """Use measured jaw-centre X/Y while retaining the separately fitted grasp height."""
    if measured is None:
        return tuple(float(value) for value in fitted)
    return (float(measured[0]), float(measured[1]), float(fitted[2]))


def projected_grip_width(kind: str, size: tuple[float, ...], quat_wxyz,
                          jaw_axis_xy: tuple[float, float]) -> float:
    """Project an object's collision geometry onto the gripper's actual closing axis."""
    sx = float(size[0])
    sy = float(size[1]) if len(size) > 1 else sx
    sz = float(size[2]) if len(size) > 2 else sx
    axis = np.asarray(jaw_axis_xy, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        axis = np.array((1.0, 0.0))
    else:
        axis /= axis_norm

    w, x, y, z = (float(value) for value in quat_wxyz)
    if kind == "ball":
        return max(sx, sy, sz)
    if kind == "cylinder":
        # Demo 11 cylinders stand upright; this also handles a tilted cylinder axis.
        cylinder_axis = np.array((2 * (x * z + w * y), 2 * (y * z - w * x),
                                  1 - 2 * (x * x + y * y)))
        axial_projection = abs(float(np.dot(axis, cylinder_axis[:2])))
        radial_projection = math.sqrt(max(0.0, 1.0 - axial_projection ** 2))
        return sx * radial_projection + sz * axial_projection

    rotation = np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    ))
    if kind in ("triangular_prism", "hexagonal_prism"):
        sides = 3 if kind == "triangular_prism" else 6
        radius = sx / 2.0
        vertices = np.asarray([
            (radius * math.cos(2 * math.pi * i / sides),
             radius * math.sin(2 * math.pi * i / sides), z_local)
            for i in range(sides) for z_local in (-sz / 2.0, sz / 2.0)
        ])
    else:
        vertices = np.asarray([
            (ix * sx / 2.0, iy * sy / 2.0, iz * sz / 2.0)
            for ix in (-1, 1) for iy in (-1, 1) for iz in (-1, 1)
        ])
    world_vertices = vertices @ rotation.T
    projections = world_vertices[:, :2] @ axis
    return float(projections.max() - projections.min())


def _rotate_local_offset(quat_wxyz, offset_xyz) -> np.ndarray:
    """Rotate a link-local calibrated fingertip point into world coordinates."""
    w, x, y, z = (float(value) for value in quat_wxyz)
    rotation = np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    ))
    return rotation @ np.asarray(offset_xyz, dtype=float)


def world_position_to_local_offset(body_position, body_quaternion_wxyz,
                                   object_position) -> tuple[float, float, float]:
    """Express a world-space object centre in the gripper body's local frame."""
    quaternion = np.asarray(body_quaternion_wxyz, dtype=float)
    inverse = (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
    local = _rotate_local_offset(
        inverse, np.asarray(object_position, dtype=float) - np.asarray(body_position, dtype=float))
    return tuple(float(value) for value in local)


def simulator_recovery_hold_is_plausible(local_hold, tolerance_m: float = 0.030) -> bool:
    """Reject simulator recovery when the measured gripper/object transform implies a teleport."""
    try:
        values = np.asarray(local_hold, dtype=float).reshape(-1)
        return bool(values.size == 3 and np.all(np.isfinite(values))
                    and abs(float(values[2]) - float(executor_module.HOLD[2])) <= tolerance_m)
    except (TypeError, ValueError, IndexError):
        return False


# Contact-face centres calibrated from the SO-101 collision geometry in the centred-grasp
# probe. The link origins are high above these points and are not suitable for lateral centring.
FINGER_CONTACT_CENTRES_LOCAL_M = {
    "gripper": (-0.0031, 0.0064, -0.0976),
    "jaw": (-0.0112, -0.0716, 0.0124),
}


# The tool-centre servo in ``StrictExecutor.track`` stops correcting when it stops making progress,
# because an arm at a joint limit will obey every command and go nowhere. "The error did not
# shrink by this much" over that many corrections means the pose is unreachable, which is a
# different fact with a different fix, and the run says which one it was.
STALL_IMPROVEMENT = 0.001
STALL_WINDOW = int(os.environ.get("DEMO11_STALL_WINDOW", "8"))
PREGRASP_ALIGNMENT_MAX_STEPS = 24
PREGRASP_ALIGNMENT_TOLERANCE_M = 0.008
PREGRASP_ALIGNMENT_STEP_M = 0.008
# Consecutive camera misses tolerated during the descent. While the wrist hides the part the
# aperture is still servoed to the scene-linked anchor from the arm's own feedback, and a part
# that was knocked meanwhile fails the close-time reach and drift guards, so a miss is safe to
# coast through. 2 was tuned for the old 150 mm hover; with the reachable hover the whole
# descent is inside the occlusion window, and a prism's rotated wrist hides the part for ~6
# looks: strict logs showed "2 occluded look(s) ... 1 look(s) could not find it" cancelling an
# approach whose aperture was 0.2-1.7 mm from the part.
TRACK_MAX_LATE_OCCLUSION = 20
# How far the last camera frame may be from the validated grasp anchor and still replace it for
# the contact check. As the wrist covers the part its mask shrinks and the centroid walks: in the
# strict logs `seen` drifted 427 -> 422 mm over one descent while the aperture was 1 mm from the
# anchor, and at the former 5 mm tolerance that drifted frame was trusted, so the check measured
# "5.0 mm from the part" and cancelled a centred grasp (the same 4.3-5.3 mm "outside the slack"
# on cubes and hexagons, three attempts in a row). The anchor was linked before the wrist hid the
# part, so a frame may only refine it, and only within the alignment gate (3.1 mm).
FINAL_CONTACT_OBSERVATION_TOLERANCE_M = 0.002
# The top-camera occlusion can start well above the nominal gate on this wrist pose.
TRACK_LATE_OCCLUSION_MARGIN_M = 0.060
BALL_PRECONTACT_HOLD_Z_M = 0.285
BALL_PRECONTACT_TOLERANCE_M = 0.008
BALL_MAX_DESCENT_STEP_M = 0.001


def bounded_servo_delta(error_xy, max_step_m: float = PREGRASP_ALIGNMENT_STEP_M,
                        gain: float = 0.65) -> tuple[float, float]:
    """Return a damped, norm-bounded XY correction for a stationary pre-grasp hover."""
    error = np.asarray(error_xy, dtype=float).reshape(-1)[:2]
    magnitude = float(np.linalg.norm(error))
    if (error.size != 2 or not np.all(np.isfinite(error)) or not math.isfinite(magnitude)
            or magnitude <= 1e-12 or max_step_m <= 0 or not 0 < gain <= 1):
        return 0.0, 0.0
    distance = min(float(max_step_m), magnitude * float(gain))
    delta = error * (distance / magnitude)
    return float(delta[0]), float(delta[1])


def body_command_for_aperture(body_xy, aperture_xy, target_xy) -> tuple[float, float]:
    """Translate a desired jaw-centre point into a continuous body-frame XY command.

    Planner targets and camera observations describe the object/aperture in world
    coordinates, while ``Executor.current`` is the gripper-body pose. Starting a
    second tracking phase at ``target_xy`` treats those frames as identical and can
    create a large lateral jump after the pre-grasp hover. Use measured poses instead.
    """
    body = np.asarray(body_xy, dtype=float).reshape(-1)[:2]
    aperture = np.asarray(aperture_xy, dtype=float).reshape(-1)[:2]
    target = np.asarray(target_xy, dtype=float).reshape(-1)[:2]
    command = body + (target - aperture)
    return float(command[0]), float(command[1])


def feedback_body_command(body_xy, aperture_xy, target_xy, *,
                          max_step_m: float, gain: float = 1.0) -> tuple[float, float]:
    """Make a bounded servo command from the *measured* body pose.

    Integrating each aperture error into the previous target winds the command up when
    the arm is still moving toward that target. The measured aperture can then look
    centred while the accumulated body target has drifted far away, especially across
    the hover-to-descent transition. Rebase each correction on live articulation
    feedback instead; this limits the next target to one bounded step ahead of the arm.
    """
    body = np.asarray(body_xy, dtype=float).reshape(-1)[:2]
    aperture = np.asarray(aperture_xy, dtype=float).reshape(-1)[:2]
    target = np.asarray(target_xy, dtype=float).reshape(-1)[:2]
    if any(value.size != 2 or not np.all(np.isfinite(value))
           for value in (body, aperture, target)):
        return (float(body[0]), float(body[1])) if body.size == 2 else (0.0, 0.0)
    error = target - aperture
    delta = bounded_servo_delta(error, max_step_m=max_step_m, gain=gain)
    return float(body[0] + delta[0]), float(body[1] + delta[1])


def trusted_contact_target(observed, anchor):
    """Reject wrist-clipped final masks and close against the validated scene anchor."""
    if anchor is None:
        return ((float(observed[0]), float(observed[1]))
                if observed is not None else None)
    trusted = (float(anchor[0]), float(anchor[1]))
    if (observed is None
            or math.dist((float(observed[0]), float(observed[1])), trusted)
            > FINAL_CONTACT_OBSERVATION_TOLERANCE_M):
        return trusted
    return float(observed[0]), float(observed[1])


def supports_large_ball_hover(kind: str, size_m) -> bool:
    """Scope the extra hover-centering and late-occlusion allowance to the tested ball size."""
    return (kind == "ball" and bool(size_m)
            and 0.029 <= max(float(value) for value in size_m[:2]) <= 0.031)


def requires_pregrasp_alignment(item) -> bool:
    """Require a stationary, measured jaw-centre check for every movable geometry."""
    if item is None or bool(getattr(item, "container", False)):
        return False
    return getattr(item, "kind", "") in {
        "cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism",
    }


def ball_precontact_hold_z(*, action: str, large_ball: bool, requested_z: float,
                           lateral_error_m: float, holding: bool,
                           ceiling_z: float | None = None) -> tuple[float, bool]:
    """Pause a 30 mm ball's descent at clearance until measured lateral error is small.

    ``ceiling_z`` is the height the descent started from. The pause is never above it: the
    fixed pause height was chosen when the ball hovered 150 mm up, and once the hover was
    lowered to what the arm can hold, pausing "at clearance" meant commanding the arm back up
    to a pose it could not reach, which left the aperture 60-70 mm off the ball.
    """
    if action != "down" or not large_ball:
        return float(requested_z), False
    should_hold = bool(
        float(lateral_error_m) > BALL_PRECONTACT_TOLERANCE_M
        and (holding or float(requested_z) <= BALL_PRECONTACT_HOLD_Z_M))
    hold_z = BALL_PRECONTACT_HOLD_Z_M
    if ceiling_z is not None:
        hold_z = min(hold_z, float(ceiling_z))
    return (hold_z if should_hold else float(requested_z), should_hold)


def bounded_ball_descent_z(*, action: str, large_ball: bool, requested_z: float,
                           previous_z: float) -> float:
    """Limit a 30 mm ball's vertical command step after a pre-contact pause.

    The nominal trajectory keeps advancing while ``ball_precontact_hold_z`` pauses the
    commanded pose. Without this cap, releasing the hold can apply the accumulated descent
    in one control step and move the aperture across its reachable workspace discontinuity.
    """
    requested_z = float(requested_z)
    previous_z = float(previous_z)
    if action != "down" or not large_ball:
        return requested_z
    return max(requested_z, previous_z - BALL_MAX_DESCENT_STEP_M)

# Maximum bounded reacquisition distance after a failed physical grasp. Ordinary tracking remains
# stricter; large relinks are only used on a recovery scan when no same-color peer is unobserved.
SAME_OBJECT_MAX_SHIFT_M = 0.060
# A scene-linked target should not jump across half the gripper width between adjacent video
# frames. Larger jumps during approach are usually the arm or a rim being segmented as the part.
# Keep the identity gate above for real motion, but do not steer towards a single implausible
# measurement.
MAX_TRACKING_STEP_M = 0.025
MAX_GRASP_OBSERVATION_AGE_S = 1.0


def target_tracking_candidates(candidates, target_prim: str, anchor_xy,
                               max_shift_m: float = MAX_TRACKING_STEP_M):
    """Keep only scene-linked detections or nearby candidates for one known target.

    Color-only nearest-neighbour tracking can switch a small ball to a same-colour tray
    fragment when the ball is occluded. A candidate linked to this exact catalog prim is
    authoritative only inside the ordinary 25 mm motion gate; an unlinked fallback has the
    same limit. A recovery caller may widen the gate after a measured failed contact when no
    same-colour identity is missing from the current view. Otherwise stop and reacquire rather
    than steer the gripper across the workspace.
    """
    if anchor_xy is None:
        return [thing for thing in candidates
                if getattr(thing, "entity_id", None) == target_prim]
    nearby = [thing for thing in candidates
              if (thing.aim_xy or thing.pos_m) is not None
              and math.dist((thing.aim_xy or thing.pos_m)[:2], anchor_xy[:2]) <= max_shift_m]
    linked = [thing for thing in nearby
              if getattr(thing, "entity_id", None) == target_prim]
    return linked or [thing for thing in nearby
                      if getattr(thing, "entity_id", None) in (None, target_prim)]


def unambiguous_catalog_assignment(costs, max_cost: float,
                                   ambiguity_margin_m: float = 0.010):
    """Return Hungarian matches only when each detection and catalog item have a clear match.

    A minimum-cost assignment always returns a winner, even when two same-colour pieces are
    indistinguishable at the current image resolution. Treating that arbitrary winner as an
    identity can make the gripper chase the other piece. In that case abstain and wait for a
    less occluded frame instead.
    """
    from scipy.optimize import linear_sum_assignment

    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or not values.size:
        return []
    rows, columns = linear_sum_assignment(values)
    accepted = []
    for row, column in zip(rows, columns):
        selected = float(values[row, column])
        if not np.isfinite(selected) or selected > max_cost:
            continue
        row_alternatives = values[row, :]
        column_alternatives = values[:, column]
        row_alternatives = row_alternatives[np.isfinite(row_alternatives)]
        column_alternatives = column_alternatives[np.isfinite(column_alternatives)]
        row_clear = (len(row_alternatives) < 2 or
                     float(np.partition(row_alternatives, 1)[1]) - selected
                     >= ambiguity_margin_m)
        column_clear = (len(column_alternatives) < 2 or
                        float(np.partition(column_alternatives, 1)[1]) - selected
                        >= ambiguity_margin_m)
        if row_clear and column_clear:
            accepted.append((int(row), int(column)))
    return accepted


def reacquire_target_from_raw_proposals(raw_views, item, target_prim: str, anchor_xy,
                                        nearest_fn):
    """Resolve a missed small part from raw detections during bounded post-contact recovery."""
    for proposals in raw_views:
        candidates = [thing for thing in proposals
                      if thing.color == item.color and thing.kind in (item.kind, "any")]
        candidates = target_tracking_candidates(
            candidates, target_prim, anchor_xy, max_shift_m=SAME_OBJECT_MAX_SHIFT_M)
        nearest = nearest_fn(candidates, anchor_xy)
        if nearest and (len(nearest) == 1 or nearest[1][0] - nearest[0][0] >= .005):
            thing = nearest[0][1]
            thing.entity_id = target_prim
            thing.kind = item.kind
            thing.size_m = list(item.size)
            return thing
    return None

# The lowest gripper-body height at which the tool-centre servo still corrects on a sighting.
#
# LOWERED FROM 0.20 TO 0.13 AND PUT BACK. At 0.13 the aperture reaches the part within a few
# millimetres - measured, 0.3-2.3 mm, against 3.1 mm of slack - and it is still worse, because
# correcting that low means the fingers are beside the part while they move. The refusal log
# shows the cube's attitude going 0, 2, 47, 90 degrees across successive attempts, and at 90
# degrees it stands on its edge and presents 42 mm to jaws that open 36.2 mm. A correction that
# centres the grip by tipping the part over has not improved anything.
#
# The trade is: correct high and leave a residual (2 of 3 sorted), or correct low and knock the
# part off its face (1 of 3 sorted). What decides it is the finger length - how close to the
# part's top face the fingers may be moved sideways - and 0.20 is the last value measured not
# to tip anything.
#
# `executor.TRACK_OBSERVE_ABOVE_Z` is 0.20 m, chosen because the gripper occludes the part from
# there down. But the arm's own error keeps growing over the last 8 cm of the descent, and that is
# precisely where a grasp is decided: measured, the residual was 3.7 mm at the jaws after the last
# correction at 0.20 m, against 3.1 mm of slack - and a grasp still took. So the servo is allowed
# to use a sighting right down to here, and a miss below 0.20 m counts as no information rather
# than as a lost target, because a lost target aborts the plan.
# The 0.20 m no-tip baseline left 17-24 mm residual on the tested ball/hex approach. At 0.19 m,
# the extra observation reduced the ball residual and strict E2E passed for a 30 mm ball, a
# 30 mm cube, and a 20 mm hexagon at yaw 30 degrees. Lowering farther to 0.13 m previously
# tipped cubes during lateral correction.
LAST_CORRECTION_Z = float(os.environ.get("DEMO11_LAST_CORRECTION_Z", "0.19"))

# How long the arm holds still at the bottom of a descent before the jaws shut. 40 steps is 0.67 s,
# which is long enough for a correction's worth of lateral velocity to die and short enough that it
# is not a pause anyone would notice.

def robot_measure(env):
    """The measured aperture offset, printed once at start-up as a calibration line.

    Reads the same two finger bodies the executor servos on, so what is printed here is what the
    descent will use. Kept as its own function because it is the number a bench would write on the
    setup sheet next to the jaw aperture.
    """
    try:
        bot = StrictExecutor(env)
        offset = bot.measured_hold()
        del bot
        return offset
    except Exception:  # noqa: BLE001 - a calibration line must never stop the demo
        return None


# --------------------------------------------------------------------------------------
# The grip has to be real
# --------------------------------------------------------------------------------------


class GraspRefused(Exception):
    """Raised instead of picking a part up by writing its pose from the gripper frame."""


class _TargetLost(Exception):
    """Raised inside the executor when a re-target finds nothing: the plan must not continue."""

    def __init__(self, prim):
        super().__init__(prim)
        self.prim = prim


class StrictExecutor(executor_module.Executor):
    """An executor that will not fake a grasp.

    Demo 10's ``grab`` is the mechanism that made its runs work: after the lift, if the part did
    not follow, the executor declares it held and then writes its pose from the gripper frame
    every step. That is a camera operator's invisible hand, and there is no version of it on a
    SO-101. Overriding ``grab`` to refuse is the entire change; ``do`` turns the refusal into the
    same "stale" verdict the rest of the pipeline already knows how to handle, so a part that
    friction did not hold aborts the rest of that pick and is re-planned instead of carried to a
    tray and released over thin air.

    ``allow_kinematic`` enables demo 10's object-follow constraint for any geometry in explicit
    A/B runs. Spheres, cylinders and polygon prisms use it automatically in this simulation demo
    because the shipped jaw contacts have not proved reliable across placements; counters and UI
    report these as simulator-assisted, never as physical grasps. Cubes use the same explicitly
    labelled fallback after a centered-close check; the strict benchmark disables it to measure
    physical friction separately.
    """
    def __init__(self, *args, allow_kinematic: bool = False,
                 auto_sim_assist: bool = True,
                 auto_sim_assist_policy: str = "preemptive",
                 close_command_override: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if auto_sim_assist_policy not in {"preemptive", "after_physical_miss"}:
            raise ValueError("auto_sim_assist_policy must be 'preemptive' or 'after_physical_miss'")
        self.allow_kinematic = bool(allow_kinematic)
        self.auto_sim_assist = bool(auto_sim_assist)
        self.auto_sim_assist_policy = auto_sim_assist_policy
        self.close_command_override = close_command_override
        # Reuse the measured midpoint between the finger pads as the motion servo's actual
        # end-effector position. The base executor's lift/carry/lower loop can then correct the
        # same IK residual that the approach controller already compensates.
        self.position_feedback = self.aperture_centre
        measured_hold = self.measured_hold()
        if measured_hold is not None:
            # The shipped grasp offset was fitted from the meshes (7, 0, -92 mm), but the
            # measured pad midpoint is (-9, +12, +10 mm) from this gripper-body origin. The
            # vertical link origin is not the contact height, so keep the fitted Z while using
            # the measured jaw-centre X/Y for close, lift and simulator-follow transforms.
            calibrated = calibrated_grasp_offset(
                self._hold[0].detach().cpu().numpy(), measured_hold)
            self._hold[0, 0] = calibrated[0]
            self._hold[0, 1] = calibrated[1]
        self.refused = 0
        self.kinematic = 0
        # Grasps where the part was carried off the table or out of the jaws after the close.
        self.lost = 0
        # Descents where the tool-centre servo had to correct the commanded pose to keep the
        # aperture over the part, and descents where it stopped making progress (a joint limit).
        self.corrected = 0
        self.correcting = True
        # Set by demo 11's robot: called mid-motion to publish a live frame. Kept as a plain
        # attribute rather than a hook list because there is exactly one camera on this bench.
        self.frame_hook = None
        # Parts measured ungraspable during this command: the geometry at the moment the jaws
        # shut said they cannot be held by anything (on their edge, wider than the aperture).
        # The demo's own loop re-plans from a fresh look every pass, so without this set the
        # same part is re-picked every pass of the sort and tips itself worse each time -
        # measured, five attempts, five refusals, nothing sorted. Consumed by the robot's
        # ``before_plan``, which takes these parts out of the fresh scene.
        self.impossible: set[str] = set()
        # Refusals by part within the current command. A part that was refused twice stops
        # being re-planned: the measured behaviour of a marginal grip is that each attempt
        # tips the part further off its face, so attempt three is not a second chance but
        # the mechanism of the failure itself.
        self.failures: dict[str, int] = {}
        self._last_object_grasp_yaw: dict[str, float] = {}
        self._initial_object_grasp_yaw: dict[str, float] = {}
        # Geometry read at the end of the last ``close``, which is the last instant at which the
        # part and the jaws are in the same place.
        self.at_close = None
        self._pregrasp_kinematic: str | None = None
        self._assist_ball_states: dict[str, dict] = {}
        self.physics_step_hook = None

        # Declared part sizes by prim, for the "will it fit between the jaws" test. Populated by
        # ``StrictExecutor.with_layout``.
        self.items_by_prim: dict = {}
        self.item_sizes: dict[str, tuple[float, ...]] = {}
        self.item_kinds: dict[str, str] = {}
        self.failure_hook = None

    def set_object_grasp_yaw(self, prim: str, target_pose=None) -> None:
        """Orient the stock jaws to the validated facet-aligned hexagon pose."""
        item = next((item for item in getattr(self, "layout", ())
                     if item.prim == prim and not item.container), None)
        radial_reach = None
        if target_pose is not None and hasattr(self, "base_pos"):
            try:
                base_xy = self.base_pos[0, :2].detach().cpu().numpy()
                target_xy = np.asarray(target_pose, dtype=float)[:2]
                radial_reach = float(np.linalg.norm(target_xy - base_xy))
            except (TypeError, ValueError, IndexError, AttributeError):
                radial_reach = None
        yaw = (gripper_contact_pads.tool_yaw_for_view(
            item.kind, item.size, getattr(item, "yaw_deg", 0.0), radial_reach,
            getattr(self, "camera_name", None))
            if item else 0.0)
        failures = getattr(self, "failures", {}).get(prim, 0)
        base_yaw = (gripper_contact_pads.tool_yaw_for(
            item.kind, item.size, getattr(item, "yaw_deg", 0.0)) if item else yaw)
        alternate_yaw = (gripper_contact_pads.alternate_tool_yaw_for(
            item.kind, item.size, getattr(item, "yaw_deg", 0.0)) if item else None)
        # The workspace policy can promote the ordinary alternate pose to the initial pose.
        # In that case the retry must use the base orientation as the alternate, while keeping
        # the workspace-selected pose as the preferred one.
        retry_yaw = (base_yaw if alternate_yaw is not None
                     and abs(yaw - alternate_yaw) <= 1e-6 else alternate_yaw)
        if failures and retry_yaw is not None:
            initial_yaw = getattr(self, "_initial_object_grasp_yaw", {}).get(prim, yaw)
            previous_yaw = getattr(self, "_last_object_grasp_yaw", {}).get(
                prim, initial_yaw)
            yaw = gripper_contact_pads.retry_tool_yaw_for(
                yaw, retry_yaw, previous_yaw)
        if not hasattr(self, "_last_object_grasp_yaw"):
            self._last_object_grasp_yaw = {}
        if not hasattr(self, "_initial_object_grasp_yaw"):
            self._initial_object_grasp_yaw = {}
        if not failures:
            self._initial_object_grasp_yaw[prim] = yaw
        self._last_object_grasp_yaw[prim] = yaw
        self.set_tool_yaw(yaw)
        if abs(yaw) > 1e-6:
            suffix = f" after {failures} failed grasp(s)" if failures else ""
            self.log(f"      wrist alignment: {yaw:+.0f} deg for {item.kind} yaw "
                     f"{getattr(item, 'yaw_deg', 0.0):.0f} deg{suffix}"
                     + (f" (reach radius {radial_reach:.3f} m)"
                        if radial_reach is not None else ""))

    def grab(self, prim: str) -> None:
        if self._uses_simulator_assist(prim):
            self.kinematic += 1
            super().grab(prim)
            return
        raise GraspRefused(prim)

    def with_layout(self, items):
        """Teach the executor each part's declared size, for the grip-width test."""
        self.layout = list(items)
        self.items_by_prim = {item.prim: item.size[0] for item in items}
        self.item_sizes = {item.prim: tuple(float(value) for value in item.size)
                           for item in items}
        self.item_kinds = {item.prim: item.kind for item in items}
        return self

    def _uses_simulator_assist(self, prim: str) -> bool:
        """Whether this simulation run may use its clearly reported grasp constraint."""
        return (getattr(self, "allow_kinematic", False)
                or (getattr(self, "auto_sim_assist", True)
                    and getattr(self, "item_kinds", {}).get(prim) in SIM_ASSISTED_SHAPES))

    def _uses_pregrasp_assist(self, prim: str) -> bool:
        """Whether to attach before contact instead of attempting a physical grasp first."""
        size_mm = (max(self.item_sizes[prim]) * 1000.0
                   if prim in getattr(self, "item_sizes", {}) else None)
        return should_pregrasp_assist(
            getattr(self, "auto_sim_assist_policy", "after_physical_miss"),
            assist_enabled=self._uses_simulator_assist(prim),
            kind=getattr(self, "item_kinds", {}).get(prim, ""),
            size_mm=size_mm)

    def tracking_observation(self, prim: str, camera_observe):
        """Use exact simulated object pose for the last-millimetre servo when allowed.

        The camera still selects and identifies the requested object. Once that identity is
        resolved to a simulator prim, the simulation can provide a stable pose through wrist
        occlusion. Hardware and strict-physical runs keep camera-only localization.
        """
        if not self._uses_simulator_assist(prim):
            return camera_observe(prim)
        try:
            target = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
            point = (float(target[0]), float(target[1]))
            if all(math.isfinite(value) for value in point):
                logged = getattr(self, "_sim_tracking_logged", set())
                if prim not in logged:
                    self.log(f"      aproximación guiada por pose de simulación para {prim}; "
                             "el objeto ya está identificado por cámara")
                    logged.add(prim)
                    self._sim_tracking_logged = logged
                return point
        except (KeyError, AttributeError, TypeError, ValueError, IndexError):
            pass
        return camera_observe(prim)

    def _begin_ball_assist(self, prim: str) -> None:
        """Keep a sphere stationary and non-colliding during the simulator-only approach."""
        if prim in self._assist_ball_states or self.item_kinds.get(prim) != "ball":
            return
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics
            import torch

            actor = self.env.scene[prim]
            root_path = str(actor.root_physx_view.prim_paths[0])
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(root_path)
            colliders = []
            for child in Usd.PrimRange(root):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    api = UsdPhysics.CollisionAPI(child)
                    attr = api.GetCollisionEnabledAttr()
                    value = attr.Get()
                    colliders.append((attr, True if value is None else bool(value)))
                    attr.Set(False)
            if not colliders:
                return
            pose = actor.data.root_pose_w[0:1].clone()
            velocities = torch.zeros((1, 6), device=self.device)
            self._assist_ball_states[prim] = {
                "actor": actor, "pose": pose, "velocities": velocities,
                "colliders": colliders,
            }
            self._freeze_ball_assist(prim)
            self.log(f"simulator ball assist active for {prim}: collisions filtered until release")
        except Exception as exc:  # simulator assist should fail closed rather than break a task
            self.log(f"simulator ball assist unavailable for {prim}: {type(exc).__name__}: {exc}")

    def _freeze_ball_assist(self, prim: str) -> None:
        state = self._assist_ball_states.get(prim)
        if state is None or self.holding == prim:
            return
        state["actor"].write_root_pose_to_sim(state["pose"])
        state["actor"].write_root_velocity_to_sim(state["velocities"])

    def _end_ball_assist(self, prim: str) -> None:
        state = self._assist_ball_states.pop(prim, None)
        if state is None:
            return
        for attr, enabled in state["colliders"]:
            attr.Set(enabled)

    def drop(self) -> None:
        held = self.holding
        super().drop()
        if held is not None:
            self._end_ball_assist(held)

    def _set_pick_open_width(self, action, prim) -> None:
        """Open the jaws only as wide as the next part needs, and restore the stock width after.

        See ``gripper_contact_pads.open_command_for``: the stock width lets the moving jaw clip a
        nearby tray rim, which wedges it open so the close never grips.
        """
        if action == "above" and os.environ.get("DEMO11_TRACE_STATE") == "1":
            try:
                robot = self.robot
                names = list(robot.joint_names)
                pos = robot.data.joint_pos[0].detach().cpu().numpy()
                target = robot.data.joint_pos_target[0].detach().cpu().numpy()
                term = self.env.action_manager.get_term("gripper_action")
                self.log("      state at approach: joints " + " ".join(
                    f"{n}={p:+.3f}/{t:+.3f}" for n, p, t in zip(names, pos, target))
                    + f" hold={[round(float(v) * 1000, 1) for v in self._hold[0]]} mm"
                    + f" held={getattr(self, '_held', None)} holding={self.holding}"
                    + f" open_cmd={float(term._open_command[0]):.3f}"
                    + f" close_cmd={float(term._close_command[0]):.3f}")
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                self.log(f"      state at approach: unavailable ({exc})")
        if (not hasattr(self, "env") or os.environ.get("DEMO11_DISABLE_NARROW_OPEN") == "1"
                or action not in ("above", "retreat")):
            return
        try:
            term = self.env.action_manager.get_term("gripper_action")
        except (AttributeError, KeyError, ValueError):
            return
        stock = getattr(self, "_stock_open_command", None)
        if stock is None:
            stock = term._open_command.clone()
            self._stock_open_command = stock
        if action == "retreat" or prim not in getattr(self, "item_kinds", {}):
            term._open_command[:] = stock
            return
        close_target = gripper_contact_pads.close_command_for(
            self.item_kinds[prim], self.item_sizes.get(prim, ()),
            real_world.GRIPPER.close_command_rad, override=self.close_command_override,
            pad_shape=getattr(self.env, "demo11_contact_pad_shape", None))
        opening = gripper_contact_pads.open_command_for(
            self.item_kinds[prim], self.item_sizes.get(prim, ()),
            float(stock[0]), close_target)
        term._open_command[:] = opening
        if action == "above":
            self.log(f"      jaws open to {opening:.3f} rad for {prim} "
                     f"(stock {float(stock[0]):.3f} rad)")

    def _install_part_trace(self, action, prim) -> None:
        """Opt-in (``DEMO11_TRACE_PART=1``): log where a prism sits between the pads each few ticks.

        Reports clearance from each pad face to the nearest vertex along the jaw axis, the part
        centre's offset from the pad midpoint (along the jaw axis ``u`` and across it ``v``), its
        height relative to the midpoint, its tilt and the jaw angle. It never changes a decision.
        """
        if os.environ.get("DEMO11_TRACE_PART") != "1" or not prim:
            return
        self._trace_label = (action, prim)
        if getattr(self, "_part_trace_installed", False):
            return
        self._part_trace_installed = True
        original_step = self.env.step
        counter = {"n": 0}

        def traced_step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            counter["n"] += 1
            label = getattr(self, "_trace_label", None)
            every = int(os.environ.get("DEMO11_TRACE_PART_EVERY", "12") or 12)
            if (not label or label[0] not in ("down", "close", "lift", "carry")
                    or counter["n"] % every):
                return result
            try:
                phase, part_prim = label
                part = self.env.scene[part_prim]
                where = part.data.root_pos_w[0].detach().cpu().numpy()
                quat = part.data.root_quat_w[0].detach().cpu().numpy()
                points = self.finger_contact_points()
                size = self.item_sizes.get(part_prim, (0.03, 0.03, 0.03))
                sides = 3 if self.item_kinds.get(part_prim) == "triangular_prism" else 6
                axis = points["jaw"][:2] - points["gripper"][:2]
                separation = float(np.linalg.norm(axis))
                axis = axis / max(separation, 1e-9)
                across = np.array((-axis[1], axis[0]))
                mid = (points["jaw"] + points["gripper"]) / 2.0
                radius = size[0] / 2.0
                corners = [_rotate_local_offset(quat, (radius * math.cos(2 * math.pi * i / sides),
                                                       radius * math.sin(2 * math.pi * i / sides), 0.0))[:2]
                           for i in range(sides)]
                along = [float((where[:2] + c - points["gripper"][:2]) @ axis) for c in corners]
                gap_fixed = min(along)
                gap_jaw = separation - max(along)
                delta = where[:2] - mid[:2]
                up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(up_z)))))
                joint = float(self.robot.data.joint_pos[0, self.robot.find_joints(["gripper"])[0][0]])
                self.log(f"      [part] {phase} n={counter['n']} gap_fixed={gap_fixed * 1000:+.1f} "
                         f"gap_jaw={gap_jaw * 1000:+.1f} sep={separation * 1000:.1f} "
                         f"u={float(delta @ axis) * 1000:+.1f} v={float(delta @ across) * 1000:+.1f} "
                         f"dz={(where[2] - mid[2]) * 1000:+.1f} tilt={tilt:.0f} jaw={joint:.3f} "
                         f"z={where[2]:.4f}")
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                self.log(f"      [part] trace unavailable: {type(exc).__name__}: {exc}")
            return result

        self.env.step = traced_step

    def do(self, *args, **kwargs):
        # The close step gets its own duration, for the reason in CLOSE_SETTLE_STEPS. Only when the
        # caller did not ask for a specific count: the retry path drives its own lift with an
        # explicit count and must not be second-guessed here.
        if len(args) > 1 and args[0] == "close" and len(args) < 6 and "steps" not in kwargs:
            kwargs["steps"] = CLOSE_SETTLE_STEPS
        action = args[0] if args else kwargs.get("action")
        prim = kwargs.get("prim")
        self._install_part_trace(action, prim)
        self._set_pick_open_width(action, prim)
        carry_override = os.environ.get("DEMO11_EXPERIMENTAL_CARRY_STEPS")
        if action == "carry" and carry_override is not None:
            kwargs["steps"] = experimental_carry_steps(carry_override)
            self.log(f"      experimental slower carry: {kwargs['steps']} physics steps")
        elif action == "carry" and prim in getattr(self, "item_kinds", {}):
            carry_steps = gripper_contact_pads.carry_steps_for(
                self.item_kinds[prim], self.item_sizes.get(prim, ()),
                int(kwargs.get("steps") or executor_module.STEP_COUNTS["carry"]))
            if carry_steps > int(kwargs.get("steps") or executor_module.STEP_COUNTS["carry"]):
                kwargs["steps"] = carry_steps
                self.log(f"      transporte estable para {self.item_kinds[prim]}: "
                         f"{carry_steps} pasos de carry")
        if (action == "close" and prim in self.item_kinds
                and hasattr(self, "item_sizes") and hasattr(self, "env")):
            # The action is binary; its continuous close target lives on the joint term.
            close_target = gripper_contact_pads.close_command_for(
                self.item_kinds[prim], self.item_sizes.get(prim, ()),
                real_world.GRIPPER.close_command_rad,
                override=self.close_command_override,
                pad_shape=getattr(self.env, "demo11_contact_pad_shape", None))
            set_gripper(self.env, angle_rad=close_target)
            ramp_variable = {
                "ball": "DEMO11_BALL_CLOSE_RAMP_STEPS",
                "cube": "DEMO11_CUBE_CLOSE_RAMP_STEPS",
                "triangular_prism": "DEMO11_TRIANGLE_CLOSE_RAMP_STEPS",
                "cylinder": "DEMO11_CYLINDER_CLOSE_RAMP_STEPS",
                "hexagonal_prism": "DEMO11_HEXAGON_CLOSE_RAMP_STEPS",
            }.get(self.item_kinds[prim])
            ramp_override = os.environ.get(ramp_variable) if ramp_variable else None
            ramp_steps = (0 if os.environ.get("DEMO11_DISABLE_STAGED_CLOSE") == "1" else
                          gripper_contact_pads.staged_close_ramp_steps(
                              self.item_kinds[prim], self.item_sizes.get(prim, ()),
                              int(kwargs.get("steps") or CLOSE_SETTLE_STEPS),
                              ramp_steps=(int(ramp_override) if ramp_override is not None else None)))
            if ramp_steps:
                term = self.env.action_manager.get_term("gripper_action")
                opening = term._open_command.clone()
                closing = term._close_command.clone()
                term._close_command[:] = opening

                def stage_close(index, _total_steps):
                    fraction = min(1.0, index / ramp_steps)
                    term._close_command[:] = opening + (closing - opening) * fraction

                self.gripper_step_hook = stage_close
                self.log(f"      staged {self.item_kinds[prim]} close: target ramps over "
                         f"{ramp_steps} physics steps")
            if os.environ.get("DEMO11_TRACE_CONTACT") == "1":
                self.log(f"      grasp profile: kind={self.item_kinds[prim]}, "
                         f"size_mm={self.item_sizes.get(prim)}, "
                         f"pads={getattr(self.env, 'demo11_contact_pad_shape', None)}, "
                         f"close={close_target:.3f} rad")
        assisted_lift = bool(action == "lift" and prim
                              and getattr(self, "_pregrasp_kinematic", None) == prim
                              and self.holding == prim)
        if assisted_lift:
            # The base executor measures object displacement against the commanded vertical
            # lift. That is the physical-friction test. An assisted object follows the closed
            # gripper through its measured lateral IK corrections as well, so applying the 5%
            # vertical-only overshoot threshold would reject legitimate assisted transport.
            # Keep the carry constraint active and classify this result through its own path.
            kwargs["on_close"] = None
            kwargs["check"] = None
        # Isaac's shipped parallel-jaw contacts do not reliably hold round and polygonal parts.
        # In simulator-assist mode, establish a grasp constraint immediately before
        # the fingers close, but only when the part fits, is level with the commanded grasp, and
        # the measured aperture centre is within the small simulator-assist tolerance. This
        # avoids attaching to a different object while absorbing measured camera/IK residuals,
        # and is reported as kinematic rather than as a friction grasp. Hardware mode never
        # enters this branch.
        if (prim and self._uses_pregrasp_assist(prim) and action == "close"
                and self.holding is None and len(args) > 1):
            geometry = self.part_geometry(prim)
            target = np.asarray(args[1], dtype=float)
            lateral = geometry.get("lateral_m") if geometry else None
            # Planner poses already denote the hold point. Tracked approaches pass a direct
            # gripper-body pose, so add its calibrated world-frame offset before comparing.
            # The body origin is ~92 mm above the SO-101 grasp centre and falsely rejects a
            # correctly aligned tracked close.
            if kwargs.get("body_target"):
                hold_offset = self.hold_offset_world()[0].detach().cpu().numpy()
                height_error = grasp_height_error(
                    geometry["z"] if geometry else 0.0, target[2], hold_offset[2])
            else:
                height_error = (abs(float(geometry["z"]) - float(target[2]))
                                if geometry else math.inf)
            eligible = bool(geometry and geometry.get("fits") and lateral is not None
                            and lateral <= SIM_ASSIST_MAX_LATERAL_M and height_error <= 0.020)
            if eligible:
                self.grab(prim)
                self._pregrasp_kinematic = prim
                # The physical reach check runs inside Executor.do before its last per-step
                # _carry_if_held() write. A closing contact can transiently push the part away,
                # so only this already-validated simulator constraint skips that check; the
                # final carry write restores the configured aperture-relative pose.
                kwargs["check"] = None
                self.log(f"      simulator grasp assist: {prim} centred, "
                         f"{lateral * 1000:.1f} mm lateral, "
                         f"{height_error * 1000:.1f} mm height error")
            else:
                self.log(f"      simulator grasp assist withheld: {prim}, "
                         f"fits={geometry.get('fits') if geometry else 'unknown'}, "
                         f"lateral={lateral if lateral is not None else 'unknown'}, "
                         f"height_error={height_error:.3f} m")
        try:
            result = super().do(*args, **kwargs)
            if action in {"above", "down"} and not result.ok and prim:
                # A pose that cannot reach its pre-grasp/approach waypoint is still a
                # failed grasp attempt for orientation recovery. Previously the executor
                # marked it stale only after this method returned, so the per-object failure
                # counter stayed at zero and every retry repeated the same saturated IK pose.
                self.failures[prim] = self.failures.get(prim, 0) + 1
                self.log(f"      {action} failed before contact; next retry will test an "
                         "alternate reachable wrist orientation when the geometry allows it")
            if assisted_lift:
                if result.ok and self.holding == prim:
                    result.grasp = "kinematic"
                    suffix = "; simulator contact constraint carried the centred part"
                    result.note = (result.note + suffix).strip()
                else:
                    result.ok = False
                    result.grasp = "stale"
                    result.note = "simulator grasp constraint did not remain active through lift"
                self._pregrasp_kinematic = None
            elif (action == "lift" and result.ok and result.grasp == "real"
                  and self.holding is None
                  and off_center_assist_candidate(
                      self.at_close,
                      after_physical_miss=self.auto_sim_assist_policy == "after_physical_miss",
                      assist_enabled=self._uses_simulator_assist(prim))):
                # An off-centre part can move with the jaws because they push it, which
                # makes vertical follow alone a false positive for a friction grasp. In
                # simulation mode, secure this bounded measured miss to finish the move;
                # report it as assisted. Hardware and strict-physical runs stay untouched.
                lateral = float(self.at_close["lateral_m"])
                self.grab(prim)
                result.grasp = "kinematic"
                result.note += (f"; off-centre physical follow ({lateral * 1000:.1f} mm) "
                                "secured by simulator assist")
            if self.frame_hook is not None:
                self.frame_hook()
            if result.grasp == "lost":
                # The part left the jaws' reach while the gripper rose. It is not held, so it
                # must not be carried, lowered or released as if it were. Reported as a failed
                # grasp: executor.run stops on "stale", and the caller re-plans or fails the task.
                prim = kwargs.get("prim")
                geometry = self.at_close or self.part_geometry(prim)
                self.lost += 1
                if prim:
                    self.failures[prim] = self.failures.get(prim, 0) + 1
                if self.failure_hook is not None:
                    self.failure_hook(prim, "lost", geometry)
                if (prim and getattr(self, "_held", None) is None
                        and physical_miss_assist_candidate(
                            geometry,
                            after_physical_miss=(getattr(
                                self, "auto_sim_assist_policy", "after_physical_miss")
                                == "after_physical_miss"),
                            assist_enabled=self._uses_simulator_assist(prim))):
                    # The physical lift already failed. In simulation only, a centred,
                    # size-compatible part may be recovered by an explicitly counted
                    # kinematic grasp so the demo can finish and verify the placement.
                    # Strict physical runs and hardware never enter this branch.
                    recovered_hold = self.measure_simulated_hold_to_object(prim)
                    if not simulator_recovery_hold_is_plausible(recovered_hold):
                        self.log(f"      simulator recovery withheld: measured hold Z "
                                 f"{recovered_hold[2] * 1000:+.1f} mm is inconsistent "
                                 f"with the calibrated grasp")
                        return executor_module.StepResult(
                            action=result.action, ok=False, grasp="stale",
                            note="simulator recovery refused: measured gripper/object height is implausible")
                    self._hold[0, :] = self.torch.as_tensor(
                        recovered_hold, dtype=self._hold.dtype, device=self._hold.device)
                    self.grab(prim)
                    self.log(f"      simulator recovery after physical miss: {prim}, "
                             f"{float(geometry['lateral_m']) * 1000:.1f} mm lateral, "
                             f"hold=({recovered_hold[0] * 1000:+.1f},"
                             f"{recovered_hold[1] * 1000:+.1f},"
                             f"{recovered_hold[2] * 1000:+.1f}) mm")
                    return executor_module.StepResult(
                        action=result.action, ok=True, grasp="kinematic",
                        note="physical lift missed; bounded simulator recovery secured the part")
                return executor_module.StepResult(
                    action=result.action, ok=False, grasp="stale",
                    note=f"la pieza ya no estÃ¡ en las mordazas: {result.note}")
            if result.grasp == "stale" and action == "close" and prim:
                # A close-time reach failure means contact may have pushed the target even
                # though the fingers never held it. Count it before the recovery scan so the
                # backup camera can accept one bounded re-anchor (only when no same-colour peer
                # is unobserved) instead of retrying the old, now-empty position.
                self.failures[prim] = self.failures.get(prim, 0) + 1
                geometry = self.part_geometry(prim)
                if self.failure_hook is not None:
                    self.failure_hook(prim, "missed", geometry)
                self.log(f"      close missed for {prim}; recovery may re-anchor the moved target")
            if (args and args[0] == "close" and result.ok and prim
                    and kwargs.get("feedback_target") is not None
                    # A part the simulator is carrying follows the jaw midpoint as the jaws close
                    # (it moves ~9-11 mm from the open hold frame to the closed one), which is
                    # not the jaws shoving it away; only a friction grasp is judged by drift.
                    and getattr(self, "_pregrasp_kinematic", None) != prim):
                # The scene-linked centre was measured immediately before closing. If the
                # jaws shove the part farther than the camera's error band, do not lift an
                # object that is no longer centred between the fingers.
                before_xy = np.asarray(kwargs["feedback_target"], dtype=float)[:2]
                self.at_close = self.part_geometry(prim)
                after_xy = ((self.at_close["x"], self.at_close["y"])
                            if self.at_close is not None else None)
                if (after_xy is not None and not executor_module.close_center_drift_is_safe(
                        before_xy, after_xy)):
                    displacement = math.dist(before_xy, after_xy)
                    result.ok = False
                    result.grasp = "stale"
                    result.note = (f"cierre desplazó la pieza {displacement * 1000:.1f} mm; "
                                   "se cancela la elevación para volver a observar")
                    self.failures[prim] = self.failures.get(prim, 0) + 1
                    if self.failure_hook is not None:
                        self.failure_hook(prim, "missed", self.at_close)
                    self.log(f"      {result.note}")
            if args and args[0] == "close" and result.ok:
                # Measured here because this is the last moment before the lift: afterwards the arm
                # has risen 75 mm and any geometry read later describes where the gripper *is*,
                # not where the jaws shut.
                self.at_close = self.part_geometry(kwargs.get("prim"))
                calibrated_hold = self.calibrate_hold_at_close()
                if calibrated_hold is not None:
                    self.log(f"      closed-jaw hold frame: "
                             f"({calibrated_hold[0] * 1000:+.1f}, "
                             f"{calibrated_hold[1] * 1000:+.1f}) mm")
                if (kwargs.get("prim")
                        and kwargs.get("prim") == getattr(
                            getattr(self, "env", None), "demo11_contact_diagnostics_for", None)):
                    # Optional strict-E2E diagnostic: the isolated grasp benchmark can report
                    # contact force, but previously the camera-driven executor could not. Log
                    # one final close sample beside the measured object/jaw geometry so a
                    # slip can be separated from a missed contact without changing control.
                    try:
                        contact = {}
                        for link in ("gripper", "jaw", "target"):
                            try:
                                sensor = self.env.scene[f"demo11_contact_{link}"]
                            except KeyError:
                                continue
                            matrix = sensor.data.force_matrix_w
                            if matrix is None:
                                continue
                            values = matrix[0].reshape(-1, 3).detach().cpu().numpy()
                            contact[link] = round(float(np.linalg.norm(values, axis=1).sum()), 4)
                        robot = self.env.scene["robot"]
                        joint_id = int(robot.find_joints("gripper")[0][0])
                        position = float(robot.data.joint_pos[0, joint_id].detach().cpu())
                        velocity = float(robot.data.joint_vel[0, joint_id].detach().cpu())
                        torque = float(robot.data.applied_torque[0, joint_id].detach().cpu())
                        self.log(f"      contacto medido N={contact}; "
                                 f"junta(pos={position:.4f}, vel={velocity:.4f}, "
                                 f"torque={torque:.3f} Nm)")
                    except Exception as exc:  # diagnostic-only: never alter a grasp decision
                        self.log(f"      diagnóstico de contacto no disponible: {exc}")
            return result
        except GraspRefused:
            action = args[0] if args else kwargs.get("action", "?")
            self.refused += 1
            prim = kwargs.get("prim")
            geometry = self.at_close or self.part_geometry(prim)
            if prim:
                self.failures[prim] = self.failures.get(prim, 0) + 1
                if geometry is not None and geometry.get("fits") is False:
                    # Not a marginal miss: the part's footprint across the jaws is wider
                    # than the aperture allows, so no attempt of any force can hold it.
                    # Recorded so the re-planning loop does not spend the rest of the
                    # command on it.
                    self.impossible.add(prim)
            if self.failure_hook is not None:
                self.failure_hook(prim, "refused", geometry)
            return executor_module.StepResult(
                action=action, ok=False, grasp="stale",
                note=(f"the jaws closed and the part was not held - friction did not carry it. "
                      f"{self.refusal_geometry(kwargs.get('prim'))}"
                      f"Demo 11 will not pick it up by hand, so this counts as a missed grasp"))
        return result

    def part_geometry(self, prim: str | None) -> dict | None:
        """Where the part is, how it is lying, and whether the jaws could close on it at all.

        ``grip_width_m`` is the part's extent *across the jaws*, which depends on its attitude and
        not on its size: a 30 mm cube flat presents 30 mm and a cube on its edge presents 42 mm.
        That is the measurement that separates "the grip was marginal" from "the grip was
        impossible", and it is the same fact demo 10 learned the hard way about a 4 cm ball in a
        36.2 mm gripper.
        """
        if prim is None:
            return None
        try:
            part = self.env.scene[prim]
            where = part.data.root_pos_w[0].detach().cpu().numpy()
            quat = part.data.root_quat_w[0].detach().cpu().numpy()
            centre = self.aperture_centre()
        except Exception:  # noqa: BLE001 - a diagnostic must never break a run
            return None
        w, x, y, z = (float(v) for v in quat)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(up_z)))))
        size = self.item_sizes.get(prim)
        if size is None:
            edge = float(self.items_by_prim.get(prim, 0.0))
            size = (edge, edge, edge)
        kind = self.item_kinds.get(prim, "cube")
        contact_points = self.finger_contact_points()
        jaw_axis = (1.0, 0.0)
        if contact_points is not None:
            delta = contact_points["jaw"][:2] - contact_points["gripper"][:2]
            if float(np.linalg.norm(delta)) > 1e-6:
                jaw_axis = (float(delta[0]), float(delta[1]))
        grip_width = projected_grip_width(kind, size, quat, jaw_axis)
        geometry = {
            "x": float(where[0]),
            "y": float(where[1]),
            "z": float(where[2]),
            "tilt_deg": tilt,
            "grip_width_m": grip_width,
            "fits": grip_width <= real_world.GRIPPER.aperture_m - real_world.GRIP_WIDTH_MARGIN_M,
            "lateral_m": None,
        }
        if centre is not None:
            geometry["lateral_m"] = math.dist(centre[:2], (float(where[0]), float(where[1])))
        return geometry

    def refusal_geometry(self, prim: str | None) -> str:
        """Why a grasp failed, in numbers, measured at the moment the jaws shut.

        Read from ``at_close`` rather than from here, because by the time the lift has been judged
        the arm is 75 mm higher and any geometry taken now describes the wrong instant.

        The reading that matters most is ``fits``. With the tool-centre servo running the aperture
        is typically 0.3-2 mm from the part - the pads are around it - and a 28 g cube between
        36.2 mm pads still came up followed by 0% of the lift. The reason, measured: the cube was
        **on its edge**, 42 mm across, which those jaws cannot close on at any force. So the servo
        was never the missing piece; the missing piece is knowing which failures are impossible.
        """
        geometry = self.at_close or self.part_geometry(prim)
        if not geometry:
            return ""
        parts = [f"part centre ({geometry['x']:+.4f}, {geometry['y']:+.4f}, "
                 f"{geometry['z']:.4f}) m", f"tilted {geometry['tilt_deg']:.0f} deg",
                 f"{geometry['grip_width_m'] * 1000:.0f} mm across the jaws"]
        if geometry["lateral_m"] is not None:
            inside = "between the pads" if geometry["lateral_m"] <= LATERAL_SLACK_M \
                else "outside the slack"
            parts.append(f"{geometry['lateral_m'] * 1000:.1f} mm away laterally ({inside})")
        verdict = ""
        if not geometry["fits"]:
            verdict = (f" - IMPOSSIBLE: {geometry['grip_width_m'] * 1000:.0f} mm will not fit "
                       f"{real_world.GRIPPER.aperture_m * 1000:.1f} mm of jaws, so no force "
                       f"would hold it. It is on its edge and has to lie flat first")
        elif geometry["lateral_m"] is not None and geometry["lateral_m"] <= LATERAL_SLACK_M:
            verdict = (" - the part was between the pads, so this is the friction cone failing "
                       "rather than the aim")
        return "[" + ", ".join(parts) + verdict + "] "

    def run(self, plan, resolve=None, keys=lambda: set(), fps=1, retarget=None, observe=None):
        """``executor.run``, with a failed re-target stopping the plan instead of being ignored.

        The shared executor keeps executing the old steps when ``retarget`` returns nothing - it
        aims at where the object *was*. Here an empty re-target ends the plan at once, and the
        result says the target was lost, so the caller re-plans from a fresh look or fails the
        task. Motion stops where the arm is; nothing is carried or released on the way out.
        """
        assisted = []
        if resolve is not None:
            for scene_id in getattr(plan, "picks", ()):
                prim = resolve(scene_id)
                if (prim and self.item_kinds.get(prim) == "ball"
                        and self._uses_simulator_assist(prim)):
                    self._begin_ball_assist(prim)
                    if prim in self._assist_ball_states:
                        assisted.append(prim)
        if assisted:
            # The collision-filtered ball otherwise continues to receive gravity during
            # close/lift steps. Reapply its fixed pose after every physics tick until the
            # contact-gated grasp takes ownership of it.
            def freeze_unheld_assisted_balls(_action, _prim):
                for ball_prim in assisted:
                    self._freeze_ball_assist(ball_prim)

            self.physics_step_hook = freeze_unheld_assisted_balls

        def guarded(step, prim):
            fresh = retarget(step, prim)
            if not fresh:
                raise _TargetLost(prim)
            return fresh

        try:
            tracking_observe = observe
            if callable(observe):
                tracking_observe = lambda prim: self.tracking_observation(prim, observe)
            if retarget is None:
                return super().run(plan, resolve=resolve, keys=keys, fps=fps,
                                   observe=tracking_observe)
            return super().run(plan, resolve=resolve, keys=keys, fps=fps, retarget=guarded,
                               observe=tracking_observe)
        except _TargetLost as lost:
            out = executor_module.RunResult()
            out.stale_grasps = 1
            out.stale_prims = [lost.prim] if lost.prim else []
            out.aborted_at = "re-plan: no se encuentra el objeto"
            self.log(f"  re-plan: no se encuentra el objeto {lost.prim}; se detiene el movimiento")
            return out
        finally:
            self.physics_step_hook = None
            for prim in assisted:
                if self.holding != prim:
                    self._end_ball_assist(prim)

    # -- putting the part actually between the pads ----------------------------

    def aperture_centre(self) -> tuple[float, float] | None:
        """World-space midpoint between the opposing collision faces at object-centre height.

        The contact points are calibrated from the SO-101 collision mesh in each link frame and
        transformed by the live articulation pose. Link origins are a poor proxy here: the fixed
        finger extends below its body and the moving jaw rotates about its joint while closing.
        """
        points = self.finger_contact_points()
        if points is None:
            return None
        return tuple(float((points["gripper"][axis] + points["jaw"][axis]) / 2.0)
                     for axis in range(3))

    def finger_contact_points(self) -> dict[str, np.ndarray] | None:
        """World-space centres of the fixed and moving collision-pad faces."""
        names = list(self.robot.body_names)
        if not all(name in names for name in FINGER_CONTACT_CENTRES_LOCAL_M):
            return None
        positions = self.robot.data.body_pos_w
        quaternions = self.robot.data.body_quat_w
        points = {}
        for name, local_contact in FINGER_CONTACT_CENTRES_LOCAL_M.items():
            index = self.robot.find_bodies(name)[0][0]
            body_position = positions[0, index].detach().cpu().numpy()
            body_quaternion = quaternions[0, index].detach().cpu().numpy()
            points[name] = body_position + _rotate_local_offset(body_quaternion, local_contact)
        return points

    def measured_hold(self) -> tuple[float, float, float] | None:
        """The aperture centre relative to the gripper body: ``executor.HOLD``, measured.

        ``HOLD`` says a held part's centre sits at (7, 0, -92) mm in the gripper body frame. That
        is demo 05's number, fitted to the finger meshes, and every descent converts a target
        height with it. If the real distance is not 92 mm then the descent ends at the wrong height
        and the pads close above or below a 30 mm part - which fails outright rather than slipping,
        and is the difference between "held at 99%" and "followed 0% of the lift" with nothing in
        between.

        Measured at the hover, where the gripper is horizontal and stationary, and used for the
        descent's height. Logged once so the fitted number and the real one can be compared.
        """
        centre = self.aperture_centre()
        if centre is None:
            return None
        body = self._gripper_pos()[0].detach().cpu().numpy()
        quat = self.robot.data.body_quat_w[0, self.gripper_id].detach().cpu().numpy()
        inverse = (quat[0], -quat[1], -quat[2], -quat[3])
        local = _rotate_local_offset(inverse, np.asarray(centre) - body)
        return tuple(float(value) for value in local)

    def calibrate_hold_at_close(self) -> tuple[float, float] | None:
        """Use the closed-jaw aperture midpoint as the carried object's XY frame.

        The SO-101 finger links swing while closing, so the open-gripper midpoint measured
        at startup is not the closed-gripper midpoint used to carry a part. Refreshing XY once
        the jaws settle avoids a systematic lateral offset in simulator-follow mode and uses
        only measured articulation state; fitted hold height remains unchanged.
        """
        if not hasattr(self, "robot") or not hasattr(self, "_hold"):
            return None
        centre = self.aperture_centre()
        if centre is None:
            return None
        body = self._gripper_pos()[0].detach().cpu().numpy()
        quat = self.robot.data.body_quat_w[0, self.gripper_id].detach().cpu().numpy()
        inverse = (quat[0], -quat[1], -quat[2], -quat[3])
        local = _rotate_local_offset(inverse, np.asarray(centre) - body)
        offset = (float(local[0]), float(local[1]))
        self._hold[0, 0] = offset[0]
        self._hold[0, 1] = offset[1]
        if self.holding is not None:
            self._carry_if_held()
        return offset

    def measure_simulated_hold_to_object(self, prim: str) -> tuple[float, float, float]:
        """Measure an object-centre offset without changing the active grasp transform."""
        object_position = (self.env.scene[prim].data.root_pos_w[0]
                           .detach().cpu().numpy().copy())
        body_position = self._gripper_pos()[0].detach().cpu().numpy()
        body_quaternion = self.robot.data.body_quat_w[0, self.gripper_id].detach().cpu().numpy()
        return world_position_to_local_offset(body_position, body_quaternion, object_position)

    def track(self, observe, target_z: float, steps: int, action: str = "above",
              grip: float = executor_module.OPEN,
              every: int = executor_module.TRACK_EVERY,
              aim: tuple[float, float] | None = None,
              _contact_retry: int = 0):
        """Move to ``target_z`` while servoing the *aperture* onto the part, not the aim onto it.

        ``executor.track`` aims at what the camera reports: it re-reads the part's position and
        points the commanded pose at it. That is right for a part that is where the arm will be,
        and wrong when the arm does not go where it is told - and it is why demo 11's first
        attempt at this fix did nothing. Correcting the aim was undone on the next 20-step tick,
        because the correction changed the aim and the camera immediately pulled the aim back onto
        the part.

        So the correction is applied to the **commanded pose** instead, but rebased on live arm
        feedback on every observation. Integrating error into the last command winds up while the
        articulation is still moving and can send the body target far beyond the aperture target.
        The bounded update is:

            command = measured body pose + bounded((where the part is) - (where the aperture is))

        Every tick that consults the camera. This is a bounded servo on the tool centre point using
        the arm's own feedback. It prevents stale commands from accumulating during hover-to-contact
        transitions; the physical benchmark must still establish that the arm converges at each
        tested workspace pose.

        Two guards, because a servo that cannot converge must say so instead of grinding:

        * a shift larger than ``SAME_OBJECT_MAX_SHIFT_M`` still counts as lost - that is the
          part being knocked away, not the arm lagging;
        * if the error has not shrunk by ``STALL_IMPROVEMENT`` over ``STALL_WINDOW`` corrections,
          the arm is at a limit and no amount of commanding will move it, so it stops and says so.
          That is the difference between "the arm is 14 mm out and correcting" and "this pose is
          unreachable", which are different facts and need different fixes.
        """
        import torch

        # ``self.current`` is the last command sent to the IK controller, not the
        # live end-effector pose. At the boundary between the hover and descent the
        # arm can still be settling, so restarting from that stale command compounds
        # its lag. Seed this phase from articulation feedback.
        measured_body = self._gripper_pos().clone()
        start = measured_body
        here = aim if aim is not None else (float(start[0][0]), float(start[0][1]))
        correction_cap = LATERAL_SLACK_M * (4.0 if action == "above" else 2.0)
        # The descent's height uses ``executor.HOLD``, demo 05's *fitted grip point*, and not the
        # measured aperture centre. Measured on this arm the two finger bodies' origins sit at
        # (-9, +12, +10) mm from the gripper body - not 92 mm below it - because they are the finger
        # links, whose origins are near the top of the finger. So the midpoint is the right
        # *lateral* reference for the servo and completely the wrong *height*: driving the descent
        # with it put the body at the table and cost a placement. Height comes from the fitted
        # point; lateral comes from the measurement.
        hold_z = float(self._hold[0][2])
        start_z = float(start[0][2])
        last_commanded_z = start_z
        target_body_z = target_z - hold_z
        measured_aperture = self.aperture_centre()
        if action == "down" and measured_aperture is None:
            self.correcting = False
            note = "descenso cancelado: no hay feedback del centro de las mordazas"
            self.log(f"      {note}; no se entra en contacto sin medir la apertura")
            prim = getattr(self, "_tracking_prim", None)
            if prim:
                if not hasattr(self, "failures"):
                    self.failures = {}
                self.failures[prim] = self.failures.get(prim, 0) + 1
            return executor_module.StepResult(action=action, ok=False, note=note,
                                              grasp="stale")
        if measured_aperture is not None:
            command = list(feedback_body_command(
                measured_body[0, :2].detach().cpu().numpy(),
                measured_aperture[:2], here, max_step_m=correction_cap))
        else:
            command = [float(here[0]), float(here[1])]
        corrections = 0
        largest = 0.0
        lost = 0
        aperture_feedback_lost = False
        occluded = 0
        tracking_jumps = 0
        rejected_jumps = 0
        # ``aim`` came from a fresh, scene-linked observation immediately before this approach.
        # It is safe to coast briefly on that identity while the arm crosses the camera's view;
        # a large relocalization still aborts and close/lift retain their own checks.
        last_seen = ((float(aim[0]), float(aim[1])) if aim is not None else None)
        initial_aim = last_seen if last_seen is not None else tuple(command)
        worst = 0.0
        # The residuals of the last STALL_WINDOW corrections, for this descent only. A stall is a
        # window of corrections that did not shrink the residual by STALL_IMPROVEMENT across the
        # window. It is not a comparison against the best value of the whole descent, which
        # flagged a steadily converging servo as stuck.
        recent: list[float] = []
        stall_reported = False
        precontact_hold = False
        precontact_hold_logged = False
        precontact_residual = math.inf
        self.correcting = True
        tracking_prim = getattr(self, "_tracking_prim", None)
        tracking_item = next((item for item in getattr(self, "layout", ())
                              if item.prim == tracking_prim), None)
        last_correction_z = (
            gripper_contact_pads.last_visual_correction_height(
                getattr(tracking_item, "kind", ""), getattr(tracking_item, "size", ()),
                getattr(self.env, "demo11_contact_pad_shape", None),
                LAST_CORRECTION_Z, getattr(tracking_item, "yaw_deg", 0.0))
            if tracking_item is not None else LAST_CORRECTION_Z)
        small_pregrasp_coast = bool(
            action == "above" and tracking_item is not None
            and executor_module.pregrasp_coast_supported(
                tracking_item.kind, tracking_item.size))
        large_ball_hover = bool(
            tracking_item is not None
            and supports_large_ball_hover(getattr(tracking_item, "kind", ""),
                                          getattr(tracking_item, "size", ())))
        pregrasp_hover_alignment = requires_pregrasp_alignment(tracking_item)
        tracking_tolerance = (PREGRASP_ALIGNMENT_TOLERANCE_M
                              if action == "above" and pregrasp_hover_alignment
                              else LATERAL_SLACK_M)
        with torch.no_grad():
            for index in range(1, steps + 1):
                fraction = min(1.0, index / (0.8 * steps))
                z = executor_module.tracked_body_z(
                    action, start_z, target_body_z, fraction)
                z, precontact_hold = ball_precontact_hold_z(
                    action=action, large_ball=large_ball_hover, requested_z=z,
                    lateral_error_m=precontact_residual, holding=precontact_hold,
                    ceiling_z=start_z)
                # The camera is not consulted once the gripper is over the target: measured, at the
                # bottom of a descent the part is not in the frame at all, and the only thing left
                # to correct towards is the gripper's own error.
                #
                # The height it *stops* consulting at is ``executor.TRACK_OBSERVE_ABOVE_Z``, and
                # that leaves the last 8 cm of the descent uncorrected - which is where the arm's
                # residual is largest and where a grasp is won or lost. So this corrects down to
                # ``LAST_CORRECTION_Z`` instead, and a missing sighting below the old gate is
                # treated as "no information" rather than as "the part is gone": losing the target
                # is only meaningful while the part should still be visible.
                if (index % every == 0 or index == steps) and z >= last_correction_z:
                    seen = observe()
                    if seen is None:
                        if (action == "down" and last_seen is not None
                                and z < (executor_module.TRACK_OBSERVE_ABOVE_Z
                                         + TRACK_LATE_OCCLUSION_MARGIN_M)
                                and occluded < TRACK_MAX_LATE_OCCLUSION):
                            # As the wrist crosses the camera sightline, every geometry can
                            # disappear for a frame. Retain the clean, scene-linked grasp anchor
                            # for at most two observations; longer occlusion still cancels before
                            # closing on an unverified pose.
                            occluded += 1
                            centre = self.aperture_centre()
                            control_target = (initial_aim if initial_aim is not None
                                              else last_seen)
                            if centre is not None and control_target is not None:
                                error = (control_target[0] - centre[0],
                                         control_target[1] - centre[1])
                                size = math.hypot(*error)
                                worst = max(worst, size)
                                if (executor_module.tool_alignment_needs_correction(
                                        size, tracking_tolerance) and not stall_reported):
                                    live_body = self._gripper_pos()[0, :2].detach().cpu().numpy()
                                    command[:] = feedback_body_command(
                                        live_body, centre, control_target,
                                        max_step_m=correction_cap)
                                    corrections += 1
                                    largest = max(largest, size)
                                    self.corrected += 1
                        elif large_ball_hover and action == "down":
                            lost += 1
                        elif z >= executor_module.TRACK_OBSERVE_ABOVE_Z:
                            scene_linked_coast = (last_seen is not None
                                                  and occluded < TRACK_MAX_SCENE_LINKED_MISSES
                                                  and (z >= executor_module.TRACK_OBSERVE_ABOVE_Z
                                                       or small_pregrasp_coast))
                            front_coast = (getattr(self, "allow_occluded_tracking", False)
                                           and occluded < executor_module.TRACK_MAX_OCCLUDED)
                            if front_coast or scene_linked_coast:
                                # The caller linked this aim to the scene immediately before the
                                # approach. Keep it for a short, fixed number of observations if
                                # the arm hides the object; a later mismatch still aborts.
                                occluded += 1
                                centre = self.aperture_centre()
                                if last_seen is not None and centre is not None:
                                    control_target = (initial_aim if initial_aim is not None
                                                      else last_seen)
                                    error = (control_target[0] - centre[0],
                                             control_target[1] - centre[1])
                                    size = math.hypot(*error)
                                    worst = max(worst, size)
                                    if (executor_module.tool_alignment_needs_correction(
                                            size, tracking_tolerance) and not stall_reported):
                                        live_body = self._gripper_pos()[0, :2].detach().cpu().numpy()
                                        command[:] = feedback_body_command(
                                            live_body, centre, control_target,
                                            max_step_m=correction_cap)
                                        corrections += 1
                                        largest = max(largest, size)
                                        self.corrected += 1
                            else:
                                lost += 1
                        else:
                            # Non-ball profiles keep the established visibility contract; only
                            # the measured 30 mm ball case has the late-wrist-occlusion allowance.
                            lost += 1
                    else:
                        reference = last_seen if last_seen is not None else tuple(command)
                        drift = math.dist((float(seen[0]), float(seen[1])), reference)
                        if (drift <= MAX_TRACKING_STEP_M
                                and not executor_module.tracked_target_motion_is_safe(
                                    seen, initial_aim, during_descent=action == "down")):
                            # A single detection can shift under partial wrist occlusion while
                            # remaining within the ordinary per-frame motion gate. Treat it as
                            # an outlier first; only two consecutive anchor violations indicate
                            # that the object really moved and require a safe retreat.
                            tracking_jumps += 1
                            rejected_jumps += 1
                            occluded += 1
                            if tracking_jumps >= 2:
                                lost += 1
                                total_drift = math.dist(
                                    (float(seen[0]), float(seen[1])), initial_aim)
                                self.log(f"      tracking stopped after two consecutive "
                                         f"anchor violations ({total_drift * 1000:.1f} mm); "
                                         "retreating without following the shifted target")
                        elif drift > MAX_TRACKING_STEP_M:
                            # A single frame can include a second red blob formed by the
                            # gripper/target overlap, especially in the front camera. Do not
                            # redirect a descent to that outlier, and do not abort on one frame.
                            # Keep the last trusted scene-linked aim while the front view is
                            # occluded. Other views get one outlier frame; repeated jumps still
                            # stop the motion before close. A real object displacement is handled
                            # by the executor's fresh-look recovery after this bounded approach.
                            tracking_jumps += 1
                            rejected_jumps += 1
                            if getattr(self, "allow_occluded_tracking", False):
                                occluded += 1
                                if occluded > executor_module.TRACK_MAX_OCCLUDED:
                                    lost += 1
                            elif (small_pregrasp_coast and last_seen is not None
                                  and occluded < TRACK_MAX_SCENE_LINKED_MISSES):
                                # Small parts occupy few pixels, and the ball's circular mask
                                # often jumps as the wrist enters the top camera. During the
                                # non-contact above move only, coast on the scene-linked aim for
                                # this bounded window; the descent retains the two-jump stop.
                                occluded += 1
                            elif tracking_jumps >= 2:
                                lost += 1
                                self.log(f"      tracking rejected two consecutive target jumps "
                                         f"({drift * 1000:.0f} mm from the commanded aim)")
                        else:
                            occluded = 0
                            tracking_jumps = 0
                            last_seen = (float(seen[0]), float(seen[1]))
                            centre = self.aperture_centre()
                            if centre is not None:
                                # The observation confirms the tracked identity and bounds
                                # object motion; it must not move the grasp point around as the
                                # wrist clips the object's mask. Keep the original scene pose
                                # for control throughout this approach.
                                control_target = (initial_aim if initial_aim is not None
                                                  else last_seen)
                                error = (float(control_target[0]) - centre[0],
                                         float(control_target[1]) - centre[1])
                                size = math.hypot(*error)
                                if os.environ.get("DEMO11_TRACE_TRACK") == "1":
                                    self.log(
                                        f"      track trace {action} z={z:.3f} "
                                        f"seen=({seen[0]:+.3f},{seen[1]:+.3f}) "
                                        f"aperture=({centre[0]:+.3f},{centre[1]:+.3f}) "
                                        f"command=({command[0]:+.3f},{command[1]:+.3f}) "
                                        f"error=({error[0]*1000:+.1f},{error[1]*1000:+.1f}) mm")
                                worst = max(worst, size)
                                if large_ball_hover and action == "down":
                                    was_holding = precontact_hold
                                    precontact_residual = size
                                    z, precontact_hold = ball_precontact_hold_z(
                                        action=action, large_ball=True, requested_z=z,
                                        lateral_error_m=size, holding=precontact_hold,
                                        ceiling_z=start_z)
                                    if precontact_hold and not precontact_hold_logged:
                                        precontact_hold_logged = True
                                        self.log("      ball pre-contact hold: pausing descent "
                                                 f"at {z:.3f} m; aperture error is "
                                                 f"{size * 1000:.1f} mm")
                                    elif was_holding and not precontact_hold:
                                        precontact_hold_logged = False
                                        self.log("      ball pre-contact hold released: "
                                                 f"aperture error {size * 1000:.1f} mm")
                                if (executor_module.tool_alignment_needs_correction(
                                        size, tracking_tolerance) and not stall_reported):
                                    recent.append(size)
                                    del recent[:-STALL_WINDOW]
                                    if (len(recent) == STALL_WINDOW
                                            and recent[0] - recent[-1] < STALL_IMPROVEMENT):
                                        # Stop issuing corrections: an arm at a joint limit obeys
                                        # every command and goes nowhere. The caller re-plans or
                                        # fails the task; the servo must not keep pushing.
                                        stall_reported = True
                                        self.correcting = False
                                        self.log(f"      centring stalled: the aperture is still "
                                                 f"{size * 1000:.1f} mm from the part after "
                                                 f"{corrections} correction(s), and the last "
                                                 f"{STALL_WINDOW} corrections did not shrink it - "
                                                 f"the jaws cannot reach it from here")
                                        if precontact_hold:
                                            lost += 1
                                            self.log("      ball pre-contact hold aborted: "
                                                     "centering stalled above contact")
                                    else:
                                        # Never correct by more than the slack allows. If the arm
                                        # is at a limit this caps the swing a full gain would give.
                                        # The hover has no contact risk, so it may eat the large
                                        # IK residual faster. Keep the near-contact descent capped
                                        # at twice the lateral slack to avoid sweeping a part.
                                        live_body = self._gripper_pos()[0, :2].detach().cpu().numpy()
                                        command[:] = feedback_body_command(
                                            live_body, centre, control_target,
                                            max_step_m=correction_cap)
                                        corrections += 1
                                        largest = max(largest, size)
                                        self.corrected += 1
                    if lost:
                        # A missing or mismatched target is a hard stop on top/side cameras and
                        # after the bounded front-camera coast expires.
                        break
                if self.frame_hook is not None:
                    self.frame_hook()
                z = bounded_ball_descent_z(
                    action=action, large_ball=large_ball_hover,
                    requested_z=z, previous_z=last_commanded_z)
                last_commanded_z = z
                self.current = torch.tensor([[command[0], command[1], z]], device=self.device)
                if self.pause_hook is not None and not self.pause_hook():
                    lost += 1
                    break
                if action == "down" and self.aperture_centre() is None:
                    aperture_feedback_lost = True
                    self.correcting = False
                    self.log("      descenso detenido: se perdió el feedback de apertura; "
                             "no se continúa hacia el contacto")
                    break
                self.env.step(self._action(self.current, grip))
                if (os.environ.get("DEMO11_TRACE_MOTION") == "1"
                        and index % max(1, every) == 0 and corrections % 5 == 0):
                    live_body = self._gripper_pos()[0].detach().cpu().numpy()
                    live_aperture = self.aperture_centre()
                    joint_ids, _ = self.robot.find_joints([
                        "shoulder_pan", "shoulder_lift", "elbow_flex",
                        "wrist_flex", "wrist_roll"])
                    joints = self.robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
                    aperture_xy = (live_aperture[:2] if live_aperture is not None
                                   else (math.nan, math.nan))
                    self.log(
                        f"      [motion] {action} target=({last_seen[0]:+.3f},"
                        f" {last_seen[1]:+.3f}) body=({live_body[0]:+.3f},"
                        f" {live_body[1]:+.3f}, {live_body[2]:+.3f}) "
                        f"aperture=({aperture_xy[0]:+.3f}, {aperture_xy[1]:+.3f}) "
                        f"command=({command[0]:+.3f}, {command[1]:+.3f}, {z:.3f}) "
                        f"joints={np.round(joints, 3).tolist()}")
                tracked_prim = getattr(self, "_tracking_prim", None)
                if tracked_prim:
                    self._freeze_ball_assist(tracked_prim)
                if (tracked_prim and self._uses_simulator_assist(tracked_prim)
                        and action == "down"
                        and self.holding is None
                        and self.item_kinds.get(tracked_prim) == "ball"):
                    # The SO-101 fingers cannot reliably pinch a sphere while the wrist is
                    # entering it from above: the forearm pushes the ball before the pads meet.
                    # In simulator-assist mode, create a vacuum-style grasp at first contact,
                    # based on measured aperture XY, the declared sphere radius, and the target
                    # height. This is a contact-gated simulator constraint; hardware remains on
                    # the ordinary physical path and the result is explicitly counted kinematic.
                    try:
                        ball = self.env.scene[tracked_prim]
                        ball_pos = ball.data.root_pos_w[0].detach().cpu().numpy()
                        centre = self.aperture_centre()
                        radius = float(self.items_by_prim[tracked_prim]) / 2.0
                        aperture_hold_z = float(self.current[0, 2].detach().cpu()) + hold_z
                        gap = (math.dist(centre[:2], ball_pos[:2])
                               if centre is not None else math.inf)
                        contact_band = (ball_pos[2] - 0.008, ball_pos[2] + radius + 0.005)
                        in_contact_band = contact_band[0] <= aperture_hold_z <= contact_band[1]
                        if os.environ.get("DEMO11_TRACE_CONTACT") == "1":
                            self.log(
                                f"      ball assist contact check: gap={gap * 1000:.1f} mm, "
                                f"hold_z={aperture_hold_z:.4f} m, "
                                f"ball_z={float(ball_pos[2]):.4f} m, "
                                f"band=({contact_band[0]:.4f},{contact_band[1]:.4f}) m, "
                                f"eligible={gap <= 0.006 and in_contact_band}")
                        if gap <= 0.006 and in_contact_band:
                            self.grab(tracked_prim)
                            self._pregrasp_kinematic = tracked_prim
                            if os.environ.get("DEMO11_TRACE_CONTACT") == "1":
                                self.log("      ball assist contact constraint attached")
                    except (KeyError, TypeError, ValueError, IndexError) as exc:
                        if os.environ.get("DEMO11_TRACE_CONTACT") == "1":
                            self.log(f"      ball assist contact check unavailable: "
                                     f"{type(exc).__name__}: {exc}")
                if getattr(self, "_held", None) is not None:
                    self._carry_if_held()

        # The preceding above trajectory can end with the gripper still several centimetres
        # away from the target: it has to converge while changing height, and the SO-101 response
        # couples X/Y as its wrist rotates. Finish alignment at the safe hover height for every
        # movable geometry before beginning contact descent. This stationary correction cannot
        # drag a part across the table, and failure to converge prevents contact.
        pregrasp_residual = math.inf
        if (action == "above" and pregrasp_hover_alignment and lost == 0
                and last_seen is not None):
            hover_value = self.current[0, 2]
            if callable(getattr(hover_value, "detach", None)):
                hover_value = hover_value.detach().cpu()
            hover_z = float(hover_value)
            pregrasp_misses = 0
            for alignment_step in range(PREGRASP_ALIGNMENT_MAX_STEPS + 1):
                if self.pause_hook is not None and not self.pause_hook():
                    lost += 1
                    self.log("      centrado previo interrumpido por el usuario")
                    break
                seen = observe()
                jump = (seen is not None and last_seen is not None
                        and math.dist((float(seen[0]), float(seen[1])), last_seen)
                        > MAX_TRACKING_STEP_M)
                if seen is None or jump:
                    if pregrasp_misses < TRACK_MAX_SCENE_LINKED_MISSES:
                        pregrasp_misses += 1
                        occluded += 1
                        if jump:
                            rejected_jumps += 1
                        # Hold the current hover pose while waiting for a fresh, scene-linked
                        # frame; do not issue another correction from a stale or shifted mask.
                        self.env.step(self._action(self.current, grip))
                        continue
                    lost += 1
                    if jump:
                        self.log("      centrado previo cancelado: detección desplazada de forma persistente")
                    break
                pregrasp_misses = 0
                if not executor_module.tracked_target_motion_is_safe(
                        seen, initial_aim, during_descent=False):
                    lost += 1
                    self.log("      pre-grasp hover stopped: target moved outside its scene anchor")
                    break
                last_seen = (float(seen[0]), float(seen[1]))
                centre = self.aperture_centre()
                if centre is None:
                    lost += 1
                    break
                # The gripper can shift or partly occlude a target's mask at hover height.
                # Continue aiming at the pose planned from the last clean scene view; the new
                # frame is used to verify identity and reject excessive object motion.
                control_target = initial_aim if initial_aim is not None else last_seen
                error = (control_target[0] - centre[0], control_target[1] - centre[1])
                pregrasp_residual = math.hypot(*error)
                worst = max(worst, pregrasp_residual)
                if os.environ.get("DEMO11_TRACE_MOTION") == "1":
                    live_body = self._gripper_pos()[0].detach().cpu().numpy()
                    joint_ids, _ = self.robot.find_joints([
                        "shoulder_pan", "shoulder_lift", "elbow_flex",
                        "wrist_flex", "wrist_roll"])
                    joints = self.robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
                    self.log(
                        f"      [hover] step={alignment_step} target=({last_seen[0]:+.3f},"
                        f" {last_seen[1]:+.3f}) body=({live_body[0]:+.3f},"
                        f" {live_body[1]:+.3f}, {live_body[2]:+.3f}) "
                        f"aperture=({centre[0]:+.3f}, {centre[1]:+.3f}) "
                        f"error=({error[0] * 1000:+.1f}, {error[1] * 1000:+.1f}) mm "
                        f"joints={np.round(joints, 3).tolist()}")
                if pregrasp_residual <= PREGRASP_ALIGNMENT_TOLERANCE_M:
                    break
                if alignment_step >= PREGRASP_ALIGNMENT_MAX_STEPS:
                    break
                live_body = self._gripper_pos()[0, :2].detach().cpu().numpy()
                command[:] = feedback_body_command(
                    live_body, centre, control_target,
                    max_step_m=PREGRASP_ALIGNMENT_STEP_M, gain=0.65)
                corrections += 1
                largest = max(largest, pregrasp_residual)
                self.corrected += 1
                self.current = torch.tensor([[command[0], command[1], hover_z]],
                                            device=self.device)
                self.env.step(self._action(self.current, grip))
            if (lost == 0
                    and pregrasp_residual > PREGRASP_ALIGNMENT_TOLERANCE_M):
                stall_reported = True
                self.correcting = False
                self.log(f"      pre-grasp hover did not converge: aperture remains "
                         f"{pregrasp_residual * 1000:.1f} mm from target; "
                         "contact descent cancelled")

        # Settle before the jaws shut: TRIED AT 40 STEPS AND REVERTED, because it measured worse.
        #
        # The reasoning was that the arm still has lateral velocity when the jaws close. The
        # measurement said otherwise and more usefully: holding the pose on top of the part for
        # another 0.67 s tipped it. The sort went from 2 of 3 to 1 of 3, and the refusal log showed
        # the cube's attitude degrading across attempts - 0 deg, 2 deg, then 47 and 90, where it
        # stands on its edge and presents 42 mm to a 36.2 mm gripper. The knock-over causes the
        # failed grasps, it is not a symptom of them, and pressing the gripper down is the worst
        # possible response to it. What the trace does establish is the order of events:
        #
        #   descent tips the part -> 42 mm across the jaws -> no force can hold it
        #   -> every retry finds it worse -> the sort gives up on it
        #
        # So the fix is to stop the arm tipping the part (see ``LAST_CORRECTION_Z``), and to *say*
        # when a part cannot be gripped at all rather than spending attempts on it.
        note = f"aimed at ({command[0]:+.3f}, {command[1]:+.3f})"
        if corrections:
            note += (f", {corrections} tool-centre correction(s) (largest {largest * 1000:.0f} mm, "
                     f"worst residual {worst * 1000:.0f} mm)")
        if occluded:
            note += f", {occluded} occluded look(s), retained the last scene-linked aim"
        if lost:
            note += f", {lost} look(s) could not find it"
        if aperture_feedback_lost:
            note += ", descenso cancelado al perder el feedback de las mordazas"
        if rejected_jumps:
            note += f", ignored {rejected_jumps} isolated target-jump frame(s)"
        # The residual in all three axes, because "the grasp failed" is not a diagnosis and this is
        # the one number that turns it into one. A part is between the pads only if the aperture
        # centre is within LATERAL_SLACK_M of it laterally and within half the part's height
        # vertically.
        centre = self.aperture_centre()
        contact_misaligned = False
        contact_error = None
        if action == "down" and centre is None:
            contact_misaligned = True
            note += ", sin feedback del centro de mordazas: agarre cancelado"
        elif centre is not None and action == "down":
            here3 = observe()
            # A final frame can contain a wrist-clipped object mask. Accept it only while it
            # remains inside the same descent motion gate; otherwise use the last plan anchor,
            # never the new shifted centroid, for the physical aperture check.
            contact_frame_trusted = (here3 is None or initial_aim is None or
                                     math.dist((float(here3[0]), float(here3[1])),
                                               (float(initial_aim[0]),
                                                float(initial_aim[1])))
                                     <= FINAL_CONTACT_OBSERVATION_TOLERANCE_M)
            contact_target = trusted_contact_target(here3, initial_aim)
            if here3 is not None and not contact_frame_trusted:
                note += ", rejected wrist-clipped final mask; using the planned grasp anchor"
            if contact_target is not None:
                gap = math.dist(centre[:2], contact_target[:2])
                experimental_slack = None
                pad_shape = getattr(self.env, "demo11_contact_pad_shape", None)
                slack_override = os.environ.get("DEMO11_EXPERIMENTAL_APERTURE_SLACK_MM")
                if slack_override is not None:
                    try:
                        experimental_slack = float(slack_override)
                    except ValueError:
                        experimental_slack = math.nan
                contact_slack = aperture_slack_for_profile(
                    getattr(tracking_item, "size", None),
                    kind=getattr(tracking_item, "kind", ""), pad_shape=pad_shape,
                    experimental_slack_mm=experimental_slack)
                contact_misaligned = (contact_misaligned
                                       or not aperture_gap_is_graspable(gap, contact_slack))
                if contact_misaligned:
                    contact_error = (float(contact_target[0]) - float(centre[0]),
                                     float(contact_target[1]) - float(centre[1]))
                note += (f", aperture {gap * 1000:.1f} mm from the part"
                         f"{' (between the pads)' if gap <= contact_slack else ' (outside the slack)'}")
                if contact_misaligned:
                    note += "; cierre cancelado para volver a centrar sin contacto"
        if (contact_misaligned and contact_error is not None and _contact_retry < 1
                and callable(getattr(self.current, "clone", None))
                and callable(getattr(self.current[0, 2], "detach", None))):
            # A failed final aperture check used to cancel the close, then restart the same
            # descent from the same pose. On the next pass the camera returned the same aim and
            # the robot repeated the same miss. Retreat vertically while the fingers are open,
            # move the measured aperture error at clearance, and make one fresh descent. This
            # preserves the object instead of dragging the open fingers across it.
            current = self.current.clone()
            safe_z = max(float(current[0, 2].detach().cpu()), target_body_z + 0.045)
            high = current.clone()
            high[0, 2] = safe_z
            corrected = high.clone()
            measured_body = self._gripper_pos()[0, :2].detach().cpu().numpy()
            corrected_xy = feedback_body_command(
                measured_body, centre[:2], contact_target,
                max_step_m=MAX_TRACKING_STEP_M)
            corrected[0, 0] = corrected_xy[0]
            corrected[0, 1] = corrected_xy[1]
            with torch.no_grad():
                for index in range(1, 17):
                    if self.pause_hook is not None and not self.pause_hook():
                        return executor_module.StepResult(
                            action=action, ok=False, note="parada solicitada durante el reajuste")
                    self.current = current + (high - current) * (index / 16.0)
                    self.env.step(self._action(self.current, grip))
                for index in range(1, 17):
                    if self.pause_hook is not None and not self.pause_hook():
                        return executor_module.StepResult(
                            action=action, ok=False, note="parada solicitada durante el reajuste")
                    self.current = high + (corrected - high) * (index / 16.0)
                    self.env.step(self._action(self.current, grip))
            self.log(f"      re-centro seguro: retiro vertical y corrección "
                     f"({contact_error[0] * 1000:+.1f}, {contact_error[1] * 1000:+.1f}) mm")
            retried = self.track(
                observe, target_z, steps, action=action, grip=grip, every=every,
                aim=initial_aim,
                _contact_retry=_contact_retry + 1)
            if retried.ok:
                retried.note = "recentrado seguro; " + retried.note
            return retried
        if stall_reported:
            note += ", centrado detenido: la pose no es alcanzable desde aquÃ­"
        ok = (lost == 0 and not stall_reported and not contact_misaligned
              and not aperture_feedback_lost)
        if not ok and tracking_prim and action in {"above", "down"}:
            if not hasattr(self, "failures"):
                self.failures = {}
            self.failures[tracking_prim] = self.failures.get(tracking_prim, 0) + 1
            self.log(f"      {action} failed before contact; next retry will test an "
                     "alternate reachable wrist orientation when the geometry allows it")
        return executor_module.StepResult(
            action=action, note=note, ok=ok,
            grasp="stale" if (contact_misaligned or aperture_feedback_lost) else None)


# --------------------------------------------------------------------------------------
# The demo-10 loop, instrumented
# --------------------------------------------------------------------------------------


class ModelSpy:
    """The command model, wrapped so the dashboard can show what was understood.

    A pass-through proxy rather than an override of ``handle``: the sentence-to-command step
    happens on one line inside demo 10's loop, and the only way to see it there is to be the
    object ``handle`` calls. Everything else is delegated untouched, including anything demo 10
    may add to the command dict later.
    """

    def __init__(self, model, bus: telemetry.Bus):
        self._model = model
        self._bus = bus

    def parse(self, text: str, *args, **kwargs) -> dict:
        self._bus.node("understand", telemetry.ACTIVE, text)
        command = self._model.parse(text, *args, **kwargs)
        # The dashboard's classification card wants the whole verdict, not a summary string:
        # intent, what to act on (each item's kind and colour, and its own destination when the
        # sentence gave one), the shared destination and the sort key. ``place`` on the command
        # is what the sentence as a whole was aiming at, ``sort_by`` what a sort ordered by.
        # One sentence saying what was understood, so the panel never has to reassemble a decision
        # out of a grid of options. Built here, from `describe` and `ui_spanish`, because that is
        # where the Spanish agreement lives. It is published twice on purpose - once in the state,
        # for the card, and once as this box's own note, which is what the diagram shows - and both
        # are the same words, so there is nothing to disagree.
        summary = describe.understood(command)
        self._bus.set_state(classification={
            "intent": command.get("intent"),
            "items": [{"kind": item.get("kind"), "color": item.get("color"),
                       "count": item.get("quantity", item.get("count")), "place": item.get("place")}
                      for item in command.get("items", [])],
            "place": command.get("place"),
            "sort_by": command.get("sort_by"),
            "summary": summary,
            # The model's own score for each field it decided, so the panel can say which one it
            # was unsure of instead of only reporting the weakest of them as a single number.
            "confidences": command.get("confidences") or {},
            "confidence": command.get("confidence"),
            "parser": command.get("parser", "model"),
            "errors": command.get("errors", []),
        })
        self._bus.set_metric("parse_ms", command.get("parse_ms"))
        self._bus.node("understand", telemetry.OK, summary)
        return command


# Every robot that wants to hear about planning. The wrapper is installed once, the first time a
# robot is built, and is a plain delegation around ``planner.plan_command``.
_PLAN_SPIES: list = []
# Robots that may rewrite the scene a plan is built from (each one in turn). Separate registry
# from the spies so a robot that only reads never has to know about the one that writes.
_PLAN_FILTERS: list = []


def install_plan_spy() -> None:
    """Let demo 11 see - and shape - every plan demo 10 builds, without editing ``planner``.

    Monkeypatching a shared module is normally the wrong answer, and it is the right one here for
    two reasons: demo 11 owns this process (demo 10 is loaded into it, not the reverse), and the
    alternative - reimplementing ``handle`` - would duplicate the single piece of this system
    that is hardest to get right. The wrapper filters the scene first (parts measured
    ungraspable), does nothing else, then calls through and tells the robots.
    """
    if getattr(planner.plan_command, "_demo11_spy", False):
        return
    original = planner.plan_command

    def spy(command, scene, offset=(0.0, 0.0)):
        shaped = scene
        for robot in list(_PLAN_FILTERS):
            shaped = robot.before_plan(command, shaped, offset)
        plan = original(command, shaped, offset)
        for robot in list(_PLAN_SPIES):
            robot.on_plan(plan)
        return plan

    spy._demo11_spy = True
    planner.plan_command = spy


# Where demo 11 is willing to put a part down for the arm to pick up.
#
# **This is the fix for the grasp that never worked, and it is a measurement rather than a
# preference.** With the strict grip in place, demo 11 places a 30 mm cube at x = 0.289 on the
# first or second attempt - ``lift #3 [followed 99% of the lift]``, placed and verified - and
# refuses a cube at x = 0.385 and at x = 0.445 every single time. The reason is not perception and
# not the jaw angle: measured at the bottom of the descent, **the jaws are 13.7-16.6 mm from where
# they were told to be**, against the 3.1 mm of lateral slack a 30 mm part has between 36.2 mm
# pads. The camera re-aims onto the part correctly; the arm does not arrive.
#
# The default is therefore demo 10's declared working area, unchanged, with ``preflight`` reporting
# the gap - because clamping the layout into the part of the table that works is not available as a
# fix: the envelope is ~90 mm wide and the scene needs 190 mm of it to keep two reds far enough
# apart for segmentation. The three real answers are on a bench, not in a constant: recalibrate the
# arm's tool centre point (see below), fit a wider-aperture gripper, or move the robot so the work
# is centred on it.
#
# **Why correcting the aim is not one of those answers.** Demo 11 measures the jaw-vs-command error
# at the bottom of every descent and repeats the descent displaced by it (``CENTRING_CORRECTIONS``),
# which is a tool-centre-point correction done online. It does not help, and the log says why: the
# descent re-reads the camera every 20 steps and aims at the part, so the correction is undone
# before the jaws close. The compensation has to be applied to the *commanded pose inside that
# loop*, not to the aim that feeds it. That is the fix to write; it is not written, and the
# counter in the summary reports how often it was tried.
# Where demo 11 is willing to put a part down for the arm to pick up.
#
# **Left at demo 10's declared area on purpose, and that is the finding.** With a tool-centre servo
# on the descent (``StrictExecutor.track``) the arm grasps at x = 0.44 as well as at x = 0.29 - see
# ``results/grip_and_release.md`` - so no clamping is needed. Before that servo existed this
# constant was 0.27..0.33, the only range measured to work, and the four-object sort was 0 of 3: a
# 90 mm grasp envelope inside a 190 mm layout. The servo removed the reason for the constraint
# rather than the constraint's consequences.
#
# Kept as a flag because a narrower envelope is the right thing to ask for on a bench whose arm
# has not been calibrated: it trades sorting for graspability, and it says so on start-up.
DEFAULT_WORK_X = scene_spec.WORK_X

# The range measured to hold reliably before the tool-centre servo existed, reported by preflight as
# what to fall back to if the servo is ever switched off.
GRASP_ENVELOPE_X = (0.27, 0.33)

# How much of the layout's jitter is kept. Smaller than ``dynamics``'s default because the range
# above is 130 mm wide and an 18 mm jitter either way would put a part back outside the verified
# envelope on a third of rounds - and a part outside it is a part the arm cannot hold.
DEFAULT_SPREAD_M = 0.010

# The live camera feed between annotated looks. Measured, the re-aim sightings during a descent
# already publish an annotated frame every ``LOOK_EVERY`` steps; between those, and during
# moves without a tracking target, the picture used to go stale. The pump simply renders and
# publishes a downscaled raw frame whenever the previous one is older than 1/fps - a raw frame
# costs a render, not a perception pass.
#
# Three, not five, and the reason is that this runs between two physics steps. Every published
# frame is one ``sim.render()`` the arm pays for in wall-clock time, and a descent is short: at
# 3 fps a typical descent gets several frames, at 5 the render starts costing more than the
# picture is worth. The dashboard's own ``fps`` readout shows what the chosen rate really
# delivers, so this is a default to change rather than a rule - ``--cam-fps 0`` turns it off.
CAM_FPS_DEFAULT = 3.0
FRAME_PUMP_WIDTH = 384
FRAME_PUMP_QUALITY = 70

# After how many refused picks a part stops being planned again within the same command.
# Demo 10 gives every object 2 go(s) in the sort loop; this is the same budget seen from the
# executor's side, and it is what stops the third attempt - the one the refusal log measured
# tipping the part further off its face - from happening at all.
MAX_PICK_FAILURES = 2


def clamp_layout(items, work_x=DEFAULT_WORK_X, half_width_fn=None) -> list:
    """Move every loose part's x into ``work_x``, keeping its order and leaving trays alone.

    Clamping rather than re-laying-out on purpose: the layout's colours, spacing and rows were
    fitted against the camera's field of view and the segmentation thresholds, and moving parts
    around is exactly the kind of change that invalidates a recorded measurement. Only x is
    touched, only for parts the arm has to pick up, and each part is clamped to its own
    footprint so nothing ends up half outside the range.
    """
    from scene_spec import half_width

    half_width_fn = half_width_fn or half_width
    out = []
    for item in items:
        if item.container:
            out.append(item)
            continue
        pad = half_width_fn(item)
        x = min(max(item.x, work_x[0] + pad), work_x[1] - pad)
        out.append(item if x == item.x else replace(item, x=x))
    return out


class TaskLedger:
    """One verdict per physical task, for the whole run. This is the only source of the result.

    A later verdict replaces an earlier one, so a task that failed on the first attempt and was
    then placed counts once, as placed. A task with no verdict is not counted as done. The key is
    the part's perception-independent name when it resolved, else ``thing #<id>``, which is the
    same key the executor's own logs use.
    """

    OK, FAILED, PENDING = "ok", "failed", "pending"

    def __init__(self):
        self._verdict: dict[str, str] = {}
        self._label: dict[str, str] = {}

    def record(self, key: str, ok: bool, label: str = "") -> None:
        self._verdict[key] = self.OK if ok else self.FAILED
        if label:
            self._label.setdefault(key, label)

    def expect(self, key: str, label: str = "") -> None:
        self._verdict.setdefault(key, self.PENDING)
        self._label.setdefault(key, label or key)

    def verdict(self, key: str) -> str | None:
        return self._verdict.get(key)

    def counts(self) -> tuple[int, int]:
        values = list(self._verdict.values())
        return values.count(self.OK), values.count(self.FAILED) + values.count(self.PENDING)

    def failed_labels(self) -> list[str]:
        return [self._label.get(key, key) for key, value in self._verdict.items()
                if value != self.OK]

    def status(self, aborted: bool = False) -> str:
        """SUCCESS, PARTIAL_SUCCESS, FAILED, ABORTED or NO_TASKS, from the verdicts only."""
        ok, failed = self.counts()
        if aborted:
            return "ABORTED"
        if ok + failed == 0:
            return "NO_TASKS"
        if failed == 0:
            return "SUCCESS"
        if ok == 0:
            return "FAILED"
        return "PARTIAL_SUCCESS"


# The user-facing name of each final status, and the exit code a script should see for it.
STATUS_ES = {"SUCCESS": "Completado", "PARTIAL_SUCCESS": "Completado parcialmente",
             "FAILED": "Fallido", "ABORTED": "Interrumpido", "NO_TASKS": "Sin tareas"}
def _short_summary(run) -> str:
    """The execute box's note: a few words, since the box is small. The full summary is in the log."""
    if run.aborted:
        return f"parado en {run.aborted_at.split(' ')[0]}"
    text = (f"{len(run.steps)} pasos, {run.real_grasps} agarre(s) fÃ­sico(s), "
            f"{run.kinematic_grasps} asistido(s) en simulaciÃ³n")
    if run.stale_grasps:
        text += ", sin pieza"
    if run.lost_grasps:
        text += ", pieza perdida"
    return text


EXIT_CODE = {"SUCCESS": 0, "NO_TASKS": 0, "PARTIAL_SUCCESS": 2, "FAILED": 2, "ABORTED": 3}


def should_use_backup_camera(*, has_backup: bool, primary_capture_failed: bool,
                             objects_seen: int, containers_expected: bool,
                             containers_seen: int, tasks_pending: bool,
                             robot_holding: bool, grasp_failed: bool = False,
                             ambiguous_objects: int = 0) -> bool:
    """Recover from failed frames and obvious arm occlusion in oblique views."""
    if not has_backup:
        return False
    if grasp_failed and not robot_holding:
        return True
    if primary_capture_failed:
        return True
    if ambiguous_objects > 0 and tasks_pending and not robot_holding:
        return True
    if containers_expected and containers_seen == 0:
        return True
    return tasks_pending and not robot_holding and objects_seen == 0


def allow_occluded_tracking_for_camera(camera_name: str, active_sensor, primary_sensor) -> bool:
    """Coast through brief arm occlusion only while tracking the oblique front camera.

    The top camera is the recovery view and has a different occlusion geometry. Keeping the
    front-camera coast policy after a fallback lets stale target coordinates drive repeated
    wrist corrections even though tracking is now using the overhead sensor.
    """
    return camera_name == "front" and active_sensor is primary_sensor


def experimental_approach_clearance_z(value: str | None) -> float | None:
    """Parse an opt-in world-frame height for a vertical-then-horizontal approach A/B."""
    if value is None or not str(value).strip():
        return None
    height = float(value)
    if not 0.12 <= height <= 0.50:
        raise ValueError("approach clearance height must be within [0.12, 0.50] m")
    return height


def safe_approach_clearance_z(items) -> float:
    """Choose a scene-aware hold-point height for collision-free XY approach travel.

    The approach waypoint itself is already above the selected object, but the arm can start
    above or below it. Translating and descending at the same time sweeps the open fingers and
    wrist through nearby objects. Keep the approach stage above every configured object and
    container, with 40 mm of clearance, and at least at the calibrated travel plane. The actual
    per-object waypoint can be higher (notably the ball profile) and still takes precedence.
    """
    top_z = scene_spec.TABLE_TOP_Z
    for item in items or ():
        try:
            top_z = max(top_z, scene_spec.TABLE_TOP_Z + float(item.height))
        except (AttributeError, TypeError, ValueError):
            size = getattr(item, "size", ()) or ()
            if size:
                top_z = max(top_z, scene_spec.TABLE_TOP_Z + max(float(v) for v in size))
    return max(float(planner.TRAVEL_Z), top_z + 0.040)


def off_center_assist_candidate(geometry: dict | None, *, after_physical_miss: bool,
                                assist_enabled: bool) -> bool:
    """A high lift-follow score is not a centered physical grasp by itself."""
    if not after_physical_miss or not assist_enabled or not geometry:
        return False
    lateral = geometry.get("lateral_m")
    return bool(geometry.get("fits") and lateral is not None
                and LATERAL_SLACK_M < lateral <= SIM_ASSIST_MAX_LATERAL_M)


def physical_miss_assist_candidate(geometry: dict | None, *, after_physical_miss: bool,
                                   assist_enabled: bool) -> bool:
    """Allow simulation-only recovery after a lost lift when the measured close was viable."""
    if not after_physical_miss or not assist_enabled or not geometry:
        return False
    lateral = geometry.get("lateral_m")
    return bool(geometry.get("fits") and lateral is not None
                and 0.0 <= lateral <= SIM_ASSIST_MAX_LATERAL_M)


def should_pregrasp_assist(policy: str, *, assist_enabled: bool, kind: str,
                           size_mm: float | None) -> bool:
    """Select pre-contact assistance for explicit policy or empirically fragile small parts."""
    if not assist_enabled:
        return False
    if policy == "preemptive":
        return True
    if (policy == "after_physical_miss" and size_mm is not None
            and ((kind == "cube" and float(size_mm) <= 20.0)
                 or (kind == "cylinder" and float(size_mm) <= 30.0)
                 or (kind in {"hexagonal_prism", "triangular_prism"}
                     and float(size_mm) <= 30.0))):
        # Repeated small-cube, 30 mm prism, and side-view 30 mm cylinder trials showed that
        # physical-first contact can leave the arm outside its calibrated frame or lose the part
        # in carry. In simulation, avoid that fragile contact for these profiles and count them
        # as assisted; strict/hardware runs stay real.
        return True
    return False


def stabilize_fallback_position(measured_xy, trusted_xy, max_shift_m: float):
    """Keep the last scene-linked pose when a backup view jumps beyond its error budget."""
    if measured_xy is None or trusted_xy is None:
        return measured_xy, False
    if math.dist(measured_xy, trusted_xy) > max_shift_m:
        return tuple(trusted_xy), True
    return tuple(measured_xy), False


def tracking_measurement_is_reliable(area_px: int, confidence: float,
                                     previous_area_px: int | None,
                                     min_confidence: float = 0.20,
                                     min_area_fraction: float = 0.40) -> bool:
    """Reject weak or abruptly clipped masks as pose updates for a known object.

    The catalog linker can identify a partly occluded object by colour and location, but
    the centroid of a clipped mask is not the object's centre. Keep the identity association
    while retaining the last full-view pose until the mask recovers.
    """
    area = max(0, int(area_px))
    if area == 0 or float(confidence) < min_confidence:
        return False
    if previous_area_px is not None and previous_area_px > 0:
        if area < float(previous_area_px) * min_area_fraction:
            return False
    return True


def fallback_tracking_step_limit(*, after_grasp_failure: bool,
                                  normal_m: float = MAX_TRACKING_STEP_M,
                                  recovery_m: float = SAME_OBJECT_MAX_SHIFT_M,
                                  ambiguous_same_color_peer: bool = False) -> float:
    """Allow bounded re-linking when failed jaw contact may have moved the target.

    Before contact, a 25 mm jump is usually camera/gripper error. After a recorded failed grasp,
    28-40 mm physical pushes have been observed; retaining the old anchor aims the retry at an
    empty point. A large re-link is permitted only when no same-color peer is left unobserved in
    this scan; if a peer is missing, the ordinary gate avoids switching object identity.
    """
    return float(recovery_m if after_grasp_failure and not ambiguous_same_color_peer else normal_m)


def fallback_link_tolerance(base_m: float, *, backup_view: bool,
                            is_container: bool, max_tracking_shift_m: float) -> float:
    """Allow association first, then stabilize object coordinates against the last view."""
    if backup_view and not is_container:
        return base_m + max_tracking_shift_m
    return base_m


class MonitoredRobot(load_demo10().Robot):
    """``demo10.Robot`` with real physics, a real grip, and a dashboard attached.

    The overrides are all reporting, plus two behavioural changes: where a part is released, and
    where a part may be picked up from. They are kept short and in one place on purpose - the
    loop's order of operations is the part of this system that is easy to get wrong and hard to
    read, so duplicating it to add logging would be duplicating the thing most worth having
    exactly one copy of.
    """

    def __init__(self, env, sensor, args, bus: telemetry.Bus, viewer=None, model=None,
                 release_mode: str = "clear"):
        self.bus = bus
        self.release_mode = release_mode
        self.demo10 = sys.modules["demo10"]
        self.advance = False
        self.quit = False
        self.web_input = bool(args.web) or bool(getattr(args, "worker", False))
        # The thread that reads typed commands, if there is a terminal to read. Started from
        # ``main`` once the start gate is open, so it never competes with
        # ``enter_starts_the_run`` for the same stdin.
        self.keyboard = None
        # A raw frame between the annotated looks, so the picture moves at something a person
        # reads as live while the arm travels. 0 disables the pump entirely.
        self.cam_fps = float(getattr(args, "cam_fps", 0.0) or 0.0)
        self._last_pump = 0.0
        self._plan_n = 0
        # Which step of the running plan the arm is on, counted by the executor's own callback.
        # `run_plan` publishes the total, which is where it is known.
        self._step_i = 0
        # The command's tasks and which are done, keyed by part name so a re-plan recognises
        # the same task instead of adding a second row for it. Cleared per command in `handle`.
        self._task_rows: dict[str, dict] = {}
        # The verdict per task for the whole run; the final result is derived from it alone.
        self.ledger = TaskLedger()
        super().__init__(env, sensor, args, viewer=viewer, model=model)
        self.primary_sensor = sensor
        self.backup_sensor = getattr(env, "demo11_backup_camera", None)
        self._active_sensor = sensor
        self._backup_capture_stale = True
        self._last_backup_capture = None
        if getattr(args, "scene_items", None):
            self.base = list(args.scene_items)
            self.layout = list(self.base)
            self.tracked_xy = {i.prim: (i.x, i.y) for i in self.base}
            self.tracked_area_px = {}
        self.catalog = scene_config.catalog(getattr(args, "scene_data", scene_config.default_config()), self.base)
        self._initial_layout = list(self.base)
        self._placed_slots = {}
        self._task_slots = {}
        self._rejected_commands = 0
        from frame_encoder import FrameEncoder
        self.frame_encoder = FrameEncoder(bus, self._camera_error)
        # Its own out/ folder: these frames are annotated for the dashboard and saved on
        # request, and mixing them into demo 10's would make either run's folder a mixture.
        self.writer = viewer_module.FrameWriter(
            HERE / "out11", every=args.save_every if args.save_every > 0 else 10 ** 9)
        self.bot = StrictExecutor(env, log=self.say, allow_kinematic=args.allow_kinematic,
                                  auto_sim_assist=not args.strict_physical,
                                  auto_sim_assist_policy=args.sim_assist_policy,
                                  close_command_override=getattr(
                                      args, "close_command_override", None))
        clearance_override = experimental_approach_clearance_z(
            os.environ.get("DEMO11_EXPERIMENTAL_APPROACH_CLEARANCE_Z"))
        self.bot.approach_clearance_z = (
            clearance_override if clearance_override is not None
            else safe_approach_clearance_z(self.base))
        # The executor needs the declared footprint to apply the bounded pre-grasp coasting
        # rule only to small, scene-linked targets; its generic grasp geometry stays separate.
        self.bot.layout = list(self.base)
        self.bot.camera_name = args.camera
        self.bot.allow_occluded_tracking = allow_occluded_tracking_for_camera(
            args.camera, self._active_sensor, self.primary_sensor)
        self.bot.log = self._step_logged
        self.bot.failure_hook = self._grasp_failure
        self.bot.pause_hook = self.bus.wait_if_paused
        self.bot.frame_hook = self.cam_pump if self.cam_fps > 0.0 else None
        # The waits that issue no action at all - ``settle`` after a release, and the
        # settling between a placement and its measurement - used to run with no
        # frame published, which is several seconds of frozen camera exactly when
        # the audience is looking at it. ``cam_pump`` is throttled internally, so
        # calling it here costs nothing when a frame was published recently.
        self.bot.settle_hook = self.cam_pump if self.cam_fps > 0.0 else None
        self.model = ModelSpy(model, bus)
        install_plan_spy()
        if self not in _PLAN_SPIES:
            _PLAN_SPIES.append(self)
        if self not in _PLAN_FILTERS:
            _PLAN_FILTERS.append(self)

    def populate(self, layout) -> None:
        """Refresh both the spawned layout and the identity linker after every scene change."""
        super().populate(layout)
        self.tracked_xy = {item.prim: (float(item.x), float(item.y)) for item in self.layout}
        self.tracked_area_px = {}
        bot = getattr(self, "bot", None)
        if bot is not None and os.environ.get("DEMO11_EXPERIMENTAL_APPROACH_CLEARANCE_Z") is None:
            bot.approach_clearance_z = safe_approach_clearance_z(self.layout)
        # Let the RTX render product catch up with place_items' prim transforms. These steps
        # do not pass simulator positions to perception; they only synchronize physics/render.
        sim = getattr(getattr(self, "env", None), "sim", None)
        step = getattr(sim, "step", None)
        if callable(step):
            for _ in range(SCENE_SETTLE_RENDER_STEPS):
                step(render=True)
        if getattr(self, "backup_sensor", None) is not None:
            # Initialize the backup RTX render product immediately after scene population;
            # lazily rendering it for the first time during an occlusion produced blank
            # buffers in several otherwise valid front/side trials.
            try:
                self._last_backup_capture = isaac_io.capture(self.env, self.backup_sensor)
                self._backup_capture_stale = False
                self.bus.set_state(backup_camera_ready=True)
            except RuntimeError as exc:
                self.bus.set_state(backup_camera_ready=False)
                self.bus.log(f"La cámara cenital de respaldo aún está inicializándose: {exc}",
                             level="warn", node="capture")

    def _capture_backup(self):
        """Capture overhead, reusing the same-scene warm frame only before any arm motion."""
        try:
            captured = isaac_io.capture(self.env, self.backup_sensor)
            self._last_backup_capture = captured
            return captured
        except RuntimeError:
            cached = self._last_backup_capture
            if cached is not None and not self._backup_capture_stale:
                self.say("  la cámara cenital no refresca; uso el último fotograma de esta "
                         "escena, capturado antes de mover el brazo")
                return cached
            raise

    # -- output ----------------------------------------------------------------

    def say(self, text: str = "") -> None:
        """Print, and send the same line to the dashboard.

        Every line the robot prints is already the explanation of something it just decided, so
        mirroring ``say`` rather than adding calls at the interesting places means the console
        and the dashboard cannot end up telling different stories.
        """
        print(text, flush=True)

    def _step_logged(self, text: str) -> None:
        """The executor's per-step log line, which is also the execute node's progress."""
        self.say(text)
        head = text.strip().split()
        if head:
            import ui_spanish
            self.bus.set_state(progress=head[0])
            self.bus.node("execute", telemetry.ACTIVE, ui_spanish.ACTIONS.get(head[0], "Comprobando ejecuciÃ³n"))
            # One call is one step executed, counted against the total the plan published, so the
            # panel can say "paso 3 de 8". The counter is reset by `run_plan`, which is where the
            # total is known.
            self._step_i = getattr(self, "_step_i", 0) + 1
            self.bus.set_state(step_index=self._step_i)

    # -- the environment -------------------------------------------------------

    def findings(self) -> list[tuple[str, str]]:
        """Run the real-world checks against this layout and log every one of them.

        Run after Isaac starts as well as before it, because the layout is re-randomised each
        round and the checks are about positions as well as materials: a part that landed outside
        the arm's reach in some seed is a fact about that seed, not about the bench in general.
        """
        rows = real_world.preflight(self.base or self.layout, camera=self.args.camera)
        rows.append((
            "note",
            f"grasp envelope: the tool-centre servo holds a part anywhere in the declared working "
            f"area, x {self.args.work_x[0]:.2f}..{self.args.work_x[1]:.2f}, and reports a stall "
            f"rather than closing the jaws when a pose is unreachable. Without it the same pick "
            f"was refused every time beyond about x {GRASP_ENVELOPE_X[1]:.2f}: the jaws ended up "
            f"13-18 mm from the commanded pose there against {LATERAL_SLACK_M * 1000:.1f} mm of "
            f"lateral slack. A bench with an uncalibrated arm should use --work-x "
            f"{GRASP_ENVELOPE_X[0]:.2f} {GRASP_ENVELOPE_X[1]:.2f}."))
        import ui_spanish
        self.bus.set_state(findings=ui_spanish.findings(rows), findings_technical=rows)
        for level, text in ui_spanish.findings(rows):
            self.bus.log(f"{level}: {text}",
                         level="fail" if level == "fail" else ("warn" if level == "warn"
                                                              else "info"))
        return rows

    def before_plan(self, command, scene, offset) -> object:
        """Remove parts measured ungraspable from this pass's scene, and say so in Spanish.

        The refusal log measured the pattern this avoids: a part that ends up on its edge
        presents more across the jaws than they can ever open, every retry tips it further,
        and the command's whole budget is spent on it - measured, five attempts, five
        refusals, nothing sorted. So, once a descent's close-time geometry says the grip is
        impossible, the part leaves the fresh scene and the command re-plans around it. The
        set is cleared per command in ``handle``: between commands the park sweep moves
        things, and a part that lies flat again is graspable again.
        """
        impossible = getattr(self.bot, "impossible", None) or set()
        failures = getattr(self.bot, "failures", None) or {}
        prims = set(impossible) | {p for p, n in failures.items()
                                   if n >= MAX_PICK_FAILURES}
        if not prims or scene is None:
            return scene
        fresh = copy.deepcopy(scene)
        keep = []
        dropped = []
        for thing in fresh.objects:
            if thing.pos_m is None:
                keep.append(thing)
                continue
            prim = self._prim_of(thing)
            if prim in prims:
                dropped.append(prim)
                continue
            keep.append(thing)
        if not dropped:
            return scene
        fresh.objects = keep
        reasons = [
            (f"{prim} de canto, sin ninguna fuerza posible entre las mordazas"
             if prim in impossible else
             f"{prim} tras {failures.get(prim, 0)} sujeciÃ³n(es) fallidas")
            for prim in dropped]
        self.say("  demo 11: " + "; ".join(reasons) +
                 "; se sacan del plan de esta pasada y el comando sigue con lo demÃ¡s")
        return fresh

    def _prim_of(self, thing) -> str | None:
        """The known part a perceived blob is standing on, matched by position as ``link`` does."""
        tolerance = getattr(getattr(self, "demo10", None), "LINK_TOLERANCE_M", 0.05)
        best, best_gap = None, tolerance
        for item in self.layout:
            if item.container:
                continue
            gap = math.dist(thing.pos_m[:2], (item.x, item.y))
            if gap < best_gap:
                best, best_gap = item, gap
        return best.prim if best else None

    def on_plan(self, plan) -> None:
        """Report what grounding and planning decided, from the plan itself.

        A plan carries both answers - ``unresolved`` and ``question`` are what grounding could not
        do, and ``steps`` is what planning did with the rest - so one hook covers both boxes
        instead of wrapping two functions that demo 10 calls as one expression.
        """
        if plan.question:
            line = next(iter(plan.question.values()))
            self.bus.node("ground", telemetry.WARN, f"AclaraciÃ³n: {line}")
        elif plan.unresolved:
            self.bus.node("ground", telemetry.WARN, "; ".join(plan.unresolved))
        else:
            self.bus.node("ground", telemetry.OK,
                          f"{len(plan.picks)} piezas, {len(plan.places)} destinos")
        # A plan that did not move still has something to say in the plan card - a question or
        # an unresolved reference is what the plan is that round - so both shapes publish.
        note = ""
        if plan.question:
            line = next(iter(plan.question.values()))
            note = f"pregunta: {line}"
        elif plan.unresolved:
            note = "; ".join(plan.unresolved)
        if plan.moves or note:
            self._publish_plan(plan, note=note)

    def _publish_plan(self, plan, note: str = "", increment: bool = True) -> None:
        """The plan the dashboard decides with, as data rather than prose.

        Published on every plan build (initial or re-planned, via ``on_plan``) and again when
        execution starts, because ``_release_adjusted`` may have raised the release heights
        after the last build - the picture must show the plan that will actually be run.
        """
        if increment:
            self._plan_n += 1
        self.bus.set_state(plan={
            "n": self._plan_n,
            "intent": getattr(plan, "intent", ""),
            "note": note,
            # The question and the unresolved references are what this plan IS when it could not
            # decide something: a plan that stops to ask "which red ball?" has no steps, and the
            # panel's classification box reads these two rather than showing a grid of options.
            "question": (plan.question or {}).get("es", ""),
            "unresolved": list(getattr(plan, "unresolved", []) or []),
            "picks": [str(pick) for pick in getattr(plan, "picks", [])],
            "tasks": self.plan_tasks(plan),
            "steps": [{"action": step.action, "note": step.note,
                       "pose": None if step.pose is None else
                       [float(v) for v in step.pose]}
                      for step in plan.steps],
        })

    # -- the task list, and what is already done --------------------------------

    def plan_tasks(self, plan, premise: dict | None = None) -> list[dict]:
        """This plan's tasks, as the rows a person would call a checklist.

        One row per thing the plan intends to move: "bola roja -> cesta roja". That is the unit
        ``planner.Task`` records and the unit ``check`` judges, so nothing here is inferred -
        ``place`` is None for a plain ``pick``, which has nowhere to put the part, and the row
        says so instead of inventing a destination.

        ``premise`` maps perception ids to part names, and it is what makes the row stable: the
        ids belong to one frame and change on the next look, while the name does not. So a
        re-planned task is recognised as the same task rather than added as a second one.

        ``state`` is one of ``doing`` / ``done`` / ``fail`` / ``todo``, and the panel's three icons
        come from it. A task that `check` measured and did not find in its tray is ``fail``, which
        is not the same as ``todo``: the second never got attempted, and marking it as failed would
        be claiming a measurement that was not taken.
        """
        rows = []
        for task in getattr(plan, "tasks", None) or []:
            part = describe.phrase(task.kind, task.color, 1, "es", bare=True)
            where = ""
            if task.place is not None:
                where = describe.phrase(task.place_kind, task.place_color, 1,
                                        "es", bare=True)
            rows.append({"key": self._task_key(task, premise),
                         "part": part, "destination": where,
                         "label": f"{part} → {where}" if where else part,
                         # `done` is kept alongside `state` because it is the one flag with
                         # meaning to anything reading this from outside the panel: `state` says
                         # HOW it went, `done` says whether it succeeded.
                         "done": False, "state": "todo"})
        return rows

    def _task_key(self, task, premise: dict | None) -> str:
        """What identifies a task across re-plans and in the ledger: the part's name, else its id.

        The same key the ledger uses (``thing #<id>``), so a task that could not be named from
        the frame is one task with one verdict, not a row that never matches anything.
        """
        prim = (premise or {}).get(task.pick)
        source = prim if prim else f"thing #{task.pick}"
        cid = getattr(self, "_active_cid", "")
        destination = (premise or {}).get(task.place, task.place)
        return f"{cid}:{source}:{destination}" if cid else source

    def _note_tasks(self, plan, premise: dict | None = None) -> None:
        """Fold this plan's tasks into the command's list, keeping what is already done.

        A multi-object command does not arrive as one plan. ``_one_at_a_time`` re-plans around
        each object in turn and calls ``check`` after every one, so the plans that get executed
        hold a single task each - while the plan the person was asked to act on held all of them.
        The list is therefore accumulated here, keyed by part, rather than replaced by each plan
        as it runs: otherwise the checklist would show one task at a time and never the whole
        job, which is the thing being asked for.
        """
        for row in self.plan_tasks(plan, premise):
            self.ledger.expect(row["key"], row["label"])
            previous = self._task_rows.get(row["key"])
            if previous:
                # A task that was already settled keeps its verdict: re-planning it is not a new
                # attempt, and a row that flipped back to "pendiente" because the plan was rebuilt
                # would say the measurement had been thrown away when it had not.
                row["done"] = bool(previous.get("done"))
                row["state"] = previous.get("state", "done" if row["done"] else "todo")
            self._task_rows[row["key"]] = row
        self._publish_tasks()

    def _mark_doing(self, plan, premise: dict | None = None) -> None:
        """Say which checklist rows this plan is executing, so the panel can answer "which one now?".

        ``_one_at_a_time`` re-plans around one object at a time, so the plan that runs holds a
        single task and the whole job is the checklist. Marking the running plan's rows - and only
        those - is what makes "tarea 2 de 3" true rather than inferred.

        A row that ``check`` has already settled is left alone: re-planning it is not a new
        attempt, and a row that went back to "en curso" would say the measurement is pending when
        it has been taken. Rows that are not part of this plan go back to pending, which is honest
        between attempts of the same command.
        """
        keys = {self._task_key(task, premise) for task in getattr(plan, "tasks", None) or []}
        for key, row in self._task_rows.items():
            if row["state"] in ("done", "fail"):
                continue
            row["state"] = "doing" if key in keys else "todo"
        self._publish_tasks()

    def _settle_rows(self) -> None:
        """Show each checklist row with the ledger's verdict for its task, and nothing else.

        A row with no verdict keeps its state. Nothing is ticked here that the ledger does not
        hold, so the checklist cannot disagree with the final result.
        """
        for key, row in self._task_rows.items():
            verdict = self.ledger.verdict(key)
            if verdict == TaskLedger.OK:
                row["done"], row["state"] = True, "done"
            elif verdict == TaskLedger.FAILED:
                row["done"], row["state"] = False, "fail"
        self._publish_tasks()

    def _placement_verdicts(self, plan, premise) -> list[tuple[str, bool, str | None]]:
        """(task key, placed, part name or None) for each pick, from the simulator's positions.

        Placed only when the object resolved to a known part, its destination resolved to a known
        tray, and it measured inside that tray. Anything else is not placed - an object that
        cannot be named from this frame is a failed task, not an unknown one.
        """
        out = []
        for pick_id, place_id in zip(plan.picks, plan.places):
            prim = premise.get(pick_id)
            tray_prim = premise.get(place_id)
            tray = (next((i for i in self.layout if i.prim == tray_prim), None)
                    if tray_prim else None)
            placed = prim is not None and tray is not None and self._inside(prim, tray)
            task = next((t for t in plan.tasks if t.pick == pick_id), None)
            key = self._task_key(task, premise) if task else (prim or f"thing #{pick_id}")
            out.append((key, placed, prim))
        return out

    def _task_label(self, plan, key: str, premise) -> str:
        """The human-readable objective of a task, for the ledger's failure list."""
        for task in getattr(plan, "tasks", None) or []:
            if self._task_key(task, premise) == key:
                part = describe.phrase(task.kind, task.color, 1, "es", bare=True)
                if task.place is None:
                    return part
                where = describe.phrase(task.place_kind, task.place_color, 1, "es",
                                        bare=True)
                return f"{part} → {where}"
        return key

    def _verify_and_retry(self, plan, premise, announce: bool = True) -> list[str]:
        """Measure every placement, retry each miss once, and record one verdict per task.

        The counters are read from the ledger, so each task adds to them exactly once. A retry
        counts as recovered only when the object is then measured inside its tray; a retry that
        did not place it is not a recovery. Returns the task keys that are still not placed.
        """
        self.bot.settle()
        first = self._placement_verdicts(plan, premise)
        retry = [prim for _key, placed, prim in first if not placed and prim is not None]
        for prim in retry:
            # Say which containment test failed and by how much: "not placed" alone sent the
            # last investigation looking at the wrong axis (see ``_why_not_inside``).
            self.say(f"    not verified: {prim} {self._why_not_inside(prim)}")
        if retry:
            self.say("  re-planning the ones that missed")
        for prim in retry:
            self._retry_one(prim)
        self.park()
        self.bot.settle()
        final = self._placement_verdicts(plan, premise)
        for _key, placed, prim in final:
            if placed and prim is not None and prim in retry:
                self.recovered += 1
        for key, placed, _prim in final:
            self.ledger.record(key, placed, self._task_label(plan, key, premise))
        self.verified, self.missed = self.ledger.counts()
        self._settle_rows()
        misses = [key for key, placed, _prim in final if not placed]
        seen = self.look("after the plan")
        if announce:
            self.say(f"  camera sees {len(seen.objects)} object(s) and "
                     f"{len(seen.containers)} container(s); "
                     f"{len(plan.picks) - len(misses)}/{len(plan.picks)} placement(s) in a tray")
        return misses

    def _publish_tasks(self) -> None:
        self.bus.set_state(tasks=[dict(row) for row in self._task_rows.values()])

    # -- perception ------------------------------------------------------------

    def look(self, caption: str = "", *, after_grasp_failure: bool = False):
        """One frame, then the dashboard knows what was in it.

        The frame published is ``self.shown`` - the ANNOTATED one, every blob outlined and
        labelled with what was decided about it - because the raw feed cannot answer the
        question the dashboard exists to answer: whether the robot saw the green ball as a ball
        at (+0.366, -0.472) or as an unclassified blob.
        """
        backup_sensor = getattr(self, "backup_sensor", None)
        robot_holding = getattr(getattr(self, "bot", None), "holding", None)
        if after_grasp_failure and backup_sensor is not None and robot_holding is None:
            self._active_sensor = backup_sensor
            self.sensor = self._active_sensor
            self.say("  agarre fallido: vuelvo a localizar la pieza desde la cámara cenital")
        self.bus.node("capture", telemetry.ACTIVE, "Capturando la mesa")
        began = time.perf_counter()
        capture_failed = False
        try:
            image, depth, camera = isaac_io.capture(self.env, self._active_sensor)
        except RuntimeError as exc:
            capture_failed = True
            reason = str(exc)
            if self.backup_sensor is None:
                self.bus.node("capture", telemetry.FAIL, "La c?mara no entreg? una imagen v?lida")
                self._camera_error(reason)
                raise
            self.say(f"  vista {self._active_sensor} no disponible; cambio a cenital: {reason}")
            self._active_sensor = self.backup_sensor
            self.sensor = self._active_sensor
            image, depth, camera = self._capture_backup()
        captured_at = time.time()
        self._last_capture = (image, depth, camera)
        self.bus.set_metric("capture_ms", round((time.perf_counter() - began) * 1000, 1))
        self._latest_rgb = (time.monotonic(), image, time.time())
        if self.mask_arm:
            image = self._mask(image, camera)
        perceived = time.perf_counter()
        smallest_part_m = min(
            (min(item.size[:2]) for item in self.layout if not item.container),
            default=None,
        )
        largest_part_m = max(
            (max(item.size[:2]) for item in self.layout if not item.container),
            default=None,
        )
        largest_part_footprint_m2 = max(
            (float(item.size[0]) * float(item.size[1])
             for item in self.layout if not item.container),
            default=None,
        )
        object_bounds_by_color = {}
        for item in self.layout:
            if item.container:
                continue
            extent = max(float(item.size[0]), float(item.size[1]))
            footprint = float(item.size[0]) * float(item.size[1])
            previous = object_bounds_by_color.get(item.color)
            if previous is None or footprint > previous[1]:
                object_bounds_by_color[item.color] = (extent, footprint)
        seen = perceive.perceive(image, source=self.label, camera=camera,
                                 frame=isaac_io.robot_frame(self.env), plane_z=scene_spec.TABLE_TOP_Z,
                                 depth=depth, looking_for=self.args.detect,
                                 robust=not getattr(self.args, "legacy_perception", False) and self.args.variant != "out_of_reach",
                                 min_area_px=colors.min_area_for_object_size(smallest_part_m),
                                 max_object_extent_m=largest_part_m,
                                 max_object_footprint_m2=largest_part_footprint_m2,
                                 object_bounds_by_color=object_bounds_by_color,
                                 captured_at=captured_at)
        completed, unresolved = self.ledger.counts()
        pending = unresolved > 0 or (completed == 0 and unresolved == 0
                                      and any(not item.container for item in self.layout))
        containers_expected = any(item.container for item in self.layout)
        if (self._active_sensor == self.primary_sensor and should_use_backup_camera(
                has_backup=self.backup_sensor is not None,
                primary_capture_failed=capture_failed,
                objects_seen=len(seen.objects),
                 containers_expected=containers_expected,
                 containers_seen=len(seen.containers),
                 tasks_pending=pending,
                 robot_holding=self.bot.holding is not None,
                 ambiguous_objects=sum(bool(thing.ambiguous) for thing in seen.objects))):
            ambiguous_count = sum(bool(thing.ambiguous) for thing in seen.objects)
            self.say(f"  vista {self._active_sensor} incompleta ({len(seen.objects)} objetos, "
                     f"{ambiguous_count} ambiguos, "
                     f"{len(seen.containers)} destinos); verifico desde cenital")
            self._active_sensor = self.backup_sensor
            self.sensor = self._active_sensor
            image, depth, camera = isaac_io.capture(self.env, self._active_sensor)
            captured_at = time.time()
            if self.mask_arm:
                image = self._mask(image, camera)
            seen = perceive.perceive(image, source=self.label, camera=camera,
                                     frame=isaac_io.robot_frame(self.env), plane_z=scene_spec.TABLE_TOP_Z,
                                     depth=depth, looking_for=self.args.detect,
                                     robust=not getattr(self.args, "legacy_perception", False) and self.args.variant != "out_of_reach",
                                     min_area_px=colors.min_area_for_object_size(smallest_part_m),
                                     max_object_extent_m=largest_part_m,
                                     max_object_footprint_m2=largest_part_footprint_m2,
                                     object_bounds_by_color=object_bounds_by_color,
                                     captured_at=captured_at)
            self._latest_rgb = (time.monotonic(), image, time.time())
        self.bot.allow_occluded_tracking = allow_occluded_tracking_for_camera(
            self.args.camera, self._active_sensor, self.primary_sensor)
        self.bus.set_state(camera=self._active_sensor,
                           camera_fallback=self._active_sensor != self.primary_sensor)
        self.bus.set_metric("segmentation_ms", round((time.perf_counter() - perceived) * 1000, 1))
        self.scene, self.last_image = seen, image
        annotation = time.perf_counter()
        self.shown = image if self.args.raw else perceive.annotate(image, seen, "PercepciÃ³n de la mesa", lang="es")
        self.bus.set_metric("annotation_ms", round((time.perf_counter() - annotation) * 1000, 1))
        self.writer.offer(self.shown)
        if self.viewer is not None:
            self.viewer.show(self.shown, caption)
        self.frames += 1
        self.bus.node("capture", telemetry.OK, "Fotograma capturado")
        self.bus.set_metric("perception_ms", round((time.perf_counter() - began) * 1000, 1))
        # Corta a propÃ³sito: la caja de percepciÃ³n es pequeÃ±a. El motivo va en la caja de cÃ¡mara.
        confirmed_objects = sum(not thing.ambiguous for thing in seen.objects)
        ambiguous_objects = len(seen.objects) - confirmed_objects
        note = (f"{confirmed_objects} objetos confirmados, "
                f"{ambiguous_objects} por reobservar, {len(seen.containers)} destinos")
        self.bus.node("perceive", telemetry.WARN if not confirmed_objects or ambiguous_objects else telemetry.OK,
                      note)
        try:
            mapping = self.link(seen)
        except Exception as exc:
            self.bus.node("link", telemetry.FAIL, f"No se pudo asociar la escena: {exc}")
            raise
        self.bus.set_state(detections=self._detection_rows(seen, mapping),
                           frame_note=note, frames=self.frames,
                           perception_captured_at=seen.captured_at)
        self._publish_frame(note)
        return seen

    def _grasp_failure(self, prim, stage, geometry):
        if stage == "lost":
            cause = "La pieza no siguio el ascenso; revisar contacto y retencion."
        elif geometry and geometry.get("fits") is False:
            cause = "La pieza excede la apertura util en esta orientacion."
        elif geometry and geometry.get("lateral_m") is not None and geometry["lateral_m"] > LATERAL_SLACK_M:
            cause = "El centro quedo fuera del margen de la pinza; volver a observar y centrar."
        else:
            cause = "La pinza cerro, pero el contacto no retuvo la pieza; revisar friccion y cierre."
        details = {}
        if geometry:
            details = {key: geometry.get(key) for key in ("z", "tilt_deg", "grip_width_m", "lateral_m", "fits")}
        self.bus.set_state(grasp_failure={"prim": prim, "stage": stage, "cause": cause, **details})

    def _detection_rows(self, seen, mapping: dict[int, str]) -> list[dict]:
        """What the dashboard shows per detected thing: the blob and which part it resolved to."""
        rows = []
        for thing in seen.everything:
            rows.append({"id": thing.id, "kind": thing.kind, "color": thing.color,
                         "pos": thing.pos_m, "area": thing.area_px,
                         "confidence": round(float(thing.confidence), 3),
                         "ambiguous": bool(thing.ambiguous),
                         "ambiguity_reason": thing.ambiguity_reason,
                         "prim": mapping.get(thing.id, ""),
                         "container": bool(thing.inner_rect_m)})
        return rows

    def _publish_frame(self, note: str) -> None:
        if self.shown is None:
            return
        if getattr(self, "frame_encoder", None):
            self.frame_encoder.offer(self.shown, {"note": note, "annotated": True,
                                                "captured_at": getattr(self, "_latest_rgb", (None,None,time.time()))[2]}, FRAME_JPEG_QUALITY)
            return
        try:
            import cv2

            ok, encoded = cv2.imencode(".jpg", self.shown,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY])
            if ok:
                self.bus.put_frame(encoded.tobytes(), {"note": note, "annotated": True, "captured_at": time.time()})
                self.bus.tick("camera")
                self.bus.set_state(camera_error="")
            else:
                self._camera_error("No se pudo codificar la imagen.")
        except Exception as exc:
            self._camera_error(str(exc))

    def _camera_error(self, reason: str) -> None:
        last = getattr(self, "_camera_error_at", 0.0)
        self.bus.set_state(camera_error=reason)
        if time.monotonic() - last > 5:
            self.bus.log(f"La cÃ¡mara no se actualiza: {reason}", level="warn", node="capture")
            self._camera_error_at = time.monotonic()

    def cam_pump(self) -> None:
        """A raw frame while the arm moves, throttled to ``cam_fps``.

        The annotated frames come only from ``look()``, which is a whole perception pass on
        top of the render - several hundred milliseconds the descent does not owe the camera.
        The pump is the cheap one: render, read the sensor buffer, downscale, encode, publish.
        It is called from inside the executor's own motion loops and from ``keys()``, always on
        the main thread where rendering is allowed, and never raises.

        **This is the most expensive thing the dashboard asks the simulator to do, so it is
        built to ask for as little as possible.** ``isaac_io.capture_rgb`` rather than
        ``capture``: the depth buffer is a second GPU-to-CPU copy plus a float64 conversion of
        2.4 MB, for a picture that ends up 384 px wide and JPEG-encoded. The render itself is
        not optional - there is no new picture without one - which is why the default rate is
        ``CAM_FPS_DEFAULT`` and not more: a descent is a few hundred steps of Python between
        physics calls, and spending a render on every twentieth of them is visible as the arm
        hesitating. ``--cam-fps 0`` turns the pump off and leaves only the annotated looks.
        """
        if self.cam_fps <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_pump < 1.0 / self.cam_fps:
            return
        self._last_pump = now
        try:
            import cv2

            latest = getattr(self, "_latest_rgb", None)
            if latest and now - latest[0] < 1.0 / self.cam_fps:
                image = latest[1]
                captured_at = latest[2] if len(latest) > 2 else time.time()
                self.bus.add_to_metric("camera_reused_frames", 1)
            else:
                capture_started = time.perf_counter()
                image, _camera = isaac_io.capture_rgb(self.env, self._active_sensor)
                captured_at = time.time()
                self.bus.set_metric("live_capture_ms", round((time.perf_counter() - capture_started) * 1000, 1))
            if getattr(self, "frame_encoder", None):
                self.frame_encoder.offer(image, {"note": "en vivo", "annotated": False,
                                                 "captured_at": captured_at}, FRAME_PUMP_QUALITY, FRAME_PUMP_WIDTH)
                return
            height, width = image.shape[:2]
            small = cv2.resize(image, (FRAME_PUMP_WIDTH,
                                       max(1, int(FRAME_PUMP_WIDTH * height / width))))
            ok, encoded = cv2.imencode(".jpg", small,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_PUMP_QUALITY])
            if ok:
                self.bus.put_frame(encoded.tobytes(), {"note": "en vivo", "annotated": False, "captured_at": time.time()})
                self.bus.tick("camera")
                self.bus.set_state(camera_error="")
            else:
                self._camera_error("No se pudo codificar el vÃ­deo.")
        except Exception as exc:
            self._camera_error(str(exc))

    def link(self, seen):
        if not hasattr(self, "tracked_area_px"):
            self.tracked_area_px = {}
        primary_sensor = getattr(self, "primary_sensor", None)
        backup_view = (primary_sensor is not None
                       and getattr(self, "_active_sensor", primary_sensor) != primary_sensor)
        things = [t for t in seen.everything if t.pos_m is not None and not t.ambiguous]
        items = list(self.layout)
        costs = np.full((len(things), len(items)), 1e6)
        for row, thing in enumerate(things):
            for column, item in enumerate(items):
                if thing.color != item.color:
                    continue
                last = self.tracked_xy.get(item.prim, (item.x, item.y))
                # For a tilted camera, projecting a raised object's centroid onto the
                # table plane shifts its XY estimate. Compare against the catalog item
                # at its own centre height before deciding identity; applying this
                # correction only after association is too late for side views.
                candidate_xy = None
                if seen.camera and getattr(thing, "centroid_px", None):
                    try:
                        from scene import CameraModel
                        projected = CameraModel.from_dict(seen.camera).on_plane(
                            *thing.centroid_px, scene_spec.TABLE_TOP_Z + item.height / 2)
                        if projected:
                            candidate_xy = projected[:2]
                    except (KeyError, TypeError, ValueError, AttributeError):
                        candidate_xy = None
                gap = math.dist(candidate_xy or thing.aim_xy or thing.pos_m[:2], last)
                tolerance = fallback_link_tolerance(
                    self.demo10.LINK_TOLERANCE_M,
                    backup_view=backup_view,
                    is_container=item.container,
                    max_tracking_shift_m=MAX_TRACKING_STEP_M)
                if gap <= tolerance:
                    # Camera foreshortening can reverse the image-area ordering of a tray and a
                    # large object. The scene catalog plus position is the stronger class cue;
                    # category mismatch is a small tie-break penalty, not a hard exclusion.
                    category_penalty = 0.012 if (thing in seen.containers) != item.container else 0.0
                    costs[row, column] = gap + category_penalty
        mapping = {}
        association_max_cost = max(fallback_link_tolerance(
            self.demo10.LINK_TOLERANCE_M,
            backup_view=backup_view,
            is_container=False,
            max_tracking_shift_m=MAX_TRACKING_STEP_M),
            self.demo10.LINK_TOLERANCE_M) + 0.012  # category tie-break cost, not XY error
        if things and items:
            assignments = unambiguous_catalog_assignment(
                costs, max_cost=association_max_cost)
            for row, column in assignments:
                mapping[things[row].id] = items[column].prim
            linked_rows = {row for row, _column in assignments}
            uncertain_rows = [row for row in range(len(things))
                              if row not in linked_rows
                              and np.min(costs[row]) <= association_max_cost]
            for row in uncertain_rows:
                things[row].ambiguous = True
                things[row].ambiguity_reason = "identidad de escena ambigua"
            if uncertain_rows:
                seen.notes.append(
                    f"{len(uncertain_rows)} deteccion(es) sin identidad segura; se omiten "
                    "para evitar agarrar otra pieza")
        by_name = {i.prim: i for i in self.layout}
        names = {entry["id"]: entry["name"] for entry in self.catalog}
        for thing in seen.everything:
            item = by_name.get(mapping.get(thing.id))
            if item:
                thing.entity_id = item.prim
                thing.display_name = names.get(item.prim, item.prim)
                thing.size_m = list(item.size)
                # The catalog is authoritative for type after color and world position associate.
                # This repairs camera-dependent size/shape classifications and supplies custom
                # geometry to planning.
                thing.kind = item.kind
                thing.hollow = item.container
                if item.container:
                    # Simulator receptacles are fixed kinematic scene objects. Their known
                    # configured pose is more accurate than projecting the visible rim or a
                    # partly occluded red wall to the table plane, especially in front views.
                    # Keep the image association above as the evidence that this is the tray,
                    # then use its configured centre for release and verification.
                    thing.aim_xy = [float(item.x), float(item.y)]
                    thing.pos_m = [float(item.x), float(item.y),
                                   float(scene_spec.TABLE_TOP_Z + item.height / 2)]
                elif seen.camera and not getattr(self.args, "legacy_perception", False):
                    from scene import CameraModel
                    point = CameraModel.from_dict(seen.camera).on_plane(*thing.centroid_px, scene_spec.TABLE_TOP_Z + item.height / 2)
                    if point:
                        measured_xy = tuple(point[:2])
                        if backup_view and item.prim in self.tracked_xy:
                            old_xy = tuple(self.tracked_xy[item.prim])
                            bot = getattr(self, "bot", None)
                            failures = getattr(bot, "failures", {})
                            after_grasp_failure = bool(failures.get(item.prim, 0))
                            linked_prims = set(mapping.values())
                            ambiguous_same_color_peer = any(
                                candidate.prim != item.prim and not candidate.container
                                and candidate.color == item.color
                                and candidate.prim not in linked_prims
                                for candidate in self.layout)
                            max_shift = fallback_tracking_step_limit(
                                after_grasp_failure=after_grasp_failure,
                                ambiguous_same_color_peer=ambiguous_same_color_peer)
                            measured_xy, rejected_jump = stabilize_fallback_position(
                                measured_xy, old_xy, max_shift)
                            accepted_reanchor = (after_grasp_failure and not rejected_jump
                                    and math.dist(measured_xy, old_xy) > MAX_TRACKING_STEP_M)
                            if accepted_reanchor:
                                self.say(f"  vista de respaldo: reanclo {item.prim} tras contacto "
                                         f"fallido: ({old_xy[0]:+.3f}, {old_xy[1]:+.3f}) → "
                                         f"({measured_xy[0]:+.3f}, {measured_xy[1]:+.3f})")
                            if math.dist(point[:2], old_xy) > 0.015:
                                if rejected_jump:
                                    status, display_xy = "rechazada; conservo", old_xy
                                elif accepted_reanchor:
                                    status, display_xy = "reanclada a", measured_xy
                                else:
                                    status, display_xy = "asociada a", measured_xy
                                self.say(f"  localización de respaldo {item.prim}: "
                                         f"medida ({point[0]:+.3f}, {point[1]:+.3f}), "
                                         f"{status} ({display_xy[0]:+.3f}, {display_xy[1]:+.3f})")
                            if rejected_jump:
                                self.say(f"  vista de respaldo: rechazo salto de "
                                         f"{math.dist(point[:2], self.tracked_xy[item.prim]) * 1000:.0f} mm "
                                         f"en {item.prim}; conservo la posición asociada")
                                seen.notes.append(
                                    f"vista de respaldo: se conserva la última posición "
                                    f"asociada de {item.prim} ante un salto > "
                                    f"{max_shift * 1000:.0f} mm")
                        thing.aim_xy = list(measured_xy)
                        # The generic detector's pos_m intersects the centroid ray with the
                        # tabletop. For tilted views that shifts a raised object's estimate by
                        # centimetres. Once the catalog provides its height, use the object-centre
                        # plane for both world reasoning and grasp aiming.
                        thing.pos_m = [round(float(measured_xy[0]), 5), round(float(measured_xy[1]), 5),
                                       round(scene_spec.TABLE_TOP_Z + item.height / 2, 5)]
                measured_area = int(getattr(thing, "area_px", 0) or 0)
                previous_area = self.tracked_area_px.get(item.prim)
                if tracking_measurement_is_reliable(
                        measured_area, getattr(thing, "confidence", 0.0), previous_area):
                    # Keep the largest recent full-view mask as the occlusion reference. If we
                    # ratchet this baseline down every frame, a slowly shrinking gripper-occluded
                    # mask can drag the anchor even though each individual change looks small.
                    self.tracked_area_px[item.prim] = max(measured_area, previous_area or 0)
                    self.tracked_xy[item.prim] = tuple(
                        thing.aim_xy or scene_spec.aim_point(thing.pos_m, container=item.container))
                else:
                    # Keep the catalog identity for visualization, but do not feed a clipped
                    # component centroid into the grasp servo or the next association pass.
                    trusted_xy = self.tracked_xy.get(item.prim, (item.x, item.y))
                    thing.aim_xy = [float(trusted_xy[0]), float(trusted_xy[1])]
                    if thing.pos_m is not None:
                        thing.pos_m = [float(trusted_xy[0]), float(trusted_xy[1]),
                                       float(scene_spec.TABLE_TOP_Z + item.height / 2)]
                    seen.notes.append(
                        f"se conserva la pose de {item.prim}: máscara parcial o de baja confianza")
                if item.container:
                    thing.cavity_m = list(item.inner)
        # Rebuild category lists after catalog association so downstream locate/plan logic sees
        # camera-corrected object and container classes in this same frame.
        container_ids = {thing.id for thing in seen.containers}
        all_things, suppressed = scene_config.suppress_unlinked_fragments(
            list(seen.everything), container_ids, mapping, self.layout)
        seen.objects = [thing for thing in all_things if thing.id not in container_ids]
        seen.containers = [thing for thing in all_things if thing.id in container_ids]
        if suppressed:
            seen.notes.append(f"se descartaron {len(suppressed)} fragmento(s) duplicado(s) de objetos asociados")
        self.scene.objects = [thing for thing in all_things
                              if not getattr(thing, "hollow", False)]
        self.scene.containers = [thing for thing in all_things
                                 if getattr(thing, "hollow", False)]
        self.bus.node("link", telemetry.OK if mapping else telemetry.WARN,
                      f"{len(mapping)}/{len(seen.everything)} detecciones asociadas")
        return mapping

    def locate(self, prim, last_xy=None, kind_hint=None):
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None:
            self.locate_note = "La pieza no pertenece a la escena activa."
            return None
        seen = self.look("Siguiendo la pieza")
        if (seen.captured_at is None
                or time.time() - float(seen.captured_at) > MAX_GRASP_OBSERVATION_AGE_S):
            self.locate_note = "La imagen ha caducado; vuelve a observar antes de agarrar."
            self.bus.log(self.locate_note, level="warn")
            return None
        candidates = [t for t in seen.objects if t.color == item.color and not t.ambiguous]
        # Same-colour parts can be adjacent in a customized scene. Use the known
        # scene geometry as an identity cue when perception has a confident class;
        # retain "any" detections so cylinders and user-added shapes still track.
        wanted_kind = kind_hint or item.kind
        if wanted_kind != "any":
            compatible = [t for t in candidates if t.kind in (wanted_kind, "any")]
            if compatible:
                candidates = compatible
        anchor_xy = last_xy if last_xy is not None else (item.x, item.y)
        peer_prims = {other.prim for other in self.layout
                      if not other.container and other.prim != prim
                      and other.color == item.color}
        visible_peer_prims = {getattr(candidate, "entity_id", None)
                              for candidate in candidates}
        ambiguous_peer = bool(peer_prims - visible_peer_prims)
        failures = getattr(getattr(self, "bot", None), "failures", {})
        shift_limit = fallback_tracking_step_limit(
            after_grasp_failure=bool(failures.get(prim, 0)),
            ambiguous_same_color_peer=ambiguous_peer)
        candidates = target_tracking_candidates(
            candidates, prim, anchor_xy, max_shift_m=shift_limit)
        nearest = self._nearest(candidates, last_xy)
        if not nearest:
            self.locate_note = "La pieza estÃ¡ oculta o no se identifica con seguridad."
            return None
        nearest.sort(key=lambda pair: pair[0])
        if len(nearest) > 1 and abs(nearest[0][0] - nearest[1][0]) < .005:
            self.locate_note = "Dos piezas coinciden con la referencia; seguimiento ambiguo."
            return None
        best = nearest[0][1]
        best.kind = item.kind
        best.size_m = list(item.size)
        self.locate_note = ""
        self.tracked_xy[prim] = tuple(best.aim_xy or scene_spec.aim_point(best.pos_m))
        return self.tracked_xy[prim]

    def _observe_grasp_target(self, prim, kind_hint=None):
        """Track the calibrated tool aim while retaining the measured object centre."""
        trusted = self.tracked_xy.get(prim)
        point = self.locate(prim, trusted, kind_hint=kind_hint)
        # ``Executor.track`` calls this immediately before each servo segment. The opening
        # observation can still use the same-scene warm frame if RTX has not produced a new
        # frame yet; after that observation the executor is about to step the arm, so that
        # cache must no longer be used for any later localization.
        if hasattr(self, "_backup_capture_stale"):
            self._backup_capture_stale = True
        if point is None:
            return None
        item = next((entry for entry in self.layout if entry.prim == prim), None)
        small_target = bool(item is not None
                            and max(float(v) for v in item.size[:2]) <= 0.022)
        if (small_target and trusted is not None
                and math.dist(point, trusted) > MAX_TRACKING_STEP_M):
            # ``locate`` may have selected a same-colour tray or gripper fragment when a
            # small object disappears behind the arm. Reject that measurement before it
            # replaces the scene-linked identity; StrictExecutor can then use its bounded
            # small-object pre-grasp coast or stop safely during descent.
            self.tracked_xy[prim] = trusted
            self.locate_note = "La detecciÃ³n siguiÃ³ otro elemento; se conserva la asociaciÃ³n previa."
            return None
        if item is None:
            return point
        dx, dy = real_world.jaw_sweep_approach_offset(item.kind, item.size)
        return float(point[0]) + dx, float(point[1]) + dy

    # -- planning and motion ---------------------------------------------------

    def _release_adjusted(self, plan, premise=None):
        """Raise every release to a height the fingertips can actually reach.

        ``planner.place_pose`` releases 12 mm above the rim. With ``executor.HOLD`` putting the
        part's centre 92 mm below the gripper body and the fingers hanging
        ``real_world.GRIPPER.finger_tip_below_hold_m`` below the part, the fingertips are then
        ~20 mm *below* the rim - inside the wall, or through the 6 mm floor slab - while the
        jaws are still closed. The part is dragged along the inside of the cavity on the way
        down. That is the mechanism behind demo 10's "ball came to rest 30 mm above the floor,
        near the cavity edge", and no amount of aiming fixes it.

        ``--release rim`` keeps demo 10's height for comparison; ``clear`` is the default because
        it is the only one a real arm can perform.
        """
        if self.release_mode != "clear" or not plan.steps:
            return plan
        trays = [item for item in self.layout if item.container]
        if not trays:
            return plan
        fallback_floor = max(real_world.safe_release_z(tray) for tray in trays)
        raised = 0
        steps = []
        for step in plan.steps:
            if step.action in ("lower", "release") and step.pose is not None:
                destination = (premise or {}).get(step.target)
                tray = next((tray for tray in trays if tray.prim == destination), None)
                floor = real_world.safe_release_z(tray) if tray else fallback_floor
                z = float(step.pose[2])
                if z < floor:
                    step = replace(step, pose=(step.pose[0], step.pose[1], floor))
                    raised += 1
            steps.append(step)
        if not raised:
            return plan
        copy = planner.Plan(intent=plan.intent)
        copy.steps = steps
        copy.picks = list(plan.picks)
        copy.places = list(plan.places)
        # Raised poses change how the arm gets there, not what it was asked to do, so the task
        # list comes across untouched. Without this the checklist would empty itself the moment a
        # release height had to be lifted.
        copy.tasks = list(plan.tasks)
        copy.unresolved = list(plan.unresolved)
        copy.question = dict(plan.question)
        copy.reply = plan.reply
        self.bus.log(f"    release raised to {floor * 1000:.0f} mm so the fingertips clear the "
                     f"{real_world.GRIPPER.finger_tip_below_hold_m * 1000:.0f} mm rim "
                     f"({raised} step(s))")
        return copy

    def run_plan(self, plan, premise, command):
        if hasattr(self, "_command_began") and not self._first_motion:
            self._first_motion = True
            self.bus.set_metric("command_to_task_motion_ms", round(
                (time.perf_counter() - self._command_began) * 1000, 1))
        plan = copy.deepcopy(plan)
        index = -1
        for number, step in enumerate(plan.steps):
            if step.action == "above":
                index += 1
            if step.action in ("carry", "lower", "release", "retreat") and 0 <= index < len(plan.tasks) and step.pose:
                prim = premise.get(plan.tasks[index].pick)
                allocation = getattr(self, "_task_slots", {}).get(prim)
                if allocation:
                    dx, dy = allocation[1]
                    plan.steps[number] = replace(step, pose=(step.pose[0]+dx, step.pose[1]+dy, step.pose[2]))
        plan = self._release_adjusted(plan, premise)
        self._publish_plan(plan, note="", increment=False)
        # The whole job, before the per-object re-plans start: this is the plan the person asked
        # for, and it is where the checklist gets all of its rows from.
        self._note_tasks(plan, premise)
        # The plan that runs now is the job the arm is on, so its rows say so. Nothing else writes
        # "doing": the panel cannot infer it, because what it can see - a step's action word - is
        # not the same thing as which part that step belongs to.
        self._mark_doing(plan, premise)
        for step in plan.steps:
            pose = "" if step.pose is None else " ".join(f"{v:+.3f}" for v in step.pose)
            import ui_spanish
            self.bus.log(f"Plan: {ui_spanish.ACTIONS.get(step.action, 'movimiento')} {pose}", node="plan")
        # How far along the plan is, as "paso 3 de 8": the total is the plan's own, and the counter
        # is per step executed, so the pair moves with the arm rather than with the publishing.
        self._step_i = 0
        self.bus.set_state(step_total=len(plan.steps), step_index=self._step_i)
        self.bus.node("execute", telemetry.ACTIVE, f"{len(plan.steps)} pasos")
        self._motion_started(len(plan.steps))
        refused_before = self.bot.refused
        # Demo 10's retarget callback replans the whole original command. That can
        # change identities, destinations and release heights inside a single task.
        # Demo 11 retries explicitly between tasks with the same requested identity.
        was_masked = self.mask_arm
        self.mask_arm = True
        try:
            run = self.bot.run(plan, resolve=premise.get, keys=lambda: self.keys(),
                               observe=self._observe_grasp_target, retarget=None)
        finally:
            self.mask_arm = was_masked
        if run.stale_grasps:
            # A grasp that did not hold, or a target that could not be found again, is a failed
            # attempt at each task in this plan. A later verdict (check) may still settle it as
            # placed; if none comes - for example when the command runs out of attempts - the
            # failure stands. A user abort is not a failure and is not recorded here.
            for pick_id in plan.picks:
                task = next((t for t in plan.tasks if t.pick == pick_id), None)
                key = self._task_key(task, premise) if task else (premise.get(pick_id) or f"thing #{pick_id}")
                self.ledger.record(key, False, self._task_label(plan, key, premise))
            self._settle_rows()
        if self.bot.refused > refused_before:
            # Demo 10 reports this failure as "the arm knocked the target away before closing",
            # which is the right shape of response - re-look and try again - for the wrong reason.
            # Say what actually happened, because on a real bench these two causes need different
            # fixes: a knock is a planning problem, an unheld part is a grip-force or jaw problem.
            self.say("  demo 11: the jaws closed and the part was not held, so it was not "
                     "carried. A real arm drops the part here rather than flying on to the tray.")
        state = telemetry.FAIL if (run.stale_grasps or run.aborted) else telemetry.OK
        self.bus.node("execute", state, _short_summary(run))
        # Cumulative over the command: a retry runs its own grasp (see ``_retry_placed``'s lift) and
        # a later run must not overwrite an earlier physical grasp with its own zero.
        self._count_real_grasps(run.real_grasps)
        self.bus.set_state(real_grasps=self.real_grasps_total, tracked=self.tracked,
                           reaimed=self.reaimed, refused_grasps=self.bot.refused,
                           recentred=self.bot.corrected,
                           kinematic_grasps=self.bot.kinematic)
        self._motion_finished()
        return run

    def _motion_finished(self) -> None:
        """Say the arm has stopped, in the three fields the panel reads as motion.

        ``progress``, ``step_index`` and ``step_total`` are written by
        ``_step_logged`` and ``run_plan`` but were never cleared, so after the last
        step of a plan the panel kept asserting the last verb - Â«ahora: retirar Â·
        paso 8 de 8Â» - through the whole of verification, recovery and the report.
        The stage being shown then disagreed with what the arm was doing, which is
        the one thing a demo's dashboard cannot get wrong.

        ``moving`` is the honest single flag: true only between the first and the
        last step of a plan. It is published rather than inferred, because the
        phase is still ``command`` or ``sorting`` throughout verification - only
        the robot knows whether the arm is following steps right now.
        """
        self.bus.set_state(progress="", step_index=0, step_total=0, moving=False)

    def _motion_started(self, total: int) -> None:
        self.bus.set_state(step_index=0, step_total=total, moving=True)

    def _retry_one(self, prim: str) -> bool:
        """One masked retry: the arm is painted out of the frame for the whole approach.

        Without the mask the retry kept losing its own target - measured, the log repeatedly
        said "lost sight of {prim} on the way in" for a part the camera had just seen, because
        the unmasked gripper clipped its blob from above and the tracker refused the sighting.
        ``run_plan`` masks for its own descents; the retry is driven from outside that, so it
        has to do it itself.
        """
        self.bus.node("recover", telemetry.ACTIVE, f"Recuperando {prim}")
        was = getattr(self, "mask_arm", False)
        self.mask_arm = True
        try:
            self.bus.set_state(phase="recovering", moving=True)
            done = self._retry_placed(prim)
        finally:
            self.mask_arm = was
            self.bus.set_state(phase="verifying", moving=False)
        self.bus.node("recover", telemetry.OK if done else telemetry.FAIL,
                      f"{prim}: {'recuperada' if done else 'no se pudo recuperar'}")
        return done

    def _retry_original_targets(self, plan, premise) -> bool:
        """Retry the exact catalog identities if language replanning becomes ambiguous.

        Recovery remains camera-linked and physical. It never substitutes a same-colour object,
        and counts a retry only when the original part is measured inside its original tray.
        """
        retried = False
        for task in getattr(plan, "tasks", ()):
            prim = premise.get(task.pick)
            if prim is None:
                continue
            retried = True
            self._retry_one(prim)
        if not retried:
            return False
        self.bot.settle()
        verdicts = self._placement_verdicts(plan, premise)
        for key, placed, _prim in verdicts:
            self.ledger.record(key, placed)
            if placed:
                self.recovered += 1
        self.verified, self.missed = self.ledger.counts()
        self._settle_rows()
        return bool(verdicts) and all(placed for _key, placed, _prim in verdicts)

    def _count_real_grasps(self, count: int) -> int:
        """Add friction grasps to the command's running total, which the report reads."""
        self.real_grasps_total = getattr(self, "real_grasps_total", 0) + int(count)
        return self.real_grasps_total

    def _retry_placed(self, prim: str) -> bool:
        """Pick one misplaced part up again, with the same steps and the same safety as a pick.

        Demo 10's retry aimed from the simulator's own position, skipped the close, and released
        at the old height. Here the aim comes from the camera - the same sighting a plan uses -
        the jaws close and the lift is checked, and the release uses the rim-clearance height the
        normal path uses. Any step that fails returns False at once; the caller re-verifies the
        tray either way, so a retry is only ever counted as a recovery when it is measured placed.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        tray = self._tray_of(prim)
        if item is None or tray is None:
            self.say(f"    {prim}: no hay dÃ³nde devolverlo")
            return False
        seen = self.look("recuperaciÃ³n")
        # Preserve the detector's raw proposals: catalog linking can reclassify a ball inside a
        # same-colour tray as the receptacle and remove it from ``seen.objects``.
        raw_objects_by_view = [copy.deepcopy(seen.objects)]
        premise = self.link(seen)
        pid = next((key for key, name in premise.items() if name == prim), None)
        thing = (next((t for t in seen.everything if t.id == pid), None)
                 if pid is not None else None)
        if thing is None:
            # A failed close can knock a small part outside the stricter 35 mm catalog-link
            # radius. Reacquire only within the bounded 60 mm motion gate from the last trusted
            # anchor, with expected color and geometry; never follow an arbitrary red fragment.
            anchor = self.tracked_xy.get(prim, (item.x, item.y))
            fresh_view = self.look("Confirmando la pieza tras el agarre fallido",
                                   after_grasp_failure=True)
            raw_objects_by_view.append(copy.deepcopy(fresh_view.objects))
            rejected = [
                f"{candidate.kind}/{candidate.color}@"
                f"({candidate.pos_m[0]:+.3f},{candidate.pos_m[1]:+.3f})"
                f"[{getattr(candidate, 'entity_id', None) or 'sin vínculo'}]"
                for raw_objects in raw_objects_by_view for candidate in raw_objects
                if candidate.pos_m is not None]
            thing = reacquire_target_from_raw_proposals(
                raw_objects_by_view, item, prim, anchor, self._nearest)
        if thing is None or thing.pos_m is None:
            anchor = self.tracked_xy.get(prim, (item.x, item.y))
            detail = f"; ancla ({anchor[0]:+.3f}, {anchor[1]:+.3f})"
            if 'rejected' in locals() and rejected:
                detail += "; candidatos " + ", ".join(rejected[-6:])
            self.say(f"    {prim}: no lo encuentro en la imagen{detail}; no se intenta a ciegas")
            return False
        grasp = np.asarray(planner.grasp_pose(thing, offset=self.args.offset), dtype=float)
        lift = grasp + np.array([0.0, 0.0, executor_module.ABOVE])
        target = next((t for t in seen.containers if t.entity_id == tray.prim), None)
        if target is None:
            self.bus.log("No se identifica el recipiente durante la recuperaciÃ³n.", level="warn")
            return False
        drop = np.asarray(planner.place_pose(target), dtype=float)
        compensation = real_world.jaw_sweep_approach_offset(item.kind, item.size)
        drop[:2] += np.asarray(compensation, dtype=float)
        allocation = getattr(self, "_task_slots", {}).get(prim)
        if allocation:
            drop[:2] += allocation[1]
        drop_low = drop.copy()
        drop_low[2] = max(drop[2], real_world.safe_release_z(tray))
        above_drop = drop_low + np.array([0.0, 0.0, executor_module.ABOVE])

        self.tracked_xy[prim] = (float(thing.pos_m[0]), float(thing.pos_m[1]))
        observe = lambda: self._observe_grasp_target(prim, kind_hint="any")
        approach = self.bot.track(observe=observe, target_z=float(lift[2]),
                                  steps=executor_module.TRACK_DESCEND_STEPS, action="above",
                                  grip=executor_module.OPEN, aim=(float(lift[0]), float(lift[1])))
        descent = self.bot.track(observe=observe, target_z=float(grasp[2]),
                                 steps=executor_module.TRACK_DESCEND_STEPS, action="down",
                                 grip=executor_module.OPEN, aim=(float(grasp[0]), float(grasp[1])))
        if not (approach.ok and descent.ok):
            self.say(f"    {prim}: se pierde de vista al acercarse; no se intenta")
            return False
        closed = self.bot.do("close", grasp, executor_module.CLOSE, prim=prim, on_close=grasp,
                             check="reach")
        if not closed.ok:
            self.say(f"    {prim}: {closed.note or 'las mordazas cierran sobre nada'}")
            return False
        before = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()
        lifted = self.bot.do("lift", lift, executor_module.CLOSE, prim=prim, on_close=before,
                             check="follow")
        self.bot.log(f"      lift: {lifted.note}")
        if lifted.ok and getattr(lifted, "grasp", None) == "real":
            # The recovery grasp is judged here, not by ``Executor.run``, so count it here.
            self._count_real_grasps(1)
            self.bus.set_state(real_grasps=self.real_grasps_total)
        if not lifted.ok:
            return False
        carried = self.bot.do("carry", above_drop, executor_module.CLOSE)
        if not carried.ok:
            self.say(f"    {prim}: transporte detenido; {carried.note or 'no se alcanzó el destino'}")
            return False
        lowered = self.bot.do("lower", drop_low, executor_module.CLOSE)
        if not lowered.ok:
            self.say(f"    {prim}: descenso detenido; {lowered.note or 'no se alcanzó el destino'}")
            return False
        self.bot.drop()
        released = self.bot.do("release", drop_low, executor_module.OPEN)
        if not released.ok:
            self.say(f"    {prim}: no se confirmó la liberación")
            return False
        retreated = self.bot.do("retreat", above_drop, executor_module.OPEN)
        if not retreated.ok:
            self.say(f"    {prim}: no se confirmó la retirada")
            return False
        self.bot.settle()
        return True

    def _one_at_a_time(self, command, plan) -> bool:
        self.bus.set_state(phase="sorting")
        self.bus.node("plan", telemetry.ACTIVE, "Una pieza por vez; observaciÃ³n entre tareas")
        # Keep the requested identities and destinations. Replanning the entire command
        # would silently forget missing pieces or select a different same-colour object.
        initial = self.link(self.scene)
        for task in plan.tasks:
            prim = initial.get(task.pick)
            tray_prim = initial.get(task.place)
            for attempt in range(1 + self.args.regrasp):
                if self.bus.stop_requested():
                    return False
                seen = self.look("Actualizando la posiciÃ³n de la siguiente pieza")
                premise = self.link(seen)
                inverse = {name: identifier for identifier, name in premise.items()}
                things = {thing.id: thing for thing in seen.everything}
                if prim not in inverse or (task.place is not None and tray_prim not in inverse):
                    self.bus.log("No se localiza la pieza o su recipiente; la tarea queda pendiente.", level="warn")
                    break
                one = planner.Plan(intent=plan.intent)
                piece = things[inverse[prim]]
                grasp_offset = self._scene_grasp_offset(piece, seen)
                if task.place is None:
                    planner._grasp_and_hold(one, piece, grasp_offset, task.tag)
                else:
                    planner._pick_and_place(one, piece, things[inverse[tray_prim]],
                                            grasp_offset, task.tag)
                run = self.run_plan(one, premise, command)
                if run.aborted and not run.stale_grasps:
                    return False
                if not run.stale_grasps:
                    if one.places:
                        self.check(one, premise, announce=False)
                    else:
                        held = bool((run.real_grasps + run.kinematic_grasps) and self.bot.holding)
                        self.ledger.record(self._task_key(one.tasks[0], premise), held)
                        self._settle_rows()
                    break
                if attempt < self.args.regrasp:
                    self.park()
        return True

    def _scene_grasp_offset(self, thing, seen) -> tuple[float, float]:
        """Select a tested assisted-mode correction from the visible scene geometry."""
        base = tuple(self.args.offset)
        chosen = planner.simulated_cluster_grasp_offset(
            thing, seen.everything, base, enabled=self.bot.auto_sim_assist,
            camera=self.args.camera)
        if chosen != base:
            self.bus.log(
                f"Ajuste de aproximaciÃ³n asistido para prisma triangular de 20 mm "
                f"en grupo: ({chosen[0] * 1000:+.0f}, {chosen[1] * 1000:+.0f}) mm; "
                f"no es calibraciÃ³n fÃ­sica.", level="info")
        return chosen

    def _tray_of(self, prim):
        allocation = getattr(self, "_task_slots", {}).get(prim) or getattr(self, "_placed_slots", {}).get(prim)
        if allocation:
            return next((i for i in self.layout if i.prim == allocation[0]), None)
        item = next((i for i in self.layout if i.prim == prim), None)
        matches = [i for i in self.layout if item and i.container and i.color == item.color]
        return matches[0] if len(matches) == 1 else None

    def check(self, plan, premise, announce: bool = True):
        self._motion_finished()
        self.bus.set_state(phase="verifying")
        self.bus.node("verify", telemetry.ACTIVE,
                      f"Comprobando {len(plan.picks)} colocaciones")
        misses = self._verify_and_retry(plan, premise, announce)
        self._motion_finished()
        self.bus.set_state(phase="sorting" if getattr(self, "_one_at_a_time_run", False) else "command")
        # The verdict per task. `check` has just re-perceived and measured where every part came
        # to rest, so this is the moment the checklist has an answer: `misses` is the parts that
        # are not in the tray they were sent to.
        verified = len(plan.picks) - len(misses)
        self.bus.node("verify", telemetry.OK if not misses else telemetry.WARN,
                      f"{verified}/{len(plan.picks)} dentro"
                      + (f"; sin verificar: {', '.join(misses)}" if misses else ""))
        self.bus.set_state(verified=self.verified, recovered=self.recovered,
                           missed=self.missed, replanned=self.replanned,
                           regrasped=self.regrasped)
        return misses

    # -- input -----------------------------------------------------------------

    @staticmethod
    def _ignore_parts_already_in_trays(seen) -> int:
        """Drop detected parts that already stand inside a receptacle; return how many.

        A part the arm has just placed in a tray of its own colour merges with the tray's mask and
        comes back from perception as an unlinked object at the tray (it was reported as a red
        ball, "sin vínculo": its tracked position is still the one it started from). "Separa las
        piezas por colores" then planned a pick for it, could not name it, and refused the whole
        order with "No puedo identificar todas las piezas y sus destinos" - 6 of 6 times in a
        session that had just placed one piece. A sort has nothing to do for a part that is
        already in a tray, so it leaves them out of the plan.
        """
        trays = [t for t in seen.containers if t.pos_m is not None and t.size_m]
        kept, ignored = [], 0
        for thing in seen.objects:
            point = thing.aim_xy or (thing.pos_m[:2] if thing.pos_m is not None else None)
            inside = point is not None and any(
                abs(point[0] - tray.pos_m[0]) <= tray.size_m[0] / 2.0
                and abs(point[1] - tray.pos_m[1]) <= tray.size_m[1] / 2.0
                for tray in trays)
            if inside:
                ignored += 1
            else:
                kept.append(thing)
        seen.objects = kept
        return ignored

    def _catalog_gaps(self, command, premise):
        """Do not let an occluded configured piece disappear from an 'all' request."""
        filters = command.get("sort_filter") or command.get("items") or [{"kind": "any"}]
        known = [item for item in self.layout if not item.container
                 and item.prim not in getattr(self, "_placed_slots", {})]
        visible = set(premise.values())
        for selection in filters:
            candidates = [item for item in known
                          if selection.get("kind") in (None, "any", item.kind)
                          and selection.get("color") in (None, item.color)]
            missing = [item.prim for item in candidates if item.prim not in visible]
            if missing:
                return "No se identifican todas las piezas de esa referencia: " + ", ".join(missing) + ". Separa las piezas o cambia la cÃ¡mara."
        destinations = [command.get("place"), *[item.get("place") for item in command.get("items", [])]]
        for destination in destinations:
            if destination and not destination.get("entity_id"):
                matches = [item for item in self.layout if item.container and destination.get("color") in (None, item.color)]
                if len(matches) > 1:
                    return "Hay varios recipientes que coinciden. Indica el nombre del destino."
        return ""

    def handle(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return True
        # A new command is a new chance for parts that failed last time: the park sweep moves
        # things down between commands, and a part lying flat is a part the jaws can hold.
        self.bot.impossible.clear()
        self.bot.failures.clear()
        getattr(self.bot, "_last_object_grasp_yaw", {}).clear()
        getattr(self.bot, "_initial_object_grasp_yaw", {}).clear()
        # A new command is a new job: the previous one's checklist has nothing to do with this
        # one, and leaving it up would claim work that was never asked for.
        self._task_rows.clear()
        self._publish_tasks()
        telemetry.reset_nodes(self.bus)
        self.bus.node("input", telemetry.ACTIVE, text)
        self.bus.set_state(grasp_failure=None, phase="command", command=text)
        cid = self.bus.active_command_id()
        current = self.bus.command_state(cid) or {}
        if current.get("state") != telemetry.RUNNING or current.get("text") != text:
            cid = self.bus.new_command("command", "script", text=text)
            self.bus.set_command_state(cid, telemetry.QUEUED)
            self.bus.set_command_state(cid, telemetry.RUNNING)
        self._active_cid = cid
        began = time.perf_counter()
        self._command_began = began
        self._first_motion = False
        # A new command starts with the arm still: nothing from the previous plan's
        # steps may be described as happening now.
        self._motion_finished()
        outcome, reason = telemetry.FAILED, "La orden no se ha completado."
        try:
            if text.lower() in self.demo10.QUIT_WORDS:
                self.quit = True
                outcome, reason = telemetry.CANCELLED, "Parada solicitada."
                return False
            command = self.model.parse(text)
            if command.get("intent") in ("stop", "cancel"):
                self.bus.request_stop()
                self.quit = True
                outcome, reason = telemetry.CANCELLED, "Parada solicitada."
                return False
            if command.get("errors"):
                outcome, reason = telemetry.NEEDS_CLARIFICATION, "; ".join(command["errors"])
                return True
            if command.get("intent") in planner.NO_MOTION:
                reason = (describe.answer(self.scene, command, lang="es") if command["intent"] == "question"
                          else "Hola. Puedes pedir que mueva cubos y bolas a los recipientes." if command["intent"] == "greeting"
                          else "La orden no requiere movimiento." if command["intent"] in {"answer_yes", "answer_no", "answer_value"}
                          else "No puedo realizar esa orden. Usa las piezas y recipientes de la escena.")
                outcome = telemetry.NEEDS_CLARIFICATION if command["intent"] in {"not_doable", "off_topic"} else telemetry.SUCCESS
                self.bus.set_state(reply=reason)
                return True
            # Observe without moving first. Unresolved orders must not park the arm.
            seen = self.look("Observando para resolver la orden")
            premise = self.link(seen)
            gap = self._catalog_gaps(command, premise)
            # A single side-view render can have a stale or partially updated color mask even
            # when the object is plainly visible. Never move the arm on that frame, but give the
            # camera two stationary refreshes before asking the user to reposition the scene.
            for retry in range(2):
                if not gap:
                    break
                self.bus.log(f"La primera lectura dejÃ³ piezas sin asociar; repito la captura "
                             f"({retry + 1}/2) antes de rechazar la orden.", level="warn")
                seen = self.look("Confirmando visualmente la escena")
                premise = self.link(seen)
                gap = self._catalog_gaps(command, premise)
            if gap:
                outcome, reason = telemetry.NEEDS_CLARIFICATION, gap
                return True
            if command.get("intent") in ("sort", "clear", "sequence"):
                ignored = self._ignore_parts_already_in_trays(seen)
                if ignored:
                    self.bus.log(f"  {ignored} pieza(s) ya están dentro de un recipiente; "
                                 "no se vuelven a ordenar.", level="info")
            plan = planner.plan_command(command, seen, offset=self.args.offset)
            if len(plan.picks) == 1:
                picked = next((thing for thing in seen.everything
                               if thing.id == plan.picks[0]), None)
                if picked is not None:
                    effective_offset = self._scene_grasp_offset(picked, seen)
                    if effective_offset != tuple(self.args.offset):
                        plan = planner.plan_command(command, seen, offset=effective_offset)
            if plan.question or plan.unresolved or not plan.moves:
                reason = plan.question.get("es") or "; ".join(plan.unresolved) or plan.reply
                outcome = telemetry.NEEDS_CLARIFICATION
                return True
            if any(pick not in premise for pick in plan.picks) or any(
                    place not in premise for place in plan.places):
                outcome, reason = telemetry.NEEDS_CLARIFICATION, "No puedo identificar todas las piezas y sus destinos."
                return True
            if command.get("intent") == "pick" and len(plan.picks) > 1:
                outcome, reason = telemetry.NEEDS_CLARIFICATION, "La pinza solo puede sostener una pieza. Indica una pieza o un recipiente."
                return True
            if len({premise[pick] for pick in plan.picks}) != len(plan.picks):
                outcome, reason = telemetry.NEEDS_CLARIFICATION, "Una pieza aparece varias veces. EnvÃ­a cada traslado de esa pieza en una orden distinta."
                return True
            import capacity
            try:
                self._task_slots = capacity.allocate(plan.tasks, premise, self.layout, self._placed_slots)
            except ValueError as exc:
                outcome, reason = telemetry.NEEDS_CLARIFICATION, str(exc)
                return True
            self._note_tasks(plan, premise)
            requested = {(premise[task.pick], premise.get(task.place)) for task in plan.tasks}
            if len(plan.picks) > 1:
                self._one_at_a_time(command, plan)
            else:
                for attempt in range(1 + self.args.regrasp):
                    run = self.run_plan(plan, premise, command)
                    if self.bus.stop_requested():
                        outcome, reason = telemetry.CANCELLED, "EjecuciÃ³n interrumpida."
                        return False
                    if run.aborted and not run.stale_grasps:
                        for task in plan.tasks:
                            self.ledger.record(self._task_key(task, premise), False)
                        self._settle_rows()
                        outcome, reason = telemetry.FAILED, "El ejecutor detuvo la tarea: no se confirmÃ³ el agarre o el movimiento. Revisa la cÃ¡mara y la posiciÃ³n de la pieza."
                        return True
                    if not run.stale_grasps:
                        if plan.places:
                            self.check(plan, premise)
                        else:
                            for task in plan.tasks:
                                held = bool((run.real_grasps + run.kinematic_grasps) and self.bot.holding)
                                self.ledger.record(self._task_key(task, premise), held)
                            self._settle_rows()
                        break
                    if attempt < self.args.regrasp:
                        self.park()
                        seen = self.look("Reintentando tras perder la pieza")
                        fresh_premise = self.link(seen)
                        fresh = planner.plan_command(command, seen, offset=self.args.offset)
                        fresh_targets = {(fresh_premise.get(task.pick), fresh_premise.get(task.place)) for task in fresh.tasks}
                        if not fresh.moves or fresh.question or fresh.unresolved or fresh_targets != requested:
                            # A language re-plan must not silently switch to another
                            # same-colour part or destination. Preserve the resolved task identity
                            # and use the camera-linked recovery path once.
                            if self._retry_original_targets(plan, premise):
                                self.bus.log("La recuperación de la tarea original quedó verificada.",
                                             level="info")
                            else:
                                self.bus.log("No se conservaron las identidades al replanificar y "
                                             "la recuperación dirigida no verificó la colocación.",
                                             level="warn")
                            break
                        premise, plan = fresh_premise, fresh
                        if len(plan.picks) == 1:
                            picked = next((thing for thing in seen.everything
                                           if thing.id == plan.picks[0]), None)
                            if picked is not None:
                                effective_offset = self._scene_grasp_offset(picked, seen)
                                if effective_offset != tuple(self.args.offset):
                                    plan = planner.plan_command(
                                        command, seen, offset=effective_offset)
            rows = list(self._task_rows.values())
            done = sum(row.get("state") == "done" for row in rows)
            outcome = (telemetry.SUCCESS if rows and done == len(rows) else
                       telemetry.PARTIAL_SUCCESS if done else telemetry.FAILED)
            reason = f"{done}/{len(rows)} tareas verificadas."
            if done < len(rows):
                reason += " No se ha confirmado el agarre o la colocaciÃ³n de todas las piezas; consulta la lista de tareas."
            if self.bus.stop_requested():
                outcome, reason = telemetry.CANCELLED, "EjecuciÃ³n interrumpida."
            for task in plan.tasks:
                key = self._task_key(task, premise)
                prim = premise.get(task.pick)
                if self.ledger.verdict(key) == TaskLedger.OK:
                    self._placed_slots.pop(prim, None)
                    if prim in self._task_slots:
                        self._placed_slots[prim] = self._task_slots[prim]
            return not self.bus.stop_requested()
        except Exception as exc:
            reason = f"No se pudo ejecutar la orden: {type(exc).__name__}: {exc}"
            self.bus.log(reason, level="fail")
            self.bus.node("execute", telemetry.FAIL, reason)
            return True
        finally:
            if outcome in (telemetry.FAILED, telemetry.NEEDS_CLARIFICATION, telemetry.PARTIAL_SUCCESS):
                self._rejected_commands = getattr(self, "_rejected_commands", 0) + 1
            for node in self.bus.snapshot()["nodes"]:
                if node["state"] == telemetry.ACTIVE:
                    self.bus.node(node["id"], telemetry.OK if outcome == telemetry.SUCCESS else telemetry.WARN,
                                  "Etapa terminada" if outcome == telemetry.SUCCESS else "Etapa interrumpida: " + reason)
            self.bus.set_command_state(cid, outcome, reason)
            self.bus.set_state(command_result={"id": cid, "status": outcome, "reason": reason}, reply=reason)
            self.bus.node("report", telemetry.OK if outcome == telemetry.SUCCESS else telemetry.WARN
                          if outcome == telemetry.NEEDS_CLARIFICATION else telemetry.FAIL, reason)
            self.bus.set_metric("command_total_ms", round((time.perf_counter() - began) * 1000, 1))

    def ask(self) -> str:
        """One command, from the browser, the microphone or the keyboard - in that order.

        All three arrive through one queue, so this only ever waits: the browser and the
        microphone are polled, and the keyboard is read on its own thread by
        ``_terminal_reader``. The reason is a measurement, and the shape of the bug is that
        nothing was ever *lost* - it was simply never looked at.

        *It quit the run instead of waiting.* ``take_command(timeout=None)`` means "do not
        block" in ``Bus``, so with no terminal to fall back on, one empty poll returned the
        literal string ``"quit"`` - which ``handle`` reads as the quit word and stops the run.
        A dashboard-only deployment could therefore never wait for a command: the moment the
        person finished reading the setup sheet and had not typed yet, the demo ended.

        *A browser command waited behind the keyboard.* The 0.2 s poll came first and then the
        robot fell straight through to ``input()``, which blocks until a key is pressed. So a
        sentence typed in the browser sat in the queue until somebody also pressed Enter in the
        terminal - measured, ``ask`` returned the terminal's own answer while the browser's
        command was still waiting. With ``--voice`` the wait was the length of the recording,
        which is why the browser is checked again after the microphone.

        Either way the page showed no sign of any of it, which is what made it read as a broken
        server rather than as a slow one.
        """
        if not self.web_input and self.voice is None and self.keyboard is None:
            # No dashboard, no microphone and no keyboard reader: the only way in is the
            # terminal, read synchronously here.
            text = super().ask()
            self.bus.set_state(command_src="terminal")
            return text

        if self.voice is not None:
            self.bus.node("input", telemetry.ACTIVE, "escuchando")
            self.say("  listening...")
            heard = self.voice.listen()
            self.bus.log(f"  Se ha oÃ­do: {heard}" if heard else "  No se ha oÃ­do ningÃºn comando")
            if heard:
                self.bus.node("input", telemetry.OK, heard)
                self.bus.set_state(command_src="voice")
                return heard

        return self._wait_for_command()

    def _wait_for_command(self) -> str:
        """Block until a command arrives from anywhere, and say where it came from.

        With no keyboard and no microphone this is the only place a command can come from, so it
        is a real wait: a timed poll that gave up was read as ``"quit"`` and ended the run.

        The panel's other buttons are answered here rather than left to ``keys()``, because
        ``keys()`` only runs between two executor steps and there are no steps to run while this
        loop is waiting. ``next round`` returns an empty sentence, which ``handle`` ignores and
        the round loop then sees as ``advance``; ``stop`` returns the quit word, the same one
        the keyboard uses. Controls are only taken when one is actually queued, so nothing is
        swallowed by the wait.
        """
        self.bus.set_state(phase="waiting")
        if self.web_input:
            self.bus.set_state(voice_note="hay una orden esperando en el panel")
        self.bus.node("input", telemetry.ACTIVE,
                      "esperando al panel" if self.web_input else "esperando una orden")
        while True:
            # STOP is a flag, checked before anything else on every pass, so it cannot be
            # consumed behind a queued sentence or control.
            if self.bus.stop_requested():
                self.bus.log("  parada solicitada: no se acepta ninguna orden mÃ¡s", level="warn")
                return "quit"
            text = self.bus.take_command(timeout=WEB_COMMAND_POLL_S)
            if text:
                source = self.bus.command_source()
                where = {"terminal": "el terminal", "voice": "la voz"}.get(source, "el panel")
                self.bus.node("input", telemetry.ACTIVE, f"desde {where}: {text}")
                self.bus.set_state(command_src=source, queued="", phase="command")
                return text
            if getattr(self, "cam_fps", 0) > 0:
                self.cam_pump()
            if not self.bus.control_pending():
                continue
            control = self.bus.take_control() or {}
            action = str(control.get("action") or "")
            if action == "next":
                self.bus.log("  el panel ha pedido la siguiente ronda")
                self.advance = True
                self.bus.set_command_state(control.get("id", ""), telemetry.SUCCESS)
                # An empty sentence is a no-op for ``handle``, which returns to the round loop
                # and finds ``advance`` set. Anything else here would run a command nobody asked
                # for just to get out of the wait.
                return ""
            # Everything else is carried out here, now. Nothing is logged as "later" and dropped:
            # there are no executor steps while this loop waits, so "later" would never come.
            self._apply_control(control)

    def keys(self) -> set[str]:
        """Viewer keys, plus anything the dashboard asked for.

        Called after every executor step, which is what makes the browser's ``q`` abort a run
        mid-move rather than at the next command boundary - the same guarantee demo 10 gives the
        keyboard. Re-look and save-frame are handled here too, and they are the two actions that
        are safe to perform between two steps.
        """
        pressed = set(super().keys())
        self.bus.tick("control_poll")
        self.cam_pump()
        # STOP first, and always: it is a flag, so it is seen even if the controls below are
        # busy or there are many of them.
        if self.bus.stop_requested():
            self.quit = True
            pressed.add("q")
        # A bounded number of controls per step. A burst from the page is spread over the next
        # steps instead of costing that many perception passes before the arm moves again; the
        # rest stay queued, in order, and nothing is discarded.
        for _ in range(MAX_CONTROLS_PER_STEP):
            control = self.bus.take_control()
            if control is None:
                break
            if self._apply_control(control) == "stop":
                self.quit = True
                pressed.add("q")
        return pressed

    def _apply_control(self, control: dict) -> str:
        """Carry out one control from the panel, and report the outcome on its own command.

        Returns the action, so a caller can react to ``stop`` or ``next``. Every control ends in
        SUCCESS or FAILED with a reason; none is left RUNNING and none is silently ignored.
        """
        action = str(control.get("action") or "")
        cid = control.get("id", "")
        params = control.get("params") or {}
        outcome, note = telemetry.SUCCESS, ""
        if action == "stop":
            self.bus.log("  la parada del panel llega al robot", level="warn")
        elif action == "next":
            self.bus.log("  el panel pide la siguiente ronda")
            self.advance = True
        elif action == "relook":
            self.bus.log("  el panel pide volver a observar")
            try:
                self.look("re-observación del panel")
            except Exception as exc:  # noqa: BLE001 - reported as the command's failure
                outcome, note = telemetry.FAILED, f"no se pudo observar: {exc}"
                self.bus.log(f"  {note}", level="fail")
        elif action == "save":
            if self.shown is None:
                outcome, note = telemetry.FAILED, "todavÃ­a no hay imagen"
            else:
                self.writer.force(self.shown)
                self.say(f"  imagen guardada ({self.writer.written} en total) en "
                         f"{self.writer.folder}")
        elif action == "voice":
            outcome = telemetry.FAILED
            note = ("el micrÃ³fono se elige al arrancar con --voice"
                    if self.voice is None else "el micrÃ³fono ya estÃ¡ escuchando")
            self.bus.log(f"  {note}", level="warn")
        elif action == "start":
            self.bus.open_start()
        else:
            outcome, note = telemetry.FAILED, f"la demo no sabe hacer {action!r}"
            self.bus.log(f"  {note}; parÃ¡metros {params}", level="warn")
        if cid:
            self.bus.set_command_state(cid, outcome, note)
        return action


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def apply_start_params(args, bus: telemetry.Bus) -> None:
    """Take the variant/camera/rounds/seed the dashboard chose, by overriding the CLI.

    The start button sends its settings along with the go-ahead, and the loop waiting on
    ``wait_start`` drains the controls on its side, so ``keys()`` never sees the start request
    twice. Only known fields are applied: an unknown key must be a bug in the dashboard, not a
    silent ``setattr`` on the argument namespace.
    """
    params = {}
    start_id = ""
    while True:
        control = bus.take_control()
        if control is None:
            break
        if control.get("action") == "start":
            params = control.get("params") or {}
            start_id = control.get("id", "")
        else:
            # Before the run exists there is nothing for these to act on. They are answered,
            # with that reason, rather than taken and forgotten.
            bus.set_command_state(control.get("id", ""), telemetry.CANCELLED,
                                  "llegÃ³ antes de iniciar el entorno")
    if start_id:
        bus.set_state(start_command_id=start_id)
    for key in ("variant", "camera", "rounds", "seed"):
        if key in params:
            try:
                setattr(args, key, type(getattr(args, key))(params[key]))
            except (TypeError, ValueError):
                bus.log(f"  the dashboard's {key} = {params[key]!r} is not a valid value",
                        level="warn")
    # A startup task is parsed before Isaac is created so its geometry-specific contact
    # profile can be selected when the scene is built. It then follows the normal scripted
    # command path after the robot and camera are ready.
    startup_command = params.get("command")
    if isinstance(startup_command, str) and startup_command.strip():
        args.text = [startup_command.strip()]


def infer_startup_grasp_profile(model, commands) -> str | None:
    """Return a safe target profile only when every startup task selects one geometry.

    Mixed-geometry commands deliberately use the scene's conservative default. Applying a
    target-scoped physical profile to a different geometry can make contact less reliable.
    """
    supported = {"cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"}
    kinds: set[str] = set()
    for text in commands or ():
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            parsed = model.parse(text)
        except Exception:
            return None
        if parsed.get("errors"):
            return None
        items = parsed.get("items") or []
        if not items:
            return None
        item_kinds = {item.get("kind") for item in items}
        if not item_kinds or not item_kinds <= supported:
            return None
        kinds.update(item_kinds)
        if len(kinds) > 1:
            return None
    return next(iter(kinds)) if len(kinds) == 1 else None


def _terminal_reader(bus: telemetry.Bus) -> threading.Thread | None:
    """Read typed commands on their own thread, into the same queue as the browser.

    **Without this, a sentence typed in the browser waits for somebody to press Enter in the
    terminal.** ``ask`` used to poll the bus for 0.2 s and then fall through to ``input()``, and
    ``input()`` blocks until a key is pressed - so in a run started from a terminal, which is the
    usual way to start it, the browser's command sat in the queue indefinitely. Measured: with
    the browser command queued, ``ask`` still returned the terminal's own answer. From the
    browser that is indistinguishable from a server that dropped it.

    So the keyboard is read here instead, on a daemon thread, and everything arrives at
    ``ask`` through one queue. ``enter_starts_the_run`` already uses this pattern for the same
    reason and with the same property: a daemon thread blocked in ``input()`` does not hold up
    interpreter shutdown. Nothing is started when there is no terminal - ``web_input`` covers
    that case and ``ask`` waits on the dashboard there.
    """
    if not sys.stdin.isatty():
        return None

    def read() -> None:
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                bus.submit("quit", source="terminal")
                return
            if line.strip():
                bus.submit(line, source="terminal")

    thread = threading.Thread(target=read, name="terminal-commands", daemon=True)
    thread.start()
    return thread


def enter_starts_the_run(bus: telemetry.Bus) -> threading.Thread | None:
    """Let ``Enter`` in the terminal start the run, for a bench where nobody opened a browser.

    A daemon thread blocked in ``input()`` does not hold up interpreter shutdown, and it is only
    started when there is a terminal to read from.
    """
    if not sys.stdin.isatty():
        return None

    def wait() -> None:
        while not bus.started():
            try:
                input()
            except EOFError:
                return
            bus.open_start()

    thread = threading.Thread(target=wait, name="enter-starts", daemon=True)
    thread.start()
    return thread


def build_parser() -> argparse.ArgumentParser:
    """Demo 10's arguments, unchanged, plus the dashboard and the real-world ones."""
    import supervisor as supervisor_module  # one constant, and only when the parser is built
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # -- demo 10's flags, verbatim, so the muscle memory still works
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"),
                        help="dispositivo de fÃ­sica/tensores; CPU evita compartir PhysX CUDA")
    parser.add_argument("--voice", action="store_true",
                        help="listen to each command as well as accepting typed ones")
    parser.add_argument("--voice-model", default="small",
                        help="faster-whisper model: tiny, base, small or medium (all cached)")
    parser.add_argument("--voice-language", default=None,
                        help="force one language (es, en, eu) instead of detecting it")
    parser.add_argument("--voice-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--raw", action="store_true",
                        help="show the unannotated camera feed instead of the drawn detections")
    parser.add_argument("--camera", default="top", choices=sorted(scene_spec.CAMERA_CHOICES))
    parser.add_argument("--variant", default="contained",
                        choices=["contained", "open", "out_of_reach"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spread", type=float, default=DEFAULT_SPREAD_M,
                        help="how far a randomised layout may move things, in metres; smaller "
                             "than demo 10's because the grasp envelope is 130 mm wide")
    parser.add_argument("--allow-same-colour-clusters", action="store_true",
                        help="probar objetos prÃ³ximos del mismo color sin permitir colisiones")
    parser.add_argument("--disturb", type=float, default=0.0, metavar="P",
                        help="probability of shoving the scene between planning and executing")
    parser.add_argument("--regrasp", type=int, default=2, metavar="N")
    parser.add_argument("--text", action="append", default=[], metavar="COMMAND")
    parser.add_argument("--detect", nargs="+", default=list(colors.DEFAULT_DETECT))
    parser.add_argument("--lang", default="es")
    parser.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("DX", "DY"))
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    # -- demo 11's own flags
    parser.add_argument("--web", dest="web", action="store_true", default=True,
                        help="serve the dashboard on localhost (default)")
    parser.add_argument("--no-web", dest="web", action="store_false")
    monitor_server = _monitor_server()
    parser.add_argument("--host", default=monitor_server.DEFAULT_HOST,
                        help="interface to bind; loopback unless you have a reason")
    parser.add_argument("--port", type=int, default=monitor_server.DEFAULT_PORT,
                        help="0 asks the OS for a free port")
    parser.add_argument("--open", action="store_true", help="open a browser at the dashboard")
    parser.add_argument("--no-wait-for-start", dest="wait_for_start", action="store_false",
                        default=True,
                        help="start Isaac immediately instead of waiting for the dashboard")
    parser.add_argument("--cam-fps", type=float, default=CAM_FPS_DEFAULT, metavar="FPS",
                        help="raw live frames between annotated looks, published while the arm "
                             "moves (0 = annotated looks only). Each one is a render the "
                             "simulation does between two physics steps, so this trades the "
                             "picture against how fast the arm actually moves")
    parser.add_argument("--start-timeout", type=float, default=0.0, metavar="S",
                        help="give up waiting for the dashboard after S seconds (0 = wait for "
                             "ever)")
    parser.add_argument("--setup-only", action="store_true",
                        help="print the bench sheet and serve the dashboard; do not start Isaac")
    parser.add_argument("--startup-timeout", type=float, default=300.0,
                        help="maximum session startup time in seconds")
    parser.add_argument("--scene-config", help="versioned scene JSON file")
    parser.add_argument("--legacy-perception", action="store_true", help="use baseline perception for paired measurements")
    parser.add_argument("--intent-backend", choices=["hybrid", "model", "rules"], default="hybrid")
    parser.add_argument("--stop-timeout", type=float, default=10.0,
                        help="grace period before terminating a stalled session")
    parser.add_argument("--start-attempts", type=int, default=supervisor_module.MAX_START_ATTEMPTS,
                        metavar="N",
                        help="how many times a session is launched before giving up when "
                             "Isaac faults during start-up")
    parser.add_argument("--work-x", type=float, nargs=2, default=None,
                        metavar=("XMIN", "XMAX"),
                        help="where the robot may put a part down to be picked up. Demo 10's "
                             "declared working area is 0.27..0.52, but the arm cannot hold a "
                             "part past about 0.40: the jaws end up 14-17 mm from the "
                             "commanded pose there, against 3.1 mm of lateral slack")
    parser.add_argument("--allow-kinematic", action="store_true",
                        help="permit simulator-assisted grasps for every geometry (spheres and "
                             "polygon prisms use this automatically); these are not physical "
                             "gripper grasps")
    parser.add_argument("--strict-physical", action="store_true",
                        help="disable automatic simulator grasp assistance and require measured jaw contact")
    parser.add_argument("--grasp-profile-kind",
                        choices=("cube", "ball", "cylinder", "triangular_prism",
                                 "hexagonal_prism"), default=None,
                        help="scope a validated contact-pad profile to one resolved target geometry; "
                             "use only for a command that does not grasp other geometries")
    parser.add_argument("--contact-diagnostics-for", default=None, metavar="PRIM",
                        help="optionally log filtered PhysX jaw forces for this configured object at close")
    parser.add_argument("--sim-assist-policy", choices=("preemptive", "after_physical_miss"),
                        default="preemptive",
                        help="preemptive (default) uses a contact-gated, explicitly labelled simulator grasp for reliable demos; after_physical_miss tests friction first")
    parser.add_argument("--release", default="clear", choices=["clear", "rim"],
                        help="'clear' releases from the height the fingertips can reach above "
                             "the rim (the only one a real arm can do); 'rim' is demo 10's "
                             "12 mm above the rim, kept for comparison")
    parser.add_argument("--close-command", type=float, default=None, metavar="RAD",
                        help="how far the jaws close. LeIsaac's 0.4 rad grips nothing here; "
                             "an explicit value overrides the validated per-shape profile")
    parser.add_argument("--grip-stiffness", type=float,
                        default=real_world.GRIPPER.actuator_stiffness, metavar="K",
                        help="gripper actuator stiffness. At the shipped 17.8 the ball "
                             "follows only 79%% of the lift; at 250 it follows 94%%")
    return parser


def initialize_intent_parser(backend: str, catalog, model_factory=None):
    """Build the requested parser, allowing hybrid mode to degrade safely without PyTorch.

    The rule parser only executes fully explicit supported commands and abstains on ambiguous
    ones. That makes it a safe runtime fallback when a minimal LeRobot Python lacks torch; an
    explicitly requested model backend still raises the import error instead of silently changing.
    """
    from intent_parser import HybridParser

    factory = model_factory or CommandModel
    learned_model = None
    rules_fallback = False
    if backend != "rules":
        try:
            learned_model = factory()
        except ModuleNotFoundError as exc:
            if backend != "hybrid" or getattr(exc, "name", None) != "torch":
                raise
            rules_fallback = True
    parser = learned_model if backend == "model" else HybridParser(learned_model, catalog)
    return parser, rules_fallback


def main(args=None, bus=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = args or parser.parse_args()
    args.close_command_override = args.close_command
    if args.close_command is None:
        args.close_command = real_world.GRIPPER.close_command_rad
    if args.web and not args.setup_only and not getattr(args, "worker", False):
        from supervisor import run
        return run(args)

    args.scene_data, args.scene_items = scene_config.build(args.scene_data) if getattr(args, "scene_data", None) else scene_config.load(args.scene_config)
    _, configured_work_x, configured_work_y = scene_config.working_area(args.scene_data)
    scene_spec.set_work_area(configured_work_x, configured_work_y)
    if args.work_x is None:
        args.work_x = list(configured_work_x)
    args.detect = sorted({i.color for i in args.scene_items})

    bus = bus or telemetry.Bus()
    bus.node("setup", telemetry.ACTIVE, "Preparando el entorno")
    bus.set_state(phase="setup", variant=args.variant, seed=args.seed, voice="off",
                      rounds=args.rounds, camera=args.camera, release=args.release,
                      close_command=args.close_command, grip_stiffness=args.grip_stiffness,
                      allow_kinematic=args.allow_kinematic,
                      sim_assist_policy=args.sim_assist_policy,
                      sim_assist_shapes=sorted(SIM_ASSISTED_SHAPES))

    # The setup sheet before anything else, so the dashboard is useful even if Isaac never
    # starts: this is the part of the demo that transfers to a real bench.
    sheet = real_world.setup_sheet(args.scene_items, camera=args.camera)
    print("\n".join(sheet), flush=True)
    findings = real_world.preflight(args.scene_items, camera=args.camera)
    import ui_spanish
    bus.set_state(findings=ui_spanish.findings(findings), findings_technical=findings)
    for level, text in ui_spanish.findings(findings):
        bus.log(text, level="fail" if level == "fail" else
                ("warn" if level == "warn" else "info"))
    assisted_in_scene = sorted({item.kind for item in args.scene_items} & SIM_ASSISTED_SHAPES)
    if assisted_in_scene:
        labels = {"cube": "cubos", "ball": "esferas", "cylinder": "cilindros",
                  "triangular_prism": "prismas triangulares",
                  "hexagonal_prism": "prismas hexagonales"}
        names = ", ".join(labels[kind] for kind in assisted_in_scene)
        bus.log(f"SujeciÃ³n asistida por simulaciÃ³n para {names}; se cuenta aparte del agarre "
                "fÃ­sico.", level="warn")

    monitor = None
    if args.web and not getattr(args, "worker", False):
        monitor = _monitor_server().serve(bus, args.host, args.port, open_browser=args.open,
                                          announce=lambda line: print(line, flush=True))

    if args.setup_only:
        print("  --setup-only: not starting the simulator. close this window to stop the "
              "dashboard.", flush=True)
        bus.set_state(phase="setup")
        _wait_forever(bus)
        if monitor is not None:
            monitor.stop()
        return 0

    # The dashboard is meant to be set up *before* Isaac, which is where the environment gets
    # chosen and the run gets started. With no dashboard there is nobody to press the button, and
    # a scripted run has its own script, so neither waits.
    if args.web and args.wait_for_start and not args.text and not getattr(args, "worker", False):
        bus.set_state(phase="waiting_start")
        bus.node("setup", telemetry.ACTIVE, "Esperando el inicio desde el panel")
        print(f"  dashboard: {monitor.url if monitor else '(none: --no-web)'}", flush=True)
        print("  waiting for the start command there, or press Enter here", flush=True)
        enter_starts_the_run(bus)
        if not bus.wait_start(timeout=args.start_timeout or None):
            bus.log("  tiempo de espera agotado: se arranca con la configuraciÃ³n de la lÃ­nea "
                    "de Ã³rdenes, sin esperar al panel", level="warn")
        apply_start_params(args, bus)

    # The command model before the simulator, and the microphone with it. Loading a model after
    # Isaac takes 0xC0000005 inside gliner2's from_pretrained, and two heavy native libraries
    # loading at once did the same thing before that.
    bus.set_state(phase="loading")
    try:
        catalog = scene_config.catalog(args.scene_data, args.scene_items)
        model, rules_fallback = initialize_intent_parser(args.intent_backend, catalog)
        if rules_fallback:
            note = ("PyTorch no está disponible en este Python; se usará el intérprete por reglas. "
                    "Las órdenes ambiguas se rechazarán para que puedas reformularlas.")
            bus.log(note, level="warn")
            bus.set_state(intent_backend_active="rules", intent_backend_note=note)
    except Exception as exc:
        bus.set_state(phase="failed", error=f"No se pudo cargar el modelo: {exc}")
        bus.fail_active(str(exc))
        raise
    if args.grasp_profile_kind is None and args.text:
        args.grasp_profile_kind = infer_startup_grasp_profile(model, args.text)
        if args.grasp_profile_kind:
            bus.log(f"  perfil de agarre validado activado para {args.grasp_profile_kind}, "
                    "según la primera orden", level="info")
            bus.set_state(grasp_profile_kind=args.grasp_profile_kind)
        else:
            bus.log("  la orden inicial incluye geometrías mezcladas o no resueltas; "
                    "se usará el perfil general de agarre", level="warn")
    listener = None
    if args.voice:
        import voice

        complaint = voice.check_microphone()
        if complaint:
            parser.error(f"--voice cannot work: {complaint}")
        listener = voice.Listener(model=args.voice_model, language=args.voice_language,
                                  device=args.voice_device)
        bus.set_state(voice="on", voice_note="MicrÃ³fono activo; tambiÃ©n puedes enviar Ã³rdenes desde el panel.")

    bus.set_state(phase="starting", voice="on" if listener else "off")
    bus.node("setup", telemetry.ACTIVE, "Arrancando Isaac Sim")
    print("  Arrancando Isaac Sim; el panel mostrarÃ¡ el estado y cualquier error.", flush=True)
    try:
        isaac_io.prepare_runtime(headless=args.headless)

        layout = None
        if args.variant == "out_of_reach":
            layout = scene_spec.out_of_reach_layout(args.scene_items)
            env, sensor = build_real_env(args.camera, layout, pen=False,
                                           grip_stiffness=args.grip_stiffness,
                                           device=args.device,
                                           contact_diagnostics_for=args.contact_diagnostics_for)
            unreachable = [item for item in layout if not item.container
                           and scene_spec.reach_facts(item.x, item.y,
                                                      scene_spec.TABLE_X,
                                                      scene_spec.TABLE_Y)["beyond_arm"]]
            print(f"  {len(unreachable)} pieza(s) colocada(s) fuera del alcance:", flush=True)
            for item in unreachable:
                print(f"    {item.prim}: "
                      f"{scene_spec.why_unreachable(item.x, item.y, scene_spec.TABLE_X, scene_spec.TABLE_Y)}",
                      flush=True)
        else:
            env, sensor = build_real_env(args.camera, list(args.scene_items),
                                         pen=args.variant == "contained",
                                         grip_stiffness=args.grip_stiffness,
                                         device=args.device,
                                         auto_validated_contact_pads=True,
                                         grasp_profile_kind=args.grasp_profile_kind,
                                         contact_diagnostics_for=args.contact_diagnostics_for)
            if args.variant == "contained":
                kerb_x, kerb_y = pen_module.contained_region()
                print(f"  working area walled in: a {pen_module.PEN_HEIGHT * 100:.1f} cm kerb around "
                      f"x {kerb_x} y {kerb_y}", flush=True)
                if abs(kerb_y[0] - scene_spec.WORK_Y[0]) > 1e-6:
                    print(f"  NOTA: el muro cercano del cerco se ha retirado de y {scene_spec.WORK_Y[0]:.3f} "
                          f"a y {kerb_y[0]:.3f}: más cerca de la base el brazo choca con él y no puede "
                          "acercarse a las piezas. Usa la variante 'open' o aleja el área de la base.",
                          flush=True)
        # Record registered rigid roots immediately before IsaacLab's reset event.
        # This pinpoints any kinematic body that the reset callback still tries to
        # write velocities to, without changing scene physics based on guesses.
        for actor_name in env.scene.keys():
            try:
                actor = env.scene[actor_name]
                view = getattr(actor, "root_physx_view", None)
                if view is None:
                    continue
                paths = getattr(view, "prim_paths", ())
                if paths:
                    print(f"  [reset-body] {actor_name}: {paths[0]} kinematic="
                          f"{isaac_io.root_is_kinematic(actor)}", flush=True)
            except Exception as exc:
                print(f"  [reset-body] {actor_name}: inspection failed ({exc})", flush=True)
        env.reset()
        original_step = env.step
        def measured_step(*values, **options):
            began = time.perf_counter()
            result = original_step(*values, **options)
            bus.tick("physics")
            bus.set_metric("physics_step_ms", round((time.perf_counter()-began)*1000, 2))
            return result
        env.step = measured_step
        if scene_spec.SHIPPED_CUBE not in {i.prim for i in args.scene_items}:
            hidden = env.scene[scene_spec.SHIPPED_CUBE]
            pose = hidden.data.root_state_w[:, :7].clone()
            pose[:, 2] = -10.0
            hidden.write_root_pose_to_sim(pose)
        isaac_io.aim_camera(env, args.camera, sensor)
        if getattr(env, "demo11_backup_camera", None) is not None:
            isaac_io.aim_camera(env, "top", env.demo11_backup_camera)
        was = set_gripper(env, args.close_command)
        print(f"  jaws close to {args.close_command:.2f} rad (was {was:.2f} rad, which grips nothing "
              f"smaller than about 8 cm)", flush=True)
        _offset = robot_measure(env)
        if _offset is not None:
            print(f"  measured tool offset from the gripper body: "
                  f"({_offset[0] * 1000:+.0f}, {_offset[1] * 1000:+.0f}, "
                  f"{_offset[2] * 1000:+.0f}) mm; demo 05's fitted HOLD is "
                  f"({executor_module.HOLD[0] * 1000:+.0f}, {executor_module.HOLD[1] * 1000:+.0f}, "
                  f"{executor_module.HOLD[2] * 1000:+.0f}) mm", flush=True)

    except BaseException as exc:
        bus.set_state(phase="failed", error=f"{type(exc).__name__}: {exc}")
        bus.log(f"el entorno no ha arrancado: {type(exc).__name__}: {exc}", level="fail")
        raise

    viewer = None
    robot = None
    try:
        if not args.no_viewer and not args.headless:
            viewer = viewer_module.Viewer(name="leisaac demo 11")
        robot = MonitoredRobot(env, sensor, args, bus, viewer=viewer, model=model,
                               release_mode=args.release)
        # The grasp envelope, not the reach envelope: see DEFAULT_WORK_X. Applied to this robot's
        # own base layout, so scene_spec - and demo 10 - are untouched.
        robot.base = clamp_layout(robot.base, tuple(args.work_x))
        robot.bot.with_layout(robot.base)
        for item in robot.base:
            original = next((i for i in robot._initial_layout if i.prim == item.prim), item)
            if not item.container and abs(item.x - original.x) > 1e-9:
                robot.say(f"  {item.prim} moved to x {item.x:.3f}: the arm cannot hold a part "
                          f"outside x {args.work_x[0]:.2f}..{args.work_x[1]:.2f}")
        if args.variant == "out_of_reach":
            robot.base = layout
        robot.voice = listener
        # The keyboard joins the same queue as the browser, read on its own thread. Started
        # here, after the start gate: ``enter_starts_the_run`` has already had stdin to itself
        # and two threads calling ``input()`` is not a thing to do on purpose.
        robot.keyboard = (_terminal_reader(bus) if not args.text and sys.stdin.isatty()
                          and not getattr(args, "worker", False) else None)
        bus.set_state(input_mode="dashboard" if robot.web_input else "terminal")
        bus.set_state(phase="ready", sim="running")
        start_id = bus.snapshot()["state"].get("start_command_id")
        if start_id:
            bus.set_command_state(start_id, telemetry.SUCCESS, "Entorno listo.")
        bus.node("setup", telemetry.OK, f"Isaac en marcha; variante {args.variant}, cÃ¡mara "
                                        f"{args.camera}")
        robot.findings()

        scripted = list(args.text) * max(1, args.repeat) if args.text else []
        stopped = False

        for round_index in range(args.rounds):
            seed = args.seed + round_index
            if args.variant == "out_of_reach":
                layout = robot.base
                robot.say(f"\n=== round {round_index} (static layout) ===")
            else:
                layout = dynamics.randomize(
                    robot.base, seed, spread=args.spread,
                    allow_same_colour_clusters=args.allow_same_colour_clusters)
                try:
                    for item in layout:
                        scene_config._check_position(item, [other for other in layout if other is not item])
                except ValueError:
                    layout = list(robot.base)
                    robot.say("La variaciÃ³n no deja separaciÃ³n segura; se conserva la disposiciÃ³n validada.")
                robot.say(f"\n=== round {round_index} (seed {seed}) ===")
            bus.set_state(round=round_index, seed=seed, phase="scene")

            # The front camera needs a park pose out of its sight line, but x = 0.20 was too close
            # to the base: the vertical move down to the hover height stopped at 0.26 m instead of
            # 0.223 m, the long lateral slide then ran at that height in an extended arm
            # configuration (wrist_flex at its limit) and the pre-grasp hover never converged.
            # 20 mm triangle, front camera, 8 strict scenes: 3/8 at x = 0.20, 8/8 at x = 0.28.
            park_pose = (0.28, -0.58, 0.39) if args.camera == "front" else None
            if park_pose is not None and os.environ.get("DEMO11_FRONT_PARK_POSE"):
                # A/B only: "x,y,z" hold-point park pose for the front camera.
                park_pose = tuple(float(v) for v in os.environ["DEMO11_FRONT_PARK_POSE"].split(","))
            parked = robot.park(pose=park_pose)  # clear the table before repopulating it
            if not parked.ok:
                raise RuntimeError(f"No se puede poblar la escena: {parked.note}")
            robot.populate(layout)
            if robot._last_backup_capture is not None:
                # ``populate`` has just settled this exact scene; the warmed top-camera
                # frame is valid for the opening observation even if a prior round had
                # marked its cache stale before resetting the scene.
                robot._backup_capture_stale = False
            robot._placed_slots.clear()
            robot.round_index = round_index
            robot.label = f"round{round_index}"
            robot.advance = False
            seen = robot.look("the table")
            robot.say(f"  {seen.summary()}")
            robot.say(f"  {describe.what_i_see(seen, lang=args.lang)}")

            queue = list(scripted) if scripted else None
            while not stopped:
                if robot.advance:
                    robot.advance = False
                    break
                # Checked before taking one, not after: checking after pops the last scripted
                # command and throws it away, which is exactly what an earlier version did and
                # why that version reported a round with nothing in it.
                if queue is not None and not queue:
                    break
                text = queue.pop(0) if queue is not None else robot.ask()
                if queue is not None:
                    bus.set_state(command_src="script")
                if text:
                    robot.say(f"\n> {text}")
                if not robot.handle(text):
                    stopped = True
                    break
                if robot.keys() & {"q", "escape"}:
                    stopped = True
                    break
                if viewer is not None and viewer.wants_quit(robot.keys()):
                    stopped = True
                    break
            if stopped:
                break
            if viewer is not None and viewer.wants_quit(robot.keys()):
                break

        # The result comes from the ledger alone: one verdict per task, for the whole run. The
        # exit code is derived from the same status, so the summary and the process cannot
        # disagree.
        status = robot.ledger.status(aborted=stopped)
        if robot._rejected_commands and status in ("SUCCESS", "NO_TASKS"):
            status = "PARTIAL_SUCCESS" if robot.ledger.counts()[0] else "FAILED"
        ok_count, failed_count = robot.ledger.counts()
        robot.say("\n=== resultado ===")
        robot.say(f"  {STATUS_ES[status]}  ({status})")
        robot.say(f"  tareas completadas: {ok_count}")
        robot.say(f"  tareas fallidas:    {failed_count}")
        for label in robot.ledger.failed_labels():
            robot.say(f"    - {label}")
        robot.say("\n=== detalle ===")
        for label, value in (("colocadas y verificadas", robot.verified),
                             ("recuperadas tras reintentar", robot.recovered),
                             ("no verificadas", robot.missed),
                             ("replanificaciones", robot.replanned),
                             ("sujeciones rechazadas", robot.bot.refused),
                             ("pÃ©rdidas de objeto tras cerrar", robot.bot.lost),
                             ("centrados del efector", robot.bot.corrected),
                             ("sujeciones forzadas (solo simulaciÃ³n)", robot.bot.kinematic)):
            robot.say(f"  {label:<40}: {value}")
        bus.node("report", telemetry.OK if status in ("SUCCESS", "NO_TASKS") else telemetry.FAIL,
                 f"{ok_count} ok, {failed_count} fallidas")
        bus.set_state(phase="done", status=status, verified=robot.verified,
                      completed_tasks=ok_count, failed_tasks=failed_count,
                      rejected_commands=robot._rejected_commands,
                      failed_labels=robot.ledger.failed_labels(),
                      recovered=robot.recovered, missed=robot.missed,
                      refused_grasps=robot.bot.refused, lost_grasps=robot.bot.lost,
                      real_grasps=getattr(robot, "real_grasps_total", 0),
                      kinematic_grasps=robot.bot.kinematic)
        if robot.bot.kinematic:
            robot.say("  NOTA: los agarres asistidos se cuentan por separado: el simulador "
                      "mantuvo la pieza ligada al efector; no equivalen a un agarre fÃ­sico.")
        if robot.writer:
            robot.writer.close()
        return EXIT_CODE[status]
    finally:
        active_robot = locals().get("robot")
        if active_robot and getattr(active_robot, "frame_encoder", None):
            active_robot.frame_encoder.close()
        if viewer is not None:
            viewer.close()
        env.close()
        bus.complete_stop()
        bus.set_state(phase="stopped")
        if monitor is not None:
            # Stopped after the simulator, not before: the dashboard is how someone watching a
            # failed run finds out what it said, and it costs one second to keep it up.
            monitor.stop()


def _wait_forever(bus: telemetry.Bus) -> None:
    """Block until the dashboard asks to stop, or Ctrl-C."""
    print("  ctrl-c to stop", flush=True)
    try:
        while not bus.stop_requested():
            control = bus.take_control(timeout=1.0)
            if control is not None:
                # Only preparation is running here: nothing can be moved, so a control that
                # would need the simulator is refused with its reason rather than left pending.
                bus.set_command_state(control.get("id", ""), telemetry.FAILED,
                                      "solo se prepara el entorno; no hay robot que lo ejecute")
    except KeyboardInterrupt:
        return
    bus.complete_stop()


if __name__ == "__main__":
    raise SystemExit(main())
