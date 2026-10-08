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
    else:
        spawn = sim_utils.CuboidCfg(size=item.size, **common)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/" + item.prim,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(item.x, item.y, item.z), rot=(1.0, 0.0, 0.0, 0.0)),
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


def build_real_env(camera: str, items: list[scene_spec.Item], pen: bool = False,
                   grip_stiffness: float = real_world.GRIPPER.actuator_stiffness):
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

    cfg = parse_env_cfg(isaac_io.TASK, device="cuda:0", num_envs=1)
    # The action manager is only installed by the teleop device, and without it the arm_action
    # term is missing and the environment cannot be built.
    cfg.use_teleop_device("so101_state_machine")
    cfg.seed = 0
    cfg.recorders = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None  # otherwise lifting an object ends the episode

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

    if pen:
        # The containment kerb. Same body shape as a tray, four walls and no floor; see pen.py.
        # Spawned last so it exists before anything is placed inside it.
        setattr(cfg.scene, pen_module.PEN_PRIM, pen_module.make_pen_asset_cfg())

    sensor = isaac_io.ensure_camera(cfg, camera)
    return gym.make(isaac_io.TASK, cfg=cfg).unwrapped, sensor


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

# How far off centre a part may be and still end up between the pads: half the aperture less half
# the part. This is a geometric fact about the gripper, not a tuned number, and it is the budget
# every grasp has to fit inside.
LATERAL_SLACK_M = (real_world.GRIPPER.aperture_m - 0.030) / 2.0

# The tool-centre servo in ``StrictExecutor.track`` stops correcting when it stops making progress,
# because an arm at a joint limit will obey every command and go nowhere. "The error did not
# shrink by this much" over that many corrections means the pose is unreachable, which is a
# different fact with a different fix, and the run says which one it was.
STALL_IMPROVEMENT = 0.001
STALL_WINDOW = 6

# The largest move of a sighted object, between two looks, that is still the same object. The
# closest pair of same-coloured objects in the scene is 64 mm apart centre to centre (demo 10,
# LINK_TOLERANCE_M's comment), so anything at or beyond that could be the neighbour. A knock
# larger than this is reported as a lost target and re-planned from a fresh look, rather than
# followed onto whatever blob is nearest.
SAME_OBJECT_MAX_SHIFT_M = 0.060

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
LAST_CORRECTION_Z = 0.20

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

    ``allow_kinematic`` restores demo 10's behaviour for A/B runs and defaults off. When it is on,
    the refusal is counted but not acted on, and the dashboard says the run is faking grasps so
    a result obtained that way is never mistaken for a result a real arm could produce.
    """
    def __init__(self, *args, allow_kinematic: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_kinematic = bool(allow_kinematic)
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
        # Geometry read at the end of the last ``close``, which is the last instant at which the
        # part and the jaws are in the same place.
        self.at_close = None
        # Declared part sizes by prim, for the "will it fit between the jaws" test. Populated by
        # ``StrictExecutor.with_layout``.
        self.items_by_prim: dict = {}

    def grab(self, prim: str) -> None:
        if self.allow_kinematic:
            self.kinematic += 1
            super().grab(prim)
            return
        raise GraspRefused(prim)

    def with_layout(self, items):
        """Teach the executor each part's declared size, for the grip-width test."""
        self.items_by_prim = {item.prim: item.size[0] for item in items}
        return self

    def do(self, *args, **kwargs):
        # The close step gets its own duration, for the reason in CLOSE_SETTLE_STEPS. Only when the
        # caller did not ask for a specific count: the retry path drives its own lift with an
        # explicit count and must not be second-guessed here.
        if len(args) > 1 and args[0] == "close" and len(args) < 6 and "steps" not in kwargs:
            kwargs["steps"] = CLOSE_SETTLE_STEPS
        try:
            result = super().do(*args, **kwargs)
            if self.frame_hook is not None:
                self.frame_hook()
            if result.grasp == "lost":
                # The part left the jaws' reach while the gripper rose. It is not held, so it
                # must not be carried, lowered or released as if it were. Reported as a failed
                # grasp: executor.run stops on "stale", and the caller re-plans or fails the task.
                self.lost += 1
                prim = kwargs.get("prim")
                if prim:
                    self.failures[prim] = self.failures.get(prim, 0) + 1
                return executor_module.StepResult(
                    action=result.action, ok=False, grasp="stale",
                    note=f"la pieza ya no está en las mordazas: {result.note}")
            if args and args[0] == "close" and result.ok:
                # Measured here because this is the last moment before the lift: afterwards the arm
                # has risen 75 mm and any geometry read later describes where the gripper *is*,
                # not where the jaws shut.
                self.at_close = self.part_geometry(kwargs.get("prim"))
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
        size = self.items_by_prim.get(prim)
        edge = float(size) if size else 0.0
        # The jaws close across the part's horizontal footprint; on its edge a square part presents
        # its diagonal, so the width grows by sqrt(2) and no force can recover it.
        grip_width = edge * (math.sqrt(2.0) if tilt > 45.0 else 1.0)
        geometry = {
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
        parts = [f"part centre z {geometry['z']:.4f}", f"tilted {geometry['tilt_deg']:.0f} deg",
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
        if retarget is None:
            return super().run(plan, resolve=resolve, keys=keys, fps=fps, observe=observe)

        def guarded(step, prim):
            fresh = retarget(step, prim)
            if not fresh:
                raise _TargetLost(prim)
            return fresh

        try:
            return super().run(plan, resolve=resolve, keys=keys, fps=fps, retarget=guarded,
                               observe=observe)
        except _TargetLost as lost:
            out = executor_module.RunResult()
            out.stale_grasps = 1
            out.stale_prims = [lost.prim] if lost.prim else []
            out.aborted_at = "re-plan: no se encuentra el objeto"
            self.log(f"  re-plan: no se encuentra el objeto {lost.prim}; se detiene el movimiento")
            return out

    # -- putting the part actually between the pads ----------------------------

    def aperture_centre(self) -> tuple[float, float] | None:
        """Where the gap between the pads actually is, in world x-y.

        Measured, not derived. ``executor.HOLD`` says a held part's centre sits 7 mm forward and
        92 mm below the gripper body, and every earlier version of this correction used that -
        rotated into the world by the wrist's own orientation. That is an assumption about where
        the fingers are, and it is wrong by enough to matter: it reported the jaws as 13-18 mm from
        the part at the far end of the table, which is most of what a 36.2 mm aperture has to give.

        The articulation has exactly one body named ``gripper`` and one named ``jaw``, and
        ``tools/probe_gripper.py`` measures the aperture as the distance between their origins. So
        their midpoint *is* the place a part has to be, whatever the wrist is doing - which is the
        one number that can be compared with the camera's estimate of the part without assuming
        anything.
        """
        names = list(self.robot.body_names)
        fingers = [name for name in ("gripper", "jaw") if name in names]
        if len(fingers) < 2:
            return None
        ids = [self.robot.find_bodies(name)[0][0] for name in fingers]
        here = [self.robot.data.body_pos_w[0, index].detach().cpu().numpy() for index in ids]
        return tuple(float((here[0][axis] + here[1][axis]) / 2.0) for axis in range(3))

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
        return (centre[0] - float(body[0]), centre[1] - float(body[1]), centre[2] - float(body[2]))

    def track(self, observe, target_z: float, steps: int, action: str = "above",
              grip: float = executor_module.OPEN,
              every: int = executor_module.TRACK_EVERY,
              aim: tuple[float, float] | None = None):
        """Move to ``target_z`` while servoing the *aperture* onto the part, not the aim onto it.

        ``executor.track`` aims at what the camera reports: it re-reads the part's position and
        points the commanded pose at it. That is right for a part that is where the arm will be,
        and wrong when the arm does not go where it is told - and it is why demo 11's first
        attempt at this fix did nothing. Correcting the aim was undone on the next 20-step tick,
        because the correction changed the aim and the camera immediately pulled the aim back onto
        the part.

        So the correction is applied to the **commanded pose** instead, which is the thing the arm
        actually fails to follow:

            command += (where the part is) - (where the aperture is)

        Every tick that consults the camera. That is a servo on the tool centre point using the
        arm's own feedback, and a fixed offset between command and reality cancels on the next tick
        rather than being fought. Measured at x = 0.29 the residual is about 1 mm and the grasp
        succeeds; at x = 0.44 it was 13-18 mm, and this is what gets rid of it.

        Two guards, because a servo that cannot converge must say so instead of grinding:

        * a shift larger than ``SAME_OBJECT_MAX_SHIFT_M`` still counts as lost - that is the
          part being knocked away, not the arm lagging;
        * if the error has not shrunk by ``STALL_IMPROVEMENT`` over ``STALL_WINDOW`` corrections,
          the arm is at a limit and no amount of commanding will move it, so it stops and says so.
          That is the difference between "the arm is 14 mm out and correcting" and "this pose is
          unreachable", which are different facts and need different fixes.
        """
        import torch

        start = self.current.clone()
        here = aim if aim is not None else (float(start[0][0]), float(start[0][1]))
        # The descent's height uses ``executor.HOLD``, demo 05's *fitted grip point*, and not the
        # measured aperture centre. Measured on this arm the two finger bodies' origins sit at
        # (-9, +12, +10) mm from the gripper body - not 92 mm below it - because they are the finger
        # links, whose origins are near the top of the finger. So the midpoint is the right
        # *lateral* reference for the servo and completely the wrong *height*: driving the descent
        # with it put the body at the table and cost a placement. Height comes from the fitted
        # point; lateral comes from the measurement.
        hold_z = float(self._hold[0][2])
        start_z = float(start[0][2])
        target_body_z = target_z - hold_z
        command = [float(here[0]), float(here[1])]
        corrections = 0
        largest = 0.0
        lost = 0
        worst = 0.0
        # The residuals of the last STALL_WINDOW corrections, for this descent only. A stall is a
        # window of corrections that did not shrink the residual by STALL_IMPROVEMENT across the
        # window. It is not a comparison against the best value of the whole descent, which
        # flagged a steadily converging servo as stuck.
        recent: list[float] = []
        stall_reported = False
        self.correcting = True

        with torch.no_grad():
            for index in range(1, steps + 1):
                fraction = min(1.0, index / (0.8 * steps))
                z = start_z + (target_body_z - start_z) * fraction
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
                if (index % every == 0 or index == steps) and z >= LAST_CORRECTION_Z:
                    seen = observe()
                    if seen is None:
                        if z >= executor_module.TRACK_OBSERVE_ABOVE_Z:
                            lost += 1
                    else:
                        drift = math.dist((float(seen[0]), float(seen[1])), tuple(command))
                        if drift > SAME_OBJECT_MAX_SHIFT_M:
                            lost += 1
                        else:
                            centre = self.aperture_centre()
                            if centre is not None:
                                error = (float(seen[0]) - centre[0], float(seen[1]) - centre[1])
                                size = math.hypot(*error)
                                worst = max(worst, size)
                                if size > 0.0005 and not stall_reported:  # 0.5 mm: arm's own noise
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
                                    else:
                                        # Never correct by more than the slack allows. If the arm
                                        # is at a limit this caps the swing a full gain would give.
                                        step = min(size, LATERAL_SLACK_M * 2.0)
                                        command[0] += error[0] / size * step
                                        command[1] += error[1] / size * step
                                        corrections += 1
                                        largest = max(largest, size)
                                        self.corrected += 1
                if self.frame_hook is not None:
                    self.frame_hook()
                self.current = torch.tensor([[command[0], command[1], z]], device=self.device)
                self.env.step(self._action(self.current, grip))

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
        if lost:
            note += f", {lost} look(s) could not find it"
        # The residual in all three axes, because "the grasp failed" is not a diagnosis and this is
        # the one number that turns it into one. A part is between the pads only if the aperture
        # centre is within LATERAL_SLACK_M of it laterally and within half the part's height
        # vertically.
        centre = self.aperture_centre()
        if centre is not None and action == "down":
            here3 = observe()
            if here3 is not None:
                gap = math.dist(centre[:2], (float(here3[0]), float(here3[1])))
                note += (f", aperture {gap * 1000:.1f} mm from the part"
                         f"{' (between the pads)' if gap <= LATERAL_SLACK_M else ' (outside the slack)'}")
        if stall_reported:
            note += ", centrado detenido: la pose no es alcanzable desde aquí"
        return executor_module.StepResult(action=action, note=note,
                                          ok=lost == 0 and not stall_reported)


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
        self._bus.set_state(classification={
            "intent": command.get("intent"),
            "items": [{"kind": item.get("kind"), "color": item.get("color"),
                       "count": item.get("quantity", item.get("count")), "place": item.get("place")}
                      for item in command.get("items", [])],
            "place": command.get("place"),
            "sort_by": command.get("sort_by"),
            # The model's own score for each field it decided, so the panel can say which one it
            # was unsure of instead of only reporting the weakest of them as a single number.
            "confidences": command.get("confidences") or {},
            "confidence": command.get("confidence"),
            "parser": command.get("parser", "model"),
            "errors": command.get("errors", []),
        })
        self._bus.set_metric("parse_ms", command.get("parse_ms"))
        import ui_spanish
        items = " ".join(describe.phrase(item.get("kind"), item.get("color"), 1, "es", bare=True) for item in command.get("items", []))
        self._bus.node("understand", telemetry.OK,
                       f"{ui_spanish.INTENTS.get(command.get('intent'), 'orden')} [{items.strip() or 'sin piezas'}]")
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
    text = f"{len(run.steps)} pasos, {run.real_grasps} agarre(s)"
    if run.stale_grasps:
        text += ", sin pieza"
    if run.lost_grasps:
        text += ", pieza perdida"
    return text


EXIT_CODE = {"SUCCESS": 0, "NO_TASKS": 0, "PARTIAL_SUCCESS": 2, "FAILED": 2, "ABORTED": 3}


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
        # The command's tasks and which are done, keyed by part name so a re-plan recognises
        # the same task instead of adding a second row for it. Cleared per command in `handle`.
        self._task_rows: dict[str, dict] = {}
        # The verdict per task for the whole run; the final result is derived from it alone.
        self.ledger = TaskLedger()
        super().__init__(env, sensor, args, viewer=viewer, model=model)
        if getattr(args, "scene_items", None):
            self.base = list(args.scene_items)
            self.layout = list(self.base)
            self.tracked_xy = {i.prim: (i.x, i.y) for i in self.base}
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
        self.bot = StrictExecutor(env, log=self.say, allow_kinematic=args.allow_kinematic)
        self.bot.log = self._step_logged
        self.bot.frame_hook = self.cam_pump if self.cam_fps > 0.0 else None
        self.model = ModelSpy(model, bus)
        install_plan_spy()
        if self not in _PLAN_SPIES:
            _PLAN_SPIES.append(self)
        if self not in _PLAN_FILTERS:
            _PLAN_FILTERS.append(self)

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
            self.bus.node("execute", telemetry.ACTIVE, ui_spanish.ACTIONS.get(head[0], "Comprobando ejecución"))

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
             f"{prim} tras {failures.get(prim, 0)} sujeción(es) fallidas")
            for prim in dropped]
        self.say("  demo 11: " + "; ".join(reasons) +
                 "; se sacan del plan de esta pasada y el comando sigue con lo demás")
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
            self.bus.node("ground", telemetry.WARN, f"Aclaración: {line}")
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

    def look(self, caption: str = ""):
        """One frame, then the dashboard knows what was in it.

        The frame published is ``self.shown`` - the ANNOTATED one, every blob outlined and
        labelled with what was decided about it - because the raw feed cannot answer the
        question the dashboard exists to answer: whether the robot saw the green ball as a ball
        at (+0.366, -0.472) or as an unclassified blob.
        """
        self.bus.node("capture", telemetry.ACTIVE, "Capturando la mesa")
        began = time.perf_counter()
        image, depth, camera = isaac_io.capture(self.env, self.sensor)
        self._last_capture = (image, depth, camera)
        self.bus.set_metric("capture_ms", round((time.perf_counter() - began) * 1000, 1))
        self._latest_rgb = (time.monotonic(), image, time.time())
        if self.mask_arm:
            image = self._mask(image, camera)
        perceived = time.perf_counter()
        seen = perceive.perceive(image, source=self.label, camera=camera,
                                frame=isaac_io.robot_frame(self.env), plane_z=scene_spec.TABLE_TOP_Z,
                                depth=depth, looking_for=self.args.detect,
                                robust=not getattr(self.args, "legacy_perception", False) and self.args.variant != "out_of_reach")
        self.bus.set_metric("segmentation_ms", round((time.perf_counter() - perceived) * 1000, 1))
        self.scene, self.last_image = seen, image
        annotation = time.perf_counter()
        self.shown = image if self.args.raw else perceive.annotate(image, seen, "Percepción de la mesa", lang="es")
        self.bus.set_metric("annotation_ms", round((time.perf_counter() - annotation) * 1000, 1))
        self.writer.offer(self.shown)
        if self.viewer is not None:
            self.viewer.show(self.shown, caption)
        self.frames += 1
        self.bus.node("capture", telemetry.OK, "Fotograma capturado")
        self.bus.set_metric("perception_ms", round((time.perf_counter() - began) * 1000, 1))
        # Corta a propósito: la caja de percepción es pequeña. El motivo va en la caja de cámara.
        note = f"{len(seen.objects)} objetos, {len(seen.containers)} cestas"
        self.bus.node("perceive", telemetry.WARN if not seen.objects else telemetry.OK, note)
        try:
            mapping = self.link(seen)
        except Exception as exc:
            self.bus.node("link", telemetry.FAIL, f"No se pudo asociar la escena: {exc}")
            raise
        self.bus.set_state(detections=self._detection_rows(seen, mapping),
                           frame_note=note, frames=self.frames)
        self._publish_frame(note)
        return seen

    def _detection_rows(self, seen, mapping: dict[int, str]) -> list[dict]:
        """What the dashboard shows per detected thing: the blob and which part it resolved to."""
        rows = []
        for thing in seen.everything:
            rows.append({"id": thing.id, "kind": thing.kind, "color": thing.color,
                         "pos": thing.pos_m, "area": thing.area_px,
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
            self.bus.log(f"La cámara no se actualiza: {reason}", level="warn", node="capture")
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
                image, _camera = isaac_io.capture_rgb(self.env, self.sensor)
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
                self._camera_error("No se pudo codificar el vídeo.")
        except Exception as exc:
            self._camera_error(str(exc))

    def link(self, seen):
        from scipy.optimize import linear_sum_assignment
        things = [t for t in seen.everything if t.pos_m is not None]
        items = list(self.layout)
        costs = np.full((len(things), len(items)), 1e6)
        for row, thing in enumerate(things):
            for column, item in enumerate(items):
                if thing.color != item.color or (thing in seen.containers) != item.container:
                    continue
                if not item.container and thing.kind not in ("any", item.kind):
                    continue
                last = self.tracked_xy.get(item.prim, (item.x, item.y))
                gap = math.dist(thing.aim_xy or thing.pos_m[:2], last)
                if gap <= self.demo10.LINK_TOLERANCE_M:
                    costs[row, column] = gap
        mapping = {}
        if things and items:
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] < 1e6:
                    mapping[things[row].id] = items[column].prim
        by_name = {i.prim: i for i in self.layout}
        names = {entry["id"]: entry["name"] for entry in self.catalog}
        for thing in seen.everything:
            item = by_name.get(mapping.get(thing.id))
            if item:
                thing.entity_id = item.prim
                thing.display_name = names.get(item.prim, item.prim)
                thing.size_m = list(item.size)
                if (item.container or thing.kind in ("ball", "cube")) and seen.camera and not getattr(self.args, "legacy_perception", False):
                    from scene import CameraModel
                    point = CameraModel.from_dict(seen.camera).on_plane(*thing.centroid_px, scene_spec.TABLE_TOP_Z + item.height / 2)
                    if point:
                        thing.aim_xy = point[:2]
                self.tracked_xy[item.prim] = tuple(thing.aim_xy or scene_spec.aim_point(thing.pos_m, container=item.container))
                if item.container:
                    thing.cavity_m = list(item.inner)
        self.bus.node("link", telemetry.OK if mapping else telemetry.WARN,
                      f"{len(mapping)}/{len(seen.everything)} detecciones asociadas")
        return mapping

    def locate(self, prim, last_xy=None, kind_hint=None):
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None:
            self.locate_note = "La pieza no pertenece a la escena activa."
            return None
        seen = self.look("Siguiendo la pieza")
        candidates = [t for t in seen.objects if t.color == item.color
                      and (kind_hint == "any" or t.kind in ("any", kind_hint or item.kind))]
        nearest = self._nearest(candidates, last_xy)
        if not nearest:
            self.locate_note = "La pieza está oculta o no se identifica con seguridad."
            return None
        nearest.sort(key=lambda pair: pair[0])
        if len(nearest) > 1 and abs(nearest[0][0] - nearest[1][0]) < .005:
            self.locate_note = "Dos piezas coinciden con la referencia; seguimiento ambiguo."
            return None
        best = nearest[0][1]
        self.locate_note = ""
        self.tracked_xy[prim] = tuple(best.aim_xy or scene_spec.aim_point(best.pos_m))
        return self.tracked_xy[prim]

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
        for step in plan.steps:
            pose = "" if step.pose is None else " ".join(f"{v:+.3f}" for v in step.pose)
            import ui_spanish
            self.bus.log(f"Plan: {ui_spanish.ACTIONS.get(step.action, 'movimiento')} {pose}", node="plan")
        self.bus.node("execute", telemetry.ACTIVE, f"{len(plan.steps)} pasos")
        refused_before = self.bot.refused
        # Demo 10's retarget callback replans the whole original command. That can
        # change identities, destinations and release heights inside a single task.
        # Demo 11 retries explicitly between tasks with the same requested identity.
        was_masked = self.mask_arm
        self.mask_arm = True
        try:
            run = self.bot.run(plan, resolve=premise.get, keys=lambda: self.keys(),
                               observe=lambda prim: self.locate(prim, self.tracked_xy.get(prim)),
                               retarget=None)
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
        self.bus.set_state(real_grasps=run.real_grasps, tracked=self.tracked,
                           reaimed=self.reaimed, refused_grasps=self.bot.refused,
                           recentred=self.bot.corrected,
                           kinematic_grasps=self.bot.kinematic)
        return run

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
            done = self._retry_placed(prim)
        finally:
            self.mask_arm = was
        self.bus.node("recover", telemetry.OK if done else telemetry.FAIL,
                      f"{prim}: {'recuperada' if done else 'no se pudo recuperar'}")
        return done

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
            self.say(f"    {prim}: no hay dónde devolverlo")
            return False
        seen = self.look("recuperación")
        premise = self.link(seen)
        pid = next((key for key, name in premise.items() if name == prim), None)
        thing = (next((t for t in seen.everything if t.id == pid), None)
                 if pid is not None else None)
        if thing is None or thing.pos_m is None:
            self.say(f"    {prim}: no lo encuentro en la imagen; no se intenta a ciegas")
            return False
        grasp = np.asarray(planner.grasp_pose(thing, offset=self.args.offset), dtype=float)
        lift = grasp + np.array([0.0, 0.0, executor_module.ABOVE])
        target = next((t for t in seen.containers if t.entity_id == tray.prim), None)
        if target is None:
            self.bus.log("No se identifica el recipiente durante la recuperación.", level="warn")
            return False
        drop = np.asarray(planner.place_pose(target), dtype=float)
        allocation = getattr(self, "_task_slots", {}).get(prim)
        if allocation:
            drop[:2] += allocation[1]
        drop_low = drop.copy()
        drop_low[2] = max(drop[2], real_world.safe_release_z(tray))
        above_drop = drop_low + np.array([0.0, 0.0, executor_module.ABOVE])

        self.tracked_xy[prim] = (float(thing.pos_m[0]), float(thing.pos_m[1]))
        observe = lambda: self.locate(prim, self.tracked_xy.get(prim), kind_hint="any")
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
            self.say(f"    {prim}: las mordazas cierran sobre nada")
            return False
        before = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()
        lifted = self.bot.do("lift", lift, executor_module.CLOSE, prim=prim, on_close=before,
                             check="follow")
        self.bot.log(f"      lift: {lifted.note}")
        if not lifted.ok:
            return False
        self.bot.do("carry", above_drop, executor_module.CLOSE)
        self.bot.do("lower", drop_low, executor_module.CLOSE)
        self.bot.drop()
        self.bot.do("release", drop_low, executor_module.OPEN)
        self.bot.do("retreat", above_drop, executor_module.OPEN)
        self.bot.settle()
        return True

    def _one_at_a_time(self, command, plan) -> bool:
        self.bus.set_state(phase="sorting")
        self.bus.node("plan", telemetry.ACTIVE, "Una pieza por vez; observación entre tareas")
        # Keep the requested identities and destinations. Replanning the entire command
        # would silently forget missing pieces or select a different same-colour object.
        initial = self.link(self.scene)
        for task in plan.tasks:
            prim = initial.get(task.pick)
            tray_prim = initial.get(task.place)
            for attempt in range(1 + self.args.regrasp):
                if self.bus.stop_requested():
                    return False
                seen = self.look("Actualizando la posición de la siguiente pieza")
                premise = self.link(seen)
                inverse = {name: identifier for identifier, name in premise.items()}
                things = {thing.id: thing for thing in seen.everything}
                if prim not in inverse or (task.place is not None and tray_prim not in inverse):
                    self.bus.log("No se localiza la pieza o su recipiente; la tarea queda pendiente.", level="warn")
                    break
                one = planner.Plan(intent=plan.intent)
                piece = things[inverse[prim]]
                if task.place is None:
                    planner._grasp_and_hold(one, piece, self.args.offset, task.tag)
                else:
                    planner._pick_and_place(one, piece, things[inverse[tray_prim]], self.args.offset, task.tag)
                run = self.run_plan(one, premise, command)
                if run.aborted and not run.stale_grasps:
                    return False
                if not run.stale_grasps:
                    if one.places:
                        self.check(one, premise, announce=False)
                    else:
                        self.ledger.record(self._task_key(one.tasks[0], premise), bool(run.real_grasps and self.bot.holding))
                        self._settle_rows()
                    break
                if attempt < self.args.regrasp:
                    self.park()
        return True

    def _tray_of(self, prim):
        allocation = getattr(self, "_task_slots", {}).get(prim) or getattr(self, "_placed_slots", {}).get(prim)
        if allocation:
            return next((i for i in self.layout if i.prim == allocation[0]), None)
        item = next((i for i in self.layout if i.prim == prim), None)
        matches = [i for i in self.layout if item and i.container and i.color == item.color]
        return matches[0] if len(matches) == 1 else None

    def check(self, plan, premise, announce: bool = True):
        self.bus.node("verify", telemetry.ACTIVE,
                      f"Comprobando {len(plan.picks)} colocaciones")
        misses = self._verify_and_retry(plan, premise, announce)
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
                return "No se identifican todas las piezas de esa referencia: " + ", ".join(missing) + ". Separa las piezas o cambia la cámara."
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
        # A new command is a new job: the previous one's checklist has nothing to do with this
        # one, and leaving it up would claim work that was never asked for.
        self._task_rows.clear()
        self._publish_tasks()
        telemetry.reset_nodes(self.bus)
        self.bus.node("input", telemetry.ACTIVE, text)
        self.bus.set_state(phase="command", command=text)
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
            if gap:
                outcome, reason = telemetry.NEEDS_CLARIFICATION, gap
                return True
            plan = planner.plan_command(command, seen, offset=self.args.offset)
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
                outcome, reason = telemetry.NEEDS_CLARIFICATION, "Una pieza aparece varias veces. Envía cada traslado de esa pieza en una orden distinta."
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
                        outcome, reason = telemetry.CANCELLED, "Ejecución interrumpida."
                        return False
                    if run.aborted and not run.stale_grasps:
                        for task in plan.tasks:
                            self.ledger.record(self._task_key(task, premise), False)
                        self._settle_rows()
                        outcome, reason = telemetry.FAILED, "El ejecutor detuvo la tarea: no se confirmó el agarre o el movimiento. Revisa la cámara y la posición de la pieza."
                        return True
                    if not run.stale_grasps:
                        if plan.places:
                            self.check(plan, premise)
                        else:
                            for task in plan.tasks:
                                self.ledger.record(self._task_key(task, premise),
                                                   bool(run.real_grasps and self.bot.holding))
                            self._settle_rows()
                        break
                    if attempt < self.args.regrasp:
                        self.park()
                        seen = self.look("Reintentando tras perder la pieza")
                        fresh_premise = self.link(seen)
                        fresh = planner.plan_command(command, seen, offset=self.args.offset)
                        fresh_targets = {(fresh_premise.get(task.pick), fresh_premise.get(task.place)) for task in fresh.tasks}
                        if not fresh.moves or fresh.question or fresh.unresolved or fresh_targets != requested:
                            self.bus.log("No se pueden conservar las identidades y destinos al reintentar; la orden termina como fallida.", level="warn")
                            break
                        premise, plan = fresh_premise, fresh
            rows = list(self._task_rows.values())
            done = sum(row.get("state") == "done" for row in rows)
            outcome = (telemetry.SUCCESS if rows and done == len(rows) else
                       telemetry.PARTIAL_SUCCESS if done else telemetry.FAILED)
            reason = f"{done}/{len(rows)} tareas verificadas."
            if done < len(rows):
                reason += " No se ha confirmado el agarre o la colocación de todas las piezas; consulta la lista de tareas."
            if self.bus.stop_requested():
                outcome, reason = telemetry.CANCELLED, "Ejecución interrumpida."
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
            self.bus.log(f"  Se ha oído: {heard}" if heard else "  No se ha oído ningún comando")
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
                self.bus.log("  parada solicitada: no se acepta ninguna orden más", level="warn")
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
                outcome, note = telemetry.FAILED, "todavía no hay imagen"
            else:
                self.writer.force(self.shown)
                self.say(f"  imagen guardada ({self.writer.written} en total) en "
                         f"{self.writer.folder}")
        elif action == "voice":
            outcome = telemetry.FAILED
            note = ("el micrófono se elige al arrancar con --voice"
                    if self.voice is None else "el micrófono ya está escuchando")
            self.bus.log(f"  {note}", level="warn")
        elif action == "start":
            self.bus.open_start()
        else:
            outcome, note = telemetry.FAILED, f"la demo no sabe hacer {action!r}"
            self.bus.log(f"  {note}; parámetros {params}", level="warn")
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
                                  "llegó antes de iniciar el entorno")
    if start_id:
        bus.set_state(start_command_id=start_id)
    for key in ("variant", "camera", "rounds", "seed"):
        if key in params:
            try:
                setattr(args, key, type(getattr(args, key))(params[key]))
            except (TypeError, ValueError):
                bus.log(f"  the dashboard's {key} = {params[key]!r} is not a valid value",
                        level="warn")


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # -- demo 10's flags, verbatim, so the muscle memory still works
    parser.add_argument("--headless", action="store_true")
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
    parser.add_argument("--work-x", type=float, nargs=2, default=list(DEFAULT_WORK_X),
                        metavar=("XMIN", "XMAX"),
                        help="where the robot may put a part down to be picked up. Demo 10's "
                             "declared working area is 0.27..0.52, but the arm cannot hold a "
                             "part past about 0.40: the jaws end up 14-17 mm from the "
                             "commanded pose there, against 3.1 mm of lateral slack")
    parser.add_argument("--allow-kinematic", action="store_true",
                        help="permit demo 10's carry-the-part-by-hand fallback. Results obtained "
                             "this way are NOT reproducible on real hardware")
    parser.add_argument("--release", default="clear", choices=["clear", "rim"],
                        help="'clear' releases from the height the fingertips can reach above "
                             "the rim (the only one a real arm can do); 'rim' is demo 10's "
                             "12 mm above the rim, kept for comparison")
    parser.add_argument("--close-command", type=float,
                        default=real_world.GRIPPER.close_command_rad, metavar="RAD",
                        help="how far the jaws close. LeIsaac's 0.4 rad grips nothing here; "
                             "0.20 rad holds these parts (measured, tools/probe_grip_hold.py)")
    parser.add_argument("--grip-stiffness", type=float,
                        default=real_world.GRIPPER.actuator_stiffness, metavar="K",
                        help="gripper actuator stiffness. At the shipped 17.8 the ball "
                             "follows only 79%% of the lift; at 250 it follows 94%%")
    return parser


def main(args=None, bus=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = build_parser()
    args = args or parser.parse_args()
    if args.web and not args.setup_only and not getattr(args, "worker", False):
        from supervisor import run
        return run(args)

    args.scene_data, args.scene_items = scene_config.build(args.scene_data) if getattr(args, "scene_data", None) else scene_config.load(args.scene_config)
    args.detect = sorted({i.color for i in args.scene_items})

    bus = bus or telemetry.Bus()
    bus.node("setup", telemetry.ACTIVE, "Preparando el entorno")
    bus.set_state(phase="setup", variant=args.variant, seed=args.seed, voice="off",
                  rounds=args.rounds, camera=args.camera, release=args.release,
                  close_command=args.close_command, grip_stiffness=args.grip_stiffness,
                  allow_kinematic=args.allow_kinematic)

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
            bus.log("  tiempo de espera agotado: se arranca con la configuración de la línea "
                    "de órdenes, sin esperar al panel", level="warn")
        apply_start_params(args, bus)

    # The command model before the simulator, and the microphone with it. Loading a model after
    # Isaac takes 0xC0000005 inside gliner2's from_pretrained, and two heavy native libraries
    # loading at once did the same thing before that.
    bus.set_state(phase="loading")
    try:
        model = CommandModel() if args.intent_backend != "rules" else None
        if args.intent_backend != "model":
            from intent_parser import HybridParser
            model = HybridParser(model, scene_config.catalog(args.scene_data, args.scene_items))
    except Exception as exc:
        bus.set_state(phase="failed", error=f"No se pudo cargar el modelo: {exc}")
        bus.fail_active(str(exc))
        raise
    listener = None
    if args.voice:
        import voice

        complaint = voice.check_microphone()
        if complaint:
            parser.error(f"--voice cannot work: {complaint}")
        listener = voice.Listener(model=args.voice_model, language=args.voice_language,
                                  device=args.voice_device)
        bus.set_state(voice="on", voice_note="Micrófono activo; también puedes enviar órdenes desde el panel.")

    bus.set_state(phase="starting", voice="on" if listener else "off")
    bus.node("setup", telemetry.ACTIVE, "Arrancando Isaac Sim")
    print("  Arrancando Isaac Sim; el panel mostrará el estado y cualquier error.", flush=True)
    try:
        isaac_io.prepare_runtime(headless=args.headless)

        layout = None
        if args.variant == "out_of_reach":
            from dataclasses import replace
            layout = [replace(item, x=scene_spec.OUT_OF_REACH_ITEMS[item.prim][0],
                              y=scene_spec.OUT_OF_REACH_ITEMS[item.prim][1])
                      if item.prim in scene_spec.OUT_OF_REACH_ITEMS else item for item in args.scene_items]
            env, sensor = build_real_env(args.camera, layout, pen=False,
                                           grip_stiffness=args.grip_stiffness)
            print("  two parts placed beyond the arm's reach, on purpose:", flush=True)
            for item in scene_spec.OUT_OF_REACH_ITEMS:
                print(f"    {item}: "
                      f"{scene_spec.why_unreachable(*scene_spec.OUT_OF_REACH_ITEMS[item])}",
                      flush=True)
        else:
            env, sensor = build_real_env(args.camera, list(args.scene_items),
                                         pen=args.variant == "contained",
                                         grip_stiffness=args.grip_stiffness)
            if args.variant == "contained":
                print(f"  working area walled in: a {pen_module.PEN_HEIGHT * 100:.1f} cm kerb around "
                      f"x {scene_spec.WORK_X} y {scene_spec.WORK_Y}", flush=True)
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
            import torch
            hidden = env.scene[scene_spec.SHIPPED_CUBE]
            pose = hidden.data.root_state_w[:, :7].clone()
            pose[:, 2] = -10.0
            hidden.write_root_pose_to_sim(pose)
            hidden.write_root_velocity_to_sim(torch.zeros_like(hidden.data.root_state_w[:, 7:13]))
        isaac_io.aim_camera(env, args.camera, sensor)
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
        bus.node("setup", telemetry.OK, f"Isaac en marcha; variante {args.variant}, cámara "
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
                layout = dynamics.randomize(robot.base, seed, spread=args.spread)
                try:
                    for item in layout:
                        scene_config._check_position(item, [other for other in layout if other is not item])
                except ValueError:
                    layout = list(robot.base)
                    robot.say("La variación no deja separación segura; se conserva la disposición validada.")
                robot.say(f"\n=== round {round_index} (seed {seed}) ===")
            bus.set_state(round=round_index, seed=seed, phase="scene")

            robot.park()  # before populate: the park sweep moves whatever is down
            robot.populate(layout)
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
                             ("pérdidas de objeto tras cerrar", robot.bot.lost),
                             ("centrados del efector", robot.bot.corrected),
                             ("sujeciones forzadas (solo simulación)", robot.bot.kinematic)):
            robot.say(f"  {label:<40}: {value}")
        bus.node("report", telemetry.OK if status in ("SUCCESS", "NO_TASKS") else telemetry.FAIL,
                 f"{ok_count} ok, {failed_count} fallidas")
        bus.set_state(phase="done", status=status, verified=robot.verified,
                      completed_tasks=ok_count, failed_tasks=failed_count,
                      rejected_commands=robot._rejected_commands,
                      failed_labels=robot.ledger.failed_labels(),
                      recovered=robot.recovered, missed=robot.missed,
                      refused_grasps=robot.bot.refused, lost_grasps=robot.bot.lost,
                      kinematic_grasps=robot.bot.kinematic)
        if robot.bot.allow_kinematic:
            robot.say("  NOTA: --allow-kinematic estaba activo: algunas piezas se llevaron "
                      "escribiendo su pose. Un brazo real no puede hacerlo.")
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
