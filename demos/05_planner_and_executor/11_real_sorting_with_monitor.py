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
import importlib.util
import math
import random
import sys
import threading
from dataclasses import replace
from pathlib import Path

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
import real_world  # noqa: E402
import scene_spec  # noqa: E402
import telemetry  # noqa: E402
import viewer as viewer_module  # noqa: E402
from command_model_v2.command_model import CommandModel  # noqa: E402
from monitor import server as monitor_server  # noqa: E402

DEMO10_PATH = HERE / "10_command_to_motion.py"

# JPEG quality for the dashboard frame. 80 keeps a 640x480 annotated frame around 40 kB, which
# is sharp enough to read a blob outline over loopback and small enough to push on every frame
# without the socket becoming the thing that is slow.
FRAME_JPEG_QUALITY = 80

# How long the command wait looks at the browser before falling back to the keyboard. Short
# enough to feel instant, long enough that a person who is about to type is not fighting it.
WEB_COMMAND_POLL_S = 0.2


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
        # Descents where the tool-centre servo had to correct the commanded pose to keep the
        # aperture over the part, and descents where it stopped making progress (a joint limit).
        self.corrected = 0
        self.correcting = True
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
            if args and args[0] == "close" and result.ok:
                # Measured here because this is the last moment before the lift: afterwards the arm
                # has risen 75 mm and any geometry read later describes where the gripper *is*,
                # not where the jaws shut.
                self.at_close = self.part_geometry(kwargs.get("prim"))
            return result
        except GraspRefused:
            action = args[0] if args else kwargs.get("action", "?")
            self.refused += 1
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

        * a shift larger than ``executor.TRACK_MAX_SHIFT_M`` still counts as lost - that is the
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
        best_recent = math.inf
        stall_reported = False

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
                        if drift > executor_module.TRACK_MAX_SHIFT_M:
                            lost += 1
                        else:
                            centre = self.aperture_centre()
                            if centre is not None:
                                error = (float(seen[0]) - centre[0], float(seen[1]) - centre[1])
                                size = math.hypot(*error)
                                worst = max(worst, size)
                                if size > 0.0005:  # 0.5 mm: below this it is the arm's own noise
                                    # Never correct by more than the slack allows. If the arm is
                                    # at a limit this caps the wild swing a full gain would give.
                                    step = min(size, LATERAL_SLACK_M * 2.0)
                                    command[0] += error[0] / size * step
                                    command[1] += error[1] / size * step
                                    corrections += 1
                                    largest = max(largest, size)
                                    if self.correcting:
                                        self.corrected += 1
                                if not stall_reported:
                                    if size < best_recent - STALL_IMPROVEMENT:
                                        best_recent = size
                                    elif corrections >= STALL_WINDOW:
                                        stall_reported = True
                                        self.correcting = False
                                        self.log(f"      centring stalled: the aperture is still "
                                                 f"{size * 1000:.1f} mm from the part after "
                                                 f"{corrections} correction(s), so this pose is "
                                                 f"outside what the arm can hold it at - the "
                                                 f"jaws cannot reach it from here")
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
        return executor_module.StepResult(action=action, note=note, ok=lost == 0)


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
        items = " ".join(str(item.get("kind")) for item in command.get("items", []))
        self._bus.node("understand", telemetry.OK,
                       f"{command.get('intent')} [{items.strip() or 'no items'}]")
        return command


# Every robot that wants to hear about planning. The wrapper is installed once, the first time a
# robot is built, and is a plain delegation around ``planner.plan_command``.
_PLAN_SPIES: list = []


def install_plan_spy() -> None:
    """Let demo 11 see every plan demo 10 builds, without editing ``planner``.

    Monkeypatching a shared module is normally the wrong answer, and it is the right one here for
    two reasons: demo 11 owns this process (demo 10 is loaded into it, not the reverse), and the
    alternative - reimplementing ``handle`` - would duplicate the single piece of this system
    that is hardest to get right. The wrapper does nothing but call through and tell the robots.
    """
    if getattr(planner.plan_command, "_demo11_spy", False):
        return
    original = planner.plan_command

    def spy(command, scene, offset=(0.0, 0.0)):
        plan = original(command, scene, offset)
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
        self.web_input = bool(args.web) and not sys.stdin.isatty()
        super().__init__(env, sensor, args, viewer=viewer, model=model)
        # Its own out/ folder: these frames are annotated for the dashboard and saved on
        # request, and mixing them into demo 10's would make either run's folder a mixture.
        self.writer = viewer_module.FrameWriter(
            HERE / "out11", every=args.save_every if args.save_every > 0 else 10 ** 9)
        self.bot = StrictExecutor(env, log=self.say, allow_kinematic=args.allow_kinematic)
        self.bot.log = self._step_logged
        self.model = ModelSpy(model, bus)
        install_plan_spy()
        if self not in _PLAN_SPIES:
            _PLAN_SPIES.append(self)

    # -- output ----------------------------------------------------------------

    def say(self, text: str = "") -> None:
        """Print, and send the same line to the dashboard.

        Every line the robot prints is already the explanation of something it just decided, so
        mirroring ``say`` rather than adding calls at the interesting places means the console
        and the dashboard cannot end up telling different stories.
        """
        print(text, flush=True)
        if text.strip():
            self.bus.log(text)

    def _step_logged(self, text: str) -> None:
        """The executor's per-step log line, which is also the execute node's progress."""
        self.say(text)
        head = text.strip().split()
        if head:
            self.bus.node("execute", telemetry.ACTIVE, head[0])

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
        self.bus.set_state(findings=rows)
        for level, text in rows:
            self.bus.log(f"{level}: {text}",
                         level="fail" if level == "fail" else ("warn" if level == "warn"
                                                              else "info"))
        return rows

    def on_plan(self, plan) -> None:
        """Report what grounding and planning decided, from the plan itself.

        A plan carries both answers - ``unresolved`` and ``question`` are what grounding could not
        do, and ``steps`` is what planning did with the rest - so one hook covers both boxes
        instead of wrapping two functions that demo 10 calls as one expression.
        """
        if plan.question:
            line = next(iter(plan.question.values()))
            self.bus.node("ground", telemetry.WARN, f"asked: {line}")
        elif plan.unresolved:
            self.bus.node("ground", telemetry.WARN, "; ".join(plan.unresolved))
        else:
            self.bus.node("ground", telemetry.OK,
                          f"{len(plan.picks)} pick(s) resolved to {len(plan.places)} "
                          f"destination(s)")
        if plan.moves:
            self.bus.node("plan", telemetry.OK, plan.summary)

    # -- perception ------------------------------------------------------------

    def look(self, caption: str = ""):
        """One frame, then the dashboard knows what was in it.

        The frame published is ``self.shown`` - the ANNOTATED one, every blob outlined and
        labelled with what was decided about it - because the raw feed cannot answer the
        question the dashboard exists to answer: whether the robot saw the green ball as a ball
        at (+0.366, -0.472) or as an unclassified blob.
        """
        self.bus.node("capture", telemetry.ACTIVE, caption or "frame")
        seen = super().look(caption)
        note = (f"{len(seen.objects)} object(s), {len(seen.containers)} container(s)"
                + (f" - {caption}" if caption else ""))
        self.bus.node("perceive", telemetry.WARN if not seen.objects else telemetry.OK, note)
        try:
            mapping = self.link(seen)
        except Exception:  # noqa: BLE001 - a bad frame must not stop the run
            mapping = {}
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
        try:
            import cv2

            ok, encoded = cv2.imencode(".jpg", self.shown,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY])
            if ok:
                self.bus.put_frame(encoded.tobytes(), {"note": note})
        except Exception:  # noqa: BLE001 - encoding is a convenience, never a requirement
            return

    def link(self, seen):
        mapping = super().link(seen)
        self.bus.node("link", telemetry.OK if mapping else telemetry.WARN,
                      f"{len(mapping)}/{len(seen.everything)} blob(s) matched a part")
        return mapping

    # -- planning and motion ---------------------------------------------------

    def _release_adjusted(self, plan):
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
        floor = min(real_world.safe_release_z(tray) for tray in trays)
        raised = 0
        steps = []
        for step in plan.steps:
            if step.action in ("lower", "release") and step.pose is not None:
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
        copy.unresolved = list(plan.unresolved)
        copy.question = dict(plan.question)
        copy.reply = plan.reply
        self.bus.log(f"    release raised to {floor * 1000:.0f} mm so the fingertips clear the "
                     f"{real_world.GRIPPER.finger_tip_below_hold_m * 1000:.0f} mm rim "
                     f"({raised} step(s))")
        return copy

    def run_plan(self, plan, premise, command):
        plan = self._release_adjusted(plan)
        for step in plan.steps:
            pose = "" if step.pose is None else " ".join(f"{v:+.3f}" for v in step.pose)
            self.bus.log(f"    plan: {step.action:<8} {step.note:<34} {pose}", node="plan")
        self.bus.node("execute", telemetry.ACTIVE, f"{len(plan.steps)} step(s)")
        refused_before = self.bot.refused
        run = super().run_plan(plan, premise, command)
        if self.bot.refused > refused_before:
            # Demo 10 reports this failure as "the arm knocked the target away before closing",
            # which is the right shape of response - re-look and try again - for the wrong reason.
            # Say what actually happened, because on a real bench these two causes need different
            # fixes: a knock is a planning problem, an unheld part is a grip-force or jaw problem.
            self.say("  demo 11: the jaws closed and the part was not held, so it was not "
                     "carried. A real arm drops the part here rather than flying on to the tray.")
        state = telemetry.FAIL if (run.stale_grasps or run.aborted) else telemetry.OK
        self.bus.node("execute", state, run.summary())
        self.bus.set_state(real_grasps=run.real_grasps, tracked=self.tracked,
                           reaimed=self.reaimed, refused_grasps=self.bot.refused,
                           recentred=self.bot.corrected,
                           kinematic_grasps=self.bot.kinematic)
        return run

    def _retry_one(self, prim: str) -> bool:
        self.bus.node("recover", telemetry.ACTIVE, f"re-picking {prim}")
        done = super()._retry_one(prim)
        self.bus.node("recover", telemetry.OK if done else telemetry.FAIL,
                      f"{prim} {'re-placed' if done else 'could not be re-placed'}")
        return done

    def _one_at_a_time(self, command, plan) -> bool:
        self.bus.set_state(phase="sorting")
        self.bus.node("plan", telemetry.ACTIVE, "one object at a time, re-looking between")
        return super()._one_at_a_time(command, plan)

    def check(self, plan, premise, announce: bool = True):
        self.bus.node("verify", telemetry.ACTIVE,
                      f"judging {len(plan.picks)} placement(s) against the tray geometry")
        misses = super().check(plan, premise, announce)
        verified = len(plan.picks) - len(misses)
        self.bus.node("verify", telemetry.OK if not misses else telemetry.WARN,
                      f"{verified}/{len(plan.picks)} inside"
                      + (f"; missed {', '.join(misses)}" if misses else ""))
        self.bus.set_state(verified=self.verified, recovered=self.recovered,
                           missed=self.missed, replanned=self.replanned,
                           regrasped=self.regrasped)
        return misses

    # -- input -----------------------------------------------------------------

    def handle(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return True
        if text.lower() in self.demo10.QUIT_WORDS:
            self.quit = True
            return False
        telemetry.reset_nodes(self.bus)
        self.bus.node("input", telemetry.ACTIVE, text)
        self.bus.set_state(phase="command", command=text)
        return super().handle(text)

    def ask(self) -> str:
        """One command, from the browser, the microphone or the keyboard - in that order.

        The browser is polled first and briefly, because a person watching a demo in a browser
        should not have to guess whether the robot is listening. With ``--web`` and no terminal
        (a service, a container, ``--no-viewer`` under a scheduler) the wait is on the browser
        alone and is therefore also how ``q`` and "next round" get through with no keyboard at
        all.
        """
        text = self.bus.take_command(timeout=WEB_COMMAND_POLL_S)
        if text:
            self.bus.node("input", telemetry.ACTIVE, f"from the dashboard: {text}")
            return text

        if self.voice is not None:
            self.bus.node("input", telemetry.ACTIVE, "listening")
            self.say("  listening...")
            heard = self.voice.listen()
            self.bus.log(f"  heard: {heard}" if heard else "  heard nothing")
            if heard:
                self.bus.node("input", telemetry.OK, heard)
                return heard

        if self.web_input:
            self.bus.set_state(phase="waiting", voice_note="waiting for a command in the browser")
            self.bus.node("input", telemetry.ACTIVE, "waiting for the browser")
            text = self.bus.take_command(timeout=None)
            return text or "quit"

        return super().ask()

    def keys(self) -> set[str]:
        """Viewer keys, plus anything the dashboard asked for.

        Called after every executor step, which is what makes the browser's ``q`` abort a run
        mid-move rather than at the next command boundary - the same guarantee demo 10 gives the
        keyboard. Re-look and save-frame are handled here too, and they are the two actions that
        are safe to perform between two steps.
        """
        pressed = set(super().keys())
        while True:
            control = self.bus.take_control()
            if control is None:
                break
            action = str(control.get("action") or "")
            params = control.get("params") or {}
            if action == "stop":
                self.bus.log("  the dashboard asked to stop", level="warn")
                self.quit = True
                pressed.add("q")
            elif action == "next":
                self.bus.log("  the dashboard asked for the next round")
                self.advance = True
            elif action == "relook":
                self.bus.log("  the dashboard asked for a re-look")
                self.look("dashboard re-look")
            elif action == "save":
                if self.shown is None:
                    self.say("  no frame yet")
                else:
                    self.writer.force(self.shown)
                    self.say(f"  frame saved ({self.writer.written} so far) to "
                             f"{self.writer.folder}")
            elif action == "voice":
                self.bus.log("  voice cannot be switched on mid-run; it is chosen with --voice"
                             if self.voice is None else
                             "  the microphone is already listening between commands")
            elif action == "start":
                self.bus.open_start()
            else:
                self.bus.log(f"  the dashboard asked for {action!r}, which this demo does not "
                             f"do; parameters were {params}", level="warn")
        return pressed


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


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
    parser.add_argument("--host", default=monitor_server.DEFAULT_HOST,
                        help="interface to bind; loopback unless you have a reason")
    parser.add_argument("--port", type=int, default=monitor_server.DEFAULT_PORT,
                        help="0 asks the OS for a free port")
    parser.add_argument("--open", action="store_true", help="open a browser at the dashboard")
    parser.add_argument("--no-wait-for-start", dest="wait_for_start", action="store_false",
                        default=True,
                        help="start Isaac immediately instead of waiting for the dashboard")
    parser.add_argument("--start-timeout", type=float, default=0.0, metavar="S",
                        help="give up waiting for the dashboard after S seconds (0 = wait for "
                             "ever)")
    parser.add_argument("--setup-only", action="store_true",
                        help="print the bench sheet and serve the dashboard; do not start Isaac")
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    bus = telemetry.Bus()
    bus.node("setup", telemetry.ACTIVE, "choosing the environment")
    bus.set_state(phase="setup", variant=args.variant, seed=args.seed, voice="off",
                  rounds=args.rounds, camera=args.camera, release=args.release,
                  close_command=args.close_command, grip_stiffness=args.grip_stiffness,
                  allow_kinematic=args.allow_kinematic)

    # The setup sheet before anything else, so the dashboard is useful even if Isaac never
    # starts: this is the part of the demo that transfers to a real bench.
    sheet = real_world.setup_sheet(scene_spec.DEFAULT_SCENE, camera=args.camera)
    print("\n".join(sheet), flush=True)
    for line in sheet:
        bus.log(line)
    findings = real_world.preflight(scene_spec.DEFAULT_SCENE, camera=args.camera)
    bus.set_state(findings=findings)
    for level, text in findings:
        bus.log(f"{level}: {text}", level="fail" if level == "fail" else
                ("warn" if level == "warn" else "info"))

    monitor = None
    if args.web:
        monitor = monitor_server.serve(bus, args.host, args.port, open_browser=args.open,
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
    if args.web and args.wait_for_start and not args.text:
        bus.set_state(phase="starting")
        bus.node("setup", telemetry.ACTIVE, "waiting for the dashboard to start the environment")
        print(f"  dashboard: {monitor.url if monitor else '(none: --no-web)'}", flush=True)
        print("  waiting for the start command there, or press Enter here", flush=True)
        enter_starts_the_run(bus)
        bus.wait_start(timeout=args.start_timeout or None)

    # The command model before the simulator, and the microphone with it. Loading a model after
    # Isaac takes 0xC0000005 inside gliner2's from_pretrained, and two heavy native libraries
    # loading at once did the same thing before that.
    model = CommandModel()
    listener = None
    if args.voice:
        import voice

        complaint = voice.check_microphone()
        if complaint:
            parser.error(f"--voice cannot work: {complaint}")
        listener = voice.Listener(model=args.voice_model, language=args.voice_language,
                                  device=args.voice_device)
        bus.set_state(voice="on", voice_note="microphone live; commands also come from the "
                                             "browser and the terminal")

    bus.set_state(phase="starting", voice="on" if listener else "off")
    bus.node("setup", telemetry.ACTIVE, "starting Isaac Sim")
    print("  starting Isaac Sim; this takes about two minutes", flush=True)
    isaac_io.prepare_runtime(headless=args.headless)

    layout = None
    if args.variant == "out_of_reach":
        layout = scene_spec.out_of_reach_scene()
        env, sensor = build_real_env(args.camera, layout, pen=False,
                                       grip_stiffness=args.grip_stiffness)
        print("  two parts placed beyond the arm's reach, on purpose:", flush=True)
        for item in scene_spec.OUT_OF_REACH_ITEMS:
            print(f"    {item}: "
                  f"{scene_spec.why_unreachable(*scene_spec.OUT_OF_REACH_ITEMS[item])}",
                  flush=True)
    else:
        env, sensor = build_real_env(args.camera, list(scene_spec.DEFAULT_SCENE),
                                     pen=args.variant == "contained",
                                     grip_stiffness=args.grip_stiffness)
        if args.variant == "contained":
            print(f"  working area walled in: a {pen_module.PEN_HEIGHT * 100:.1f} cm kerb around "
                  f"x {scene_spec.WORK_X} y {scene_spec.WORK_Y}", flush=True)
    env.reset()
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
            if not item.container and abs(item.x - scene_spec.DEFAULT_SCENE[
                    [i.prim for i in scene_spec.DEFAULT_SCENE].index(item.prim)].x) > 1e-9:
                robot.say(f"  {item.prim} moved to x {item.x:.3f}: the arm cannot hold a part "
                          f"outside x {args.work_x[0]:.2f}..{args.work_x[1]:.2f}")
        if args.variant == "out_of_reach":
            robot.base = layout
        robot.voice = listener
        bus.set_state(phase="ready", sim="running")
        bus.node("setup", telemetry.OK, f"Isaac running; variant {args.variant}, camera "
                                        f"{args.camera}")
        robot.findings()

        scripted = list(args.text) * max(1, args.repeat) if args.text else []
        failures = 0
        stopped = False

        for round_index in range(args.rounds):
            seed = args.seed + round_index
            if args.variant == "out_of_reach":
                layout = robot.base
                robot.say(f"\n=== round {round_index} (static layout) ===")
            else:
                layout = dynamics.randomize(robot.base, seed, spread=args.spread)
                robot.say(f"\n=== round {round_index} (seed {seed}) ===")
            bus.set_state(round=round_index, seed=seed, phase="scene")

            robot.park()  # before populate: the park sweep moves whatever is down
            robot.populate(layout)
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

        robot.say("\n=== summary ===")
        for label, value in (("placements verified", robot.verified),
                             ("needed a retry", robot.recovered),
                             ("never verified", robot.missed),
                             ("replanned mid-task", robot.replanned),
                             ("regrasp after a stale target", robot.regrasped),
                             ("re-aimed at a knocked target", robot.reaimed),
                             ("moves that re-tracked the target", robot.tracked),
                             ("grasps refused (no friction hold)", robot.bot.refused),
                             ("descents re-centred on the part", robot.bot.corrected),
                             ("grasps carried by hand", robot.bot.kinematic)):
            robot.say(f"  {label:<40}: {value}")
        bus.node("report", telemetry.FAIL if robot.missed else telemetry.OK,
                 f"{robot.verified} verified, {robot.missed} missed, "
                 f"{robot.bot.refused} grasp(es) refused")
        bus.set_state(phase="done", verified=robot.verified, recovered=robot.recovered,
                      missed=robot.missed, refused_grasps=robot.bot.refused,
                      kinematic_grasps=robot.bot.kinematic)
        if robot.bot.allow_kinematic:
            robot.say("  NOTE: --allow-kinematic was on, so some grasps were carried by writing "
                      "the part's pose. A real arm cannot do that.")
        failures += robot.missed
        if robot.writer:
            robot.writer.close()
        return 2 if failures else 0
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
        bus.set_state(phase="stopped")
        if monitor is not None:
            # Stopped after the simulator, not before: the dashboard is how someone watching a
            # failed run finds out what it said, and it costs one second to keep it up.
            monitor.stop()


def _wait_forever(bus: telemetry.Bus) -> None:
    """Block until the dashboard asks to stop, or Ctrl-C."""
    print("  ctrl-c to stop", flush=True)
    try:
        while True:
            control = bus.take_control(timeout=1.0)
            if control and control.get("action") == "stop":
                return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    raise SystemExit(main())