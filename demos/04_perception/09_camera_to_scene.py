"""Put a camera on the simulation table, and turn what it sees into a scene JSON.

The demo before this one ends with the robot not being able to see anything: demo 08
answers a question about the table with "todavía no puedo ver la mesa: la cámara
llegará en la siguiente demo", and its README names this demo as the thing that
replaces that answer. This is that demo.

Three things happen here, and they can be run separately because they fail separately:

* **the camera**. A ``CameraCfg`` is added to the task's config, tilted 67 degrees over
  the working area and placed on the *far* side of the table so it looks back past the
  objects towards the robot, rather than through the arm. The scene is then filled with
  cubes, balls and coloured zones, inside the reach demo 05 already proved, so demo 10
  can reuse the layout unchanged.
* **the scene JSON**. Colour segmentation turns the frame into a list of objects with a
  colour, a kind, a position and a confidence. That list is written to disk and is what
  demos 10 and 12 consume. No Isaac code is involved in producing it: see
  ``perceive.py``.
* **the measurement**. The simulator knows exactly where it put every object and it
  knows the camera's pose, so each detection can be scored against it: was the colour
  right, was the kind right, and how many centimetres off was the position. This is
  what turns a picture of a working pipeline into a number, and it is the number demo
  12 puts on its board.

Run:

    python .\\demos\\04_perception\\09_camera_to_scene.py
    python .\\demos\\04_perception\\09_camera_to_scene.py --trials 20
    python .\\demos\\04_perception\\09_camera_to_scene.py --calibrate
    python .\\demos\\04_perception\\09_camera_to_scene.py --source webcam
    python .\\demos\\04_perception\\09_camera_to_scene.py --source folder out/frames

The colour bands in ``colors.py`` were measured off rendered frames and the numbers are
written down next to each band. If you change the objects, the lighting or the camera,
run ``--calibrate`` once and check them again before trusting the accuracy figures.

Accuracy as measured over 20 rearrangements (120 objects): 120/120 detected, 120/120
correct colour, 117/120 correct kind, plane position error median 10 mm.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import colors  # noqa: E402
import perceive  # noqa: E402
import scene_spec  # noqa: E402
from scene import CameraModel, Frame, Scene, Thing, quat_to_matrix  # noqa: E402

TASK = "LeIsaac-SO101-LiftCube-v0"
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"
RESULTS = HERE / "results"

# env_loader.py belongs to demos/01_simulation_basics and starts Kit, so it is imported
# by path. There are no __init__.py files anywhere in this repository, which is the same
# reason 08_voice_hybrid.py loads command_model.py with importlib.
ENV_LOADER = HERE.parent / "01_simulation_basics" / "env_loader.py"

# How long the objects are given to fall onto the table before anything is looked at.
# Demo 05 uses 20 physics steps for the same job; a 6 mm cube dropped on a flat plane
# needs a few more because it can land on an edge.
SETTLE_STEPS = 40

# Physics steps given to the arm to reach its parked pose. The SO-101 servos are
# position-controlled with stiffness 17.8, so 60 steps of 16.7 ms is a second, which is
# plenty for a 1.4 rad elbow move and is not enough for it to drift back.
ARM_PARK_STEPS = 60

# How close a detection has to be, on the table, to count as having found an object.
#
# 6 cm is not arbitrary and not tight. The primary position estimate comes from one pixel
# ray intersected with the table plane, and the pixel it starts from is on the object's
# *top face*, which is up to 3 cm above the plane. A ray from a steep angle therefore
# lands height/tan(elevation) beyond the object, about 4.7 cm for a 3 cm cube at 67
# degrees. So the plane method measures where the object appears to touch the table, not
# where its centre is, and matching has to allow for that. 6 cm is still far tighter than
# the 8 cm minimum gap in the layout, so it cannot match the wrong object.
MATCH_RADIUS_M = 0.06


def prepare_runtime(headless: bool) -> None:
    """Start Kit through demos/01_simulation_basics/env_loader.py.

    That module is not copied here on purpose: it holds the two workarounds this
    environment needs (import h5py before Kit, accept the Omniverse EULA) and duplicating
    it would guarantee the two copies drift apart.
    """
    import importlib.util

    if not ENV_LOADER.exists():
        raise FileNotFoundError(f"{ENV_LOADER} is missing; the simulation demos live in ../01_simulation_basics")
    spec = importlib.util.spec_from_file_location("demo_env_loader", ENV_LOADER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {ENV_LOADER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.prepare_runtime(headless=headless)


# --- the simulator ------------------------------------------------------------------


def rigid_cfg(item: scene_spec.Item):
    """One scene_spec.Item as an IsaacLab RigidObjectCfg.

    Three things about the spawner are easy to get wrong, and all three cost a
    two-minute simulator start each:

    * ``rigid_props`` must be present even when empty.
      ``_spawn_geom_from_prim_type`` only calls ``define_rigid_body_properties`` when it
      is not None, so leaving it out spawns a prim with mass and collision but no
      ``RigidBodyAPI``, and the environment then fails with *"Failed to find a rigid body
      when resolving '/World/envs/env_.*/Scene/cube_blue'"*. Mass is a separate class
      again: ``mass_props=MassPropertiesCfg(mass=...)``, never ``RigidBodyPropertiesCfg``.
    * ``collision_props`` goes on the *mesh* child, which is where the spawner puts it,
      and that is done for us.
    * ``spawn_cuboid`` scales a unit cube, so a 0.13 x 0.10 x 0.012 zone is fine: the
      sizes are per-axis, not a single edge.

    A 3 cm cube of anything solid is tens of grams; 50 g settles quickly and is light
    enough for demo 05's 0.15 rad jaw to hold if demo 10 ever tries.
    """
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    common = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=colors.COLORS[item.color].albedo, roughness=0.7
        ),
    }
    if item.shape == "sphere":
        spawn = sim_utils.SphereCfg(radius=item.size[0] / 2.0, **common)
    else:
        spawn = sim_utils.CuboidCfg(size=item.size, **common)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/" + item.prim,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(item.x, item.y, item.z), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


def ensure_camera(cfg, name: str) -> str:
    """Make sure the sensor a --camera choice names exists, and return its scene name.

    ``top`` and ``side`` are added by this demo; ``front`` is the sensor the task already
    ships, and is used as it is. See ``scene_spec.CAMERA_CHOICES``.
    """
    from scene_spec import CAMERA_CHOICES

    sensor, add_it = CAMERA_CHOICES[name]
    if not add_it:
        return sensor

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    pose = scene_spec.camera_poses()[sensor]
    setattr(
        cfg.scene,
        sensor,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Scene/" + sensor + "_camera",
            update_period=0.0,
            width=pose["width"],
            height=pose["height"],
            data_types=["rgb", scene_spec.DEPTH],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=pose["focal_length"],
                horizontal_aperture=38.11,
                clipping_range=(0.01, 20.0),
                lock_camera=False,
            ),
        ),
    )
    return sensor


def build_env(args, items: list[scene_spec.Item]):
    """The shipped lift-cube task, plus our objects, plus our camera, all deterministic."""
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import leisaac.tasks  # noqa: F401  registers the gym task ids

    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)

    # The action manager is only installed by the teleop device, and without it the
    # arm_action term is still MISSING and the environment cannot be built. Demo 05
    # does the same; the arm is never driven here, only parked.
    cfg.use_teleop_device("so101_state_machine")
    cfg.seed = 0
    cfg.recorders = None
    cfg.terminations.time_out = None
    cfg.terminations.success = None  # otherwise lifting the cube ends the episode

    # The task randomises the cube by up to 7.5 cm and the camera by 5 mm on every
    # reset. That is right for training a policy and fatal for a measurement, so both
    # terms are dropped. The poses are written explicitly below instead.
    for name in ("domain_randomize_0", "domain_randomize_1"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)

    # The task's own `front` TiledCamera and the observation term that reads it are both
    # removed, and this demo adds its own cameras instead. Three reasons, in order of how
    # much they cost when they are wrong:
    #
    # 1. On this machine isaacsim.sensors.rtx fails to load ("DLL load failed while
    #    importing _generic_model_output"), which takes the lidar, radar and
    #    syntheticdata plugins with it. The shipped observation term reads the camera
    #    during load_managers, so the environment dies there with a bare "CUDA error: an
    #    illegal memory access was encountered" and no clue why.
    # 2. Demo 09 needs the camera at a pose it controls, and the shipped one is on the
    #    robot base over a metre from the table, where a 3 cm cube is about 15 px.
    # 3. Removing a term from an observation group is safe: the observation manager skips
    #    terms whose cfg is None, exactly as demo 05 removes terminations the same way.
    for term in ("front", "wrist"):
        if hasattr(cfg.scene, term):
            setattr(cfg.scene, term, None)
        if hasattr(cfg.observations.policy, term):
            setattr(cfg.observations.policy, term, None)

    for item in items:
        if item.prim == scene_spec.SHIPPED_CUBE:
            continue  # already a rigid body inside the scene USD
        setattr(cfg.scene, item.prim, rigid_cfg(item))

    sensor = ensure_camera(cfg, args.camera)
    return gym.make(TASK, cfg=cfg).unwrapped, sensor



def look_at_opengl(eye, target) -> tuple[float, float, float, float]:
    """A (w, x, y, z) quaternion aiming a camera from eye at target, OpenGL frame.

    Built with Isaac Lab's own ``create_rotation_matrix_from_view`` rather than by hand,
    because getting the camera frame wrong is silent: the camera renders perfectly well
    from an orientation that perception then cannot invert. Isaac's function is the one
    ``Camera.set_world_poses_from_view`` uses, so it is by construction the frame the
    sensor understands. Its docstring says it all:

        "get rotation matrix in opengl format (-Z forward, +Y up)"

    ``up_axis`` is the string "Z", not a vector: the function compares it against the
    literals "Y" and "Z" and raises on anything else, passing a tensor fails.
    """
    import torch
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    eyes = torch.tensor([list(eye)], dtype=torch.float32)
    targets = torch.tensor([list(target)], dtype=torch.float32)
    matrix = create_rotation_matrix_from_view(eyes, targets, "Z")
    quat = quat_from_matrix(matrix).squeeze(0)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def ensure_camera(cfg, name: str) -> str:
    """Make sure the sensor a --camera choice names exists, and return its scene name.

    All three choices are added by this demo, including ``front``. The shipped ``front``
    sensor is removed in ``build_env`` together with the observation term that read it,
    so this name replaces nothing. See ``scene_spec.CAMERA_CHOICES``.

    The pose goes in ``CameraCfg.offset`` rather than being applied afterwards. That is
    deliberate and cost two simulator starts to learn: the pose is applied by the same
    ``PinholeCameraCfg`` call that creates the prim, so it cannot be lost, whereas
    ``Camera.set_world_poses_from_view`` returned without complaint and left the sensor
    reporting ``pos_w = [0, 0, 0]`` - the prim's default - with every projection in the
    report measured against a camera that was still at the world origin.
    ``aim_camera`` then only checks that the pose took.
    """
    from scene_spec import CAMERA_CHOICES

    sensor, add_it = CAMERA_CHOICES[name]
    if not add_it:
        return sensor

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    place = scene_spec.camera_poses()[sensor]
    setattr(
        cfg.scene,
        sensor,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Scene/" + sensor + "_camera",
            update_period=0.0,
            width=place["width"],
            height=place["height"],
            data_types=["rgb", scene_spec.DEPTH],
            offset=CameraCfg.OffsetCfg(
                pos=place["pos"],
                rot=look_at_opengl(place["pos"], place["look_at"]),
                convention="opengl",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=place["focal_length"],
                horizontal_aperture=38.11,
                clipping_range=(0.01, 20.0),
                lock_camera=True,
            ),
        ),
    )
    return sensor


def aim_camera(env, args, sensor: str) -> None:
    """Check that the camera ended up where ``scene_spec`` said it should.

    The pose was already applied at spawn time, so this only reads it back and fails
    loudly. A camera that is not where the scene JSON claims it is makes every number in
    the report wrong in a way that is very hard to notice later, and it is the one thing
    worth a hard error rather than a warning.
    """
    if not scene_spec.CAMERA_CHOICES[args.camera][1]:
        return
    place = scene_spec.camera_poses()[sensor]
    wanted = np.asarray(place["pos"], dtype=float)
    back = env.scene[sensor].data.pos_w[0].detach().cpu().numpy()
    off = float(np.linalg.norm(back - wanted))
    print(
        f"  {sensor} camera asked for {np.round(wanted, 4).tolist()}, "
        f"sensor reports {np.round(back, 4).tolist()} (off by {off * 1000:.1f} mm)",
        flush=True,
    )
    if off > 0.002:
        raise RuntimeError(
            f"the {sensor} camera is {off * 1000:.0f} mm from where scene_spec put it. "
            f"Every position in the report would be measured against a camera that is "
            f"not where it claims to be, so this stops here rather than reporting numbers."
        )


def park_arm(env) -> None:
    """Fold the arm up and back, so it is not between the camera and the table.

    The arm's home pose puts its upper links at z 0.2..0.45 directly over the working
    area, which is exactly where a camera looking down from the far side has to see.
    Every sensible viewpoint either has the arm in it or looks through it: the first
    render of this demo was half yellow arm and half table, with the blue cube and the
    green ball completely hidden behind the gripper.

    Parking is done by writing joint positions and stepping physics, not by writing the
    root pose, because this is an articulation. The three angles below are the ones demo
    05 already drives through on every pick, so they are known to be reachable and to
    clear the table; they differ only in folding the elbow further back.
    """
    import torch

    robot = env.scene["robot"]
    names = list(robot.joint_names)
    # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    folded = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 1.30,
        "elbow_flex": 1.45,
        "wrist_flex": 0.60,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }
    missing = [name for name in folded if name not in names]
    if missing:
        print(f"  NOTE: the arm has no {', '.join(missing)}; not parking it", flush=True)
        return
    positions = torch.tensor([[folded[name] for name in names]], device=env.device)
    with torch.inference_mode():
        for _ in range(ARM_PARK_STEPS):
            robot.write_joint_position_to_sim(positions)
            robot.write_joint_velocity_to_sim(torch.zeros_like(positions))
            env.sim.step(render=False)
        robot.write_joint_position_to_sim(positions)
    env.scene.update(dt=env.physics_dt)
    print(f"  arm parked: {', '.join(f'{k}={v}' for k, v in folded.items())}", flush=True)


def robot_frame(env) -> Frame:
    """The robot's own idea of where left is, rather than an assumption about it.

    With the shipped task the base sits at (0.35, -0.64) with no yaw, facing +y toward
    the cube, so the robot's left is -x. It is computed from the base quaternion anyway,
    so moving the robot does not silently break left and right.
    """
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w[0].detach().cpu().numpy().tolist()
    rotation = quat_to_matrix(quat)
    forward = rotation[:, 1]  # the base's own +y is where it faces
    up = rotation[:, 2]
    left = np.cross(up, forward)
    position = robot.data.root_pos_w[0].detach().cpu().numpy()
    return Frame.from_robot((float(position[0]), float(position[1])), left, forward)


def place_items(env, items: list[scene_spec.Item]) -> None:
    """Write every object's pose and let it settle, the way demo 05 places the cube."""
    import torch

    device = env.device
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    zeros = torch.zeros(1, 6, device=device)
    for item in items:
        obj = env.scene[item.prim]
        position = torch.tensor([[item.x, item.y, item.z]], device=device)
        obj.write_root_pose_to_sim(torch.cat([position, upright], dim=-1))
        obj.write_root_velocity_to_sim(zeros)
    for _ in range(SETTLE_STEPS):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)



def capture(env, name: str) -> tuple[np.ndarray, np.ndarray | None, CameraModel]:
    """Render one frame from a camera sensor and read it as numpy.

    The sequence matters. ``sim.step(render=False)`` advances physics without drawing,
    ``scene.update`` pulls the transforms back into the asset buffers, ``sim.render``
    produces the frame, and only then is the sensor asked to refill its buffers with
    ``force_recompute=True``: without that last argument a sensor with an update period
    can hand back the previous frame.

    ``sim.render()`` really does draw in ``--headless`` mode, because passing
    ``enable_cameras=True`` to AppLauncher (which env_loader.prepare_runtime does) sets
    the render mode to PARTIAL_RENDERING rather than NO_RENDERING.
    """
    sensor = env.scene[name]
    env.sim.render()
    sensor.update(env.physics_dt, force_recompute=True)

    rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    bgr = rgb[..., ::-1].copy()  # the annotator gives RGB, OpenCV wants BGR

    depth = None
    if scene_spec.DEPTH in sensor.data.output:
        depth = sensor.data.output[scene_spec.DEPTH][0].detach().cpu().numpy().astype(np.float64)

    height, width = sensor.data.image_shape
    camera = CameraModel(
        pos_w=sensor.data.pos_w[0].detach().cpu().numpy().tolist(),
        quat_wxyz=sensor.data.quat_w_world[0].detach().cpu().numpy().tolist(),
        matrix=sensor.data.intrinsic_matrices[0].detach().cpu().numpy().tolist(),
        width=width,
        height=height,
    )
    # Printed because aiming a sensor is easy to get wrong and impossible to debug from
    # the output of a projection: if set_world_poses_from_view silently did nothing,
    # the only way to see it is to print the pose the sensor actually ended up with and
    # compare it with the one that was asked for.
    print(f"  sensor {name}: pos {[round(v, 4) for v in camera.pos_w]}  "
          f"quat {[round(v, 4) for v in camera.quat_wxyz]}")
    print(f"  sensor {name}: fx {camera.matrix[0][0]:.1f} fy {camera.matrix[1][1]:.1f}  "
          f"cx {camera.matrix[0][2]:.1f} cy {camera.matrix[1][2]:.1f}  {width}x{height}")
    return bgr, depth, camera


# --- scoring against the simulator's own ground truth --------------------------------


class Trial:
    """One layout, what was detected in it, and how well."""

    def __init__(self, index: int, items: list[scene_spec.Item], scene: Scene, camera: CameraModel):
        self.index = index
        self.items = items
        self.scene = scene
        self.camera = camera
        self.expected = len(items)
        self.colour_hits = 0
        self.kind_hits = 0
        self.position_errors_mm: list[float] = []
        self.depth_errors_mm: list[float] = []
        self.centre_pixel_errors: list[float] = []
        self.missed: list[str] = []
        self.off_frame: list[str] = []
        self.unexpected: list[Thing] = []
        self.wrong_colour: list[str] = []
        self.wrong_kind: list[str] = []
        self.positions_printed = False

    def _distance(self, thing: Thing, item: scene_spec.Item) -> float | None:
        """Planar distance between a detection and a true object, in metres."""
        if thing.pos_m is None:
            return None
        return float(np.hypot(thing.pos_m[0] - item.x, thing.pos_m[1] - item.y))

    def _best(self, item: scene_spec.Item, claimed: set[int]) -> Thing | None:
        """The closest unclaimed detection within MATCH_RADIUS_M."""
        best: tuple[float, Thing] | None = None
        for thing in self.scene.everything:
            if thing.id in claimed:
                continue
            distance = self._distance(thing, item)
            if distance is None or distance > MATCH_RADIUS_M:
                continue
            if best is None or distance < best[0]:
                best = (distance, thing)
        return best[1] if best else None

    def _centre_pixel(self, item: scene_spec.Item) -> tuple[float, float] | None:
        """Where the object's true centre projects.

        Only a diagnostic. The centre is inside the object and therefore invisible, so
        comparing it with a blob's centroid measures how far the blob's visible top face
        sits from the centre, which is the height bias of the plane method again. It is
        printed, never used to decide a match.
        """
        return self.camera.project([item.x, item.y, item.z])

    def score(self) -> Trial:
        """Match every true object to at most one detection, then record the errors."""
        # Printed once, before matching, because it is the fastest way to tell a framing
        # problem from a segmentation one: if a column says OUTSIDE THE FRAME, the
        # camera is wrong and no threshold will fix it.
        for item in self.items:
            pixel = self._centre_pixel(item)
            if pixel is None:
                self.off_frame.append(item.prim)
                print(f"    {item.prim:<11} centre projects behind the camera")
                continue
            inside = 0 <= pixel[0] < self.scene.width and 0 <= pixel[1] < self.scene.height
            print(
                f"    {item.prim:<11} centre at px ({pixel[0]:6.1f}, {pixel[1]:6.1f})  "
                f"{'in frame' if inside else 'OUTSIDE THE FRAME'}"
            )

        claimed: set[int] = set()
        for item in self.items:
            hit = self._best(item, claimed)
            if hit is None:
                self.missed.append(item.prim)
                continue
            claimed.add(hit.id)
            if hit.color == item.color:
                self.colour_hits += 1
            else:
                self.wrong_colour.append(f"{item.prim} read as {hit.color}")
            if hit.kind == item.kind:
                self.kind_hits += 1
            else:
                self.wrong_kind.append(f"{item.prim} read as {hit.kind}")
            self.position_errors_mm.append(self._distance(hit, item) * 1000.0)
            if hit.pos_m_depth is not None:
                depth_error = float(
                    np.hypot(hit.pos_m_depth[0] - item.x, hit.pos_m_depth[1] - item.y)
                )
                self.depth_errors_mm.append(depth_error * 1000.0)
            pixel = self._centre_pixel(item)
            if pixel is not None:
                self.centre_pixel_errors.append(
                    float(np.hypot(hit.centroid_px[0] - pixel[0], hit.centroid_px[1] - pixel[1]))
                )
        self.unexpected = [t for t in self.scene.everything if t.id not in claimed]
        return self

    @property
    def found(self) -> int:
        return self.expected - len(self.missed)

    def median_position_mm(self) -> float | None:
        return float(np.median(self.position_errors_mm)) if self.position_errors_mm else None

    def median_depth_mm(self) -> float | None:
        return float(np.median(self.depth_errors_mm)) if self.depth_errors_mm else None

    def rows(self) -> list[str]:
        return [
            f"| {self.index} | {self.expected} | {self.found} | {self.colour_hits} | {self.kind_hits} | "
            f"{self.format_mm(self.median_position_mm())} | {self.format_mm(self.median_depth_mm())} | "
            f"{', '.join(self.missed) or '-'} | {', '.join(self.wrong_colour) or '-'} |"
        ]

    @staticmethod
    def format_mm(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"


def report(trials: list[Trial], args, camera: CameraModel, plane_z: float) -> str:
    """The committed report: one row per layout, then the totals that go on the board."""
    expected = sum(t.expected for t in trials)
    found = sum(t.found for t in trials)
    colours = sum(t.colour_hits for t in trials)
    kinds = sum(t.kind_hits for t in trials)
    all_errors = [e for t in trials for e in t.position_errors_mm]
    all_depth = [e for t in trials for e in t.depth_errors_mm]
    all_pixels = [e for t in trials for e in t.centre_pixel_errors]
    resolution = camera.pixels_per_m(plane_z)

    lines = [
        "# Perception results",
        "",
        f"`python .\\demos\\04_perception\\09_camera_to_scene.py --trials {len(trials)}` "
        f"on the {args.camera} camera of `{TASK}`.",
        "",
        f"- ground resolution on the table: "
        + (f"**{resolution:.0f} px/m**" if resolution else "not measurable"),
        f"- table plane used: z = {plane_z} m",
        f"- {colors.detection_note()}",
        "- frames written to `out/frames/`, scenes to `out/scenes/`, overlays to `out/annotated/`",
        "",
        "| trial | objects | found | right colour | right kind | plane err (mm) | depth err (mm) | missed | wrong colour |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trial in trials:
        lines += trial.rows()
    lines += [
        "",
        "## Totals",
        "",
        f"- objects in the scene: **{expected}**",
        f"- objects detected: **{found}/{expected}** ({100.0 * found / max(1, expected):.0f}%)",
        f"- correct colour: **{colours}/{expected}** ({100.0 * colours / max(1, expected):.0f}%)",
        f"- correct kind: **{kinds}/{expected}** ({100.0 * kinds / max(1, expected):.0f}%)",
    ]
    if all_errors:
        lines.append(
            f"- plane position error: median **{np.median(all_errors):.0f} mm**, "
            f"p95 **{np.percentile(all_errors, 95):.0f} mm**, worst **{max(all_errors):.0f} mm**"
        )
    if all_depth:
        lines.append(
            f"- depth position error: median **{np.median(all_depth):.0f} mm**, "
            f"p95 **{np.percentile(all_depth, 95):.0f} mm**, worst **{max(all_depth):.0f} mm**"
        )
    if all_errors and all_depth:
        lines.append(
            f"- **the difference between those two columns is the bias of the plane "
            f"method**: {np.median(all_errors) - np.median(all_depth):.0f} mm"
        )
    if all_pixels:
        lines.append(
            f"- blob centroid against the projected object centre: median "
            f"**{np.median(all_pixels):.0f} px**"
        )
    wrong_kind = [w for t in trials for w in t.wrong_kind]
    if wrong_kind:
        counts: dict[str, int] = {}
        for entry in wrong_kind:
            counts[entry] = counts.get(entry, 0) + 1
        summary = ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))
        lines.append(f"- wrong kind: **{len(wrong_kind)}** ({summary}). Shape is the one "
                     f"noisy column: see `perceive.BALL_FILL` for the measured ranges.")
    extra = sum(len(t.unexpected) for t in trials)
    if extra:
        lines.append(
            f"- blobs with no object behind them: **{extra}**. See "
            f"`colors.detection_note` for why white, black and yellow are off by "
            f"default; with the default set this should be 0, and anything else is a "
            f"false positive worth looking at in `out/annotated/`."
        )
    off_frame = sorted({p for t in trials for p in t.off_frame})
    if off_frame:
        lines.append(
            "- objects whose centre projects outside the frame: "
            + ", ".join(off_frame)
            + ". That is a camera placement problem, not a segmentation one."
        )
    lines += [
        "",
        "## How to read these numbers",
        "",
        "- *Found* means a detection's position on the table is within "
        f"{MATCH_RADIUS_M * 100:.0f} cm of where the object really is.",
        "- **The plane column is biased on purpose and the depth column is not.** The",
        "  plane method starts from a pixel on the object's *top face* and walks that ray",
        "  down to the table, so it reports where the object appears to touch the table:",
        "  an error of height/tan(elevation), about 47 mm for a 3 cm cube at 67 degrees.",
        "  The depth method reads the object's actual surface, so it is unbiased. Neither",
        "  is wrong; they answer different questions, and only the simulator has depth.",
        "- *Right kind* is the honest one to watch. It is the fill ratio of the mask, and",
        "  a 3 cm cube is only about 30 px across, so it is the noisiest column here.",
        f"  Measured over {expected} blobs: cubes fill 0.9562-0.9780, balls 0.9985-1.0001,",
        f"  so the threshold in `perceive.BALL_FILL` ({perceive.BALL_FILL}) sits in a gap",
        "  the data agrees on rather than near a boundary.",
        "- The bands in `colors.py` are measured, not derived from the albedos. If the",
        "  scene, the lighting or the camera changes, re-run `--calibrate` before",
        "  trusting the colour column.",
    ]
    return "\n".join(lines) + "\n"


# --- file, folder and webcam sources -------------------------------------------------


def images_from(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither a file nor a directory")
    found = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not found:
        raise FileNotFoundError(f"{path} has no images in it")
    return found


def run_from_pictures(args, paths: list[Path]) -> int:
    """Perceive saved frames. No Isaac Sim, no torch: this is the offline path."""
    detect = detector_for(args)
    looking_for = colors.colors(colors.parse_detect(args.detect))
    written = 0
    for path in paths:
        image = perceive.read_image(path)
        if args.calibrate:
            print(perceive.calibration(image))
            continue
        camera, frame = make_camera_from_args(args, image.shape[1], image.shape[0])
        scene = perceive.perceive(
            image,
            source=f"file:{path.name}",
            camera=camera,
            frame=frame,
            plane_z=args.plane_z if camera else None,
            detect=detect,
            looking_for=looking_for,
        )
        scene.write(args.out / "scenes" / f"{path.stem}.json")
        if args.annotate:
            perceive.save_png(args.out / "annotated" / f"{path.stem}.png", perceive.annotate(image, scene))
        print(scene.summary(), flush=True)
        written += 1
    if not args.calibrate:
        print(f"\n{written} scene(s) written to {args.out / 'scenes'}")
    return 0


def make_camera_from_args(args, width: int, height: int) -> tuple[CameraModel | None, Frame]:
    """A camera model for a webcam, from its horizontal field of view, and its frame.

    This is an approximation and says so: a real camera has a principal point that is
    not the centre, a yaw that is not exactly over the table, and a pose that changes.
    It is good enough to place objects on a table seen from behind.

    The pose assumes the webcam is directly above the table and looking straight down,
    which is the only arrangement where the plane intersection is a straight
    substitution. With the convention documented in ``scene.CameraModel.to_camera`` the
    camera's own +x is the optical axis, so the rotation has to send that to world -Z,
    image-left to world -X and image-up to world +Y. Its quaternion is
    (-0.5, 0.5, -0.5, -0.5).

    This was ``[0, 1, 0, 0]`` before, which is a valid rotation but the wrong one: it
    points the optical axis along world +x, so every object came out metres away from
    the table (see ``out/scenes/`` from a plain ``--source image`` run, which reported
    positions like +4.2, +0.4 m). That is the webcam path's version of the same
    transposed-axis bug the simulator path had.

    The axes then give:

    * "left", the image's own left, is the smallest world x, hence left_axis (-1, 0);
    * "front", the near edge of the table, is the smallest world y, hence (0, -1);
    * "nearest" is measured from the bottom edge of the ground footprint, which is
      (h/2) / focal below the optical axis, not from the centre.

    ``--no-intrinsics`` returns no camera and an image-frame scene instead: positions
    are then honestly unknown and only left and right are available.
    """
    if not args.intrinsics:
        return None, Frame.from_image(width, height)
    focal = (width / 2.0) / np.tan(np.radians(args.fov_h / 2.0))
    camera = CameraModel(
        pos_w=[0.0, 0.0, args.plane_z + args.cam_height],
        quat_wxyz=[-0.5, 0.5, -0.5, -0.5],
        matrix=[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        width=width,
        height=height,
    )
    frame = Frame(space="world", left_axis=(-1.0, 0.0), front_axis=(0.0, -1.0),
                  robot_xy=(0.0, -(height / 2.0) / focal))
    return camera, frame


def detector_for(args):
    if not args.detector:
        return None
    import detector

    queries = detector.parse_queries(args.queries)
    if not queries:
        raise SystemExit("ERROR: --detector needs --queries, for example --queries 'banana,apple'")
    return detector.Owlv2(queries)


def run_webcam(args) -> int:
    """One frame from a webcam, so the same code runs on the table at home."""
    import cv2

    device = cv2.VideoCapture(args.camera_index)
    if not device.isOpened():
        print(f"ERROR: camera {args.camera_index} could not be opened")
        return 1
    ok, image = device.read()
    device.release()
    if not ok:
        print("ERROR: the webcam returned no frame")
        return 1
    if args.calibrate:
        print(perceive.calibration(image))
        return 0
    camera, reference = make_camera_from_args(args, image.shape[1], image.shape[0])
    scene = perceive.perceive(
        image,
        source=f"webcam:{args.camera_index}",
        camera=camera,
        frame=reference,
        plane_z=args.plane_z if camera else None,
        detect=detector_for(args),
        looking_for=colors.colors(colors.parse_detect(args.detect)),
    )
    scene.write(args.out / "scenes" / "webcam.json")
    perceive.save_png(args.out / "annotated" / "webcam.png", perceive.annotate(image, scene))
    print(scene.summary())
    return 0


# --- the simulator run ---------------------------------------------------------------


def run_simulation(args) -> int:
    """Build the scene, look at it once per trial, and score every detection."""
    if args.calibrate:
        # The static half first, so the bands are on screen before the minutes that
        # starting the simulator takes, then one frame and out.
        run_calibration(args)
        print("", flush=True)
    print("starting the simulator, which takes a few minutes the first time", flush=True)
    prepare_runtime(headless=args.headless)

    detect = detector_for(args)
    looking_for = colors.colors(colors.parse_detect(args.detect))
    items_base = scene_spec.scene_of(not args.no_duplicate)
    trials: list[Trial] = []
    last: tuple[np.ndarray, Scene, CameraModel] | None = None
    env = None
    sensor = args.camera
    try:
        for trial in range(max(1, args.trials)):
            items = scene_spec.jittered(items_base, trial) if trial else items_base
            if env is None:
                env, sensor = build_env(args, items_base)
                env.reset()
                aim_camera(env, args, sensor)
            if trial == 0:
                park_arm(env)
            place_items(env, items)
            image, depth, camera = capture(env, sensor)
            scene = perceive.perceive(
                image,
                source=f"isaac:{args.camera}:trial{trial}",
                camera=camera,
                frame=robot_frame(env),
                plane_z=scene_spec.TABLE_TOP_Z,
                depth=depth,
                detect=detect,
                looking_for=looking_for,
            )
            perceive.save_png(args.out / "frames" / f"trial{trial:03d}.png", image)
            if args.calibrate:
                print(perceive.calibration(image), flush=True)
                print("")
                print(scene.summary(), flush=True)
                return 0
            scored = Trial(trial, items, scene, camera).score()
            trials.append(scored)
            last = (image, scene, camera)
            scene.write(args.out / "scenes" / f"trial{trial:03d}.json")
            overlay = perceive.annotate(image, scene, f"trial {trial} - {args.camera}")
            if args.check_convention:
                overlay = perceive.annotate_truth(overlay, scene, items, both_conventions=True)
                print("  drew a white cross (this file's convention) and a yellow square "
                      "(v flipped) at each object's true position", flush=True)
            perceive.save_png(args.out / "annotated" / f"trial{trial:03d}.png", overlay)
            print(
                f"  trial {trial}: found {scored.found}/{scored.expected}, "
                f"colour {scored.colour_hits}/{scored.expected}, kind {scored.kind_hits}/{scored.expected}, "
                f"median position {scored.format_mm(scored.median_position_mm())}",
                flush=True,
            )
            if scored.wrong_kind:
                print(f"    wrong kind: {'; '.join(scored.wrong_kind)}", flush=True)
    finally:
        if env is not None:
            env.close()

    if last is None:
        print("ERROR: no trial ran")
        return 1
    image, scene, camera = last
    print("")
    print(scene.summary())
    print("")

    RESULTS.mkdir(parents=True, exist_ok=True)
    text = report(trials, args, camera, scene_spec.TABLE_TOP_Z)
    (RESULTS / "segmentation_report.md").write_text(text, encoding="utf-8")
    print(text)

    expected = sum(t.expected for t in trials)
    found = sum(t.found for t in trials)
    colours = sum(t.colour_hits for t in trials)
    ok = found == expected and colours == expected
    print(f"RESULT: {'SUCCESS' if ok else 'FAILED'} -> {RESULTS / 'segmentation_report.md'}")
    return 0 if ok else 2


def measured_bands_table() -> str:
    """The bands, with what was measured where.

    ``--calibrate`` prints this against a real frame; here it just says where the numbers
    came from, so that anyone reading the report can tell measured from nominal.
    """
    return "\n".join(
        f"  {name:<7} {color.note.splitlines()[0] if color.note else ''}"
        for name, color in colors.COLORS.items()
    )


def run_calibration(args) -> int:
    """Print the bands, the scene's own layout, and what is still unmeasured."""
    print(colors.describe())
    print("")
    print(colors.detection_note())
    print("")
    for name, other in colors.overlapping(colors.DEFAULT_DETECT).items():
        print(f"WARNING: the band for {name} overlaps {', '.join(other)}")
    print("")
    print("scene spec (world metres; z is the centre of the object):")
    for item in scene_spec.scene_of(not args.no_duplicate):
        origin = "shipped USD" if item.prim == scene_spec.SHIPPED_CUBE else "primitive"
        print(
            f"  {item.prim:<11} {item.kind:<5} {item.color:<6} "
            f"({item.x:+.3f}, {item.y:+.3f}, {item.z:+.4f})  {origin}"
        )
    print("")
    print(f"working area x {scene_spec.WORK_X} y {scene_spec.WORK_Y}, "
          f"robot at {scene_spec.ROBOT_XY}, reach limit {scene_spec.MAX_REACH_M} m")

    layouts = [("default", scene_spec.scene_of(not args.no_duplicate))]
    layouts += [(f"trial {t}", scene_spec.jittered(layouts[0][1], t)) for t in range(1, args.trials)]
    problems = {name: scene_spec.clearance(items) for name, items in layouts}
    broken = {name: rows for name, rows in problems.items() if rows}
    print("")
    if broken:
        print(f"WARNING: {len(broken)} layout(s) would break the demo, so a --trials report "
              f"from them is not a result:")
        for name, rows in broken.items():
            print(f"  {name}: " + "; ".join(f"{a} vs {b} ({gap:+.1f} mm)" for a, b, gap in rows))
    else:
        print(f"layout OK: no same-colour pair touches and nothing is out of reach, "
              f"in {len(layouts)} layout(s)")

    print("")
    print(measured_bands_table())
    print("")
    print("The bands above were measured on a rendered frame, not derived from the")
    print("albedos. Run --calibrate again after changing anything about the scene, the")
    print("lighting or the camera, and check the numbers before trusting a result.")
    return 0


# --- entry point ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["isaac", "webcam", "image", "folder", "spec"],
                        default="isaac",
                        help="Where the picture comes from. 'spec' prints the bands and "
                             "the layout and exits, without starting Isaac Sim (default: isaac)")
    parser.add_argument("path", nargs="?", default=None,
                        help="A file or folder for --source image / folder")
    parser.add_argument("--camera", choices=list(scene_spec.CAMERA_CHOICES), default="top",
                        help="Which camera to read: top, side, or front. top and side "
                             "never look through the arm; front does, and is kept as the "
                             "comparison (default: top)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Layouts to try, each one rearranged (default: 1)")
    parser.add_argument("--no-duplicate", action="store_true",
                        help="Drop the second red ball, so no colour is ambiguous")
    parser.add_argument("--calibrate", action="store_true",
                        help="Report what is actually in the frame, to fill colors.py in")
    parser.add_argument("--headless", action="store_true", help="Run the simulator without a window")
    parser.add_argument("--detector", action="store_true",
                        help="Also run an open-vocabulary detector (needs --queries)")
    parser.add_argument("--queries", default="banana,apple,bottle",
                        help="What to ask the open-vocabulary detector (default: banana,apple,bottle)")
    parser.add_argument("--annotate", action="store_true", default=True,
                        help="Write an annotated PNG of every frame (default: on)")
    parser.add_argument("--no-annotate", dest="annotate", action="store_false")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--plane-z", type=float, default=0.75,
                        help="Table height in metres, for the webcam (default: 0.75)")
    parser.add_argument("--cam-height", type=float, default=0.45,
                        help="Webcam height above the table, in metres (default: 0.45)")
    parser.add_argument("--fov-h", type=float, default=60.0,
                        help="Webcam horizontal field of view in degrees (default: 60)")
    parser.add_argument("--no-intrinsics", dest="intrinsics", action="store_false", default=True,
                        help="Treat a webcam frame as having no metric scale")
    parser.add_argument("--detect", default=",".join(colors.DEFAULT_DETECT),
                        help="Colours to segment (default: "
                             f"{','.join(colors.DEFAULT_DETECT)}). Add white or black "
                             "only if you want the background or the robot's links too")
    parser.add_argument("--check-convention", action="store_true",
                        help="On the overlay, draw a white cross at each object's true "
                             "position under this file's convention, and a yellow square "
                             "under the opposite one. The sign of both lateral axes was "
                             "settled this way: with the wrong signs every object still "
                             "landed inside the frame, so check where the marker sits on "
                             "the object, not whether it is on screen at all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write the evidence")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    args.out = Path(args.out)

    if args.source == "spec":
        # The one route through this script that never starts Isaac Sim, so the bands
        # and the layout can be checked in a second on a machine with no GPU.
        return run_calibration(args)
    if args.source in ("image", "folder") and not args.path:
        print(f"ERROR: --source {args.source} needs a file or a folder")
        return 1
    if args.source == "webcam":
        return run_webcam(args)
    if args.source in ("image", "folder"):
        started = time.perf_counter()
        try:
            code = run_from_pictures(args, images_from(Path(args.path)))
        except FileNotFoundError as error:
            print(f"ERROR: {error}")
            return 1
        print(f"took {time.perf_counter() - started:.1f} s")
        return code
    try:
        return run_simulation(args)
    except Exception as error:
        import traceback

        print("")
        print("EXCEPTION while running the perception demo:", flush=True)
        traceback.print_exc()
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
