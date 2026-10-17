"""Demo 12 — scene configuration.

The single source of truth for what the cell contains. Pure Python, no Isaac imports, so
it can be validated, unit-tested and round-tripped to JSON without a GPU.

The flow this models is the one the demo is built around:

    CONFIGURE ENVIRONMENT -> LAUNCH ISAAC LAB -> EXECUTE TASK -> OBSERVE RESULTS

so the configuration is complete *before* anything is simulated, and changing it after
launch means relaunching the simulator rather than mutating a running world.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from . import materials as M

# Workspace bounds. These are the measured table extents of the shipped scene
# (demos/05_planner_and_executor/scene_spec.py: TABLE_X / TABLE_Y), kept so that a scene
# configured here is dimensionally the same one Demo 11 measured its physics against.
DEFAULT_TABLE_SIZE_M = (0.70, 0.65)
DEFAULT_TABLE_CENTER_XY = (0.35, -0.35)

# The shared working area on the tabletop: the strip the robots actually reach into.
# Objects and destination bins are placed inside it.
WORK_AREA_X = (0.20, 0.50)
WORK_AREA_Y = (-0.55, -0.38)

# A SO-101 base has to sit outside the work area or the arm starts inside the parts.
ROBOT_BASE_Y = -0.64
ROBOT_BASE_XS = (0.20, 0.36, 0.52)

MAX_ROBOTS = 3
MAX_PARTS = 12

PART_KINDS = ("cube", "cylinder", "ball")

# Default part size in metres. 30 mm is the size this project has measured grasping for
# (a 30 mm cube is 28 g and follows a lift; a 40 mm ball does not fit the 36.2 mm
# aperture at all).
DEFAULT_PART_SIZE_M = 0.030

# How far a part must start from a bin's cavity edge. Measured behaviour, not taste: the
# 10 g PET sphere on the melamine top rolls about 10 mm while the scene settles, so
# anything closer than this ends up in a tray.
ROLL_CLEARANCE_M = 0.015


class SceneError(ValueError):
    """A configuration that cannot be built into a physical cell."""


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


@dataclass
class PartConfig:
    """One physical object on the table.

    ``size_m`` is the full extent, so a cube has three equal entries and a cylinder has
    (diameter, diameter, height). Mass is *not* stored: it is derived from size and
    material so it can never disagree with the geometry.
    """

    id: str
    kind: str = "cube"
    material: str = "abs"
    color: str = "#e0483c"
    x: float = 0.35
    y: float = -0.425
    yaw_deg: float = 0.0
    size_m: tuple[float, float, float] = (DEFAULT_PART_SIZE_M,) * 3

    # -- derived physical properties ------------------------------------

    @property
    def material_spec(self) -> M.Material:
        mat = M.PART_MATERIALS.get(self.material)
        if mat is None:
            raise SceneError(f"unknown material {self.material!r} on part {self.id}")
        return mat

    @property
    def mass_kg(self) -> float:
        return M.part_mass_kg(self.kind, self.size_m, self.material_spec)

    @property
    def footprint_radius_m(self) -> float:
        """Half the diagonal of the footprint: the radius of the smallest circle it fits in."""
        return math.hypot(self.size_m[0], self.size_m[1]) / 2.0

    @property
    def grasp_width_m(self) -> float:
        """The dimension the gripper has to straddle.

        For a cube and a cylinder that is the side; for a ball it is the diameter. This
        is what decides whether the part is graspable at all.
        """
        return self.size_m[0]

    @property
    def graspable(self) -> bool:
        return M.width_fits_gripper(self.grasp_width_m)

    @property
    def label_es(self) -> str:
        return {
            "cube": "Cubo",
            "cylinder": "Cilindro",
            "ball": "Bola",
        }.get(self.kind, self.kind)


# ---------------------------------------------------------------------------
# Workspace furniture
# ---------------------------------------------------------------------------


@dataclass
class BinConfig:
    """A destination tray: a real hollow box with a floor and four walls.

    The cavity is what a part has to end up inside. ``wall`` is the thickness of each
    wall, so the outer footprint is ``inner + 2*wall``.
    """

    id: str
    label: str = "Destino"
    color: str = "#4a7fb5"
    x: float = 0.35
    y: float = -0.51
    inner_m: tuple[float, float] = (0.075, 0.075)
    wall_m: float = 0.006
    height_m: float = 0.045
    floor_m: float = 0.006
    yaw_deg: float = 0.0

    @property
    def outer_size_m(self) -> tuple[float, float]:
        return (self.inner_m[0] + 2 * self.wall_m, self.inner_m[1] + 2 * self.wall_m)

    @property
    def outer_size_with_height_m(self) -> tuple[float, float, float]:
        return (self.outer_size_m[0], self.outer_size_m[1], self.height_m)

    @property
    def rim_z_m(self) -> float:
        return M.TABLE_TOP_Z_M + self.height_m

    @property
    def cavity_floor_z_m(self) -> float:
        return M.TABLE_TOP_Z_M + self.floor_m

    @property
    def accepts(self) -> str:
        return f"{self.inner_m[0] * 1000:.0f}x{self.inner_m[1] * 1000:.0f} mm"

    def contains_footprint(self, x: float, y: float, radius_m: float) -> bool:
        """Whether a part of the given footprint radius is fully inside the cavity.

        Tested radially against the cavity's *half-diagonal*, not per-axis. A per-axis test
        passes a part sitting in a cavity corner -- 30 mm out in x and 40 mm out in y is
        50 mm from the centre, inside a 75 mm cavity, but per-axis it reads as 30 mm out of
        a 37.5 mm half-width with only 7.5 mm to spare on each axis. The demo's default
        scene had exactly that: a ball that started inside a tray and was reported as
        resting on the table.
        """
        half_diagonal = math.hypot(self.inner_m[0], self.inner_m[1]) / 2.0
        return self.distance_from_center_m(x, y) + radius_m <= half_diagonal

    def distance_from_center_m(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)

    def slot_positions(self, count: int) -> list[tuple[float, float]]:
        """Up to ``count`` well-separated drop positions inside the cavity.

        A grid, because a person drops parts into the corners of a tray rather than
        stacking them in the middle.
        """
        if count <= 1:
            return [(self.x, self.y)]
        cols = 2
        rows = 2
        out: list[tuple[float, float]] = []
        for i in range(min(count, cols * rows)):
            cx = i % cols
            cy = i // cols
            fx = 0.0 if cols == 1 else (-0.25 if cx == 0 else 0.25)
            fy = 0.0 if rows == 1 else (-0.25 if cy == 0 else 0.25)
            out.append((self.x + fx * self.inner_m[0], self.y + fy * self.inner_m[1]))
        return out


# ---------------------------------------------------------------------------
# Table and robots
# ---------------------------------------------------------------------------


@dataclass
class TableConfig:
    size_m: tuple[float, float] = DEFAULT_TABLE_SIZE_M
    center_xy: tuple[float, float] = DEFAULT_TABLE_CENTER_XY

    @property
    def bounds_x(self) -> tuple[float, float]:
        cx, cy = self.center_xy
        return (cx - self.size_m[0] / 2.0, cx + self.size_m[0] / 2.0)

    @property
    def bounds_y(self) -> tuple[float, float]:
        cx, cy = self.center_xy
        return (cy - self.size_m[1] / 2.0, cy + self.size_m[1] / 2.0)

    def holds(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        x0, x1 = self.bounds_x
        y0, y1 = self.bounds_y
        return (x0 + margin_m <= x <= x1 - margin_m) and (y0 + margin_m <= y <= y1 - margin_m)


@dataclass
class RobotConfig:
    """One SO-101 follower base.

    The arm is a fixed-base articulation (``fix_root_link=True`` in the LeIsaac asset),
    so a robot's position *is* its base pose and nothing else moves it.

    ``base_yaw_deg`` defaults to 180 because of something that is easy to get wrong and
    looks like a broken robot: at every joint angle of zero the SO-101's arm points along
    **-Y**, so a robot placed on the bench with zero yaw reaches *away* from the table. A
    cell whose arms fold back over their own bases is not a reachability problem and no
    amount of IK tuning fixes it.
    """

    id: str
    label: str = "SO-101"
    base_xy: tuple[float, float] = (ROBOT_BASE_XS[0], ROBOT_BASE_Y)
    base_yaw_deg: float = 180.0
    enabled: bool = True

    @property
    def reach(self) -> M.ReachEnvelope:
        return M.ReachEnvelope(self.base_xy)


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------


@dataclass
class MoveTask:
    """One part has to end up in one bin. This is the whole task vocabulary."""

    id: str
    part_id: str
    bin_id: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MissionConfig:
    tasks: list[MoveTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": [t.to_dict() for t in self.tasks]}


# ---------------------------------------------------------------------------
# Whole scene
# ---------------------------------------------------------------------------


@dataclass
class SceneConfig:
    table: TableConfig = field(default_factory=TableConfig)
    robots: list[RobotConfig] = field(default_factory=list)
    parts: list[PartConfig] = field(default_factory=list)
    bins: list[BinConfig] = field(default_factory=list)
    mission: MissionConfig = field(default_factory=MissionConfig)
    name: str = "Celda Demo 12"

    # -- lookups ---------------------------------------------------------

    def robot(self, robot_id: str) -> RobotConfig:
        for r in self.robots:
            if r.id == robot_id:
                return r
        raise SceneError(f"no robot with id {robot_id!r}")

    def part(self, part_id: str) -> PartConfig:
        for p in self.parts:
            if p.id == part_id:
                return p
        raise SceneError(f"no part with id {part_id!r}")

    def bin(self, bin_id: str) -> BinConfig:
        for b in self.bins:
            if b.id == bin_id:
                return b
        raise SceneError(f"no bin with id {bin_id!r}")

    @property
    def enabled_robots(self) -> list[RobotConfig]:
        return [r for r in self.robots if r.enabled]

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": {
                "sizeM": list(self.table.size_m),
                "centerXy": list(self.table.center_xy),
            },
            "robots": [
                {
                    "id": r.id,
                    "label": r.label,
                    "baseXy": list(r.base_xy),
                    "baseYawDeg": r.base_yaw_deg,
                    "enabled": r.enabled,
                }
                for r in self.robots
            ],
            "parts": [
                {
                    "id": p.id,
                    "kind": p.kind,
                    "material": p.material,
                    "color": p.color,
                    "x": p.x,
                    "y": p.y,
                    "yawDeg": p.yaw_deg,
                    "sizeM": list(p.size_m),
                    "massG": round(p.mass_kg * 1000.0, 1),
                    "graspable": p.graspable,
                }
                for p in self.parts
            ],
            "bins": [
                {
                    "id": b.id,
                    "label": b.label,
                    "color": b.color,
                    "x": b.x,
                    "y": b.y,
                    "innerM": list(b.inner_m),
                    "wallM": b.wall_m,
                    "heightM": b.height_m,
                    "floorM": b.floor_m,
                    "yawDeg": b.yaw_deg,
                }
                for b in self.bins
            ],
            "mission": self.mission.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneConfig":
        """Rebuild a scene from the JSON the interface sends.

        Every field is validated on the way in, so an invalid scene is rejected with a
        message naming the field rather than crashing the simulator three minutes later.
        """
        table_raw = data.get("table") or {}
        table = TableConfig(
            size_m=_pair(table_raw.get("sizeM"), DEFAULT_TABLE_SIZE_M, "table.sizeM"),
            center_xy=_pair(table_raw.get("centerXy"), DEFAULT_TABLE_CENTER_XY, "table.centerXy"),
        )

        robots = [
            RobotConfig(
                id=str(r.get("id") or f"robot_{i + 1}"),
                label=str(r.get("label") or f"SO-101 #{i + 1}"),
                base_xy=_pair(r.get("baseXy"), (ROBOT_BASE_XS[i], ROBOT_BASE_Y), "robot.baseXy"),
                base_yaw_deg=float(r.get("baseYawDeg", 180.0)),
                enabled=bool(r.get("enabled", True)),
            )
            for i, r in enumerate(data.get("robots") or [])
        ][:MAX_ROBOTS]

        parts: list[PartConfig] = []
        for i, p in enumerate(data.get("parts") or []):
            kind = str(p.get("kind") or "cube")
            if kind not in PART_KINDS:
                raise SceneError(f"parts[{i}].kind must be one of {PART_KINDS}, got {kind!r}")
            material = str(p.get("material") or M.SHAPE_DEFAULT_MATERIAL[kind])
            if material not in M.PART_MATERIALS:
                raise SceneError(
                    f"parts[{i}].material must be one of {sorted(M.PART_MATERIALS)}, "
                    f"got {material!r}"
                )
            default_size = (DEFAULT_PART_SIZE_M,) * 3
            size = _triple(p.get("sizeM"), default_size, f"parts[{i}].sizeM")
            parts.append(
                PartConfig(
                    id=str(p.get("id") or f"part_{i + 1}"),
                    kind=kind,
                    material=material,
                    color=str(p.get("color") or "#e0483c"),
                    x=float(p.get("x", 0.35)),
                    y=float(p.get("y", -0.425)),
                    yaw_deg=float(p.get("yawDeg", 0.0)),
                    size_m=size,
                )
            )
        if len(parts) > MAX_PARTS:
            raise SceneError(f"at most {MAX_PARTS} parts, got {len(parts)}")

        bins: list[BinConfig] = []
        for i, b in enumerate(data.get("bins") or []):
            bins.append(
                BinConfig(
                    id=str(b.get("id") or f"bin_{i + 1}"),
                    label=str(b.get("label") or f"Destino {i + 1}"),
                    color=str(b.get("color") or "#4a7fb5"),
                    x=float(b.get("x", 0.35)),
                    y=float(b.get("y", -0.51)),
                    inner_m=_pair(b.get("innerM"), (0.075, 0.075), f"bins[{i}].innerM"),
                    wall_m=float(b.get("wallM", 0.006)),
                    height_m=float(b.get("heightM", 0.045)),
                    floor_m=float(b.get("floorM", 0.006)),
                    yaw_deg=float(b.get("yawDeg", 0.0)),
                )
            )

        mission_raw = data.get("mission") or {}
        tasks = [
            MoveTask(
                id=str(t.get("id") or f"task_{i + 1}"),
                part_id=str(t["partId"]),
                bin_id=str(t["binId"]),
                label=str(t.get("label") or ""),
            )
            for i, t in enumerate(mission_raw.get("tasks") or [])
            if t.get("partId") and t.get("binId")
        ]

        scene = cls(table=table, robots=robots, parts=parts, bins=bins,
                    mission=MissionConfig(tasks=tasks),
                    name=str(data.get("name") or "Celda Demo 12"))
        scene.validate()
        return scene

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        """Reject a scene that cannot be built, with a message naming what is wrong.

        This runs before Isaac Sim starts. A layout that will not work should cost the
        user a sentence, not a three-minute simulator boot and a stack trace.
        """
        errors = self.collect_errors()
        if errors:
            raise SceneError("; ".join(errors))

    def collect_errors(self) -> list[str]:
        """Every problem with this scene, so the interface can show them all at once."""
        errs: list[str] = []

        if not self.enabled_robots:
            errs.append("at least one robot must be enabled")

        ids: set[str] = set()
        for r in self.robots:
            if r.id in ids:
                errs.append(f"duplicate robot id {r.id!r}")
            ids.add(r.id)

        part_ids = {p.id for p in self.parts}
        bin_ids = {b.id for b in self.bins}

        if not self.bins:
            errs.append("the cell needs at least one destination bin")

        for p in self.parts:
            if not self.table.holds(p.x, p.y):
                errs.append(f"part {p.id} at ({p.x:.3f}, {p.y:.3f}) is off the table")
            for r in self.enabled_robots:
                if r.reach.contains((p.x, p.y)):
                    break
            else:
                errs.append(
                    f"part {p.id} at ({p.x:.3f}, {p.y:.3f}) is out of reach of every robot"
                )
            if not p.graspable:
                errs.append(
                    f"part {p.id} is {p.grasp_width_m * 1000:.0f} mm across; this gripper "
                    f"takes {M.MIN_GRASP_WIDTH_M * 1000:.0f}-"
                    f"{M.GRASP_WIDTH_LIMIT_M * 1000:.0f} mm parts"
                )

        for b in self.bins:
            if not self.table.holds(b.x, b.y):
                errs.append(f"bin {b.id} at ({b.x:.3f}, {b.y:.3f}) is off the table")
            if b.wall_m <= 0 or b.floor_m <= 0:
                errs.append(f"bin {b.id} needs a positive wall and floor thickness")
            if b.floor_m >= b.height_m:
                errs.append(f"bin {b.id} floor is thicker than the tray is tall")

        # Nothing may start overlapping something else: an interpenetrating pair is a
        # PhysX explosion on frame one, not a soft failure.
        for i, a in enumerate(self.parts):
            for b in self.parts[i + 1:]:
                gap = math.hypot(a.x - b.x, a.y - b.y) - (a.footprint_radius_m + b.footprint_radius_m)
                if gap < 0.005:
                    errs.append(
                        f"parts {a.id} and {b.id} overlap by {-gap * 1000:.0f} mm; "
                        f"move them at least 5 mm apart"
                    )
        for part in self.parts:
            for b in self.bins:
                if b.contains_footprint(part.x, part.y, part.footprint_radius_m):
                    errs.append(f"part {part.id} starts inside bin {b.id}")
                # A sphere rolls. Anything this close to a tray cavity will find its way in
                # during the first settle, and then be reported as sitting on the table
                # while it is physically on the tray floor.
                elif b.distance_from_center_m(part.x, part.y) - (
                        math.hypot(*b.inner_m) / 2.0 + part.footprint_radius_m
                ) < ROLL_CLEARANCE_M:
                    errs.append(
                        f"part {part.id} is within {ROLL_CLEARANCE_M * 1000:.0f} mm of bin "
                        f"{b.id} and may roll into it"
                    )

        for b in self.bins:
            for other in self.bins:
                if other.id <= b.id:
                    continue
                gap = math.hypot(b.x - other.x, b.y - other.y) - (
                    math.hypot(*b.outer_size_m) / 2.0 + math.hypot(*other.outer_size_m) / 2.0
                )
                if gap < 0.010:
                    errs.append(f"bins {b.id} and {other.id} are too close together")

        for t in self.mission.tasks:
            if t.part_id not in part_ids:
                errs.append(f"task {t.id} references unknown part {t.part_id!r}")
            if t.bin_id not in bin_ids:
                errs.append(f"task {t.id} references unknown bin {t.bin_id!r}")

        return errs


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_COLORS = ["#e0483c", "#3d7fd6", "#3aa76d", "#d98c1f"]


def default_scene(robot_count: int = 2) -> SceneConfig:
    """A cell that is physically valid as configured, used as the editor's starting point.

    The layout obeys the constraints ``validate`` enforces: everything on the table,
    everything inside some robot's reach envelope, parts and bins clear of each other,
    and parts small enough for a 36.2 mm gripper.
    """
    robots = [
        RobotConfig(
            id=f"robot_{i + 1}",
            label=f"SO-101 #{i + 1}",
            base_xy=(ROBOT_BASE_XS[i], ROBOT_BASE_Y),
            base_yaw_deg=180.0,
            enabled=i < robot_count,
        )
        for i in range(min(robot_count, MAX_ROBOTS))
    ]

    # Parts in the near row (y = -0.425), bins in the far row (y = -0.512).
    # Rows rather than columns because a 45 mm tall bin standing in the near row would
    # sit between the camera and everything behind it.
    #
    # Every x here is chosen so that each item is inside the reach annulus of *every*
    # enabled robot, which is what makes the cell reconfigurable: enabling or disabling a
    # robot never strands a part or a tray.
    parts = [
        PartConfig(id="part_1", kind="cube", material="abs", color=DEFAULT_COLORS[0],
                   x=0.290, y=-0.425),
        PartConfig(id="part_2", kind="cube", material="abs", color=DEFAULT_COLORS[1],
                   x=0.430, y=-0.425),
        PartConfig(id="part_3", kind="cylinder", material="abs", color=DEFAULT_COLORS[2],
                   x=0.360, y=-0.425),
        # The ball sits in the second row between the two trays, at 20 mm rather than
        # 30 mm. Both of those were found by running the scene rather than by choosing
        # them: a sphere on melamine rolls, and at 30 mm this one was 33 mm from a tray's
        # cavity edge and rolled 9.5 mm into it during the first settle -- ending up on
        # the tray floor while the telemetry still called it a table part. ``validate``
        # now enforces the margin instead of leaving it to be rediscovered each run.
        PartConfig(id="part_4", kind="ball", material="pet", color=DEFAULT_COLORS[3],
                   x=0.360, y=-0.472, size_m=(0.020, 0.020, 0.020)),
    ]

    bins = [
        BinConfig(id="bin_1", label="Rojo", color=DEFAULT_COLORS[0], x=0.255, y=-0.520),
        BinConfig(id="bin_2", label="Azul", color=DEFAULT_COLORS[1], x=0.455, y=-0.520),
    ]

    # Task by colour: the intuitive reading of a sorting cell.
    tasks: list[MoveTask] = []
    for i, p in enumerate(parts):
        match = next((b for b in bins if b.color.lower() == p.color.lower()), None)
        if match is None:
            continue
        tasks.append(MoveTask(
            id=f"task_{i + 1}", part_id=p.id, bin_id=match.id,
            label=f"{p.label_es} {p.id.split('_')[-1]} -> {match.label}",
        ))

    scene = SceneConfig(robots=robots, parts=parts, bins=bins, mission=MissionConfig(tasks))
    scene.validate()
    return scene


def _pair(value: Any, default: tuple[float, float], field: str) -> tuple[float, float]:
    if value is None:
        return default
    try:
        seq = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise SceneError(f"{field} must be two numbers") from exc
    if len(seq) != 2:
        raise SceneError(f"{field} must have exactly two values, got {len(seq)}")
    return (seq[0], seq[1])


def _triple(value: Any, default: tuple[float, float, float], field: str) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        seq = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise SceneError(f"{field} must be three numbers") from exc
    if len(seq) != 3:
        raise SceneError(f"{field} must have exactly three values, got {len(seq)}")
    if any(v <= 0 for v in seq):
        raise SceneError(f"{field} must be positive, got {seq}")
    return (seq[0], seq[1], seq[2])


def catalog() -> dict[str, Any]:
    """What the interface needs to offer its choices: part kinds, materials, colours.

    Sent once so the frontend never hard-codes a list that physics has to agree with.
    """
    return {
        "partKinds": [
            {"id": k,
             "labelEs": {"cube": "Cubo", "cylinder": "Cilindro", "ball": "Bola"}[k],
             "defaultMaterial": M.SHAPE_DEFAULT_MATERIAL[k]}
            for k in PART_KINDS
        ],
        "materials": [
            {
                "id": mat.key,
                "labelEs": mat.label_es,
                "labelEn": mat.label_en,
                "densityKgM3": mat.density_kg_m3,
                "staticFriction": mat.static_friction,
                "dynamicFriction": mat.dynamic_friction,
                "restitution": mat.restitution,
                "note": mat.note,
            }
            for mat in M.PART_MATERIALS.values()
        ],
        "colors": DEFAULT_COLORS,
        "limits": {
            "maxRobots": MAX_ROBOTS,
            "maxParts": MAX_PARTS,
            "defaultPartSizeM": DEFAULT_PART_SIZE_M,
            "workAreaX": list(WORK_AREA_X),
            "workAreaY": list(WORK_AREA_Y),
            "robotBaseY": ROBOT_BASE_Y,
            "robotBaseXs": list(ROBOT_BASE_XS),
            "tableTopZM": M.TABLE_TOP_Z_M,
            "gripperApertureM": M.GRIPPER_APERTURE_M,
            "minPartMm": M.MIN_GRASP_WIDTH_M * 1000,
            "maxPartMm": M.GRASP_WIDTH_LIMIT_M * 1000,
            "minReachM": M.MIN_REACH_M,
            "maxReachM": M.MAX_REACH_M,
        },
    }

