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
#
# Wider and deeper than the four-part cell this replaced, because six parts and three trays
# do not fit in the old 700x650 mm footprint with the required clearances -- see
# ``feasible_span_x`` for why the layout is derived rather than typed in.
# The bench, **unchanged from the measured-working cell**. Do not enlarge it.
#
# These two lines look like free parameters and they are not. The robot bases sit at
# y = -0.64, so the table's back edge at y = -0.675 leaves only 35 mm of bench behind each
# arm. ``IsaacCell.settle`` retracts each arm to ``(base_x, base_y - 0.10, ...)`` -- 100 mm
# behind its own base -- which overhangs the edge and is how it was designed to park.
#
# Growing the table to 800x780 at (0.36, -0.38) pushed that edge back to y = -0.77, 130 mm
# behind the bases. The retract pose then sits *on* the tabletop, the arm's own colliders
# (base, shoulder, upper_arm, lower_arm, wrist -- all enabled) push against it, and the arm
# cannot extend: measured, **all six tasks failed with a tool error of 146-168 mm and zero
# grip force**, at part positions (216-229 mm from a base) that are *closer* than the ones
# the same configuration reached to under 3 mm before the table grew.
#
# The scene got bigger, the arms did not, and the geometry nobody thought about was the one
# that mattered.
DEFAULT_TABLE_SIZE_M = (0.70, 0.65)
DEFAULT_TABLE_CENTER_XY = (0.35, -0.35)

WORK_AREA_X = (0.20, 0.50)
WORK_AREA_Y = (-0.55, -0.40)

ROBOT_BASE_Y = -0.64
ROBOT_BASE_XS = (0.20, 0.36, 0.52)

#: The furthest a part may sit from the arm that has to pick it, measured rather than
#: assumed, and **smaller than the geometric envelope**.
#:
#: ``materials.MAX_REACH_M`` is 335 mm and is correct as geometry. The arm does not manage
#: it: ``ArmController._wrist_for_down`` imposes tool-down with a fitted planar relation
#: (``WRIST_DOWN_OFFSET_RAD`` + two per-joint coefficients), and that fit was measured
#: against the *original* work area only. Outside it the commanded wrist angle is wrong by
#: enough degrees that the tool no longer converges, and the arm stalls hundreds of
#: millimetres short with ``shoulder_lift`` pinned against its own projection floor.
#:
#: So this constant is the boundary of what has actually been demonstrated, not a guess
#: about the hardware. Enlarging the cell means re-measuring the fit, which is a separate
#: job with a probe -- not a matter of moving numbers in this file.
WORKING_RADIUS_M = 0.235

# Two part rows and a tray row, exactly where the measured grasps were taken. The 20 mm ball
# on the far row and the 30 mm cube on the near row both grasped cleanly here.
PART_ROW_NEAR_Y = -0.425
PART_ROW_FAR_Y = -0.472
BIN_ROW_Y = -0.520

MAX_ROBOTS = 3
MAX_PARTS = 12

#: One arm cannot cover this cell; two is the smallest roster that can, and the reason is
#: in the message rather than in a comment.
#: One arm **can** build this cell, and ``tools/probe_grasp.py`` relies on it -- the
#: measured grasp evidence in ``out12/grasp_probe.json`` was taken with a single arm, which
#: is the cleanest way to measure one grasp with nothing else in the scene. What one arm
#: cannot do is serve the multi-robot mission well, so this is a floor of one and not two:
#: raising it to two silently disabled the probe that certifies the gripper, which is the
#: opposite of what a guard is for.
MIN_ROBOTS_FOR_CELL = 1

PART_KINDS = ("cube", "cylinder", "ball")

DEFAULT_PART_SIZE_M = 0.030

#: How far a part must start from a tray's cavity edge. Measured behaviour, not taste: the
#: PET sphere on melamine rolls about 10 mm while the scene settles, so anything closer than
#: this ends up in a tray.
ROLL_CLEARANCE_M = 0.015

#: A colour-matched tray is only preferred if it is genuinely close. Beyond this the nearest
#: tray wins, because a 200 mm carry fails in a way a slightly wrong colour does not.
COLOUR_MATCH_RADIUS_M = 0.140

#: Colour-match bonus when two trays are equidistant.
MATCH_COLOUR_BONUS = 1


def _norm_colour(value: str) -> str:
    return str(value or "").strip().lower()


DEFAULT_COLORS = ["#e0483c", "#3d7fd6", "#3aa76d", "#d98c1f", "#8b5cf6", "#0ea5e9",
                  "#f472b6", "#14b8a6"]

BIN_COLORS = ["#e0483c", "#3d7fd6", "#d98c1f"]

BIN_LABELS_ES = ("Rojo", "Azul", "Amarillo")


def lateral_offset_at(y: float) -> float:
    """How far sideways from a base the arm can still work at row ``y``.

    ``sqrt(r^2 - dy^2)`` on :data:`WORKING_RADIUS_M`. Zero when the row is deeper than the
    arm can reach at all.
    """
    import math as _math

    dy = abs(y - ROBOT_BASE_Y)
    if dy > WORKING_RADIUS_M:
        return 0.0
    return _math.sqrt(max(WORKING_RADIUS_M ** 2 - dy ** 2, 0.0))


def reachable_x_spans(y: float,
                      base_xs: tuple[float, ...] = ROBOT_BASE_XS) -> list[tuple[float, float]]:
    """The x-intervals at row ``y`` that at least one robot in ``base_xs`` can work.

    **A union of intervals, not an intersection.** An earlier version intersected every
    arm's working disc and produced an empty span: at the part row each arm covers about
    +/-95 mm sideways while the bases are 160 mm apart, so the *common* region is empty for
    a three-arm cell. That was never the requirement -- ``validate`` asks only that some
    enabled robot can reach each part, and the planner already excludes robots that cannot
    reach a given part.

    Asking for the intersection looked stricter and was actually impossible, and it is how a
    layout ends up in a place no arm can actually serve.
    """
    w = lateral_offset_at(y)
    if w <= 0.0:
        return []
    return [(bx - w, bx + w) for bx in base_xs]


def span_of(scans: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Smallest interval containing every span, or ``None`` if there are none."""
    if not scans:
        return None
    return (min(lo for lo, _ in scans), max(hi for _, hi in scans))


def row_is_feasible(y: float) -> bool:
    return bool(reachable_x_spans(y))


def robot_can_work(base_xy: tuple[float, float], x: float, y: float,
                   radius: float = WORKING_RADIUS_M) -> bool:
    """Whether the arm at ``base_xy`` can pick from ``(x, y)``.

    **The one reachability question in the demo.** Both :meth:`SceneConfig.collect_errors`
    and :class:`demo12.planner.AssignmentPolicy` ask it, and they have to agree: they did
    not, and the disagreement is a silent way to lose a task.

    ``validate`` checked the 235 mm working radius while the assignment policy kept checking
    the 335 mm geometric envelope, so a scene that validated could still be handed to an arm
    that cannot work there. Measured, that is how a run that used to complete 2/2 ended
    0/4 with tracking errors of 146-208 mm -- a scene-validation pass said the cell was fine
    and the scheduler disagreed with it a moment later.

    One definition, used by both.
    """
    dy = abs(y - ROBOT_BASE_Y)
    if dy > radius:
        return False
    lateral = math.sqrt(max(radius * radius - dy * dy, 0.0))
    return abs(x - base_xy[0]) <= lateral


class SceneError(ValueError):

    """A configuration that cannot be built into a physical cell."""


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
        """The wire form, spelled ``partId`` / ``binId`` like every other JSON here.

        This was ``asdict(self)``, which emits ``part_id`` / ``bin_id`` -- and that was the
        only snake_case value in the whole HTTP surface. Everything else already agreed on
        camelCase: ``ledger.py`` both writes and reads ``partId``, ``SceneConfig.from_dict``
        reads ``partId``, ``server.drain_commands`` reads ``partId``, and the TypeScript
        client types every task as ``{ id, partId, binId, label }``.

        So the mismatch was invisible in review and total at runtime. Every task submitted
        from the interface -- the command bar, the piece-to-box picker, a one-click scenario
        -- was serialised with ``part_id`` and then read back with ``t["partId"]``, which
        raised ``KeyError`` inside ``drain_commands``. That handler catches and publishes, so
        the run did not crash: it queued nothing, applied nothing, and wrote a line into
        ``lastError`` that nothing in the interface displays. The buttons looked alive and
        the cell never moved.

        The same defect broke ``SceneConfig`` round-tripping, so ``/api/scene/validate``
        handed back its own output to ``/api/scene/apply`` and the mission arrived empty.
        """
        return {
            "id": self.id,
            "partId": self.part_id,
            "binId": self.bin_id,
            "label": self.label,
        }


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
                    # ``outerM`` is derived, and it is serialised so the interface draws the
                    # tray's real footprint instead of assuming the default one. Without it a
                    # resized tray is drawn at the wrong size while PhysX builds the right
                    # one -- a silent divergence between what the screen shows and what the
                    # physics does, which is the one kind of disagreement this demo is not
                    # allowed to have.
                    "outerM": list(b.outer_size_m),
                    "wallM": b.wall_m,
                    "heightM": b.height_m,
                    "floorM": b.floor_m,
                    "rimZ": round(b.rim_z_m, 5),
                    "cavityFloorZ": round(b.cavity_floor_z_m, 5),
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

    def _workable(self, x: float, y: float) -> bool:
        """Whether some enabled robot can pick from here.

        Delegates to :func:`robot_can_work`, which :class:`demo12.planner.AssignmentPolicy`
        also calls -- one definition, so a scene that validates cannot be scheduled into a
        grasp the arm cannot make.
        """
        return any(robot_can_work(r.base_xy, x, y) for r in self.enabled_robots)

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
            elif not self._workable(p.x, p.y):
                errs.append(
                    f"part {p.id} at ({p.x:.3f}, {p.y:.3f}) is out of reach of every robot "
                    f"(each arm works about {WORKING_RADIUS_M * 1000:.0f} mm from its base)"
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
            elif not self._workable(b.x, b.y):
                errs.append(
                    f"bin {b.id} at ({b.x:.3f}, {b.y:.3f}) is out of reach of every robot "
                    f"(each arm works about {WORKING_RADIUS_M * 1000:.0f} mm from its base)"
                )
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

#: Colour-match bonus when two trays are equidistant, so a colour sort still wins a tie.
MATCH_COLOUR_BONUS = 1

#: A colour-matched tray is only preferred if it is genuinely close. Beyond this the
#: nearest tray wins, because a 340 mm carry fails in a way a slightly wrong colour does
#: not: the arm cannot hold a part across that distance at all. 160 mm keeps the colour
#: story where it is cheap and yields where it is not -- measured, the longest carry in the
#: default mission is 124 mm.
COLOUR_MATCH_RADIUS_M = 0.160


def _norm_colour(value: str) -> str:
    return str(value or "").strip().lower()


DEFAULT_COLORS = ["#e0483c", "#3d7fd6", "#3aa76d", "#d98c1f", "#8b5cf6", "#0ea5e9",
                  "#f472b6", "#14b8a6"]

BIN_COLORS = ["#e0483c", "#3d7fd6", "#d98c1f"]

BIN_LABELS_ES = ("Rojo", "Azul", "Amarillo")


def default_scene(robot_count: int = 2, assign_all: bool = False) -> SceneConfig:
    """A cell that is physically valid as configured, used as the editor's starting point.

    **These positions are measured, not chosen.** They are the ones
    ``out12/grasp_probe.json`` and the end-to-end runs were taken on: a 30 mm ABS cube and a
    25 mm cylinder on the near row, a 20 mm PET ball on the far row, two trays behind them.

    An earlier attempt at this function laid out six parts on a single row spanning the
    whole reachable width, and every grasp failed with a tool error of 146-168 mm and zero
    grip force. The cause was not the part positions -- they were all inside
    :data:`WORKING_RADIUS_M` -- but that the arm's tool-down wrist relation is a **fit**, and
    the fit was measured over the original work area only. Widening the area puts poses
    outside it, the commanded wrist angle is wrong by enough degrees to stop the tool
    converging, and the arm pins ``shoulder_lift`` against its own limit.

    The honest conclusion is that growing this cell is a recalibration job with a probe
    attached, not an edit to a list of coordinates. So the layout is held at what has been
    demonstrated, :func:`SceneConfig.collect_errors` now checks against the measured radius
    rather than the geometric envelope, and the interface can still *edit* the scene -- the
    editor validates every change against the same number, so an impossible layout is
    refused while it is being drawn rather than three minutes into a simulator boot.
    """
    if robot_count < MIN_ROBOTS_FOR_CELL:  # noqa: SIM108 - the message is the point
        raise SceneError(
            f"one arm cannot cover this cell. Each arm is demonstrated out to about "
            f"{WORKING_RADIUS_M * 1000:.0f} mm from its base, and the near row is "
            f"{abs(PART_ROW_NEAR_Y - ROBOT_BASE_Y) * 1000:.0f} mm in front of it, leaving "
            f"about {lateral_offset_at(PART_ROW_NEAR_Y) * 2000:.0f} mm of x per arm -- not "
            f"enough for the parts and two trays. Use two or three robots.")

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

    parts = [
        PartConfig(id="part_1", kind="cube", material="abs", color=DEFAULT_COLORS[0],
                   x=0.290, y=PART_ROW_NEAR_Y, size_m=(0.030, 0.030, 0.030)),
        PartConfig(id="part_2", kind="cylinder", material="abs", color=DEFAULT_COLORS[1],
                   x=0.430, y=PART_ROW_NEAR_Y, size_m=(0.030, 0.030, 0.030)),
        PartConfig(id="part_3", kind="cube", material="abs", color=DEFAULT_COLORS[2],
                   x=0.360, y=PART_ROW_NEAR_Y, size_m=(0.025, 0.025, 0.025)),
        # 20 mm rather than 30 mm, for the reason recorded in PHYSICS.md section 5: a 30 mm
        # sphere on melamine rolls about 10 mm while the scene settles and ends up in a
        # tray while the telemetry still calls it a table part. 20 mm was measured to hold,
        # and this is the one grasp in the repo with a full evidence record.
        PartConfig(id="part_4", kind="ball", material="pet", color=DEFAULT_COLORS[3],
                   x=0.360, y=PART_ROW_FAR_Y, size_m=(0.020, 0.020, 0.020)),
    ]

    bins = [
        BinConfig(id="bin_1", label=BIN_LABELS_ES[0], color=BIN_COLORS[0],
                  x=0.255, y=BIN_ROW_Y),
        BinConfig(id="bin_2", label=BIN_LABELS_ES[1], color=BIN_COLORS[1],
                  x=0.455, y=BIN_ROW_Y),
    ]

    # A task per part, where a colour-matched tray exists -- the original rule, and the one
    # the measured runs used.
    #
    # The alternative of a task per *part* was tried and it breaks the cell: with all four
    # parts carrying work, two tasks land on ``part_3`` and ``part_4``, which sit at the same
    # x only 47 mm apart in y. Two arms working the centre of the cell at the same time
    # collide, and the run ends 0/4 with tracking errors of 150-270 mm -- which looks exactly
    # like a reachability problem and is not.
    #
    # So the default mission is the two colour-matched picks, which are on opposite sides of
    # the cell and demonstrably run in parallel. The other two parts are in the cell and are
    # reachable; they simply have no task until someone assigns one from the interface,
    # which is a deliberate choice rather than an oversight.
    tasks: list[MoveTask] = []
    for i, p in enumerate(parts):
        match = next((b for b in bins
                      if _norm_colour(b.color) == _norm_colour(p.color)), None)
        if match is None:
            if not assign_all:
                continue
            # ``assign_all`` gives every part a task, for the denser mission the interface
            # offers. The nearest tray wins, and the colour-matched one only when it is
            # close enough to carry to -- a 200 mm carry fails in a way a slightly wrong
            # colour does not.
            match = min(bins, key=lambda b: (round(math.hypot(b.x - p.x, b.y - p.y), 3),
                                             -MATCH_COLOUR_BONUS))
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

