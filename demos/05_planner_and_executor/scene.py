"""What perception believes is on the table: objects, containers, and where they are.

This module is the contract between the camera half of the demo and everything
downstream. Demo 09 writes a scene as JSON; demo 10 grounds a command in it, demo
11 uses it to place a grasp, and demo 12 shows it on the dashboard. None of them
import Isaac Sim, and neither does this file, so the same scene JSON is produced by
the simulator, by a webcam and by a folder of saved frames.

Two coordinate spaces are supported and the scene says which one it is in:

* ``world``, in metres, when a camera pose and a table height are known. Positions
  are then real and the arm can be driven to them.
* ``image``, normalised to 0..1, when only a picture is available. Left and right
  still mean something there, which is all that grounding needs, but "nearest" is
  measured from the bottom of the image and is therefore a weaker claim. The space
  is recorded in the JSON so nothing downstream has to guess.

Camera geometry follows Isaac Lab exactly: a pixel plus an orthogonal depth ``d``
gives the camera-frame point ``K^-1 [u, v, 1] d``, and the world point is
``R(quat) p_cam + pos``. That is the same pair of formulas as
``isaaclab.utils.math.unproject_depth`` and ``transform_points``, so the projection
here is the exact inverse of what the simulator renders and the numbers can be
checked against it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

# Names used by demo 03's command model for the kind of an item, kept identical so
# that grounding is a lookup and not a translation. See vocabulary.yaml `objects`.
KINDS = ("ball", "cube", "any", "other")

# The words demo 03 puts in an item's `position` field, which decide which of
# several matching objects is meant.
POSITIONS = ("left", "right", "front", "back", "nearest", "farthest")


def matrix_to_quat(matrix: list[list[float]] | np.ndarray) -> list[float]:
    """A rotation matrix to a (w, x, y, z) quaternion, for building test poses.

    Only used by the offline checks and by the camera placement, where a rotation is
    easier to reason about as three axes than as four numbers.
    """
    m = np.asarray(matrix, dtype=float)[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2
        return [float(v) for v in (0.25 * s, (m[2, 1] - m[1, 2]) / s,
                                   (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)]
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    quat = [0.0, 0.0, 0.0, 0.0]
    quat[0] = (m[k, j] - m[j, k]) / s
    quat[i + 1] = 0.25 * s
    quat[j + 1] = (m[j, i] + m[i, j]) / s
    quat[k + 1] = (m[k, i] + m[i, k]) / s
    return quat


def quat_to_matrix(quat_wxyz: list[float] | tuple[float, ...]) -> np.ndarray:
    """A rotation matrix from a (w, x, y, z) quaternion, as Isaac Lab orders them."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("a zero quaternion has no rotation")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class CameraModel:
    """Everything needed to turn a pixel into a point on the table.

    ``matrix`` is the 3x3 intrinsic matrix Isaac Lab builds from the camera config,
    so no focal-length arithmetic has to be repeated here.
    """

    pos_w: list[float]
    quat_wxyz: list[float]
    matrix: list[list[float]]
    width: int
    height: int

    @property
    def k(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.quat_wxyz)

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.pos_w, dtype=float)

    def to_camera(self, point: list[float] | np.ndarray) -> np.ndarray | None:
        """A world point in camera coordinates, or None if it is behind the camera.

        The convention is **not** the one in Isaac Lab's ``Camera._update_poses``
        docstring (front +Z, up -Y). It was measured against rendered objects instead,
        and both lateral signs are negative:

        * the camera looks along its own **+x**, so x is the depth;
        * **-y is image right**, so u = cx - fx * y / x;
        * **-z is image down**, so v = cy - fy * z / x.

        Negating both lateral axes is the same as turning the image 180 degrees, so it
        is easy to get wrong in a way that looks fine: with the opposite signs every
        object still landed comfortably in frame, because the layout is compact and
        centred. What gave it away was comparing against the *measured* blob centroids
        of the same frame, where each object was 130 px off on average and
        ``640 - u, 480 - v`` put it exactly on top of the blob (median 1.9 px over all
        six objects). ``--check-convention`` draws the truth positions on the overlay;
        reading that against the detections is what settles the signs, and checking
        only that nothing left the frame is not enough.
        """
        target = np.asarray(point, dtype=float).reshape(3)
        camera = self.rotation.T @ (target - self.origin)
        if camera[0] <= 1e-6:  # behind the camera, or exactly on its plane
            return None
        return camera

    def ray(self, u: float, v: float) -> np.ndarray:
        """The ray through a pixel, in camera coordinates, with x = 1 forward.

        The two lateral signs are negative to match ``to_camera``: they are the exact
        inverse of ``project``, which is what ``smoke_perception`` checks.
        """
        return np.array([1.0, -(float(u) - self.k[0, 2]) / self.k[0, 0],
                         -(float(v) - self.k[1, 2]) / self.k[1, 1]])

    def project(self, point: list[float] | np.ndarray) -> tuple[float, float] | None:
        """Where a world point lands in the image, or None if it is behind the camera."""
        camera = self.to_camera(point)
        if camera is None:
            return None
        u = self.k[0, 2] - self.k[0, 0] * camera[1] / camera[0]
        v = self.k[1, 2] - self.k[1, 1] * camera[2] / camera[0]
        return float(u), float(v)

    def on_plane(self, u: float, v: float, plane_z: float) -> list[float] | None:
        """The point where the ray through a pixel meets a horizontal plane.

        Every object in this demo rests on the table, so one ray and one known plane
        height is enough to place it. This is the method that also works on a webcam,
        where there is no depth buffer at all.
        """
        direction = self.rotation @ self.ray(u, v)
        if abs(direction[2]) < 1e-9:
            return None
        scale = (plane_z - self.origin[2]) / direction[2]
        if scale <= 0.0:
            return None
        return list(self.origin + scale * direction)

    def from_depth(self, u: float, v: float, depth: float) -> list[float] | None:
        """The point a pixel with a distance-to-camera reading maps to."""
        if depth <= 0.0 or not np.isfinite(depth):
            return None
        camera = self.ray(u, v) * depth
        return list(self.rotation @ camera + self.origin)

    def pixels_per_m(self, plane_z: float, at: tuple[float, float] = (0.375, -0.41)) -> float | None:
        """Ground resolution near a table point, measured by projecting two points.

        Measured rather than derived from the focal length, because a tilted camera
        foreshortens along one direction and the number a judge asks about ("how many
        pixels is a centimetre") is not the same in both axes.
        """
        here = self.project([at[0], at[1], plane_z])
        there = self.project([at[0] + 0.1, at[1], plane_z])
        if here is None or there is None:
            return None
        return float(np.hypot(there[0] - here[0], there[1] - here[1]) / 0.1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CameraModel:
        return cls(**data)


@dataclass
class Frame:
    """What "left", "front" and "nearest" mean for this scene.

    The axes are 2D vectors in whatever space the scene's anchors live in, so the
    ranking code does not need to know whether that space is metres or pixels. In the
    simulator they are read off the robot's own pose rather than assumed: with the
    SO-101 at (0.35, -0.64) facing +y, the robot's left is -x.
    """

    space: str
    left_axis: tuple[float, float]
    front_axis: tuple[float, float]
    robot_xy: tuple[float, float]

    @classmethod
    def from_robot(cls, robot_xy: tuple[float, float], left_axis, front_axis) -> Frame:
        """Build a world-space frame from the robot's base position and heading."""
        left = np.asarray(left_axis, dtype=float)[:2]
        front = np.asarray(front_axis, dtype=float)[:2]
        for name, axis in (("left_axis", left), ("front_axis", front)):
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                raise ValueError(f"{name} is a zero vector; the robot's pose is degenerate")
        return cls("world", tuple(left), tuple(front), robot_xy)

    @classmethod
    def from_image(cls, width: int, height: int) -> Frame:
        """The frame to assume when there is no camera pose at all.

        It assumes the picture was taken from behind the table looking across it,
        which is where a person stands when they point at something and say "that
        one". With that assumption, image-left is robot-left and the bottom of the
        image is the near edge, so the axes are -x and +y in normalised coordinates.
        """
        return cls("image", (-1.0, 0.0), (0.0, 1.0), (0.5, 1.0))

    def to_dict(self) -> dict:
        return {
            "space": self.space,
            "left_axis": list(self.left_axis),
            "front_axis": list(self.front_axis),
            "robot_xy": list(self.robot_xy),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Frame:
        return cls(
            space=data["space"],
            left_axis=tuple(data["left_axis"]),
            front_axis=tuple(data["front_axis"]),
            robot_xy=tuple(data["robot_xy"]),
        )


@dataclass
class Thing:
    """One detected thing. Objects and containers are the same record plus a kind."""

    id: int
    kind: str
    color: str
    centroid_px: list[float]
    bbox_px: list[int]
    area_px: int
    mean_hsv: list[int]
    confidence: float
    source: str
    track_id: int | None = None
    pos_m: list[float] | None = None
    pos_m_depth: list[float] | None = None
    inner_rect_m: list[float] | None = None
    elongation: float | None = None
    fill: float | None = None
    hollow: bool | None = None
    footprint_m: list[float] | None = None
    base_centre_m: list[float] | None = None
    entity_id: str | None = None
    display_name: str | None = None
    size_m: list[float] | None = None
    cavity_m: list[float] | None = None
    aim_xy: list[float] | None = None

    def anchor(self, width: int, height: int) -> np.ndarray:
        """A 2D position in the scene's space: metres if known, else the pixel."""
        if self.pos_m is not None:
            return np.asarray(self.pos_m[:2], dtype=float)
        return np.array([self.centroid_px[0] / width, self.centroid_px[1] / height])

    def matches(self, kind: str | None, color: str | None) -> bool:
        """Whether this thing answers to one of demo 03's item descriptions.

        ``any`` matches everything and ``other`` matches anything perception did not
        call a ball or a cube, which is how a named thing that only an open-vocabulary
        detector could name is handled without changing this code.
        """
        if kind in (None, "", "any"):
            kind_ok = True
        elif kind in ("ball", "cube"):
            kind_ok = self.kind == kind
        else:  # "other", and anything unexpected from the model
            kind_ok = self.kind not in ("ball", "cube")
        return kind_ok and (color in (None, "") or self.color == color)

    def to_dict(self) -> dict:
        data = asdict(self)
        for optional in ("inner_rect_m", "track_id", "pos_m", "pos_m_depth", "elongation", "fill", "hollow", "footprint_m", "base_centre_m", "entity_id", "display_name", "size_m", "cavity_m", "aim_xy"):
            if getattr(self, optional) is None:
                del data[optional]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Thing:
        data = dict(data)
        for optional in ("inner_rect_m", "track_id", "pos_m", "pos_m_depth", "elongation", "fill", "hollow", "footprint_m", "base_centre_m", "entity_id", "display_name", "size_m", "cavity_m", "aim_xy"):
            data.setdefault(optional, None)
        return cls(**data)


@dataclass
class Scene:
    """What is on the table, as perception believes it at one instant."""

    frame: Frame
    source: str
    width: int = 0
    height: int = 0
    objects: list[Thing] = field(default_factory=list)
    containers: list[Thing] = field(default_factory=list)
    camera: dict | None = None
    ground_px_per_m: float | None = None
    plane_z: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def everything(self) -> list[Thing]:
        return self.objects + self.containers

    def anchor_of(self, thing: Thing) -> np.ndarray:
        """A thing's 2D position, falling back to normalised pixels when it has none."""
        return thing.anchor(self.width or 1, self.height or 1)

    def candidates(self, kind: str | None, color: str | None) -> list[Thing]:
        """Objects matching an item description, largest first.

        Only objects, never containers: "put the red cube in the tray" must not be
        able to pick the tray.
        """
        found = [o for o in self.objects if o.matches(kind, color)]
        return sorted(found, key=lambda o: -o.area_px)

    def container(self, kind: str | None, color: str | None) -> list[Thing]:
        """Containers matching a place description, largest first."""
        found = [c for c in self.containers if c.matches(kind, color)]
        return sorted(found, key=lambda c: -c.area_px)

    def score(self, thing: Thing, position: str) -> float:
        """One number for a thing under one position word, along that word's own axis.

        Everything about left/right/front/back/nearest is decided here, so the ranking
        and the "does this word actually tell them apart" test can never disagree about
        which axis a word uses. Comparing the two axes together would make "left" claim
        to separate two objects that differ only in depth.
        """
        point = self.anchor_of(thing)
        if position in ("left", "right"):
            return float(np.dot(point, np.asarray(self.frame.left_axis, dtype=float)))
        if position in ("front", "back"):
            return float(np.dot(point, np.asarray(self.frame.front_axis, dtype=float)))
        if position in ("nearest", "farthest"):
            return float(np.linalg.norm(point - np.asarray(self.frame.robot_xy, dtype=float)))
        raise ValueError(f"unknown position word {position!r}; expected one of {POSITIONS}")

    def rank(self, things: list[Thing], position: str | None) -> list[Thing]:
        """Order things by a position word, best first.

        ``left`` is smallest along the robot's left axis, so it is the thing nearest
        the robot's own left hand, whatever direction the camera happens to face.
        """
        if not things or position in (None, ""):
            return list(things)
        ordered = sorted(things, key=lambda thing: self.score(thing, position))
        if position in ("right", "back", "farthest"):
            ordered.reverse()
        return ordered

    def separates(self, things: list[Thing], position: str) -> bool:
        """Whether a position word really tells these things apart along its own axis."""
        if len(things) < 2 or position not in POSITIONS:
            return False
        return abs(self.score(things[0], position) - self.score(things[-1], position)) > 1e-9

    def disambiguate(self, kind: str | None, color: str | None, position: str | None) -> list[Thing]:
        """Apply a position word to the matching objects."""
        return self.rank(self.candidates(kind, color), position)

    def relations(self) -> list[dict]:
        """Left/right/front/back between things that could be confused with each other.

        Only pairs that share a kind and a colour are reported, because those are the
        only pairs a person ever has to disambiguate. Reporting every pair would put
        O(n^2) entries in a file nobody reads.
        """
        out: list[dict] = []
        groups: dict[tuple[str, str], list[Thing]] = {}
        for thing in self.objects:
            groups.setdefault((thing.kind, thing.color), []).append(thing)
        for members in groups.values():
            if len(members) < 2:
                continue
            for first in members:
                for second in members:
                    if first.id >= second.id:
                        continue
                    for predicate, axis in (
                        ("left_of", self.frame.left_axis),
                        ("front_of", self.frame.front_axis),
                    ):
                        a = float(np.dot(self.anchor_of(first), np.asarray(axis, dtype=float)))
                        b = float(np.dot(self.anchor_of(second), np.asarray(axis, dtype=float)))
                        if abs(a - b) <= 1e-9:
                            continue
                        near, far = (first, second) if a < b else (second, first)
                        opposite = predicate.replace("left", "right").replace("front", "back")
                        out.append({"subject": near.id, "predicate": predicate, "object": far.id})
                        out.append({"subject": far.id, "predicate": opposite, "object": near.id})
        return sorted(out, key=lambda r: (r["subject"], r["predicate"]))

    def summary(self) -> str:
        """One line per thing, for the terminal."""
        lines = [f"scene from {self.source} ({self.frame.space} frame, {len(self.objects)} objects, "
                 f"{len(self.containers)} containers)"]
        for thing in self.everything:
            where = f"{thing.pos_m[0]:+.3f}, {thing.pos_m[1]:+.3f}" if thing.pos_m else (
                f"px {thing.centroid_px[0]:.0f}, {thing.centroid_px[1]:.0f}"
            )
            lines.append(
                f"  #{thing.id:<2} {thing.kind:<5} {thing.color:<6} {where}  "
                f"area={thing.area_px:<6} conf={thing.confidence:.2f} [{thing.source}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema": "leisaac-demo/perception/scene/1",
            "source": self.source,
            "frame": self.frame.to_dict(),
            "width": self.width,
            "height": self.height,
            "camera": self.camera,
            "plane_z": self.plane_z,
            "ground_px_per_m": self.ground_px_per_m,
            "objects": [o.to_dict() for o in self.objects],
            "containers": [c.to_dict() for c in self.containers],
            "relations": self.relations(),
            "notes": self.notes,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        return cls(
            frame=Frame.from_dict(data["frame"]),
            source=data["source"],
            width=data.get("width", 0),
            height=data.get("height", 0),
            objects=[Thing.from_dict(o) for o in data.get("objects", [])],
            containers=[Thing.from_dict(c) for c in data.get("containers", [])],
            camera=data.get("camera"),
            ground_px_per_m=data.get("ground_px_per_m"),
            plane_z=data.get("plane_z"),
            notes=data.get("notes", []),
        )

    @classmethod
    def read(cls, path: str | Path) -> Scene:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
