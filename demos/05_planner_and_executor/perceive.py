"""A camera image becomes a scene: colour masks, blobs, kinds, positions, objects.

The whole file is deliberately free of Isaac Sim and of torch, for two reasons.
The perception half of the demo then runs on the hackathon laptop even if the
simulator is broken, and the same code path is exercised by the simulator, by a
webcam and by a folder of saved frames, so what is tuned on one is what runs on the
others.

The pipeline, and why each step is there:

1. **HSV, not RGB.** The dome light is intensity 1000, so brightness varies with
   facing and distance. Hue survives that; saturation is what separates a painted
   object from the light wood of the table.
2. **One mask per named colour**, from ``colors.COLORS``. Overlapping bands are
   reported by ``colors.overlapping`` rather than left to cause a wrong detection.
3. **A morphological opening**, because two objects of the same colour touching merge
   into one blob and the demo then claims there is one thing when there are two.
4. **Connected components**, which give area, bounding box and centroid for free.
5. **Shape from the second moments of the mask**, not from the bounding box: a square
   rotated inside its own bounding box has an extent anywhere between 0.5 and 1.0, so
   the extent test cannot tell a cube from a ball, while the second moments of a disc
   are isotropic and equal ``r^2/4`` for any orientation.
6. **Position from one ray and the table plane.** Every object rests on the table, so
   knowing the camera pose and the table height is enough, and it is the method that
   also works on a webcam with no depth sensor. Where a depth image is available the
   result is computed a second way and stored beside the first, because the gap
   between them is a measurement worth having in the README.

Known weak spot, and it is measured rather than hidden: telling a cube from a ball is
the least reliable step. A disc and a square differ by only about 4.5% in the fill
ratio used below, and at the top camera's resolution a 3 cm cube is around 36 pixels
across, so the second moments are noisy. ``--trials`` scores kind accuracy against
the simulation's own ground truth, and ``"any"`` is a legal answer so that grounding
falls back on colour and position instead of trusting a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import colors
from scene import CameraModel, Frame, Scene, Thing

# Below this area the second moments are too noisy to call a shape, so the kind is
# still reported but with its confidence cut. 400 px is about a 2.3 cm blob at 1206
# px/m, which is a little under the smallest object the scene spec spawns.
KIND_MIN_AREA_PX = 400

# A disc has both second moments equal, and its extent is pi/4 of its bounding box; a
# square rotated by any angle has equal moments too but an extent from 0.5 to 1.0, so
# the extent test cannot tell a cube from a ball while the second moments can.
#
# The fill ratio is what separates them, and the numbers are measured over 20 rendered
# trials rather than reasoned from ideal shapes, because the mask is thresholded and
# antialiased and an ideal square is not what a cube's silhouette looks like.
#
#     3 cm cubes, 80 blobs:  fill 0.9562 .. 0.9780   elong 0.739 .. 0.863
#     4 cm balls, 80 blobs:  fill 0.9985 .. 1.0001   elong 0.961 .. 0.998
#
# The two ranges do not touch, so any threshold in (0.9780, 0.9985) separates them.
# 0.985 sits in the middle of that gap. An ideal disc gives 0.9988-1.000 and an ideal
# square 0.954-0.957, which is the same story: it is the antialiased thresholded cube,
# not the geometry, that needs the room.
#
# The old value was 0.975, which is *below* the top of the cube range and so called the
# largest blue cube a ball. 0.985 is not a fudge to hide that: it is in the middle of a
# gap that 160 measured blobs agree on. If a future scene moves the cubes closer to the
# camera, re-measure this rather than nudging it.
BALL_ELONGATION = 0.80
BALL_FILL = 0.985
BOX_ELONGATION = 0.55

# What a square measures, kept as its own constant because confidence_of scores
# distance from either of the two expected values, and because it is the number to
# check after changing BALL_FILL.
CUBE_FILL = 0.954

# Fraction of a container's bounding box taken to be wall rather than interior, applied
# to each side of each dimension, so the interior is (1 - 2 * this) of the footprint.
#
# MEASURED, AND FOUND NOT TO BE ENOUGH ON ITS OWN.
#
# History, because the gap between "declared" and "used" here caused a bug. Demo 09
# declared this constant, documented the intent, and then never applied it: it returned the
# outer footprint, which is why its own readme could promise that demo 10 would consume
# inner_rect_m while the value pointed at the outside edge of the container.
#
# With 0.12 the interior comes out 0.76 of the footprint. The trays in this directory have
# a cavity/footprint ratio of 0.800, which would want 0.100. But the detected footprint is
# not the declared footprint: a 7 cm tray 5 cm tall, seen from 67 degrees, has a colour
# blob whose bounding box came to 8.0 x 8.0 cm, inflated by the tray's height as well as
# its footprint. Measured against the truth (results/tray_check.md), inner_rect_m came out
# 4.8 cm for a 5.6 cm cavity on one tray and 6.0 x 6.9 cm on the other - right on one side,
# 1.3 cm wrong on the other.
#
# So this constant is a reasonable estimate and is not the placement mechanism. The
# executor aims at the container's ``pos_m`` and uses the DECLARED cavity size from
# scene_spec, then verifies where the object actually came to rest. That was measured to
# work: dropping at the perceived centre, 9-11 mm out, lands inside both cavities.
WALL_INSET_FRACTION = 0.12

# A container's footprint this close to square is called a box. A 10 x 8 cm tray is
# 1.25, an 8 x 8 cm box is 1.0. Nothing is called a tray or a basket from shape in
# this version: with three containers on a table the colour is what a person uses,
# and guessing "tray" from a rectangle would be inventing a class.
BOX_ASPECT = 1.25


@dataclass
class Blob:
    """One connected component of one colour mask."""

    color: str
    area_px: int
    bbox_px: list[int]
    centroid_px: list[float]
    mean_hsv: list[int]
    elongation: float
    fill: float
    mask: np.ndarray = field(repr=False, default=None)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 to HSV uint8, the only conversion this file needs."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def shape_of(mask: np.ndarray) -> tuple[float, float]:
    """Elongation and fill ratio of a binary mask, from its second moments.

    Elongation is the square root of the ratio of the smaller to the larger moment, so
    it is 1 for a disc or a square in any rotation and falls towards 0 for a thin
    sliver. Fill is the area over the area of the ellipse with the same moments, which
    is 1 for a disc and about 0.954 for a square.

    ``cv2.moments`` returns the *raw* central moments, which are the variance times the
    area, so they are divided by ``m00`` before being read as variances. Without that
    division both ratios come out around 0.0007 and every blob looks like a sliver.
    """
    binary = (mask > 0).astype(np.uint8)
    moments = cv2.moments(binary, binaryImage=True)
    area = moments["m00"]
    if area <= 0:
        return 0.0, 0.0
    mu20 = moments["mu20"] / area
    mu02 = moments["mu02"] / area
    mu11 = moments["mu11"] / area
    # eigenvalues of the 2x2 covariance matrix
    half = 0.5 * (mu20 + mu02)
    spread = np.sqrt(max(0.0, half * half - (mu20 * mu02 - mu11 * mu11)))
    big, small = half + spread, max(half - spread, 1e-9)
    elongation = float(np.sqrt(small / big))
    a, b = 2.0 * np.sqrt(big), 2.0 * np.sqrt(small)
    return elongation, float(area / (np.pi * a * b))


def kind_of(elongation: float, fill: float) -> str:
    """Call a shape a ball, a cube, or admit it is neither."""
    if elongation >= BALL_ELONGATION and fill >= BALL_FILL:
        return "ball"
    if elongation >= BOX_ELONGATION:
        return "cube"
    return "any"


def blobs_of(hsv: np.ndarray, color: colors.Color, opening_px: int) -> list[Blob]:
    """Every blob of one colour, after opening the mask to separate touching objects."""
    mask = colors.mask_of(hsv, color)
    if opening_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opening_px + 1,) * 2)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    found: list[Blob] = []
    total, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    for label in range(1, total):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < colors.MIN_AREA_PX:
            continue
        blob_mask = labels == label
        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        mean = cv2.mean(hsv, mask=blob_mask.astype(np.uint8))[:3]
        elongation, fill = shape_of(blob_mask)
        found.append(
            Blob(
                color=color.name,
                area_px=area,
                bbox_px=[x, y, w, h],
                centroid_px=[float(centroids[label][0]), float(centroids[label][1])],
                mean_hsv=[int(round(v)) for v in mean],
                elongation=elongation,
                fill=fill,
                mask=blob_mask,
            )
        )
    return sorted(found, key=lambda b: -b.area_px)


def confidence_of(blob: Blob, distance_to_edge: float) -> float:
    """How much to trust a blob, from its shape purity and its size.

    A blob that fills its own ellipse (a ball) or sits well clear of the frame edge
    scores high; one that is clipped by the frame, too small to measure, or neither
    round nor square scores low. The number is reported, not thresholded, so that the
    evaluator can show which detections it did not trust.
    """
    score = 1.0
    purity = min(abs(blob.fill - BALL_FILL), abs(blob.fill - CUBE_FILL)) / 0.05
    score *= float(np.clip(0.55 + 0.45 * min(1.0, purity), 0.0, 1.0))
    if blob.area_px < KIND_MIN_AREA_PX:
        score *= 0.6
    score *= float(np.clip(1.15 - distance_to_edge, 0.0, 1.0))
    return round(float(np.clip(score, 0.05, 0.99)), 3)


def distance_to_edge(centroid: tuple[float, float], width: int, height: int) -> float:
    """How far a centroid is from the frame border, as a fraction of the half-diagonal."""
    half = 0.5 * float(np.hypot(width, height))
    return min(centroid) / half if half else 0.0


def base_band(camera: CameraModel, blob: Blob, depth: np.ndarray,
              tol: float = 0.004, trim: float = 2.0):
    """The blob's lowest points in world coordinates, as an (N, 3) array.

    Shared by ``base_centre`` and ``base_footprint``. Returns None without a usable depth
    image.
    """
    ys, xs = np.nonzero(blob.mask)
    if len(xs) < 40:
        return None
    readings = np.asarray(depth, dtype=np.float64)
    if readings.ndim != 2:
        return None
    readings = readings[ys, xs]
    ok = np.isfinite(readings) & (readings > 0)
    if int(ok.sum()) < 40:
        return None
    xs, ys, readings = xs[ok], ys[ok], readings[ok]

    # from_depth, vectorised: camera.ray puts x = 1 forward, so the camera-frame point is
    # the ray scaled by the reading, and the world point is that rotated into the world.
    rays = np.stack([
        np.ones_like(xs, dtype=float),
        -(xs - camera.k[0, 2]) / camera.k[0, 0],
        -(ys - camera.k[1, 2]) / camera.k[1, 1],
    ], axis=-1)
    points = (camera.rotation @ (rays * readings[:, None]).T).T + camera.origin

    lowest = float(np.percentile(points[:, 2], trim))
    band = points[points[:, 2] <= lowest + tol]
    if len(band) < 40:
        band = points[points[:, 2] <= lowest + 2.0 * tol]
    return band if len(band) >= 40 else None


def base_centre(camera: CameraModel, blob: Blob, depth: np.ndarray) -> list[float] | None:
    """A container's centre on the table, from the floor pixels rather than the blob's middle.

    A container's colour blob is not its footprint. A 6 cm tray 5 cm tall, seen from 67
    degrees, presents its rim, the inside of its far wall and part of its floor, and the
    blob's area centroid sits off the tray's centre because that presentation is not
    symmetric about it. Measured against ground truth on the two trays in this directory:
    the plane-ray estimate ``pos_m`` lands 6 mm and 12 mm away from the truth.

    The floor is the part of the blob that lies where the tray really is: it is the lowest
    surface, it sits at a known height, and its pixels back-project exactly. Taking the
    MEDIAN of the lowest band rather than its bounding box keeps stray boundary pixels -
    where the depth reading straddles the tray and the table beside it - from dragging the
    answer.

    Returns None without a depth image; the caller falls back to ``pos_m``.
    """
    band = base_band(camera, blob, depth)
    if band is None:
        return None
    return [round(float(np.median(band[:, 0])), 5), round(float(np.median(band[:, 1])), 5)]


def base_footprint(camera: CameraModel, blob: Blob, depth: np.ndarray,
                   trim: float = 2.0) -> list[float] | None:
    """A container's measured footprint on the table, as ``[x, y, w, d]``.

    DIAGNOSTIC ONLY, AND KNOWN TO BE UNRELIABLE. Kept because it is the natural way to
    estimate a cavity and the numbers say how far off it is, but two instances of the same
    geometry disagree by more than a factor of two on one axis: on the two trays in this
    directory it returned 5.2 x 5.2 cm and 6.8 x 7.9 cm for a 6.0 x 6.0 cm footprint.

    The trouble is that what the camera can see of a tray's base is not its base. The near
    wall hides its own foot, and the cavity floor of this scene's tray spans the full outer
    footprint because the floor slab runs under the walls, so the lowest visible pixels are
    a partial, asymmetric view of it. Robust percentiles do not fix an asymmetric sample.

    So the executor does NOT use this to decide where to drop something. It uses
    ``base_centre`` for the position and the declared cavity from scene_spec for the size,
    and then verifies the result. See planner.py.

    ``pos_m`` and the bounding box remain the honest fallbacks when there is no depth.
    """
    band = base_band(camera, blob, depth, trim=trim)
    if band is None:
        return None
    x0, x1 = (float(np.percentile(band[:, 0], trim)),
              float(np.percentile(band[:, 0], 100.0 - trim)))
    y0, y1 = (float(np.percentile(band[:, 1], trim)),
              float(np.percentile(band[:, 1], 100.0 - trim)))
    return [round(x0, 5), round(y0, 5), round(x1 - x0, 5), round(y1 - y0, 5)]


def inner_rect(thing: Thing, pixels_per_m: float | None, footprint: list[float] | None = None) -> list[float] | None:
    """A container's usable interior: its footprint inset by the wall fraction.

    ``footprint`` is ``base_footprint``'s answer when a depth image was available, and is
    the better of the two measurements because it is not inflated by the container's
    height. Without it the detected bounding box is used, which is right for a flat
    coloured zone and wrong for anything with walls.

    Only meaningful with a real scale: without ``pixels_per_m`` there are no metres to
    report and the caller gets None, which is the honest answer for a webcam frame.

    Demo 09's copy of this function documented the inset and never applied it: it
    returned the outer footprint, which is harmless for a solid coloured zone but points
    the gripper straight into a wall for a hollow tray, because the bounding box of a
    tray is its outside edge. ``WALL_INSET_FRACTION`` was declared and never referenced.
    Both are fixed here, since this demo places objects inside real trays.

    The interior is clamped so a very thin detection cannot produce a negative extent.
    """
    if pixels_per_m is None or pixels_per_m <= 0 or not thing.pos_m:
        return None
    if footprint is not None:
        fx, fy, fw, fd = footprint
        centre = ((fx + fw / 2.0), (fy + fd / 2.0))
    else:
        centre = (thing.pos_m[0], thing.pos_m[1])
        fw = thing.bbox_px[2] / pixels_per_m
        fd = thing.bbox_px[3] / pixels_per_m
    keep = max(0.0, 1.0 - 2.0 * WALL_INSET_FRACTION)
    inner_w = max(0.0, fw * keep)
    inner_d = max(0.0, fd * keep)
    return [round(centre[0] - inner_w / 2.0, 5), round(centre[1] - inner_d / 2.0, 5),
            round(inner_w, 5), round(inner_d, 5)]


def is_container(blob: Blob, min_area: int) -> bool:
    """Whether a blob is a receptacle rather than an object to pick up.

    Size first, then one shape exclusion. Measured over 120 blobs in 20 frames
    (``tools/measure_shapes.py``):

    * objects to pick up are 612..687 px;
    * receptacles are 1900..2600 px.

    That is a clean 3x gap and ``colors.CONTAINER_MIN_AREA_PX`` sits in the middle of
    it, so size does the real work.

    Shape is used to rule out one case only: a ball is never a receptacle. Cubes and
    boxes are *not* separable by shape here, which was worth measuring rather than
    assuming. A solid near-square coloured zone measures fill 0.958..0.985 and
    elongation 0.749..0.935, which is the same range as a 3 cm cube (0.9562..0.9780),
    so a shape test of the form "not ball and not cube" rejects every tray on the
    table. Nothing was detected as a container at all until this was measured.

    Keeping the area floor is also what makes the occluded-object case harmless: hiding
    half a cube behind the gripper changes its shape but cannot multiply its area by
    three, so a part-occluded object stays below the floor and is never called a tray.
    """
    if blob.area_px < min_area:
        return False
    return kind_of(blob.elongation, blob.fill) != "ball"


def perceive(
    image_bgr: np.ndarray,
    source: str = "",
    camera: CameraModel | None = None,
    frame: Frame | None = None,
    plane_z: float | None = None,
    depth: np.ndarray | None = None,
    opening_px: int = colors.OPENING_PX,
    container_min_area: int = colors.CONTAINER_MIN_AREA_PX,
    detect=None,
    looking_for: list[str] | None = None,
) -> Scene:
    """One image to one scene.

    ``depth`` is an (H, W) orthogonal depth image in metres, only available in the
    simulator. It is never the source of a position: ``pos_m`` always comes from the
    ray and the table plane so that the same code works on a webcam, and the depth
    result is stored beside it in ``pos_m_depth`` for comparison.

    ``looking_for`` restricts which named colours are segmented, and ``None`` means
    ``colors.DEFAULT_DETECT``: red, green and blue. White and black match the
    background and the robot's own links, and yellow shares its hue band with the wood
    table, so including all six turns the whole frame into one detection.

    ``detect`` is an optional callable ``(image_bgr) -> list[Thing]`` from
    ``detector.py``, merged with the colour results after de-duplication.
    """
    height, width = image_bgr.shape[:2]
    hsv = to_hsv(image_bgr)
    looking_for = colors.colors(looking_for if looking_for is not None else colors.DEFAULT_DETECT)
    if frame is None:
        frame = Frame.from_image(width, height)

    found: list[Blob] = []
    for name in colors.colors(looking_for):
        found.extend(blobs_of(hsv, colors.COLORS[name], opening_px))

    # Canonical order: biggest first, then left to right. Ids are therefore stable
    # while nothing moves, which is what lets demo 10 say "object #3" and mean it.
    found.sort(key=lambda b: (-b.area_px, b.centroid_px[0]))

    pixels_per_m = None
    if camera is not None and plane_z is not None:
        pixels_per_m = camera.pixels_per_m(plane_z)

    notes: list[str] = [colors.detection_note()]
    clashes = colors.overlapping(looking_for)
    for name, others in clashes.items():
        notes.append(f"colour bands for {name} overlap {', '.join(others)}; calibrate before trusting them")

    things: list[Thing] = []
    is_container_flags: list[bool] = []
    for index, blob in enumerate(found):
        x, y, w, h = blob.bbox_px
        aspect = max(w, h) / max(1.0, min(w, h))
        container = is_container(blob, container_min_area)
        if container:
            kind = "box" if aspect <= BOX_ASPECT else "any"
        else:
            kind = kind_of(blob.elongation, blob.fill)
        thing = Thing(
            id=index,
            kind=kind,
            color=blob.color,
            centroid_px=blob.centroid_px,
            bbox_px=blob.bbox_px,
            area_px=blob.area_px,
            mean_hsv=blob.mean_hsv,
            confidence=confidence_of(blob, distance_to_edge(tuple(blob.centroid_px), width, height)),
            source="segmentation",
            elongation=round(blob.elongation, 4),
            fill=round(blob.fill, 4),
        )
        if camera is not None and plane_z is not None:
            thing.pos_m = camera.on_plane(blob.centroid_px[0], blob.centroid_px[1], plane_z)
            if thing.pos_m is not None:
                thing.pos_m = [round(v, 5) for v in thing.pos_m]
        if camera is not None and depth is not None:
            thing.pos_m_depth = _depth_position(camera, blob, depth)
        if container:
            footprint = None
            centre = None
            if camera is not None and depth is not None:
                footprint = base_footprint(camera, blob, depth)
                centre = base_centre(camera, blob, depth)
            thing.footprint_m = footprint
            thing.base_centre_m = centre
            thing.inner_rect_m = inner_rect(thing, pixels_per_m, footprint)
            thing.hollow = True
        things.append(thing)
        is_container_flags.append(container)

    if detect is not None:
        extra = _merge_detected(things, detect(image_bgr))
        things.extend(extra)
        is_container_flags.extend([False] * len(extra))

    return Scene(
        frame=frame,
        source=source,
        width=width,
        height=height,
        objects=[t for t, flag in zip(things, is_container_flags) if not flag],
        containers=[t for t, flag in zip(things, is_container_flags) if flag],
        camera=camera.to_dict() if camera is not None else None,
        ground_px_per_m=round(pixels_per_m, 1) if pixels_per_m else None,
        plane_z=plane_z,
        notes=notes,
    )


def _depth_position(camera: CameraModel, blob: Blob, depth: np.ndarray) -> list[float] | None:
    """Median depth over a blob's pixels, back-projected, or None if unusable."""
    values = depth[blob.mask]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 20:
        return None
    point = camera.from_depth(blob.centroid_px[0], blob.centroid_px[1], float(np.median(values)))
    return [round(v, 5) for v in point] if point else None


def _iou(a: Thing, b: Thing) -> float:
    ax, ay, aw, ah = a.bbox_px
    bx, by, bw, bh = b.bbox_px
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    return overlap / float(aw * ah + bw * bh - overlap)


def _merge_detected(things: list[Thing], detected: list[Thing]) -> list[Thing]:
    """Add detector boxes that do not already cover a colour blob.

    An open-vocabulary detector and colour segmentation disagree about which of the
    two is right about a flat red ball, so a box that overlaps a blob by more than
    half is dropped: the name the detector gives is kept only when the colour route
    found nothing there, which is the case it exists for.
    """
    kept: list[Thing] = []
    next_id = max((t.id for t in things), default=-1) + 1
    for candidate in sorted(detected, key=lambda t: -t.area_px):
        if any(_iou(candidate, existing) > 0.5 for existing in things):
            continue
        candidate.id = next_id
        next_id += 1
        kept.append(candidate)
    return kept


# --- drawing and calibration --------------------------------------------------------


DRAW_COLORS = {
    "red": (60, 60, 235),
    "green": (90, 200, 90),
    "blue": (235, 140, 60),
    "yellow": (40, 220, 240),
    "white": (240, 240, 240),
    "black": (40, 40, 40),
}


def annotate_truth(image_bgr: np.ndarray, scene: Scene, items, both_conventions: bool = False) -> np.ndarray:
    """Draw where the simulator says each object is, for working out the camera.

    Two markers, deliberately different shapes:

    * a cross at ``project()``, which is the convention this file believes;
    * a hollow square at the same pixel with the image's v axis flipped.

    Running once with ``both_conventions`` and looking at the frame settles which one the
    renderer uses, with no argument from first principles. After that, pass False and the
    square disappears. This exists because the first real render proved the convention
    could not be reasoned out from the docs: ``unproject_depth`` builds
    ``K^-1 [u, v, 1] d`` with a positive ``d``, which fixes the sign of z in the camera
    frame but says nothing about whether v grows downwards.
    """
    canvas = image_bgr.copy()
    if not scene.camera:
        return canvas
    camera = CameraModel.from_dict(scene.camera)
    for item in items:
        pixel = camera.project([item.x, item.y, item.z])
        if pixel is None:
            continue
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        cv2.drawMarker(canvas, (u, v), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(canvas, item.prim, (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if both_conventions:
            flipped = int(round(2 * scene.height / 2 - pixel[1]))
            cv2.rectangle(canvas, (u - 9, flipped - 9), (u + 9, flipped + 9), (0, 255, 255), 2)
    return canvas


def annotate(image_bgr: np.ndarray, scene: Scene, title: str = "") -> np.ndarray:
    """A copy of the image with every detection drawn on it, for the evidence PNGs."""
    canvas = image_bgr.copy()
    for thing in scene.everything:
        x, y, w, h = thing.bbox_px
        tint = DRAW_COLORS.get(thing.color, (200, 200, 200))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), tint, 2)
        cu, cv_ = int(round(thing.centroid_px[0])), int(round(thing.centroid_px[1]))
        cv2.drawMarker(canvas, (cu, cv_), tint, cv2.MARKER_CROSS, 12, 2)
        label = f"#{thing.id} {thing.kind} {thing.color} {thing.confidence:.2f}"
        cv2.putText(canvas, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tint, 1, cv2.LINE_AA)
    if scene.ground_px_per_m:
        footer = f"{scene.ground_px_per_m:.0f} px/m on the table"
        cv2.putText(canvas, footer, (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_png(path, image_bgr: np.ndarray) -> None:
    """Write a PNG. cv2.imwrite is avoided because it fails silently on odd paths."""
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError(f"could not encode a PNG for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def read_image(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # fromfile keeps non-ASCII paths working
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"{path} could not be read as an image")
    return image


def calibration(image_bgr: np.ndarray, opening_px: int = colors.OPENING_PX) -> str:
    """Report what is actually in a frame, so the bands in colors.py can be measured.

    Prints, per colour: how many pixels the band claims, and the area and mean HSV of
    each blob it found. Also prints the hue histogram of the low-saturation pixels,
    which is how the table's own colour is found: whatever hue the wood sits at is
    the hue a "yellow" band must avoid.
    """
    height, width = image_bgr.shape[:2]
    hsv = to_hsv(image_bgr)
    rows = [f"image {width}x{height}", "", colors.detection_note(), ""]
    for name in colors.colors():
        spec = colors.COLORS[name]
        blobs = blobs_of(hsv, spec, opening_px)
        claimed = int(colors.mask_of(hsv, spec).sum())
        rows.append(f"{name}: band claims {claimed} px, {len(blobs)} blob(s)")
        for blob in blobs:
            rows.append(
                f"    area={blob.area_px:<7} bbox={blob.bbox_px} meanHSV={blob.mean_hsv} "
                f"elongation={blob.elongation:.3f} fill={blob.fill:.3f}"
            )
    rows.append("")
    rows.append("hue histogram of the low-saturation pixels (this is the table and the room)")
    flat = hsv.reshape(-1, 3)
    wood = flat[flat[:, 1] < 90]
    if wood.size:
        counts = np.bincount(wood[:, 0], minlength=180)
        top = np.argsort(counts)[::-1][:12]
        summary = ", ".join(f"H{h}:{counts[h]}" for h in top if counts[h] > 0)
        rows.append(f"    {len(wood)} px below saturation 90")
        rows.append(f"    top hues: {summary}")
    else:
        rows.append("    no low-saturation pixels: the whole frame is saturated, check the lighting")
    return "\n".join(rows)
