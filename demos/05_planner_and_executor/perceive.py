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

from dataclasses import dataclass, field, replace
import math

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
# Those were demo 09's numbers, on a scene whose receptacles were flat coloured slabs. This
# copy replaced them with hollow trays 7 cm across and 5 cm tall, which changes what the
# camera sees around the objects, and the thresholds were re-measured rather than nudged -
# tools/measure_kinds.py, 20 jittered trials, 40 cubes and 40 balls labelled against the
# simulator's own ground truth (results/kind_calibration.md):
#
#     3 cm cubes, 40 blobs:  fill 0.9629 .. 0.9850   elong 0.749 .. 0.861
#     4 cm balls, 40 blobs:  fill 0.9980 .. 0.9999   elong 0.952 .. 0.996
#
# The cube ceiling rose from 0.9780 to 0.9850, which is exactly where demo 09 had put the
# threshold: `fill >= 0.985` was already false for every cube demo 09 measured, and now a
# cube sitting at the new ceiling fails the test by nothing at all. So the threshold has
# moved to the middle of the new gap, 0.9915, the same rule demo 09 used and the same reason.
#
# Any threshold in (0.9850, 0.9980) separates the two ranges. 0.9915 is the middle.
BALL_ELONGATION = 0.907
BALL_FILL = 0.9915
BOX_ELONGATION = 0.55

# Calibrated against 444 paired captures (826 color/position-matched loose objects) from the
# top, front and side cameras. Depth distinguishes the spherical surface of a ball from the
# circular top of a cylinder; ellipse-normalized contour harmonics distinguish polygonal
# silhouettes without assuming their axis-aligned bounding boxes.
BALL_DEPTH_TOP_FRACTION = 0.35
TRIANGLE_HARMONIC_MIN = 0.035
CUBE_HARMONIC_MIN = 0.035
HEXAGON_HARMONIC_MIN = 0.011
HEXAGON_HARMONIC_LOW_RES_MIN = 0.0075
HEXAGON_HARMONIC_DOMINANCE = 1.05

# What a square measures, kept as its own constant because confidence_of scores distance
# from either of the two expected values, and because it is the number to check after
# changing BALL_FILL. Measured midpoint of the 40 cubes above, 0.9629..0.9850.
CUBE_FILL = 0.974

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
# Oblique views foreshorten square receptacles; the measured square red tray in the
# top-camera fixture projects to 1.33:1. Keep enough margin to call that a box.
BOX_ASPECT = 1.40
# A lone 30 mm cube seen by the oblique top camera covers 2,762 px at yaw 0 but 2,985 px at yaw
# 30 degrees, where a second side face comes into view (1.49x its plan footprint). At 1.10 the
# 1.30 split trigger sat at 2,869 px, so that rotated cube was flagged "ambiguous" - no
# separation seeds - and dropped by the catalog linker, which rejected the command. At 1.25 the
# trigger is 1.63x the footprint; two touching 30 mm parts are ~1.9x or more, so they still
# split or stay flagged.
SINGLE_OBJECT_AREA_ALLOWANCE = 1.25
# How many times a declared part's size one blob may span and still be a single part. A cube at
# yaw 30 degrees spans sqrt(2) x its side in plan plus its visible side face in the oblique top
# view: 65 x 74 px (49.6 mm) for a 30 mm cube, over the former 1.5x (67 px) limit. Two parts side
# by side span 2x or more and normally have separation seeds, so 1.8x keeps them distinct.
SINGLE_OBJECT_EXTENT_FACTOR = 1.8


def projected_single_object_area_px(footprint_m2: float, pixels_per_m: float) -> float:
    """Conservative projected area for one configured part in an oblique RGB view."""
    return max(0.0, float(footprint_m2)) * float(pixels_per_m) ** 2 * SINGLE_OBJECT_AREA_ALLOWANCE


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
    polygon_sides: int = 0
    is_container_override: bool | None = None
    ambiguous_group: bool = False
    ambiguity_reason: str | None = None


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 to HSV uint8, the only conversion this file needs."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def image_pixel_scale(width: int, height: int) -> float:
    """Linear pixel scale relative to the calibrated 640x480 camera images."""
    if width <= 0 or height <= 0:
        return 1.0
    return float(np.sqrt((int(width) * int(height)) / float(640 * 480)))


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


def kind_of(elongation: float, fill: float, polygon_sides: int = 0) -> str:
    """Classify the stable top-view silhouettes; leave ambiguous blobs as ``any``."""
    if elongation >= BALL_ELONGATION and fill >= BALL_FILL:
        return "ball"
    if polygon_sides == 3:
        return "triangular_prism"
    if polygon_sides == 6:
        return "hexagonal_prism"
    if elongation >= BOX_ELONGATION:
        return "cube"
    return "any"


def polygon_sides(mask: np.ndarray) -> int:
    """Approximate the dominant object's outer contour; zero means no stable polygon."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0
    corners = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    count = len(corners)
    return count if count in (3, 4, 5, 6, 7, 8) else 0


def blobs_of(hsv: np.ndarray, color: colors.Color, opening_px: int,
             robust: bool = False, roi: np.ndarray | None = None,
             min_area_px: int = colors.MIN_AREA_PX,
             closing_px: int = 1) -> list[Blob]:
    """Every blob of one colour, after opening the mask to separate touching objects."""
    spec = replace(color, val_min=max(40, int(color.val_min * .65))) if robust else color
    mask = colors.mask_of(hsv, spec)
    if roi is not None:
        mask &= roi
    if opening_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opening_px + 1,) * 2)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    if robust:
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(
                                   cv2.MORPH_ELLIPSE,
                                   (2 * closing_px + 1, 2 * closing_px + 1))).astype(bool)
    found: list[Blob] = []
    total, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    for label in range(1, total):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        blob_mask = labels == label
        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        w, h = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        local = blob_mask[y:y+h, x:x+w]
        mean = cv2.mean(hsv[y:y+h, x:x+w], mask=local.astype(np.uint8))[:3]
        elongation, fill = shape_of(local)
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
                polygon_sides=polygon_sides(local),
            )
        )
    return sorted(found, key=lambda b: -b.area_px)


def split_touching(blob: Blob, max_extent_px: int = 160,
                   min_area_px: int = colors.MIN_AREA_PX,
                   single_object_extent_px: int | None = None,
                   single_object_area_px: float | None = None) -> list[Blob]:
    """Split a small multi-object colour clump using its distance-transform peaks.

    The previous implementation only tried elongatedness < .72 and only accepted
    cube/ball shaped children. That missed side-by-side clusters and made the split
    useless for cylinders and custom geometries. A single convex object normally
    has one distance-transform peak; a touching clump has one peak per lobe. We cap
    the image-space extent so large tray/rim masks are not treated as object groups.
    ``max_extent_px`` is scaled from the camera when available, with a conservative
    default for saved/synthetic frames.
    """
    x, y, w, h = blob.bbox_px
    extent = max(w, h)
    exceeds_single_part = (
        (single_object_extent_px is not None and extent > single_object_extent_px)
        or (single_object_area_px is not None
            and blob.area_px > single_object_area_px * 1.30)
    )

    def keep_unsplit(reason: str) -> list[Blob]:
        # A large region that could not be separated must remain visible to the caller as
        # ambiguous. Quietly returning it as one ordinary object lets the catalog assign it
        # to one of several same-colour parts and can send the gripper to a false centroid.
        if exceeds_single_part:
            blob.ambiguous_group = True
            blob.ambiguity_reason = reason
        return [blob]

    if max(w, h) > max_extent_px:
        return keep_unsplit("grupo mayor que el límite de segmentación")
    # Shading seams on one small cube can create two distance peaks even though its
    # mask is continuous. If its entire projected silhouette fits a single supported
    # part, keep it intact; true touching groups span farther than one part plus margin.
    within_single_extent = (single_object_extent_px is not None
                            and extent <= single_object_extent_px)
    within_single_area = (single_object_area_px is None
                          or blob.area_px <= single_object_area_px * 1.30)
    if within_single_extent and within_single_area:
        return [blob]
    if blob.area_px < min_area_px * 2:
        return keep_unsplit("área insuficiente para confirmar las instancias")
    mask = blob.mask[y:y+h, x:x+w].astype(np.uint8)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    # A single global threshold loses the peak of a smaller object touching a larger
    # one. Detect local maxima instead, with a floor to ignore shallow pixel noise.
    radius = max(3, int(round(float(distance.max()) * .6)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    local_max = cv2.dilate(distance.astype(np.float32), kernel)
    peak_floor = max(2.0, float(distance.max()) * .35)
    peaks = ((distance >= local_max - 1e-4) & (distance >= peak_floor)).astype(np.uint8)
    count, markers = cv2.connectedComponents(peaks)
    if count < 3 or count > 5:
        return keep_unsplit("no hay semillas de separación fiables")
    # Surface seams, shading boundaries and aliased cube corners can create multiple
    # distance-transform maxima inside one object. Real separate objects need distinct
    # centres: for touching disks their peaks are roughly a radius apart or more. Close
    # maxima are therefore treated as one irregular silhouette instead of split into
    # duplicate detections. This is especially important for matte cubes whose red face
    # segmentation can form two lobes divided by a narrow low-saturation seam.
    peak_centres = []
    for label in range(1, count):
        py, px = np.nonzero(markers == label)
        if len(px):
            peak_centres.append((float(px.mean()), float(py.mean())))
    minimum_peak_separation = max(3.0, float(distance.max()) * .65)
    if any(math.dist(first, second) < minimum_peak_separation
           for index, first in enumerate(peak_centres)
           for second in peak_centres[index + 1:]):
        return keep_unsplit("semillas de separación demasiado próximas")
    # Flood the negative distance surface. cv2.watershed uses image gradients,
    # which does not implement this topography and can swallow one of the peaks.
    import heapq
    heap = []
    serial = 0
    for yy, xx in zip(*np.nonzero(markers)):
        heapq.heappush(heap, (-float(distance[yy, xx]), serial, int(yy), int(xx)))
        serial += 1
    while heap:
        _, _, yy, xx = heapq.heappop(heap)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = yy + dy, xx + dx
            if 0 <= ny < padded.shape[0] and 0 <= nx < padded.shape[1] and padded[ny, nx] and markers[ny, nx] == 0:
                markers[ny, nx] = markers[yy, xx]
                heapq.heappush(heap, (-float(distance[ny, nx]), serial, ny, nx))
                serial += 1
    results = []
    for label in range(1, count):
        local = (markers[1:-1, 1:-1] == label) & (mask > 0)
        child_area = int(local.sum())
        if child_area < min_area_px:
            return keep_unsplit("la separación produciría una pieza demasiado pequeña")
        ys, xs = np.nonzero(local)
        full = np.zeros_like(blob.mask)
        full[y:y+h, x:x+w] = local
        elongation, fill = shape_of(local)
        candidate = Blob(blob.color, int(local.sum()), [x+int(xs.min()), y+int(ys.min()),
                         int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)],
                         [x+float(xs.mean()), y+float(ys.mean())], blob.mean_hsv,
                         elongation, fill, full, polygon_sides(local),
                         is_container_override=False)
        results.append(candidate)
    # A watershed can create a plausible-looking sliver along a specular edge. Keep
    # the original component unless every child has meaningful area and the split
    # explains almost all of it. This is geometry agnostic: cylinders/prisms may
    # correctly remain kind="any" until scene metadata supplies their class.
    if not results or max(part.area_px for part in results) / min(part.area_px for part in results) > 6.0:
        return keep_unsplit("las áreas separadas no son coherentes")
    if abs(sum(part.area_px for part in results) - blob.area_px) > max(20, int(blob.area_px * .08)):
        return keep_unsplit("la separación no conserva suficiente máscara")
    return results


def split_depth_layered_container(blob: Blob, depth: np.ndarray, max_extent_px: int,
                                  min_area_px: int = colors.MIN_AREA_PX,
                                  container_min_area: int = colors.CONTAINER_MIN_AREA_PX,
                                  single_object_extent_px: int | None = None,
                                  single_object_area_px: float | None = None
                                  ) -> list[Blob]:
    """Recover a small same-colour part touching a larger tray using depth layers.

    A front/side camera can project a red cube against the red wall of its tray. Their colour
    masks then become one connected component, which the area heuristic labels only as a box.
    Depth separates the nearer part surface from the tray even though HSV cannot. Keep the
    original component as the container and return only compact, supported-shape child masks;
    depth fragments of the tray itself are rejected by geometry classification and extent.
    """
    depth = np.asarray(depth)
    if (not is_container(blob, container_min_area) or depth.ndim != 2
            or depth.shape != blob.mask.shape):
        return []
    valid = blob.mask & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    if len(xs) < max(container_min_area, min_area_px * 3):
        return []
    values = depth[ys, xs].astype(np.float32)
    if float(np.percentile(values, 90) - np.percentile(values, 10)) < 0.012:
        return []
    child_min_area = max(int(min_area_px), KIND_MIN_AREA_PX // 2)

    candidates: list[Blob] = []
    # Deterministic 1-D k-means avoids a random initialization changing the instance count from
    # one render to the next. More than four depth modes mostly partitions the tray's own walls.
    for clusters in (2, 3, 4):
        centres = np.quantile(values, np.linspace(.12, .88, clusters)).astype(np.float32)
        labels = np.zeros(len(values), dtype=np.int32)
        for _ in range(24):
            labels = np.abs(values[:, None] - centres[None, :]).argmin(axis=1)
            updated = centres.copy()
            for label in range(clusters):
                group = values[labels == label]
                if len(group):
                    updated[label] = float(group.mean())
            if np.allclose(updated, centres, atol=1e-4):
                break
            centres = updated

        for label in range(clusters):
            if int(np.count_nonzero(labels == label)) < child_min_area:
                continue
            layer = np.zeros_like(blob.mask, dtype=np.uint8)
            selected = labels == label
            layer[ys[selected], xs[selected]] = 1
            count, components, stats, centroids = cv2.connectedComponentsWithStats(
                layer, connectivity=8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                x = int(stats[component, cv2.CC_STAT_LEFT])
                y = int(stats[component, cv2.CC_STAT_TOP])
                w = int(stats[component, cv2.CC_STAT_WIDTH])
                h = int(stats[component, cv2.CC_STAT_HEIGHT])
                max_object_area = max(container_min_area * 2.5, (max_extent_px * .35) ** 2)
                if (area < child_min_area
                        or area > max_object_area or max(w, h) > max_extent_px):
                    continue
                # A recovered part is a declared part: larger than one is the tray's own
                # cavity floor or wall layer. From the top camera that layer is a 4,300 px
                # region on a 70 mm tray, was emitted as a "ball" or "cube", and competed with
                # the real tray for its catalog identity, so the order was refused.
                if ((single_object_area_px is not None
                     and area > single_object_area_px * 1.30)
                        or (single_object_extent_px is not None
                            and max(w, h) > single_object_extent_px)):
                    continue
                child_mask = components == component
                local = child_mask[y:y+h, x:x+w]
                elongation, fill = shape_of(local)
                child = Blob(
                    blob.color, area, [x, y, w, h],
                    [float(centroids[component][0]), float(centroids[component][1])],
                    blob.mean_hsv, elongation, fill, child_mask, polygon_sides(local))
                if (is_container(child, container_min_area)
                        or robust_kind(child) not in {
                            "cube", "ball", "cylinder", "triangular_prism",
                            "hexagonal_prism"}):
                    continue
                # A depth layer is only a valid second instance when the original colour mask
                # has nearby pixels outside this child. This rejects isolated tray-wall texture.
                nearby = cv2.dilate(child_mask.astype(np.uint8),
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
                if not np.any((nearby > 0) & blob.mask & ~child_mask):
                    continue
                candidates.append(child)

    # Different cluster counts often find the same physical object. Merge those proposals by
    # centre distance and retain the largest complete child mask.
    unique: list[Blob] = []
    for candidate in sorted(candidates, key=lambda part: -part.area_px):
        if any(math.dist(candidate.centroid_px, kept.centroid_px) <= 6.0
               for kept in unique):
            continue
        unique.append(candidate)
    return unique


def confidence_of(blob: Blob, distance_to_edge: float, kind: str | None = None,
                  camera: CameraModel | None = None,
                  depth: np.ndarray | None = None) -> float:
    """Score evidence for the *classified shape*, plus size and border visibility.

    The old score measured only how close a blob's fill ratio was to a ball or cube.
    Triangular prisms and hexagonal prisms therefore received low scores even when
    their harmonic classifier had strong, correct vertex evidence. That made the
    monitor label good detections as uncertain and made scores incomparable between
    supported geometries. This value is a calibrated evidence score, not a probability.
    """
    if kind in {"triangular_prism", "hexagonal_prism", "cube"}:
        h3, h4, h6 = contour_harmonics(blob.mask, blob.bbox_px)
        if kind == "triangular_prism":
            evidence = h3 / max(TRIANGLE_HARMONIC_MIN * 1.8, 1e-6)
            purity = float(blob.fill < 0.95)
        elif kind == "hexagonal_prism":
            evidence = h6 / max(HEXAGON_HARMONIC_MIN * 1.8, 1e-6)
            purity = float(blob.fill >= 0.94)
        else:
            evidence = h4 / max(CUBE_HARMONIC_MIN * 1.8, 1e-6)
            purity = max(0.0, 1.0 - abs(blob.fill - CUBE_FILL) / 0.04)
        shape_quality = float(np.clip(evidence, 0.0, 1.0)) * float(purity)
    elif kind == "ball":
        # Rendering, antialiasing and occlusion move the silhouette fill by a few
        # thousandths even for a true sphere. Keep this broad enough that a measured
        # round top is not reported as uncertain solely from raster noise.
        fill_quality = max(0.0, 1.0 - abs(blob.fill - BALL_FILL) / 0.05)
        round_quality = float(np.clip(blob.elongation / max(BALL_ELONGATION, 1e-6), 0.0, 1.0))
        shape_quality = fill_quality * round_quality
    elif kind == "cylinder":
        top_fraction = depth_top_fraction(blob, camera, depth)
        flat_cap = float(top_fraction is not None and top_fraction >= BALL_DEPTH_TOP_FRACTION)
        round_quality = float(np.clip(blob.elongation / max(BALL_ELONGATION, 1e-6), 0.0, 1.0))
        shape_quality = flat_cap * round_quality
    elif kind == "box":
        shape_quality = float(blob.area_px >= colors.CONTAINER_MIN_AREA_PX)
    else:
        # Unknown geometry should remain visibly less certain than a supported class.
        purity = min(abs(blob.fill - BALL_FILL), abs(blob.fill - CUBE_FILL)) / 0.05
        shape_quality = float(np.clip(1.0 - min(1.0, purity), 0.0, 1.0))

    score = 0.50 + 0.45 * shape_quality
    if blob.area_px < KIND_MIN_AREA_PX:
        score *= 0.6
    # ``distance_to_edge`` grows as a blob moves farther inside the image. The previous
    # expression inverted that relationship, assigning the lowest confidence near the
    # image centre and the highest confidence at the frame border. Keep at least 0.55
    # edge confidence and rise monotonically to 1.0 for an un-clipped central object.
    edge_quality = float(np.clip(0.55 + distance_to_edge, 0.55, 1.0))
    score *= edge_quality
    return round(float(np.clip(score, 0.05, 0.99)), 3)


def distance_to_edge(centroid: tuple[float, float], width: int, height: int) -> float:
    """How far a centroid is from the frame border, as a fraction of the half-diagonal."""
    half = 0.5 * float(np.hypot(width, height))
    return min(centroid[0], centroid[1], width-centroid[0], height-centroid[1]) / half if half else 0.0


def contour_harmonics(mask: np.ndarray, bbox_px: list[int] | None = None
                      ) -> tuple[float, float, float]:
    """Return normalized 3/4/6-fold contour evidence after removing ellipse perspective.

    A regular polygon projected obliquely is no longer a regular polygon in pixel space. We fit
    the contour's ellipse first, then measure radial variation in that normalized frame. The
    harmonic amplitudes are robust to rotation: triangles, squares and hexagons contribute most
    strongly at orders 3, 4 and 6, while circular silhouettes have little energy at any of them.
    """
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if bbox_px is not None:
        x, y, width, height = bbox_px
        binary = binary[y:y + height, x:x + width]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return 0.0, 0.0, 0.0
    (cx, cy), (major, minor), angle = cv2.fitEllipse(contour)
    if major <= 0.0 or minor <= 0.0:
        return 0.0, 0.0, 0.0
    theta = math.radians(angle)
    points = contour[:, 0, :].astype(np.float64)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    u = (dx * math.cos(theta) + dy * math.sin(theta)) / max(major / 2.0, 1.0)
    v = (-dx * math.sin(theta) + dy * math.cos(theta)) / max(minor / 2.0, 1.0)
    radius = np.hypot(u, v)
    mean_radius = float(radius.mean())
    if mean_radius <= 1e-9:
        return 0.0, 0.0, 0.0
    phase = np.arctan2(v, u)
    residual = radius - mean_radius
    return tuple(float(abs(np.mean(residual * np.exp(-1j * order * phase))) / mean_radius)
                 for order in (3, 4, 6))


def hexagonal_harmonic_evidence(h3: float, h4: float, h6: float,
                                fill: float) -> bool:
    """Recover low-resolution hexagons when their sixth harmonic dominates.

    In the 30-scene camera/variant matrix, two 30 mm hexagons fell below the absolute
    sixth-harmonic threshold after segmentation and projection. Their h6 amplitudes were
    .0088 and .0108, but both remained stronger than h3/h4. Cylinder controls had a
    sixth harmonic no stronger than .0086 and a clearly stronger fourth harmonic.
    Requiring a compact mask plus sixth-harmonic dominance recovers those hexagons while
    keeping the measured cylinder controls separate.
    """
    strongest_competitor = max(float(h3), float(h4))
    return (float(fill) >= .97
            and float(h6) >= HEXAGON_HARMONIC_LOW_RES_MIN
            and float(h6) >= HEXAGON_HARMONIC_DOMINANCE * strongest_competitor)


def surface_top_fraction(world_z: np.ndarray, band_m: float = 0.0015) -> float | None:
    """Fraction of visible depth samples on the object's nearly level upper surface."""
    heights = np.asarray(world_z, dtype=np.float64).reshape(-1)
    heights = heights[np.isfinite(heights)]
    if heights.size < 40:
        return None
    top = float(np.percentile(heights, 95))
    return float(np.mean(heights >= top - float(band_m)))


def depth_world_z_samples(blob: Blob, camera: CameraModel | None,
                          depth: np.ndarray | None) -> np.ndarray | None:
    """World-height samples covered by a blob's valid depth pixels."""
    if camera is None or depth is None:
        return None
    readings = np.asarray(depth, dtype=np.float64)
    if readings.ndim != 2 or readings.shape != blob.mask.shape:
        return None
    x0, y0, width, height = blob.bbox_px
    ys, xs = np.nonzero(blob.mask[y0:y0 + height, x0:x0 + width])
    if len(xs) < 40:
        return None
    distance = readings[ys + y0, xs + x0]
    valid = np.isfinite(distance) & (distance > 0.0)
    if int(valid.sum()) < 40:
        return None
    xs, ys, distance = xs[valid] + x0, ys[valid] + y0, distance[valid]
    # Project only the world-Z row to avoid allocating a full 3-D ray per pixel.
    ray_y = -(xs - camera.k[0, 2]) / camera.k[0, 0]
    ray_z = -(ys - camera.k[1, 2]) / camera.k[1, 1]
    rotation_z = camera.rotation[2]
    return distance * (
        rotation_z[0] + rotation_z[1] * ray_y + rotation_z[2] * ray_z
    ) + camera.origin[2]


def object_center_z(blob: Blob, camera: CameraModel | None,
                    depth: np.ndarray | None, plane_z: float,
                    fallback_height_m: float = 0.015) -> float:
    """Estimate the vertical centre used to unproject an object's image centroid.

    A pixel centroid on a raised object is not on the table plane. Intersecting it with
    the table shifts the grasp point horizontally in oblique views. Depth gives the top
    surface height; the centroid of supported balls and vertical prisms lies near half
    their height. RGB-only cameras use the calibrated 15 mm centre-height fallback.
    """
    world_z = depth_world_z_samples(blob, camera, depth)
    if world_z is not None:
        top_z = float(np.percentile(world_z, 95))
        height = top_z - float(plane_z)
        if 0.004 <= height <= 0.060:
            return float(plane_z) + height * 0.5
    return float(plane_z) + float(fallback_height_m)


def depth_top_fraction(blob: Blob, camera: CameraModel | None,
                       depth: np.ndarray | None) -> float | None:
    """Estimate whether the visible object has a flat cap or a curved spherical surface."""
    world_z = depth_world_z_samples(blob, camera, depth)
    if world_z is None:
        return None
    return surface_top_fraction(world_z)


def robust_kind(blob: Blob, camera: CameraModel | None = None,
                depth: np.ndarray | None = None) -> str:
    """Classify from depth topology and ellipse-normalized contour shape when available."""
    top_fraction = depth_top_fraction(blob, camera, depth)
    # Contour harmonics also work for an RGB-only camera. They distinguish polygonal
    # footprints without mistaking rasterized corners on a round object for a polygon.
    h3, h4, h6 = contour_harmonics(blob.mask, blob.bbox_px)
    if top_fraction is not None:
        # A ball's projected outline can look polygonal at low resolution, but its visible
        # world-height profile is curved: very few mask pixels lie at the uppermost 1.5 mm.
        if top_fraction < BALL_DEPTH_TOP_FRACTION and blob.elongation >= .75:
            return "ball"
        # The square's fourth harmonic wins when a shaded cube also has a noisy third harmonic.
        if h4 >= CUBE_HARMONIC_MIN and blob.fill >= .94:
            return "cube"
        if h3 >= TRIANGLE_HARMONIC_MIN and blob.fill < .95:
            return "triangular_prism"
        if h6 >= HEXAGON_HARMONIC_MIN and blob.fill >= .97:
            return "hexagonal_prism"
        if hexagonal_harmonic_evidence(h3, h4, h6, blob.fill):
            return "hexagonal_prism"
        # A tall, flat circular cap is the remaining supported round-top geometry.
        if top_fraction >= BALL_DEPTH_TOP_FRACTION and blob.fill >= .97:
            return "cylinder"
    else:
        # A depthless RGB view still gives reliable polygon cues for the tested classes.
        # Apply the same calibrated harmonic thresholds used by the depth path before its
        # coarse rotated-rectangle fallback, which loses vertices on small/oblique prisms.
        # These thresholds were calibrated with known camera intrinsics. Uncalibrated binary
        # masks (including deliberately aliasing cluster fixtures) retain their older rules.
        if camera is not None:
            if h4 >= CUBE_HARMONIC_MIN and blob.fill >= .94:
                return "cube"
            if h3 >= TRIANGLE_HARMONIC_MIN and blob.fill < .95:
                return "triangular_prism"
            if h6 >= HEXAGON_HARMONIC_MIN and blob.fill >= .97:
                return "hexagonal_prism"
            if hexagonal_harmonic_evidence(h3, h4, h6, blob.fill):
                return "hexagonal_prism"

    # Rotated-rectangle occupancy is the fallback when depth is unavailable.
    x, y, w, h = blob.bbox_px
    contours, _ = cv2.findContours(blob.mask[y:y+h, x:x+w].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "any"
    contour = max(contours, key=cv2.contourArea)
    _, (rw, rh), _ = cv2.minAreaRect(contour)
    extent = cv2.contourArea(contour) / max(1.0, rw * rh)
    if blob.elongation < .72:
        return "any"
    # A near-circular silhouette is not enough to call a ball: an overhead cylinder has
    # the same outline. The calibrated ball range starts at BALL_ELONGATION; round shapes
    # below it remain untyped unless depth reveals a flat cap or a curved sphere surface.
    minimum_ball_elongation = BALL_ELONGATION if camera is not None else .75
    minimum_ball_fill = BALL_FILL if camera is not None else .97
    if (extent < .85 and blob.fill >= minimum_ball_fill
            and blob.elongation >= minimum_ball_elongation):
        return "ball"
    if blob.polygon_sides == 3:
        return "triangular_prism"
    if blob.polygon_sides == 6:
        return "hexagonal_prism"
    if extent > .85 and blob.fill < BALL_FILL:
        return "cube"
    return "any"


def workspace_roi(camera, plane_z, width, height):
    import scene_spec
    pad = .025
    corners = [(scene_spec.WORK_X[0]-pad, scene_spec.WORK_Y[0]-pad),
               (scene_spec.WORK_X[1]+pad, scene_spec.WORK_Y[0]-pad),
               (scene_spec.WORK_X[1]+pad, scene_spec.WORK_Y[1]+pad),
               (scene_spec.WORK_X[0]-pad, scene_spec.WORK_Y[1]+pad)]
    pixels = [camera.project([x, y, plane_z]) for x, y in corners]
    if any(p is None for p in pixels):
        raise ValueError("La cámara no cubre el área de trabajo.")
    roi = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(roi, np.rint(pixels).astype(np.int32), 1)
    return roi.astype(bool)


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
    if blob.is_container_override is not None:
        return blob.is_container_override
    if blob.area_px < min_area:
        return False
    return kind_of(blob.elongation, blob.fill, blob.polygon_sides) != "ball"


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
                   robust: bool = False,
    min_area_px: int = colors.MIN_AREA_PX,
    max_object_extent_m: float | None = None,
    max_object_footprint_m2: float | None = None,
    object_bounds_by_color: dict[str, tuple[float, float]] | None = None,
    captured_at: float | None = None,
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
    # Scale calibrated pixel thresholds only when intrinsics identify a real camera
    # image. Cropped/synthetic fixtures without a camera model have no known field of
    # view, so their apparent dimensions must retain the legacy thresholds.
    resolution_scale = image_pixel_scale(width, height) if camera is not None else 1.0
    # Pixel areas and morphology kernels were calibrated against 640x480 frames.
    # Scale areas quadratically and radii linearly so switching to a sharper sensor
    # increases shape detail without admitting four times as much speckle.
    min_area_px = max(1, int(round(min_area_px * resolution_scale ** 2)))
    container_min_area = max(1, int(round(container_min_area * resolution_scale ** 2)))
    opening_radius = max(1, int(round((1 if robust else opening_px) * resolution_scale)))
    closing_radius = max(1, int(round(resolution_scale)))
    hsv = to_hsv(image_bgr)
    looking_for = colors.colors(looking_for if looking_for is not None else colors.DEFAULT_DETECT)
    if frame is None:
        frame = Frame.from_image(width, height)

    found: list[Blob] = []
    roi = workspace_roi(camera, plane_z, width, height) if robust and camera is not None and plane_z is not None else None
    for name in colors.colors(looking_for):
        found.extend(blobs_of(hsv, colors.COLORS[name], opening_radius,
                              robust=robust, roi=roi, min_area_px=min_area_px,
                              closing_px=closing_radius))
    if robust:
        # A camera-aware limit keeps receptacles out of the instance splitter while
        # allowing a compact group of several objects to be separated.
        max_extent = 160
        ppm = (camera.pixels_per_m(plane_z)
               if camera is not None and plane_z is not None else None)
        if ppm is not None:
            max_extent = max(64, min(256, int(ppm * .20)))
        single_object_extent = None
        single_object_area = None
        if ppm is not None:
            # The size-aware 80 px floor is used for <=22 mm objects; the ordinary
            # floor covers the configured scene size. A 1.5x allowance covers silhouette
            # and projection variation without admitting two ordinary parts.
            object_size_m = (.022 if min_area_px <= colors.SMALL_OBJECT_MIN_AREA_PX
                             else float(max_object_extent_m or .036))
            if max_object_extent_m is not None:
                object_size_m = max(.006, float(max_object_extent_m))
            single_object_extent = max(8, int(round(
                ppm * object_size_m * SINGLE_OBJECT_EXTENT_FACTOR)))
            if max_object_footprint_m2 is not None:
                # Oblique views include visible side faces and HSV masks can retain
                # antialiased edge pixels. A 1.10 projected-footprint factor avoids
                # treating one real 30 mm cube (measured at 2,762 px in the top
                # camera) as an unresolved cluster, while two touching parts still
                # exceed the 1.30 split trigger or the single-part extent bound.
                single_object_area = projected_single_object_area_px(
                    max_object_footprint_m2, ppm)

        def part_bounds(blob: Blob) -> tuple[int | None, float | None]:
            extent_m, footprint_m2 = (object_bounds_by_color or {}).get(
                blob.color, (max_object_extent_m, max_object_footprint_m2))
            extent_px = (max(8, int(round(ppm * float(extent_m) * SINGLE_OBJECT_EXTENT_FACTOR)))
                         if extent_m is not None and ppm is not None
                         else single_object_extent)
            area_px = (projected_single_object_area_px(footprint_m2, ppm)
                       if footprint_m2 is not None and ppm is not None
                       else single_object_area)
            return extent_px, area_px

        instances: list[Blob] = []
        for blob in found:
            part_extent_px, part_area_px = part_bounds(blob)
            if is_container(blob, container_min_area):
                # The configured scene is a strong prior here: when no movable part
                # has this colour, neither watershed nor depth-layer recovery can
                # reveal a catalogued target. Avoid repeatedly clustering the tray's
                # own wall/floor depth on every camera frame.
                has_matching_part = (not object_bounds_by_color
                                     or blob.color in object_bounds_by_color)
                if not has_matching_part:
                    instances.append(blob)
                    continue
                # A large container can itself have several distance-transform peaks
                # (most often its coloured walls and floor separated by shading). Only
                # replace it with children when every child is compact enough to be one
                # supported part. This prevents a top-view tray from disappearing as two
                # giant "objects" while retaining the same-colour cluster recovery.
                split = split_touching(
                    blob, max_extent, min_area_px=min_area_px,
                    single_object_extent_px=part_extent_px,
                    single_object_area_px=part_area_px)
                compact_cluster = len(split) > 1
                if compact_cluster:
                    # The largest supported object is 36 mm. A 55 mm image-space limit
                    # leaves projection/segmentation margin while rejecting tray-wall
                    # fragments that span most of a 60-84 mm receptacle. Saved frames
                    # without camera calibration retain the historical extent limit.
                    part_limit = (max(18, int(round(camera.pixels_per_m(plane_z) * .055)))
                                  if camera is not None and plane_z is not None
                                  else max_extent)
                    compact_cluster = all(max(part.bbox_px[2:]) <= part_limit for part in split)
                if compact_cluster:
                    instances.extend(split)
                else:
                    # The size prior passed to the splitter describes movable parts, not
                    # receptacles. Keep an unsplit tray as a valid container observation; its
                    # inner depth-layer pass below may still recover a part resting in it.
                    blob.ambiguous_group = False
                    blob.ambiguity_reason = None
                    instances.append(blob)
                if depth is not None:
                    instances.extend(split_depth_layered_container(
                        blob, depth, max_extent, min_area_px=min_area_px,
                        container_min_area=container_min_area,
                        single_object_extent_px=part_extent_px,
                        single_object_area_px=part_area_px))
            else:
                instances.extend(split_touching(
                    blob, max_extent, min_area_px=min_area_px,
                    single_object_extent_px=part_extent_px,
                    single_object_area_px=part_area_px))
        found = instances

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
            kind = (robust_kind(blob, camera=camera, depth=depth) if robust else
                    kind_of(blob.elongation, blob.fill, blob.polygon_sides))
        thing = Thing(
            id=index,
            kind=kind,
            color=blob.color,
            centroid_px=blob.centroid_px,
            bbox_px=blob.bbox_px,
            area_px=blob.area_px,
            mean_hsv=blob.mean_hsv,
            confidence=confidence_of(
                blob, distance_to_edge(tuple(blob.centroid_px), width, height),
                kind=kind, camera=camera, depth=depth),
            source="segmentation",
            elongation=round(blob.elongation, 4),
            fill=round(blob.fill, 4),
            ambiguous=bool(blob.ambiguous_group),
            ambiguity_reason=blob.ambiguity_reason,
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
            if robust and camera is not None and plane_z is not None:
                # A partial floor band can bias the median towards one wall. Use
                # the projected silhouette at mid-height instead; Demo 11 supplies
                # each matched container's declared height rather than this default.
                target = camera.on_plane(*blob.centroid_px, plane_z + .025)
                if target:
                    thing.aim_xy = target[:2]
        elif robust and camera is not None and plane_z is not None and kind != "any":
            # The image centroid lies above the table for every raised object. Correct
            # that ray height for cylinders/prisms too; using the table plane directly
            # displaced front-view grasps by up to 10 mm in the 20 mm cylinder trials.
            centre_z = object_center_z(blob, camera, depth, plane_z)
            target = camera.on_plane(*blob.centroid_px, centre_z)
            if target:
                thing.aim_xy = target[:2]
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
        captured_at=captured_at,
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


def annotate(image_bgr: np.ndarray, scene: Scene, title: str = "", lang: str | None = None) -> np.ndarray:
    """A copy of the image with every detection drawn on it, for the evidence PNGs."""
    canvas = image_bgr.copy()
    for thing in scene.everything:
        x, y, w, h = thing.bbox_px
        tint = DRAW_COLORS.get(thing.color, (200, 200, 200))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), tint, 2)
        cu, cv_ = int(round(thing.centroid_px[0])), int(round(thing.centroid_px[1]))
        cv2.drawMarker(canvas, (cu, cv_), tint, cv2.MARKER_CROSS, 12, 2)
        label = (f"#{thing.id} {'AMBIGUO' if thing.ambiguous else thing.kind} "
                 f"{thing.color} {thing.confidence:.2f}")
        if lang:
            import describe
            import unicodedata
            name = ("grupo ambiguo" if thing.ambiguous else
                    describe.phrase(thing.kind, thing.color, 1, lang, bare=True))
            label = f"#{thing.id} {name} {thing.confidence:.2f}"
            label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
        cv2.putText(canvas, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, tint, 1, cv2.LINE_AA)
    if scene.ground_px_per_m:
        footer = f"{scene.ground_px_per_m:.0f} px/m " + ("sobre la mesa" if lang == "es" else "on the table")
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
