"""Find the cube in a front-camera frame, and say where it is.

Why this module exists
----------------------

The camera that produced the demonstrations is pinned and calibrated
(``il_env`` / LeIsaac's task config: ``lock_camera=True``, ``focal_length=40.6``,
``horizontal_aperture=38.11``), so where the cube is in the image and where it is
on the table are connected by a known projection. Recovering the cube's position
does not need to be learned, and this is the measured reason: a linear ridge
regression on a 24x32 grayscale downsample of 100 demonstration frames recovers
the cube's world x at **R2 = 0.995** and y at **R2 = 0.999**. The information is
plainly there; the imitation-learning policy simply has to be asked for it in a
form it can use.

What it gives back
------------------

``detect_cube`` returns the **pixel centroid** of the red cube, not a world
position. That is deliberate:

* No calibration is needed, so nothing has to be re-fitted when the camera moves.
* Within a fixed table row the centroid is linear in world x to **<= 5.3 px**
  (R2 = 0.95-0.97), and the fitted slope is ~540-600 px per **metre** along that
  axis, so that residual is **<= 10 mm** against a 30 mm placement tolerance -
  measured, see ``tools/17_validate_detector.py``. It is not sub-millimetre: the
  table is seen obliquely, so the cube's 3 cm spans ~16 px along x while its
  silhouette is ~33 px across diagonally.
* A world position is a re-parameterisation of the same two numbers, available
  through :func:`Calibration` when enough samples exist to fit one.

Measured accuracy on the recorded demonstrations
------------------------------------------------

=========================  ==========================
quantity                   value
=========================  ==========================
detection rate             100 / 100 frames
blob area                  938 px mean (814-1030)
repeatability, same cube   0.9 px median, 6.8 px worst
linearity in world x       R2 = 0.95-0.97
residual                   <= 5.3 px  (<= 10 mm)
=========================  ==========================

The scene is bright and low saturation overall (mean saturation 0.11), and the
table itself is orange at hue 40-60, so a naive "is it reddish" test fires on
61 % of the frame. The saturation gate below is what makes it work.
"""

from __future__ import annotations

from dataclasses import dataclass

# The cube is the only strongly saturated red thing in the scene. Everything else
# is either desaturated (the table and background, mean saturation 0.11) or a
# different hue entirely (the table sits at 40-60 deg).
HUE_MAX_DEG = 20.0
HUE_WRAP_DEG = 335.0
MIN_SATURATION = 0.45

# A 3 cm cube is ~30 px across from this camera. Demo 05's ``colors.MIN_AREA_PX``
# is 150 px and is kept as the floor; the ceiling rejects a mask that has merged
# the cube with something else, which is the failure mode worth guarding against.
# Measured on the recorded demonstrations: mean 938 px, range 814-1030.
MIN_AREA_PX = 150
MAX_AREA_PX = 4000
CUBE_SIZE_M = 0.03


@dataclass(frozen=True)
class Detection:
    """Where the cube is in one camera frame."""

    u: float
    """Centroid column, pixels from the left edge."""
    v: float
    """Centroid row, pixels from the top edge."""
    area_px: int
    width_px: float
    height_px: float

    @property
    def plausible(self) -> bool:
        """Whether the blob is cube-shaped rather than a merge or a stray."""
        if not (MIN_AREA_PX <= self.area_px <= MAX_AREA_PX):
            return False
        longer = max(self.width_px, self.height_px)
        shorter = min(self.width_px, self.height_px)
        return longer <= 3.0 * max(shorter, 1e-6)


def hsv(rgb):
    """(H, W, 3) uint8 RGB -> hue in degrees (0-360), saturation, value.

    Written out rather than taken from ``cv2`` or ``matplotlib`` so the module has
    no dependency beyond NumPy and the result is defined in the same terms the
    thresholds below are written in.
    """
    import numpy as np

    x = np.asarray(rgb, dtype=np.float32) / 255.0
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    delta = mx - mn

    hue = np.zeros_like(mx)
    nonzero = delta > 1e-6

    is_r = (mx == x[..., 0]) & nonzero
    is_g = (mx == x[..., 1]) & nonzero
    is_b = (mx == x[..., 2]) & nonzero

    hue[is_r] = (((x[..., 1] - x[..., 2])[is_r] / delta[is_r]) % 6) * 60.0
    hue[is_g] = (((x[..., 2] - x[..., 0])[is_g] / delta[is_g]) + 2) * 60.0
    hue[is_b] = (((x[..., 0] - x[..., 1])[is_b] / delta[is_b]) + 4) * 60.0

    saturation = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0.0)
    return hue, saturation, mx


def red_mask(rgb):
    """Boolean mask of the strongly saturated red pixels."""
    import numpy as np

    hue, saturation, _ = hsv(rgb)
    return (saturation > MIN_SATURATION) & (
        (hue < HUE_MAX_DEG) | (hue > HUE_WRAP_DEG)
    )


def detect_cube(rgb) -> Detection | None:
    """Largest saturated-red blob's centroid, or ``None`` if the cube is not there.

    "Largest" rather than "first found" so a few stray pixels from elsewhere in
    the frame cannot pre-empt the real cube. Returns ``None`` rather than a bad
    guess when the blob is out of the area bounds, so a caller can react instead
    of acting on noise.
    """
    import numpy as np
    from scipy import ndimage

    mask = red_mask(rgb)
    if not mask.any():
        return None

    labels, count = ndimage.label(mask)
    if count == 0:
        return None

    # Component sizes, then the index of the biggest.
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] > MAX_AREA_PX:
        return None

    rows, cols = np.nonzero(labels == best)
    if len(rows) < MIN_AREA_PX:
        return None

    return Detection(
        u=float(cols.mean()),
        v=float(rows.mean()),
        area_px=int(len(rows)),
        width_px=float(cols.max() - cols.min() + 1),
        height_px=float(rows.max() - rows.min() + 1),
    )


class Calibration:
    """Pixel centroid -> table position, by a homography on the table plane.

    Fitted, not assumed, and fitted *only when the samples justify it*: a plane
    to image map has 8 degrees of freedom and needs the samples to span a
    two-dimensional region. The demonstrations this demo started from only ever
    used two distinct table rows, which is not enough - a fit to them is
    unsolvable and comes out with ~400 px errors. :meth:`usable` refuses rather
    than returning something wrong.

    The policy does not need this. ``detect_cube``'s pixel centroid is already
    linear in world x to 5 mm; this is for reporting a position in metres, and
    for the day the workspace is widened.
    """

    def __init__(self, matrix, samples, pixels_per_metre: float):
        import numpy as np

        self._h = np.asarray(matrix, dtype=np.float64)
        self.samples = int(samples)
        self.pixels_per_metre = float(pixels_per_metre)

    @classmethod
    def fit(cls, pixels, table_points, min_rows: int = 3):
        """Fit from ``(u, v)`` centroids and their matching ``(x, y)`` positions.

        Returns ``None`` when the samples do not span enough distinct table rows,
        which is the case that silently produces a wrong answer.
        """
        import numpy as np

        pixels = np.asarray(pixels, dtype=np.float64)
        table_points = np.asarray(table_points, dtype=np.float64)
        rows = np.unique(np.round(table_points[:, 1], 3))
        if len(rows) < min_rows:
            return None

        design, rhs = [], []
        for (u, v), (x, y) in zip(pixels, table_points):
            design.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
            rhs.append(u)
            design.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
            rhs.append(v)
        solution, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(rhs), rcond=None)
        matrix = np.append(solution, 1.0).reshape(3, 3)

        ppm = float(np.ptp(pixels[:, 0]) / np.ptp(table_points[:, 0])) if np.ptp(
            table_points[:, 0]
        ) > 0 else 0.0
        return cls(matrix, len(pixels), ppm)

    def usable(self) -> bool:
        return self.samples > 0 and self.pixels_per_metre > 0

    def to_world(self, u: float, v: float) -> tuple[float, float]:
        """Pixel centroid -> ``(x, y)`` on the table plane."""
        import numpy as np

        point = self._h @ np.array([u, v, 1.0])
        return float(point[0] / point[2]), float(point[1] / point[2])

    def residual_px(self, pixels, table_points) -> float:
        """Largest centroid error, in pixels, over the samples it was fitted on."""
        import numpy as np

        pixels = np.asarray(pixels, dtype=np.float64)
        table_points = np.asarray(table_points, dtype=np.float64)
        worst = 0.0
        for (u, v) in pixels:
            x, y = self.to_world(u, v)
            back = self._h @ np.array([x, y, 1.0])
            bu, bv = back[0] / back[2], back[1] / back[2]
            worst = max(worst, float(np.hypot(bu - u, bv - v)))
        return worst