"""Percepción con la cámara cenital RGB-D: piezas, caja y contenido de la caja.

Entrada: imagen RGB, profundidad (distancia al plano de imagen) y la
calibración de la cámara (intrínsecos y pose en el mundo). Salida: lista de
detecciones en coordenadas del mundo. Nada de aquí lee el simulador; con una
cámara RGB-D calibrada sobre la mesa real funciona igual.

Pasos:
1. HSV: los colores de las piezas están muy saturados; la mesa gris, la caja
   marrón y el robot amarillo pálido no llegan al umbral de saturación.
2. Componentes conexas, filtro por área esperada a la distancia de la mesa.
3. Profundidad mediana de la cara superior → punto 3D → mundo.
4. Forma por relleno del rectángulo mínimo (cuadrado ≈ 1, círculo ≈ π/4,
   triángulo ≈ 1/2) y giro por ``cv2.minAreaRect``.
5. Dentro o fuera de la caja por la huella interior de la caja detectada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# tono OpenCV (0–180) de referencia por color; medido en tools/capture_perception.py
# (rojo 1, verde 68, azul 106–107, amarillo 29). El SO-101 es amarillo
# (tono 28, saturación hasta 179): el amarillo sólo se separa con la máscara del brazo.
HUES = {"red": 0.0, "yellow": 29.0, "green": 67.0, "blue": 107.0}
HUE_TOL = 14.0
MIN_SAT = 150
MIN_VAL = 50
BOX_HUE = (5.0, 26.0)
BOX_SAT = (60, 175)
BOX_MIN_VAL = 60
TOP_FACE_M = 0.004
PARTIAL_FRACTION = 0.8


@dataclass
class Camera:
    """Cámara pinhole calibrada. ``rot`` lleva del marco óptico (x der., y abajo, z delante) al mundo."""

    k: np.ndarray
    rot: np.ndarray
    pos: np.ndarray
    width: int
    height: int

    def backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        x = (u - self.k[0, 2]) * depth / self.k[0, 0]
        y = (v - self.k[1, 2]) * depth / self.k[1, 1]
        return self.rot @ np.array([x, y, depth]) + self.pos

    def project(self, pts_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Puntos del mundo (N×3) → píxeles (N×2) y profundidad (N)."""
        pc = (pts_w - self.pos) @ self.rot
        z = pc[:, 2]
        uv = np.stack([self.k[0, 0] * pc[:, 0] / z + self.k[0, 2], self.k[1, 1] * pc[:, 1] / z + self.k[1, 2]], 1)
        return uv, z

    def px_per_m(self, distance: float) -> float:
        return self.k[0, 0] / distance


@dataclass
class Detection:
    color: str
    shape: str              # "cube" | "cylinder" | "prism" | "?"
    pos: np.ndarray         # centro estimado de la pieza (mundo)
    yaw: float              # giro de una cara (cubos); 0 para cilindros
    width_m: float
    height_m: float
    area_px: int
    pixel: tuple[float, float]
    in_box: bool = False
    box_id: str | None = None
    merged: int = 1         # >1 si la mancha parece contener varias piezas juntas
    fill: float = 0.0
    airborne: bool = False
    partial: bool = False   # sólo se ve una parte: posición y giro poco fiables

    def as_dict(self) -> dict:
        return {"color": self.color, "shape": self.shape, "pos": [round(float(v), 4) for v in self.pos],
                "yaw_deg": round(math.degrees(self.yaw), 1), "width_mm": round(self.width_m * 1000, 1),
                "height_mm": round(self.height_m * 1000, 1), "in_box": self.in_box, "box": self.box_id,
                "merged": self.merged, "pixel": [round(self.pixel[0], 1), round(self.pixel[1], 1)]}


@dataclass
class BoxDetection:
    id: str
    center: np.ndarray       # centro del suelo interior (mundo)
    yaw: float
    outer_m: tuple[float, float]
    inner_m: tuple[float, float]
    rim_z: float
    floor_z: float

    def contains_xy(self, p: np.ndarray, margin: float = 0.0) -> bool:
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        dx, dy = p[0] - self.center[0], p[1] - self.center[1]
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= self.inner_m[0] / 2 - margin and abs(ly) <= self.inner_m[1] / 2 - margin


@dataclass
class Observation:
    pieces: list[Detection] = field(default_factory=list)
    boxes: list[BoxDetection] = field(default_factory=list)
    occluded_px: int = 0
    notes: list[str] = field(default_factory=list)

    def resting(self) -> list[Detection]:
        return [d for d in self.pieces if not d.airborne]

    def on_table(self) -> list[Detection]:
        return [d for d in self.resting() if not d.in_box]

    def in_box(self, box_id: str | None = None) -> list[Detection]:
        return [d for d in self.resting() if d.in_box and (box_id is None or d.box_id == box_id)]


# --------------------------------------------------------------- segmentación
def color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32)
    sat_ok = (hsv[..., 1] >= MIN_SAT) & (hsv[..., 2] >= MIN_VAL)
    masks = {}
    for name, hue in HUES.items():
        d = np.abs(h - hue)
        d = np.minimum(d, 180 - d)
        masks[name] = ((d <= HUE_TOL) & sat_ok).astype(np.uint8) * 255
    return masks


def box_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= BOX_HUE[0]) & (hsv[..., 0] <= BOX_HUE[1]) & (hsv[..., 1] >= BOX_SAT[0])
         & (hsv[..., 1] <= BOX_SAT[1]) & (hsv[..., 2] >= BOX_MIN_VAL))
    return m.astype(np.uint8) * 255


def _clean(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def classify_shape(contour: np.ndarray) -> tuple[str, float, float]:
    """(forma, relleno del rectángulo mínimo, giro de una arista en la imagen, rad).

    Medido sobre 127 piezas (results/perception_eval.json): cubos enteros
    0.97–0.99, cilindros 0.78–0.82. Los cubos tapados en parte dan valores
    intermedios; por eso ``detect`` marca antes las manchas demasiado estrechas.
    """
    area = cv2.contourArea(contour)
    (_, _), (w, h), ang = cv2.minAreaRect(contour)
    fill = area / max(w * h, 1e-6)
    yaw = math.radians(ang)
    if fill > 0.845:
        return "cube", fill, yaw
    if fill > 0.75:
        return "cylinder", fill, 0.0
    if fill > 0.40:
        return "prism", fill, yaw
    return "?", fill, yaw


# ------------------------------------------------------------------ detección
@dataclass
class Expectation:
    """Lo que la percepción sabe de antemano: tamaños posibles, la mesa y la caja."""

    table_z: float
    piece_sizes_m: tuple[float, ...] = (0.025,)
    box_outer_m: tuple[float, float, float] = (0.11, 0.11, 0.045)
    box_wall_m: float = 0.004
    box_floor_m: float = 0.004
    # colores del catálogo de la célula: un color que no está no se busca
    colors: tuple[str, ...] = tuple(HUES)


def detect_boxes(rgb: np.ndarray, depth: np.ndarray, cam: Camera, exp: Expectation) -> list[BoxDetection]:
    m = _clean(box_mask(rgb))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    z_rim = exp.table_z + exp.box_outer_m[2]
    ppm = cam.px_per_m(float(cam.pos[2] - z_rim))
    expected = (exp.box_outer_m[0] * ppm) * (exp.box_outer_m[1] * ppm)
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.25 * expected:
            continue
        ys, xs = np.nonzero(labels == i)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.float32))
        (u, v), (w, h), ang = cv2.minAreaRect(hull)
        if not 0.6 < (w * h) / expected < 1.6:
            continue
        c = cam.backproject(u, v, float(cam.pos[2] - z_rim))
        outer = exp.box_outer_m[:2]
        inner = (outer[0] - 2 * exp.box_wall_m, outer[1] - 2 * exp.box_wall_m)
        boxes.append(BoxDetection("caja" if not boxes else f"caja_{len(boxes)+1}",
                                  np.array([c[0], c[1], exp.table_z + exp.box_floor_m]),
                                  _image_yaw_to_world(math.radians(ang), cam), outer, inner, z_rim,
                                  exp.table_z + exp.box_floor_m))
    return boxes


def detect(rgb: np.ndarray, depth: np.ndarray, cam: Camera, exp: Expectation,
           robot_px: np.ndarray | None = None) -> Observation:
    obs = Observation()
    obs.boxes = detect_boxes(rgb, depth, cam, exp)
    d_table = float(cam.pos[2] - exp.table_z)
    nominal = max(exp.piece_sizes_m)
    expected_area = (nominal * cam.px_per_m(d_table)) ** 2
    min_area = 0.3 * expected_area * (min(exp.piece_sizes_m) / nominal) ** 2
    if robot_px is not None:
        obs.occluded_px = int(np.count_nonzero(robot_px))
    for color, raw in color_masks(rgb).items():
        if color not in exp.colors:
            continue
        m = _clean(raw)
        if robot_px is not None:
            m[robot_px > 0] = 0
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comp = labels == i
            dvals = depth[comp]
            ok = np.isfinite(dvals) & (dvals > 0)
            if not ok.any():
                continue
            # la cámara ve también las caras laterales (perspectiva): sólo cuentan los
            # píxeles de la cara superior, a menos de TOP_FACE_M de la profundidad mínima
            d_near = float(np.percentile(dvals[ok], 5))
            top = comp & (depth < d_near + TOP_FACE_M)
            if np.count_nonzero(top) < 0.25 * area:
                top = comp
            ys, xs = np.nonzero(top)
            contour = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
            shape, fill, yaw_img = classify_shape(contour)
            d_top = float(np.median(depth[top]))
            u, v = float(xs.mean()), float(ys.mean())
            top_pt = cam.backproject(u, v, d_top)
            merged = max(1, int(round(np.count_nonzero(top) / (expected_area * (d_table / d_top) ** 2))))
            (_, _), (w_px, h_px), _ = cv2.minAreaRect(contour)
            width = min(w_px, h_px) / cam.px_per_m(d_top)
            partial = width < PARTIAL_FRACTION * min(exp.piece_sizes_m)
            if partial:
                # tapada en parte (brazo, borde de la imagen): la forma no es fiable
                shape = "?"
            det = Detection(color, shape, top_pt.copy(), _image_yaw_to_world(yaw_img, cam) if shape == "cube" else 0.0,
                            width, 0.0, area, (u, v), merged=merged, fill=fill, partial=partial)
            support_z = exp.table_z
            for b in obs.boxes:
                if b.contains_xy(top_pt, margin=-0.003):
                    det.in_box, det.box_id, support_z = True, b.id, b.floor_z
                    break
            det.height_m = max(0.0, float(top_pt[2]) - support_z)
            det.pos[2] = support_z + det.height_m / 2
            # más alta que cualquier pieza apoyada: está en el aire (en la pinza) o es el brazo
            det.airborne = det.height_m > max(exp.piece_sizes_m) + 0.015
            obs.pieces.append(det)
    return obs


def _image_yaw_to_world(yaw_img: float, cam: Camera) -> float:
    """Giro de una arista en la imagen → giro en el mundo (cámara mirando hacia abajo).

    Un cuadrado es simétrico a 90°: se devuelve en [−45°, 45°).
    """
    d_w = cam.rot @ np.array([math.cos(yaw_img), math.sin(yaw_img), 0.0])
    a = math.atan2(d_w[1], d_w[0])
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def robot_pixels(cam: Camera, points_w: np.ndarray, dilate_px: int = 9) -> np.ndarray:
    """Máscara de los píxeles que tapa el brazo, a partir de sus colisionadores proyectados."""
    m = np.zeros((cam.height, cam.width), np.uint8)
    uv, z = cam.project(points_w)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    uv = uv[ok].astype(np.int32)
    m[uv[:, 1], uv[:, 0]] = 255
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)))
    return m
