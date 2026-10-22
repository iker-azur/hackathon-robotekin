"""Segmentación por instancias de la cámara cenital: HSV con profundidad, YOLO y verdad del simulador.

Los tres producen lo mismo (``list[Instance]``) y comparten ``finalize``, que da a
cada máscara su color, forma, posición 3D, giro y visibilidad. Así cambiar de
segmentador (``--segmenter hsv`` | ``yolo:<pesos>``) no cambia nada aguas abajo, y
la política recibe máscaras con el mismo formato en el entrenamiento y en el uso.

* ``HsvSegmenter`` (por defecto, sin dependencias nuevas): el detector del demo 08,
  más una separación de manchas que antes no existía (piezas pegadas o apiladas).
* ``YoloSegmenter``: envuelve ``ultralytics`` (YOLO-seg ajustado con
  ``colab/01_entrenar_yolo_seg.ipynb`` o YOLOE con texto). **No hay en el repositorio
  ningún modelo YOLO entrenado**; hasta que exista, el demo usa HSV.
* ``ground_truth``: máscaras analíticas desde las poses del simulador. Sólo para
  etiquetar (dataset de YOLO) y evaluar; el agente nunca las usa.

Qué se ve y qué no, con una sola cámara cenital: la máscara es la parte *visible* de
la cara superior. Una pieza tapada por otra o por la pared del contenedor tiene
``visibility`` < 1 y su posición/giro son menos fiables. Una pieza completamente
debajo de otra no existe para esta cámara (``docs/03_limitaciones.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from . import attributes as A
from .config import COLORS, PIECE_SIZE_M
from .instances import Instance
from recoverykit import perception as P  # noqa: E402

TOP_FACE_M = 0.004            # grosor de la cara superior que cuenta (demo 08)
DEPTH_STEP_M = 0.006          # salto de profundidad entre píxeles vecinos que separa dos piezas
AIR_ABOVE_TABLE_M = 0.075     # una cara superior más alta que esto (dos capas = 0.054) está en el aire / es el brazo


# --------------------------------------------------------------- cámara
def overhead_camera(cam_cfg) -> P.Camera:
    """La ``P.Camera`` de una ``CameraCfg`` cenital, calculada (igual que la calibración que da Isaac)."""
    fx = cam_cfg.width * cam_cfg.focal_length_mm / cam_cfg.horizontal_aperture_mm
    k = np.array([[fx, 0, cam_cfg.width / 2], [0, fx, cam_cfg.height / 2], [0, 0, 1.0]])
    rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])      # derecha = +x, abajo = −y, mira a −z
    return P.Camera(k, rot, np.array(cam_cfg.pos_m, dtype=float), cam_cfg.width, cam_cfg.height)


@dataclass
class Context:
    """Lo que hay que saber de la escena para interpretar máscaras."""

    cam: P.Camera
    table_z: float
    region_fn: Callable[[float, float, float], str]
    piece_size_m: float = PIECE_SIZE_M

    def expected_top_px(self, depth_m: float, shape: str = "cube") -> float:
        side = self.piece_size_m * self.cam.px_per_m(depth_m)
        return side * side * (math.pi / 4 if shape == "cylinder" else 1.0)


class Segmenter(Protocol):
    name: str

    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> list[Instance]: ...


# ------------------------------------------------------------ separar manchas
def _edges_from_depth(comp: np.ndarray, depth: np.ndarray, step_m: float) -> np.ndarray:
    """Píxeles junto a un salto brusco de profundidad dentro de la mancha (borde entre piezas a distinta altura)."""
    d = np.where(comp & np.isfinite(depth), depth, np.nan)
    edge = np.zeros(comp.shape, bool)
    dx = np.abs(d[:, 1:] - d[:, :-1])
    dy = np.abs(d[1:, :] - d[:-1, :])
    jx = np.nan_to_num(dx, nan=0.0) > step_m
    jy = np.nan_to_num(dy, nan=0.0) > step_m
    edge[:, 1:] |= jx
    edge[:, :-1] |= jx
    edge[1:, :] |= jy
    edge[:-1, :] |= jy
    return edge


def _assign_to_nearest(comp: np.ndarray, labels: np.ndarray, n: int) -> list[np.ndarray]:
    """Reparte todos los píxeles de ``comp`` entre los ``n`` núcleos de ``labels`` (1..n) por cercanía."""
    seeds = (labels > 0).astype(np.uint8)
    inv = (1 - seeds).astype(np.uint8)
    _dist, near = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # etiqueta de píxel del núcleo más cercano → etiqueta de núcleo
    lut = np.zeros(int(near.max()) + 1, np.int32)
    lut[near[seeds > 0]] = labels[seeds > 0]
    owner = lut[near]
    return [comp & (owner == i) for i in range(1, n + 1)]


def _split_by_depth(comp: np.ndarray, depth: np.ndarray, exp_area: float) -> list[np.ndarray]:
    edge = _edges_from_depth(comp, depth, DEPTH_STEP_M)
    if not edge.any():
        return [comp]
    core = comp & ~edge
    n, lab = cv2.connectedComponents(core.astype(np.uint8), connectivity=4)
    # una pieza de abajo tapada a medias deja un fragmento pequeño: se conserva si tiene cuerpo de pieza
    keep = [i for i in range(1, n) if (lab == i).sum() >= 0.12 * exp_area]
    if len(keep) < 2:
        return [comp]
    remap = np.zeros(n, np.int32)
    for new, old in enumerate(keep, 1):
        remap[old] = new
    return _assign_to_nearest(comp, remap[lab], len(keep))


def _split_aligned(p: np.ndarray, n: int) -> list[np.ndarray] | None:
    """n piezas cuadradas pegadas en fila forman un rectángulo n:1: se parte por igual a lo largo."""
    cnts, _ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    long_, short = max(w, h), max(min(w, h), 1e-6)
    if cv2.contourArea(c) / (w * h + 1e-9) < 0.85 or abs(long_ / short - n) > 0.3 * n:
        return None
    # eje largo
    theta = math.radians(ang if w >= h else ang + 90.0)
    axis = np.array([math.cos(theta), math.sin(theta)])
    ys, xs = np.nonzero(p)
    t = (xs - cx) * axis[0] + (ys - cy) * axis[1]
    cuts = np.floor((t + long_ / 2) / (long_ / n)).clip(0, n - 1).astype(int)
    out = []
    for i in range(n):
        m = np.zeros_like(p)
        m[ys[cuts == i], xs[cuts == i]] = True
        out.append(m)
    return out


def _split_watershed(p: np.ndarray, n: int) -> list[np.ndarray]:
    dist = cv2.distanceTransform(p.astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return [p]
    labels, k = None, 0
    for frac in np.linspace(0.75, 0.35, 9):
        k, lab = cv2.connectedComponents((dist > frac * dist.max()).astype(np.uint8))
        if k - 1 >= n:
            labels = lab
            break
    if labels is None or k - 1 < 2:
        return [p]
    sizes = [(int((labels == i).sum()), i) for i in range(1, k)]
    sizes.sort(reverse=True)
    markers = np.zeros(p.shape, np.int32)
    markers[~p] = 1
    for new, (_s, old) in enumerate(sizes[:n], 2):
        markers[labels == old] = new
    img = cv2.cvtColor((255 - 255 * dist / dist.max()).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)
    out = [p & (markers == i) for i in range(2, 2 + min(n, len(sizes)))]
    return [m for m in out if m.sum() > 0] or [p]


def _split_kmeans(p: np.ndarray, n: int) -> list[np.ndarray]:
    """Reparto de la mancha en ``n`` grupos compactos de píxeles (k-means sobre las coordenadas, determinista)."""
    ys, xs = np.nonzero(p)
    if len(xs) < n:
        return [p]
    pts = np.stack([xs, ys], 1).astype(np.float32)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _c, lab, _centers = cv2.kmeans(pts, n, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    out = []
    for i in range(n):
        m = np.zeros_like(p)
        sel = lab.ravel() == i
        m[ys[sel], xs[sel]] = True
        if m.sum() > 0:
            out.append(m)
    return out or [p]


def split_blob(comp: np.ndarray, depth: np.ndarray, exp_area: float) -> list[np.ndarray]:
    """Una mancha de un color → una máscara por pieza.

    1) los saltos de profundidad separan piezas a distinta altura (apiladas);
    2) una mancha mucho mayor que una pieza se parte: en partes iguales si es un
       rectángulo n:1, por watershed sobre la transformada de distancia si no.
    Es heurístico y su límite está medido en ``tests/test_segmentation.py``: piezas
    pegadas del mismo color, a la misma altura y con las caras alineadas en L con
    tres o más piezas se separan peor.
    """
    out: list[np.ndarray] = []
    for p in _split_by_depth(comp, depth, exp_area):
        area = float(p.sum())
        n = int(round(area / exp_area))
        if n >= 2 and area > 1.6 * exp_area:
            parts = _split_aligned(p, n) or _split_watershed(p, n)
            if len(parts) < 2:
                # dos cubos pegados y escalonados no tienen costura de color ni dos picos de distancia (medido en renders
                # de Isaac): lo único que queda es repartir la mancha en n grupos compactos
                parts = _split_kmeans(p, n)
            out.extend(parts)
        else:
            out.append(p)
    return out


# -------------------------------------------------------------- finalizar
def finalize(rgb: np.ndarray, depth: np.ndarray, ctx: Context, masks: list[np.ndarray],
             scores: list[float] | None = None, colors: list[str | None] | None = None,
             source: str = "", ids: list[str] | None = None) -> list[Instance]:
    """Máscaras visibles → ``Instance`` con color, forma, posición 3D, giro y visibilidad."""
    cam = ctx.cam
    d_table = float(cam.pos[2] - ctx.table_z)
    min_area = 0.25 * ctx.expected_top_px(d_table - ctx.piece_size_m)
    out: list[Instance] = []
    for i, m in enumerate(masks):
        m = m.astype(bool)
        area = int(m.sum())
        if area < min_area:
            continue
        dv = depth[m]
        ok = np.isfinite(dv) & (dv > 0)
        if not ok.any():
            continue
        d_near = float(np.percentile(dv[ok], 5))
        top = m & np.isfinite(depth) & (depth < d_near + TOP_FACE_M)
        if top.sum() < 0.25 * area:
            top = m
        ys, xs = np.nonzero(top)
        d_top = float(np.median(depth[top][np.isfinite(depth[top])]))
        u, v = float(xs.mean()), float(ys.mean())
        top_pt = cam.backproject(u, v, d_top)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
        (_, _), (w_px, h_px), _ = cv2.minAreaRect(hull)
        fill = cv2.contourArea(hull) / max(w_px * h_px, 1e-6)
        _, _, ang = cv2.minAreaRect(hull)
        ppm = cam.px_per_m(d_top)
        width_m = min(w_px, h_px) / ppm
        shape = A.classify_shape_from_fill(fill)
        expected = ctx.expected_top_px(d_top, shape if shape != "?" else "cube")
        visibility = float(min(1.0, top.sum() / expected))
        if visibility < 0.7:
            shape = "?"                      # una pieza tapada no enseña su contorno: no se adivina la forma
        color, conf = (colors[i], 1.0) if colors is not None and colors[i] else A.classify_color(rgb, m)
        top_z = float(top_pt[2])
        pos = np.array([top_pt[0], top_pt[1], top_z - ctx.piece_size_m / 2])
        airborne = top_z - ctx.table_z > AIR_ABOVE_TABLE_M
        region = "aire" if airborne else ctx.region_fn(float(pos[0]), float(pos[1]), top_z)
        yaw = P._image_yaw_to_world(math.radians(ang), cam) if shape == "cube" else 0.0
        out.append(Instance(idx=len(out), mask=m, score=float(scores[i]) if scores else 1.0, color=color,
                            color_conf=float(conf), shape=shape, pixel=(u, v), area_px=area, pos=pos, top_z=top_z,
                            yaw=yaw, width_m=float(width_m), visibility=visibility, region=region, source=source,
                            track=ids[i] if ids else None))
    return out


# ------------------------------------------------------------------- HSV
class HsvSegmenter:
    """Color saturado + profundidad. El color de cada instancia es el del canal que la produjo."""

    name = "hsv"

    def __init__(self, ctx: Context, colors: tuple[str, ...] = COLORS):
        self.ctx = ctx
        self.colors = colors

    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> list[Instance]:
        cam = self.ctx.cam
        d_top_nominal = float(cam.pos[2] - self.ctx.table_z - self.ctx.piece_size_m)
        exp_area = self.ctx.expected_top_px(d_top_nominal)
        masks: list[np.ndarray] = []
        colors: list[str] = []
        for color, raw in P.color_masks(rgb).items():
            if color not in self.colors:
                continue
            cleaned = P._clean(raw)
            n, labels = cv2.connectedComponents(cleaned)
            for i in range(1, n):
                comp = labels == i
                if comp.sum() < 0.25 * exp_area:
                    continue
                for part in split_blob(comp, depth, exp_area):
                    masks.append(part)
                    colors.append(color)
        return finalize(rgb, depth, self.ctx, masks, colors=colors, source="hsv")


# ------------------------------------------------------------------ YOLO
class YoloSegmenter:
    """Instancias de una red YOLO de segmentación (ultralytics, carga perezosa).

    ``predict_fn(rgb) -> [(mask bool HxW, score)]`` se puede inyectar (pruebas); por
    defecto se construye con ``ultralytics.YOLO(weights)`` (o ``YOLOE`` si se dan
    ``prompts``). El color y el resto de atributos NO los decide la red: se miden en la
    máscara (``attributes.classify_color``), de modo que la red sólo tiene que
    encontrar «piezas» y se puede reutilizar con otros colores.

    Ejemplo para el cuaderno de Colab::

        YoloSegmenter(ctx, weights="piezas_seg.pt")
        YoloSegmenter(ctx, weights="yoloe-11s-seg.pt", prompts=["industrial part on a tray"])  # sin entrenar
    """

    def __init__(self, ctx: Context, weights: str | None = None, prompts: list[str] | None = None,
                 conf: float = 0.25, imgsz: int = 800, device: str = "cpu",
                 predict_fn: Callable[[np.ndarray], list[tuple[np.ndarray, float]]] | None = None):
        self.ctx = ctx
        self.name = f"yolo:{weights}" if weights else "yolo"
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self._predict = predict_fn or self._build_predictor(weights, prompts)

    def _build_predictor(self, weights, prompts):
        if not weights:
            raise ValueError("YoloSegmenter necesita 'weights' o 'predict_fn'")
        try:
            from ultralytics import YOLO, YOLOE
        except ImportError as exc:  # noqa: BLE001
            raise ImportError("falta ultralytics: pip install ultralytics --no-deps (ver README, apartado YOLO)") from exc
        if prompts:
            model = YOLOE(weights)
            model.set_classes(prompts, model.get_text_pe(prompts))
        else:
            model = YOLO(weights)

        def predict(rgb):
            res = model.predict(np.ascontiguousarray(rgb[..., ::-1]), conf=self.conf, imgsz=self.imgsz,
                                device=self.device, retina_masks=True, verbose=False)[0]
            if res.masks is None:
                return []
            data = res.masks.data.cpu().numpy() > 0.5
            scores = res.boxes.conf.cpu().numpy()
            return [(data[i], float(scores[i])) for i in range(len(scores))]

        return predict

    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> list[Instance]:
        found = sorted(self._predict(rgb), key=lambda t: -t[1])
        taken = np.zeros(rgb.shape[:2], bool)
        masks, scores = [], []
        for m, s in found:                       # sin solapes: gana la de más confianza
            m = np.asarray(m, bool) & ~taken
            if m.sum() == 0:
                continue
            taken |= m
            masks.append(m)
            scores.append(s)
        return finalize(rgb, depth, self.ctx, masks, scores=scores, source="yolo")


def make_segmenter(spec: str, ctx: Context):
    """``hsv`` | ``yolo:<pesos.pt>`` | ``yoloe:<pesos.pt>`` (YOLOE con el texto «industrial part»)."""
    if spec == "hsv":
        return HsvSegmenter(ctx)
    kind, _, weights = spec.partition(":")
    if kind == "yolo" and weights:
        return YoloSegmenter(ctx, weights=weights)
    if kind == "yoloe" and weights:
        return YoloSegmenter(ctx, weights=weights, prompts=["industrial part", "cube", "cylinder"])
    raise ValueError(f"segmentador desconocido: {spec!r} (usa hsv, yolo:<pesos> o yoloe:<pesos>)")


# ---------------------------------------------------- verdad del simulador
@dataclass
class PieceTruth:
    id: str
    color: str
    shape: str
    size: float
    height: float
    pos: np.ndarray            # centro, mundo
    quat: np.ndarray           # (w, x, y, z)


def _rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def silhouette(cam: P.Camera, t: PieceTruth) -> np.ndarray:
    """Silueta (bool HxW) de una pieza proyectada a la imagen, sin oclusiones."""
    r = _rot(t.quat)
    if t.shape == "cylinder":
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        loc = np.array([[t.size / 2 * math.cos(a), t.size / 2 * math.sin(a), s * t.height / 2]
                        for a in th for s in (-1, 1)])
    else:
        loc = np.array([[sx * t.size / 2, sy * t.size / 2, sz * t.height / 2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pts = loc @ r.T + t.pos
    uv, z = cam.project(pts)
    m = np.zeros((cam.height, cam.width), np.uint8)
    if (z <= 0).any():
        return m.astype(bool)
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def ground_truth(cam: P.Camera, pieces: list[PieceTruth]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(mapa de etiquetas, siluetas completas).

    El mapa vale ``i + 1`` en los píxeles visibles de ``pieces[i]`` (0 = fondo); las más altas se
    pintan al final, así que tapan a las bajas. Las siluetas dicen cuánto de cada pieza se ve.
    """
    labels = np.zeros((cam.height, cam.width), np.int32)
    sils: dict[str, np.ndarray] = {}
    order = sorted(range(len(pieces)), key=lambda i: pieces[i].pos[2])
    for i in order:
        s = silhouette(cam, pieces[i])
        sils[pieces[i].id] = s
        labels[s] = i + 1
    return labels, sils


def gt_instances(rgb: np.ndarray, depth: np.ndarray, ctx: Context, pieces: list[PieceTruth]) -> list[Instance]:
    """Instancias «perfectas» (máscaras analíticas + color y forma reales) para evaluar otros segmentadores."""
    labels, sils = ground_truth(ctx.cam, pieces)
    masks = [labels == i + 1 for i in range(len(pieces))]
    return finalize(rgb, depth, ctx, masks, colors=[p.color for p in pieces], source="gt",
                    ids=[p.id for p in pieces])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0
