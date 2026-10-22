"""Imágenes para el panel: la cámara cenital girada, con cajas, piezas y segmentación dibujadas.

La imagen cruda de la cámara tiene +x a la derecha y +y arriba, así que la **izquierda del robot** (+y)
queda arriba. El panel la gira 90° (antihorario) para que el robot «mire hacia arriba» y su izquierda
esté a la izquierda de la pantalla, que es la que usan las órdenes («caja izquierda»). El texto se dibuja
*después* de girar, para que se lea derecho.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config as C
from .instances import Instance

COLOR_BGR = {"red": (40, 40, 230), "green": (60, 190, 60), "blue": (230, 120, 30)}
NEUTRAL_BGR = (200, 200, 200)
LABELS = {C.BIN_ID: "CONTENEDOR", "caja_izquierda": "IZQUIERDA", "caja_derecha": "DERECHA"}


def rotate_view(img: np.ndarray) -> np.ndarray:
    """Gira la imagen de la cámara 90° antihorario: (+x arriba, +y izquierda)."""
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rot_pt(u: float, v: float, width: int) -> tuple[int, int]:
    """Píxel (u, v) de la imagen original → píxel de la imagen girada con ``rotate_view``."""
    return int(round(v)), int(round(width - 1 - u))


def _container_polys(scene, cam) -> dict[str, np.ndarray]:
    out = {}
    for c in scene.containers:
        z = scene.table.top_z_m + c.outer_m[2]
        cx, cy = c.center_xy_m
        pts = np.array([[cx + sx * c.outer_m[0] / 2, cy + sy * c.outer_m[1] / 2, z]
                        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        uv, _ = cam.project(pts)
        out[c.id] = uv
    return out


def annotate(rgb: np.ndarray, instances: list[Instance], scene, cam, mode: str = "rgb",
             target_track: str | None = None, rotate: bool = True) -> np.ndarray:
    """BGR listo para JPEG. ``mode``: ``rgb`` (marcas sobre la imagen) o ``seg`` (máscaras de instancia de color)."""
    h, w = rgb.shape[:2]
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if mode == "seg":
        layer = np.zeros_like(img)
        solid = np.zeros((h, w), bool)
        for i in instances:
            layer[i.mask] = COLOR_BGR.get(i.color or "", NEUTRAL_BGR)
            solid |= i.mask
        img = np.where(solid[..., None], (0.35 * img + 0.65 * layer).astype(np.uint8), (0.45 * img).astype(np.uint8))
    for i in instances:
        if i.track is not None and i.track == target_track:
            cnts, _ = cv2.findContours(i.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 255), 3)
    polys = _container_polys(scene, cam)
    if rotate:
        img = rotate_view(img)
        pt = lambda u, v: rot_pt(u, v, w)  # noqa: E731
    else:
        pt = lambda u, v: (int(round(u)), int(round(v)))  # noqa: E731
    for cid, uv in polys.items():
        pts = np.array([pt(*p) for p in uv], np.int32)
        cv2.polylines(img, [pts], True, (240, 240, 240), 2)
        cx, cy = pts.mean(0)
        label = LABELS.get(cid, cid)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(img, label, (int(cx - size[0] / 2), int(pts[:, 1].min() - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    for i in instances:
        u, v = pt(*i.pixel)
        col = COLOR_BGR.get(i.color or "", NEUTRAL_BGR)
        if mode == "rgb":
            cv2.circle(img, (u, v), 22, col, 2, cv2.LINE_AA)
        tag = (i.track or f"#{i.idx}") + ("" if i.visibility > 0.85 else f" {i.visibility * 100:.0f}%")
        cv2.putText(img, tag, (u + 20, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def to_jpeg(bgr: np.ndarray, quality: int = 80) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def plain_jpeg(rgb: np.ndarray, rotate: bool = False, quality: int = 80) -> bytes | None:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return to_jpeg(rotate_view(img) if rotate else img, quality)
