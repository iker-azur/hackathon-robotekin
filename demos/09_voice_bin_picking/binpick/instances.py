"""Lo que la percepción entrega: una lista de instancias, una por pieza visible.

Es el contrato entre segmentación (HSV, YOLO, verdad del simulador), selección,
seguimiento, política y panel. Un segmentador nuevo sólo tiene que producir
máscaras; ``segmentation.finalize`` las convierte en ``Instance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instance:
    """Una pieza vista en una imagen.

    ``mask`` es la parte **visible** (otra pieza encima la recorta). ``pos`` es el
    centro estimado de la pieza en el mundo; ``top_z`` la altura de su cara
    superior medida con profundidad; ``visibility`` la fracción de su cara
    superior esperada que se ve (1 = entera, <0.6 = muy tapada).
    """

    idx: int                                   # índice dentro de esta imagen
    mask: np.ndarray                           # (H, W) bool
    score: float = 1.0
    color: str | None = None
    color_conf: float = 0.0
    shape: str | None = None                   # "cube" | "cylinder" | "?"
    attributes: dict = field(default_factory=dict)   # otros atributos registrados
    pixel: tuple[float, float] = (0.0, 0.0)    # centroide (u, v) de la máscara
    area_px: int = 0
    pos: np.ndarray | None = None              # (x, y, z_centro) en el mundo
    top_z: float | None = None
    yaw: float = 0.0                           # giro de una cara, rad en [-π/4, π/4)
    width_m: float = 0.0
    visibility: float = 1.0
    region: str = "mesa"                       # origen | izquierda | derecha | mesa | aire
    source: str = ""                           # hsv | yolo | gt
    track: str | None = None                   # identificador estable (lo pone el tracker)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @property
    def occlusion(self) -> float:
        return float(max(0.0, 1.0 - self.visibility))

    def label(self) -> str:
        from .config import COLOR_ES_M
        shape = {"cube": "cubo", "cylinder": "cilindro"}.get(self.shape or "", "pieza")
        color = COLOR_ES_M.get(self.color or "", ("", ""))[0] if self.color else "sin color"
        return f"{shape} {color}".strip()

    def as_dict(self) -> dict:
        """Versión ligera (sin máscara) para el panel y los informes."""
        return {"id": self.track or f"#{self.idx}", "color": self.color, "shape": self.shape, "region": self.region,
                "pos": None if self.pos is None else [round(float(v), 4) for v in self.pos],
                "top_z": None if self.top_z is None else round(float(self.top_z), 4),
                "yaw_deg": round(float(np.degrees(self.yaw)), 1), "visibility": round(float(self.visibility), 2),
                "score": round(float(self.score), 2), "pixel": [round(float(self.pixel[0]), 1), round(float(self.pixel[1]), 1)],
                "source": self.source}
