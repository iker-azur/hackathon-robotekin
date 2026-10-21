"""Mallas sencillas del SO-101 para la vista 3D del panel (sin Isaac).

    python tools/build_robot_meshes.py

Una envolvente convexa por eslabón, a partir de los colisionadores extraídos
del USD (``assets/so101/colliders.npz``), en el marco de cada eslabón. Escribe
``server/static/robot_meshes.json``. Sólo es para dibujar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import kinematics as K  # noqa: E402


def main() -> int:
    out = {}
    for link, pts in K.colliders().items():
        hull = ConvexHull(pts)
        used = np.unique(hull.simplices)
        remap = {int(o): i for i, o in enumerate(used)}
        faces = [[remap[int(a)] for a in tri] for tri in hull.simplices]
        verts = pts[used]
        center = verts.mean(0)
        # caras con la normal hacia fuera
        for f in faces:
            a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
            if np.dot(np.cross(b - a, c - a), a - center) < 0:
                f[1], f[2] = f[2], f[1]
        out[link] = {"v": np.round(verts, 4).ravel().tolist(), "f": np.asarray(faces).ravel().tolist()}
        print(f"{link}: {len(verts)} vértices, {len(faces)} caras")
    path = ROOT / "server" / "static" / "robot_meshes.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"-> {path} ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
