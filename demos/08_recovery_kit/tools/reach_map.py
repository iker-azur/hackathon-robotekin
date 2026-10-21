"""Mapa de alcance con la pinza vertical (sin Isaac).

    python tools/reach_map.py

Para cada punto (radio, ángulo) alrededor de la base y cada altura del ciclo
(agarre, aproximación, subida, tránsito) comprueba con la cinemática inversa si
algún giro de la pinza compatible con un cubo (cualquiera de 4 orientaciones
probadas) llega. De aquí sale la corona de trabajo de ``config.WorkArea``.
Escribe ``results/reach_map.json`` y lo dibuja en la consola.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import config, kinematics as K, skills  # noqa: E402
from recoverykit.workcell import base_transform  # noqa: E402

HEIGHTS = {"agarre": 0.0125, "aproximación": 0.065, "subida": 0.0775, "tránsito": 0.09}


def main() -> int:
    cfg = config.preset("demo")
    base = base_transform(cfg)
    top = cfg.table.top_z_m
    local = K.tcp_local(0.025)
    radii = [round(r, 3) for r in np.arange(0.11, 0.271, 0.01)]
    angles = list(range(-75, 76, 15))
    seed = np.array(cfg.robot.start_joints[:5])
    reach = {}
    for name, h in HEIGHTS.items():
        grid = []
        for r in radii:
            row = []
            for a in angles:
                p = np.array([r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), top + h])
                ok = True
                for cube_yaw in (0.0, 22.5, 45.0, 67.5):
                    yaws = [math.radians(cube_yaw) + k * math.pi / 2 for k in range(4)]
                    res, _ = skills.solve_topdown(base, p, yaws, local, seed)
                    if res is None:
                        ok = False
                        break
                row.append(ok)
            grid.append(row)
        reach[name] = grid
    all_h = [[all(reach[n][i][j] for n in HEIGHTS) for j in range(len(angles))] for i in range(len(radii))]
    out = {"heights_m": HEIGHTS, "radii_m": radii, "angles_deg": angles, "reachable": reach, "all_heights": all_h,
           "work_area": cfg.to_dict()["work_area"]}
    (ROOT / "results" / "reach_map.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("radio  " + " ".join(f"{a:>4d}" for a in angles) + "   (# = alcanza a todas las alturas)")
    for i, r in enumerate(radii):
        print(f"{r:.2f}   " + " ".join("   #" if v else "   ." for v in all_h[i]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
