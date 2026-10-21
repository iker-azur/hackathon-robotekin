"""Puntúa la percepción sobre los fotogramas grabados (sin Isaac).

    python tools/eval_perception.py

Empareja cada detección con la pieza verdadera del mismo color más cercana y
mide: piezas encontradas, falsas detecciones, error de posición, forma, giro y
si está dentro o fuera de la caja. Las piezas tapadas por el brazo (según su
proyección) se cuentan aparte. Escribe ``results/perception_eval.json``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import config, kinematics as K, perception as P  # noqa: E402
from recoverykit.workcell import base_transform  # noqa: E402

MATCH_M = 0.02


def arm_points(q: list[float], base: np.ndarray) -> np.ndarray:
    poses = K.fk(dict(zip(K.ARM, q)), base)
    pts = []
    for link in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"):
        v = K.colliders()[link][::3]
        t = poses[link]
        pts.append(v @ t[:3, :3].T + t[:3, 3])
    return np.concatenate(pts)


def main() -> int:
    cfg = config.preset("demo")
    base = base_transform(cfg)
    frames = sorted((ROOT / "results" / "perception_frames").glob("frame_*.npz"))
    if not frames:
        print("no hay fotogramas: ejecuta antes tools/capture_perception.py")
        return 1
    rows, errs, yaw_errs = [], [], []
    totals = {"truth": 0, "visible": 0, "found": 0, "false": 0, "shape_ok": 0, "shape_unknown": 0,
              "shape_wrong": 0, "box_ok": 0, "occluded": 0}
    for f in frames:
        d = np.load(f)
        meta = json.loads(str(d["truth_json"]))
        rgb, depth = d["rgb"], d["depth"]
        cam = P.Camera(d["k"], d["rot"], d["pos"], rgb.shape[1], rgb.shape[0])
        exp = P.Expectation(table_z=meta["table_z"], colors=tuple(sorted({p["color"] for p in meta["pieces"]})))
        mask = P.robot_pixels(cam, arm_points(meta["arm_q"], base))
        obs = P.detect(rgb, depth, cam, exp, robot_px=mask)
        obs.pieces = obs.resting()
        used = set()
        for t in meta["pieces"]:
            pos = np.array(t["pos"])
            if pos[2] < meta["table_z"] - 0.1:
                continue    # aparcada fuera de la escena
            totals["truth"] += 1
            uv, _ = cam.project(pos[None])
            u, v = int(uv[0, 0]), int(uv[0, 1])
            hidden = 0 <= u < cam.width and 0 <= v < cam.height and mask[v, u] > 0
            cands = sorted(((i, float(np.linalg.norm(o.pos[:2] - pos[:2]))) for i, o in enumerate(obs.pieces)
                            if o.color == t["color"] and i not in used), key=lambda c: c[1])
            if hidden:
                totals["occluded"] += 1
                if cands and cands[0][1] < MATCH_M:
                    used.add(cands[0][0])
                continue
            totals["visible"] += 1
            if not cands or cands[0][1] > MATCH_M:
                rows.append({"frame": f.name, "id": t["id"], "result": "no encontrada"})
                continue
            i, _ = cands[0]
            used.add(i)
            o = obs.pieces[i]
            totals["found"] += 1
            errs.append(float(np.linalg.norm(o.pos - pos)))
            totals["shape_ok"] += o.shape == t["shape"]
            totals["shape_unknown"] += o.shape == "?"
            totals["shape_wrong"] += o.shape not in ("?", t["shape"])
            totals["box_ok"] += o.in_box == t["in_box"]
            if t["shape"] == "cube":
                e = (math.radians(t["yaw_deg"]) - o.yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
                yaw_errs.append(abs(math.degrees(e)))
            rows.append({"frame": f.name, "id": t["id"], "result": "ok", "err_mm": round(errs[-1] * 1000, 2),
                         "shape": o.shape, "in_box": o.in_box})
        false = [o.as_dict() for k, o in enumerate(obs.pieces) if k not in used]
        totals["false"] += len(false)
        for fd in false:
            rows.append({"frame": f.name, "result": "falsa", "det": fd})
    summary = {
        "frames": len(frames),
        "recall_visible": totals["found"] / max(1, totals["visible"]),
        "false_detections": totals["false"],
        "occluded_by_arm": totals["occluded"],
        "position_error_mm_mean": 1000 * float(np.mean(errs)) if errs else None,
        "position_error_mm_p95": 1000 * float(np.percentile(errs, 95)) if errs else None,
        "position_error_mm_max": 1000 * float(np.max(errs)) if errs else None,
        "shape_accuracy": totals["shape_ok"] / max(1, totals["found"]),
        "in_box_accuracy": totals["box_ok"] / max(1, totals["found"]),
        "cube_yaw_error_deg_mean": float(np.mean(yaw_errs)) if yaw_errs else None,
        "cube_yaw_error_deg_max": float(np.max(yaw_errs)) if yaw_errs else None,
        "totals": totals,
    }
    (ROOT / "results" / "perception_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                                                         encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}: {v}")
    for r in [r for r in rows if r["result"] != "ok"][:10]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
