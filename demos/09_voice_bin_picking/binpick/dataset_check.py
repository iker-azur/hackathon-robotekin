"""Comprobaciones de un dataset de demostraciones, sin simulador y sin decodificar vídeo (sólo parquet y metadatos).

Se ejecuta antes de gastar GPU en entrenar (``tools/validate_dataset.py``, y la primera celda de la libreta de
Colab): que las claves y formas son las que espera ``policy.py``, que no hay NaN, que las unidades son las de
LeRobot, que las acciones se mueven (un dataset de acciones casi constantes entrena una política quieta) y que las
marcas de tiempo son las de 30 Hz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import joints
from . import observation as O

MIN_MOTION_STD = 0.5            # unidades de LeRobot: una articulación que no se mueve en todo el dataset es sospechosa
MIN_ACTION_STATE_GAP = 0.02     # |acción − estado| medio: si es ~0 la acción es una copia del estado


def _read_frames(root: Path):
    import pyarrow.parquet as pq

    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        return None
    tables = [pq.read_table(f).to_pandas() for f in files]
    import pandas as pd

    return pd.concat(tables, ignore_index=True)


def check(root: str | Path) -> dict:
    root = Path(root)
    problems: list[str] = []
    warnings: list[str] = []
    stats: dict = {}
    info_f = root / "meta" / "info.json"
    if not info_f.is_file():
        return {"ok": False, "problems": [f"{root} no es un dataset de LeRobot (falta meta/info.json)"], "warnings": [], "stats": {}}
    info = json.loads(info_f.read_text(encoding="utf-8"))
    feats = info.get("features", {})
    for key in (O.IMAGE_KEYS["overhead"], O.IMAGE_KEYS["target"], O.STATE_KEY, O.ACTION_KEY):
        if key not in feats:
            problems.append(f"falta la característica «{key}»")
    for key in (O.STATE_KEY, O.ACTION_KEY):
        if key in feats and list(feats[key]["shape"]) != [6]:
            problems.append(f"«{key}» debería tener forma [6] y tiene {feats[key]['shape']}")
    if info.get("fps") != C.FPS:
        problems.append(f"fps {info.get('fps')} ≠ {C.FPS}")
    stats["episodios"] = info.get("total_episodes")
    stats["fotogramas"] = info.get("total_frames")

    meta = root / "meta" / "binpick_info.json"
    if meta.is_file():
        bi = json.loads(meta.read_text(encoding="utf-8"))
        stats["segmentador"] = bi.get("segmenter")
        stats["modo_de_observacion"] = bi.get("obs_mode")
        stats["recorte"] = bi.get("roi")
    else:
        warnings.append("falta meta/binpick_info.json: no se podrá comprobar el preprocesado al ejecutar la política")

    for stream in ("overhead", "target"):
        vids = list((root / "videos").glob(f"{O.IMAGE_KEYS[stream]}/*/*.mp4")) if (root / "videos").exists() else []
        if not vids:
            warnings.append(f"no hay vídeos de «{O.IMAGE_KEYS[stream]}»")

    df = _read_frames(root)
    if df is None:
        problems.append("no hay ficheros de datos (data/*/*.parquet)")
        return {"ok": False, "problems": problems, "warnings": warnings, "stats": stats}
    state = np.stack(df[O.STATE_KEY].to_numpy()).astype(np.float64)
    action = np.stack(df[O.ACTION_KEY].to_numpy()).astype(np.float64)
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        problems.append("hay NaN o infinitos en el estado o la acción")
    lo = np.array([joints.MOTOR_LIMITS[n][0] for n in joints.JOINT_NAMES]) - 1.0
    hi = np.array([joints.MOTOR_LIMITS[n][1] for n in joints.JOINT_NAMES]) + 1.0
    for name, arr in (("estado", state), ("acción", action)):
        bad = ((arr < lo) | (arr > hi)).any(axis=1).sum()
        if bad:
            problems.append(f"{bad} fotogramas con {name} fuera de los límites de LeRobot (¿unidades en radianes?)")
    stats["estado_min"] = state.min(0).round(1).tolist()
    stats["estado_max"] = state.max(0).round(1).tolist()
    std = action.std(0)
    stats["accion_std"] = std.round(2).tolist()
    moving = (std > MIN_MOTION_STD)
    if not moving[:4].all():
        warnings.append("alguna de las 4 articulaciones principales apenas se mueve en todo el dataset")
    gap = float(np.abs(action - state).mean())
    stats["acción_vs_estado"] = round(gap, 3)
    if gap < MIN_ACTION_STATE_GAP:
        problems.append("la acción es una copia del estado: la política aprendería a quedarse quieta")

    ep = df["episode_index"].to_numpy()
    lengths = np.bincount(ep)[np.unique(ep)]
    stats["longitud_episodio"] = {"min": int(lengths.min()), "mediana": float(np.median(lengths)), "max": int(lengths.max())}
    if len(np.unique(ep)) != int(ep.max()) + 1:
        problems.append("los índices de episodio no son contiguos")
    if lengths.min() < 30:
        warnings.append(f"hay episodios de sólo {int(lengths.min())} fotogramas (¿se cortó la grabación?)")
    ts = df["timestamp"].to_numpy()
    fi = df["frame_index"].to_numpy()
    first = np.concatenate([[True], np.diff(ep) != 0])
    if not (fi[first] == 0).all():
        problems.append("algún episodio no empieza en el fotograma 0")
    dts = np.diff(ts)[~first[1:]]
    if len(dts) and not np.allclose(dts, 1.0 / C.FPS, atol=2e-3):
        problems.append("las marcas de tiempo no avanzan a 1/30 s")
    log = root / "meta" / "binpick_episodes.jsonl"
    if log.is_file():
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        okn = sum(1 for r in rows if r.get("success"))
        stats["intentos"] = len(rows)
        stats["validos"] = okn
        stats["tasa_de_éxito_del_experto"] = round(okn / max(len(rows), 1), 3)
        by_reason: dict[str, int] = {}
        for r in rows:
            if not r.get("success"):
                by_reason[str(r.get("reason"))[:60]] = by_reason.get(str(r.get("reason"))[:60], 0) + 1
        stats["descartes"] = by_reason
        stats["con_ruido"] = sum(1 for r in rows if r.get("success") and r.get("noisy"))
        layers: dict[int, int] = {}
        for r in rows:
            if r.get("success"):
                layers[r.get("layer", 0)] = layers.get(r.get("layer", 0), 0) + 1
        stats["por_capa"] = layers
    return {"ok": not problems, "problems": problems, "warnings": warnings, "stats": stats}


def render_report(rep: dict) -> str:
    lines = ["DATASET " + ("VÁLIDO" if rep["ok"] else "CON PROBLEMAS")]
    lines += [f"  ✘ {p}" for p in rep["problems"]] + [f"  ! {w}" for w in rep["warnings"]]
    lines += [f"  · {k}: {v}" for k, v in rep["stats"].items()]
    return "\n".join(lines)
