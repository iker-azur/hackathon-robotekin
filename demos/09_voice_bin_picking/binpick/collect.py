"""Bucle de recogida de demostraciones: escena aleatoria → objetivo → experto → criterio físico → dataset.

Cada episodio es una escena nueva (semilla propia: piezas, colores, montones) y un objetivo elegido **al azar**
entre las piezas agarrables y alcanzables (no «la mejor»): la política tiene que ver también piezas poco
cómodas. Las escenas se asientan con física (``cell.BinCell``) y el experto actúa con la física real; sólo se
guarda el episodio si la pieza se levantó y viajó con la pinza (``recorder.is_success``). Los demás se registran
con su causa y no entran en el dataset.

Reanudable: el registro ``meta/binpick_episodes.jsonl`` guarda cada intento con su semilla; ``--resume`` salta
las semillas hechas y añade episodios al mismo dataset. Pensado para sesiones de ~40 min que se reparten por
bloques de semillas (``--first-seed``) y se pueden juntar sin que se pisen.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import commands as K
from . import config as C
from . import observation as O
from . import recorder as R
from . import scene_gen as G
from . import selection as SEL
from recoverykit import skills  # noqa: E402

PICK_ORDER = K.parse("las piezas a la izquierda")       # cualquier pieza: el destino da igual, sólo se graba el agarre


def nearest_truth(robot, inst) -> tuple[str, np.ndarray]:
    """Pieza del simulador más cercana a lo que se percibió (sólo para filtrar demostraciones, nunca para actuar)."""
    best, best_d = None, 1e9
    for pid, (p, _q) in robot.bcell.truth_poses().items():
        d = float(np.linalg.norm(p[:2] - inst.pos[:2]))
        if d < best_d:
            best, best_d = pid, d
    return best, robot.bcell.truth_poses()[best][0]


def run_episode(robot, rec: R.EpisodeRecorder, rng: np.random.Generator, seed: int, cfg: C.BinPickCfg,
                noisy: bool) -> tuple[list[R.Frame] | None, dict]:
    """Un intento. Devuelve ``(fotogramas | None, registro)``."""
    spec = G.sample_scene(cfg, seed)
    rec_log: dict = {"seed": seed, "mode": spec.mode, "n": len(spec.drops), "colors": spec.colors(), "noisy": noisy}
    res = robot.new_scene(spec)
    rec_log["layout"] = res["layout"]
    if res["problems"]:
        rec_log.update(success=False, reason="escena no válida: " + "; ".join(res["problems"][:2]))
        return None, rec_log
    robot.home()
    obs = robot.observe("inicio")
    sel = SEL.rank(obs.instances, PICK_ORDER, robot.scene)
    usable = list(sel.usable)
    rng.shuffle(usable)
    cand = None
    for c in usable:
        if robot.reach(c) is None:
            cand = c
            break
    if cand is None:
        rec_log.update(success=False, reason="ninguna pieza agarrable y alcanzable en esta escena")
        return None, rec_log
    plan, _slot = robot._plans[cand.track]
    pid, _ = nearest_truth(robot, cand.inst)
    rec_log.update(track=cand.track, piece=pid, color=cand.inst.color, shape=cand.inst.shape,
                   layer=int(round((cand.inst.pos[2] - (robot.top + robot.bin.floor_m + C.PIECE_SIZE_M / 2)) / C.PIECE_SIZE_M)),
                   clearance_mm=round(cand.grasp.clearance_m * 1000, 1), visibility=round(cand.inst.visibility, 2))
    seg = O.target_seg_full(robot.cam, robot.scene, obs.instances, cand.inst.idx)
    rec.start(seg, noisy)
    t0 = robot.cell.sim_time
    out = skills.execute_pick(robot.ctrl, plan, on_phase=rec.on_phase)
    frames = rec.stop()
    rec_log["sim_s"] = round(robot.cell.sim_time - t0, 2)
    rec_log["frames"] = len(frames)
    if out.status == "abortado" and out.detail.startswith("detenido"):
        rec_log.update(success=False, reason="detenido por el usuario", stopped=True)
        return None, rec_log
    pro = robot.cell.proprio()
    piece_pos = robot.bcell.cell.truth(pid)["pos"]
    ok, why = R.is_success(plan, robot.base, pro["arm_q"], pro["gripper_q"], piece_pos)
    if not out.ok:
        ok, why = False, f"el experto informa «{out.status}»: {out.detail}"
    rec_log.update(success=ok, reason=why)
    return (frames if ok else None), rec_log


def collect(robot, root: Path, dataset, cfg: C.BinPickCfg, episodes: int, first_seed: int, noise: R.NoiseCfg,
            rng_seed: int = 0, max_attempts: int | None = None, say: Callable[[str], None] = print,
            stop: Callable[[], bool] = lambda: False) -> dict:
    """Recoge ``episodes`` demostraciones válidas más. Devuelve un resumen."""
    rng = np.random.default_rng(rng_seed + first_seed)
    done = {r["seed"] for r in R.read_log(root)}
    prev_ok = sum(1 for r in R.read_log(root) if r.get("success"))
    max_attempts = max_attempts or max(episodes * 4, episodes + 10)
    rec = R.EpisodeRecorder(robot.cell, robot.spec, robot._grab, noise=noise, rng=rng)
    kept = attempts = 0
    t_start = time.perf_counter()
    seed = first_seed
    try:
        while kept < episodes and attempts < max_attempts and not stop():
            if seed in done:
                seed += 1
                continue
            attempts += 1
            noisy = bool(rng.random() < noise.fraction)
            frames, log = run_episode(robot, rec, rng, seed, cfg, noisy)
            log["episode"] = prev_ok + kept if frames else None
            if frames:
                R.add_episode(dataset, frames)
                kept += 1
            R.append_log(root, log)
            wall = time.perf_counter() - t_start
            eta = wall / max(attempts, 1) * (episodes - kept) * (attempts / max(kept, 1))
            say(f"[{prev_ok + kept:4d}] semilla {seed} {'✔' if frames else '✘'} {log.get('mode', '')} n={log.get('n', '?')} "
                f"{log.get('color', '')}/{log.get('shape', '')} capa {log.get('layer', '-')} "
                f"{'ruido' if noisy else 'limpio'} · {log.get('frames', 0)} fot. · "
                f"{'' if frames else log.get('reason', '')} · {wall / 60:.1f} min, quedan ~{eta / 60:.0f} min")
            if log.get("stopped"):
                break
            seed += 1
    finally:
        rec.detach()
    return {"kept": kept, "attempts": attempts, "total_kept": prev_ok + kept, "wall_min": round((time.perf_counter() - t_start) / 60, 1)}
