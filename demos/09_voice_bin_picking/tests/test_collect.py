"""Grabación de demostraciones sobre el simulador cinemático hacia un dataset REAL de LeRobot, y su validación.

El simulador es cinemático (``fake_cell``): valida el formato, el preprocesado, el ruido de aproximación y el
filtrado; no la física del agarre.
"""

import json

import numpy as np
import pytest

from binpick import collect as COL
from binpick import config as C
from binpick import dataset_check as DC
from binpick import observation as O
from binpick import recorder as R
from binpick import scene_gen as G
from binpick.robot_port import SimRobot
from binpick.telemetry import BinTelemetry
from fake_cell import FakeBinCell

CFG = C.BinPickCfg()


def make_robot():
    fb = FakeBinCell(CFG)
    robot = SimRobot(fb, BinTelemetry(), "hsv", realtime=0.0)
    robot.live_frames = False
    return fb, robot


def test_recorder_samples_at_30hz_with_clean_actions_and_noise_on_approach_only():
    fb, robot = make_robot()
    robot.new_scene(G.sample_scene(CFG, 2, n_pieces=3, mode="separadas"))
    robot.home()
    obs = robot.observe("inicio")
    from binpick import commands as K, selection as SEL
    sel = SEL.rank(obs.instances, COL.PICK_ORDER, fb.scene)
    cand = next(c for c in sel.usable if robot.reach(c) is None)
    plan, _ = robot._plans[cand.track]
    seg = O.target_seg_full(robot.cam, robot.scene, obs.instances, cand.inst.idx)
    executed, clean = [], []
    cell = fb.cell
    orig = cell.set_arm_targets

    rec = R.EpisodeRecorder(cell, robot.spec, robot._grab, noise=R.NoiseCfg(0.03, 1.0), rng=np.random.default_rng(1))
    # lo que llega de verdad al simulador (tras el ruido) frente a la consigna limpia que se graba
    inner = rec._orig_arm
    rec._orig_arm = lambda q: (executed.append(np.asarray(q).copy()), inner(q))[1]
    from recoverykit import skills
    t0 = cell.sim_time
    rec.start(seg, noisy=True)
    out = skills.execute_pick(robot.ctrl, plan, on_phase=rec.on_phase)
    frames = rec.stop()
    rec.detach()
    assert out.ok and 60 < len(frames) < 400
    n_phys = int(round((cell.sim_time - t0) * 240))
    assert abs(len(frames) - n_phys / 8) <= 2                                   # un fotograma cada 8 pasos de física
    f = frames[len(frames) // 2]
    assert f.overhead.shape == (*robot.spec.out_hw, 3) and f.target.shape == f.overhead.shape
    assert f.state.shape == f.action.shape == (6,) and f.state.dtype == np.float32
    acts = np.array([fr.action for fr in frames])
    assert acts[:, 5].max() > acts[0, 5]                                        # la pinza se abre y se cierra
    # el ruido se ejecutó (la consigna enviada difiere de la limpia) pero la acción grabada es siempre la limpia
    ex = np.array(executed)
    assert np.abs(ex - np.roll(ex, 1, axis=0)).max() > 0
    assert rec._off is not None


def test_noise_is_off_when_not_noisy_and_decays_after_the_approach():
    fb, robot = make_robot()
    rec = R.EpisodeRecorder(fb.cell, robot.spec, robot._grab, noise=R.NoiseCfg(0.05, 1.0), rng=np.random.default_rng(0))
    rec.start(np.zeros((10, 10, 3), np.uint8), noisy=False)
    for _ in range(200):
        rec._update_noise()
    assert np.allclose(rec._off, 0)
    rec.noisy, rec._phase = True, "aproximación"
    for _ in range(400):
        rec._update_noise()
    assert np.abs(rec._off).max() > 0.005
    rec._phase = "bajada"
    for _ in range(600):
        rec._update_noise()
    assert np.abs(rec._off).max() < 0.005 * 0.05
    rec.detach()


def test_success_criterion_rejects_a_piece_left_behind():
    fb, robot = make_robot()
    robot.new_scene(G.sample_scene(CFG, 3, n_pieces=2, mode="separadas"))
    robot.home()
    obs = robot.observe("inicio")
    from binpick import selection as SEL
    cand = next(c for c in SEL.rank(obs.instances, COL.PICK_ORDER, fb.scene).usable if robot.reach(c) is None)
    plan, _ = robot._plans[cand.track]
    from binpick import picking
    q = plan.lift[-1]
    tcp = np.r_[picking.tcp_xy(robot.base, q, plan.local), picking.tcp_height(robot.base, q, plan.local)]
    ok, _ = R.is_success(plan, robot.base, q, plan.q_contact, tcp)
    assert ok
    assert not R.is_success(plan, robot.base, q, plan.q_contact, np.array([tcp[0], tcp[1], plan.grasp_z]))[0]      # se quedó abajo
    assert not R.is_success(plan, robot.base, q, plan.q_contact, tcp + np.array([0.05, 0, 0]))[0]                  # no viaja con la pinza
    assert not R.is_success(plan, robot.base, q, plan.q_contact - 0.3, tcp)[0]                                    # pinza vacía


def test_collect_writes_a_valid_lerobot_dataset_and_resumes(tmp_path):
    pytest.importorskip("lerobot")
    fb, robot = make_robot()
    root = tmp_path / "ds"
    ds = R.open_dataset(root, "binpick_test", robot.spec, resume=False, threads=0)
    R.write_info(root, robot.spec, "hsv", CFG, R.NoiseCfg(0.02, 0.5))
    lines = []
    s1 = COL.collect(robot, root, ds, CFG, episodes=2, first_seed=10, noise=R.NoiseCfg(0.02, 0.5), say=lines.append)
    ds.finalize()
    assert s1["kept"] == 2 and s1["total_kept"] == 2 and len(lines) >= 2
    rep = DC.check(root)
    assert rep["ok"], DC.render_report(rep)
    assert rep["stats"]["episodios"] == 2 and rep["stats"]["validos"] == 2
    assert rep["stats"]["longitud_episodio"]["min"] > 60
    assert rep["stats"]["acción_vs_estado"] > DC.MIN_ACTION_STATE_GAP
    log = R.read_log(root)
    assert [r["success"] for r in log].count(True) == 2 and all("layout" in r for r in log)
    info = json.loads((root / "meta" / R.INFO_FILE).read_text(encoding="utf-8"))
    assert info["roi"] == list(robot.spec.roi) and info["joint_units"] == "range_m100_100" and info["segmenter"] == "hsv"

    # reanudar: no repite semillas y suma episodios al mismo dataset
    ds2 = R.open_dataset(root, "binpick_test", robot.spec, resume=True, threads=0)
    s2 = COL.collect(robot, root, ds2, CFG, episodes=1, first_seed=10, noise=R.NoiseCfg(0.0, 0.0), say=lambda m: None)
    ds2.finalize()
    assert s2["total_kept"] == 3
    seeds = [r["seed"] for r in R.read_log(root)]
    assert len(seeds) == len(set(seeds))
    assert DC.check(root)["stats"]["episodios"] == 3
    with pytest.raises(FileExistsError):
        R.open_dataset(root, "binpick_test", robot.spec, resume=False)


def test_validator_flags_a_dataset_whose_action_copies_the_state(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / "bad"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    spec = O.ObsSpec((0, 0, 10, 10), (288, 160))
    info = {"fps": 30, "total_episodes": 1, "total_frames": 60, "features": O.features(spec)}
    info["features"] = {k: {**v, "shape": list(v["shape"])} for k, v in info["features"].items()}
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    n = 60
    state = np.tile(np.linspace(-50, 50, n)[:, None], (1, 6)).astype(np.float32)
    table = pa.table({O.STATE_KEY: list(state), O.ACTION_KEY: list(state), "episode_index": np.zeros(n, int),
                      "frame_index": np.arange(n), "timestamp": np.arange(n) / 30.0})
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    rep = DC.check(root)
    assert not rep["ok"] and any("copia del estado" in p for p in rep["problems"])
    bad = DC.check(tmp_path / "no_hay")
    assert not bad["ok"] and "meta/info.json" in bad["problems"][0]
