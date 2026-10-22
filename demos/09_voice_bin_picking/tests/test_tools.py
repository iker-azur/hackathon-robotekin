"""Las herramientas de línea de órdenes: compilan, y lo que se puede probar sin Isaac ni entrenar, se prueba."""

import importlib.util
import json
import py_compile
import sys
from pathlib import Path

import numpy as np
import pytest

from binpick import commands as K
from binpick import config as C

ROOT = Path(__file__).resolve().parents[1]
TOOLS = sorted((ROOT / "tools").glob("*.py"))
SCENE = C.build_scene()


def load(name):
    spec = importlib.util.spec_from_file_location(f"tool_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path", TOOLS + [ROOT / "18_voice_bin_picking.py"], ids=lambda p: p.name)
def test_every_script_compiles(path, tmp_path):
    py_compile.compile(str(path), cfile=str(tmp_path / "x.pyc"), doraise=True)


# ---------------------------------------------------------- juicio con la verdad
def test_truth_check_flags_wrong_box_and_pending_pieces():
    mod = load("eval_sorting")
    order = K.parse("las rojas a la izquierda y las verdes a la derecha")
    ids = {"r1": "rojo_c1", "r2": "rojo_c2", "g1": "verde_c1", "b1": "azul_c1"}
    where_ok = {"rojo_c1": "izquierda", "rojo_c2": "izquierda", "verde_c1": "derecha", "azul_c1": "origen"}
    assert mod.truth_check(order, where_ok, SCENE)["ok"]
    wrong = dict(where_ok, rojo_c2="derecha")                              # una roja en la caja equivocada
    r = mod.truth_check(order, wrong, SCENE)
    assert r["wrong"] == 1 and not r["ok"]
    pending = dict(where_ok, rojo_c2="origen")                             # una roja sin mover
    r = mod.truth_check(order, pending, SCENE)
    assert r["pending"] == 1 and r["wrong"] == 0 and not r["ok"]
    blue_in_box = dict(where_ok, azul_c1="izquierda")                      # una azul que nadie pidió mover
    assert mod.truth_check(order, blue_in_box, SCENE)["wrong"] == 1


def test_truth_check_with_quantity_and_rest():
    mod = load("eval_sorting")
    order = K.parse("pon dos piezas rojas en la caja izquierda")
    where = {"rojo_c1": "izquierda", "rojo_c2": "origen", "rojo_c3": "origen", "azul_c1": "origen"}
    assert mod.truth_check(order, where, SCENE)["pending"] == 1             # piden dos, hay una dentro
    where["rojo_c2"] = "izquierda"
    assert mod.truth_check(order, where, SCENE)["ok"]
    rest = K.parse("las rojas a la izquierda y las demás a la derecha")
    all_in = {"rojo_c1": "izquierda", "azul_c1": "derecha", "verde_c1": "derecha"}
    assert mod.truth_check(rest, all_in, SCENE)["ok"]
    assert mod.truth_check(rest, dict(all_in, azul_c1="izquierda"), SCENE)["wrong"] == 1


def test_default_orders_are_the_four_from_the_brief_and_all_parse():
    mod = load("eval_sorting")
    assert len(mod.ORDERS) == 4
    for text in mod.ORDERS:
        assert K.parse(text, default_dest="derecha").ok, text      # la 4.ª no dice caja: la herramienta de medida la fija


def test_adapt_order_swaps_missing_colors_for_present_ones_and_still_parses():
    mod = load("eval_sorting")
    # la escena no tiene rojo: «rojas» pasa a un color que hay, «verdes» se queda
    out = mod.adapt_order("Clasifica las piezas rojas a la izquierda y las verdes a la derecha.", {"green": 4, "blue": 2})
    assert "rojas" not in out and "verdes" in out and "azules" in out
    assert K.parse(out).ok
    # si los colores que pide ya están, la orden no cambia
    same = "Pon las piezas azules en la caja derecha."
    assert mod.adapt_order(same, {"blue": 1, "red": 3}) == same
    # un solo color pedido y ausente
    assert "verdes" in mod.adapt_order("Recoge todas las piezas azules.", {"green": 3, "red": 1})


# --------------------------------------------------------------- entrenamiento
def args_for(root, **kw):
    import argparse

    base = dict(dataset_root=str(root), repo_id="binpick_shared", obs_mode="rgb_mask", steps=100, batch=2, lr=1e-5, chunk=10,
                action_steps=5, no_vae=False, no_state=False, kl_weight=10.0, episodes=None, output="/tmp/nunca_se_crea", device="cpu", seed=1,
                workers=0, save_freq=None, log_freq=10, init_from=None, no_pretrained_backbone=True)
    base.update(kw)
    return argparse.Namespace(**base)


def test_train_config_selects_inputs_per_mode_without_training(recorded_dataset, tmp_path):
    pytest.importorskip("lerobot")
    mod = load("train_act")
    root, _ = recorded_dataset
    expect = {"rgb_mask": 3, "rgb_only": 2, "mask_only": 2}
    for mode, n in expect.items():
        cfg = mod.build_cfg(args_for(root, obs_mode=mode, output=str(tmp_path / mode)))
        assert len(cfg.policy.input_features) == n and "observation.state" in cfg.policy.input_features
        assert set(cfg.policy.output_features) == {"action"}
        assert cfg.policy.chunk_size == 10 and cfg.policy.n_action_steps == 5 and cfg.dataset.root == str(root.resolve())
        cfg.validate()                                          # la validación que hace train() antes de empezar
        assert cfg.policy.n_obs_steps == 1 and cfg.batch_size == 2
    nostate = mod.build_cfg(args_for(root, no_state=True, output=str(tmp_path / "ns")))
    assert "observation.state" not in nostate.policy.input_features and len(nostate.policy.input_features) == 2
    cfg = mod.build_cfg(args_for(root, episodes="0", no_vae=True, output=str(tmp_path / "x")))
    assert cfg.dataset.episodes == [0] and cfg.policy.use_vae is False
    assert mod.parse_episodes("0-3,7") == [0, 1, 2, 3, 7] and mod.parse_episodes(None) is None


def test_train_info_is_written_next_to_every_checkpoint(recorded_dataset, tmp_path):
    mod = load("train_act")
    root, _ = recorded_dataset
    out = tmp_path / "run"
    (out / "checkpoints" / "000100" / "pretrained_model").mkdir(parents=True)
    (out / "checkpoints" / "last" / "pretrained_model").mkdir(parents=True)
    mod.write_info(args_for(root, obs_mode="rgb_only", steps=100), root, out)
    for p in (out / "binpick_info.json", out / "checkpoints" / "last" / "pretrained_model" / "binpick_info.json"):
        info = json.loads(p.read_text(encoding="utf-8"))
        assert info["obs_mode"] == "rgb_only" and info["roi"] and info["joint_units"] == "range_m100_100"
    from binpick import policy as POL
    assert POL.load_info(out / "checkpoints" / "last" / "pretrained_model")["obs_mode"] == "rgb_only"


def test_offline_eval_runs_on_a_saved_policy(recorded_dataset, tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from test_lerobot_compat import CHUNK, tiny_act

    root, _ = recorded_dataset
    ds = LeRobotDataset("binpick_shared", root=str(root), delta_timestamps={"action": [i / 30 for i in range(CHUNK)]})
    cfg = tiny_act(ds.meta.features)
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    out = tmp_path / "pm"
    ACTPolicy(cfg).save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    mod = load("offline_eval_policy")
    res = tmp_path / "rep.json"
    monkeypatch.setattr(sys, "argv", ["x", "--policy", str(out), "--dataset-root", str(root), "--repo-id", "binpick_shared",
                                      "--episodes", "0-1", "--device", "cpu", "--stride", "20", "--out", str(res)])
    assert mod.main() == 0
    rep = json.loads(res.read_text(encoding="utf-8"))
    assert rep["fotogramas_evaluados"] > 3 and rep["bloque"] == CHUNK
    assert "paso 0" in rep["error_medio_por_articulación_y_paso"]
    assert np.isfinite(rep["resumen"]["error_política"])                  # una red sin entrenar: sólo se comprueba que corre
