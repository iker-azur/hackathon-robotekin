"""¿Acepta ACT de LeRobot exactamente este dataset, y reproduce la inferencia el mismo preprocesado? (sin entrenar)

Se graba un dataset real de 2 episodios en el simulador cinemático, se instancia un ACT diminuto sin pesos
preentrenados (no hay descarga ni entrenamiento), se hace UNA pasada hacia delante con un lote del dataset, y se
guarda y vuelve a cargar con ``ACTPickPolicy`` para comprobar la ruta de inferencia. No dice nada de si una política
entrenada de verdad funciona: eso se mide con ``tools/eval_sorting.py``.
"""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot")
torch = pytest.importorskip("torch")

from binpick import collect as COL
from binpick import config as C
from binpick import observation as O
from binpick import policy as POL
from binpick import recorder as R
from binpick.robot_port import SimRobot
from binpick.telemetry import BinTelemetry
from fake_cell import FakeBinCell

CFG = C.BinPickCfg()
CHUNK = 8


@pytest.fixture()
def recorded(recorded_dataset):
    return recorded_dataset


def tiny_act(features):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig

    pf = dataset_to_policy_features(features)
    out = {k: v for k, v in pf.items() if v.type is FeatureType.ACTION}
    inp = {k: v for k, v in pf.items() if k not in out}
    return ACTConfig(input_features=inp, output_features=out, chunk_size=CHUNK, n_action_steps=CHUNK, dim_model=64,
                     n_heads=4, dim_feedforward=128, n_encoder_layers=1, n_decoder_layers=1, use_vae=False,
                     pretrained_backbone_weights=None, device="cpu", push_to_hub=False)


def test_act_accepts_the_recorded_dataset_forward_pass(recorded):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root, robot = recorded
    ds = LeRobotDataset("binpick_shared", root=str(root), delta_timestamps={"action": [i / C.FPS for i in range(CHUNK)]})
    assert ds.num_episodes == 2 and ds.fps == C.FPS
    cfg = tiny_act(ds.meta.features)
    assert set(cfg.input_features) == {O.IMAGE_KEYS["overhead"], O.IMAGE_KEYS["target"], O.STATE_KEY}
    pre, _post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    item = ds[5]
    h, w = robot.spec.out_hw
    assert item[O.IMAGE_KEYS["overhead"]].shape == (3, h, w) and item[O.IMAGE_KEYS["target"]].shape == (3, h, w)
    assert item[O.ACTION_KEY].shape == (CHUNK, 6) and item[O.STATE_KEY].shape == (6,)
    batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in item.items()}
    loss, _ = ACTPolicy(cfg).forward(pre(batch))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("mode", ["rgb_mask", "rgb_only", "mask_only"])
def test_every_observation_mode_builds_a_valid_policy(recorded, mode):
    """Los tres modos del experimento son subconjuntos de las características grabadas: no hace falta volver a grabar."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy

    root, _ = recorded
    meta = LeRobotDatasetMetadata("binpick_shared", root=str(root))
    cfg = tiny_act(meta.features)
    keep = {O.STATE_KEY, *(O.IMAGE_KEYS[s] for s in O.OBS_MODES[mode])}
    cfg.input_features = {k: v for k, v in cfg.input_features.items() if k in keep}
    ACTPolicy(cfg)                                          # no lanza


def test_inference_path_roundtrip_matches_the_documented_preprocessing(recorded, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root, robot = recorded
    ds = LeRobotDataset("binpick_shared", root=str(root), delta_timestamps={"action": [i / C.FPS for i in range(CHUNK)]})
    cfg = tiny_act(ds.meta.features)
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    policy = ACTPolicy(cfg)
    out = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    (tmp_path / "run" / POL.INFO_FILE).write_text(json.dumps(
        {"obs_mode": "rgb_mask", "roi": list(robot.spec.roi), "out_hw": list(robot.spec.out_hw), "fps": C.FPS,
         "segmenter": "hsv", "joint_units": "range_m100_100"}), encoding="utf-8")

    loaded = POL.ACTPickPolicy(out, device="cpu")
    assert loaded.kind == "act" and loaded.obs_mode == "rgb_mask"
    errors, warnings = POL.check_compat(loaded.info, robot.spec, "hsv")
    assert errors == [] and warnings == []
    img = np.random.default_rng(0).integers(0, 255, (*robot.spec.out_hw, 3), dtype=np.uint8)
    obs = O.PolicyObs(img, img, np.zeros(6, np.float32))
    loaded.reset()
    a = loaded.act(obs)
    assert a.shape == (6,) and np.isfinite(a).all()
    # las órdenes están en radianes del simulador: dentro de los límites de las articulaciones (tras desnormalizar)
    assert np.abs(a[:5]).max() < 10
    a2 = loaded.act(obs)                                    # el bloque de acciones se consume sin reentrar a la red
    assert a2.shape == (6,)
    # un recorte distinto se rechaza antes de ejecutar nada
    bad = O.ObsSpec((0, 0, 50, 50), robot.spec.out_hw)
    assert POL.check_compat(loaded.info, bad, "hsv")[0]


def test_policy_refuses_mode_that_does_not_match_the_checkpoint(recorded, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root, _ = recorded
    ds = LeRobotDataset("binpick_shared", root=str(root))
    cfg = tiny_act(ds.meta.features)
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    out = tmp_path / "m"
    ACTPolicy(cfg).save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    with pytest.raises(ValueError):
        POL.ACTPickPolicy(out, obs_mode="rgb_only")        # el modelo espera dos imágenes


def test_policy_without_the_joint_state_as_input_still_trains_forward_and_infers(recorded, tmp_path):
    """La variante anti copia de pose: sin ``observation.state`` entre las entradas (la clave sigue en el lote)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root, robot = recorded
    ds = LeRobotDataset("binpick_shared", root=str(root), delta_timestamps={"action": [i / C.FPS for i in range(CHUNK)]})
    cfg = tiny_act(ds.meta.features)
    cfg.input_features = {k: v for k, v in cfg.input_features.items() if k != O.STATE_KEY}
    pre, post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    policy = ACTPolicy(cfg)
    item = ds[3]
    batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in item.items()}
    loss, _ = policy.forward(pre(batch))
    assert torch.isfinite(loss)
    out = tmp_path / "ns"
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    loaded = POL.ACTPickPolicy(out, device="cpu")
    img = np.zeros((*robot.spec.out_hw, 3), np.uint8)
    a = loaded.act(O.PolicyObs(img, img, np.zeros(6, np.float32)))
    assert a.shape == (6,) and np.isfinite(a).all()
