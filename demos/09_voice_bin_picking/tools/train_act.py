"""Entrena ACT (LeRobot) sobre un dataset de demostraciones de este demo. Sirve en Colab (Linux) y en local. NO lo lanza nadie por ti.

    python train_act.py --dataset-root data/bin_pick_v1 --steps 50000 --batch 16 --output outputs/act_rgbmask_v1
    python train_act.py ... --obs-mode rgb_only                       # variante de control (sin saber qué pieza)
    python train_act.py ... --no-state                                # sin articulaciones como entrada (anti copia de pose)
    python train_act.py ... --episodes 0-449                          # deja 450-499 para offline_eval_policy.py
    python train_act.py ... --init-from outputs/act_v1/checkpoints/last/pretrained_model --steps 20000   # afinar con más datos

Variantes de entrada (``--obs-mode``; ver ``docs/02_arquitectura.md``): ``rgb_mask`` (recomendada: RGB + segmentación del objetivo),
``rgb_only`` (control) y ``mask_only``. Son subconjuntos de lo grabado: no hay que volver a grabar para compararlas.

Qué decide este script y por qué (cada valor es una elección, no un resultado medido en este dataset):

* ``chunk_size 50`` a 30 Hz = 1,7 s por bloque, lo que dura aproximar, bajar y cerrar; ``n_action_steps 25`` re-observa a mitad de bloque
  (lazo más cerrado que el de 100 de LeRobot por defecto). La política que funcionó en el demo 06 fue de lazo abierto y se hundía con
  cualquier desvío;
* VAE activado como en LeRobot (con 500 episodios ya no sobra: el demo 06 lo apagó con sólo 48). Prueba ``--no-vae`` como segundo experimento;
* tasa de aprendizaje 1e-5 (la de ACT, afinada para ResNet preentrenado), sin aumentos de imagen (``ImageTransforms`` también tocaría
  los colores de la máscara de segmentación).

Escribe ``<output>/`` con los checkpoints de LeRobot y, junto a cada ``pretrained_model``, ``binpick_info.json`` (modo de observación,
recorte, segmentador, unidades) que ``binpick/policy.py`` comprueba al cargar. Necesita ``lerobot==0.4.4`` (con el que se grabó).
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path

IMAGE_KEYS = {"overhead": "observation.images.overhead", "target": "observation.images.target"}
STATE_KEY, ACTION_KEY = "observation.state", "action"
OBS_MODES = {"rgb_mask": ("overhead", "target"), "rgb_only": ("overhead",), "mask_only": ("target",)}


def parse_episodes(text: str | None) -> list[int] | None:
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def resolve_image_inputs(feature_keys: set[str], obs_mode: str) -> list[str]:
    """Resolve visual inputs for custom bin-pick data and regular LeRobot camera keys."""
    available = sorted(key for key in feature_keys if key.startswith("observation.images."))
    if obs_mode == "rgb_mask":
        required = [IMAGE_KEYS["overhead"], IMAGE_KEYS["target"]]
        missing = [key for key in required if key not in feature_keys]
        if missing:
            raise SystemExit(
                f"RGB + target mask requires {required}; this dataset has {available or 'no camera features'}. "
                "Choose RGB only for ordinary camera recordings, or record masks with tools/collect_demos.py."
            )
        return required
    if obs_mode == "mask_only":
        key = IMAGE_KEYS["target"]
        if key not in feature_keys:
            raise SystemExit(f"Mask-only mode requires {key}; available camera features: {available or 'none'}.")
        return [key]
    if IMAGE_KEYS["overhead"] in feature_keys:
        return [IMAGE_KEYS["overhead"]]
    for suffix in ("front", "camera", "main"):
        key = f"observation.images.{suffix}"
        if key in feature_keys:
            return [key]
    ordinary_cameras = [key for key in available if key != IMAGE_KEYS["target"]]
    if ordinary_cameras:
        return [ordinary_cameras[0]]
    raise SystemExit("The dataset does not contain an RGB camera feature for ACT.")


def report_stage_progress(stage: str, current: int, total: int) -> None:
    print(f"[Progress] stage={stage} current={current} total={total}", flush=True)


def patch_last_checkpoint_windows() -> None:
    """Apply Windows checkpoint fixes and allow warm-starting an incomplete saved checkpoint."""
    import shutil

    import lerobot.scripts.lerobot_train as lerobot_train
    import lerobot.utils.train_utils as train_utils
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import LAST_CHECKPOINT_LINK

    def update_last_checkpoint(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        link = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
        step = checkpoint_dir.name
        staged = checkpoint_dir.parent / f"t{step}"
        backup = checkpoint_dir.parent / f"b{step}"

        try:
            shutil.copytree(checkpoint_dir, staged)
            if link.exists() or link.is_symlink():
                os.replace(link, backup)
            os.replace(staged, link)
        except (OSError, shutil.Error) as exc:
            if backup.exists() and not link.exists():
                try:
                    os.replace(backup, link)
                except OSError:
                    pass
            shutil.rmtree(staged, ignore_errors=True)
            print(
                f"[Checkpoint] Could not refresh 'last' because Windows/OneDrive could not copy or replace it: {exc}. "
                f"The numbered checkpoint at {checkpoint_dir} is saved; training will continue.",
                flush=True,
            )
            return link

        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        return link

    train_utils.update_last_checkpoint = update_last_checkpoint
    lerobot_train.update_last_checkpoint = update_last_checkpoint

    def make_pre_post_processors_with_partial_checkpoint(policy_cfg, pretrained_path=None, **kwargs):
        checkpoint = Path(pretrained_path) if pretrained_path else None
        required_configs = ("policy_preprocessor.json", "policy_postprocessor.json")
        if checkpoint and (checkpoint / "config.json").is_file():
            missing = [name for name in required_configs if not (checkpoint / name).is_file()]
            if missing:
                print(
                    "[Prepare] Checkpoint has model weights but no complete processor files; "
                    "rebuilding processors from the selected dataset statistics. Optimizer state will start fresh.",
                    flush=True,
                )
                return make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=None, **kwargs)
        return make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=pretrained_path, **kwargs)

    lerobot_train.make_pre_post_processors = make_pre_post_processors_with_partial_checkpoint


def patch_live_validation(args, cfg) -> None:
    """Log ACT loss on a fixed sample of held-out episodes while training."""
    import random

    import torch
    import lerobot.scripts.lerobot_train as lerobot_train
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from torch.utils.data import DataLoader, Subset

    validation_episodes = parse_episodes(args.validation_episodes)
    if not validation_episodes:
        raise SystemExit("[Validate] Select held-out validation episodes to calculate live validation loss.")

    repo_id = args.repo_id or Path(args.dataset_root).name
    metadata = LeRobotDatasetMetadata(repo_id, root=Path(args.dataset_root))
    validation_dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(args.dataset_root),
        episodes=validation_episodes,
        delta_timestamps=resolve_delta_timestamps(cfg.policy, metadata),
    )
    if not len(validation_dataset):
        raise SystemExit("[Validate] The selected validation episodes contain no frames.")

    sample_size = min(64, len(validation_dataset))
    sample_indices = sorted(random.Random(args.seed).sample(range(len(validation_dataset)), sample_size))
    validation_loader = DataLoader(
        Subset(validation_dataset, sample_indices),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    print(
        f"Live validation: {len(validation_episodes)} held-out episodes, {sample_size} fixed frames, "
        f"every {args.validation_freq} steps.",
        flush=True,
    )

    state = {"preprocessor": None, "step": 0, "train_sums": {}, "train_count": 0}
    original_processor_factory = lerobot_train.make_pre_post_processors
    original_update_policy = lerobot_train.update_policy

    def capture_preprocessor(*factory_args, **factory_kwargs):
        preprocessor, postprocessor = original_processor_factory(*factory_args, **factory_kwargs)
        state["preprocessor"] = preprocessor
        return preprocessor, postprocessor

    def update_policy_with_validation(
        train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
        lr_scheduler=None, lock=None, rabc_weights_provider=None,
    ):
        train_metrics, output_dict = original_update_policy(
            train_metrics,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            accelerator,
            lr_scheduler=lr_scheduler,
            lock=lock,
            rabc_weights_provider=rabc_weights_provider,
        )
        state["step"] += 1
        step = state["step"]
        progress_interval = max(1, args.steps // 100)
        if step == 1 or step % progress_interval == 0 or step == args.steps:
            report_stage_progress("train", step, args.steps)
        for key in ("l1_loss", "kld_loss"):
            if key in output_dict:
                state["train_sums"][key] = state["train_sums"].get(key, 0.0) + float(output_dict[key])
        state["train_count"] += 1

        if step != 1 and step % args.validation_freq and step != args.steps:
            return train_metrics, output_dict

        train_values = {
            key: value / state["train_count"]
            for key, value in state["train_sums"].items()
        }
        train_loss = train_values.get("l1_loss", 0.0) + args.kl_weight * train_values.get("kld_loss", 0.0)
        preprocessor = state["preprocessor"]
        if preprocessor is None:
            print(f"Live validation unavailable at step {step}: ACT preprocessor is not ready.", flush=True)
            return train_metrics, output_dict

        loss_sum = 0.0
        component_sums = {}
        sample_count = 0
        was_training = policy.training
        policy.eval()
        try:
            cuda_devices = [torch.cuda.current_device()] if args.device == "cuda" and torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(args.seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(args.seed)
                with torch.no_grad(), accelerator.autocast():
                    for validation_batch in validation_loader:
                        validation_batch = preprocessor(validation_batch)
                        loss, components = policy.forward(validation_batch)
                        count = int(validation_batch[ACTION_KEY].shape[0])
                        loss_sum += float(loss.detach().float().item()) * count
                        sample_count += count
                        for key in ("l1_loss", "kld_loss"):
                            if key in components:
                                component_sums[key] = component_sums.get(key, 0.0) + float(components[key]) * count
        except Exception as error:
            print(f"Live validation unavailable at step {step}: {error}", flush=True)
            return train_metrics, output_dict
        finally:
            policy.train(was_training)

        fields = [f"step: {step}", f"loss: {train_loss:.6f}"]
        if "l1_loss" in train_values:
            fields.append(f"train_l1_loss: {train_values['l1_loss']:.6f}")
        if "kld_loss" in train_values:
            fields.append(f"train_kld_loss: {train_values['kld_loss']:.6f}")
        fields.append(f"eval_loss: {loss_sum / sample_count:.6f}")
        if "l1_loss" in component_sums:
            fields.append(f"eval_l1_loss: {component_sums['l1_loss'] / sample_count:.6f}")
        if "kld_loss" in component_sums:
            fields.append(f"eval_kld_loss: {component_sums['kld_loss'] / sample_count:.6f}")
        fields.append(f"grad_norm: {train_metrics.grad_norm.val:.6f}")
        fields.append(f"lr: {train_metrics.lr.val:.8g}")
        elapsed = train_metrics.update_s.val + train_metrics.dataloading_s.val
        if elapsed > 0:
            fields.append(f"samples_per_s: {args.batch / elapsed:.4f}")
        if args.device == "cuda" and torch.cuda.is_available():
            fields.append(f"gpu_mem_gb: {torch.cuda.max_memory_allocated() / 1024**3:.3f}")
        print(" ".join(fields), flush=True)
        state["train_sums"] = {}
        state["train_count"] = 0
        return train_metrics, output_dict

    lerobot_train.make_pre_post_processors = capture_preprocessor
    lerobot_train.update_policy = update_policy_with_validation


def build_cfg(args):
    """``TrainPipelineConfig`` de LeRobot para estos argumentos (sin entrenar nada)."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig

    root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or root.name
    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    pf = dataset_to_policy_features(meta.features)
    # el estado articular es una entrada más; sin él (``--no-state``) la red no puede copiar la pose de su propio brazo
    # (la «confusión causal» que hundió a las políticas con cámara + estado del demo 06). LeRobot 0.4.4 lee igualmente la
    # clave del lote, que sigue estando en el dataset: sólo se saca de las entradas de la red.
    image_inputs = resolve_image_inputs(set(pf), args.obs_mode)
    args.resolved_image_inputs = image_inputs
    keep = set(image_inputs) | (set() if args.no_state else {STATE_KEY})
    missing = keep - set(pf)
    if missing:
        raise SystemExit(f"el dataset no tiene {sorted(missing)}; ¿se grabó con tools/collect_demos.py?")
    output_features = {k: v for k, v in pf.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in pf.items() if k in keep}
    optional_policy_values = {
        "vision_backbone": args.vision_backbone,
        "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
        "replace_final_stride_with_dilation": args.replace_final_stride,
        "feedforward_activation": args.feedforward_activation,
        "optimizer_weight_decay": args.weight_decay,
        "optimizer_lr_backbone": args.backbone_lr,
        "dropout": args.dropout,
        "dim_model": args.dim_model,
        "n_heads": args.attention_heads,
        "dim_feedforward": args.feedforward_dim,
        "n_encoder_layers": args.encoder_layers,
        "n_decoder_layers": args.decoder_layers,
        "latent_dim": args.latent_dim,
        "n_vae_encoder_layers": args.vae_encoder_layers,
        "pre_norm": args.pre_norm,
        "use_amp": args.use_amp,
    }
    baseline_policy_values = {
        "vision_backbone": "resnet18",
        "temporal_ensemble_coeff": None,
        "replace_final_stride_with_dilation": False,
        "feedforward_activation": "relu",
        "optimizer_weight_decay": 1e-4,
        "optimizer_lr_backbone": 1e-5,
        "dropout": 0.1,
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "pre_norm": False,
        "use_amp": False,
    }
    try:
        supported = {field.name for field in dataclasses.fields(ACTConfig)}
    except TypeError:
        supported = set(inspect.signature(ACTConfig).parameters)
    unsupported = [
        f"{name} (requested {value!r}, baseline {baseline_policy_values[name]!r})"
        for name, value in optional_policy_values.items()
        if name not in supported and value != baseline_policy_values[name]
    ]
    if unsupported:
        raise SystemExit(
            "La versión instalada de LeRobot no admite estos ajustes ACT: "
            + ", ".join(unsupported)
            + ". Instala la versión indicada por el proyecto o restaura esos valores a su base."
        )
    policy_values = {name: value for name, value in optional_policy_values.items() if name in supported}
    policy = ACTConfig(
        input_features=input_features, output_features=output_features, device=args.device,
        chunk_size=args.chunk, n_action_steps=min(args.action_steps, args.chunk), use_vae=not args.no_vae,
        kl_weight=args.kl_weight, optimizer_lr=args.lr, push_to_hub=False,
        pretrained_backbone_weights=None if args.no_pretrained_backbone else {
            "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
            "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
            "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
        }[args.vision_backbone],
        **policy_values,
    )
    if args.init_from:
        policy.pretrained_path = Path(args.init_from)
    out = Path(args.output).resolve()
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, root=str(root), episodes=parse_episodes(args.episodes)),
        policy=policy, output_dir=out, job_name=out.name, seed=args.seed, batch_size=args.batch, steps=args.steps,
        num_workers=args.workers, save_freq=args.save_freq or max(1000, args.steps // 5), log_freq=args.log_freq,
        eval_freq=args.steps + 1)       # LeRobot no tiene aquí un entorno de evaluación: se evalúa con tools/eval_sorting.py


def write_info(args, root: Path, out: Path) -> None:
    src = root / "meta" / "binpick_info.json"
    info = json.loads(src.read_text(encoding="utf-8")) if src.is_file() else {}
    info.update({"obs_mode": args.obs_mode, "image_inputs": getattr(args, "resolved_image_inputs", []),
                 "trained_on": root.name, "steps": args.steps, "batch": args.batch, "chunk": args.chunk,
                 "n_action_steps": min(args.action_steps, args.chunk), "use_vae": not args.no_vae, "use_state": not args.no_state, "lr": args.lr,
                 "act_parameters": {"weight_decay": args.weight_decay, "backbone_lr": args.backbone_lr, "dropout": args.dropout,
                                    "vision_backbone": args.vision_backbone,
                                    "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
                                    "replace_final_stride_with_dilation": args.replace_final_stride,
                                    "feedforward_activation": args.feedforward_activation,
                                    "dim_model": args.dim_model, "attention_heads": args.attention_heads,
                                    "feedforward_dim": args.feedforward_dim, "encoder_layers": args.encoder_layers,
                                    "decoder_layers": args.decoder_layers, "latent_dim": args.latent_dim,
                                    "vae_encoder_layers": args.vae_encoder_layers, "pre_norm": args.pre_norm,
                                    "use_amp": args.use_amp, "kl_weight": args.kl_weight},
                 "episodes": args.episodes or "todos",
                 "validation_episodes": parse_episodes(args.validation_episodes) or [],
                 "validation_freq": args.validation_freq,
                 "lerobot": _lerobot_version()})
    text = json.dumps(info, indent=2, ensure_ascii=False)
    (out / "binpick_info.json").write_text(text, encoding="utf-8")
    for ck in (out / "checkpoints").glob("*/pretrained_model"):
        (ck / "binpick_info.json").write_text(text, encoding="utf-8")


def _lerobot_version() -> str:
    try:
        import lerobot
        return getattr(lerobot, "__version__", "?")
    except Exception:  # noqa: BLE001
        return "?"


def configure_local_hf_cache() -> None:
    """Use a short, writable local cache path to avoid profile ACL and Windows path-length errors."""
    default_home = Path(tempfile.gettempdir()) / "rkhf"
    hf_home = Path(os.environ.setdefault("HF_HOME", str(default_home))).expanduser()
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "ds"))
    os.environ.setdefault("HF_ASSETS_CACHE", str(hf_home / "assets"))
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_ASSETS_CACHE"):
        Path(os.environ[variable]).expanduser().mkdir(parents=True, exist_ok=True)


def validate_runtime(args) -> None:
    try:
        import torch
    except ImportError as error:
        raise SystemExit(f"Falta PyTorch en este intérprete: {sys.executable}.") from error
    try:
        import lerobot
    except ImportError as error:
        raise SystemExit(f"Falta LeRobot en este intérprete: {sys.executable}; instala lerobot==0.4.4.") from error
    version = str(getattr(lerobot, "__version__", ""))
    if version != "0.4.4":
        raise SystemExit(f"Este entrenador requiere LeRobot 0.4.4; el entorno tiene {version or 'versión desconocida'}.")
    print(f"[Validate] Python: {sys.executable}", flush=True)
    print(f"[Validate] PyTorch {torch.__version__} · CUDA {torch.version.cuda or 'CPU'}", flush=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[Validate] Se solicitó CUDA pero PyTorch no detecta una GPU disponible.")
    print(f"[Validate] Dispositivo disponible: {args.device}", flush=True)
    if os.name == "nt" and args.workers > 1:
        print(
            f"[Validate] Aviso: {args.workers} workers de DataLoader en Windows pueden consumir mucha RAM; "
            "se recomienda --workers 0 si aparece MemoryError.",
            flush=True,
        )


def validate_dataset(args) -> None:
    root = Path(args.dataset_root).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"No se encuentra el metadato del dataset: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"No se puede leer {info_path}: {error}") from error
    features = info.get("features") or {}
    if not isinstance(features, dict):
        raise SystemExit("El campo 'features' de meta/info.json no tiene el formato esperado.")
    feature_keys = set(features)
    for key in (STATE_KEY, ACTION_KEY):
        if key not in feature_keys:
            raise SystemExit(f"Falta la característica necesaria {key} en el dataset.")
    image_inputs = resolve_image_inputs(feature_keys, args.obs_mode)
    args.resolved_image_inputs = image_inputs
    total = int(info.get("total_episodes", 0))
    if total < 1:
        raise SystemExit("El dataset no contiene episodios.")
    selected = parse_episodes(args.episodes)
    if selected is not None and any(index < 0 or index >= total for index in selected):
        raise SystemExit(f"Los episodios seleccionados deben estar entre 0 y {total - 1}.")
    validation = parse_episodes(args.validation_episodes)
    if args.stage in {"train", "all"} and not validation:
        raise SystemExit("[Validate] Select at least one held-out episode for live validation loss.")
    if validation is not None:
        if any(index < 0 or index >= total for index in validation):
            raise SystemExit(f"Los episodios de validación deben estar entre 0 y {total - 1}.")
        training_ids = set(selected) if selected is not None else set(range(total))
        if training_ids & set(validation):
            raise SystemExit("Los episodios de entrenamiento y validación no pueden solaparse.")
    print(
        f"[Validate] Dataset: {root.name} · {total} episodios · "
        f"train: {len(selected) if selected is not None else total} · "
        f"validation: {len(validation or [])} · entradas visuales: {image_inputs}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--obs-mode", default="rgb_mask", choices=sorted(OBS_MODES))
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--vision-backbone", choices=("resnet18", "resnet34", "resnet50"), default="resnet18")
    ap.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    ap.add_argument("--replace-final-stride", action="store_true")
    ap.add_argument("--feedforward-activation", choices=("relu", "gelu"), default="relu")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--dim-model", type=int, default=512)
    ap.add_argument("--attention-heads", type=int, default=8)
    ap.add_argument("--feedforward-dim", type=int, default=3200)
    ap.add_argument("--encoder-layers", type=int, default=4)
    ap.add_argument("--decoder-layers", type=int, default=1)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--vae-encoder-layers", type=int, default=4)
    ap.add_argument("--pre-norm", action="store_true")
    ap.add_argument("--use-amp", action="store_true")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--action-steps", type=int, default=25)
    ap.add_argument("--no-vae", action="store_true")
    ap.add_argument("--no-state", action="store_true", help="no dar las articulaciones como entrada (experimento anti copia de pose)")
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--episodes", default=None, help="p. ej. 0-449 (deja el resto sin ver para la evaluación offline)")
    ap.add_argument("--validation-episodes", default=None, help="episodios held-out para la métrica live eval_loss")
    ap.add_argument("--output", default=None, help="carpeta nueva (LeRobot no reutiliza una existente)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=None)
    ap.add_argument("--log-freq", type=int, default=200)
    ap.add_argument("--validation-freq", type=int, default=None, help="frecuencia de cálculo de métricas de validación")
    ap.add_argument("--init-from", default=None, help="carpeta pretrained_model desde la que continuar (afinado)")
    ap.add_argument("--no-pretrained-backbone", action="store_true", help="no descargar los pesos ImageNet del ResNet (sin red)")
    ap.add_argument("--stage", choices=("validate", "prepare", "train", "all"), default="all",
                    help="validate runtime/data, prepare ACT config, train, or run all stages")
    args = ap.parse_args()
    args.validation_freq = args.validation_freq or args.log_freq
    if args.validation_freq < 1:
        ap.error("--validation-freq must be positive")
    configure_local_hf_cache()

    if args.dim_model % args.attention_heads:
        ap.error("--dim-model must be divisible by --attention-heads")
    if not 0.0 <= args.dropout < 1.0:
        ap.error("--dropout must be in the range [0, 1)")
    if args.weight_decay < 0 or args.backbone_lr <= 0 or args.lr <= 0:
        ap.error("learning rates must be positive and weight decay cannot be negative")
    if args.temporal_ensemble_coeff is not None:
        if args.temporal_ensemble_coeff <= 0 or args.action_steps != 1:
            ap.error("temporal ensembling requires a positive coefficient and --action-steps 1")

    args.output = args.output or f"outputs/act_{args.obs_mode}_{time.strftime('%Y%m%d_%H%M')}"
    if args.stage in {"train", "all"} and Path(args.output).exists():
        print(f"ERROR: {args.output} ya existe (LeRobot no reutiliza carpetas); usa otro --output")
        return 1

    if args.stage == "all":
        print("[Validate] Comprobando entorno y dataset...", flush=True)
    report_stage_progress("validate", 0, 2)
    validate_runtime(args)
    report_stage_progress("validate", 1, 2)
    validate_dataset(args)
    report_stage_progress("validate", 2, 2)
    if args.stage == "validate":
        print("[Validate] Validación completada correctamente.", flush=True)
        return 0

    print("[Prepare] Construyendo la configuración ACT...", flush=True)
    report_stage_progress("prepare", 0, 2)
    cfg = build_cfg(args)
    report_stage_progress("prepare", 1, 2)
    print(f"[Prepare] Dataset: {cfg.dataset.root} · entradas: {sorted(cfg.policy.input_features)} · salida: {args.output}", flush=True)
    report_stage_progress("prepare", 2, 2)
    if args.stage == "prepare":
        print("[Prepare] Preparación completada correctamente; no se inició el entrenamiento.", flush=True)
        return 0

    print(f"[Train] Bloque {args.chunk}/{args.action_steps} · VAE {'no' if args.no_vae else 'sí'} · "
          f"{args.steps} pasos × lote {args.batch} · {args.device}", flush=True)
    if sys.platform == "win32":
        patch_last_checkpoint_windows()
    patch_live_validation(args, cfg)
    from lerobot.scripts.lerobot_train import train

    t0 = time.time()
    report_stage_progress("train", 0, args.steps)
    train(cfg)
    write_info(args, Path(args.dataset_root).resolve(), Path(args.output).resolve())
    print(f"entrenado en {(time.time() - t0) / 60:.1f} min · modelo: {Path(args.output) / 'checkpoints' / 'last' / 'pretrained_model'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
