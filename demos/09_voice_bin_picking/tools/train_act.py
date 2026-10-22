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
import json
import sys
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


def patch_last_checkpoint_windows() -> None:
    """LeRobot enlaza ``checkpoints/last`` con un symlink; en Windows sin permisos falla. Se sustituye por una copia."""
    import shutil

    import lerobot.scripts.lerobot_train as lerobot_train
    import lerobot.utils.train_utils as train_utils
    from lerobot.utils.constants import LAST_CHECKPOINT_LINK

    def update_last_checkpoint(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        link = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        shutil.copytree(checkpoint_dir, link)
        return link

    train_utils.update_last_checkpoint = update_last_checkpoint
    lerobot_train.update_last_checkpoint = update_last_checkpoint


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
    keep = {*(IMAGE_KEYS[s] for s in OBS_MODES[args.obs_mode])} | (set() if args.no_state else {STATE_KEY})
    missing = keep - set(pf)
    if missing:
        raise SystemExit(f"el dataset no tiene {sorted(missing)}; ¿se grabó con tools/collect_demos.py?")
    output_features = {k: v for k, v in pf.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in pf.items() if k in keep}
    policy = ACTConfig(
        input_features=input_features, output_features=output_features, device=args.device,
        chunk_size=args.chunk, n_action_steps=min(args.action_steps, args.chunk), use_vae=not args.no_vae,
        kl_weight=args.kl_weight, optimizer_lr=args.lr, push_to_hub=False,
        pretrained_backbone_weights=None if args.no_pretrained_backbone else "ResNet18_Weights.IMAGENET1K_V1")
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
    info.update({"obs_mode": args.obs_mode, "trained_on": root.name, "steps": args.steps, "batch": args.batch, "chunk": args.chunk,
                 "n_action_steps": min(args.action_steps, args.chunk), "use_vae": not args.no_vae, "use_state": not args.no_state, "lr": args.lr,
                 "episodes": args.episodes or "todos", "lerobot": _lerobot_version()})
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--obs-mode", default="rgb_mask", choices=sorted(OBS_MODES))
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--action-steps", type=int, default=25)
    ap.add_argument("--no-vae", action="store_true")
    ap.add_argument("--no-state", action="store_true", help="no dar las articulaciones como entrada (experimento anti copia de pose)")
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--episodes", default=None, help="p. ej. 0-449 (deja el resto sin ver para la evaluación offline)")
    ap.add_argument("--output", default=None, help="carpeta nueva (LeRobot no reutiliza una existente)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-freq", type=int, default=None)
    ap.add_argument("--log-freq", type=int, default=200)
    ap.add_argument("--init-from", default=None, help="carpeta pretrained_model desde la que continuar (afinado)")
    ap.add_argument("--no-pretrained-backbone", action="store_true", help="no descargar los pesos ImageNet del ResNet (sin red)")
    args = ap.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA no está disponible: se entrena en CPU (muy lento).")
        args.device = "cpu"
    args.output = args.output or f"outputs/act_{args.obs_mode}_{time.strftime('%Y%m%d_%H%M')}"
    if Path(args.output).exists():
        print(f"ERROR: {args.output} ya existe (LeRobot no reutiliza carpetas); usa otro --output")
        return 1
    cfg = build_cfg(args)
    print(f"dataset {cfg.dataset.root} · entradas: {sorted(cfg.policy.input_features)} · bloque {args.chunk}/{args.action_steps} · "
          f"VAE {'no' if args.no_vae else 'sí'} · {args.steps} pasos × lote {args.batch} · {args.device}", flush=True)
    if sys.platform == "win32":
        patch_last_checkpoint_windows()
    from lerobot.scripts.lerobot_train import train

    t0 = time.time()
    train(cfg)
    write_info(args, Path(args.dataset_root).resolve(), Path(args.output).resolve())
    print(f"entrenado en {(time.time() - t0) / 60:.1f} min · modelo: {Path(args.output) / 'checkpoints' / 'last' / 'pretrained_model'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
