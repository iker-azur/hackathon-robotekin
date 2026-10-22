"""Comprobación SIN simulador de una política ACT entrenada: ¿predice las acciones del experto en episodios que no vio?

    python offline_eval_policy.py --policy outputs/act_v1/checkpoints/last/pretrained_model --dataset-root data/bin_pick_v1 --episodes 450-499

Para cada fotograma de los episodios indicados (que NO deben haber entrado en el entrenamiento: ``train_act.py --episodes 0-449``)
predice un bloque de acciones y lo compara con las acciones grabadas del experto, en unidades de LeRobot (−100…100; la pinza 0…100).
Se da el error absoluto medio por articulación y por paso del bloque, y se compara con dos líneas base triviales:

* «copiar el estado» (la acción = la posición actual: lo que haría una política que se queda quieta);
* «mantener la última acción» (la acción de hace un paso).

**Qué dice y qué no.** Que el error sea mucho menor que el de «copiar el estado» a horizontes largos es una buena señal de que la red
aprendió algo. NO dice que agarre: un error pequeño en la trayectoria no garantiza un agarre por fricción de 25 mm, y una política con
error pequeño puede fallar en lazo cerrado por desviaciones acumuladas (demo 06). El juez es ``tools/eval_sorting.py`` en Isaac.

No importa ``binpick``: sólo ``lerobot`` y numpy, para poder usarlo tal cual en Colab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STATE_KEY, ACTION_KEY = "observation.state", "action"
HORIZONS = (0, 5, 10, 25, 49)


def parse_episodes(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="carpeta pretrained_model")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--episodes", required=True, help="p. ej. 450-499")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=5, help="un fotograma de cada N")
    ap.add_argument("--out", default=None, help="fichero .json de salida")
    args = ap.parse_args()

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    root = Path(args.dataset_root).resolve()
    repo_id = args.repo_id or root.name
    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.device = device
    policy = ACTPolicy.from_pretrained(args.policy, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    chunk = policy.config.chunk_size
    fps = 30
    ds = LeRobotDataset(repo_id, root=str(root), episodes=parse_episodes(args.episodes),
                        delta_timestamps={ACTION_KEY: [i / fps for i in range(chunk)]})
    # error por [articulación, paso del bloque] acumulado, y de las líneas base
    err = np.zeros((6, chunk))
    base_state = np.zeros((6, chunk))
    base_last = np.zeros((6, chunk))
    n = 0
    for i in range(0, len(ds), args.stride):
        item = ds[i]
        valid = ~item["action_is_pad"].numpy() if "action_is_pad" in item else np.ones(chunk, bool)
        batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in item.items()
                 if k.startswith("observation.")}
        with torch.no_grad():
            pred = post(policy.predict_action_chunk(pre(batch)))            # (1, chunk, 6) desnormalizado
        pred = pred.squeeze(0).cpu().numpy()
        truth = item[ACTION_KEY].numpy()
        state = item[STATE_KEY].numpy()
        for t in range(chunk):
            if not valid[t]:
                continue
            err[:, t] += np.abs(pred[t] - truth[t])
            base_state[:, t] += np.abs(state - truth[t])
            base_last[:, t] += np.abs(truth[0] - truth[t])
        n += 1
    names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "pinza"]
    rep = {"episodios": args.episodes, "fotogramas_evaluados": n, "bloque": chunk, "unidades": "LeRobot (−100…100; pinza 0…100)",
           "error_medio_por_articulación_y_paso": {}}
    for h in [x for x in HORIZONS if x < chunk]:
        rep["error_medio_por_articulación_y_paso"][f"paso {h}"] = {
            "política": dict(zip(names, (err[:, h] / n).round(2).tolist())),
            "copiar_el_estado": dict(zip(names, (base_state[:, h] / n).round(2).tolist())),
            "mantener_la_última_acción": dict(zip(names, (base_last[:, h] / n).round(2).tolist())),
        }
    pol, cp = err.mean() / n, base_state.mean() / n
    rep["resumen"] = {"error_política": round(float(pol), 3), "error_copiar_estado": round(float(cp), 3),
                      "mejora_frente_a_copiar_el_estado": round(float(1 - pol / cp), 3) if cp else None}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
