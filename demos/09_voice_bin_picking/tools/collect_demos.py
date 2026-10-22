"""Graba demostraciones del experto en escenas aleatorias, en formato LeRobot (dataset v3). REQUIERE ISAAC SIM.

    # prueba de humo: 3 episodios (≈ 5 min), no toca el dataset principal
    python tools/collect_demos.py --headless --episodes 3 --dataset humo

    # el dataset inicial: 500 episodios válidos, por bloques que se pueden repetir y reanudar
    python tools/collect_demos.py --headless --episodes 100 --first-seed 0
    python tools/collect_demos.py --headless --episodes 100 --first-seed 1000 --resume    # mismo dataset, semillas nuevas
    ...

Cada episodio: escena nueva con piezas que caen y se asientan por física (separadas, juntas, apiladas), un objetivo
al azar entre las agarrables y alcanzables, el experto (``recoverykit.skills``, agarre por fricción) y el criterio
físico de éxito con la verdad del simulador; sólo se guardan los que levantan y transportan la pieza. En el 50 % se
añade ruido a la aproximación (se ejecuta con ruido y se graba la orden limpia: ver ``binpick/recorder.py``).

Escribe en ``data/<dataset>/`` (LeRobot v3: parquet + vídeos) y, en ``meta/``: ``binpick_info.json`` (recorte, unidades,
segmentador, ruido) y ``binpick_episodes.jsonl`` (cada intento con su semilla, causa de descarte, capa, holgura).
Tiempo estimado: no medido; el experto tarda ~6 s simulados por episodio y se renderizan 2 imágenes por fotograma, así
que cuenta con varios segundos de reloj por episodio. La primera tanda dirá el ritmo real (lo imprime).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import binpick  # noqa: E402,F401
from binpick import collect as COL, config as C, dataset_check as DC, recorder as R  # noqa: E402


def main() -> int:
    from recoverykit import runtime

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runtime.add_args(ap)
    ap.add_argument("--episodes", type=int, default=500, help="episodios VÁLIDOS que añadir en esta ejecución")
    ap.add_argument("--dataset", default="bin_pick_v1", help="nombre en data/ (también el repo_id local)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="añadir al dataset existente (salta las semillas ya hechas)")
    ap.add_argument("--noise-std", type=float, default=0.02, help="rad; ruido de aproximación (0 = sin ruido)")
    ap.add_argument("--noise-fraction", type=float, default=0.5)
    ap.add_argument("--segmenter", default="hsv", help="hsv | yolo:<pesos>: las máscaras de objetivo se graban con éste")
    ap.add_argument("--max-attempts", type=int, default=None)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    root = ROOT / "data" / args.dataset
    if not args.resume and (root / "meta" / "info.json").exists():
        print(f"ERROR: {root} ya existe. Usa --resume para añadir o --dataset otro_nombre.")
        return 1
    from binpick.robot_port import open_session

    cfg = C.BinPickCfg()
    s = open_session(args, cfg, segmenter=args.segmenter, realtime=0.0)
    robot = s.robot
    noise = R.NoiseCfg(args.noise_std, args.noise_fraction if args.noise_std > 0 else 0.0)
    ds = R.open_dataset(root, args.dataset, robot.spec, resume=args.resume)
    if not args.resume:
        R.write_info(root, robot.spec, robot.segmenter_name, cfg, noise)
    print(f"[collect] {args.dataset}: recorte {robot.spec.roi} → {robot.spec.out_hw}, segmentador {robot.segmenter_name}, "
          f"ruido {noise.as_dict()}", flush=True)
    summary = COL.collect(robot, root, ds, cfg, args.episodes, args.first_seed, noise, rng_seed=args.rng_seed,
                          max_attempts=args.max_attempts, stop=lambda: s.tele.stop_requested)
    ds.finalize()
    print("[collect] resumen:", summary, flush=True)
    print(DC.render_report(DC.check(root)), flush=True)
    s.rt.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
