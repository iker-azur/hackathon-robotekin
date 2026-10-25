"""How smooth are the commands a pick policy sends, and what do chunk blending and a target filter change?

Takes the held-out episodes of ``data/real_so101/act_pick_super``, predicts one chunk per recorded frame with the
checkpoint, and replays the strategies from those chunks (the real ``smooth_motion.BlendedPolicy`` /
``TargetFilter``, so what is measured is what the app runs):

    open N    what the apps did: N actions out of each prediction, open loop, then predict again
    blend N   a prediction every N steps, all the overlapping chunks averaged (N = 1 is ACT's temporal ensembling)
    + filter  the critically damped target filter on top

Per strategy, on the five arm joints (degrees, 30 fps): mean speed, RMS acceleration and jerk, the 99th percentile
of the acceleration (a snap shows there), and the distance to the leader arm's recorded action. The row ``leader``
is the human teleoperation itself: the demonstrations are what "smooth" means for this arm. The recorded
observations do not depend on what the policy commands, so this compares command streams, not task success.

    python .\\demos\\06_imitation_learning\\analyze_motion_smoothness.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
DATASET_REPO_ID = "local/so101_act"  # must match 19_build_so101_act_dataset.py
FPS = 30
RESULTS = DEMO_DIR / "results" / "motion_smoothness.json"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), DEMO_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Cached:
    """Stands in for the policy: ``predict_action_chunk`` returns the chunk predicted at the current frame."""

    def __init__(self, chunks):
        self.chunks, self.frame, self.config = chunks, 0, type("C", (), {"n_action_steps": 1})

    def predict_action_chunk(self, batch):
        import torch

        return torch.from_numpy(self.chunks[self.frame])[None]

    def reset(self):
        pass


def stream(chunks, strategy: str, stride: int, filter_hz: float | None, coeff: float | None = None) -> np.ndarray:
    """The commands one strategy sends over an episode, from the chunk predicted at every frame."""
    smooth = load("smooth_motion.py")
    policy = Cached(chunks)
    blended = smooth.BlendedPolicy(policy, stride, smooth.DEFAULT_COEFF if coeff is None else coeff) \
        if strategy == "blend" else None
    target_filter = smooth.TargetFilter(filter_hz) if filter_hz else None
    out, queue = [], []
    for t in range(len(chunks)):
        policy.frame = t
        if strategy == "open":
            if not queue:
                queue = list(chunks[t][:stride])
            action = queue.pop(0)
        else:
            action = blended.select_action(None)[0].numpy()
        out.append(target_filter(action, 1 / FPS) if target_filter else action)
    return np.asarray(out)


def smoothness(episodes: list[np.ndarray], truth: list[np.ndarray]) -> dict:
    speed, accel, jerk, error = [], [], [], []
    for y, a in zip(episodes, truth):
        arm = y[:, :5]
        speed.append(np.abs(np.diff(arm, axis=0)) * FPS)
        accel.append(np.diff(arm, 2, axis=0) * FPS ** 2)
        jerk.append(np.diff(arm, 3, axis=0) * FPS ** 3)
        error.append(np.abs(arm - a[:, :5]))
    speed, accel, jerk, error = (np.concatenate(v) for v in (speed, accel, jerk, error))
    return {"speed_deg_s": float(speed.mean()), "accel_rms_deg_s2": float(np.sqrt((accel ** 2).mean())),
            "accel_p99_deg_s2": float(np.percentile(np.abs(accel), 99)),
            "jerk_rms_deg_s3": float(np.sqrt((jerk ** 2).mean())), "error_to_leader_deg": float(error.mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="IkerAzurmendi/act-pick-v2-30000")
    parser.add_argument("--dataset", default="act_pick_super")
    parser.add_argument("--episodes", type=int, default=8, help="how many held-out episodes")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    try:  # this PC's Python does not trust the Windows certificate store by itself
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = load("24_pick_and_classify_app.py")
    policy, pre, post = app.load_pick_components(args.policy, device)
    root = DEMO_DIR / "data" / "real_so101" / args.dataset
    sources = json.loads((root / "meta" / "sources.json").read_text(encoding="utf-8"))
    held = [s["episode_index"] for s in sources if s["holdout"]][:args.episodes]
    chunks, truth, millis = [], [], []
    for episode in held:
        dataset = LeRobotDataset(DATASET_REPO_ID, root=root, episodes=[episode])
        policy.reset()
        episode_chunks, episode_truth = [], []
        for i in range(len(dataset)):
            item = dataset[i]
            batch = {"observation.state": item["observation.state"][None].to(device), "task": "", "robot_type": ""}
            for name in ("front", "wrist"):
                batch[f"observation.images.{name}"] = item[f"observation.images.{name}"][None].to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(pre(batch)))[0].cpu().numpy()
            millis.append(1000 * (time.perf_counter() - started))
            episode_chunks.append(chunk)
            episode_truth.append(item["action"].numpy())
        chunks.append(np.asarray(episode_chunks))
        truth.append(np.asarray(episode_truth))
        print(f"episode {episode}: {len(dataset)} frames", flush=True)

    # (kind, stride, filter Hz, weight coefficient); None = the default of smooth_motion.py (-0.2)
    strategies = [("open", 30, None, None), ("open", 10, None, None), ("open", 30, 6.0, None),
                  ("blend", 30, None, 0.01), ("blend", 5, None, 0.01), ("blend", 1, None, 0.01),
                  ("blend", 5, 4.0, 0.01), ("blend", 3, 4.0, 0.01),
                  ("blend", 5, None, None), ("blend", 3, None, None), ("blend", 2, None, None),
                  ("blend", 5, 4.0, None), ("blend", 3, 4.0, None), ("blend", 3, 6.0, None), ("blend", 2, 4.0, None)]
    report = {"policy": args.policy, "episodes": held, "frames": int(sum(len(t) for t in truth)),
              "inference_ms_median": float(np.median(millis)), "inference_ms_p95": float(np.percentile(millis, 95)),
              "leader": smoothness(truth, truth), "strategies": {}}
    for kind, stride, hz, coeff in strategies:
        name = (f"{kind} {stride}" + (f" w{coeff:+g}" if kind == "blend" and coeff is not None else "")
                + (f" + filter {hz:g} Hz" if hz else ""))
        report["strategies"][name] = smoothness([stream(c, kind, stride, hz, coeff) for c in chunks], truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['frames']} frames, inference {report['inference_ms_median']:.0f} ms median "
          f"({report['inference_ms_p95']:.0f} ms p95) on {device}")
    print(f"{'':28}{'speed':>8}{'acc rms':>10}{'acc p99':>10}{'jerk rms':>11}{'err':>7}")
    rows = {"leader (human)": report["leader"], **report["strategies"]}
    for name, m in rows.items():
        print(f"{name:28}{m['speed_deg_s']:8.1f}{m['accel_rms_deg_s2']:10.0f}{m['accel_p99_deg_s2']:10.0f}"
              f"{m['jerk_rms_deg_s3']:11.0f}{m['error_to_leader_deg']:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
