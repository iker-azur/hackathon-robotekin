"""Run a policy in the SO-101 simulation and score it on physics alone.

This is the end of the playground. Three things can be run:

``--type act``       a LeRobot ACT checkpoint from demo 15.
``--type bc``        the Behavior Cloning checkpoint from demo 14.
``--type scripted``  the *teacher*, with its privileged knowledge of where the cube
                     is. It is not a learned policy and every report says so; it is
                     here so a learned number has an upper bound next to it.

Every step it is the same three lines:

    observe  ->  policy  ->  IK command  ->  step the simulator

What is different from the first version of this script
-------------------------------------------------------

* **No kinematic hold.** The teacher pins the cube to the gripper while it is
  closed. A policy gets nothing of the sort, and neither does the scripted
  baseline here: the cube is held by friction or not at all.
* **Success is a lift, a held carry and a placement** (``il_metrics.py``), not the
  cube's final x, y. A cube that is pushed to the drop point is a failure.
* **Held-out cube positions.** Evaluation episode ``i`` uses a cube position and
  yaw that no demonstration used (``il_env.sample_eval_spot``). The old evaluation
  re-used the demonstrations' own spots. The distance to the nearest training spot
  is printed per episode.
* **A report per run, never overwritten.** Each run writes
  ``results/runs/<name>.json`` and ``.md`` and refuses to replace an existing one,
  so a two-episode diagnostic cannot erase a ten-episode benchmark.
* **The action queue is reset between episodes.** ACT keeps the rest of an
  unfinished chunk; without ``policy.reset()`` the next episode starts by replaying
  the end of the previous one.

Watching it
-----------

Leave ``--headless`` off and Isaac Sim opens its window; add ``--speed`` to pace it:

    python .\\demos\\06_imitation_learning\\16_evaluate_policy.py ^
        --policy .\\outputs\\15_act_clock\\checkpoints\\last\\pretrained_model ^
        --episodes 1 --speed 0.5

The task's own camera is aimed at ``(0.9, 0, -0.3)``, which is empty floor in this
scene; the defaults here were measured against the arm's actual workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEMO_DIR = Path(__file__).resolve().parent
RUNS_DIR = DEMO_DIR / "results" / "runs"

# The plan is 640 steps; the extra steps let a lowered cube settle so the judge can
# see that it stopped.
DEFAULT_STEPS = 720

# What each input kind means, written into every report. "PRIVILEGED" marks a signal
# a real robot would not have.
INPUT_DESCRIPTIONS = {
    "image_state": "camera image + joint state (nothing privileged)",
    "phase": "camera image + joint state + PRIVILEGED plan phase index",
    "detector": "camera image + joint state + cube pixel centroid from the colour detector "
                "(perception, not privileged)",
    "clock_cube": "PRIVILEGED plan clock + cube pixel position seen once at the start by the colour "
                  "detector; no camera image and no joint state go into the network, so it cannot "
                  "react to the arm",
    "scripted": "SCRIPTED teacher with the cube's true pose (privileged) - NOT a learned policy",
    "bc": "Behavior Cloning network (demo 14): camera image + joint state + PRIVILEGED plan phase index",
}


def phase_bounds() -> list[tuple[int, int, str]]:
    """The teacher's phase schedule as ``(first_step, last_step, name)`` rows."""
    import expert as expert_module

    rows, start = [], 0
    for phase in expert_module.PLAN:
        rows.append((start, start + phase.steps, phase.name))
        start += phase.steps
    return rows


def phase_at(bounds, frame: int) -> str:
    for first, last, name in bounds:
        if first <= frame < last:
            return name
    return "past plan"


def load_act_policy(path: Path, device: str, action_steps: int | None = None):
    """Load a LeRobot checkpoint with its pre- and post-processors.

    The processors matter: they normalise the observations with the statistics
    computed at training time and un-normalise the action the network outputs.
    Loading only the weights silently gives a policy that speaks a different
    language from the one it was trained in.

    The device is forced at load time because it is baked into the checkpoint's
    ``config.json``: a policy saved from a CUDA run would otherwise load onto the
    GPU, and that CUDA context then collides with Isaac Lab's own (an illegal
    memory access inside the camera update). On the CPU a chunk costs a few tens of
    milliseconds.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    saved = PreTrainedConfig.from_pretrained(str(path))
    saved.device = device
    if action_steps is not None:
        saved.n_action_steps = min(action_steps, saved.chunk_size)
    policy = ACTPolicy.from_pretrained(str(path), config=saved)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def load_bc_policy(path: Path, device: str):
    """Load the plain-PyTorch checkpoint written by demo 14."""
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    normalisation = torch.load(
        path.parent / "normalisation.pt", map_location=device, weights_only=True
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_bc", DEMO_DIR / "14_train_behavior_cloning.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_bc"] = module
    spec.loader.exec_module(module)
    policy = module.BCPolicy(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        phase_dim=checkpoint.get("phase_dim", 0),
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, normalisation["action_mean"].to(device), normalisation["action_std"].to(device)


def place_cube(env, spot, yaw: float = 0.0) -> None:
    """Put the cube on its spot, let it settle, and refresh the camera.

    ``yaw`` matters: demo 13 sweeps the cube's turn over +-30 degrees because it is
    the only source of variety in an otherwise static image.
    """
    import torch

    import il_env

    from isaaclab.utils.math import quat_from_euler_xyz

    cube = env.scene["cube"]
    resting = quat_from_euler_xyz(
        torch.zeros(1, device=env.device),
        torch.zeros(1, device=env.device),
        torch.tensor([yaw], device=env.device),
    )
    pose = torch.tensor([[spot[0], spot[1], il_env.CUBE_REST_Z]], device=env.device)
    cube.write_root_pose_to_sim(torch.cat([pose, resting], dim=-1))
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(20):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    il_env.refresh_camera(env)


def _clip_ik_command(command):
    """Clip an 8-number IK command to the arm's reach, in the robot base frame.

    Only the position (first three numbers) is bounded, against a box well beyond
    the arm's measured reach of 0.33 m; the orientation is a unit quaternion the
    teacher always sent the same, so it is left alone.
    """
    import numpy as np

    reach = 0.45
    clipped = np.array(command, dtype=np.float32).copy()
    clipped[0] = float(np.clip(clipped[0], -reach, reach))
    clipped[1] = float(np.clip(clipped[1], -reach, reach))
    clipped[2] = float(np.clip(clipped[2], -0.6, reach))
    return clipped


def hold_still_command(env):
    """An IK command that keeps the gripper where it is, jaw open.

    Used when an episode has no perception to act on (the detector found no cube):
    the arm waits, and the judge records a failure rather than the run crashing.
    """
    import torch

    from isaaclab.utils.math import quat_apply, quat_inv

    robot = env.scene["robot"]
    gripper = robot.find_bodies("gripper")[0][0]
    position = quat_apply(
        quat_inv(robot.data.root_quat_w), robot.data.body_pos_w[:, gripper] - robot.data.root_pos_w
    )
    grip_quat = quat_inv(robot.data.root_quat_w)
    open_jaw = torch.ones(1, 1, device=env.device)
    return torch.cat([position, grip_quat, open_jaw], dim=-1).squeeze(0).cpu().numpy()


def git_state() -> dict:
    """Commit and dirty flag of the checkout, so a report says what produced it."""
    def run(*cmd):
        try:
            return subprocess.run(
                ["git", *cmd], cwd=DEMO_DIR, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - a missing git must not stop an evaluation
            return ""

    return {
        "commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain", "--", ".")),
    }


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-")[:60]


def default_report_name(args, policy_path: Path | None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if args.type == "scripted":
        who = "scripted"
    else:
        # .../outputs/<run>/checkpoints/<step>/pretrained_model -> "<run>-<step>"
        parts = policy_path.parts
        if "checkpoints" in parts:
            index = parts.index("checkpoints")
            who = f"{parts[index - 1]}-{parts[index + 1]}"
        else:
            who = policy_path.stem
    first = f"from{args.first_episode}_" if getattr(args, "first_episode", 0) else ""
    return f"{slug(who)}_{args.spots}_{first}{args.episodes}ep_{stamp}"


def write_report(name: str, record: dict) -> tuple[Path, Path]:
    """Write ``results/runs/<name>.json`` and ``.md``. Refuses to replace either."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json_path, md_path = RUNS_DIR / f"{name}.json", RUNS_DIR / f"{name}.md"
    for path in (json_path, md_path):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Reports are never overwritten; pass --report-name "
                "with a different name."
            )
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(record), encoding="utf-8")
    return json_path, md_path


def render_markdown(record: dict) -> str:
    run, summary = record["run"], record["summary"]
    lines = [
        f"# Evaluación `{run['name']}`",
        "",
        f"- Fecha (UTC): {run['utc']}",
        f"- Código: commit `{run['git']['commit']}`"
        f"{' (con cambios sin confirmar)' if run['git']['dirty'] else ''}, rama `{run['git']['branch']}`",
        f"- Política: `{run['policy']}` (tipo `{run['type']}`)",
        f"- **Entradas de la política:** {run['inputs']}",
        f"- Posiciones del cubo: {run['spots_description']}",
        f"- Semilla: {run['seed']}; pasos por episodio: {run['steps']}",
        "",
        f"**Resultado: {summary['successes']}/{summary['episodes']} "
        f"({100 * summary['rate']:.0f} %).** "
        + ("Esto es el guion del profesor, no una política aprendida. " if run["type"] == "scripted" else "")
        + ("La ejecución se interrumpió antes de terminar. " if run.get("interrupted") else "")
        + "Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del "
        "objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.",
        "",
    ]
    if run.get("scripted_grasp_bias_mm") is not None:
        lines.insert(8, "- Desplazamiento del punto de agarre del guion respecto al centro del cubo "
                        f"(mm, x e y del mundo): {run['scripted_grasp_bias_mm']}")
    if summary["failures"]:
        lines.append(
            "Fallos por causa: "
            + ", ".join(f"`{reason}` × {count}" for reason, count in sorted(summary["failures"].items()))
            + "."
        )
        lines.append("")
    lines += [
        "| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) "
        "| transporte (pasos) | sujeción | error final (cm) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for episode in record["episodes"]:
        v = episode["verdict"]
        lines.append(
            f"| {episode['episode']} | {episode['spot'][0]:.3f}, {episode['spot'][1]:.3f} "
            f"| {episode['yaw_deg']:.0f} | {episode['nearest_train_cm']:.1f} "
            f"| {'sí' if v['success'] else 'no'} | {v['failure'] or '-'} | {v['max_lift_cm']:.1f} "
            f"| {v['longest_carry_steps']} | {v['held_fraction']:.2f} | {v['final_error_cm']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def predict(args, policy, preprocessor, postprocessor, action_mean, action_std, env_kind,
            state, image, step, device, cube_uv):
    """One observation in, one 8-number IK command out."""
    import torch

    import il_dataset as data

    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    image_t = torch.as_tensor(data.image_to_policy_input(image), device=device).unsqueeze(0)
    phase_t = torch.as_tensor(
        data.phase_vector(step), dtype=torch.float32, device=device
    ).unsqueeze(0)

    if args.type == "bc":
        predicted = policy(image_t, state_t, phase_t) * action_std + action_mean
        return predicted.squeeze(0).cpu().numpy()

    batch = {}
    wanted = policy.config.input_features
    if data.STATE_KEY in wanted:
        batch[data.STATE_KEY] = state_t
    elif env_kind == "clock_cube":
        # LeRobot 0.4.4's ACT reads ``batch["observation.state"].device`` even with no
        # robot state configured. The network never uses it (it is not an input
        # feature of this checkpoint), so it gets zeros rather than the real state.
        batch[data.STATE_KEY] = torch.zeros_like(state_t)
    if data.IMAGE_KEY in wanted:
        batch[data.IMAGE_KEY] = image_t
    if data.ACT_ENV_STATE_KEY in wanted:
        if env_kind == "phase":
            env_state = phase_t
        elif env_kind == "detector":
            import cube_detect

            detection = cube_detect.detect_cube(image)
            values = (0.0, 0.0, 0.0) if detection is None else (
                *data.normalised_uv(detection, image.shape), 1.0)
            env_state = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
        elif env_kind == "clock_cube":
            env_state = torch.as_tensor(
                (data.plan_clock(step), *cube_uv), dtype=torch.float32, device=device
            ).unsqueeze(0)
        else:
            raise ValueError(f"unknown environment-state kind {env_kind!r}")
        batch[data.ACT_ENV_STATE_KEY] = env_state
    action = policy.select_action(preprocessor(batch))
    return postprocessor(action).squeeze(0).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", default=None, help="checkpoint directory or .pt file "
                        "(not needed for --type scripted)")
    parser.add_argument("--type", choices=["act", "bc", "scripted"], default="act")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"simulator steps per episode (default {DEFAULT_STEPS}; the plan is 640)")
    parser.add_argument("--spots", choices=["heldout", "outside", "train"], default="heldout",
                        help="heldout: cube positions no demonstration used, interleaved with the "
                             "training ones (default); outside: only positions outside the training box, in the 1 cm "
                             "ring around it, so the policy must extrapolate; train: the demonstrations' own positions")
    parser.add_argument("--first-episode", type=int, default=0,
                        help="index of the first evaluation cube position (default 0). Use 10 after a "
                             "10-episode run to evaluate on different unseen positions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-name", default=None,
                        help="name of the files under results/runs/ (default: policy, spots, "
                             "episodes and a timestamp). An existing name is refused.")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window; omit it to watch the policy in the viewport")
    parser.add_argument(
        "--speed", type=float, default=0.0,
        help="0 = as fast as the machine allows, 1 = real time (60 steps per second), "
             "0.25 = quarter speed. Ignored with --headless.",
    )
    parser.add_argument("--viewer-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--viewer-lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--device", default="cuda:0", help="device for the simulator")
    parser.add_argument("--inference-device", default="cpu",
                        help="device for the policy (default cpu: sharing the GPU with Isaac Sim "
                             "makes PhysX crash with an illegal memory access on this machine)")
    parser.add_argument("--action-steps", type=int, default=None,
                        help="how many of the predicted chunk to run before asking the policy "
                             "again (default: the checkpoint's own value)")
    parser.add_argument("--scripted-bias-mm", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="--type scripted only: override where the expert aims relative to the "
                             "cube centre (world x, y in mm). 0 0 is the original, unshifted "
                             "expert, whose grasp point sat at the edge of what the fingers tolerate")
    parser.add_argument("--stall-seconds", type=float, default=300.0,
                        help="if no simulation step completes for this long, write the episodes "
                             "finished so far and exit with code 3 (default 300)")
    parser.add_argument("--trace", action="store_true",
                        help="print the command the network produces every --trace-every steps")
    parser.add_argument("--trace-every", type=int, default=20)
    args = parser.parse_args()

    import numpy as np
    import torch

    import expert as expert_module
    import il_dataset as data
    import il_env
    import il_metrics

    from il_env import prepare_runtime

    if args.type != "scripted" and not args.policy:
        print("ERROR: --policy is required unless --type scripted")
        return 1
    policy_path = Path(args.policy).resolve() if args.policy else None
    if policy_path is not None and not policy_path.exists():
        print(f"ERROR: no checkpoint at {policy_path}")
        return 1

    report_name = args.report_name or default_report_name(args, policy_path)
    # Fail before the minute of Kit start-up, not after the episodes have been run.
    for suffix in (".json", ".md"):
        if (RUNS_DIR / f"{report_name}{suffix}").exists():
            print(f"ERROR: results/runs/{report_name}{suffix} already exists. Reports are never "
                  "overwritten; pass --report-name with a different name.")
            return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.scripted_bias_mm is not None:
        if args.type != "scripted":
            print("ERROR: --scripted-bias-mm only applies to --type scripted")
            return 1
        expert_module.GRASP_BIAS_M = (args.scripted_bias_mm[0] / 1000.0, args.scripted_bias_mm[1] / 1000.0, 0.0)

    viewer = il_env.default_viewer()
    if args.viewer_eye:
        viewer["eye"] = tuple(args.viewer_eye)
    if args.viewer_lookat:
        viewer["lookat"] = tuple(args.viewer_lookat)

    inference_device = args.inference_device
    if inference_device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to the CPU. This will be slow.")
        inference_device = "cpu"

    # The policy is loaded before the simulator starts: it is small, the simulator
    # is not, and a wrong path should not cost a minute of Kit.
    policy = preprocessor = postprocessor = action_mean = action_std = None
    if args.type == "act":
        print(f"loading act policy from {policy_path}")
        policy, preprocessor, postprocessor = load_act_policy(
            policy_path, inference_device, action_steps=args.action_steps
        )
        features = getattr(policy.config, "input_features", {})
        env_feature = features.get(data.ACT_ENV_STATE_KEY)
        if env_feature is None:
            env_kind = "image_state"
        else:
            env_kind = data.resolve_env_state_kind(policy_path, int(env_feature.shape[0]))
            if env_kind is None:
                print(f"ERROR: cannot tell what the {env_feature.shape[0]}-wide environment state "
                      "of this checkpoint holds; there is no il_inputs.json next to it")
                return 1
    elif args.type == "bc":
        print(f"loading bc policy from {policy_path}")
        policy, action_mean, action_std = load_bc_policy(policy_path, inference_device)
        env_kind = "bc"
    else:
        env_kind = "scripted"
    print(f"inputs: {INPUT_DESCRIPTIONS[env_kind]}")

    prepare_runtime(headless=args.headless)

    # The same environment the teacher ran in, so a predicted action is applied
    # exactly as the teacher applied it.
    env = il_env.build_env(kind="ik_teacher", record=False, device=args.device, viewer=viewer)
    il_env.assert_joint_order(env)

    if args.type == "act":
        n_steps = args.action_steps if args.action_steps is not None else policy.config.n_action_steps
        n_chunk = policy.config.chunk_size
        print(f"chunk {n_chunk}, {n_steps} actions run before the policy observes again")

    speed = args.speed if not args.headless else 0.0
    step_seconds = float(env.cfg.sim.dt) * float(env.cfg.decimation)
    pace_seconds = step_seconds / speed if speed > 0 else 0.0

    drop_x, drop_y = expert_module.DROP_OFFSET_XY
    bounds = phase_bounds()

    if args.spots == "heldout":
        spot_of, yaw_of = il_env.sample_eval_spot, il_env.sample_eval_yaw
        spots_description = (
            f"fuera del entrenamiento (índices Halton {il_env.EVAL_EPISODE_OFFSET + args.first_episode} "
            f"a {il_env.EVAL_EPISODE_OFFSET + args.first_episode + args.episodes - 1}; ninguna posición "
            "de las demostraciones)"
        )
    elif args.spots == "outside":
        spot_of, yaw_of = il_env.sample_outside_spot, il_env.sample_outside_yaw
        spots_description = (
            "fuera de la caja de entrenamiento, en el anillo de 1 cm que la rodea (la política "
            "tiene que extrapolar)"
        )
    else:
        def spot_of(i):
            return il_env.sample_cube_spot(i, 0)

        def yaw_of(i):
            return il_env.sample_cube_yaw(i, 0)

        spots_description = "las mismas posiciones que las demostraciones (mide memoria, no generalización)"

    episodes: list[dict] = []
    started = time.time()
    progress = {"time": time.time(), "finalizing": False}
    partial_path = RUNS_DIR / f"{report_name}.partial.jsonl"

    def finalize(status: str) -> int:
        """Write the report from the episodes finished so far and return an exit code.

        ``status`` is ``""`` for a complete run, otherwise why it is not (``interrupted``
        or ``stalled``); the report says so. Called once, either at the end of the loop
        or by the watchdog below.
        """
        if progress["finalizing"]:
            return 0
        progress["finalizing"] = True
        if not episodes:
            print("no episode finished; nothing to report", flush=True)
            return 1
        successes = sum(1 for e in episodes if e["verdict"]["success"])
        failures: dict[str, int] = {}
        for e in episodes:
            reason = e["verdict"]["failure"]
            if reason:
                failures[reason] = failures.get(reason, 0) + 1
        record = {
            "run": {
                "name": report_name,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git": git_state(),
                "policy": str(policy_path) if policy_path else "scripted teacher",
                "type": args.type,
                "inputs": INPUT_DESCRIPTIONS[env_kind],
                "spots": args.spots,
                "spots_description": spots_description,
                "seed": args.seed,
                "steps": args.steps,
                "action_steps": args.action_steps,
                "scripted_grasp_bias_mm": [round(v * 1000, 1) for v in expert_module.GRASP_BIAS_M[:2]]
                if args.type == "scripted" else None,
                "interrupted": bool(status),
                "status": status or "complete",
                "episodes_requested": args.episodes,
                "criteria": {
                    "lift_min_m": il_metrics.LIFT_MIN_M,
                    "carry_z_m": il_metrics.CARRY_Z_M,
                    "carry_min_steps": il_metrics.CARRY_MIN_STEPS,
                    "held_max_dist_m": il_metrics.HELD_MAX_DIST_M,
                    "place_tol_m": il_metrics.PLACE_TOL_M,
                },
            },
            "episodes": list(episodes),
            "summary": {
                "episodes": len(episodes),
                "successes": successes,
                "rate": successes / len(episodes),
                "failures": failures,
            },
        }
        _, md_path = write_report(report_name, record)
        print("")
        label = "success" if not status else f"PARTIAL ({status}, {len(episodes)} of {args.episodes} episodes)"
        print(f"{label}: {successes}/{len(episodes)} ({100 * successes / len(episodes):.0f}%)")
        print(f"report:  {md_path.relative_to(DEMO_DIR.parent.parent)}")
        print(f"total time: {time.time() - started:.0f}s", flush=True)
        return 0

    def watchdog() -> None:
        # Isaac Sim can hang inside a step or a reset with the CPU still busy and no
        # message (seen once in a 20-episode run, after 17 good episodes). Nothing in
        # Python can interrupt that, so a thread notices that no step has completed
        # and writes what was finished before the whole process is left.
        while not progress["finalizing"]:
            time.sleep(10)
            if time.time() - progress["time"] > args.stall_seconds:
                print(f"STALLED: no simulation step completed for {args.stall_seconds:.0f} s; "
                      "writing the episodes finished so far and exiting", flush=True)
                code = finalize("stalled")
                sys.stdout.flush()
                os._exit(3 if code == 0 else code)

    threading.Thread(target=watchdog, daemon=True).start()

    try:
        for episode in range(args.first_episode, args.first_episode + args.episodes):
            progress["time"] = time.time()
            spot, yaw = spot_of(episode), yaw_of(episode)
            env.reset()
            il_env.reset_arm(env)
            target = (spot[0] + drop_x, spot[1] + drop_y)
            teacher = None
            if args.type == "scripted":
                # ``bind`` places the cube and plans from where it really is. The
                # pin is never called: this baseline is held by friction alone.
                teacher = expert_module.Expert(env)
                teacher.bind(spot, yaw)
                il_env.refresh_camera(env)
            else:
                place_cube(env, spot, yaw)
                if hasattr(policy, "reset"):
                    policy.reset()

            # Perception done once, at the start, for the input kinds that use it.
            cube_uv = None
            if env_kind == "clock_cube":
                cube_uv = il_env.perceive_cube(env)
                if cube_uv is None:
                    print(f"  episode {episode}: the detector found no cube; scored as a failure")

            judge = il_metrics.EpisodeJudge(target_xy=target)
            episode_start = time.time()
            pace_deadline = time.perf_counter()
            with torch.no_grad():
                for step in range(args.steps):
                    if teacher is not None:
                        command = teacher.action()
                        command = teacher.last_action if command is None else command
                        command = command.squeeze(0).cpu().numpy()
                    elif env_kind == "clock_cube" and cube_uv is None:
                        command = hold_still_command(env)
                    else:
                        observation = env.unwrapped.observation_manager.compute()
                        state, image = data.observation_from_env(observation)
                        command = predict(
                            args, policy, preprocessor, postprocessor, action_mean, action_std,
                            env_kind, state, image, step, inference_device, cube_uv,
                        )

                    env_action = torch.as_tensor(
                        _clip_ik_command(command), dtype=torch.float32, device=env.device
                    ).unsqueeze(0)
                    il_env.step_obs(env, env_action)
                    progress["time"] = time.time()

                    cube = env.scene["cube"].data.root_pos_w[0]
                    judge.update(cube.cpu().numpy(), il_env.cube_hold_distance(env))

                    if args.trace and (step + 1) % args.trace_every == 0:
                        raw = np.asarray(command).reshape(-1)
                        print(
                            f"    step {step + 1:4d} [{phase_at(bounds, step)}] "
                            f"xyz=({raw[0]:+.3f},{raw[1]:+.3f},{raw[2]:+.3f}) grip={raw[7]:+.2f} "
                            f"cube=({float(cube[0]):.3f},{float(cube[1]):.3f},{float(cube[2]):.3f})",
                            flush=True,
                        )
                    if pace_seconds:
                        overrun = time.perf_counter() - pace_deadline
                        if overrun < pace_seconds:
                            time.sleep(pace_seconds - overrun)
                        pace_deadline = time.perf_counter() + pace_seconds

            verdict = judge.verdict()
            episodes.append({
                "episode": episode,
                "spot": [round(spot[0], 4), round(spot[1], 4)],
                "yaw_deg": round(float(np.degrees(yaw)), 1),
                "nearest_train_cm": round(il_env.nearest_training_spot_m(spot) * 100, 2),
                "cube_uv": None if cube_uv is None else [round(float(v), 4) for v in cube_uv],
                "verdict": verdict.as_dict(),
                "wall_clock_s": round(time.time() - episode_start, 1),
            })
            # Appended as soon as the episode ends, so a crash or a hang loses nothing.
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episodes[-1], ensure_ascii=False) + "\n")
            progress["time"] = time.time()
            print(
                f"  episode {episode}: {'ok' if verdict.success else 'FAILED'}"
                f"{'' if verdict.success else ' (' + verdict.failure + ')'} "
                f"lift={verdict.max_lift_m * 100:.1f}cm carry={verdict.longest_carry_steps}steps "
                f"held={verdict.held_fraction:.2f} final_error={verdict.final_error_m * 100:.1f}cm "
                f"({time.time() - episode_start:.0f}s)",
                flush=True,
            )
    except KeyboardInterrupt:
        print("interrupted; reporting the episodes finished so far", flush=True)
        sys.stdout.flush()
        os._exit(finalize("interrupted"))

    code = finalize("")
    sys.stdout.flush()
    # ``env.close()`` can hang for minutes with cameras (Kit shutdown). The reports
    # are already on disk, so leave without it.
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
