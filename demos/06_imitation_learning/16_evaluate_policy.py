"""Run a trained policy in the SO-101 simulation and score it.

This is the end of the playground. The policy sees only the front camera and the
joint positions, and has to pick the cube up and put it down 10 cm sideways.
Nothing tells it where the cube is.

Two policies are supported:

``--type act``    a checkpoint produced by LeRobot's training (demo 15). Loaded
                  with ``ACTPolicy.from_pretrained``, together with the pre- and
                  post-processors saved next to the weights.
``--type bc``     the checkpoint from demo 14, loaded with the class that trained
                  it, for comparison.

Every step it is the same three lines:

    observe  ->  policy  ->  IK command  ->  step the simulator

Watching it
-----------

Leave ``--headless`` off and Isaac Sim opens its window: omit it, add
``--speed``, and you are looking at the policy instead of waiting for it.

    python .\\demos\\06_imitation_learning\\16_evaluate_policy.py ^
        --policy .\\outputs\\15_act\\checkpoints\\last\\pretrained_model ^
        --episodes 1 --steps 400 --speed 0.5

Two things make the difference between a window you can watch and one you cannot:

* **``--speed``.** The simulator runs about 30 steps a second when it has a GPU
  to itself, so 800 steps are gone in half a minute. ``--speed 1`` paces the
  episode to real time (the task runs 60 physics steps per second), ``--speed 0.25``
  in quarter speed.
* **``--viewer-eye`` / ``--viewer-lookat``.** The task's own camera is aimed at
  ``(0.9, 0, -0.3)``, which is empty floor in this scene; the defaults here were
  measured against the arm's actual workspace.

Output
------

A per-episode table and ``results/16_evaluation.md``. An episode counts as a
success when the cube finishes within 3 cm of where the demonstrations put it -
the same criterion used to accept a demonstration - and the cube's highest point
is reported beside it, because a policy that flings the cube has already failed
even if it ends in the right place.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same criterion as the demonstrations, imported from the expert's module so the
# two cannot drift apart.
PLACEMENT_TOLERANCE_M = 0.03

# How many simulator steps an episode may take. The demonstrations' plan is 640
# steps, so 800 leaves room for a policy that hesitates but ends the episode.
DEFAULT_STEPS = 800


def phase_bounds() -> list[tuple[int, int, str]]:
    """The teacher's phase schedule as ``(first_step, last_step, name)`` rows.

    A reference for reading a trace, not a rule: the policy is under no obligation
    to be in the phase the demonstrations were in at the same step. It answers
    "the arm stopped making progress during *carry*" rather than "step 312".
    """
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

    The processors matter: they normalise the images and the state with the
    statistics computed at training time, and un-normalise the action the network
    outputs. Loading only the weights - which is tempting - silently gives a
    policy that speaks a different language from the one it was trained in.

    ``action_steps`` overrides how many of the predicted chunk are executed
    before the policy is asked again. It can be low enough to change the
    behaviour of a trained policy without retraining it, which is the cheapest
    way to test a hypothesis about the closed loop.

    Two things have to be forced at load time, not afterwards, because both are
    baked into the checkpoint's ``config.json``:

    * the device it was trained on, and
    * the device of the saved pre-processor pipeline.

    A policy saved from a CUDA run loads its weights onto the GPU even when the
    evaluator asks for the CPU afterwards, and that surviving CUDA context then
    collides with Isaac Lab's own: the camera's ``_update_outdated_buffers``
    fails with a CUDA illegal memory access that cannot be traced back to
    anything this script did. The policy is therefore started on the CPU here,
    which costs a few tens of milliseconds per chunk and buys a run that finishes.
    """
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Read the checkpoint's own config and change only where it runs, and how
    # much of the chunk to replay. Building a new ACTConfig instead would drop the
    # input features the architecture was trained with, and ACTConfig validates
    # against their absence at construction time.
    from lerobot.configs.policies import PreTrainedConfig

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
    # The class is imported from the script that trained it, rather than
    # re-declared here, so there is exactly one definition of the network.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_bc", Path(__file__).resolve().parent / "14_train_behavior_cloning.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_bc"] = module
    spec.loader.exec_module(module)
    policy = module.BCPolicy(
        state_dim=checkpoint["state_dim"], action_dim=checkpoint["action_dim"]
    ).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, normalisation["action_mean"].to(device), normalisation["action_std"].to(device)


def place_cube(env, spot, yaw: float = 0.0) -> None:
    """Put the cube on its spot and let it settle before the episode starts.

    The demonstrations' ``Expert.bind`` does exactly this and then goes on to
    plan; the evaluation calls it directly because it has no plan to run.

    ``yaw`` matters and used to be missing. Demo 13 sweeps the cube's turn over
    +-30 degrees because it is the only source of variety in an otherwise static
    image; this function always wrote the upright quaternion, so every evaluated
    episode started with the cube axis-aligned - the middle of the training sweep
    and its easiest case. Pass ``il_env.sample_cube_yaw`` to test the same
    distribution the policy was trained on.
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


def _clip_ik_command(env, command):
    """Clip an 8-number IK command to the arm's reach, in the robot base frame.

    Only the first three numbers - the target position - are bounded, against a
    conservative box in front of the base. The orientation is left alone because
    it is a unit quaternion the teacher always sent the same; constraining it
    here would be inventing a limit the action term does not impose.
    """
    import numpy as np

    reach = 0.45  # metres; the arm's measured reach is 0.33 and 0.45 leaves slack
    clipped = np.array(command, dtype=np.float32).copy()
    clipped[0] = float(np.clip(clipped[0], -reach, reach))
    clipped[1] = float(np.clip(clipped[1], -reach, reach))
    clipped[2] = float(np.clip(clipped[2], -0.6, reach))
    return clipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", required=True, help="checkpoint directory or .pt file")
    parser.add_argument("--type", choices=["act", "bc"], default="act")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"simulator steps per episode (default {DEFAULT_STEPS})")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window; omit it to watch the policy in the "
                             "Isaac Sim viewport")
    parser.add_argument(
        "--speed", type=float, default=0.0,
        help="slow the run down so it can be watched. 0 = as fast as the machine "
             "allows, 1 = real time (the task runs 60 physics steps per second), "
             "0.25 = quarter speed. Ignored when --headless is set.",
    )
    parser.add_argument(
        "--viewer-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="where to put the window camera; defaults to a measured position that "
             "frames the workspace",
    )
    parser.add_argument(
        "--viewer-lookat", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="what the window camera looks at",
    )
    parser.add_argument("--device", default="cuda:0", help="device for the simulator")
    parser.add_argument("--inference-device", default="cpu",
                        help="device for the policy (default cpu: sharing the GPU with "
                             "Isaac Sim makes PhysX crash with an illegal memory access "
                             "on this machine)")
    parser.add_argument("--action-steps", type=int, default=None,
                        help="override how many of the predicted chunk to run before "
                             "asking the policy again (default: the checkpoint's own value)")
    parser.add_argument("--trace", action="store_true",
                        help="print the full command the network produces every "
                             "--trace-every steps, all 8 dimensions, so a value "
                             "running away can be seen rather than inferred")
    parser.add_argument("--trace-every", type=int, default=20,
                        help="steps between --trace lines (default 20)")
    args = parser.parse_args()

    import numpy as np
    import torch

    import expert as expert_module
    import il_dataset as data
    import il_env

    from il_env import prepare_runtime

    policy_path = Path(args.policy).resolve()
    if not policy_path.exists():
        print(f"ERROR: no checkpoint at {policy_path}")
        return 1

    # The window camera. Built here rather than inside ``build_env`` so the
    # defaults and the two overrides are decided in one visible place.
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
    print(f"loading {args.type} policy from {policy_path}")
    if args.type == "act":
        policy, preprocessor, postprocessor = load_act_policy(
            policy_path, inference_device, action_steps=args.action_steps
        )
        action_mean = action_std = None
    else:
        policy, action_mean, action_std = load_bc_policy(policy_path, inference_device)
        preprocessor = postprocessor = None

    prepare_runtime(headless=args.headless)

    # The same environment the teacher ran in, so that a predicted action is
    # applied exactly as the teacher applied it. This is the reason the dataset's
    # action is the IK command rather than a joint position: the two spaces are
    # not interchangeable, and translating between them silently reintroduces the
    # lag and the unit mixes that make a policy stand still.
    env = il_env.build_env(
        kind="ik_teacher",
        record=False,
        device=args.device,
        viewer=viewer,
    )
    il_env.assert_joint_order(env)

    if not args.headless:
        print(f"viewer: eye={viewer['eye']} lookat={viewer['lookat']}")
        if args.speed:
            print(f"running at {args.speed}x real time; press Ctrl+C to stop")
        else:
            print("running at full speed; use --speed 0.5 to watch it")

    # The one setting worth warning about. ACT runs ``n_action_steps`` of the
    # ``chunk_size`` actions it predicts before it looks at the camera again. When
    # the two are equal nothing is ever re-observed: a 640-step episode becomes 32
    # consecutive open-loop commitments, and one bad chunk cannot be recovered
    # from. The checkpoint trained by demo 15 ships with both at 20.
    if args.type == "act":
        n_steps = args.action_steps if args.action_steps is not None else policy.config.n_action_steps
        n_chunk = policy.config.chunk_size
        if n_steps >= n_chunk:
            print(
                f"WARNING: running open-loop - the policy executes {n_steps} of the "
                f"{n_chunk} actions it predicts before observing again, so an episode "
                f"is {args.steps // max(1, n_steps)} chained open-loop chunks with no "
                f"chance to correct. Pass --action-steps 4 to close the loop."
            )

    # The demonstration's pick-and-put criterion, and the drop offset its plan used.
    tolerance = expert_module.PLACEMENT_TOLERANCE_M or PLACEMENT_TOLERANCE_M
    drop_x, drop_y = expert_module.DROP_OFFSET_XY

    # Seconds of simulated time per physics step, read from the task rather than
    # assumed: the task runs ``dt = 1/60`` with ``decimation = 1``. ``--speed`` is
    # a multiple of real time, so 1.0 makes an episode last as long as it would
    # in reality and 0.25 shows it in slow motion.
    speed = args.speed if not args.headless else 0.0
    step_seconds = float(env.cfg.sim.dt) * float(env.cfg.decimation)
    pace_seconds = step_seconds / speed if speed > 0 else 0.0
    if pace_seconds:
        print(
            f"pacing: {1 / pace_seconds:.1f} steps per second "
            f"(the task advances {1 / step_seconds:.0f} steps per second)"
        )

    results: list[dict] = []
    bounds = phase_bounds()
    started = time.time()

    try:
        for episode in range(args.episodes):
            env.reset()
            il_env.reset_arm(env)
            spot = il_env.sample_cube_spot(episode, args.episodes)
            yaw = il_env.sample_cube_yaw(episode, args.episodes)
            place_cube(env, spot, yaw)
            target = (spot[0] + drop_x, spot[1] + drop_y)

            episode_start = time.time()
            max_height = 0.0
            steps_taken = 0
            pace_deadline = time.perf_counter()
            # Every episode is scored on where the cube ended up, how high it
            # went, and how many steps it took. The episode runs out of steps
            # rather than ending on the task's own success term, because the
            # demonstrations put the cube back down again and the interesting
            # question is where a policy that keeps moving ends up.
            with torch.no_grad():
                for steps_taken in range(1, args.steps + 1):
                    # ``observation_manager.compute()`` is the current
                    # observation, and after ``reset`` it is the reset itself.
                    # It is asked for explicitly rather than kept from the
                    # previous ``step`` return value, which drops the image tensors
                    # from the tuple when a recorder is disabled.
                    observation = env.unwrapped.observation_manager.compute()
                    state, image = data.observation_from_env(observation)
                    state_t = torch.as_tensor(
                        state, dtype=torch.float32, device=inference_device
                    ).unsqueeze(0)
                    # Channel-first, the order the saved pre-processor was built
                    # with. Getting this wrong is not caught by a shape check
                    # inside the policy, only by a dtype error in the normaliser.
                    image_t = torch.as_tensor(
                        data.image_to_policy_input(image), device=inference_device
                    ).unsqueeze(0)

                    if args.type == "act":
                        action = policy.select_action(
                            preprocessor({data.STATE_KEY: state_t, data.IMAGE_KEY: image_t})
                        )
                        command = postprocessor(action).squeeze(0).cpu().numpy()
                    else:
                        predicted = policy(image_t, state_t) * action_std + action_mean
                        command = predicted.squeeze(0).cpu().numpy()

                    # The action space of this environment is the teacher's, so
                    # the command goes to the simulator as it is. Clipping only
                    # the position components, and with the joint limits read
                    # from the articulation, keeps the arm inside its range
                    # without inventing numbers for the orientation ones.
                    clipped = _clip_ik_command(env, command)
                    env_action = torch.as_tensor(clipped, dtype=torch.float32, device=env.device)
                    if env_action.dim() == 1:
                        env_action = env_action.unsqueeze(0)
                    il_env.step_obs(env, env_action)

                    cube = env.scene["cube"].data.root_pos_w[0]
                    max_height = max(max_height, float(cube[2]))

                    if args.trace and steps_taken % args.trace_every == 0:
                        # Printed before clipping, so a dimension that the clip is
                        # quietly saving is still visible. Four of the eight are
                        # constant in the dataset (the orientation quaternion and
                        # a second gripper channel), which means their unnormaliser
                        # divides by ~0 and multiplies any stray output by 1e8.
                        raw = np.asarray(command).reshape(-1)
                        print(
                            f"    step {steps_taken:4d} "
                            f"[{phase_at(bounds, steps_taken - 1)}] "
                            f"xyz=({raw[0]:+.3f},{raw[1]:+.3f},{raw[2]:+.3f}) "
                            f"quat=({raw[3]:+.2e},{raw[4]:+.2e},{raw[5]:+.2e}) "
                            f"g6={raw[6]:+.3f} g7={raw[7]:+.2f} "
                            f"cube=({float(cube[0]):.3f},{float(cube[1]):.3f},"
                            f"{float(cube[2]):.3f})",
                            flush=True,
                        )

                    if pace_seconds:
                        # Hold the episode to the requested speed. Sleeping the
                        # remainder after the step rather than sleeping a fixed
                        # amount first means a slow frame makes the run lag
                        # behind rather than stuttering through the motion.
                        overrun = time.perf_counter() - pace_deadline
                        if overrun < pace_seconds:
                            time.sleep(pace_seconds - overrun)
                        pace_deadline = time.perf_counter() + pace_seconds
                    elif steps_taken % 50 == 0:
                        # Only when not paced: at full speed the per-step progress
                        # would bury the episode table, and when it is paced the
                        # motion is slow enough to read off the window itself.
                        print(
                            f"    step {steps_taken}/{args.steps} "
                            f"cube=({float(cube[0]):.3f}, {float(cube[1]):.3f}, "
                            f"{float(cube[2]):.3f}) target=({target[0]:.3f}, {target[1]:.3f})",
                            flush=True,
                        )

            cube_end = env.scene["cube"].data.root_pos_w[0]
            error = float(
                ((cube_end[0] - target[0]) ** 2 + (cube_end[1] - target[1]) ** 2) ** 0.5
            )
            ok = error < tolerance
            results.append(
                {
                    "episode": episode,
                    "success": bool(ok),
                    "final_error_m": round(error, 4),
                    "max_cube_height_m": round(max_height, 4),
                    "steps": steps_taken,
                    "wall_clock_s": round(time.time() - episode_start, 1),
                }
            )
            print(
                f"  episode {episode}: {'ok' if ok else 'FAILED'} "
                f"final_error={error * 100:.1f}cm max_cube_height={max_height * 100:.1f}cm "
                f"steps={steps_taken} ({time.time() - episode_start:.0f}s)"
            )

    finally:
        env.close()

    successes = sum(1 for record in results if record["success"])
    write_results(Path(__file__).resolve().parent / "results", args, policy_path, results)
    print("")
    print(f"success: {successes}/{len(results)} "
          f"({100 * successes / max(1, len(results)):.0f}%)")
    print(f"report:  results/16_evaluation.md")
    print(f"total time: {time.time() - started:.0f}s")
    return 0


def write_results(directory: Path, args, policy_path: Path, results) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    successes = sum(1 for record in results if record["success"])
    lines = [
        f"Fecha: {time.strftime('%Y-%m-%d')}. Rama: prepare-demos. Entorno Conda "
        "hackathon-robotekin, Windows, Isaac Sim 5.1, RTX 5060 Ti.",
        "",
        f"Politica evaluada: `{policy_path}` (tipo `{args.type}`).",
        f"Episodios: {len(results)}. Exito: {successes}/{len(results)} "
        f"({100 * successes / max(1, len(results)):.0f}%).",
        "",
        "Un episodio cuenta como exito si el cubo acaba a menos de "
        f"{PLACEMENT_TOLERANCE_M * 100:.0f} cm del punto donde las demostraciones lo "
        "dejaban. La altura maxima se anota al lado porque un cubo lanzado lejos ya "
        "ha fallado, aunque acabe en el sitio.",
        "",
        "| episodio | exito | error final (cm) | altura maxima del cubo (cm) | pasos |",
        "|---|---|---|---|---|",
    ]
    for record in results:
        lines.append(
            f"| {record['episode']} | {'si' if record['success'] else 'no'} "
            f"| {record['final_error_m'] * 100:.1f} | {record['max_cube_height_m'] * 100:.1f} "
            f"| {record['steps']} |"
        )
    lines.append("")
    (directory / "16_evaluation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
