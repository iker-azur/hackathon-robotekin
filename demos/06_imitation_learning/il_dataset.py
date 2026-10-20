"""Read and write the LeRobot dataset these demos produce.

Everything that touches a dataset goes through this module, so there is exactly
one place where the units, the feature names and the (observation, action)
pairing are defined.

Feature layout
--------------

=========================  =============================================  ===============
key                        contents                                       shape
=========================  =============================================  ===============
``observation.state``      the 6 SO-101 joint positions, motor degrees      ``(6,)``
``observation.images.front``  the front camera, RGB                         ``(480, 640, 3)``
``observation.phase``      which phase of the plan this step belongs to      ``(1,)``
``observation.plan_clock`` where in the 640-step plan this step is, [-1, 1]  ``(1,)``
``observation.cube_uv``    the cube's pixel position at the start, [-1, 1]   ``(2,)``
``action``                 the teacher's 8-number IK command               ``(8,)``
``timestamp``              seconds since the episode start                ``(1,)``
=========================  =============================================  ===============

``plan_clock`` and ``cube_uv`` exist only in datasets recorded with
``features(with_perception=True)`` (demo 13 does). ``plan_clock`` is privileged;
``cube_uv`` comes from the colour detector, not from the simulator's state.

The state is in motor degrees, matching LeIsaac's own export
(``leisaac.utils.robot_utils``), so it is interchangeable with LeIsaac's
recordings and with the SO-101 datasets on the Hub. The **action is not** in
joint degrees: it is the command the teacher sent to the environment's inverse-
kinematics action term, with the first three numbers the target position of the
``gripper`` body in the robot base frame, the next four its orientation, and the
last one the open/close command.

Why there is a phase, stated plainly
------------------------------------

``observation.phase`` says which of the eight steps of the teacher's plan the
robot is in, and ``observation.plan_clock`` says how far through the whole plan it
is. A real robot is not told either, and that is the honest limitation of the
demo: **it is imitation learning with the plan's clock handed over, not imitation
learning from observation alone.**

Why they are here. The arm passes through the same configuration in different
phases of the plan - frames 299 and 300 of a demonstration are identical to 8e-6
degrees while the plan switches from ``lift`` to ``carry`` - so from joint angles
alone the expert's action is ambiguous, and ACT (one observation per prediction,
``configuration_act.py:149``) cannot be given a history. A clock settles that.

What it did *not* fix on its own, measured in ``results/diagnosis.md``: a camera +
joint-state policy given the phase still failed 0/5, because it learned to copy
the arm's own pose instead of reading the cube. The clock + cube policy is given
the clock and the cube's position and nothing else, so that shortcut is closed;
``results/`` says how well it does.

Why the action is the command and not the joints it produced
-----------------------------------------------------------

The two are not the same action, and the difference is not a detail:

* The teacher's command is a Cartesian pose; the joints are what the arm manages
  to do with it, a step or more later. Recording the outcome as the action
  teaches "repeat what I just did", not "send what I sent".
* The outcome barely moves: the arm advances less than a degree per step, so the
  action ends up nearly identical to the state. A policy trained on that learns
  the identity and stands still - which is exactly what the first version of this
  demo did, and why it was measured and changed.

The pairing
-----------

Frame ``i`` holds observation ``i`` and the action chosen at ``i``. The
observation is read *before* the step, so the policy learns "from what you see,
choose what to do next" rather than "explain what just happened".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import il_env as env_cfg  # noqa: E402

STATE_KEY = "observation.state"
IMAGE_KEY = "observation.images.front"
PHASE_KEY = "observation.phase"
# LeRobot ACT only embeds its reserved environment-state feature in addition to
# robot state and images. Demo 15 maps auxiliary observations to this key.
ACT_ENV_STATE_KEY = "observation.environment_state"
# Compatibility alias for older demo code that mapped phase to this reserved key.
ACT_PHASE_KEY = ACT_ENV_STATE_KEY
ACTION_KEY = "action"

# Which phase of the expert's plan a step belongs to, as an index into
# ``expert.PLAN``. It is recorded, fed to the policy, and **what it costs is worth
# saying plainly**: the policy is being told where it is in the plan, which is not
# something a real robot gets. It is here because the task cannot be fitted
# without it - see the module docstring below and ``results/diagnosis.md``.
PHASE_DIM = 1

# Two inputs added for the clock + cube policy (see ``results/diagnosis.md``):
#
# ``observation.plan_clock``  where in the 640-step plan this step is, scaled to
#                             [-1, 1]. PRIVILEGED in the same way the phase is: a
#                             real robot is not told what time it is in the plan.
#                             Unlike the phase it is continuous, so it also says
#                             *how far into* a phase the plan is, which is what the
#                             arm's own joint angles were being used for - and that
#                             is the shortcut that made the earlier policies drift.
# ``observation.cube_uv``     the cube's pixel centroid from ``cube_detect``, seen
#                             once at the start of the episode (averaged over a few
#                             renders) and repeated on every frame, scaled to
#                             [-1, 1]. Perception, not privileged.
PLAN_CLOCK_KEY = "observation.plan_clock"
CUBE_UV_KEY = "observation.cube_uv"

# A small file next to a dataset or checkpoint saying what its environment-state
# input holds. ACT gives that input no name - only a width - so without this the
# evaluator would have to guess.
INPUTS_FILE = "il_inputs.json"
ENV_KINDS = {
    "phase": ["phase"],
    "detector": ["cube_u_normalized", "cube_v_normalized", "visible"],
    "clock_cube": ["plan_clock", "cube_u_normalized", "cube_v_normalized"],
}
LEGACY_ENV_KIND_BY_WIDTH = {1: "phase", 3: "detector"}


def plan_steps() -> int:
    """Length of the teacher's plan in steps (640)."""
    return phase_bounds()[-1][1]


def plan_clock(step: int) -> float:
    """The plan clock for ``step``, in [-1, 1]; stays at 1 once the plan is over."""
    last = plan_steps() - 1
    return 2.0 * min(max(step, 0), last) / last - 1.0


def normalised_uv(detection, image_shape) -> tuple[float, float]:
    """A detection's pixel centroid scaled to [-1, 1] (``image_shape`` is (H, W, C))."""
    return (
        2.0 * detection.u / (image_shape[1] - 1) - 1.0,
        2.0 * detection.v / (image_shape[0] - 1) - 1.0,
    )


def resolve_env_state_kind(policy_path, width: int) -> str | None:
    """What a checkpoint's environment-state input holds, or ``None`` if unknowable.

    Looks for ``il_inputs.json`` in the checkpoint folder and up to three folders
    above it (``outputs/<run>/checkpoints/<step>/pretrained_model``), then falls back
    on the two widths the earlier checkpoints used: 1 is the phase, 3 the detector.
    """
    folder = Path(policy_path).resolve()
    if folder.is_file():
        folder = folder.parent
    for candidate in [folder, *list(folder.parents)[:3]]:
        sidecar = candidate / INPUTS_FILE
        if sidecar.is_file():
            kind = json.loads(sidecar.read_text(encoding="utf-8")).get("env_state")
            if kind in ENV_KINDS and len(ENV_KINDS[kind]) == width:
                return kind
            raise ValueError(
                f"{sidecar} says the environment state is {kind!r}, which does not match the "
                f"checkpoint's width {width}"
            )
    return LEGACY_ENV_KIND_BY_WIDTH.get(width)


# The demo's task description, written into every frame of the dataset. LeRobot
# uses it to build the language annotation of a policy, and it is the one string
# a human reads to know what the robot was being taught.
TASK_DESCRIPTION = "Pick the red cube up and put it down 10 cm to the side."

ROBOT_TYPE = "so101_follower"

# The front camera is 640x480 RGB, as the task config declares.
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Rate the simulator actually advances. The task runs its physics at 60 steps a
# second (``dt = 1/60``), ``decimation = 1``, so one recorded step is 1/60 s.
FPS = 60


def convert_to_motor_degrees(joint_radians):
    """Sim radians (6,) -> motor degrees (6,), the dataset's unit."""
    import numpy as np
    import torch

    if isinstance(joint_radians, torch.Tensor):
        joint_radians = joint_radians.detach().cpu().numpy()
    tensor = torch.as_tensor(np.asarray(joint_radians, dtype=np.float32).reshape(1, -1))
    from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

    return convert_leisaac_action_to_lerobot(tensor).astype(np.float32).reshape(-1)


def convert_to_sim_radians(motor_degrees):
    """Motor degrees (6,) -> sim radians (6,), the environment's unit."""
    import numpy as np
    import torch

    if isinstance(motor_degrees, torch.Tensor):
        motor_degrees = motor_degrees.detach().cpu().numpy()
    tensor = torch.as_tensor(np.asarray(motor_degrees, dtype=np.float32).reshape(1, -1))
    from leisaac.utils.robot_utils import convert_lerobot_action_to_leisaac

    return convert_lerobot_action_to_leisaac(tensor).astype(np.float32).reshape(-1)


def features(with_perception: bool = False) -> dict:
    """The feature dictionary ``LeRobotDataset.create`` expects.

    ``with_perception`` adds the plan clock and the cube's pixel position, which
    the clock + cube policy is trained on. Off by default so a dataset recorded
    before they existed keeps the layout it was written with.
    """
    layout = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (env_cfg.STATE_DIM,),
            "names": [f"{joint}.pos" for joint in env_cfg.JOINT_NAMES],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (env_cfg.ACTION_DIM,),
            "names": list(env_cfg.ACTION_NAMES),
        },
        PHASE_KEY: {
            "dtype": "float32",
            "shape": (PHASE_DIM,),
            "names": ["phase"],
        },
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": IMAGE_HEIGHT,
                "video.width": IMAGE_WIDTH,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(FPS),
                "video.channels": 3,
                "has_audio": False,
            },
        },
    }
    if with_perception:
        layout[PLAN_CLOCK_KEY] = {"dtype": "float32", "shape": (1,), "names": ["plan_clock"]}
        layout[CUBE_UV_KEY] = {
            "dtype": "float32", "shape": (2,), "names": ["cube_u_normalized", "cube_v_normalized"],
        }
    return layout


def observation_from_env(obs) -> tuple:
    """Extract (state, image) from an Isaac Lab observation.

    Returns NumPy in the dataset's own formats: state in motor degrees, image as
    ``(H, W, 3)`` uint8.
    """
    import numpy as np

    flat = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
    state = convert_to_motor_degrees(flat["joint_pos"][0])
    image = np.asarray(flat["front"][0].cpu().numpy(), dtype=np.uint8)
    return state, image


def phase_bounds() -> list[tuple[int, int, str]]:
    """The expert's plan as ``(first_step, past_last_step, name)`` rows.

    Defined here rather than in ``expert`` so the recorder and the evaluator read
    the same boundaries and cannot drift apart. ``expert`` is not imported at
    module level because it pulls in the simulator-facing environment module.
    """
    import expert as expert_module

    rows, start = [], 0
    for phase in expert_module.PLAN:
        rows.append((start, start + phase.steps, phase.name))
        start += phase.steps
    return rows


def phase_at(step: int, bounds=None) -> int:
    """Index of the plan phase that ``step`` falls in.

    Returns ``len(bounds)`` past the end of the plan rather than raising, because
    the evaluator is allowed to run past the demonstrations' length and wants a
    sensible number to feed the policy instead of an exception.
    """
    rows = bounds if bounds is not None else phase_bounds()
    for index, (first, last, _) in enumerate(rows):
        if first <= step < last:
            return index
    return len(rows)


def phase_vector(step: int, bounds=None) -> "np.ndarray":
    """``phase_at`` as the ``(1,)`` float32 array the dataset stores."""
    import numpy as np

    return np.array([float(phase_at(step, bounds))], dtype=np.float32)


def image_to_policy_input(image_hwc):
    """(H, W, C) uint8 from the simulator -> (C, H, W) float32 in [0, 1].

    This is the form a LeRobot policy is fed, and both parts matter:

    * channel-first, because that is the order the saved pre-processor was built
      with;
    * float in zero to one, because the image statistics in the dataset (mean
      0.85, std 0.23) were computed on that scale. Handing over uint8 makes the
      normaliser adopt the image's dtype for its own statistics, and the frame
      count in those statistics then overflows an 8-bit integer.
    """
    import numpy as np

    array = np.asarray(image_hwc)
    return np.ascontiguousarray(array.transpose(2, 0, 1)).astype(np.float32) / 255.0


def action_from_teacher(teacher_action) -> "np.ndarray":
    """The teacher's action, as a numpy array in the simulator's own units.

    No conversion, deliberately. The action here is the command the environment
    was given - the 8-number IK target - so it is already exactly what the policy
    has to reproduce. Passing it through the joint-to-motor conversion instead
    treats a command for one space as if it belonged to another, which is how the
    first version of this script produced actions of 135 degrees on a joint whose
    limit is 95.
    """
    import numpy as np

    if isinstance(teacher_action, np.ndarray):
        return teacher_action.astype(np.float32).reshape(-1)
    return teacher_action.cpu().numpy().astype(np.float32).reshape(-1)


def action_from_joint_pos(joint_pos) -> "np.ndarray":
    """The next joint position, converted to the dataset's unit.

    Kept for the case where a dataset is wanted in joint space: it maps the
    articulation's positions onto the motor degrees of a real SO-101. Note that
    the resulting action is close to the state - the arm moves a fraction of a
    degree per step - which is what makes a policy trained on it converge to a
    self-attractor instead of moving. See the README for the details.
    """
    import numpy as np

    if isinstance(joint_pos, np.ndarray):
        raw = joint_pos
    else:
        raw = joint_pos.cpu().numpy()
    return convert_to_motor_degrees(raw[0] if raw.ndim == 2 else raw).astype(np.float32)


def default_dataset_dir(name: str = "so101_lift_cube_physical") -> Path:
    """Where datasets live. Under the demo folder, and outside git."""
    return Path(__file__).resolve().parent / "data" / name


def create_dataset(repo_id: str, root: Path, with_perception: bool = False):
    """Create an empty LeRobot dataset on disk, ready for ``add_frame``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features(with_perception=with_perception),
        root=str(root),
        robot_type=ROBOT_TYPE,
        use_videos=True,
    )
    return dataset


def load_dataset(repo_id: str, root: Path):
    """Load an existing LeRobot dataset, local or on the Hub."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=str(root))


def write_episode_log(path: Path, records: list[dict]) -> None:
    """Write one JSON line per episode, so the dataset can be audited without Isaac.

    A dataset directory only says how many frames it holds; it does not say which
    episodes the teacher actually succeeded on. This file does.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_episode_log(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
