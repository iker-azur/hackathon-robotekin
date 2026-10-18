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
``action``                 the teacher's 8-number IK command               ``(8,)``
``timestamp``              seconds since the episode start                ``(1,)``
=========================  =============================================  ===============

The state is in motor degrees, matching LeIsaac's own export
(``leisaac.utils.robot_utils``), so it is interchangeable with LeIsaac's
recordings and with the SO-101 datasets on the Hub. The **action is not** in
joint degrees: it is the command the teacher sent to the environment's inverse-
kinematics action term, with the first three numbers the target position of the
``gripper`` body in the robot base frame, the next four its orientation, and the
last one the open/close command.

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
ACTION_KEY = "action"

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


def features() -> dict:
    """The feature dictionary ``LeRobotDataset.create`` expects."""
    return {
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


def default_dataset_dir(name: str = "so101_lift_cube_demos") -> Path:
    """Where datasets live. Under the demo folder, and outside git."""
    return Path(__file__).resolve().parent / "data" / name


def create_dataset(repo_id: str, root: Path):
    """Create an empty LeRobot dataset on disk, ready for ``add_frame``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features(),
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
