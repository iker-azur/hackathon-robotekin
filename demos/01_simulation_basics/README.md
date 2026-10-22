# 01 — Simulation basics

## Goal

Learn the smallest useful LeRobot simulation interface for an SO-101. These demos check imports, create the official LeRobot EnvHub SO-101 lift-cube task, inspect observations/actions, and apply a few deterministic commands.

The current task is `LeIsaac-SO101-LiftCube-v0`: it is a lift-cube benchmark. It is not yet a general scripted pick-and-place scene with a user-defined target area.

## What is tested

| Script | Expected result |
| --- | --- |
| `01_check_lerobot.py` | Prints Python/package versions and reports missing imports clearly. |
| `02_inspect_environment.py` | Loads the EnvHub task, resets it, and prints observation/action spaces. |
| `03_basic_actions.py` | Applies zero and small bounded actions, printing step results. |

## Installation

The repository does not manage environments or downloads. Run these commands manually inside your activated virtual environment.

The version matrix is fixed by Isaac Sim, and it is the main source of confusion:

| Package | Version | Why |
| --- | --- | --- |
| Python | 3.11 | `isaacsim` 5.1 declares `requires_python == 3.11.*` |
| `isaacsim[all,extscache]` | 5.1.0 | simulator runtime |
| `torch` / `torchvision` | 2.7.0 / 0.22.0 (`cu128`) | Blackwell (RTX 50-series) support |
| IsaacLab | v2.3.0, from source | the release that targets Isaac Sim 5.1 |
| `leisaac` | 0.4.0 | SO-101 tasks, from git |
| `numpy` | 1.26.0 | pinned exactly by `isaacsim-kernel`; IsaacLab also needs `<2` |
| `lerobot` | 0.4.4 | last release installable on Python 3.11 |

Create a Python 3.11 environment, then install Isaac Sim and PyTorch:

```powershell
conda create -n leisaac python=3.11
conda activate leisaac
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
python -m pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

`isaacsim` must be installed in the same environment that runs these demos; Isaac Sim 5.1 and LeIsaac are the simulation, LeRobot is only the dataset/policy layer.

### IsaacLab from source

IsaacLab is **not** fully installable from PyPI. The `isaaclab` wheel only ships the
`isaaclab` package; `isaaclab_tasks` and `isaaclab_assets` are not published, and
LeIsaac imports `isaaclab_tasks`. Clone the matching tag and use the helper script:

```powershell
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
.\isaaclab.bat --install none
```

Use `none` instead of the default `all` so that `robomimic` is skipped; it has no
Windows wheels. The helper reinstalls `torch`/`torchvision` as part of the run, which
removes `torchaudio`; put it back afterwards with:

```powershell
python -m pip install "torchaudio==2.7.0" --index-url https://download.pytorch.org/whl/cu128
```

### LeIsaac and LeRobot

```powershell
python -m pip install --no-deps "leisaac @ git+https://github.com/LightwheelAI/leisaac.git#subdirectory=source/leisaac"
python -m pip install "lerobot==0.4.4"
```

Install `leisaac` with `--no-deps`. Its `isaaclab` extra pulls `isaaclab` again from
PyPI and re-resolves `torch`, which undoes the pinned versions above.

### Why these demos do not use `lerobot.envs.make_env`

The official LeIsaac x EnvHub page loads tasks with:

```python
from lerobot.envs import make_env
```

That API cannot be used here. Isaac Sim 5.1 requires Python 3.11, and
`lerobot.envs.make_env` only exists from LeRobot 0.6.0 onwards, which requires
Python >= 3.12. On Python 3.11 the newest installable LeRobot is 0.4.4, and its
`lerobot.envs` package only exports `AlohaEnv`, `EnvConfig` and `PushtEnv`.

`env_loader.py` therefore does what the LeRobot helper does internally: download the
Hub file, import it, and call the `make_env` function it defines. The behaviour is the
same; only the wrapper differs.

## Running the demos

From the repository root, with the environment activated:

```powershell
python .\demos\01_simulation_basics\01_check_lerobot.py
python .\demos\01_simulation_basics\02_inspect_environment.py
python .\demos\01_simulation_basics\03_basic_actions.py
```

Add `--headless` to the last two to run without a renderer window. The first run of
`02` and `03` downloads about 0.9 GB of USD assets from the Hub and starts Kit, which
takes a few minutes. Both accept `--source` to target another Hub task, for example
`LightwheelAI/leisaac_env:envs/so101_pick_orange.py`.

The environment loader is equivalent to:

```python
from env_loader import load_environment

env = load_environment("LightwheelAI/leisaac_env:envs/so101_lift_cube.py")
env.reset()
env.close()
```

`env_loader.py` also handles two details that are easy to miss:

- it sets `OMNI_KIT_ACCEPT_EULA=YES`, otherwise the first launch blocks on a stdin
  prompt and fails with `Unable to bootstrap inner kit kernel`;
- it imports `h5py` **before** Kit starts. Isaac Sim ships its own `hdf5.dll`, which
  otherwise shadows the copy bundled with h5py and breaks
  `ImportError: DLL load failed while importing _errors` from
  `isaaclab.managers.recorder_manager`.

## Observations and actions

The demos deliberately print the real spaces instead of assuming a fixed schema. For
`LeIsaac-SO101-LiftCube-v0` with `n_envs=1` the installed task reports:

- `observation` is a dict with `policy` and `subtask_terms`;
- `policy` holds `joint_pos`, `joint_vel`, `joint_pos_rel`, `joint_vel_rel`,
  `actions`, `ee_frame_state` and `joint_pos_target`, each `(1, 6)` except
  `ee_frame_state` which is `(1, 7)`;
- `policy.front` is a `(1, 480, 640, 3)` `uint8` camera image;
- `subtask_terms.pick_cube` is a `(1,)` boolean success flag;
- `action_space` is `Box(-inf, inf, (1, 6), float32)`, so six joints, and the leading
  `1` is the `n_envs` batch dimension.

Two consequences for any new demo:

- actions must be **PyTorch tensors on the simulation device**, not NumPy arrays. Isaac
  Lab raises `AttributeError: 'numpy.ndarray' object has no attribute 'to'` otherwise;
- `reward`, `terminated` and `truncated` come back as tensors of shape `(n_envs,)`, so
  test them with `bool(terminated.any())` rather than `if terminated:`.

## Dataset/model downloads

No dataset or learned model is needed for these initial scripted checks. Do not download one yet.

Later, a useful dataset would be a small set of successful SO-101 demonstrations in the same task, saved as a LeRobot dataset under a user-chosen local directory such as:

```text
data/
└── so101_lift_cube_demo/
    ├── meta/
    ├── data/
    └── videos/
```

For this milestone there is no exact dataset download command to run. When data collection is the next goal, use the LeRobot/LeIsaac recorder workflow and document its generated dataset identifier here rather than guessing one.

## Common problems

- **`python` or `pip` is not found:** activate the virtual environment, or use the interpreter by its full path.
- **`No module named lerobot`:** install LeRobot in the active environment and rerun `01_check_lerobot.py`.
- **`No module named lerobot.envs`** or **`cannot import name 'make_env'`:** expected. See [Why these demos do not use `lerobot.envs.make_env`](#why-these-demos-do-not-use-lerobotenvsmake_env).
- **`No module named isaacsim`:** Isaac Sim is not installed in the environment running the script, or a different interpreter is being used.
- **`No module named isaaclab_tasks`:** IsaacLab was installed from PyPI only. `isaaclab_tasks` is not published, so clone IsaacLab and run `.\isaaclab.bat --install none`.
- **`Unable to bootstrap inner kit kernel: EOF when reading a line`:** the Omniverse EULA prompt blocked startup. Set `OMNI_KIT_ACCEPT_EULA=YES`; `env_loader.py` does this for you.
- **`ImportError: DLL load failed while importing _errors`** while loading `h5py` from `isaaclab.managers`: Isaac Sim's `hdf5.dll` shadowed h5py's. Import `h5py` before Kit starts; `env_loader.py` does this for you.
- **`AttributeError: 'numpy.ndarray' object has no attribute 'to'`:** the action was a NumPy array. Isaac Lab requires a torch tensor on the simulation device.
- **`ERROR: Could not find a version that satisfies the requirement isaacsim==5.1.0`:** the interpreter is not Python 3.11. Isaac Sim 5.1 requires `python==3.11.*`, and pip hides the incompatible wheels rather than reporting a version mismatch.
- **Asset or network errors on the first run:** `02` and `03` download about 0.9 GB of USD assets from `LightwheelAI/leisaac_env` before Kit starts. Check Hub access and retry.
- **Renderer/window problems:** add `--headless`. `isaacsim.sensors.rtx` may log a non-fatal `_generic_model_output` import warning on some Windows driver/GPU combinations; the environment still runs.
- **LeIsaac is only documented and tested on Linux.** If a task fails to build a scene on Windows after the environment imports correctly, move the same environment to WSL2 or Ubuntu rather than debugging the task config.

## Pick-and-place boundary

This iteration intentionally stops before a fake scripted pick-and-place implementation. The official EnvHub task available here is “lift cube”; its public example shows standard reset/step interaction, but not a simple generic API for placing the cube in an arbitrary target area. The next implementation should inspect the installed LeIsaac task source and state-machine examples, then add a deterministic baseline only if the actual task exposes the required cube pose, gripper control, and success condition.

## Next logical step

Run the three scripts successfully, record the printed observation keys and action bounds, and then make the next demo a tiny task-specific scripted controller for the actual installed LeIsaac lift-cube environment. Only after that should the project add demonstrations or learned policies.

## References

- [LeRobot installation](https://github.com/huggingface/lerobot/blob/main/docs/source/installation.mdx)
- [LeRobot EnvHub / LeIsaac](https://huggingface.co/docs/lerobot/main/envhub_leisaac)
- [LeIsaac installation](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)
