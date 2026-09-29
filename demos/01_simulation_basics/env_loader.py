"""Load a LeIsaac SO-101 task from the Hugging Face EnvHub repository.

The published LeRobot example uses ``lerobot.envs.make_env``. That helper does not
exist in the LeRobot releases that are installable next to Isaac Sim 5.1, because
Isaac Sim 5.1 requires Python 3.11 while ``lerobot.envs.make_env`` only ships from
LeRobot 0.6.0, which requires Python >= 3.12. This module therefore calls the
``make_env`` function defined by the Hub file directly, which is exactly what the
LeRobot helper does after downloading it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_ENV_SOURCE = "LightwheelAI/leisaac_env:envs/so101_lift_cube.py"
HUB_REPOSITORY = "LightwheelAI/leisaac_env"


def parse_source(source: str) -> tuple[str, str]:
    """Split a ``repository:path`` EnvHub reference into its two parts."""
    repository, separator, path = source.partition(":")
    if not separator or not path:
        raise ValueError(
            f"Expected an EnvHub source like 'owner/repo:envs/task.py', got {source!r}"
        )
    return repository, path


def accept_omniverse_eula() -> None:
    """Accept the NVIDIA Omniverse EULA non-interactively.

    Without this, the first Kit launch blocks on a stdin prompt and the Isaac Sim
    bootstrap fails with 'Unable to bootstrap inner kit kernel'.
    """
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def _import_hub_module(local_file: str) -> Any:
    """Import a downloaded Hub file as a standalone module."""
    spec = importlib.util.spec_from_file_location("leisaac_hub_env", local_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load a Python module from {local_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_runtime(headless: bool = False) -> Path:
    """Download the Hub assets, accept the EULA, and start Kit.

    Returns the assets root. After this returns, Isaac Lab and LeIsaac modules can
    be imported, which is what lets a caller build its own environment config
    instead of using the one the Hub file hardcodes.
    """
    accept_omniverse_eula()

    if headless or os.environ.get("HEADLESS") == "1":
        # The Hub file builds AppLauncher with a fixed dict, so --headless has to be
        # passed through sys.argv, which AppLauncher also parses.
        if "--headless" not in sys.argv:
            sys.argv.append("--headless")

    # Isaac Sim ships its own hdf5.dll, which shadows the copy bundled with h5py and
    # breaks `import h5py` inside isaaclab.managers. Load h5py while its own DLLs are
    # still the ones being resolved, i.e. before Kit boots.
    import h5py  # noqa: F401

    from huggingface_hub import snapshot_download

    assets_root = Path(snapshot_download(repo_id=HUB_REPOSITORY)) / "assets"
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    # The Hub env file only calls AppLauncher inside its make_env(), so importing
    # it does not start Kit. Start it here instead, otherwise `import omni` fails
    # for every isaaclab module that reaches for it at import time.
    from isaaclab.app import AppLauncher

    _ = AppLauncher({"enable_cameras": True})

    return assets_root


def load_environment(source: str = DEFAULT_ENV_SOURCE, headless: bool = False):
    """Create the SO-101 simulation environment described by an EnvHub source.

    Returns the wrapped Gymnasium environment. Use ``.unwrapped`` to reach the
    underlying Isaac Lab environment.
    """
    accept_omniverse_eula()

    if headless or os.environ.get("HEADLESS") == "1":
        if "--headless" not in sys.argv:
            sys.argv.append("--headless")

    import h5py  # noqa: F401

    from huggingface_hub import hf_hub_download, snapshot_download

    repository, path = parse_source(source)
    assets_root = Path(snapshot_download(repo_id=repository)) / "assets"
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    local_file = hf_hub_download(repo_id=repository, filename=path)
    module = _import_hub_module(local_file)

    # Importing the module calls AppLauncher, which starts Kit. Isaac Lab modules
    # such as isaaclab_tasks must only be imported after that happens.
    make_env = getattr(module, "make_env", None)
    if make_env is None:
        raise RuntimeError(f"{local_file} does not define a make_env() function")

    return make_env(n_envs=1, use_async_envs=False)
