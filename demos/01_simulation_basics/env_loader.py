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
import hashlib
import os
import re
import sys
import tomllib
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


# Extensions Kit crashes on while loading, and which this project never uses.
#
# Measured on this machine, every Kit start-up - successful or not - reports
#
#   [Error] [omni.ext.plugin] Could not load ... generic_mo_io.dll
#   ImportError: DLL load failed while importing _generic_model_output
#   [Error] [carb] [Plugin: omni.sensors.nv.lidar.plugin.dll] ... a dependent
#                  library failed to load
#   [Error] [carb] [Plugin: omni.sensors.nv.radar.plugin.dll] ... a dependent
#                  library failed to load
#   Windows fatal exception: code 0xc0000139        (STATUS_ENTRYPOINT_NOT_FOUND)
#   Windows fatal exception: access violation
#
# with two Python threads caught inside ``omni.graph.tools``'s own extension
# scanner, i.e. on a thread with no frame of this repository's. Those extensions
# are lidar, radar, ultrasonic and the RTX generic-model-output sensor: none of
# them is a dependency of a pick-and-place on a table, and every one of them tries
# to resolve a native DLL that is not on this machine's loader path.
#
# ``0xc0000139`` is ``STATUS_ENTRYPOINT_NOT_FOUND``: a DLL that loaded but does
# not export a symbol another DLL expects. It is the signature of a *partially
# satisfiable* native dependency - a dependent library that LoadLibrary accepts
# and GetProcAddress then fails on - and it cannot be caught or handled, because
# it happens on Kit's loader thread, before any Python frame exists.
#
# Excluding them is the fix, and the reliable lever is Kit's indexed
# ``--/app/extensions/excluded/<index>=<extension>`` command-line setting, injected
# through ``AppLauncher(kit_args=...)`` (isaaclab/app/app_launcher.py appends it to
# ``sys.argv``, which is what ``SimulationApp`` parses). A previous scalar `disabled`
# setting looked plausible but did not change Kit's excluded-extension array.
DISABLED_EXTENSIONS = (
    "isaacsim.sensors.rtx",          # generic_model_output / _generic_model_output
    "omni.sensors.nv.lidar",        # lidar plugin DLL
    "omni.sensors.nv.radar",        # radar plugin DLL
    "omni.sensors.nv.ultrasonic",   # same sensor family, same loader path
)


def disable_unused_sensor_extensions() -> str:
    """Return Kit arguments that stop the sensor extensions above from loading.

    Deliberately a function rather than a module-level constant so the string is
    built once, at the moment it is used, and can be logged by whoever calls it.
    """
    # Kit expects this setting to be an array under `extensions/excluded`. The former
    # `/app/extensions/disabled=foo,bar` argument was accepted as an untyped string and did
    # not exclude anything; the RTX, lidar, and radar extensions still started and attempted
    # to load native DLLs unavailable in this Windows environment.
    exclusions = " ".join(
        f"--/app/extensions/excluded/{index}={extension}"
        for index, extension in enumerate(DISABLED_EXTENSIONS)
    )
    return exclusions


def make_demo11_kit_experience(source_app: Path, cache_root: Path | None = None) -> Path:
    """Make a local copy of IsaacLab's Python app without its broken RTX sensor dependency.

    ``isaaclab.python.kit`` declares ``isaacsim.sensors.rtx`` as a direct app dependency.
    Kit's excluded list cannot remove an explicitly enabled root dependency, so this copy
    removes only that declaration while retaining the rest of the installed IsaacLab app.
    The camera extension used by Demo 11 remains enabled.
    """
    source_app = Path(source_app)
    source = source_app.read_text(encoding="utf-8")
    dependency = re.compile(r"(?m)^\s*\"isaacsim\.sensors\.rtx\"\s*=\s*\{\}\s*\r?\n")
    source, removed = dependency.subn("", source)
    if removed != 1:
        raise RuntimeError(
            f"Expected exactly one RTX dependency in IsaacLab app {source_app}; found {removed}"
        )

    # Isaac Sim 5.1's Windows RTX path can fail during Hydra startup on newer, non-validated
    # Vulkan driver branches. Demo 11 still uses RTX sensors; select the supported Direct3D
    # backend for this derived Windows app rather than changing the machine-wide driver.
    if sys.platform == "win32":
        source, changed = re.subn(
            r"(?m)^(\s*vulkan\s*=\s*)true(\s*(?:#.*)?)$",
            r"\1false\2", source, count=1,
        )
        if not changed:
            app_section = re.search(r"(?m)^\[settings\.app\]\s*$", source)
            if app_section:
                end = app_section.end()
                source = source[:end] + "\nvulkan = false" + source[end:]
            else:
                source += "\n[settings.app]\nvulkan = false\n"

    # Demo 11 uses an RGB camera in both GUI and headless runs. Deriving from the minimal
    # IsaacLab app bypasses the usual *.rendering.kit experience, so carry over its camera
    # prerequisites and the setting checked by IsaacLab's Camera implementation.
    for extension in (
        "omni.replicator.core", "omni.syntheticdata", "omni.kit.material.library",
        "omni.kit.viewport.rtx",
    ):
        if not re.search(rf'(?m)^\s*"{re.escape(extension)}"\s*=\s*\{{', source):
            source = re.sub(
                r"(?m)^\[dependencies\]\s*$",
                f'[dependencies]\n"{extension}" = {{}}', source, count=1,
            )
    if not re.search(r"(?m)^\[settings\.isaaclab\]\s*$", source):
        source += '\n[settings.isaaclab]\ncameras_enabled = true\n'
    elif not re.search(r"(?m)^cameras_enabled\s*=", source):
        source = re.sub(
            r"(?m)^\[settings\.isaaclab\]\s*$",
            '[settings.isaaclab]\ncameras_enabled = true', source, count=1,
        )

    # Kit normally resolves each Isaac Lab dependency to the newest registered version.
    # That can silently combine the checkout's older API with newer extensions from the
    # Isaac Sim registry (for example, LeIsaac imports the checkout's tasks but Python then
    # imports IsaacLab 4.5's `isaaclab.sensors`). Pin the five coordinated Lab extensions to
    # the versions declared by the adjacent source checkout when it is present.
    source_root = source_app.parent.parent / "source"
    if source_root.is_dir():
        # The generated experience can live in the demo's writable cache instead of
        # beside the shared IsaacLab checkout. Preserve the checkout's extension
        # discovery explicitly because `${app}/../source` would otherwise point into
        # that cache and Kit could fetch a mismatched registry release.
        source = source.replace('"${app}/../source"', f'"{source_root.as_posix()}"')
        for extension in (
            "isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_rl"
        ):
            manifest = source_root / extension / "config" / "extension.toml"
            if not manifest.is_file():
                continue
            with manifest.open("rb") as manifest_file:
                version = tomllib.load(manifest_file).get("package", {}).get("version")
            if not version:
                raise RuntimeError(f"IsaacLab extension {manifest} does not declare a package version")
            entry = re.compile(
                rf'(?m)^(\s*"{re.escape(extension)}"\s*=\s*\{{)([^}}]*)(\}}\s*)$'
            )

            def pin_version(match: re.Match[str], version: str = str(version)) -> str:
                options = match.group(2).strip()
                options = re.sub(r'\bversion\s*=\s*"[^"]*"\s*,?\s*', "", options)
                if options:
                    options = f' version = "{version}", {options}'
                else:
                    options = f' version = "{version}"'
                return f"{match.group(1)}{options}{match.group(3)}"

            source, _ = entry.subn(pin_version, source, count=1)

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    output_name = f"isaaclab.python.demo11-{digest}.kit"
    if cache_root is None:
        # Keep the derived app beside the installed IsaacLab experience. Kit uses that
        # directory to resolve the user's editable IsaacLab extension folders; putting the
        # app under %TEMP% makes Kit miss them and pull an incompatible public release.
        output = source_app.parent / output_name
    else:
        output = Path(cache_root) / f"demo11-isaaclab-app-{digest}" / "apps" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text(encoding="utf-8") != source:
        output.write_text(source, encoding="utf-8")
    return output


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
    import isaaclab
    from isaaclab.app import AppLauncher

    isaaclab_root = Path(isaaclab.__file__).resolve().parents[3]
    base_experience = isaaclab_root / "apps" / "isaaclab.python.kit"
    # Generate the temporary Kit experience under the project workspace. The base
    # IsaacLab checkout may be read-only (or shared with another process), and Demo 11
    # must not need write access outside its own workspace just to start Isaac Sim.
    project_root = Path(__file__).resolve().parents[2]
    experience_cache = (project_root / "demos" / "05_planner_and_executor"
                        / "out11" / "kit_cache")
    demo11_experience = make_demo11_kit_experience(base_experience, experience_cache)

    # The one change that stops Kit's loader thread from taking the process with it:
    # the sensor extensions whose native DLLs this machine cannot resolve are told
    # not to load. ``kit_args`` is the route in - AppLauncher appends it to sys.argv,
    # and sys.argv is what SimulationApp reads - so the setting lands in Kit's own
    # argument parser, where a carb setting at runtime is ignored.
    try:
        _ = AppLauncher({"enable_cameras": True,
                         "experience": str(demo11_experience),
                         "kit_args": disable_unused_sensor_extensions()})
    finally:
        # AppLauncher has consumed the experience file once SimulationApp is initialized.
        # Keep the shared IsaacLab checkout clean while retaining the active Kit session.
        demo11_experience.unlink(missing_ok=True)

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
