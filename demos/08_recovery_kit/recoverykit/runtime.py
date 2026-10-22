"""Arranque de Isaac Sim para RecoveryKit.

Kit tiene que arrancar antes de importar cualquier módulo de ``isaaclab`` que
toque ``omni`` o ``pxr``. Este módulo es lo único que el resto del demo llama
antes de construir la célula.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Runtime:
    app: object
    headless: bool

    def running(self) -> bool:
        return bool(self.app.is_running())

    def close(self, code: int = 0) -> None:
        """Termina el proceso.

        ``SimulationApp.close()`` se queda colgado minutos en este equipo cuando
        hay cámaras (Replicator sigue descartando fotogramas). Todo lo que hay
        que guardar ya está en disco cuando se llama, así que se vacían los
        flujos y se sale directamente.
        """
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headless", action="store_true", help="sin ventana de Isaac Sim")
    parser.add_argument("--device", default="cpu", help="dispositivo de física: cpu o cuda:0")
    parser.add_argument("--vulkan", action="store_true", help="usar Vulkan en Windows (por defecto Direct3D 12)")


def _experience(headless: bool, cameras: bool) -> str:
    """Escribe un .kit que hereda el de Isaac Lab y fija sus extensiones.

    La caché de extensiones de Kit tiene copias del registro (isaaclab 4.5.x)
    con versión mayor que el Isaac Lab instalado desde el código fuente; sin
    fijar la versión, Kit carga esas copias y ``import isaaclab`` deja de ser el
    instalado. Se fija cada extensión a la versión de su ``extension.toml``.
    """
    import importlib.util
    import re

    src = Path(importlib.util.find_spec("isaaclab").submodule_search_locations[0]).parents[1]
    apps = src.parent / "apps"
    base = "isaaclab.python" + (".headless" if headless else "") + (".rendering" if cameras else "")
    pins = []
    for ext in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_rl"):
        toml = src / ext / "config" / "extension.toml"
        if toml.exists():
            version = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8"), re.M).group(1)
            pins.append(f'"{ext}" = {{ version = "{version}", exact = true }}')
    text = "\n".join([
        "[package]",
        'title = "RecoveryKit"',
        'version = "1.0.0"',
        "",
        "[dependencies]",
        f'"{base}" = {{}}',
        *pins,
        "",
        # misma identidad que Isaac Sim: comparte su carpeta de datos y su caché
        # de extensiones; con otro nombre Kit crea rutas que superan MAX_PATH
        "[settings]",
        'app.versionFile = "${exe-path}/VERSION"',
        'app.folder = "${exe-path}/"',
        'app.name = "Isaac-Sim"',
        'app.version = "5.1.0"',
        "",
        "[settings.app.exts]",
        f'folders.\'++\' = ["{apps.as_posix()}", "{src.as_posix()}"]',
        "",
    ])
    out = Path(__file__).resolve().parents[1] / "out" / f"recoverykit.{base}.kit"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)


def boot(args: argparse.Namespace, cameras: bool = True) -> Runtime:
    """Arranca Kit con cámaras activadas y devuelve el runtime."""
    # con la salida redirigida Windows usa cp1252, que no tiene «≤» ni «°»
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from isaaclab.app import AppLauncher

    # En Windows, Vulkan revienta al activar el motor RTX con cámaras en este
    # equipo (RTX 5060 Ti); Direct3D 12 arranca bien.
    kit_args = "--/app/vulkan=false" if sys.platform == "win32" and not getattr(args, "vulkan", False) else ""
    launcher = AppLauncher(headless=args.headless, enable_cameras=cameras, device=args.device,
                           kit_args=kit_args, experience=_experience(args.headless, cameras))
    sys.stdout.flush()
    return Runtime(app=launcher.app, headless=args.headless)
