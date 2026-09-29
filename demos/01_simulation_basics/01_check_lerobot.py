"""Check that the Python environment can import the packages used by LeRobot demos."""

from __future__ import annotations

import importlib
from importlib import metadata


REQUIRED_PACKAGES = ("lerobot", "gymnasium", "numpy")
OPTIONAL_PACKAGES = ("torch", "isaaclab", "leisaac")


def package_version(package_name: str) -> str:
    """Return an installed distribution version, or a useful placeholder."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    """Import required packages and report their versions."""
    print("SO-101 Autonomous Agent - LeRobot sanity check")
    print(f"Python: {__import__('sys').version.split()[0]}")

    missing: list[str] = []
    for package_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package_name)
        except ImportError as error:
            missing.append(package_name)
            print(f"[MISSING] {package_name}: {error}")
        else:
            print(
                f"[OK] {package_name}: distribution {package_version(package_name)} "
                f"({getattr(module, '__file__', 'no __file__')})"
            )

    for package_name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(package_name)
        except ImportError:
            print(f"[optional] {package_name}: not installed")
        else:
            print(f"[optional] {package_name}: {package_version(package_name)}")

    if missing:
        print("\nLeRobot is not ready for the next demos.")
        print("Activate your virtual environment and follow README.md installation steps.")
        print("Missing required packages: " + ", ".join(missing))
        return 1

    print("\nImports look good. The SO-101 simulation itself is checked by 02_inspect_environment.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
