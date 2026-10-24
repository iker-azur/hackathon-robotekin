"""Install the repository SOFollower telemetry hook in the active virtualenv."""

from pathlib import Path
import site


REPOSITORY = Path(__file__).resolve().parents[1]
SITE_PACKAGES = Path(site.getsitepackages()[0])
HOOK = SITE_PACKAGES / "robotekin_hmi_telemetry.pth"
HOOK.write_text(
    f"{REPOSITORY}\nimport comandos_so101.hmi_autopublish; comandos_so101.hmi_autopublish.install()\n",
    encoding="utf-8",
)
print(f"Instalado hook HMI en {HOOK}")
