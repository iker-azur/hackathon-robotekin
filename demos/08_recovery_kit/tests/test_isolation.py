"""El agente no usa información privilegiada del simulador.

Los módulos que deciden (percepción, mundo, plan, supervisor, habilidades,
agente) no pueden importar ``audit`` ni leer poses verdaderas de las piezas
(``truth``, ``root_pos_w``, ``viz_poses``…). Sólo ven la cámara y los servos.
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "recoverykit"
AGENT_SIDE = ["agent.py", "perception.py", "world_model.py", "planner.py", "supervisor.py", "skills.py",
              "control.py", "kit.py", "kinematics.py"]
FORBIDDEN = [r"\baudit\b", r"\.truth\(", r"root_pos_w", r"root_quat_w", r"root_lin_vel_w", r"viz_poses",
             r"\.pieces\[", r"\bperturb\b"]


def test_agent_side_has_no_privileged_access():
    offenders = []
    for name in AGENT_SIDE:
        src = (PKG / name).read_text(encoding="utf-8")
        code = re.sub(r'"""[\s\S]*?"""', "", src)   # los docstrings pueden nombrarlo para explicarlo
        code = "\n".join(line.split("#")[0] for line in code.splitlines())
        for pat in FORBIDDEN:
            if re.search(pat, code):
                offenders.append(f"{name}: {pat}")
    assert offenders == [], offenders
