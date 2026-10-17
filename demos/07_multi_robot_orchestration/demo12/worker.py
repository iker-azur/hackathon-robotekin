"""Demo 12 — Isaac Sim process owning Kit."""

from __future__ import annotations

import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "01_simulation_basics"))
import env_loader

env_loader.prepare_runtime(headless=True)

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from demo12.config import default_scene
from demo12.isaac_env import IsaacCell
from demo12.orchestrator import MultiRobotOrchestrator


def main() -> int:
    scene = default_scene(2)
    cell = IsaacCell(scene, device="cuda:0", headless=True)
    cell.settle(10)
    orch = MultiRobotOrchestrator(cell, scene.mission.tasks)
    while not orch.all_tasks_completed():
        orch.step()
    print("ALL TASKS COMPLETED")
    os._exit(0)


if __name__ == "__main__":
    main()
