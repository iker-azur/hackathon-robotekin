"""Demo 12 -- the Isaac Sim process. Owns Kit, runs the loop, prints what happened.

This is the headless end-to-end entry point: build the cell, let it settle, hand it to the
orchestrator, and step until every task is done or the run times out. It is also the
honest test of the whole demo -- if this does not finish, the demo does not work, whatever
the interface says.

Run with the Isaac interpreter, not the PATH python::

    set PYTHONPATH=demos\\07_multi_robot_orchestration
    C:\\Users\\Usuario\\miniconda3\\envs\\hackathon-robotekin\\python.exe -m demo12.worker --robots 2

Exit code 0 means every task completed. Anything else is a failure, and the JSON on stdout
says which task, in which state, and why.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_DEMO_ROOT = _HERE.parents[2]        # demos/
_REPO_ROOT = _DEMO_ROOT.parent
_PARENT = _HERE.parent               # 07_multi_robot_orchestration/

# Kit has to be up before anything imports isaaclab. env_loader is the only bootstrap that
# works on this machine; see its docstring for the sensor extensions it has to exclude.
for _p in (str(_PARENT), str(_DEMO_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo 12 multi-robot run")
    parser.add_argument("--robots", type=int, default=2,
                        help="how many SO-101 followers to build (1-3)")
    parser.add_argument("--max-seconds", type=float, default=180.0,
                        help="give up after this much simulated time")
    parser.add_argument("--max-wall-seconds", type=float, default=900.0,
                        help="give up after this much real time, whatever the sim clock "
                             "says -- a task that never terminates would otherwise hang here")
    parser.add_argument("--report-every", type=int, default=250,
                        help="print a progress line every N control ticks")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="run without a Kit window (the default)")
    parser.add_argument("--gui", dest="headless", action="store_false",
                        help="show Isaac Sim, so the cell can be watched")
    parser.add_argument("--all-tasks", action="store_true",
                        help="give every part a task, not only the colour-matched ones. A "
                             "denser mission and a harder one: two arms end up working the "
                             "middle of the cell at the same time, which is the contention "
                             "worth watching and also the part most likely to collide.")
    args = parser.parse_args()

    env_loader.prepare_runtime(headless=args.headless)

    import json

    from demo12.config import default_scene
    from demo12.isaac_env import ARM_BODIES_WITH_COLLISION, IsaacCell
    from demo12.orchestrator import MultiRobotOrchestrator

    scene = default_scene(args.robots, assign_all=args.all_tasks)
    cell = IsaacCell(scene, device="cuda:0", headless=args.headless)
    cell.settle(90)

    orch = MultiRobotOrchestrator(cell, scene, scene.mission.tasks)

    print(f"Demo 12: {len(scene.robots)} robot(s), "
          f"{len(scene.parts)} parts, {len(scene.bins)} bins, "
          f"{len(orch.tasks)} tasks", flush=True)
    print(f"  arm collision kept on: {len(ARM_BODIES_WITH_COLLISION)} body name(s)",
          flush=True)

    last_report = 0
    deadline = time.monotonic() + args.max_wall_seconds
    timed_out = None

    while orch.work_outstanding():
        orch.step()

        if orch.control_ticks - last_report >= args.report_every:
            last_report = orch.control_ticks
            active = ", ".join(
                f"{t.id}:{t.state}" +
                (f"/{t.runner.phase}" if t.runner else "")
                for t in orch.tasks.values())
            print(f"  t={orch.sim_time_s:6.1f}s  ticks={orch.control_ticks:6d}  {active}",
                  flush=True)

        if orch.sim_time_s > args.max_seconds:
            timed_out = f"simulated time limit ({args.max_seconds:.0f} s) reached"
            break
        if time.monotonic() > deadline:
            # A wall-clock limit as well as a simulated one. Simulated time advances only
            # when the orchestrator ticks, so a task that never terminates would otherwise
            # sit here until someone kills the process.
            timed_out = (f"wall-clock limit ({args.max_wall_seconds:.0f} s) reached "
                         f"after {orch.control_ticks} ticks")
            break

    if timed_out:
        print(f"  TIMED OUT: {timed_out}", flush=True)
    elif not orch.all_tasks_completed():
        failed = orch.failed_tasks
        print(f"  finished, but not successfully: {len(failed)} task(s) failed: "
              f"{', '.join(failed) if failed else 'none'}", flush=True)

    report = {
        "allTasksCompleted": orch.all_tasks_completed(),
        "snapshot": orch.snapshot(),
        "cell": cell.snapshot(),
    }
    print(json.dumps(report, indent=2), flush=True)

    if not orch.all_tasks_completed():
        print("FAILED: not every task completed", flush=True)
    else:
        print(f"OK: {len(orch.tasks)} task(s) completed in "
              f"{orch.sim_time_s:.1f} s simulated, "
              f"{orch.control_ticks} control ticks", flush=True)

    # os._exit rather than a normal return: Kit's shutdown path on this machine can hang or
    # fault in a loader thread, and a demo that has already reported its result should not
    # be held up by its own teardown.
    os._exit(0 if orch.all_tasks_completed() else 1)


if __name__ == "__main__":
    main()