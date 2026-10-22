"""Does the arm's motion step pay for a render it does not use?

The P0 performance report left one measurement undone: attribute the 2x real-time factor
(a physics step costs ~33 ms against a 1/60 s = 16.7 ms nominal). The executor's motion loop,
``Executor.do`` and ``Executor.track``, calls ``env.step(action)``, which in this Isaac setup
renders every physics step; the settle path already uses ``env.sim.step(render=False)``. The
camera's own frame comes from an explicit ``isaac_io.capture_rgb``/``sim.render`` inside
``cam_pump`` and during tracking, NOT from the motion step's render. So the per-step render in
the motion loop may be pure overhead.

This boots the real demo 11 environment once and times, with the arm held at the park pose:

  * ``env.step(hold_action)``      - the motion loop's step (render on)
  * ``env.sim.step(render=False)`` - physics only

If the first is ~2x the second, the render is the overhead and skipping it in the motion loop
is the safe win (physics identical - render does not change the simulation). If they are the
same, the step is physics-bound and no step-count or render change will help; that is a property
of this articulated arm on this GPU.

    python tools/probe_step_cost.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import scene_spec  # noqa: E402


def time_steps(step_fn, n: int) -> float:
    for _ in range(3):                      # warm up the solver before timing
        step_fn()
    started = time.perf_counter()
    for _ in range(n):
        step_fn()
    return (time.perf_counter() - started) * 1000.0 / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    out = {"steps": args.steps, "physics_dt": None, "decimation": None}
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", list(scene_spec.DEFAULT_SCENE))
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        bot = executor_module.Executor(env, log=print)
        bot.park()
        out["physics_dt"] = float(getattr(env, "physics_dt", 0.0))
        out["decimation"] = int(getattr(env, "cfg", None).decimation) if getattr(env, "cfg", None) else None
        hold = bot._action(bot.current, executor_module.OPEN)

        out["env_step_ms"] = round(time_steps(lambda: env.step(hold), args.steps), 3)
        out["sim_step_no_render_ms"] = round(
            time_steps(lambda: env.sim.step(render=False), args.steps), 3)
        t0 = time.perf_counter()
        image, _cam = isaac_io.capture_rgb(env, sensor)
        out["capture_rgb_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if image is not None:
            out["frame_w"] = int(image.shape[1])
        Path(HERE.parent / "out11").mkdir(exist_ok=True) if (HERE.parent / "out11").exists() else None
        target = Path(r"C:\Users\Usuario\AppData\Local\Temp\opencode\step_cost.json")
        target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
