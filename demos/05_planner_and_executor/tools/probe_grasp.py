"""Can the robot actually pick things up, and where? Needs Isaac.

This is the measurement the demo's two open questions both need answered by, not by argument:

* **"the ball getting must be improved so much"** - how often does a real grasp succeed, for a
  cube and for a ball, and how far does the arm knock the target on the way in?
* **"i want the working area to be bigger"** - the working area can only be as big as the
  region where a grasp actually works. ``MAX_REACH_M`` arithmetic says the arm can reach a
  33 cm disc; ``tools/probe_ball_physics.py`` measured that the region it really reaches, in
  free space and approached from the parked pose, is smaller and lopsided. This runs the whole
  pick at each point of a grid and reports which ones worked, which is the only version of
  "can it work here" that means anything.

It drives ``Executor.run`` with the same ``observe`` callback the demo uses, so what it measures
is the code that ships, not a simplified version of it.

    python tools/probe_grasp.py --headless
    python tools/probe_grasp.py --headless --grid 3 --kinds ball
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import planner  # noqa: E402
import scene as scene_module  # noqa: E402
import scene_spec  # noqa: E402

# The two objects the benchmark moves around. Green and blue because each appears once in the
# default scene, so the camera cannot confuse this object with another of the same colour and
# the measurement is of grasping rather than of re-identification.
SUBJECTS = {"ball": "ball_green", "cube": "cube_blue"}

# How long the arm takes to get from one test point to the next. Generous, because a hurried
# arm arrives in a different configuration and a grasp failure caused by the benchmark's own
# timing would be reported as the robot's.
TRAVEL_HOME_STEPS = 110


class Bench:
    """One subject, a grid of positions, and the same pick cycle the demo runs."""

    def __init__(self, env, sensor, robot):
        self.env = env
        self.robot = robot
        self.bot = robot.bot
        self.results: list[dict] = []

    def place(self, prim: str, x: float, y: float) -> None:
        """Put the subject at a known point and park everything else well out of the way."""
        import torch

        upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.env.device)
        away = torch.tensor([[0.35, -0.60, -0.60]], device=self.env.device)
        for item in scene_spec.DEFAULT_SCENE:
            obj = self.env.scene[item.prim]
            if item.prim == prim:
                continue
            obj.write_root_pose_to_sim(torch.cat([away, upright], dim=-1))
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
        item = next(i for i in scene_spec.DEFAULT_SCENE if i.prim == prim)
        obj = self.env.scene[prim]
        obj.write_root_pose_to_sim(torch.cat(
            [torch.tensor([[x, y, scene_spec.TABLE_TOP_Z + item.height / 2.0]],
                          device=self.env.device), upright], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
        for _ in range(45):
            self.env.sim.step(render=False)
        self.env.scene.update(dt=self.env.physics_dt)
        self.bot.current = self.bot._gripper_pos().clone()

    def truth(self, prim: str) -> tuple[float, float]:
        where = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        return float(where[0]), float(where[1])

    def pick(self, prim: str, x: float, y: float) -> dict:
        """One full pick, reporting what the arm did to the target and whether it held it.

        The plan is built and run through ``Executor.run`` exactly as ``planner`` and
        ``10_command_to_motion`` do, including the closed-loop ``observe``, so a number here is a
        property of the shipped behaviour.
        """
        self.place(prim, x, y)
        self.robot.tracked_xy = {prim: (x, y)}
        before = self.truth(prim)

        # Aim at what the camera sees, not at the declared point: that is the whole loop, and
        # using the simulator's own position here would hide the perception error this demo is
        # about.
        seen = self.robot.locate(prim, None)
        if seen is None:
            return {"x": x, "y": y, "prim": prim, "seen": False, "grasp": "not seen"}

        thing = scene_module.Thing(id=0, kind=next(i.kind for i in scene_spec.DEFAULT_SCENE
                                                   if i.prim == prim),
                                  color=next(i.color for i in scene_spec.DEFAULT_SCENE
                                             if i.prim == prim),
                                  centroid_px=[], bbox_px=[], area_px=0, mean_hsv=[],
                                  confidence=1.0, source="bench", pos_m=[seen[0], seen[1], 0.0])
        grasp = planner.grasp_pose(thing)
        above = (grasp[0], grasp[1], planner.TRAVEL_Z)

        plan = planner.Plan(intent="pick")
        for action, pose, grip in (
            ("above", above, executor_module.OPEN),
            ("down", grasp, executor_module.OPEN),
            ("close", grasp, executor_module.CLOSE),
            ("lift", above, executor_module.CLOSE),
        ):
            plan.steps.append(planner.Step(action, pose, grip, 0, f"bench {action}"))
        plan.picks.append(0)

        observe = lambda p: self.robot.locate(p, self.robot.tracked_xy.get(p))
        run = self.bot.run(plan, resolve=lambda _id: prim, observe=observe, keys=set)

        after_approach = None
        # Re-measure where the target ended up relative to where the camera last put it, which
        # is the honest "did the arm move it" number: the simulator's truth is used here only
        # to score the pick, never to aim it.
        landed = self.truth(prim)
        height = next(i.height for i in scene_spec.DEFAULT_SCENE if i.prim == prim)
        z = float(self.env.scene[prim].data.root_pos_w[0][2])
        # Off the table is the test that does not depend on how the lift was achieved: a
        # kinematic carry holds the object's pose from the gripper frame, and a friction grasp
        # holds it through contact, but either way it ends up in the air.
        lifted = z > scene_spec.TABLE_TOP_Z + height * 0.5 + 0.004

        grasp_kind = "stale" if run.stale_grasps else (
            "kinematic" if run.kinematic_grasps else (
                "real" if run.real_grasps else "none"))
        return {
            "x": x, "y": y, "prim": prim, "seen": True, "grasp": grasp_kind,
            "aim_error_mm": math.dist(before, seen) * 1000,
            "tracked": run.tracked,
            "lifted": bool(lifted),
            "landed_mm": math.dist(before, landed) * 1000,
            "notes": "; ".join(s.note for s in run.steps if s.note),
        }

    def go_home(self) -> None:
        self.bot.do("above", np.asarray(executor_module.PARK_POSE, dtype=float),
                    executor_module.OPEN, steps=TRAVEL_HOME_STEPS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--kinds", nargs="+", default=["cube", "ball"], choices=list(SUBJECTS))
    parser.add_argument("--grid", type=int, default=5,
                        help="points per axis; the grid spans the working area")
    parser.add_argument("--area", nargs=2, type=float, default=None, metavar=("X0", "X1"),
                        help="x range to sweep, y taken from the working area")
    parser.add_argument("--y-area", nargs=2, type=float, default=None, metavar=("Y0", "Y1"),
                        help="y range to sweep, x taken from the working area")
    parser.add_argument("--rows", type=int, default=4, help="points along y")
    args = parser.parse_args()

    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", list(scene_spec.DEFAULT_SCENE))
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)

        # The Robot class is reused rather than copied, so ``locate`` - the closed loop this
        # benchmark is measuring - is the same function the demo runs. Only ``args`` is faked,
        # and only the four attributes ``look`` and ``locate`` read.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "demo10_main", ROOT / "10_command_to_motion.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        stub = argparse.Namespace(detect=list(colors.DEFAULT_DETECT), raw=False,
                                  save_every=0, headless=args.headless)
        robot = module.Robot(env, sensor, stub, viewer=None, model=None)

        bench = Bench(env, sensor, robot)
        x0, x1 = args.area if args.area else scene_spec.WORK_X
        y0, y1 = args.y_area if args.y_area else scene_spec.WORK_Y
        xs = np.linspace(x0, x1, args.grid)
        ys = np.linspace(y0, y1, args.rows)

        print(f"\nsweeping {len(xs) * len(ys)} points per kind, "
              f"x {xs[0]:+.3f}..{xs[-1]:+.3f}, y {ys[0]:+.3f}..{ys[-1]:+.3f}")
        print(f"balls are {scene_spec._ball('b', 'red', 0, 0).size[0] * 100:.0f} cm and the "
              f"gripper aperture is 36.2 mm\n")

        for kind in args.kinds:
            prim = SUBJECTS[kind]
            corner = "\u005cx"
            print(f"| {corner + ' y':>6} | " + " | ".join(f"{x:+.2f}" for x in xs) + " |")
            print("|---" * (len(xs) + 1) + "|")
            won = 0
            tried = 0
            for y in ys:
                cells = []
                for x in xs:
                    bench.go_home()
                    out = bench.pick(prim, float(x), float(y))
                    bench.results.append(out)
                    tried += 1
                    mark = {"real": "R", "kinematic": "k", "stale": "x",
                            "none": ".", "not seen": "?"}.get(out["grasp"], "?")
                    if out["grasp"] in ("real", "kinematic") and out.get("lifted"):
                        mark = mark.upper()
                        won += 1
                    cells.append(f"{mark:^5}")
                print(f"| {y:+.2f} | " + " | ".join(cells) + " |")
            print(f"\n{kind}: lifted {won} of {tried}")
            knocked = [r for r in bench.results if r.get("seen")]
            if knocked:
                print(f"  how far the camera's aim was from the truth when the plan was made: "
                      f"{np.mean([r['aim_error_mm'] for r in knocked]):.1f} mm mean")
                print(f"  how far the target ended up from where it started: "
                      f"{np.mean([r['landed_mm'] for r in knocked]):.1f} mm mean")
            print()

        print("R = lifted by friction, k = lifted only by the kinematic fallback, "
              "x = jaws closed on nothing,")
        print(". = did not lift, ? = the camera could not find it at all")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())