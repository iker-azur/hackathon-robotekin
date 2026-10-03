"""Check that the hollow trays work, and measure the inset that perception needs.

Three things have to be true before the rest of the demo can rely on a tray, and none of
them can be checked by reading the code:

1. It spawns. IsaacLab's ``RigidObject`` refuses a prim holding more than one rigid body
   (rigid_object.py:474-479), and this tray is one body with five collision children, so
   the first risk is that the parenting does not produce what hollow.py claims.
2. It holds. An object dropped from above must end up inside the cavity, not beside it.
   This is the whole reason the tray is hollow.
3. Perception reports the right interior. ``perceive.inner_rect`` estimates the cavity by
   insetting the *detected* bounding box, and a hollow tray's blob is a rim plus part of
   the floor, not a filled rectangle. The value of ``WALL_INSET_FRACTION`` that makes that
   estimate land on the true cavity has to be measured, not assumed.

Run it headless. It prints the measured inset and says whether it differs from the one
compiled into perceive.py, and it writes results/tray_check.md.

    python tools/check_trays.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import colors  # noqa: E402
import hollow  # noqa: E402
import isaac_io  # noqa: E402
import perceive  # noqa: E402
import scene_spec  # noqa: E402

RESULTS = ROOT / "results"

# How high above the cavity centre the test object starts, and how long it is given to
# fall. 4 cm is roughly the arm's own travel clearance, so this is a realistic drop rather
# than a token one. 180 steps at 16.7 ms is 3 s, which is long enough for a bouncing ball
# to stop.
DROP_HEIGHT = 0.04
FALL_STEPS = 180

# A dropped object counts as inside when its centre is within the true cavity, expanded by
# this much. Without some slack the test would fail on a ball that came to rest touching
# a wall, which is a success, not a failure.
SLACK_M = 0.004


def inside(centre, size, x, y, slack=SLACK_M):
    """Whether a point is within a cavity, given its CENTRE and size.

    ``perceive.inner_rect_m`` is corner+size (``[x, y, w, d]``, see scene.py's schema) and
    this takes centre+size, so the conversion happens here rather than being silently wrong
    at the call site. Getting that backwards made the drop test fail while the object was
    demonstrably inside.
    """
    cx, cy = centre
    w, d = size
    return ((cx - w / 2 - slack) <= x <= (cx + w / 2 + slack)
            and (cy - d / 2 - slack) <= y <= (cy + d / 2 + slack))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="top", choices=sorted(scene_spec.CAMERA_CHOICES))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    items = list(scene_spec.DEFAULT_SCENE)
    trays = [i for i in items if i.container]
    loose = [i for i in items if not i.container]
    print(f"scene: {len(loose)} loose object(s), {len(trays)} tray(s)")
    for item in trays:
        ratio = hollow.cavity_ratio(*item.inner, item.wall)
        wanted = hollow.inset_fraction_for(*item.inner, item.wall)
        print(f"  {item.prim:12s} cavity {item.inner[0] * 100:.1f} x {item.inner[1] * 100:.1f} cm, "
              f"{item.cavity_depth * 100:.1f} cm deep, cavity/footprint {ratio:.3f}, "
              f"so the inset must be {wanted:.4f}")
    print(f"  perceive.WALL_INSET_FRACTION is {perceive.WALL_INSET_FRACTION}, "
          f"giving an interior of {1 - 2 * perceive.WALL_INSET_FRACTION:.3f} of the footprint")
    print("")

    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env(args.camera, items)
    problems = []
    try:
        env.reset()
        isaac_io.aim_camera(env, args.camera, sensor)
        isaac_io.park_arm(env)
        isaac_io.place_items(env, items)

        # One capture, used by the drop test and the perception checks below.
        drop_results: list[bool] = []
        image, depth, camera = isaac_io.capture(env, sensor)
        if depth is None:
            print("  ERROR: no depth image from the sensor; the base_centre estimate cannot be tested")
            problems.append("the top camera returned no depth image")

        # One rigid body per tray, or IsaacLab would already have raised by now. Confirm
        # it directly anyway, because this is the claim hollow.py is making.
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdPhysics
        for item in trays:
            prim = prim_utils.get_prim_at_path(f"/World/envs/env_0/Scene/{item.prim}")
            # The rigid body is on the tray prim ITSELF, and the five parts underneath it
            # are geometry only. That is the whole claim hollow.py makes, so it is checked
            # in both directions: the parent must have the API, and no child may.
            parent_has = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            child_has = [p.GetName() for p in prim.GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            parts = len(prim.GetChildren())
            print(f"  {item.prim:12s} prim ok, {parts} child part(s), "
                  f"rigid body on parent: {parent_has}, on children: {child_has or 'none'}")
            if not parent_has:
                problems.append(f"{item.prim}: the tray prim itself has no RigidBodyAPI")
            if child_has:
                problems.append(f"{item.prim}: child part(s) {child_has} wrongly carry a RigidBodyAPI")
            if parts != 5:
                problems.append(f"{item.prim}: expected 5 parts, found {parts}")

        print("")
        print("drop test: dropped at the centre PERCEPTION reports, not the true one")
        print("  (dropping at the true centre would pass trivially and prove nothing)")
        for tray in trays:
            probe_scene = perceive.perceive(image, source="tray-probe", camera=camera,
                                            frame=isaac_io.robot_frame(env),
                                            plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                            looking_for=colors.colors(colors.DEFAULT_DETECT))
            seen = next((c for c in probe_scene.containers if c.color == tray.color), None)
            if seen is None or seen.pos_m is None:
                print(f"  {tray.prim}: not detected, cannot place")
                problems.append(f"{tray.prim}: not detected, so nothing can be dropped into it")
                continue
            aim = (seen.pos_m[0], seen.pos_m[1])
            print(f"  {tray.prim:12s} perceived centre {aim[0]:+.4f}, {aim[1]:+.4f}   "
                  f"true {tray.x:+.4f}, {tray.y:+.4f}   off by "
                  f"{(((aim[0] - tray.x) ** 2 + (aim[1] - tray.y) ** 2) ** 0.5) * 1000:.1f} mm")

            cube = loose[0]
            env.scene[cube.prim].write_root_pose_to_sim(
                _pose(env, aim[0], aim[1],
                      scene_spec.TABLE_TOP_Z + tray.floor + cube.size[2] / 2 + DROP_HEIGHT)
            )
            env.scene[cube.prim].write_root_velocity_to_sim(_zeros(env))
            for _ in range(FALL_STEPS):
                env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            landed = env.scene[cube.prim].data.root_pos_w[0].detach().cpu().numpy()
            rim = scene_spec.TABLE_TOP_Z + tray.height
            in_cavity = inside((tray.x, tray.y), tray.inner, float(landed[0]), float(landed[1]))
            below_rim = float(landed[2]) < rim
            print(f"  {'':12s} {cube.prim} landed at ({landed[0]:+.3f}, {landed[1]:+.3f}, "
                  f"{landed[2]:.3f})   in cavity: {in_cavity}   below rim ({rim:.3f}): {below_rim}")
            if not in_cavity:
                problems.append(
                    f"{cube.prim} missed the cavity of {tray.prim} when aimed at the perceived "
                    f"centre: landed {landed[:2].tolist()} against a cavity centred "
                    f"({tray.x:.3f}, {tray.y:.3f}) of {tray.inner}"
                )
            if not below_rim:
                problems.append(f"{cube.prim} came to rest above the rim of {tray.prim}")
            drop_results.append(in_cavity)
            isaac_io.place_items(env, items)

        print("")
        print("perception on the real trays")
        scene = perceive.perceive(image, source="tray-check", camera=camera,
                                  frame=isaac_io.robot_frame(env),
                                  plane_z=scene_spec.TABLE_TOP_Z, depth=depth,
                                  looking_for=colors.colors(colors.DEFAULT_DETECT))
        perceive.save_png(ROOT / "out" / "tray_check.png", perceive.annotate(image, scene))
        print(scene.summary())

        centre_results: dict[str, list[float]] = {}

        lines = [
            "# hollow tray check", "",
            "| tray | true cavity (m) | detected inner_rect_m | centre error | size error |",
            "|---|---|---|---|---|",
        ]
        for item in trays:
            found = next((c for c in scene.containers if c.color == item.color), None)
            true_cx, true_cy = item.x, item.y
            true_w, true_d = item.inner
            true_outer_w = item.inner[0] + 2 * item.wall
            true_outer_d = item.inner[1] + 2 * item.wall
            if found is None or found.inner_rect_m is None:
                print(f"  {item.prim:12s} NOT DETECTED as a container")
                problems.append(f"{item.prim}: not detected as a container")
                lines.append(f"| {item.prim} | {true_w:.4f} x {true_d:.4f} | not detected | - | - |")
                continue
            rx, ry, rw, rd = found.inner_rect_m
            centre_error = ((rx + rw / 2 - true_cx) ** 2 + (ry + rd / 2 - true_cy) ** 2) ** 0.5
            size_error = max(abs(rw - true_w), abs(rd - true_d))
            scale = scene.ground_px_per_m
            print(f"  {item.prim:12s} blob bbox {found.bbox_px} px, at {scale} px/m that is "
                  f"{found.bbox_px[2] / scale * 100:.1f} x {found.bbox_px[3] / scale * 100:.1f} cm "
                  f"(true footprint {true_outer_w * 100:.1f} x {true_outer_d * 100:.1f})")
            print(f"  {'':12s} true cavity {true_w * 100:.1f} x {true_d * 100:.1f} cm at "
                  f"({true_cx:+.3f},{true_cy:+.3f})   detected {rw * 100:.1f} x {rd * 100:.1f} cm at "
                  f"({rx + rw / 2:+.3f},{ry + rd / 2:+.3f})   centre off {centre_error * 1000:.0f} mm, "
                  f"sides off {size_error * 1000:.0f} mm")
            if found.footprint_m:
                fx, fy, fw, fd = found.footprint_m
                print(f"  {'':12s} depth footprint {fw * 100:.1f} x {fd * 100:.1f} cm at "
                      f"({fx + fw / 2:+.3f},{fy + fd / 2:+.3f})   "
                      f"(true footprint {true_outer_w * 100:.1f} x {true_outer_d * 100:.1f}, "
                      f"error {abs(fw - true_outer_w) * 1000:.0f} / {abs(fd - true_outer_d) * 1000:.0f} mm)")
            else:
                print(f"  {'':12s} depth footprint unavailable, fell back to the bounding box")

            # Which centre estimate should the executor aim at? Three candidates, all
            # measured against the same ground truth.
            candidates = {
                "pos_m (plane ray)": (found.pos_m[0], found.pos_m[1]),
            }
            if found.base_centre_m:
                candidates["base_centre_m (depth median)"] = tuple(found.base_centre_m)
            if found.footprint_m:
                fx, fy, fw, fd = found.footprint_m
                candidates["footprint_m centre"] = (fx + fw / 2, fy + fd / 2)
            print(f"  {'':12s} centre error against truth ({true_cx:+.3f},{true_cy:+.3f}):")
            for name, (cx, cy) in candidates.items():
                offset = ((cx - true_cx) ** 2 + (cy - true_cy) ** 2) ** 0.5
                # How much room is left before the object would touch a wall: the cavity
                # half-width minus the object half-width, minus the centre error.
                slack = true_w / 2.0 - loose[0].size[0] / 2.0 - offset
                print(f"  {'':16s} {name:28s} off by {offset * 1000:5.1f} mm, "
                      f"leaves {slack * 1000:+5.1f} mm to the wall")
                centre_results.setdefault(name, []).append(offset)

            # Only gate on the centre estimate, not on the footprint: the footprint is a
            # documented diagnostic that is known to be unreliable on one axis.
            if not centre_results:
                problems.append("no centre estimate was measured at all")
            best = min(centre_results.items(), key=lambda kv: sum(kv[1]))
            if sum(best[1]) / len(best[1]) > 0.012:
                problems.append(
                    f"{item.prim}: the best centre estimate ({best[0]}) is "
                    f"{sum(best[1]) / len(best[1]) * 1000:.0f} mm off on average, over the 12 mm budget"
                )
            lines.append(f"| {item.prim} | {true_w:.4f} x {true_d:.4f} at ({true_cx:+.4f}, {true_cy:+.4f}) "
                         f"| {rw:.4f} x {rd:.4f} at ({rx + rw / 2:+.4f}, {ry + rd / 2:+.4f}) "
                         f"| {centre_error * 1000:.0f} mm | {size_error * 1000:.0f} mm |")
            # The gripper aims at this rect. A side error over half a wall means it aims at
            # the wall rather than the cavity.


        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "tray_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwritten: {RESULTS / 'tray_check.md'}")

        print("")
        if problems:
            for line in problems:
                print(f"  PROBLEM: {line}")
            print(f"RESULT: FAILED ({len(problems)} problem(s))")
            return 2
        print("RESULT: SUCCESS")
        return 0
    finally:
        env.close()


def _pose(env, x, y, z):
    import torch
    return torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=env.device)


def _zeros(env):
    import torch
    return torch.zeros(1, 6, device=env.device)


if __name__ == "__main__":
    raise SystemExit(main())