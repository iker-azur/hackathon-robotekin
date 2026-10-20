"""Measure Demo 11's ball/table contact without robot approach motion.

Unlike ``probe_ball_physics.py`` this uses Demo 11's real mass, restitution, friction and
damping. It reports the simulator's raycast surface normals and the first two seconds of a
stationary, exactly table-height 20 mm ball.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import executor  # noqa: E402
import isaac_io  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402


def load_demo11():
    spec = importlib.util.spec_from_file_location(
        "demo11_ball_contact_probe", ROOT / "11_real_sorting_with_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--restitution", type=float, default=None,
                        help="optional ball restitution override for an A/B physics probe")
    parser.add_argument("--rolling-damping", type=float, default=None,
                        help="optional ball angular damping override")
    parser.add_argument("--spawn-gap-mm", type=float, default=0.0,
                        help="start this many millimetres above nominal table contact")
    parser.add_argument("--bind-table-material", action="store_true",
                        help="bind Demo 11's declared melamine material to the tabletop collider")
    parser.add_argument("--set-default-table-material", action="store_true",
                        help="temporarily set the scene default material to the declared table values")
    parser.add_argument("--disable-robot-collisions", action="store_true",
                        help="diagnostic control: disable robot collision shapes before ball contact")
    parser.add_argument("--replace-table-collision-with-box", action="store_true",
                        help="diagnostic control: replace the imported top mesh with a flat box collider")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.restitution is not None and not 0.0 <= args.restitution <= 1.0:
        parser.error("--restitution must be between zero and one")
    if args.rolling_damping is not None and args.rolling_damping < 0:
        parser.error("--rolling-damping must be non-negative")
    if args.restitution is not None:
        real_world.PET_BALL = replace(real_world.PET_BALL, restitution=args.restitution)
    if args.rolling_damping is not None:
        real_world.PET_BALL = replace(real_world.PET_BALL,
                                      rolling_damping=args.rolling_damping)

    item = scene_spec._ball("probe_ball_20", "green", .38, -.435, .02)
    items = [item]
    module = load_demo11()
    isaac_io.prepare_runtime(headless=args.headless)
    env, _sensor = module.build_real_env(
        "top", items, grip_stiffness=500, device="cuda:0",
        contact_diagnostics_for=item.prim,
        contact_diagnostics_all_robot_links=True)
    lines: list[str] = []

    def log(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    try:
        env.reset()
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        log("scene prims matching cube (USD path, schemas, visible, collision):")
        for prim in stage.Traverse():
            if "cube" not in str(prim.GetPath()).lower():
                continue
            try:
                visibility = prim.GetAttribute("visibility").Get()
            except Exception:
                visibility = None
            try:
                collision = prim.GetAttribute("physics:collisionEnabled").Get()
            except Exception:
                collision = None
            log(f"  {prim.GetPath()} type={prim.GetTypeName()} "
                f"schemas={tuple(prim.GetAppliedSchemas())} "
                f"visibility={visibility} collision={collision}")
        ball_prim = stage.GetPrimAtPath(
            f"/World/envs/env_0/Scene/{item.prim}")
        for prim in (ball_prim, stage.GetPrimAtPath(f"{ball_prim.GetPath()}/geometry"),
                     stage.GetPrimAtPath(f"{ball_prim.GetPath()}/geometry/mesh")):
            if not prim or not prim.IsValid():
                continue
            props = []
            for attr in prim.GetAttributes():
                name = str(attr.GetName()).lower()
                if any(key in name for key in ("mass", "damping", "restitution", "friction")):
                    try:
                        props.append(f"{attr.GetName()}={attr.Get()}")
                    except Exception:
                        pass
            print(f"physics prim {prim.GetPath()} schemas={tuple(prim.GetAppliedSchemas())} "
                  f"attrs={props}", flush=True)
        if args.disable_robot_collisions:
            from pxr import Usd, UsdPhysics

            robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
            disabled = 0
            for prim in Usd.PrimRange(robot_root):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    api = UsdPhysics.CollisionAPI(prim)
                    api.CreateCollisionEnabledAttr().Set(False)
                    disabled += 1
            print(f"disabled {disabled} robot collision prims for the isolated-contact control",
                  flush=True)
        if args.replace_table_collision_with_box:
            from pxr import Gf, UsdGeom, UsdPhysics

            table_prim = stage.GetPrimAtPath(
                "/World/envs/env_0/Scene/counter_right_main_group/geometry_0")
            if not table_prim or not table_prim.IsValid():
                raise RuntimeError("could not find the imported tabletop collider")
            UsdPhysics.CollisionAPI.Apply(table_prim).CreateCollisionEnabledAttr().Set(False)
            plane = UsdGeom.Cube.Define(stage, "/World/Demo11Probe/table_top_box")
            plane.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(plane.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(.40, -.435, .030911))
            xf.AddScaleOp().Set(Gf.Vec3f(.5, .5, .02))
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
            print("disabled imported tabletop mesh and added flat 1 m x 1 m box "
                  "with top at z=0.040911 m", flush=True)
        if args.set_default_table_material:
            from pxr import UsdPhysics

            candidates = ("/physicsScene/defaultMaterial",
                          "/World/physicsScene/defaultMaterial")
            default_prim = next((stage.GetPrimAtPath(path) for path in candidates
                                 if stage.GetPrimAtPath(path).IsValid()), None)
            if default_prim is None:
                default_prim = next((prim for prim in stage.Traverse()
                                     if prim.GetName() == "defaultMaterial"), None)
            if default_prim is None or not default_prim.IsValid():
                raise RuntimeError("could not locate the effective PhysX default material")
            material = UsdPhysics.MaterialAPI.Apply(default_prim)
            material.GetStaticFrictionAttr().Set(real_world.TABLE.static_friction)
            material.GetDynamicFrictionAttr().Set(real_world.TABLE.dynamic_friction)
            material.GetRestitutionAttr().Set(real_world.TABLE.restitution)
            print(f"set default material {default_prim.GetPath()} to melamine "
                  f"mu={real_world.TABLE.static_friction}/{real_world.TABLE.dynamic_friction}, "
                  f"e={real_world.TABLE.restitution}", flush=True)
        if args.bind_table_material:
            from pxr import Sdf, UsdPhysics, UsdShade, PhysxSchema
            from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            material_path = Sdf.Path("/World/PhysicsMaterials/Demo11MelamineProbe")
            material_prim = UsdShade.Material.Define(stage, material_path).GetPrim()
            physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
            physics_material.CreateStaticFrictionAttr().Set(real_world.TABLE.static_friction)
            physics_material.CreateDynamicFrictionAttr().Set(real_world.TABLE.dynamic_friction)
            physics_material.CreateRestitutionAttr().Set(real_world.TABLE.restitution)
            PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
            table_prim = stage.GetPrimAtPath(
                "/World/envs/env_0/Scene/counter_right_main_group/geometry_0")
            if not table_prim or not table_prim.IsValid():
                raise RuntimeError("could not find Demo 11 tabletop collision geometry")
            add_physics_material_to_prim(stage, table_prim, material_path)
            print(f"bound melamine material mu={real_world.TABLE.static_friction}/"
                  f"{real_world.TABLE.dynamic_friction}, e={real_world.TABLE.restitution}",
                  flush=True)
        bot = module.StrictExecutor(env, allow_kinematic=False).with_layout(items)
        parked = bot.park(steps=executor.PARK_STEPS)
        isaac_io.hold_arm_pose(env)
        log(f"park={parked.ok} {parked.note}")

        # Move the only loose body below the tabletop before raycasting so it cannot occlude it.
        import torch
        actor = env.scene[item.prim]
        hidden = torch.tensor([[.38, -.435, -.30, 1.0, 0.0, 0.0, 0.0]], device=env.device)
        actor.write_root_pose_to_sim(hidden)
        actor.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        hidden_pos = actor.data.root_pos_w[0].detach().cpu().numpy()
        log(f"ball moved clear for table rays: z={hidden_pos[2]:.6f}")

        from omni.physx import get_physx_scene_query_interface
        query = get_physx_scene_query_interface()
        log("PhysX table raycast (z and normal in world coordinates):")
        for x, y in ((.34, -.435), (.38, -.435), (.42, -.435),
                     (.38, -.47), (.38, -.455), (.38, -.415), (.38, -.40)):
            hit = query.raycast_closest((x, y, .30), (0.0, 0.0, -1.0), .6)
            if not hit or not hit.get("hit"):
                log(f"  ({x:.3f},{y:.3f}) no hit")
                continue
            p, n = hit["position"], hit["normal"]
            log(f"  ({x:.3f},{y:.3f}) z={p[2]:.6f} n=({n[0]:+.5f},{n[1]:+.5f},{n[2]:+.5f}) "
                f"collision={hit.get('collision')} rigidBody={hit.get('rigidBody')} "
                f"material={hit.get('material')} keys={tuple(hit)}")
        hits = []

        def collect_hit(hit):
            hits.append(hit)
            return True

        query.raycast_all((.38, -.435, .30), (0.0, 0.0, -1.0), .6, collect_hit)
        log("  all hits at the ball's XY (to reveal the surface behind stale ray-query shapes):")
        if hits:
            log(f"    hit attributes: {tuple(name for name in dir(hits[0]) if not name.startswith('_'))}")
        for hit in hits[:8]:
            p, n = hit.position, hit.normal
            log(f"    z={p[2]:.6f} n=({n[0]:+.5f},{n[1]:+.5f},{n[2]:+.5f}) "
                f"collision={getattr(hit, 'collision', None)} "
                f"rigidBody={getattr(hit, 'rigid_body', None)} "
                f"material={getattr(hit, 'material', None)}")

        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        position = torch.tensor([[.38, -.435,
                                  scene_spec.TABLE_TOP_Z + item.height / 2
                                  + args.spawn_gap_mm / 1000]],
                                device=env.device)
        pose = torch.cat((position, quat), dim=-1)
        zeros = torch.zeros(1, 6, device=env.device)
        actor.write_root_pose_to_sim(pose)
        actor.write_root_velocity_to_sim(zeros)
        env.scene.update(dt=env.physics_dt)
        initial_velocity = actor.data.root_lin_vel_w[0].detach().cpu().numpy()
        log(f"post-teleport velocity before physics: ({initial_velocity[0]:+.7f},"
            f"{initial_velocity[1]:+.7f},{initial_velocity[2]:+.7f})")
        log(f"ball material: mass={real_world.mass_kg(item):.7f} kg, "
            f"restitution={real_world.PET_BALL.restitution}, "
            f"linear_damping={real_world.PET_BALL.linear_damping}, "
            f"angular_damping={real_world.PET_BALL.rolling_damping}, "
            f"spawn_gap={args.spawn_gap_mm:g}mm")
        log("stationary placement ticks:")
        for step in range(1, args.steps + 1):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step == 1 or step % 5 == 0 or step == args.steps:
                pos = actor.data.root_pos_w[0].detach().cpu().numpy()
                vel = actor.data.root_lin_vel_w[0].detach().cpu().numpy()
                ang = actor.data.root_ang_vel_w[0].detach().cpu().numpy()
                log(f"  {step:3d}: p=({pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}) "
                    f"v=({vel[0]:+.5f},{vel[1]:+.5f},{vel[2]:+.5f}) "
                    f"w=({ang[0]:+.4f},{ang[1]:+.4f},{ang[2]:+.4f})")
                contact_forces = []
                for link in ("shoulder", "upper_arm", "lower_arm", "wrist",
                             "gripper", "jaw"):
                    try:
                        matrix = env.scene[f"demo11_contact_{link}"].data.force_matrix_w
                        force = float(torch.linalg.norm(matrix[0]).detach().cpu())
                        if force > 1e-5:
                            contact_forces.append(f"{link}={force:.5f}N")
                    except Exception:
                        continue
                if contact_forces:
                    log("       robot/ball contact: " + ", ".join(contact_forces))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
