"""Vuelca masas, inercias, límites y drives de un asset USD del SO-101.

    python tools/inspect_robot_asset.py --headless --usd <ruta.usd> --out results/asset_x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recoverykit import runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rt = runtime.boot(args, cameras=False)

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(args.usd)
    report = {"usd": args.usd, "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
              "links": [], "joints": [], "colliders": [], "materials": []}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(prim)
            report["links"].append({
                "path": path,
                "mass": m.GetMassAttr().Get(),
                "density": m.GetDensityAttr().Get(),
                "com": _vec(m.GetCenterOfMassAttr().Get()),
                "diag_inertia": _vec(m.GetDiagonalInertiaAttr().Get()),
            })
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        if revolute or prim.IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.RevoluteJoint(prim) if revolute else UsdPhysics.PrismaticJoint(prim)
            entry = {
                "path": path,
                "type": prim.GetTypeName(),
                "body0": [str(t) for t in j.GetBody0Rel().GetTargets()],
                "body1": [str(t) for t in j.GetBody1Rel().GetTargets()],
                "lower": j.GetLowerLimitAttr().Get(),
                "upper": j.GetUpperLimitAttr().Get(),
            }
            kind = "angular" if revolute else "linear"
            if prim.HasAPI(UsdPhysics.DriveAPI, kind):
                d = UsdPhysics.DriveAPI(prim, kind)
                entry["drive"] = {
                    "stiffness": d.GetStiffnessAttr().Get(),
                    "damping": d.GetDampingAttr().Get(),
                    "max_force": d.GetMaxForceAttr().Get(),
                    "type": d.GetTypeAttr().Get(),
                }
            vel = prim.GetAttribute("physxJoint:maxJointVelocity")
            if vel and vel.IsValid():
                entry["max_joint_velocity_deg_s"] = vel.Get()
            report["joints"].append(entry)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            approx = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            report["colliders"].append({"path": path, "type": prim.GetTypeName(), "approximation": approx})
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            mat = UsdPhysics.MaterialAPI(prim)
            report["materials"].append({
                "path": path,
                "static": mat.GetStaticFrictionAttr().Get(),
                "dynamic": mat.GetDynamicFrictionAttr().Get(),
                "restitution": mat.GetRestitutionAttr().Get(),
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[inspect] {len(report['links'])} cuerpos, {len(report['joints'])} articulaciones, "
          f"{len(report['colliders'])} colisionadores -> {args.out}", flush=True)
    rt.close()
    return 0


def _vec(v):
    return None if v is None else [float(x) for x in v]


if __name__ == "__main__":
    raise SystemExit(main())
