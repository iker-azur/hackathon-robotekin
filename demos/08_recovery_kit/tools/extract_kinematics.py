"""Extrae del USD la cadena cinemática y los colisionadores del SO-101.

    python tools/extract_kinematics.py --headless

Escribe ``assets/so101/kinematics.json`` (marcos de cada articulación) y
``assets/so101/colliders.npz`` (vértices de colisión en el marco de su eslabón).
Con eso ``recoverykit.kinematics`` hace cinemática directa e inversa sin Isaac,
igual que se haría en el robot real con el mismo URDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    args = parser.parse_args()
    rt = runtime.boot(args, cameras=False)

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from recoverykit.config import RobotCfg
    from recoverykit.workcell import resolve_usd

    stage = Usd.Stage.Open(resolve_usd(RobotCfg().usd))
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    joints = []
    links: dict[str, object] = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            links[prim.GetName()] = prim
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            b0 = j.GetBody0Rel().GetTargets()[0].name
            b1 = j.GetBody1Rel().GetTargets()[0].name
            q0, q1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
            joints.append({
                "name": prim.GetName(), "parent": b0, "child": b1,
                "pos0": list(j.GetLocalPos0Attr().Get()), "rot0": [q0.GetReal(), *q0.GetImaginary()],
                "pos1": list(j.GetLocalPos1Attr().Get()), "rot1": [q1.GetReal(), *q1.GetImaginary()],
                "axis": j.GetAxisAttr().Get(),
                "lower": float(np.deg2rad(j.GetLowerLimitAttr().Get())),
                "upper": float(np.deg2rad(j.GetUpperLimitAttr().Get())),
            })

    verts: dict[str, np.ndarray] = {}
    for name, link in links.items():
        link_inv = xf.GetLocalToWorldTransform(link).GetInverse()
        col = stage.GetPrimAtPath(str(link.GetPath()) + "/collisions")
        if not col.IsValid():
            continue
        pts = []
        # los colisionadores del asset son instancias: hay que recorrer sus proxies
        for prim in Usd.PrimRange(col, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                m = xf.GetLocalToWorldTransform(prim) * link_inv
                for p in UsdGeom.Mesh(prim).GetPointsAttr().Get():
                    q = m.Transform(Gf.Vec3d(*p))
                    pts.append([q[0], q[1], q[2]])
        if pts:
            # rejilla de 1 mm: suficiente para la geometría y cabe en el repositorio
            grid = np.unique(np.round(np.asarray(pts) / 0.001).astype(np.int32), axis=0)
            verts[name] = grid.astype(np.float32) * 0.001

    out = ROOT / "assets" / "so101"
    out.mkdir(parents=True, exist_ok=True)
    root = [n for n in links if all(j["child"] != n for j in joints)]
    (out / "kinematics.json").write_text(json.dumps({"root": root[0], "joints": joints}, indent=2, default=float),
                                         encoding="utf-8")
    np.savez(out / "colliders.npz", **verts)
    print(f"[kin] {len(joints)} articulaciones, colisionadores de {sorted(verts)} -> {out}", flush=True)
    rt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
