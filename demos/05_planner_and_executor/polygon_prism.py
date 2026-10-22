"""Small regular-prism geometry helpers for Demo 11's editable scene catalog."""
from __future__ import annotations

import math

import numpy as np


def mesh_arrays(width: float, height: float, sides: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a watertight, centred regular prism mesh with diameter ``width``."""
    if sides < 3:
        raise ValueError("A prism needs at least three sides.")
    if width <= 0 or height <= 0:
        raise ValueError("Prism dimensions must be positive.")
    radius = width / 2
    ring = [(radius * math.cos(2 * math.pi * i / sides),
             radius * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
    vertices = np.asarray([(x, y, z) for z in (-height / 2, height / 2)
                           for x, y in ring], dtype=np.float64)
    faces: list[tuple[int, int, int]] = []
    for i in range(1, sides - 1):
        faces.append((0, i + 1, i))
        faces.append((sides, sides + i, sides + i + 1))
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend(((i, j, sides + j), (i, sides + j, sides + i)))
    return vertices, np.asarray(faces, dtype=np.int64)


def spawn_cfg(sim_utils, common: dict, width: float, height: float, sides: int):
    """Build an IsaacLab MeshCfg whose custom spawner creates this convex prism."""
    import trimesh
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
    from isaaclab.sim.utils import clone

    vertices, faces = mesh_arrays(width, height, sides)

    @clone
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, None,
                                   **kwargs)
        return prim_utils.get_prim_at_path(prim_path)

    return sim_utils.MeshCfg(func=spawn, **common)
