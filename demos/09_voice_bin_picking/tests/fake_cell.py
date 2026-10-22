"""Un simulador CINEMÁTICO de prueba que imita lo que ``SimRobot`` usa de ``recoverykit.workcell.Workcell``.

No es física: las articulaciones siguen la consigna con límite de velocidad, la pinza agarra por contacto
geométrico (una pieza a menos de 12 mm de su punto de agarre cuando se ordena cerrar) y la sostiene sin
fricción; la cámara es el render sintético de ``synth.py``. Sirve para ejercitar de punta a punta
percepción → selección → planificación → colisión → ``skills.execute_pick/place`` → verificación con
cámara, que es donde se esconden los errores de integración, sin lanzar Isaac. Lo que NO prueba: fricción,
contactos, estabilidad de agarre, ni nada que dependa de PhysX.
"""

from __future__ import annotations

import numpy as np

from binpick import config as C
from binpick import scene_gen as G
from binpick.cell import ResetResult
from binpick.segmentation import Context, PieceTruth, overhead_camera
from recoverykit import kinematics as K
from synth import render

JOINT_SPEED = 2.5          # rad/s
GRIPPER_SPEED = 3.0
CATCH_XY, CATCH_Z = 0.012, 0.015


class FakeCell:
    def __init__(self, scene):
        self.cfg = scene
        self.scene = scene
        self.cam = overhead_camera(scene.cameras[0])
        self.ctx = Context(self.cam, scene.table.top_z_m, lambda x, y, z: C.region_of(scene, x, y, z))
        self.cameras = {C.CAMERA: None}
        self.step_hooks: list = []
        self.sim_time = 0.0
        self.q = np.array(scene.robot.start_joints[:5], float)
        self.g = float(scene.robot.start_joints[5])
        self.q_t, self.g_t = self.q.copy(), self.g
        self.truths: dict[str, PieceTruth] = {}
        self.pieces: dict[str, object] = {}
        self.held: str | None = None
        self.local = K.tcp_local(C.PIECE_SIZE_M)
        self.q_contact = K.q_for_gap(C.PIECE_SIZE_M, -0.0955)
        self.base = None

    # ------------------------------------------------------------------ escena
    def set_pieces(self, truths: list[PieceTruth]) -> None:
        self.truths = {t.id: t for t in truths}
        self.pieces = {t.id: None for t in truths}
        self.held = None
        self.q = np.array(self.scene.robot.start_joints[:5], float)
        self.g = float(self.scene.robot.start_joints[5])
        self.q_t, self.g_t = self.q.copy(), self.g

    # ------------------------------------------------------------------- robot
    def proprio(self) -> dict:
        return {"arm_q": self.q.copy(), "arm_qd": np.zeros(5), "gripper_q": float(self.g), "gripper_qd": 0.0,
                "gripper_effort": 0.0, "arm_effort": np.zeros(5)}

    def set_arm_targets(self, q) -> None:
        self.q_t = np.asarray(q, float).copy()

    def set_gripper_target(self, g) -> None:
        self.g_t = float(g)

    def _tcp(self) -> np.ndarray:
        from recoverykit.workcell import base_transform

        if self.base is None:
            self.base = base_transform(self.scene)
        t = K.fk(dict(zip(K.ARM, self.q)), self.base)["gripper"]
        return t[:3, :3] @ self.local + t[:3, 3]

    def step(self, render: bool = False) -> None:
        for hook in self.step_hooks:
            hook()
        dt = self.scene.physics.dt
        self.q += np.clip(self.q_t - self.q, -JOINT_SPEED * dt, JOINT_SPEED * dt)
        dg = np.clip(self.g_t - self.g, -GRIPPER_SPEED * dt, GRIPPER_SPEED * dt)
        tcp = self._tcp()
        if self.held is None:
            self.g += dg
            if dg < 0 and self.g_t < self.q_contact:
                for pid, t in self.truths.items():
                    if np.linalg.norm(t.pos[:2] - tcp[:2]) < CATCH_XY and abs(t.pos[2] - tcp[2]) < CATCH_Z:
                        self.held = pid
                        self.g = max(self.g, self.q_contact)
                        break
        else:
            self.g = max(self.g + dg, self.q_contact) if self.g_t < self.q_contact else self.g + dg
            if self.g > self.q_contact + 0.03:                      # se abre: suelta
                self._release(self.held)
                self.held = None
        if self.held is not None:
            self.truths[self.held].pos = tcp.copy()
        self.sim_time += dt

    def _release(self, pid: str) -> None:
        t = self.truths[pid]
        region = C.region_of(self.scene, float(t.pos[0]), float(t.pos[1]), float(t.pos[2]))
        top = self.scene.table.top_z_m
        floor = {"origen": C.container(self.scene, C.BIN_ID).floor_m, "izquierda": 0.004, "derecha": 0.004}.get(region, 0.0)
        t.pos = np.array([t.pos[0], t.pos[1], top + floor + t.height / 2])

    # ------------------------------------------------------------------ cámara
    def render(self) -> None:
        return None

    def frame(self, name: str):
        rgb, depth = render(self.scene, self.ctx, list(self.truths.values()))
        return rgb, depth

    def camera_calibration(self, name: str) -> dict:
        c = self.cam
        return {"k": c.k, "rot": c.rot, "pos": c.pos, "width": c.width, "height": c.height}

    def truth(self, name: str) -> dict:
        t = self.truths[name]
        return {"pos": t.pos.copy(), "quat": t.quat.copy(), "lin_vel": np.zeros(3), "ang_vel": np.zeros(3)}


class FakeBinCell:
    """Lo que ``SimRobot`` espera de ``binpick.cell.BinCell``, sobre el simulador cinemático."""

    def __init__(self, cfg: C.BinPickCfg | None = None):
        self.cfg = cfg or C.BinPickCfg()
        self.scene = C.build_scene(self.cfg)
        self.cell = FakeCell(self.scene)

    def reset_scene(self, spec: G.SceneSpec, max_attempts: int = 1) -> ResetResult:
        truths = []
        for d, (pos, quat, _layer) in zip(spec.drops, G.approximate_settle(spec, self.scene).values()):
            pc = self.scene.piece(d.piece_id)
            truths.append(PieceTruth(d.piece_id, pc.color, pc.shape, pc.size_m, pc.height, pos, quat))
        self.cell.set_pieces(truths)
        poses = {t.id: (t.pos.copy(), t.quat.copy()) for t in truths}
        return ResetResult(spec, poses, G.analyze_layout(poses), [], 1, 0.0)

    def truth_poses(self) -> dict:
        return {pid: (t.pos.copy(), t.quat.copy()) for pid, t in self.cell.truths.items()}

    def audit(self) -> dict:
        regions: dict[str, dict[str, int]] = {}
        where = {}
        for pid, t in self.cell.truths.items():
            r = C.region_of(self.scene, float(t.pos[0]), float(t.pos[1]), float(t.pos[2]))
            where[pid] = r
            regions.setdefault(r, {})
            regions[r][t.color] = regions[r].get(t.color, 0) + 1
        return {"regions": regions, "where": where}
