"""El robot simulado como ``RobotPort``: ver, alcanzar, coger, dejar. Une la célula de Isaac con el orquestador.

Reparto del sistema híbrido (decisión razonada en ``docs/02_arquitectura.md``):

* **coger** lo hace la política aprendida (ACT, ``policy='act'``) o, si se pide, el guion del experto
  (``policy='guion'``, que NO es aprendizaje y así se etiqueta en el panel y en los informes);
* **llevar a la caja y dejar** es programado (cinemática inversa del demo 08): el destino es una caja fija
  y conocida, lo único que hay que decidir es un hueco libre alcanzable.

El agente ve sólo la imagen RGB-D cenital y la propiocepción de los servos (lo que reportaría un STS3215
real). Nunca lee poses de piezas del simulador: eso lo hace ``BinCell.audit`` y sólo para evaluar.
"""

from __future__ import annotations

import time

import numpy as np

from . import config as C
from . import observation as O
from . import picking, viz
from . import policy as POL
from . import segmentation as S
from .instances import Instance
from .orchestrator import Observation, PickResult, PlaceResult, Stopped
from .pick_runner import PickRunner
from .place import Placer
from .selection import Candidate
from .telemetry import BinTelemetry, piece_rows
from .tracker import Tracker
from recoverykit import kinematics as K  # noqa: E402
from recoverykit import perception as P  # noqa: E402
from recoverykit import skills  # noqa: E402
from recoverykit.control import Controller  # noqa: E402
from recoverykit.workcell import base_transform  # noqa: E402

ARM_LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
LIVE_FRAME_S = 0.25
VIEW_FRAME_S = 0.2


class SimRobot:
    def __init__(self, bcell, tele: BinTelemetry, segmenter: str = "hsv", policy=None, realtime: float = 1.0,
                 act_info_check: bool = True):
        self.bcell, self.cell, self.scene, self.cfg = bcell, bcell.cell, bcell.scene, bcell.cfg
        self.tele, self.realtime = tele, realtime
        self.base = base_transform(self.scene)
        self.top = self.scene.table.top_z_m
        self.bin = C.container(self.scene, C.BIN_ID)
        self.look_q = np.array(self.scene.robot.start_joints[:5])
        self.home_gripper = float(self.scene.robot.start_joints[5])
        self.ctrl = Controller(self.cell, monitor=self._monitor)
        self.cell.step_hooks.append(self._on_step)
        cal = self.cell.camera_calibration(C.CAMERA)
        self.cam = P.Camera(cal["k"], cal["rot"], cal["pos"], cal["width"], cal["height"])
        self.ctx = S.Context(self.cam, self.top, lambda x, y, z: C.region_of(self.scene, x, y, z))
        self.segmenter_name = segmenter
        self.segmenter = S.make_segmenter(segmenter, self.ctx)
        self.spec = O.make_spec(self.cam, self.top, self.cfg)
        self.tracker = Tracker()
        self.placer = Placer(self.scene, self.base, self.look_q)
        self.policy = policy or POL.ScriptedPolicy()
        self.has_view = C.VIEW_CAMERA in self.cell.cameras
        self.last: Observation | None = None
        self._plans: dict[str, tuple[skills.PickPlan, tuple[float, float]]] = {}
        self._held: tuple[Candidate, skills.PickPlan, tuple[float, float]] | None = None
        self._wall0, self._sim0 = time.perf_counter(), self.cell.sim_time
        self._last_live = self._last_view = -1.0
        self.last_pick_stats: dict = {}
        self.live_frames = True          # imágenes en vivo para el panel durante el movimiento (cuestan renders)
        if act_info_check and getattr(self.policy, "kind", "") == "act":
            errors, warnings = POL.check_compat(getattr(self.policy, "info", {}), self.spec, segmenter)
            if errors:
                raise RuntimeError("; ".join(errors))
            for w in warnings:
                self.tele.event("aviso", w)

    # =================================================================== tiempo y panel
    def _on_step(self) -> None:
        while self.tele.paused and not self.tele.stop_requested:
            time.sleep(0.05)
        if self.realtime > 0:
            ahead = (self.cell.sim_time - self._sim0) / self.realtime - (time.perf_counter() - self._wall0)
            if ahead > 0.002:
                time.sleep(ahead)

    def sync_clock(self) -> None:
        self._wall0, self._sim0 = time.perf_counter(), self.cell.sim_time

    def _monitor(self, info: dict) -> str | None:
        if self.tele.stop_requested:
            return "detenido por el usuario"
        t = self.cell.sim_time
        if self.live_frames and t - self._last_live >= LIVE_FRAME_S:
            self._last_live = t
            self.cell.render()
            rgb, _ = self.cell.frame(C.CAMERA)
            self._push_live(rgb)
            self.tele.set(sim_time=round(t, 2))
        return None

    def _push_live(self, rgb: np.ndarray) -> None:
        self.tele.push_named("cam", viz.to_jpeg(viz.annotate(rgb, [], self.scene, self.cam, "rgb")))
        if self.has_view and self.cell.sim_time - self._last_view >= VIEW_FRAME_S:
            self._last_view = self.cell.sim_time
            self.tele.push_named("vista", viz.plain_jpeg(self.cell.frame(C.VIEW_CAMERA)[0]))

    def idle_tick(self) -> None:
        """En reposo la física sigue (si algo empuja una pieza, se mueve) y el panel recibe imagen."""
        self.cell.step()
        t = time.perf_counter()
        if t - getattr(self, "_idle_t", 0.0) > 0.25:
            self._idle_t = t
            self.cell.render()
            rgb, _ = self.cell.frame(C.CAMERA)
            self._push_live(rgb)

    # ==================================================================== percepción
    def _arm_points(self) -> np.ndarray:
        poses = K.fk(dict(zip(K.ARM, self.ctrl.q())), self.base)
        pts = []
        for link in ARM_LINKS:
            v = K.colliders()[link][::2]
            t = poses[link]
            pts.append(v @ t[:3, :3].T + t[:3, 3])
        return np.concatenate(pts)

    def _grab(self) -> tuple[np.ndarray, np.ndarray]:
        # la cámara devuelve el render anterior al pedido: dos renders (demo 06, demo 08)
        self.cell.render()
        self.cell.render()
        return self.cell.frame(C.CAMERA)

    def _grab_settled(self) -> tuple[np.ndarray, np.ndarray]:
        """Imagen tras un cambio brusco (escena nueva, pieza movida): varios renders para que no quede el fantasma de lo anterior."""
        for _ in range(C.OBSERVE_RENDERS - 2):
            self.cell.render()
        return self._grab()

    def observe(self, why: str) -> Observation:
        rgb, depth = self._grab_settled()
        instances = self.segmenter.segment(rgb, depth)
        mask = P.robot_pixels(self.cam, self._arm_points(), dilate_px=15)

        def visible(p) -> bool:
            uv, _ = self.cam.project(np.asarray(p)[None])
            u, v = int(uv[0, 0]), int(uv[0, 1])
            return 0 <= u < self.cam.width and 0 <= v < self.cam.height and mask[v, u] == 0

        for e in self.tracker.update(instances, self.cell.sim_time, visible=visible):
            if e.kind in ("movida", "perdida"):
                self.tele.event("ver", f"{e.track}: {e.detail}", track=e.track)
        self.last = Observation(instances, frame=(rgb, depth))
        self._publish(rgb, instances)
        return self.last

    def _publish(self, rgb, instances: list[Instance], target_track: str | None = None) -> None:
        self.tele.push_named("cam", viz.to_jpeg(viz.annotate(rgb, instances, self.scene, self.cam, "rgb", target_track)))
        self.tele.push_named("seg", viz.to_jpeg(viz.annotate(rgb, instances, self.scene, self.cam, "seg", target_track)))
        if self.has_view:
            self.tele.push_named("vista", viz.plain_jpeg(self.cell.frame(C.VIEW_CAMERA)[0]))
        self.tele.set(instances=piece_rows(instances), has_view=self.has_view, sim_time=round(self.cell.sim_time, 2))
        self.tele.set_boxes(instances)

    # ================================================================ RobotPort
    def home(self) -> None:
        """Postura de observación con la pinza como al empezar un episodio (la política se entrenó desde ahí)."""
        r = self.ctrl.move_joints(self.look_q)
        if r.aborted:
            raise Stopped()
        self.ctrl.gripper(self.home_gripper, duration_s=0.3, hold_s=0.05)

    def capacity(self, dest: str) -> int:
        return self.placer.capacity(dest)

    def _occupied(self, dest: str) -> list[tuple[float, float]]:
        insts = self.last.instances if self.last else []
        return [(float(i.pos[0]), float(i.pos[1])) for i in insts if i.region == dest and i.pos is not None]

    def _target(self, inst: Instance) -> tuple[skills.Target, float, float]:
        shape = inst.shape if inst.shape in ("cube", "cylinder") else "cube"
        floor = self.top + (self.bin.floor_m if inst.region == "origen" else 0.0)
        t = skills.Target(float(inst.pos[0]), float(inst.pos[1]), float(inst.yaw), C.PIECE_SIZE_M, C.PIECE_SIZE_M, shape)
        return t, float(inst.pos[2]), floor

    def _plan(self, cand: Candidate, strategy: dict | None = None, check_collisions: bool = True):
        strategy = strategy or {}
        target, center_z, floor = self._target(cand.inst)
        return picking.find_plan(self.base, self.scene, target, center_z, floor, self.ctrl.q(), cand.grasp.yaws_ok,
                                 wall_top_z=self.top + self.bin.outer_m[2],
                                 grasp_lower_m=strategy.get("grasp_lower_m", 0.0), yaw_offset=strategy.get("yaw_offset", 0.0),
                                 check_collisions=check_collisions)

    def reach_relaxed(self, cand: Candidate) -> str | None:
        """Como ``reach`` pero sin exigir que la pinza no roce una pared ni que haya hueco para los dedos (último recurso, lo avisa)."""
        why = self.reach(cand, check_collisions=False)
        if why is None:
            self.tele.event("aviso", f"Último recurso: {cand.track} no cumple la holgura de pared o de dedos que pide el plan "
                                     f"(se agarran ~4 de cada 10 así); lo intento igualmente")
        return why

    def reach(self, cand: Candidate, check_collisions: bool = True) -> str | None:
        plan, why = self._plan(cand, check_collisions=check_collisions)
        if plan is None:
            return why
        occupied = self._occupied(cand.dest)
        slot = self.placer.choose(cand.dest, occupied)
        if slot is None:
            return f"la caja {cand.dest} no tiene un hueco libre alcanzable"
        place = self.placer.plan(cand.dest, slot, plan.lift[-1], occupied)
        if isinstance(place, str):
            return f"el hueco en la caja {cand.dest} no es alcanzable: {place}"
        self._plans[cand.track] = (plan, slot)
        return None

    def pick(self, cand: Candidate, strategy: dict, policy: str) -> PickResult:
        track = cand.track
        self._target_track = track
        if float(np.max(np.abs(self.ctrl.q() - self.look_q))) > 0.05:
            self.home()      # los planes y la política parten de la postura de observación
        plan, slot = self._plans.get(track, (None, None))
        if strategy:                                       # reintento con otra estrategia: se vuelve a planificar
            plan2, _why = self._plan(cand, strategy)
            plan = plan2 or plan
        if plan is None:
            return PickResult(False, "fallo", "no hay plan de agarre", policy)
        if slot is None:
            slot = self.placer.choose(cand.dest, self._occupied(cand.dest))
        self.tele.set(target=track)
        if self.last is not None:
            self._publish(*self.last.frame[:1], self.last.instances, track)

        def on_phase(name: str) -> None:
            self.tele.stage("ACTUAR", f"{name}: {track}")

        if policy == "act":
            if getattr(self.policy, "kind", "") != "act":
                return PickResult(False, "fallo", "no hay política aprendida cargada (--policy act:<carpeta>)", "act")
            seg = O.target_seg_full(self.cam, self.scene, self.last.instances, cand.inst.idx)
            runner = PickRunner(self.cell, self.ctrl._tick, self._grab, self.spec, self.base, self.top)
            out = runner.run(self.policy, plan, seg)
            self.last_pick_stats = out.limiter
            if out.stopped:
                raise Stopped()
            res = PickResult(out.ok, out.status, out.detail, "act", out.steps)
        else:
            out = skills.execute_pick(self.ctrl, plan, lift_speed=strategy.get("lift_speed", skills.LIFT_SPEED),
                                      on_phase=on_phase)
            if out.status == "abortado" and out.detail.startswith("detenido"):
                raise Stopped()
            status = {"vacío": "vacío", "deslizada": "deslizada", "abortado": "abortado"}.get(out.status, "fallo")
            res = PickResult(out.ok, "ok" if out.ok else status, out.detail, "guion")
        if res.ok:
            self._held = (cand, plan, slot)
        else:
            self._back_off(plan)
        return res

    def place(self, cand: Candidate) -> PlaceResult:
        if self._held is None:
            return PlaceResult(False, "deslizada", "no llevaba ninguna pieza")
        held, plan, slot = self._held
        self._held = None
        occupied = self._occupied(cand.dest)
        if slot is None or any(np.hypot(slot[0] - o[0], slot[1] - o[1]) < 0.03 for o in occupied):
            slot = self.placer.choose(cand.dest, occupied)       # el hueco previsto se ocupó mientras tanto
        if slot is None:
            self._back_off(plan)
            return PlaceResult(False, "llena", f"la caja {cand.dest} no tiene hueco")
        place = self.placer.plan(cand.dest, slot, self.ctrl.q(), occupied)
        if isinstance(place, str):
            self._back_off(plan)
            return PlaceResult(False, "inalcanzable", place)

        def on_phase(name: str) -> None:
            self.tele.stage("ACTUAR", f"{name}: {cand.track} → caja {cand.dest}")

        out = skills.execute_place(self.ctrl, plan, place, on_phase=on_phase)
        if out.status == "abortado" and out.detail.startswith("detenido"):
            raise Stopped()
        if out.status == "deslizada":
            self.ctrl.gripper(plan.q_open)
        self.home()          # la cámara tiene que ver el contenedor y la siguiente política arranca siempre desde casa
        return PlaceResult(out.ok, "ok" if out.ok else out.status, out.detail)

    def _back_off(self, plan: skills.PickPlan) -> None:
        """Soltar lo que haya, subir y volver a la postura de observación."""
        self.ctrl.gripper(plan.q_open, duration_s=0.3, hold_s=0.05)
        self.ctrl.move_joints(plan.q_pre, speed=0.8)

    # ================================================================== escenas
    def new_scene(self, spec) -> dict:
        """Reinicia: aparca, deja caer las piezas de ``spec`` (física) y mira."""
        self.tracker.reset()
        self._plans.clear()
        self._held = None
        res = self.bcell.reset_scene(spec)
        self.ctrl.gripper_cmd = self.home_gripper
        self.last = None
        self.sync_clock()
        self.tele.set(seed=spec.seed, scene=res.as_dict())
        return res.as_dict()


# ====================================================================== sesión de Isaac
def _pose_arm(self, q5: np.ndarray, gripper: float = 0.0, settle_s: float = 0.3) -> None:
    """Coloca el brazo sin pasar por la dinámica (sólo para tomar imágenes de etiquetado con el brazo a la vista)."""
    q = np.zeros(len(self.cell.robot.joint_names), dtype=np.float32)
    for i, jid in enumerate(self.cell.arm_ids):
        q[jid] = q5[i]
    q[self.cell.gripper_id] = gripper
    self.cell.write_joint_state(q)
    for _ in range(int(settle_s / self.cell.cfg.physics.dt)):
        self.cell.step()


SimRobot.pose_arm = _pose_arm


def arm_pixels(robot: SimRobot, dilate_px: int = 15) -> np.ndarray:
    """Máscara de los píxeles que tapa el brazo ahora mismo (proyección de sus colisionadores)."""
    return P.robot_pixels(robot.cam, robot._arm_points(), dilate_px=dilate_px)



class Session:
    """Todo lo necesario para una herramienta que usa Isaac: runtime, escena, célula, telemetría y robot."""

    def __init__(self, rt, cfg, scene, bcell, tele, robot):
        self.rt, self.cfg, self.scene, self.bcell, self.tele, self.robot = rt, cfg, scene, bcell, tele, robot


def open_session(args, cfg: C.BinPickCfg | None = None, segmenter: str = "hsv", policy=None, realtime: float = 0.0,
                 view: bool = False) -> Session:
    """Arranca Isaac (``recoverykit.runtime``) y construye la escena y el robot. ``args`` trae --headless, --device, --vulkan."""
    from recoverykit import physics, runtime

    from .cell import BinCell

    rt = runtime.boot(args)
    import carb
    from isaaclab.sim import SimulationContext

    carb.settings.get_settings().set("/rtx/post/aa/op", C.RTX_AA_OP)      # sin acumulación temporal: ver config.OBSERVE_RENDERS

    cfg = cfg or C.BinPickCfg()
    cfg.view_camera = view or cfg.view_camera
    scene = C.build_scene(cfg)
    sim = SimulationContext(physics.sim_cfg(scene))
    bcell = BinCell.build(cfg, sim)
    tele = BinTelemetry()
    robot = SimRobot(bcell, tele, segmenter, policy=policy, realtime=realtime)
    robot.live_frames = False
    return Session(rt, cfg, scene, bcell, tele, robot)
