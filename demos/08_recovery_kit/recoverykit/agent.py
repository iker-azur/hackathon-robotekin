"""El agente RecoveryKit: VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE.

Bucle cerrado. Cada ciclo empieza mirando la mesa desde la postura de
observación y planifica con lo que la cámara confirma dentro de la caja; nada
se da por hecho porque el robot "crea" haberlo hecho.

Lo que el agente usa:
- la cámara cenital RGB-D (``perception``) y la proyección de su propio brazo
  para saber qué tapa;
- la propiocepción de los servos (posición y carga de la pinza);
- nada más. No importa ``audit`` ni lee poses del simulador; la vista 3D del
  panel la publica el lanzador (``viz``), es una ventana a la simulación.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import kinematics as K
from . import kit, perception as P, planner, skills
from .control import Controller
from .supervisor import Failure, Supervisor
from .telemetry import Stopped, Telemetry
from .world_model import WorldModel

CAMERA = "cenital"
WATCH_EVERY_S = 0.25          # comprobación con cámara durante la aproximación
LIVE_FRAME_S = 0.2            # imagen en directo para el panel
POSES_EVERY_S = 0.066         # poses para la vista 3D
MAX_CYCLES = 40
MAX_KIT_UNITS = 4             # huecos de la caja (2×2)
ARM_LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw")
COLOR_BGR = {"red": (40, 40, 230), "blue": (230, 120, 30), "green": (60, 190, 60), "yellow": (40, 220, 230)}


@dataclass
class RunResult:
    status: str                       # "completado" | "incompleto" | "fallido" | "detenido" | "no entendido"
    text: str
    goal: dict = field(default_factory=dict)
    duration_s: float = 0.0
    metrics: dict = field(default_factory=dict)
    verified: list = field(default_factory=list)


class Agent:
    def __init__(self, cell, tele: Telemetry | None = None, perturber=None, realtime: float = 0.0, viz=None):
        """``viz``: función del lanzador que publica la vista 3D (fuera del agente)."""
        self.cell = cell
        self.viz = viz
        self.cfg = cell.cfg
        self.tele = tele or Telemetry()
        self.perturber = perturber
        self.realtime = realtime          # 0 = lo más rápido posible; 1 = tiempo real
        self.ctrl = Controller(cell, monitor=self._monitor)
        cell.step_hooks.append(self._on_step)
        cal = cell.camera_calibration(CAMERA)
        self.cam = P.Camera(cal["k"], cal["rot"], cal["pos"], cal["width"], cal["height"])
        box = self.cfg.containers[0]
        self.exp = P.Expectation(table_z=self.cfg.table.top_z_m,
                                 piece_sizes_m=tuple(sorted({p.size_m for p in self.cfg.pieces})) or (0.025,),
                                 box_outer_m=box.outer_m, box_wall_m=box.wall_m, box_floor_m=box.floor_m,
                                 colors=tuple(sorted({p.color for p in self.cfg.pieces})))
        self.look_q = np.array(self.cfg.robot.start_joints[:5])
        self.reset_memory()
        self._last_live = -1.0
        self._last_poses = -1.0
        self._watch: str | None = None
        self._last_watch = 0.0
        self._wall0 = time.perf_counter()
        self._sim0 = cell.sim_time

    def reset_memory(self) -> None:
        self.world = WorldModel()
        self.sup = Supervisor()
        self.blacklist: set[str] = set()
        self.tasks: list[planner.Task] = []
        self._left_box: list = []
        self.metrics = {"grasp_attempts": 0, "grasps_ok": 0, "places": 0, "places_verified": 0,
                        "failures": {}, "recoveries": 0, "recovered": 0, "looks": 0}

    # ================================================================ percepción
    def _arm_points(self) -> np.ndarray:
        poses = K.fk(dict(zip(K.ARM, self.ctrl.q())), self.ctrl.base)
        pts = []
        for link in ARM_LINKS:
            v = K.colliders()[link][::2]
            t = poses[link]
            pts.append(v @ t[:3, :3].T + t[:3, 3])
        return np.concatenate(pts)

    def _grab(self) -> tuple[np.ndarray, np.ndarray]:
        self.cell.render()
        self.cell.render()
        return self.cell.frame(CAMERA)

    def look(self, why: str = "") -> list:
        """Imagen nueva → detecciones → modelo del mundo. Devuelve los eventos."""
        rgb, depth = self._grab()
        mask = P.robot_pixels(self.cam, self._arm_points(), dilate_px=15)
        obs = P.detect(rgb, depth, self.cam, self.exp, robot_px=mask)

        def visible(p):
            uv, _ = self.cam.project(np.asarray(p)[None])
            u, v = int(uv[0, 0]), int(uv[0, 1])
            return 0 <= u < self.cam.width and 0 <= v < self.cam.height and mask[v, u] == 0

        events = self.world.update(obs, self.cell.sim_time, visible=visible)
        self._left_box += [e for e in events if e.kind == "fuera_de_caja"]
        self.metrics["looks"] += 1
        self._publish_frame(rgb, obs)
        self.tele.set(world=self.world.snapshot())
        for e in events:
            if e.kind != "nueva" or why == "inicio":
                self.tele.event("percepción", e.detail[0].upper() + e.detail[1:], track=e.track, what=e.kind)
        return events

    def _publish_frame(self, rgb: np.ndarray, obs: P.Observation | None) -> None:
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if obs is not None:
            for b in obs.boxes:
                c, s = math.cos(b.yaw), math.sin(b.yaw)
                corners = [[b.center[0] + c * sx * b.inner_m[0] / 2 - s * sy * b.inner_m[1] / 2,
                            b.center[1] + s * sx * b.inner_m[0] / 2 + c * sy * b.inner_m[1] / 2, b.rim_z]
                           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                uv, _ = self.cam.project(np.array(corners))
                cv2.polylines(img, [uv.astype(np.int32)], True, (255, 255, 255), 2)
            for d in obs.resting():
                tr = next((t for t in self.world.tracks.values() if np.linalg.norm(t.pos[:2] - d.pos[:2]) < 0.01), None)
                u, v = int(d.pixel[0]), int(d.pixel[1])
                cv2.circle(img, (u, v), 24, COLOR_BGR.get(d.color, (255, 255, 255)), 2)
                if tr is not None:
                    cv2.putText(img, tr.id, (u + 26, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.tele.push_frame(jpg.tobytes())

    # ============================================================ cada paso
    def _on_step(self) -> None:
        while self.tele.paused and not self.tele.stop_requested:
            time.sleep(0.05)
        t = self.cell.sim_time
        if self.realtime > 0:
            ahead = (t - self._sim0) / self.realtime - (time.perf_counter() - self._wall0)
            if ahead > 0.002:
                time.sleep(ahead)
        if t - self._last_poses >= POSES_EVERY_S:
            self._last_poses = t
            self.tele.set(sim_time=round(t, 2))
            if self.viz is not None:
                self.viz()

    def _monitor(self, info: dict) -> str | None:
        if self.tele.stop_requested:
            return "detenido por el usuario"
        t = self.cell.sim_time
        if self._watch is not None and t - self._last_watch >= WATCH_EVERY_S:
            self._last_watch = t
            for e in self.look("vigilancia"):
                if e.track == self._watch and e.kind in ("movida", "perdida"):
                    return f"{e.kind}|{e.detail}"
        elif self._watch is None and t - self._last_live >= LIVE_FRAME_S:
            self._last_live = t
            rgb, _ = self._grab()
            self._publish_frame(rgb, None)
        return None

    # ============================================================== bucle
    def run(self, text: str) -> RunResult:
        t_start = self.cell.sim_time
        self._wall0, self._sim0 = time.perf_counter(), self.cell.sim_time
        self.tele.clear_stop()
        self.tele.set(status="ejecutando", command=text)
        try:
            return self._run(text, t_start)
        except Stopped:
            self.tele.set(status="detenido")
            self.tele.event("control", "Detenido por el usuario")
            return RunResult("detenido", text, duration_s=self.cell.sim_time - t_start, metrics=self.metrics)

    def _check_stop(self) -> None:
        if self.tele.stop_requested:
            raise Stopped()

    def _run(self, text: str, t_start: float) -> RunResult:
        # ---------------------------------------------------------- ENTENDER
        self.tele.stage("ENTENDER", "interpretando el pedido")
        req = kit.parse(text)
        if req.stop:
            raise Stopped()
        if not req.ok:
            self.tele.set(status="fallido", kit_errors=req.errors)
            for e in req.errors:
                self.tele.event("entender", e)
            return RunResult("no entendido", text)
        units = sum(v for v in req.goal.values() if isinstance(v, int))
        self.tele.set(kit=req.summary(), kit_errors=[])
        self.tele.event("entender", f"Kit pedido: {req.summary()}")
        if units > MAX_KIT_UNITS:
            self.tele.set(status="fallido", kit_errors=["en la caja caben como máximo 4 piezas"])
            self.tele.event("entender", "En la caja caben como máximo 4 piezas separadas; pide un kit más pequeño")
            return RunResult("fallido", text)

        # --------------------------------------------------------------- VER
        self.tele.stage("VER", "mirando la mesa")
        self.ctrl.move_joints(self.look_q)
        self.look("inicio")
        if not self.world.boxes:
            self.tele.set(status="fallido")
            self.tele.event("ver", "No encuentro la caja en la imagen")
            return RunResult("fallido", text)

        status = "incompleto"
        for _ in range(MAX_CYCLES):
            self._check_stop()
            # una pieza ya verificada que ha salido de la caja es un fallo detectado
            # por la cámara; el plan la vuelve a pedir sin regla especial
            for e in self._left_box:
                tr = self.world.tracks[e.track]
                f = Failure("fuera_de_caja", None, tr.id, f"{tr.the()} ya no está dentro de la caja", tr.shape)
                self.tele.stage("RECUPERAR", f.detail)
                d = self.sup.decide(f, self._alternatives(tr.id))
                self.metrics["recoveries"] += 1
                self.metrics["failures"]["fuera_de_caja"] = self.metrics["failures"].get("fuera_de_caja", 0) + 1
                self.tele.recovery({"kind": f.kind, "detail": f.detail, "task": "kit"}, d.as_dict())
                self.tele.event("recuperar", d.reason, action=d.action)
                if d.blacklist:
                    self.blacklist.add(d.blacklist)
            self._left_box.clear()
            # ------------------------------------------------------ PLANIFICAR
            self.tele.stage("PLANIFICAR", "comparando el pedido con lo que hay en la caja")
            kp = planner.plan(req.goal, self.world, self._slots(), self._reach, self.blacklist, self.tasks)
            self._publish_plan(kp)
            if kp.complete:
                # ---------------------------------------- VERIFICAR (final)
                self.tele.stage("VERIFICAR", "comprobación final de la caja")
                self.ctrl.wait(0.4)
                self.look("final")
                kp = planner.plan(req.goal, self.world, self._slots(), self._reach, self.blacklist, self.tasks)
                self._publish_plan(kp)
                if kp.complete:
                    status = "completado"
                    self.tele.event("verificar", f"Kit completo y verificado con la cámara: {req.summary()}")
                    break
                continue
            task = kp.next_task()
            if task is None:
                notes = "; ".join(f"{t.label}: {t.note}" for t in kp.tasks if t.state == "bloqueada")
                self.tele.event("planificar", f"No puedo completar el kit — {notes}")
                break
            if task.kind == "retirar":
                self.tele.event("planificar", f"En la caja sobra {self.world.tracks[task.track].the()}; lo saco")
            failure = self._do(task)
            if failure is None:
                continue
            # ------------------------------------------------------- RECUPERAR
            self.tele.stage("RECUPERAR", failure.detail)
            decision = self.sup.decide(failure, self._alternatives(failure.track))
            self.metrics["recoveries"] += 1
            self.metrics["failures"][failure.kind] = self.metrics["failures"].get(failure.kind, 0) + 1
            self.tele.recovery({"kind": failure.kind, "detail": failure.detail, "task": task.label}, decision.as_dict())
            self.tele.event("recuperar", decision.reason, action=decision.action)
            if decision.blacklist:
                self.blacklist.add(decision.blacklist)
            if decision.action == "abandonar":
                task.state = "fallida"
                self._publish_tasks()
                status = "fallido"
                break
            self.ctrl.move_joints(self.look_q)
            self.look("recuperación")

        verified = [t.as_dict() for t in self.world.tracks.values() if t.state == "caja"]
        duration = self.cell.sim_time - t_start
        self.metrics["duration_s"] = round(duration, 2)
        self.tele.set(status=status, metrics=self.metrics)
        self.tele.stage(None, "")
        return RunResult(status, text, {f"{k[0]}:{k[1]}": v for k, v in req.goal.items()}, duration, self.metrics,
                         verified)

    # ============================================================ una tarea
    def _ctx(self, tr, pick: skills.PickPlan) -> dict:
        return {"target_xy": (float(tr.pos[0]), float(tr.pos[1])), "yaw": pick.yaw, "q_open": pick.q_open}

    def _do(self, task: planner.Task) -> Failure | None:
        tr = self.world.tracks[task.track]
        strat = self.sup.strategy_for(tr.id, tr.shape)
        size = min(self.exp.piece_sizes_m, key=lambda s: abs(s - tr.width)) if tr.width > 0 else self.exp.piece_sizes_m[0]
        height = size if tr.height < 0.5 * size else tr.height
        shape = tr.shape if tr.shape != "?" else "cube"
        target = skills.Target(float(tr.pos[0]), float(tr.pos[1]), tr.yaw, size, height, shape)
        support = self.world.boxes[0].floor_z if tr.state == "caja" and self.world.boxes else self.cfg.table.top_z_m
        pick = skills.plan_pick(self.ctrl.base, support, target, self.ctrl.q(),
                                grasp_lower_m=strat.get("grasp_lower_m", 0.0), yaw_offset=strat.get("yaw_offset", 0.0))
        if isinstance(pick, str):
            return Failure("inalcanzable", task.id, tr.id, f"{tr.the()}: {pick}", tr.shape)
        # la colocación se planifica ANTES de coger: desde la postura del final de la
        # subida, primero el hueco asignado y luego cualquier otro libre
        place = self._plan_place(task, pick, height, shape)
        if isinstance(place, str):
            return Failure("inalcanzable", task.id, tr.id, f"{tr.the()}: {place}", tr.shape)
        task.state, task.attempts = "ejecutando", task.attempts + 1
        self._publish_tasks()
        self.metrics["grasp_attempts"] += 1
        self.tele.stage("ACTUAR", f"cogiendo {tr.the()}")
        self.tele.event("actuar", f"Voy a por {tr.the()} ({task.label})"
                        + (f" — estrategia: {self._strategy_text(strat)}" if strat else ""))

        def on_pick_phase(phase: str) -> None:
            self.tele.stage("ACTUAR", f"{phase}: {tr.label()}")
            self._watch = tr.id if phase == "aproximación" else None
            self._last_watch = self.cell.sim_time
            self._notify_world(phase, self._ctx(tr, pick))

        out = skills.execute_pick(self.ctrl, pick, lift_speed=strat.get("lift_speed", skills.LIFT_SPEED),
                                  on_phase=on_pick_phase)
        self._watch = None
        if out.status == "abortado":
            if out.detail.startswith("detenido"):
                raise Stopped()
            kind, _, detail = out.detail.partition("|")
            self._back_off(pick)
            task.state = "pendiente"
            self.tele.event("percepción", f"Detectado durante la aproximación: {detail}")
            return Failure("movida" if kind == "movida" else "perdida", task.id, tr.id, detail, tr.shape)
        if not out.ok:
            self.tele.stage("VERIFICAR", "comprobando el agarre")
            q = f" (pinza en {out.gripper_q:.2f} rad, contacto esperado en {pick.q_contact:.2f})" \
                if out.gripper_q is not None else ""
            self.tele.event("verificar", f"Agarre fallido: {out.detail}{q}")
            self._back_off(pick)
            task.state = "pendiente"
            kind = out.status if out.status in ("vacío", "deslizada") else "movimiento"
            return Failure(kind, task.id, tr.id, out.detail, tr.shape)
        self.metrics["grasps_ok"] += 1
        self.world.grasped(tr.id)
        self.tele.stage("VERIFICAR", "agarre confirmado")
        self.tele.event("verificar", f"Agarre confirmado: la pinza se ha bloqueado sobre la pieza ({out.gripper_q:.2f} rad)")

        # ------------------------------------------------------------ dejar
        def on_place_phase(phase: str) -> None:
            self.tele.stage("ACTUAR", f"{phase}: {tr.label()}")
            self._notify_world(phase, self._ctx(tr, pick))

        out2 = skills.execute_place(self.ctrl, pick, place, on_phase=on_place_phase)
        if out2.status == "abortado" and out2.detail.startswith("detenido"):
            raise Stopped()
        if out2.status == "deslizada":
            self.tele.event("verificar", "La pinza ha perdido la pieza durante el transporte")
            self.ctrl.gripper(pick.q_open)
        self.world.released()
        self.metrics["places"] += 1
        self.ctrl.move_joints(self.look_q)
        self.tele.stage("VERIFICAR", "mirando dónde ha quedado la pieza")
        self.look("verificación")
        self._notify_world("colocada", {"target_xy": (float(tr.pos[0]), float(tr.pos[1]))})
        tr = self.world.tracks[tr.id]
        ok = tr.state == ("mesa" if task.kind == "retirar" else "caja")
        if ok:
            self.metrics["places_verified"] += 1
            task.state = "completada"
            self._publish_tasks()
            where = "fuera de la caja" if task.kind == "retirar" else "dentro de la caja"
            self.tele.event("verificar", f"Verificado con la cámara: {tr.the()} está {where}")
            if self.sup.grasp_attempts.get(tr.id) or self.sup.place_attempts.get(tr.id):
                self.metrics["recovered"] += 1
            return None
        task.state = "pendiente"
        if tr.state == "mesa":
            why = "se soltó durante el transporte y " if out2.status == "deslizada" else ""
            return Failure("fuera_de_caja", task.id, tr.id, f"{tr.the()} {why}ha quedado fuera de la caja", tr.shape)
        return Failure("perdida", task.id, tr.id, f"no encuentro {tr.the()} después de soltarlo", tr.shape)

    # ============================================================ ayudas
    def _notify_world(self, phase: str, ctx: dict) -> None:
        """Avisa al perturbador (parte del mundo) de la fase; el agente no recibe nada de vuelta."""
        if self.perturber is None:
            return
        self.perturber.notify(phase, ctx)
        self.tele.set(armed_perturbations=self.perturber.pending(), perturbations=list(self.perturber.log))

    def _plan_place(self, task: planner.Task, pick: skills.PickPlan, height: float, shape: str):
        top = self.cfg.table.top_z_m
        seed = pick.lift[-1]
        if task.kind == "retirar":
            return skills.plan_place(self.ctrl.base, top, self._free_table_spot(), top, height, pick.local,
                                     skills._yaw_candidates(0.0, shape), seed)
        box = self.world.boxes[0]
        slots = self._slots()
        busy = [t.pos[:2] for t in self.world.tracks.values() if t.state == "caja"]
        others = [i for i in range(len(slots)) if i != task.slot
                  and all(np.hypot(slots[i][0] - p[0], slots[i][1] - p[1]) > 0.025 for p in busy)]
        last = "no hay hueco libre en la caja"
        for i in [task.slot, *others]:
            if i is None:
                continue
            place = skills.plan_place(self.ctrl.base, top, slots[i], box.floor_z, height, pick.local,
                                      skills._yaw_candidates(box.yaw, shape), seed)
            if not isinstance(place, str):
                task.slot = i
                return place
            last = place
        return last

    def _back_off(self, pick: skills.PickPlan) -> None:
        """Abrir, subir y volver a la postura de observación."""
        self.ctrl.gripper(pick.q_open, duration_s=0.3, hold_s=0.05)
        self.ctrl.move_joints(pick.q_pre, speed=0.8)
        self.ctrl.move_joints(self.look_q)

    def _slots(self) -> list[tuple[float, float]]:
        if self.world.boxes:
            b = self.world.boxes[0]
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            out = []
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                lx, ly = dx * b.inner_m[0] / 4, dy * b.inner_m[1] / 4
                out.append((float(b.center[0] + c * lx - s * ly), float(b.center[1] + s * lx + c * ly)))
            return out
        box = self.cfg.containers[0]
        return skills.box_slots(box.center_xy_m, box.inner_m)

    def _reach(self, tr) -> str | None:
        size = self.exp.piece_sizes_m[0]
        support = self.world.boxes[0].floor_z if tr.state == "caja" and self.world.boxes else self.cfg.table.top_z_m
        target = skills.Target(float(tr.pos[0]), float(tr.pos[1]), tr.yaw, size, size,
                               tr.shape if tr.shape != "?" else "cube")
        p = skills.plan_pick(self.ctrl.base, support, target, self.look_q)
        return p if isinstance(p, str) else None

    def _alternatives(self, track: str | None) -> int:
        if track is None or track not in self.world.tracks:
            return 0
        me = self.world.tracks[track]
        return sum(1 for t in self.world.tracks.values()
                   if t.id != track and t.state == "mesa" and t.id not in self.blacklist
                   and t.color == me.color and (t.shape == me.shape or "?" in (t.shape, me.shape)))

    def _free_table_spot(self) -> tuple[float, float]:
        area = self.cfg.work_area
        best, best_d = (0.18, -0.08), -1.0
        for r in np.linspace(area.r_m[0] + 0.01, area.r_m[1] - 0.01, 3):
            for a in np.linspace(area.angle_deg[0], area.angle_deg[1], 9):
                x, y = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
                d = min((float(np.hypot(x - t.pos[0], y - t.pos[1])) for t in self.world.tracks.values()), default=1.0)
                if d > best_d:
                    best, best_d = (float(x), float(y)), d
        return best

    @staticmethod
    def _strategy_text(s: dict) -> str:
        parts = []
        if "yaw_offset" in s:
            parts.append("pinza girada")
        if "grasp_lower_m" in s:
            parts.append(f"{s['grasp_lower_m']*1000:.0f} mm más abajo")
        if "lift_speed" in s:
            parts.append("subida lenta")
        return ", ".join(parts)

    def _publish_tasks(self) -> None:
        self.tele.set(tasks=[t.as_dict() for t in self.tasks])

    def _publish_plan(self, kp: planner.KitPlan) -> None:
        self.tasks = kp.tasks
        self._publish_tasks()
        self.tele.set(kit_status={"required": len(kp.units), "verified": len(kp.covered),
                                  "complete": kp.complete, "extras": kp.extras})
