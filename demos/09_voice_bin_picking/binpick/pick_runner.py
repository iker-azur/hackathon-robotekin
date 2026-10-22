"""Ejecuta una política de agarre contra el simulador a 30 Hz: observar → decidir → limitar → avanzar la física.

Un paso de control = ``STEPS_PER_FRAME`` pasos de física (240/30). Se reutiliza para evaluar ACT y
para cualquier ``PickPolicy`` (``policy.CallablePolicy`` en las pruebas). El runner no sabe de Isaac:
recibe ``cell`` (``proprio``, ``set_arm_targets``, ``set_gripper_target``, ``sim_time``), una función que
avanza un paso de física (``step``) y otra que da la imagen (``grab``). Así se prueba con un robot falso.

El éxito se decide con lo que un SO-101 real también sabría: la altura del punto de agarre (cinemática
directa de las articulaciones leídas) y la posición de la pinza (cerrada sobre la pieza o en vacío). No
se mira dónde está la pieza en el simulador; eso lo hace el auditor aparte.

Causas de fallo: ``no llegó`` (nunca bajó a la pieza), ``vacío`` (cerró sin pieza), ``no levantó``
(la sujetó pero no subió), ``deslizada`` (la soltó al subir), ``tiempo`` (se acabó el presupuesto).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import config as C
from . import joints
from . import observation as O
from . import picking
from .policy import ActionLimiter, PickPolicy
from recoverykit import skills  # noqa: E402

MAX_PICK_STEPS = 270             # 9 s a 30 Hz: el experto tarda ~6 s
HOLD_FRAMES = 6                  # fotogramas seguidos arriba y sujeta para dar el agarre por bueno
LIFT_FRACTION = 0.8              # parte de la subida que hay que completar
DESCENDED_M = 0.012              # el punto de agarre está a menos de esto de la altura de agarre
OVER_PIECE_M = 0.015             # y a menos de esto de la pieza en el plano


@dataclass
class PickOutcome:
    ok: bool
    status: str = "ok"
    detail: str = ""
    steps: int = 0
    limiter: dict = field(default_factory=dict)
    max_height_m: float = 0.0
    stopped: bool = False


class PickRunner:
    def __init__(self, cell, step: Callable[[int], str | None], grab: Callable[[], tuple[np.ndarray, np.ndarray]],
                 spec: O.ObsSpec, base: np.ndarray, table_z: float, steps_per_frame: int = C.STEPS_PER_FRAME,
                 max_steps: int = MAX_PICK_STEPS, on_frame: Callable[[dict], None] | None = None):
        self.cell, self.step, self.grab = cell, step, grab
        self.spec, self.base, self.table_z = spec, base, table_z
        self.spf, self.max_steps, self.on_frame = steps_per_frame, max_steps, on_frame
        self._n = 0

    def _clearance(self, q5: np.ndarray) -> float:
        return skills.min_clearance(self.base, q5, self.table_z)

    def run(self, policy: PickPolicy, plan: skills.PickPlan, seg_full: np.ndarray) -> PickOutcome:
        grasp_h = picking.tcp_height(self.base, plan.descent[-1], plan.local)
        lift_h = picking.tcp_height(self.base, plan.lift[-1], plan.local)
        need_h = grasp_h + LIFT_FRACTION * (lift_h - grasp_h)
        pro = self.cell.proprio()
        limiter = ActionLimiter(clearance_fn=self._clearance)
        limiter.reset(joints.join(pro["arm_q"], pro["gripper_q"]))
        policy.reset()
        held_frames, was_held_up, descended, closed_empty = 0, False, False, False
        max_h = -1e9
        for k in range(self.max_steps):
            rgb, _depth = self.grab()
            pro = self.cell.proprio()
            obs = O.build_obs(rgb, seg_full, pro["arm_q"], pro["gripper_q"], self.spec)
            safe = limiter(policy.act(obs))
            self.cell.set_arm_targets(safe[:5])
            self.cell.set_gripper_target(float(safe[5]))
            reason = None
            for _ in range(self.spf):
                reason = self.step(self._n)
                self._n += 1
                if reason:
                    break
            if reason:
                return PickOutcome(False, "abortado", reason, k + 1, limiter.stats(), max_h, stopped=reason.startswith("detenido"))
            pro = self.cell.proprio()
            h = picking.tcp_height(self.base, pro["arm_q"], plan.local)
            max_h = max(max_h, h)
            over = float(np.linalg.norm(picking.tcp_xy(self.base, pro["arm_q"], plan.local)
                                        - np.array([plan.target.x, plan.target.y]))) < OVER_PIECE_M
            descended = descended or (over and h <= grasp_h + DESCENDED_M)   # bajar *encima de la pieza*, no cruzar esa altura en tránsito
            closing = float(safe[5]) < plan.q_contact          # la política está ordenando cerrar
            # «sujeta» sólo cuenta tras haber bajado a la pieza y ordenado cerrar: a la altura de casa la pinza
            # cerrada en vacío (0 rad) también está cerca del contacto y no puede dar por bueno un agarre
            held = skills.held(pro["gripper_q"], plan) and descended and closing
            if descended and not held and pro["gripper_q"] < plan.q_contact - skills.HELD_TOLERANCE_RAD:
                closed_empty = True
            if self.on_frame:
                self.on_frame({"k": k, "h": h, "held": held, "gripper_q": pro["gripper_q"]})
            if h >= need_h and held:
                held_frames += 1
                was_held_up = True
                if held_frames >= HOLD_FRAMES:
                    return PickOutcome(True, "ok", "", k + 1, limiter.stats(), max_h)
            else:
                held_frames = 0
                if was_held_up and not held:
                    return PickOutcome(False, "deslizada", "la pieza se soltó al subir", k + 1, limiter.stats(), max_h)
        st = limiter.stats()
        if not descended:
            return PickOutcome(False, "fallo", "no llegó a la pieza (la política no bajó)", self.max_steps, st, max_h)
        if closed_empty:
            return PickOutcome(False, "vacío", "cerró la pinza sin pieza", self.max_steps, st, max_h)
        return PickOutcome(False, "fallo", "sujetó la pieza pero no la levantó a tiempo", self.max_steps, st, max_h)
