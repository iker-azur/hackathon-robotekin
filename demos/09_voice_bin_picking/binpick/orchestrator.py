"""La tarea de clasificar: una máquina de estados con recuperación (sin Isaac).

    ENTENDER → VER → PLANIFICAR → ACTUAR (coger) → VERIFICAR el agarre → ACTUAR (dejar)
                 ▲                                                          │
                 └──── RECUPERAR ◄─────────── VERIFICAR con la cámara ◄─────┘

Habla con el robot a través de un ``RobotPort`` (``observe``, ``reach``, ``pick``, ``place``,
``capacity``, ``home``), así que la misma lógica corre con el robot simulado en Isaac
(``robot_port.py``) y con el ``FakeRobot`` de las pruebas. Esa separación es lo que permite probar
la lógica (selección, reintentos, cajas llenas, órdenes con cantidad) sin simulador, y es también
la costura por la que entraría un SO-101 real.

Principios:

* **Nunca se da por hecho.** Cada pieza se verifica con la cámara después de dejarla (su pista
  tiene que aparecer en la caja que toca); si no, es un fallo y se vuelve a mirar.
* **Cada ciclo vuelve a observar.** La escena cambia (la pieza de debajo se mueve cuando se saca la
  de arriba), así que no se planifica más de una pieza por imagen.
* **La política se nombra.** Cada intento queda registrado como ``act`` (política aprendida) o
  ``guion`` (el experto programado, que NO es aprendizaje); el informe cuenta ambos por separado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import commands as K
from . import selection as SEL
from .instances import Instance
from recoverykit.supervisor import Failure, Supervisor  # noqa: E402

STAGES = ("ENTENDER", "VER", "PLANIFICAR", "ACTUAR", "VERIFICAR", "RECUPERAR")
POLICY_MODES = ("act", "guion", "act+guion")


class Stopped(Exception):
    """El usuario ha pedido parar."""


@dataclass
class Observation:
    instances: list[Instance]
    frame: object = None


@dataclass
class PickResult:
    ok: bool
    status: str = "ok"            # ok | vacío | deslizada | movida | abortado | fallo
    detail: str = ""
    policy: str = "guion"         # act | guion
    steps: int = 0


@dataclass
class PlaceResult:
    ok: bool
    status: str = "ok"            # ok | deslizada | abortado | inalcanzable | llena
    detail: str = ""


class RobotPort(Protocol):
    def observe(self, why: str) -> Observation: ...
    def reach(self, cand: SEL.Candidate) -> str | None: ...
    def pick(self, cand: SEL.Candidate, strategy: dict, policy: str) -> PickResult: ...
    def place(self, cand: SEL.Candidate) -> PlaceResult: ...
    def capacity(self, dest: str) -> int: ...
    def home(self) -> None: ...


class NullSink:
    def stage(self, name, detail=""): ...
    def event(self, kind, text, **data): ...
    def set(self, **kw): ...


@dataclass
class PieceRecord:
    track: str
    color: str | None
    shape: str | None
    dest: str
    attempts: int = 0
    policies: list[str] = field(default_factory=list)
    ok: bool = False
    detail: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SortReport:
    status: str                    # completado | incompleto | fallido | detenido | no entendido | nada que hacer
    text: str
    summary: str = ""
    records: list[PieceRecord] = field(default_factory=list)
    moved: int = 0
    remaining: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    recoveries: int = 0
    recovered: int = 0
    picks: dict[str, int] = field(default_factory=lambda: {"act": 0, "guion": 0})
    picks_ok: dict[str, int] = field(default_factory=lambda: {"act": 0, "guion": 0})
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["records"] = [r.as_dict() for r in self.records]
        return d


class SortingTask:
    def __init__(self, port: RobotPort, scene, sink=None, policy_mode: str = "guion", act_attempts: int = 2,
                 max_cycles: int = 40, stop: Callable[[], bool] = lambda: False,
                 clock: Callable[[], float] = lambda: 0.0):
        if policy_mode not in POLICY_MODES:
            raise ValueError(f"modo de política desconocido: {policy_mode} (usa {', '.join(POLICY_MODES)})")
        self.port, self.scene = port, scene
        self.sink = sink or NullSink()
        self.policy_mode, self.act_attempts, self.max_cycles = policy_mode, act_attempts, max_cycles
        # fallos de ACT seguidos (con cualquier pieza) a partir de los cuales ya no se confía en ella en esta orden
        self.act_streak_limit = act_attempts
        self._act_streak = 0
        self.stop, self.clock = stop, clock

    # ------------------------------------------------------------------ ayudas
    def _check_stop(self) -> None:
        if self.stop():
            raise Stopped()

    def _policy_for(self, track: str, attempts: dict[str, int]) -> str:
        if self.policy_mode == "guion":
            return "guion"
        if self.policy_mode == "act":
            return "act"
        # Medido en Isaac: con varias piezas ACT falla alternando entre ellas y el supervisor da la orden por fallida a los
        # 3 fallos seguidos, así que ninguna pieza llegaba a su 3.er intento (el del guion). Una política que falla
        # `act_streak_limit` veces seguidas no está funcionando, sea cual sea la pieza.
        if self._act_streak >= self.act_streak_limit:
            return "guion"
        return "act" if attempts.get(track, 0) < self.act_attempts else "guion"

    def _load(self, obs: Observation) -> dict[str, int]:
        out = {"izquierda": 0, "derecha": 0}
        for i in obs.instances:
            if i.region in out:
                out[i.region] += 1
        return out

    def _full(self, obs: Observation) -> set[str]:
        load = self._load(obs)
        return {d for d, n in load.items() if n >= self.port.capacity(d)}

    # ---------------------------------------------------------------- bucle
    def run(self, order: K.SortOrder) -> SortReport:
        t0 = self.clock()
        self._act_streak = 0
        report = SortReport(status="incompleto", text=order.text, summary=order.summary(), warnings=list(order.warnings))
        try:
            self._run(order, report)
        except Stopped:
            report.status = "detenido"
            self.sink.event("control", "Detenido por el usuario")
        report.duration_s = self.clock() - t0
        self.sink.set(report=report.as_dict())
        self.sink.stage(None, "")
        return report

    def _run(self, order: K.SortOrder, rep: SortReport) -> None:
        self.sink.stage("ENTENDER", "interpretando la orden")
        if order.intent == "stop":
            raise Stopped()
        if not order.ok:
            rep.status = "no entendido"
            for e in order.errors:
                self.sink.event("entender", e)
            return
        self.sink.event("entender", f"Entendido: {order.summary()}")
        for w in order.warnings:
            self.sink.event("entender", w)

        self.sink.stage("VER", "mirando el contenedor")
        self.port.home()
        obs = self.port.observe("inicio")
        in_bin = [i for i in obs.instances if i.region in ("origen", "mesa")]
        self.sink.event("ver", f"Veo {len(in_bin)} piezas por clasificar: " + self._census(in_bin))
        pieces = [(i.color, i.shape, i.attributes) for i in in_bin]
        for r in order.rules:
            if not any(r.selector.rest or r.selector.matches(*p) for p in pieces):
                note = (f"No veo piezas que cumplan «{r.selector.describe()}» en el contenedor "
                        f"(una pieza totalmente tapada por otras no se puede ver).")
                rep.notes.append(note)
                self.sink.event("ver", note)
        load = self._load(obs)
        for w in K.capacity_warnings(order, pieces, min(self.port.capacity("izquierda"), self.port.capacity("derecha")),
                                     already=load):
            rep.warnings.append(w)
            self.sink.event("planificar", w)

        done: dict[int, int] = {}
        attempts: dict[str, int] = {}
        records: dict[str, PieceRecord] = {}
        sup = Supervisor()
        blacklist: set[str] = set()
        problem_pieces: set[str] = set()

        for _cycle in range(self.max_cycles):
            self._check_stop()
            self.sink.stage("PLANIFICAR", "eligiendo la siguiente pieza")
            sel = SEL.rank(obs.instances, order, self.scene, done, blacklist, attempts, self._full(obs))
            for n in sel.notes:
                if n not in rep.notes:
                    rep.notes.append(n)
                    self.sink.event("planificar", n)
            if sel.pending == 0:
                rep.status = "completado"
                break
            cand, why = SEL.first_reachable(sel, self.port.reach, getattr(self.port, "reach_relaxed", None))
            if cand is None:
                reasons = [f"{c.track}: {c.blocked}" for c in sel.blocked] + why
                msg = "No puedo coger las piezas que quedan — " + ("; ".join(reasons) if reasons else "ninguna es candidata")
                rep.notes.append(msg)
                self.sink.event("planificar", msg)
                rep.status = "incompleto"
                break
            track = cand.track
            rec = records.setdefault(track, PieceRecord(track, cand.inst.color, cand.inst.shape, cand.dest))
            policy = self._policy_for(track, attempts)
            attempts[track] = attempts.get(track, 0) + 1
            rec.attempts = attempts[track]
            rec.policies.append(policy)
            strategy = sup.strategy_for(track, cand.inst.shape or "cube")
            self.sink.event("planificar", f"Voy a por {cand.inst.label()} ({cand.why()}) → caja {cand.dest}", track=track)
            self.sink.stage("ACTUAR", f"cogiendo {track} con {'la política aprendida (ACT)' if policy == 'act' else 'el guion'}")
            pick = self.port.pick(cand, strategy, policy)
            rep.picks[pick.policy] = rep.picks.get(pick.policy, 0) + 1
            if pick.policy == "act":
                self._act_streak = 0 if pick.ok else self._act_streak + 1
            self._check_stop()
            if not pick.ok:
                self.sink.stage("VERIFICAR", "comprobando el agarre")
                self.sink.event("verificar", f"Agarre fallido ({pick.policy}): {pick.status}. {pick.detail}", track=track)
                obs = self._recover(Failure(self._kind(pick.status, pick.policy), None, track, pick.detail or pick.status,
                                            cand.inst.shape or "cube"), sup, rep, blacklist, sel, cand, problem_pieces)
                if rep.status == "fallido":
                    break
                continue
            rep.picks_ok[pick.policy] = rep.picks_ok.get(pick.policy, 0) + 1
            self.sink.event("verificar", f"Agarre confirmado ({pick.policy})", track=track)
            self.sink.stage("ACTUAR", f"llevando {track} a la caja {cand.dest}")
            place = self.port.place(cand)
            self._check_stop()
            self.sink.stage("VERIFICAR", "mirando dónde ha quedado la pieza")
            obs = self.port.observe("verificación")
            where = next((i.region for i in obs.instances if i.track == track), None)
            if where == cand.dest:
                done[cand.rule] = done.get(cand.rule, 0) + 1
                rec.ok = True
                rep.moved += 1
                if sup.grasp_attempts.get(track) or sup.place_attempts.get(track) or attempts[track] > 1:
                    rep.recovered += 1
                self.sink.event("verificar", f"Verificado con la cámara: {track} está en la caja {cand.dest}", track=track)
                continue
            kind = "fuera_de_caja" if where == "mesa" else ("deslizada" if where == "origen" else "perdida")
            detail = {"mesa": "ha caído fuera de las cajas", "origen": "ha vuelto al contenedor",
                      None: "no la encuentro después de soltarla"}.get(where, f"está en {where}")
            if place.status != "ok":
                detail = f"{detail} ({place.status}: {place.detail})" if place.detail else detail
            obs = self._recover(Failure(kind, None, track, f"{track} {detail}", cand.inst.shape or "cube"),
                                sup, rep, blacklist, sel, cand, problem_pieces)
            if rep.status == "fallido":
                break
        else:
            rep.notes.append("Se alcanzó el máximo de ciclos sin terminar.")

        self.sink.stage("VERIFICAR", "comprobación final")
        self.port.home()
        final = self.port.observe("final")
        sel = SEL.rank(final.instances, order, self.scene, done, blacklist, attempts, self._full(final))
        rep.remaining = sel.pending
        if rep.status in ("completado", "incompleto"):
            rep.status = "completado" if sel.pending == 0 else "incompleto"
            if rep.status == "completado" and rep.moved == 0:
                rep.status = "nada que hacer"
        rep.records = list(records.values())
        if rep.status == "completado":
            self.sink.event("verificar", f"Orden completada según la cámara: {rep.moved} piezas movidas y no veo más que cumplan "
                                         f"la orden (una pieza totalmente tapada por otras no se puede ver)")
        elif rep.status == "nada que hacer":
            self.sink.event("verificar", "No veo piezas que cumplan la orden, así que no hay nada que mover "
                                         "(una pieza totalmente tapada por otras no se puede ver)")
        else:
            self.sink.event("verificar", f"Orden {rep.status}: {rep.moved} movidas, {rep.remaining} sin mover")

    # ------------------------------------------------------------ recuperación
    @staticmethod
    def _kind(status: str, policy: str = "guion") -> str:
        kind = {"vacío": "vacío", "deslizada": "deslizada", "movida": "movida", "abortado": "movimiento"}.get(status)
        if kind:
            return kind
        # «no llegó / no bajó» de la política aprendida es un fallo de la política, no una avería del brazo: con «movimiento»
        # el supervisor abandonaba la pieza a los 2 fallos y el guion de respaldo nunca llegaba a intentarla (medido en Isaac)
        return "vacío" if policy == "act" and status == "fallo" else "movimiento"

    def _recover(self, f: Failure, sup: Supervisor, rep: SortReport, blacklist: set[str], sel: SEL.Selection,
                 cand: SEL.Candidate, problem_pieces: set[str]) -> Observation:
        self.sink.stage("RECUPERAR", f.detail)
        alternatives = max(0, len(sel.usable) - 1)
        d = sup.decide(f, alternatives)
        rep.recoveries += 1
        rep.failures[f.kind] = rep.failures.get(f.kind, 0) + 1
        problem_pieces.add(f.track or "")
        self.sink.event("recuperar", d.reason, action=d.action, track=f.track)
        if d.blacklist:
            blacklist.add(d.blacklist)
        if d.action == "abandonar":
            rep.status = "fallido"
        self.port.home()
        return self.port.observe("recuperación")

    @staticmethod
    def _census(insts: list[Instance]) -> str:
        n: dict[str, int] = {}
        for i in insts:
            n[i.label()] = n.get(i.label(), 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(n.items())) or "ninguna"
