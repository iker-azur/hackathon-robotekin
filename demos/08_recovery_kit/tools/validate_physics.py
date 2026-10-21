"""Fase 2: mide el comportamiento físico real de la célula.

    python tools/validate_physics.py --headless

No comprueba valores de configuración: mueve cosas y mide lo que pasa.
Escribe ``results/physics_validation.json`` con cada medida y si pasa.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    runtime.add_args(parser)
    parser.add_argument("--out", default=str(ROOT / "results" / "physics_validation.json"))
    args = parser.parse_args()
    rt = runtime.boot(args)

    from isaaclab.sim import SimulationContext

    from recoverykit import config, physics, workcell
    from recoverykit.config import PieceCfg
    from recoverykit.control import Controller, trapezoid

    cfg = config.preset("simple")
    cfg.pieces = [PieceCfg("rojo_1", "red", pos_xy_m=(0.19, -0.06), yaw_deg=0.0),
                  PieceCfg("azul_1", "blue", pos_xy_m=(0.19, 0.0), yaw_deg=20.0),
                  PieceCfg("verde_1", "green", shape="cylinder", pos_xy_m=(0.22, -0.12), yaw_deg=0.0)]
    sim = SimulationContext(physics.sim_cfg(cfg))
    cell = workcell.build(cfg, sim)
    top = cfg.table.top_z_m
    dt = cfg.physics.dt
    res: dict = {"generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                 "config": cfg.to_dict()["physics"], "servo_cfg": cfg.to_dict()["robot"]["arm_servo"]}
    checks: list[dict] = []

    def check(name, value, limit, ok):
        checks.append({"name": name, "value": value, "limit": limit, "pass": bool(ok)})
        print(f"[física] {'OK ' if ok else 'MAL'} {name}: {value} (límite {limit})", flush=True)

    layout = {p.id: (p.pos_xy_m[0], p.pos_xy_m[1], np.radians(p.yaw_deg)) for p in cfg.pieces}
    cell.reset_layout(layout)

    # ---------------------------------------------------------------- caída
    h = cfg.pieces[0].height
    cell.place_for_test("rojo_1", 0.19, -0.06, top + h / 2 + 0.10, 0.0)
    cell.step()
    cell.arm_watchdog()
    zs, speeds = [], []
    for _ in range(int(2.0 / dt)):
        cell.step()
        t = cell.truth("rojo_1")
        zs.append(float(t["pos"][2]))
        speeds.append(float(np.linalg.norm(t["lin_vel"])))
    zs, speeds = np.array(zs), np.array(speeds)
    rest_z = top + h / 2
    first_contact = int(np.argmax(zs < rest_z + 0.001))
    rebound = float(zs[first_contact:].max() - rest_z) if first_contact > 0 else 0.0
    moving = np.where(speeds > 0.001)[0]
    rest_time = float((moving[-1] + 1) * dt) if len(moving) else 0.0
    penetration = float(rest_z - zs[-1])
    # altura de rebote ∝ e² con la restitución combinada (promedio)
    e = (cfg.physics.piece_material.restitution + cfg.table.material.restitution) / 2
    res["drop"] = {"height_m": 0.10, "rest_time_s": rest_time, "penetration_m": penetration,
                   "rebound_m": rebound, "expected_rebound_m": 0.10 * e * e, "restitution_combined": e,
                   "final_quat": cell.truth("rojo_1")["quat"].tolist()}
    check("caída: reposo en < 1 s", round(rest_time, 3), 1.0, rest_time < 1.0)
    check("caída: penetración < 1 mm", round(penetration * 1000, 3), 1.0, abs(penetration) < 0.001)
    check("caída: rebote ≤ e²·h + 2 mm", round(rebound * 1000, 2), round((0.10 * e * e + 0.002) * 1000, 2),
          rebound <= 0.10 * e * e + 0.002)

    # ------------------------------------------------------------ fricción
    cell.place_for_test("azul_1", 0.19, 0.0, top + h / 2 + 0.0005, 0.0)
    for _ in range(int(0.5 / dt)):
        cell.step()
    cell.arm_watchdog()
    m = cfg.pieces[1].mass_kg
    g = 9.81
    start = cell.truth("azul_1")["pos"].copy()
    f_onset = None
    ramp_s = 3.0
    f_max = 0.8 * m * g
    for n in range(int(ramp_s / dt)):
        f = f_max * n * dt / ramp_s
        cell.apply_force("azul_1", (f, 0.0, 0.0))
        cell.step()
        if np.linalg.norm(cell.truth("azul_1")["pos"][:2] - start[:2]) > 0.0005:
            f_onset = f
            break
    cell.clear_force("azul_1")
    for _ in range(int(0.5 / dt)):
        cell.step()
    mu_s_expected = (cfg.physics.piece_material.static_friction + cfg.table.material.static_friction) / 2
    mu_d_expected = (cfg.physics.piece_material.dynamic_friction + cfg.table.material.dynamic_friction) / 2
    mu_s = f_onset / (m * g) if f_onset is not None else None
    # dinámica: fuerza constante hacia +x (mesa libre; en +y está la caja),
    # aceleración medida en régimen
    f_const = 1.3 * mu_s_expected * m * g
    cell.apply_force("azul_1", (f_const, 0.0, 0.0))
    vel = []
    for _ in range(int(0.4 / dt)):
        cell.step()
        vel.append(float(cell.truth("azul_1")["lin_vel"][0]))
    cell.clear_force("azul_1")
    for _ in range(int(0.8 / dt)):
        cell.step()
    vel = np.array(vel)
    k0, k1 = int(0.1 / dt), len(vel) - 1
    acc = (vel[k1] - vel[k0]) / ((k1 - k0) * dt)
    mu_d = (f_const - m * acc) / (m * g)
    res["friction"] = {"mass_kg": m, "onset_force_n": f_onset, "mu_static_measured": mu_s,
                       "mu_static_expected": mu_s_expected, "constant_force_n": f_const,
                       "acceleration_m_s2": float(acc), "mu_dynamic_measured": float(mu_d),
                       "mu_dynamic_expected": mu_d_expected}
    check("fricción estática medida ≈ configurada (±15 %)", None if mu_s is None else round(mu_s, 3),
          round(mu_s_expected, 3), mu_s is not None and abs(mu_s - mu_s_expected) <= 0.15 * mu_s_expected)
    check("fricción dinámica medida ≈ configurada (±15 %)", round(float(mu_d), 3), round(mu_d_expected, 3),
          abs(mu_d - mu_d_expected) <= 0.15 * mu_d_expected)

    # ----------------------------------------------------- deriva en reposo
    cell.reset_layout(layout)
    p0 = {n: cell.truth(n)["pos"].copy() for n in cell.pieces}
    for _ in range(int(10.0 / dt)):
        cell.step()
    drift = {n: float(np.linalg.norm(cell.truth(n)["pos"] - p0[n])) for n in cell.pieces}
    res["rest_drift"] = {"seconds": 10.0, "drift_m": drift}
    worst = max(drift.values())
    check("deriva en reposo 10 s < 0.5 mm", round(worst * 1000, 4), 0.5, worst < 0.0005)

    # ------------------------------------------------------------ servos
    ctrl = Controller(cell)
    q_start = ctrl.q().copy()
    # destino absoluto dentro de los límites articulares (la postura de partida ya
    # tiene el hombro girado 1.85 rad; sumarle algo lo sacaría del límite de 1.92)
    goal = np.array([0.6, -0.9, 0.9, 0.6, 0.8])
    qd_peak = np.zeros(5)
    track_err = 0.0
    total, at = trapezoid(q_start, goal, cfg.robot.max_joint_speed, cfg.robot.max_joint_accel)
    for n in range(int(total / dt) + 1):
        cell.set_arm_targets(at(n * dt))
        cell.step()
        p = cell.proprio()
        qd_peak = np.maximum(qd_peak, np.abs(p["arm_qd"]))
        track_err = max(track_err, float(np.max(np.abs(p["arm_q"] - at(n * dt)))))
    for _ in range(int(1.0 / dt)):
        cell.step()
    servo_only_err = (goal - ctrl.q()).tolist()
    # el mismo destino con el controlador completo (corrección integral leyendo la posición)
    moved = ctrl.move_joints(goal)
    final_err = float(np.max(np.abs(ctrl.q() - goal)))
    res["servo"] = {"trajectory_s": total, "peak_joint_speed_rad_s": qd_peak.tolist(),
                    "trajectory_speed_limit": cfg.robot.max_joint_speed,
                    "max_tracking_error_rad": track_err, "steady_error_servo_only_rad": servo_only_err,
                    "final_error_closed_loop_rad": final_err, "closed_loop_settle_s": moved.duration_s}
    check("velocidad articular ≤ límite del servo (4.71 rad/s)", round(float(qd_peak.max()), 3),
          cfg.robot.arm_servo.no_load_speed_rad_s, qd_peak.max() <= cfg.robot.arm_servo.no_load_speed_rad_s + 1e-3)
    check("error de seguimiento < 0.1 rad", round(track_err, 4), 0.1, track_err < 0.1)
    check("error estacionario del servo solo < 0.03 rad (servo P con carga)",
          round(float(np.max(np.abs(servo_only_err))), 4), 0.03, np.max(np.abs(servo_only_err)) < 0.03)
    check("error final con corrección < 0.005 rad", round(final_err, 4), 0.005, final_err < 0.005)

    # escalón crudo del codo: tiempo de subida, sobreoscilación
    base_q = ctrl.q().copy()
    step_q = base_q.copy()
    step_q[2] += 0.3
    cell.set_arm_targets(step_q)
    trace = []
    for _ in range(int(1.0 / dt)):
        cell.step()
        trace.append(cell.proprio()["arm_q"][2] - base_q[2])
    trace = np.array(trace)
    t90 = float(np.argmax(trace >= 0.27) * dt) if (trace >= 0.27).any() else None
    overshoot = float(max(0.0, trace.max() - 0.3))
    res["servo_step"] = {"step_rad": 0.3, "t90_s": t90, "overshoot_rad": overshoot,
                         "steady_error_rad": float(abs(0.3 - trace[-1]))}
    check("escalón de 0.3 rad: sobreoscilación < 0.03 rad", round(overshoot, 4), 0.03, overshoot < 0.03)

    # sostener el brazo extendido contra la gravedad
    ext = np.array([0.0, 0.9, -0.9, 0.0, 0.0])
    ctrl.move_joints(ext)
    ctrl.wait(1.0)
    sag = (ext - ctrl.q()).tolist()
    res["gravity_hold"] = {"pose": ext.tolist(), "error_rad": sag, "effort_nm": cell.proprio()["arm_effort"].tolist(),
                           "stall_torque_nm": cfg.robot.arm_servo.stall_torque_nm}
    check("brazo extendido: caída por gravedad < 0.05 rad", round(float(np.max(np.abs(sag))), 4), 0.05,
          np.max(np.abs(sag)) < 0.05)
    ctrl.move_joints(np.array(cfg.robot.start_joints[:5]))

    # ----------------------------------------------------------- rendimiento
    n = 1200
    t0 = time.perf_counter()
    for _ in range(n):
        cell.step()
    no_render = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for i in range(n):
        cell.step()
        if i % 12 == 0:
            cell.render()
    with_render = (time.perf_counter() - t0) / n
    res["performance"] = {"dt": dt, "ms_per_step_physics": no_render * 1000,
                          "ms_per_step_with_camera_20hz": with_render * 1000,
                          "real_time_factor_physics": dt / no_render, "real_time_factor_with_camera": dt / with_render}
    check("factor de tiempo real (física + cámara 20 Hz)", round(dt / with_render, 3), 0.5, dt / with_render >= 0.5)

    res["watchdog"] = cell.watchdog.summary()
    check("vigilante: 0 violaciones tras reinicio", res["watchdog"]["violations"], 0, res["watchdog"]["violations"] == 0)
    res["checks"] = checks
    res["passed"] = all(c["pass"] for c in checks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"[física] {'TODO CORRECTO' if res['passed'] else 'HAY FALLOS'} -> {args.out}", flush=True)
    rt.close(0 if res["passed"] else 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
