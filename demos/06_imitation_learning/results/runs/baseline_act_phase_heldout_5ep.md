# Evaluación `baseline_act_phase_heldout_5ep`

- Fecha (UTC): 2026-10-19T18:48:39Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act_phase\checkpoints\last\pretrained_model` (tipo `act`)
- **Entradas de la política:** camera image + joint state + PRIVILEGED plan phase index
- Posiciones del cubo: fuera del entrenamiento (índices Halton 1000 a 1004; ninguna posición de las demostraciones)
- Semilla: 0; pasos por episodio: 720

**Resultado: 0/5 (0 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

Fallos por causa: `misplaced` × 1, `never_lifted` × 4.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.343, -0.394 | -18 | 0.5 | no | never_lifted | 0.4 | 0 | 0.00 | 8.1 |
| 1 | 0.321, -0.444 | -6 | 0.7 | no | misplaced | 6.4 | 162 | 1.00 | 6.8 |
| 2 | 0.366, -0.414 | 6 | 0.7 | no | never_lifted | 0.5 | 0 | 0.00 | 10.6 |
| 3 | 0.310, -0.384 | 18 | 0.7 | no | never_lifted | 0.7 | 0 | 0.00 | 10.8 |
| 4 | 0.355, -0.434 | -27 | 0.7 | no | never_lifted | 0.0 | 0 | 0.00 | 9.8 |
