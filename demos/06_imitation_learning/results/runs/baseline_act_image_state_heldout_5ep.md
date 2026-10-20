# Evaluación `baseline_act_image_state_heldout_5ep`

- Fecha (UTC): 2026-10-19T18:55:47Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act\checkpoints\last\pretrained_model` (tipo `act`)
- **Entradas de la política:** camera image + joint state (nothing privileged)
- Posiciones del cubo: fuera del entrenamiento (índices Halton 1000 a 1004; ninguna posición de las demostraciones)
- Semilla: 0; pasos por episodio: 720

**Resultado: 0/5 (0 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

Fallos por causa: `never_lifted` × 5.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.343, -0.394 | -18 | 0.5 | no | never_lifted | -0.0 | 0 | 0.00 | 10.0 |
| 1 | 0.321, -0.444 | -6 | 0.7 | no | never_lifted | -0.0 | 0 | 0.00 | 9.9 |
| 2 | 0.366, -0.414 | 6 | 0.7 | no | never_lifted | -0.0 | 0 | 0.00 | 10.1 |
| 3 | 0.310, -0.384 | 18 | 0.7 | no | never_lifted | -0.0 | 0 | 0.00 | 9.8 |
| 4 | 0.355, -0.434 | -27 | 0.7 | no | never_lifted | -0.0 | 0 | 0.00 | 10.0 |
