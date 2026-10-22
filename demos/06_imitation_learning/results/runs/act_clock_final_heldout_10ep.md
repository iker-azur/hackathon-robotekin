# Evaluación `act_clock_final_heldout_10ep`

- Fecha (UTC): 2026-10-19T17:59:04Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model` (tipo `act`)
- **Entradas de la política:** PRIVILEGED plan clock + cube pixel position seen once at the start by the colour detector; no camera image and no joint state go into the network, so it cannot react to the arm
- Posiciones del cubo: fuera del entrenamiento (índices Halton 1000 a 1009; ninguna posición de las demostraciones)
- Semilla: 0; pasos por episodio: 720

**Resultado: 10/10 (100 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.343, -0.394 | -18 | 0.5 | sí | - | 6.1 | 158 | 1.00 | 0.3 |
| 1 | 0.321, -0.444 | -6 | 0.7 | sí | - | 6.0 | 165 | 1.00 | 0.6 |
| 2 | 0.366, -0.414 | 6 | 0.7 | sí | - | 6.1 | 161 | 1.00 | 0.7 |
| 3 | 0.310, -0.384 | 18 | 0.7 | sí | - | 6.1 | 158 | 1.00 | 0.7 |
| 4 | 0.355, -0.434 | -27 | 0.7 | sí | - | 5.9 | 162 | 1.00 | 0.2 |
| 5 | 0.332, -0.404 | -15 | 0.7 | sí | - | 6.1 | 159 | 1.00 | 0.4 |
| 6 | 0.377, -0.374 | -3 | 0.7 | sí | - | 6.1 | 158 | 1.00 | 1.1 |
| 7 | 0.295, -0.450 | 9 | 0.7 | sí | - | 6.0 | 167 | 1.00 | 0.8 |
| 8 | 0.341, -0.420 | 21 | 0.7 | sí | - | 6.0 | 161 | 1.00 | 0.9 |
| 9 | 0.318, -0.390 | -25 | 0.7 | sí | - | 6.0 | 158 | 1.00 | 0.5 |
