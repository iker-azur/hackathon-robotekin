# Evaluación `act_clock_final_wide_10ep`

- Fecha (UTC): 2026-10-20T15:37:59Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model` (tipo `act`)
- **Entradas de la política:** PRIVILEGED plan clock + cube pixel position seen once at the start by the colour detector; no camera image and no joint state go into the network, so it cannot react to the arm
- Posiciones del cubo: caja 1 cm más ancha que la de entrenamiento por cada lado (algunas posiciones obligan a extrapolar)
- Semilla: 0; pasos por episodio: 720

**Resultado: 10/10 (100 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.340, -0.450 | -18 | 0.2 | sí | - | 6.0 | 164 | 1.00 | 0.4 |
| 1 | 0.313, -0.413 | -6 | 0.9 | sí | - | 6.1 | 161 | 1.00 | 0.6 |
| 2 | 0.368, -0.376 | 6 | 0.4 | sí | - | 6.1 | 158 | 1.00 | 0.5 |
| 3 | 0.299, -0.437 | 18 | 0.9 | sí | - | 6.2 | 166 | 1.00 | 0.7 |
| 4 | 0.354, -0.401 | -27 | 0.7 | sí | - | 6.0 | 159 | 1.00 | 0.2 |
| 5 | 0.326, -0.364 | -15 | 0.7 | sí | - | 6.1 | 157 | 1.00 | 0.8 |
| 6 | 0.381, -0.458 | -3 | 1.6 | sí | - | 6.2 | 167 | 1.00 | 1.5 |
| 7 | 0.292, -0.421 | 9 | 1.0 | sí | - | 6.1 | 164 | 1.00 | 0.9 |
| 8 | 0.347, -0.385 | 21 | 0.1 | sí | - | 6.2 | 158 | 1.00 | 0.5 |
| 9 | 0.320, -0.446 | -25 | 0.5 | sí | - | 6.0 | 165 | 1.00 | 0.6 |
