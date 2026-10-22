# Evaluación `act_clock_final_outside_10ep`

- Fecha (UTC): 2026-10-20T15:47:36Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model` (tipo `act`)
- **Entradas de la política:** PRIVILEGED plan clock + cube pixel position seen once at the start by the colour detector; no camera image and no joint state go into the network, so it cannot react to the arm
- Posiciones del cubo: fuera de la caja de entrenamiento, en el anillo de 1 cm que la rodea (la política tiene que extrapolar)
- Semilla: 0; pasos por episodio: 720

**Resultado: 8/10 (80 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

Fallos por causa: `misplaced` × 1, `never_lifted` × 1.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.326, -0.364 | -15 | 0.7 | sí | - | 6.2 | 158 | 1.00 | 0.9 |
| 1 | 0.381, -0.458 | -3 | 1.6 | sí | - | 6.0 | 165 | 1.00 | 1.5 |
| 2 | 0.388, -0.360 | -23 | 2.5 | no | misplaced | 5.5 | 158 | 1.00 | 3.1 |
| 3 | 0.283, -0.454 | -11 | 1.9 | sí | - | 6.0 | 169 | 1.00 | 1.0 |
| 4 | 0.290, -0.356 | 28 | 2.0 | no | never_lifted | 0.0 | 0 | 0.00 | 9.6 |
| 5 | 0.345, -0.464 | -29 | 1.3 | sí | - | 6.1 | 167 | 1.00 | 0.8 |
| 6 | 0.387, -0.440 | -15 | 1.5 | sí | - | 6.0 | 163 | 1.00 | 1.0 |
| 7 | 0.287, -0.403 | -3 | 0.6 | sí | - | 6.1 | 162 | 1.00 | 0.8 |
| 8 | 0.314, -0.460 | 21 | 1.2 | sí | - | 6.0 | 167 | 1.00 | 0.8 |
| 9 | 0.383, -0.374 | 23 | 1.2 | sí | - | 6.2 | 158 | 1.00 | 1.4 |
