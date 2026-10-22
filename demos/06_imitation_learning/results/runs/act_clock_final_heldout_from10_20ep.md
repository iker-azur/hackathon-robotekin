# Evaluación `act_clock_final_heldout_from10_20ep`

- Fecha (UTC): 2026-10-19T18:35:34Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model` (tipo `act`)
- **Entradas de la política:** PRIVILEGED plan clock + cube pixel position seen once at the start by the colour detector; no camera image and no joint state go into the network, so it cannot react to the arm
- Posiciones del cubo: fuera del entrenamiento (índices Halton 1010 a 1029; ninguna posición de las demostraciones)
- Semilla: 0; pasos por episodio: 720

**Resultado: 20/20 (100 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.363, -0.440 | -13 | 0.7 | sí | - | 6.0 | 162 | 1.00 | 0.6 |
| 11 | 0.307, -0.410 | -1 | 0.7 | sí | - | 6.1 | 161 | 1.00 | 0.6 |
| 12 | 0.352, -0.380 | 11 | 0.7 | sí | - | 6.2 | 158 | 1.00 | 0.5 |
| 13 | 0.329, -0.430 | 23 | 0.7 | sí | - | 5.9 | 162 | 1.00 | 0.9 |
| 14 | 0.374, -0.400 | -22 | 0.7 | sí | - | 6.3 | 160 | 1.00 | 0.2 |
| 15 | 0.301, -0.370 | -10 | 0.7 | sí | - | 6.2 | 157 | 1.00 | 0.7 |
| 16 | 0.346, -0.447 | 2 | 0.7 | sí | - | 6.0 | 164 | 1.00 | 0.4 |
| 17 | 0.324, -0.417 | 14 | 0.7 | sí | - | 6.0 | 161 | 1.00 | 0.8 |
| 18 | 0.369, -0.387 | 26 | 0.7 | sí | - | 6.2 | 159 | 1.00 | 0.2 |
| 19 | 0.312, -0.437 | -20 | 0.7 | sí | - | 6.1 | 165 | 1.00 | 0.7 |
| 20 | 0.357, -0.407 | -8 | 0.7 | sí | - | 6.0 | 160 | 1.00 | 0.3 |
| 21 | 0.335, -0.377 | 4 | 0.6 | sí | - | 6.2 | 157 | 1.00 | 0.7 |
| 22 | 0.380, -0.427 | 16 | 0.7 | sí | - | 6.5 | 163 | 1.00 | 0.8 |
| 23 | 0.290, -0.397 | 28 | 0.5 | sí | - | 6.4 | 162 | 1.00 | 0.9 |
| 24 | 0.335, -0.367 | -29 | 0.5 | sí | - | 5.9 | 156 | 1.00 | 0.7 |
| 25 | 0.312, -0.453 | -17 | 0.5 | sí | - | 6.0 | 166 | 1.00 | 0.6 |
| 26 | 0.357, -0.423 | -5 | 0.5 | sí | - | 6.3 | 162 | 1.00 | 0.3 |
| 27 | 0.301, -0.393 | 7 | 0.5 | sí | - | 6.2 | 159 | 1.00 | 0.8 |
| 28 | 0.346, -0.443 | 19 | 0.5 | sí | - | 6.0 | 163 | 1.00 | 0.7 |
| 29 | 0.324, -0.413 | -27 | 0.5 | sí | - | 6.0 | 160 | 1.00 | 0.3 |
