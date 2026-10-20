# Evaluación `scripted_original_grasp_point_heldout_10ep`

- Fecha (UTC): 2026-10-19T18:45:45Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `scripted teacher` (tipo `scripted`)
- **Entradas de la política:** SCRIPTED teacher with the cube's true pose (privileged) - NOT a learned policy
- Posiciones del cubo: fuera del entrenamiento (índices Halton 1000 a 1009; ninguna posición de las demostraciones)
- Semilla: 0; pasos por episodio: 720
- Desplazamiento del punto de agarre del guion respecto al centro del cubo (mm, x e y del mundo): [0.0, 0.0]

**Resultado: 10/10 (100 %).** Esto es el guion del profesor, no una política aprendida. Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.343, -0.394 | -18 | 0.5 | sí | - | 6.1 | 162 | 1.00 | 0.6 |
| 1 | 0.321, -0.444 | -6 | 0.7 | sí | - | 6.0 | 161 | 1.00 | 0.5 |
| 2 | 0.366, -0.414 | 6 | 0.7 | sí | - | 6.0 | 162 | 1.00 | 0.6 |
| 3 | 0.310, -0.384 | 18 | 0.7 | sí | - | 6.1 | 162 | 1.00 | 0.6 |
| 4 | 0.355, -0.434 | -27 | 0.7 | sí | - | 6.0 | 161 | 1.00 | 0.7 |
| 5 | 0.332, -0.404 | -15 | 0.7 | sí | - | 5.9 | 161 | 1.00 | 0.5 |
| 6 | 0.377, -0.374 | -3 | 0.7 | sí | - | 6.2 | 161 | 1.00 | 0.4 |
| 7 | 0.295, -0.450 | 9 | 0.7 | sí | - | 6.0 | 161 | 1.00 | 0.4 |
| 8 | 0.341, -0.420 | 21 | 0.7 | sí | - | 5.9 | 161 | 1.00 | 0.7 |
| 9 | 0.318, -0.390 | -25 | 0.7 | sí | - | 6.0 | 162 | 1.00 | 0.4 |
