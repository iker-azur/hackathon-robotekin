# Evaluación `scripted_outside_10ep`

- Fecha (UTC): 2026-10-20T15:52:05Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `scripted teacher` (tipo `scripted`)
- **Entradas de la política:** SCRIPTED teacher with the cube's true pose (privileged) - NOT a learned policy
- Posiciones del cubo: fuera de la caja de entrenamiento, en el anillo de 1 cm que la rodea (la política tiene que extrapolar)
- Semilla: 0; pasos por episodio: 720
- Desplazamiento del punto de agarre del guion respecto al centro del cubo (mm, x e y del mundo): [-10.0, 2.0]

**Resultado: 10/10 (100 %).** Esto es el guion del profesor, no una política aprendida. Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.326, -0.364 | -15 | 0.7 | sí | - | 6.2 | 162 | 1.00 | 0.7 |
| 1 | 0.381, -0.458 | -3 | 1.6 | sí | - | 5.9 | 161 | 1.00 | 0.7 |
| 2 | 0.388, -0.360 | -23 | 2.5 | sí | - | 6.3 | 161 | 1.00 | 1.6 |
| 3 | 0.283, -0.454 | -11 | 1.9 | sí | - | 6.1 | 161 | 1.00 | 0.7 |
| 4 | 0.290, -0.356 | 28 | 2.0 | sí | - | 6.2 | 162 | 1.00 | 0.9 |
| 5 | 0.345, -0.464 | -29 | 1.3 | sí | - | 6.1 | 161 | 1.00 | 0.7 |
| 6 | 0.387, -0.440 | -15 | 1.5 | sí | - | 6.4 | 163 | 1.00 | 0.5 |
| 7 | 0.287, -0.403 | -3 | 0.6 | sí | - | 6.0 | 161 | 1.00 | 0.8 |
| 8 | 0.314, -0.460 | 21 | 1.2 | sí | - | 6.1 | 161 | 1.00 | 0.8 |
| 9 | 0.383, -0.374 | 23 | 1.2 | sí | - | 6.2 | 161 | 1.00 | 0.6 |
