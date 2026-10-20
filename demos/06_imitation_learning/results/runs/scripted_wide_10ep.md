# Evaluación `scripted_wide_10ep`

- Fecha (UTC): 2026-10-20T15:42:22Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `scripted teacher` (tipo `scripted`)
- **Entradas de la política:** SCRIPTED teacher with the cube's true pose (privileged) - NOT a learned policy
- Posiciones del cubo: caja 1 cm más ancha que la de entrenamiento por cada lado (algunas posiciones obligan a extrapolar)
- Semilla: 0; pasos por episodio: 720
- Desplazamiento del punto de agarre del guion respecto al centro del cubo (mm, x e y del mundo): [-10.0, 2.0]

**Resultado: 10/10 (100 %).** Esto es el guion del profesor, no una política aprendida. Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.340, -0.450 | -18 | 0.2 | sí | - | 6.0 | 161 | 1.00 | 0.6 |
| 1 | 0.313, -0.413 | -6 | 0.9 | sí | - | 6.1 | 162 | 1.00 | 0.8 |
| 2 | 0.368, -0.376 | 6 | 0.4 | sí | - | 6.1 | 162 | 1.00 | 0.7 |
| 3 | 0.299, -0.437 | 18 | 0.9 | sí | - | 6.1 | 161 | 1.00 | 0.6 |
| 4 | 0.354, -0.401 | -27 | 0.7 | sí | - | 6.0 | 162 | 1.00 | 0.4 |
| 5 | 0.326, -0.364 | -15 | 0.7 | sí | - | 6.2 | 162 | 1.00 | 0.7 |
| 6 | 0.381, -0.458 | -3 | 1.6 | sí | - | 5.9 | 161 | 1.00 | 0.7 |
| 7 | 0.292, -0.421 | 9 | 1.0 | sí | - | 6.1 | 161 | 1.00 | 0.9 |
| 8 | 0.347, -0.385 | 21 | 0.1 | sí | - | 6.3 | 162 | 1.00 | 0.7 |
| 9 | 0.320, -0.446 | -25 | 0.5 | sí | - | 6.2 | 163 | 1.00 | 0.7 |
