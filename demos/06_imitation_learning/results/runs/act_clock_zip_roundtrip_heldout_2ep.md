# Evaluación `act_clock_zip_roundtrip_heldout_2ep`

- Fecha (UTC): 2026-10-20T15:32:35Z
- Código: commit `a3db19f` (con cambios sin confirmar), rama `prepare-demos`
- Política: `C:\Users\Usuario\AppData\Local\Temp\claude\C--Users-Usuario-Documents-Workspace-LeRobot-\1ad4b6ab-6a18-496e-9d3a-440357f592fe\scratchpad\portable\model` (tipo `act`)
- **Entradas de la política:** PRIVILEGED plan clock + cube pixel position seen once at the start by the colour detector; no camera image and no joint state go into the network, so it cannot react to the arm
- Posiciones del cubo: fuera del entrenamiento (índices Halton 1000 a 1001; ninguna posición de las demostraciones)
- Semilla: 0; pasos por episodio: 720

**Resultado: 2/2 (100 %).** Éxito = el cubo se levanta ≥ 4 cm, se transporta sujeto ≥ 1 s y se deja a < 3 cm del objetivo, en reposo (`il_metrics.py`). Sin sujeción cinemática.

| ep | cubo x, y (m) | yaw (°) | a la demo más cercana (cm) | éxito | fallo | elevación máx (cm) | transporte (pasos) | sujeción | error final (cm) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.343, -0.394 | -18 | 0.5 | sí | - | 6.1 | 158 | 1.00 | 0.3 |
| 1 | 0.321, -0.444 | -6 | 0.7 | sí | - | 6.0 | 165 | 1.00 | 0.6 |
