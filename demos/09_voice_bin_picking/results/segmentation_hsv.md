# Segmentación — hsv (conjunto val, 6 imágenes, IoU ≥ 0.5)

- Instancias reales: 28 · recall **0.821** · precisión **0.92** (FP 2, FN 5) · IoU medio 0.798
- Color acertado: 1.0 · forma acertada: 0.87 (0 marcadas «?» por estar tapadas)
- Error de posición en el plano (mm): {'media': 1.35, 'p95': 2.08} · latencia por imagen (ms): {'media': 230.5, 'p95': 680.5}

| visibilidad de la pieza | recall | n |
|---|---|---|
| <50 % | 0.0 | 2 |
| 50–85 % | 1.0 | 2 |
| ≥85 % | 0.875 | 24 |

| tipo de escena | recall | n |
|---|---|---|
| separadas | 0.846 | 13 |
| tocándose | 0.821 | 28 |
| apiladas | 0.8 | 15 |
