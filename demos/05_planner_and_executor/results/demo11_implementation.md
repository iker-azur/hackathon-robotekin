# Demo 11: implementación y verificación

Fecha: 2026-10-08. Rama: `prepare-demos`. Entorno Conda `hackathon-robotekin`, Windows, Isaac Sim 5.1 y RTX 5060 Ti.

## Cambios por fase

1. **Resultados fiables.** Las órdenes terminan con éxito, fallo, cancelación, aclaración o éxito parcial y un motivo visible. El ledger distingue cada orden, pieza física y destino. Los destinos inexistentes, referencias ambiguas, cantidades insuficientes y planes parcialmente resueltos se rechazan antes del movimiento. La verificación final se hace después de aparcar el brazo.
2. **Servidor persistente.** `supervisor.py` conserva el panel y ejecuta Isaac en un proceso por sesión. Iniciar, detener y reiniciar funcionan desde HTTP. Un fallo o cierre nativo deja informe y registro descargables. El inicio solo se confirma después de tener escena y cámara listas; se corrige la salida prematura causada por EOF en el proceso trabajador.
3. **Escena configurable.** `scene_config.py` valida el JSON, coloca unidades adicionales y genera el catálogo que comparten planner, percepción, parser y ejemplos. El panel permite editar piezas y recipientes, guardar y restaurar una escena sencilla. Se admiten cubos/bolas rojos, verdes y azules y recipientes con nombre/alias; los destinos del mismo color necesitan nombre para desambiguar. Se comprueba espacio disponible en la cavidad y se reservan posiciones entre órdenes.
4. **Percepción y tiempos.** Segmentación restringida al espacio de trabajo, umbral de brillo más tolerante, morfología y características calculadas en regiones pequeñas. Separación conservadora de algunos componentes unidos; clasificación desconocida cuando no hay evidencia suficiente. Asociación uno a uno, rechazo de identidades ambiguas y corrección de posición con la altura declarada. JPEG en un trabajador con un único fotograma pendiente, reutilización de capturas recientes y actualización del DOM solo cuando cambia una sección.
5. **Panel y ejecución.** Estados, planes, ejemplos, diagnóstico y resultados en español; cámara con antigüedad y aviso de fallo. El diagrama deja de conservar etapas activas al terminar una orden. Se mantiene disponible después de fallos de agarre. Los reintentos de Demo 11 conservan la identidad y el destino originales; no usan la replanificación global heredada que podía cambiar la tarea. La altura de liberación depende del recipiente concreto.

No se modificaron `10_command_to_motion.py` ni `executor.py`. No se cambiaron pesos del modelo.

## Medidas de percepción: comparación sobre las mismas imágenes

El benchmark carga el detector anterior desde Git (`979fd5b^`) y aplica ambos a nueve escenas iniciales capturadas en Isaac, semillas 100–108. Cada escena se evalúa con brillo original y multiplicado digitalmente por 0,7: 18 imágenes evaluadas, 56 observaciones de piezas y 36 de recipientes. Las poses de física se usan exclusivamente para puntuar, nunca como entrada del detector. Asociación por color, categoría y distancia XY máxima de 40 mm, sin forzar coincidencia de tipo. Una pasada de calentamiento y cinco repeticiones por imagen para medir tiempo.

| Medida | Detector anterior | Detector mejorado |
|---|---:|---:|
| Piezas localizadas y asociadas | 52/56 | 54/56 |
| Piezas con tipo correcto, sobre todas las esperadas | 33/56 (58,9 %) | 53/56 (94,6 %) |
| Recipientes localizados y asociados | 36/36 | 36/36 |
| Recipientes con tipo correcto | 31/36 | 34/36 |
| Error XY medio, coincidencias de piezas y recipientes | 5,502 mm | 4,164 mm |
| Tiempo medio de segmentación y clasificación | 17,332 ms | 16,572 ms |

La optimización del detector legacy por sí sola mide 15,151 ms y conserva su exactitud original. Son medidas de CPU sobre imágenes guardadas; no incluyen captura, anotación, JPEG ni transporte. Hay escenas utilizadas durante la calibración, por lo que esto no es una estimación independiente de generalización. Quedan dos piezas sin localizar en la escena de seis piezas. La protección del catálogo evita presentar una orden «todas» como completada cuando falta identificar una pieza.

Los cambios de brillo son digitales. El solapamiento de bolas se comprueba con imágenes sintéticas en unit tests; no se ha medido exactitud con oclusión severa en simulación ni con otras luces reales.

## Decisión sobre el modelo de intención

El modelo extrae intención, referencias de piezas (`kind`, `color`, `quantity`, `position`), destinos globales/por pieza, `sort_by` y `sort_filter`, con puntuaciones por campo. Esas puntuaciones no son probabilidades calibradas. Su vocabulario aprendido no permite asignar libremente nombres nuevos a cajas. El fallback conserva usos del modelo para frases generales; las reglas completas añaden destinos del catálogo y validación conservadora. Los colores o referencias desconocidos, negaciones de movimiento y cantidades inválidas producen aclaración.

Se corrigió el método del benchmark para incluir `sort_filter` y excluir coincidencias normalizadas con entrenamiento. De las 136 frases de `test_real.jsonl`, 49 coinciden y quedan 87. Esto elimina duplicados de texto, no garantiza independencia semántica respecto al entrenamiento.

| Parser, mismas 87 frases | Comando exacto | Intención | Media por frase |
|---|---:|---:|---:|
| Modelo actual | 63/87 | 80/87 | 76,033 ms |
| Solo reglas | 14/87 | 15/87 | 0,289 ms |
| Híbrido restringido | 62/87 | 80/87 | 68,297 ms |

El híbrido rehúsa «Coge el cubo azul, no, mejor el rojo», que el conjunto etiqueta como acción roja. Ese rechazo explica una diferencia con el modelo. Un prototipo de reglas más general obtuvo 53/87 y se descartó. Las reglas rápidas del híbrido se limitan a parada y referencias explícitas completas de un cubo/bola con color; las frases generales siguen pasando por el modelo.

Se congelaron además 30 casos nuevos antes de evaluarlos: cantidades 1–6 y un recipiente nombrado, en español/inglés/euskera, más 12 casos que deben rechazarse. Véase [demo11_extension_cases.json](demo11_extension_cases.json).

| Parser, extensión de escena | Correctos | Rechazos seguros | Media |
|---|---:|---:|---:|
| Modelo actual | 3/30 | 3/12 | 67,506 ms |
| Solo reglas | 30/30 | 12/12 | 0,323 ms |
| Híbrido | 30/30 | 12/12 | 15,217 ms |

**Decisión:** mantener los pesos y el parser híbrido. Sustituir el modelo por reglas pierde demasiada cobertura general. El catálogo necesita resolución en tiempo de ejecución, no un reentrenamiento por cada caja nueva. Los 30 casos cubren pocas familias de frases; su resultado no demuestra exactitud general del lenguaje. El evaluador histórico `command_model_v2/evaluate.py` conserva su comparación antigua; para estas medidas se debe usar `tools/benchmark_demo11.py intent`.

## Rendimiento del panel y de la ejecución

| Prueba comparable | Antes | Después |
|---|---:|---:|
| JavaScript, DOM simulado, 1000 snapshots que solo cambian métricas; media de cinco ejecuciones | 256,176 ms | 84,886 ms |
| Escrituras HTML en esa prueba | 12013 | 1016 |
| Websocket localhost, 100 publicaciones, tiempo al primer snapshot | 12,595 ms | 12,597 ms |
| Mensajes extra en 0,5 s inactivo | 0 | 0 |

El DOM simulado mide trabajo de JavaScript, no pintura ni interacción de un navegador. El websocket aislado no mide latencia de cámara ni HTTP hasta movimiento. Los tiempos del sistema dependen de carga y caché; las mejoras anteriores no prueban que toda la simulación sea más rápida.

Perfil inicial: proceso 76,439 s, `look` medio 42,405 ms y captura RGB 21,864 ms. Perfil final de clasificación de dos cubos (semilla 108): proceso hasta informe 106,463 s, orden 61,864 s, 60 capturas RGB con media 21,680 ms, 28 llamadas `look` con media 49,503 ms. Este benchmark incluye escritura de imágenes en disco; las órdenes, escenas y carga no son iguales a las iniciales. **No hay una mejora demostrada de duración total ni del tiempo hasta el primer movimiento físico.** La métrica de entrada a ejecución es 231,4 ms en ese caso; el JPEG final 0,53 ms y se reutilizaron 25 imágenes. Son muestras de esa ejecución, no medias universales.

## Verificación física y de ciclo de vida

- Isaac headless se ejecutó realmente, con imágenes y poses físicas registradas. Se observaron éxitos, fallos de agarre y resultados parciales a través de registros y estados; no se inspeccionó visualmente la animación del brazo en un navegador.
- Clasificación de dos cubos con el preset sencillo centrado, semilla 108, sin dispersión: **SUCCESS, 2/2 piezas verificadas, cero agarres rechazados y cero sujeciones cinemáticas**. Una ejecución anterior con el rojo a la izquierda (semilla 107) terminó 2/2 tras rechazos y reintentos; otra con el cubo nativo (106) terminó 1/2. El preset usa dos cubos creados con el mismo material y una posición roja más centrada; el material por sí solo no eliminó los rechazos.
- Las bolas siguen teniendo riesgo de rodar y perder el agarre. Hubo órdenes físicas de bola que terminaron FAILED con motivo visible y el entorno siguió aceptando órdenes. No se promete éxito de agarre para cualquier configuración aceptada.
- Se ejercitaron por HTTP el arranque, rechazo de magenta, movimiento, parada y reinicio de dos sesiones en el mismo servidor. El informe descargable permanece después de parar. Las mediciones concretas de la repetición final se incluyen en [demo11_metrics.json](demo11_metrics.json).

| Sesión HTTP final | Arranque | Orden | Estado de orden | Entrada a ejecución | Salud HTTP media (15 consultas) | Parada |
|---|---:|---:|---|---:|---:|---:|
| 0 | 45.580 s | 18.895 s | FAILED | 40.3 ms | 6.646 ms | 2.986 s |
| 1 | 45.447 s | 31.586 s | SUCCESS | 38.9 ms | 7.091 ms | 2.451 s |

- El navegador disponible mediante la herramienta de automatización devolvió «Browser is not available: chrome»; no fue posible revisar visualmente el dashboard. Se verificaron JavaScript con Node/DOM simulado, endpoints y websocket. No se probó la grabación real del micrófono ni los permisos del navegador.

## Pruebas de regresión

Se ejecutaron las suites durante las fases. Registros intermedios: Demo 11/planner, 501/1474, 507/834, 524/839 y 525/839 comprobaciones, todos SUCCESS. La caída de comprobaciones del planner coincide con aplicar realmente los filtros de ordenación: genera menos pasos; se añadieron aserciones explícitas sobre filtros, cantidades y destinos inexistentes.

Resultado final:

```text
537 checks over the demo 11 real-world model, the bus and the dashboard
RESULT: SUCCESS
839 checks over 20 recorded scene(s)
RESULT: SUCCESS
```

Incluye catálogo, alias ambiguos, cantidades, capacidad, estados terminales, forwarding IPC, parada prioritaria, sesión independiente, cámara, separación sintética, altura por recipiente, verificación posterior a aparcar y ausencia de etapas activas obsoletas.

## Uso y reproducción

Desde `demos/05_planner_and_executor`, con el entorno Conda indicado:

```powershell
python 11_real_sorting_with_monitor.py --headless --no-viewer --spread 0
```

Abrir `http://127.0.0.1:8765`, guardar «escena simple», iniciar entorno y enviar «separa las piezas por colores». Para reproducir el preset observado usar semilla 108 y dispersión cero. La configuración guardada se conserva en `out11/scene_config.json`; `--scene-config ruta.json` permite usar otro fichero. Si solo existen assets en caché, establecer `HF_HUB_OFFLINE=1`; esto no descarga dependencias ausentes.

El JSON tiene `version: 1`, `objects` y `containers`. Cada pieza declara `id`, `kind`, `color`, `count`, `size_mm` y opcionalmente `x/y` en metros. Cada recipiente declara `id`, `name`, `aliases`, `color`, `size_mm`, `cavity_mm`, `wall_mm`, `floor_mm` y opcionalmente posición. El editor avanzado y los errores muestran el contrato de validación. Los cambios de escena se guardan con el entorno detenido. Los límites geométricos de validación no garantizan que la pinza agarre todos los tamaños permitidos; aquí se probaron físicamente piezas de 30 mm.

```powershell
python tests/test_demo11.py
python tests/test_planner.py
python tools/benchmark_demo11.py intent
python tools/benchmark_demo11.py extensions
python tools/benchmark_demo11.py monitor
python tools/verify_demo11_session.py
python tools/benchmark_demo11.py capture --headless --no-viewer --seed 108 --spread 0 --scene-config ruta.json --text "separa las piezas por colores"
python tools/benchmark_demo11.py perception
```

Las capturas y registros brutos se guardan localmente en `out11/benchmarks` y están ignorados por Git; se versionan el resumen y los casos de evaluación. El benchmark de percepción compara con la revisión original indicada y necesita las capturas. El verificador HTTP restaura la configuración que había antes de empezar.

## Pendiente manual y riesgos abiertos

- Revisar disposición visual, cámara, diagrama en movimiento y voz real en Chrome/Edge; enviar órdenes en los tres idiomas.
- Repetir el preset con las semillas previstas para el hackathon y evaluar tasa de éxito físico con más repeticiones. Dos piezas verificadas en una ejecución no constituyen una tasa de éxito general.
- Probar parada durante cada etapa de movimiento, fallo real de cámara, descarga del informe y recuperación de un cierre nativo desde la interfaz visual.
- Probar cajas de alturas mezcladas, varias cajas del mismo color, más piezas y tamaños distintos en Isaac. La capacidad y geometría tienen pruebas unitarias; esos casos no tienen una campaña física completa.
- Iluminación distinta, bolas que ruedan, oclusión severa y escenas densas siguen siendo límites. La paleta queda restringida a rojo/verde/azul hasta medir colores nuevos.
- La carga fría del modelo y de assets sigue dominando el arranque. Hubo avisos de DLL de plugins Isaac y un problema de certificados online; se mantuvieron los avisos en el registro técnico.
