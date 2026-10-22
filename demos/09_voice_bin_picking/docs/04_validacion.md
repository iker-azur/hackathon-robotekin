# 4 · Validación: qué está probado, con qué, y cómo validar el resto

Regla de este demo: **no se afirma que algo funciona hasta que se ha medido**, y cada medida dice con qué «mundo» se hizo.

| «mundo» de la prueba | qué es | qué puede demostrar | qué NO puede demostrar |
|---|---|---|---|
| **lógico** (`sim_stub.FakeRobot`) | piezas en una lista; agarres que salen bien o mal por probabilidad o guion | la lógica del orquestador: qué elige, cómo reintenta, cajas llenas, parada, informe | cualquier cosa del brazo, la cámara o la física |
| **cinemático** (`tests/fake_cell.py`) | las articulaciones siguen la consigna, la pinza coge por contacto geométrico, la cámara es un render sintético | que los módulos encajan de punta a punta (percepción → plan → colisión → `skills` → verificación → informe) | fricción, contactos, estabilidad del agarre, PhysX |
| **sintético** (`tests/synth.py`) | render con OpenCV de piezas sobre la mesa | la lógica de segmentación, color, posición, visibilidad | cómo se ve un render de Isaac, y menos una cámara real |
| **física** (Isaac Sim) | PhysX, servos con par, cámara RTX | todo lo anterior de verdad | **ejecutado el 22 de octubre de 2026** (sección «Medido en Isaac»); sigue sin cubrir un robot ni una cámara reales |

## Lo verificado (sin Isaac): `python -m pytest tests -q`

| fichero | pruebas | qué comprueba |
|---|---|---|
| `test_commands.py` | 33 | los 4 ejemplos del enunciado y variantes («las demás», cantidades, formas), erratas de Whisper, parada, errores claros, atributo «defecto» sin clasificador, reparto previsto y capacidad |
| `test_segmentation.py` | 17 | color, posición (<4 mm), forma, **piezas pegadas → 2 instancias, apiladas → 2 alturas**, visibilidad de la tapada, la mesa y el contenedor no son piezas, ruido de sensor, YOLO inyectado (el color lo mide la máscara) |
| `test_seg_eval.py` | 3 | el evaluador sobre un conjunto exportado: **recall 1,0 con visibilidad ≥ 85 %, precisión 1,0, 0 falsos positivos**; las visibles < 35 % se pierden |
| `test_selection_tracker.py` | 18 | filtro por regla y cantidad, «las demás», lista negra, caja llena, la de arriba primero, la tapada se bloquea, hueco de dedos junto a la pared y entre vecinas, IK real sobre el contenedor, seguimiento y eventos |
| `test_collision.py` | 7 | el cuerpo de la pinza contra las paredes a lo largo del trayecto real; con los giros que dejan hueco a los dedos 22 de 24 posiciones tienen plan libre |
| `test_scene_gen.py` | 12 | semillas reproducibles y variadas, modos (separadas/juntas/apiladas/mixto), geometría de las caídas, medición de la configuración resultante, validación |
| `test_joints_observation.py` | 12 | conversión de articulaciones ida y vuelta (la de LeIsaac), recorte con las 3 cajas dentro, codificación del objetivo, **claves = características del dataset**, modos de observación |
| `test_place_policy.py` | 13 | huecos en cajas (con dedos, paredes y alcance), capacidad 5 con la cinemática real, limitador de acciones, compatibilidad del checkpoint |
| `test_pick_runner.py` | 8 | el ejecutor a 30 Hz: éxito, «no llegó», «vacío», «deslizada», «no levantó», política descontrolada recortada, parada |
| `test_orchestrator.py` | 22 | clasificación completa en 40 escenas, reintento tras agarre fallido, pieza movida, caída fuera (se recupera), **un colocado que miente no cuenta (manda la cámara)**, ACT→guion etiquetado, caja llena, parada, reproducibilidad |
| `test_integration.py` | 6 | **de punta a punta en el simulador cinemático**: el guion clasifica y la verdad coincide; tubería ACT con una reproducción del experto; ACT sin política cargada falla con honestidad; parada |
| `test_collect.py` | 5 | grabación a 30 Hz, ruido sólo en la aproximación con la consigna limpia grabada, criterio de éxito, **dataset real de LeRobot** válido y **reanudable**, validador |
| `test_lerobot_compat.py` | 7 | **ACT de LeRobot acepta el dataset** (pasada hacia delante finita), los tres modos y `--no-state` construyen una política válida, guardar → `ACTPickPolicy` → orden de 6 articulaciones |
| `test_tools.py` | 20 | todos los scripts compilan; el juicio con la verdad; `train_act.build_cfg` valida con la comprobación de `train()`; evaluación offline corre sobre un checkpoint |
| `test_ui.py` | 13 | panel en español, órdenes (409 si trabaja o arranca), voz→orden con micrófono falso, WebSocket con imágenes etiquetadas, la imagen girada pone la caja izquierda a la izquierda |

**Total: 196 pruebas, ~2,7 min** (la mayor parte es cinemática inversa y codificación de vídeo). Estas pruebas encontraron y corrigieron, entre otros: un criterio de éxito del ejecutor que daba por bueno un agarre sin haber bajado; la confusión del marrón de
las cajas con el rojo; una pieza de abajo tapada a medias que se fundía con su vecina; capacidad de caja que ignoraba los dedos; y la postura de observación que tapaba una caja.

## Medidas de diseño (sin Isaac, reproducibles)

| medida | valor | cómo |
|---|---|---|
| alcance de agarre en el contenedor, suelo / 2.ª capa | 75/75 · 65/75 | `python tools/check_layout_reach.py --step 0.012` → `results/layout_reach.json` |
| alcance de una colocación en las cajas (rejilla de 36) | 33/36 izquierda · 36/36 derecha | ídem |
| el brazo plegado tapa contenedor o cajas | 0 solapes | ídem (`brazo_plegado.solapa`) |
| capacidad de cada caja con la cinemática real | 5 | `place.Placer.capacity` |
| piezas agarrables a la vista (separadas / mixto / apiladas / juntas) | 79 % / 62 % / 35 % / 17 % | `selection.rank` sobre `FakeRobot`, 60 semillas × 5 piezas |
| clasificación completa en lógica, 60 escenas | rojas→izq + verdes→der: 41 completadas, 19 incompletas (dicen por qué) | `SortingTask` + `FakeRobot` |

## Medido en Isaac (22 de octubre de 2026)

Todo con el **guion programado del experto (no es aprendizaje)**, salvo la prueba de la ruta ACT. Una GPU RTX 5060 Ti, Isaac Sim 5.1, `hackathon-robotekin`. Los resultados completos están en `results/`.

| qué | resultado |
|---|---|
| Escenas aleatorias (30 semillas, `check_scene_randomization.py`) | **30/30 válidas**, 0 violaciones del vigilante físico, asentamiento medio 0,79 s; 16 con piezas en contacto, 13 con apiladas, 12 con alguna tapada; modos: mixto 14, separadas 11, apiladas 5. Agarrables de entrada 94/139, alcanzables sin choque 73/139 |
| Contornos de verdad vs. render (`export_yolo_dataset.py --debug`) | coinciden con las piezas, también con el brazo delante |
| Segmentación HSV sobre el render (`eval_segmentation.py`, 36 imágenes) | recall 0,90 y precisión 0,92 sin brazo en piezas ≥ 85 % visibles (85/94); color 100 %; forma 90–93 %; error de posición 1,3–1,6 mm. Falla en piezas del mismo color que se tocan |
| Clasificación completa, 4 órdenes del enunciado (`eval_sorting.py`) | **14/24 y 16/24** escenas correctas según la verdad del simulador en dos ejecuciones con el mismo código (la física no es determinista entre ejecuciones): separadas 8/9, apiladas 2/5, mixto 4–6/10. **0 piezas en caja equivocada.** Agarres 48/73 y 51/79 (≈ 65 %). El robot acierta al creerse acabado en 18–19 de 24; el resto son piezas totalmente tapadas |
| Línea base antes de las correcciones (12 escenas) | 5/12; agarres 9/16 |
| Banco de agarres (`bench_grasp.py`) | cubos admitidos por el modelo ~75 %, cilindros ~65–70 %, piezas que el modelo de dedos descarta 7/19, piezas que descarta el de pared 11/16 |
| 3 demostraciones (`collect_demos.py`) | 3/3 válidas, 0,7 min (≈ 14 s por episodio), formato LeRobot v3 válido, éxito del experto 100 % en esa muestra |
| Entrenamiento ACT diminuto (40 pasos, lote 4) sobre ese dataset | corre, 52 M de parámetros, ~7,5 pasos/s; genera `binpick_info.json` |
| Evaluación offline (`offline_eval_policy.py`) de ese modelo | error 10,8 frente a 6,4 de copiar el estado (mejora −0,69): **lo rechaza**, como debe |
| Ruta ACT en Isaac (`eval_sorting.py --policy act:…`, mismo modelo) | corre sin caídas, etiquetada «APRENDIDA (ACT)», 0/10 agarres («la política no bajó»); con `act+guion` pasa al guion y completa escenas, etiquetando cada agarre |
| Panel y API (`18_voice_bin_picking.py`, `curl`) | imágenes cenital, segmentación y 3/4; errores 503/400/409 claros; detener, aleatorizar y reiniciar; una orden completa hasta su informe y el auditor del simulador coincide. La captura del panel con el navegador **no** se pudo hacer (extensión de Playwright sin conectar) |
| Voz (`faster-whisper small`, CPU) | 6/6 órdenes con voz sintética en español (Microsoft Helena), ~1,3 s por frase; el micrófono se detecta. **Sin persona ni ruido de sala** |

**Lo que Isaac encontró y se corrigió** (ninguna de estas causas era visible sin simulador):

1. **La postura de observación elegida con cinemática era físicamente imposible**: los eslabones se tocaban, el par se saturaba (1,91 N·m) y el brazo no se desplegaba nunca («no alcanza la consigna», 0 agarres en varias escenas).
   Nueva postura `(0, −1,1; 1,2; 1,4; 0)`: reposo a < 0,025 rad, par < 0,65 N·m, despliegue con 0,004 rad de error y solape 0 con contenedor y cajas.
2. **Piezas fantasma en la imagen**: con el antialiasing por defecto (DLSS) quedaban 850–1900 píxeles de la escena anterior tras 2 renders. Ahora FXAA y 8 renders al observar.
3. **El separador de manchas no partía dos cubos del mismo color pegados y escalonados** (sin costura de color ni de profundidad): se añadió un reparto por k-means como último recurso.
4. **El modelo de colisión con la pared era conservador** (11 de 16 piezas descartadas se agarraron): ahora esas piezas, y las que no tienen hueco para los dedos (7/19), se intentan sólo como **último recurso**, avisándolo.
5. **`act+guion` nunca llegaba al guion** con varias piezas (ACT fallaba alternando entre ellas y el supervisor cortaba) y un «la política no bajó» se trataba como avería del brazo: ahora cuentan los fallos seguidos de ACT.
6. **Evaluador**: la métrica «el robot acierta al creerlo» trataba «nada que hacer» como error, y las órdenes pedían colores que no había (3 de 12 escenas perdidas): las órdenes se adaptan ahora a los colores de la escena.
7. **Mensajes que sonaban más seguros de lo que son** con piezas tapadas: ahora dicen que una pieza totalmente tapada no se puede ver.

**Probado y descartado como palanca** (no cambió el resultado): holgura de apertura 12/16/20 mm, profundidad del agarre 0/3/6 mm, usar la pose real en vez de la percibida.

## Cómo validar lo que falta (en este orden, lo ejecutas tú)

**Estado tras la ejecución del 22 de octubre:** los pasos 1 a 5 están hechos (ver arriba; el 4 queda en 58–67 %, en el límite de su criterio), el 9 está hecho con voz sintética y por API, y **faltan el 6 (las 500
demostraciones), el 7 y el 8 (entrenar y medir una ACT de verdad)**. Cada paso tiene un criterio. Hasta que se cumplan **no hay que afirmar que el sistema completo funciona**.

Herramientas de diagnóstico nuevas: `tools/inspect_scenes.py` (verdad frente a percepción por pieza, con el motivo de cada descarte) y `tools/bench_grasp.py` (agarres físicos por variante).

| # | comando | criterio de aceptación | si no se cumple |
|---|---|---|---|
| 1 | `python tools/check_scene_randomization.py --headless --scenes 30` | ≥ 90 % de escenas válidas; hay escenas con contacto, con apiladas y con piezas tapadas; 0 violaciones físicas; asentamiento < 3 s | revisar `cell._drop`, la altura de caída, o el vigilante |
| 2 | `python tools/export_yolo_dataset.py --headless --scenes 6 --views 2 --out yolo_humo --debug 6` y mirar `data/yolo_humo/debug/*.png` | los contornos cuadran con las piezas del render | revisar `segmentation.ground_truth` o la calibración |
| 3 | `python tools/eval_segmentation.py --data data/yolo_humo --segmenter hsv` | precisión ≈ 1,0 y recall alto en piezas ≥ 85 % visibles, **sobre el render de Isaac** | ajustar umbrales de color (`attributes.py`) y de profundidad (`segmentation.py`) |
| 4 | `python tools/eval_sorting.py --headless --seeds 10` (guion) | éxito según la verdad ≥ 60 % de las escenas; **cero piezas en la caja equivocada**; el robot acierta al creerse completado | mirar `results/eval_sorting_*.json`: causa de fallo por escena |
| 5 | `python tools/collect_demos.py --headless --episodes 3 --dataset humo` | 3 episodios válidos, formato válido, y el ritmo (min/episodio) | decidir cuántas horas cuestan los 500 |
| 6 | los 500, por bloques; `python tools/validate_dataset.py data/bin_pick_v1` | «VÁLIDO», tasa de éxito del experto razonable, capas 0 y 1 representadas | si el experto falla mucho en la capa 1: el agarre en pilas es el cuello de botella, no la red |
| 7 | `colab/02_entrenar_act.ipynb` (+ `offline_eval_policy.py`) | error de la política a pasos 25/49 claramente < «copiar el estado» | más pasos/episodios o variantes (`--no-state`, `--no-vae`) |
| 8 | `python tools/eval_sorting.py --headless --seeds 20 --policy act:…` | éxito según la verdad; comparar con el paso 4 y entre `rgb_mask`/`rgb_only`/`mask_only` | DAgger y más episodios (`03_limitaciones.md`) |
| 9 | `python 18_voice_bin_picking.py --voice --view --open` y hablarle | la orden aparece en «He oído» y se entiende bien en «He entendido» | ajustar modelo (`--voice-model`), micro, reglas |

## Distinguir en un informe qué es real y qué es de sustitución

* `eval_sorting.py` imprime y guarda siempre **el tipo**: «APRENDIDA (ACT)» o «GUION programado (NO aprendizaje)», y las tasas de agarre de cada una por separado.
* El panel muestra una etiqueta fija «Política aprendida» / «Guion programado · no es aprendizaje», y el resultado separa los agarres `act` de los `guion`.
* Las pruebas de `tests/` se llaman por su mundo (lógico, cinemático, sintético) en este documento y en sus docstrings; ninguna de sus cifras se cita como resultado del sistema.
