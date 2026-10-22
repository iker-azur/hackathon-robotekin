# Demo 18 — Clasificación de piezas por voz

Una orden en español → el robot mira un contenedor con piezas amontonadas → las segmenta → coge la elegida con una política
de imitación → la lleva a la caja izquierda o derecha → **comprueba con la cámara** que está donde toca, y si no, vuelve a mirar y reintenta.

```
 voz ─► texto ─► orden ─► cámara ─► segmentación ─► elegir pieza ─► política de agarre ─► llevar y dejar ─► verificar
 Whisper     reglas     cenital     HSV | YOLO       puntuación        ACT (aprendida)      programado          cámara
                                                     por visibilidad   o guion (no aprendido)                    │
                                                     y hueco de dedos                                            └─► si falla: recuperarse
```

> **Léelo antes de presentarlo.** Este demo se ha **ejecutado en Isaac Sim (22 de octubre de 2026)**, y eso corrigió varios fallos que las pruebas sin simulador no veían
> (ver [`docs/04_validacion.md`](docs/04_validacion.md)). Con el **guion del experto** (que *no es aprendizaje*; el panel lo dice en una etiqueta fija) clasifica bien
> **entre 14 y 16 de cada 24 escenas** (58–67 %; separadas 8/9, apiladas 2/5, mixto 4–6/10), con **0 piezas en la caja equivocada**; los agarres por fricción salen en ≈ 65 %.
> **No hay ningún modelo ACT ni YOLO entrenado**: la ruta ACT se probó con un modelo de 40 pasos que sólo demuestra que la tubería corre. **No se ha probado con micrófono real ni con un SO-101 real.**
> Una pieza totalmente tapada por otras no se ve, y la orden queda sin cumplir (el robot lo dice).

## Qué está verificado y qué no

| | estado | cómo se sabe |
|---|---|---|
| Interpretar órdenes en español (los 4 ejemplos del enunciado y variantes, erratas de Whisper, errores claros) | ✅ verificado | 33 pruebas |
| Segmentar piezas pegadas/apiladas/tapadas, color, posición, visibilidad | ✅ en imágenes **sintéticas** (17 + 3 pruebas); en renders de Isaac, ver la fila de segmentación más abajo | 17 + 3 pruebas |
| Elegir pieza (altura, visibilidad, hueco para los dedos, alcance, choques con paredes) | ✅ verificado con la cinemática real del SO-101 | 16 + 7 pruebas |
| Máquina de estados con reintentos, caja llena, verificación por cámara, parada | ✅ lógica; ✅ de punta a punta en un simulador **cinemático** | 19 + 6 pruebas |
| ACT de LeRobot acepta el dataset grabado, la inferencia reproduce el preprocesado, la config de entrenamiento valida | ✅ (sin entrenar) | 7 + 17 pruebas |
| Panel en español, voz→orden, imágenes, WebSocket | ✅ verificado (sin micrófono real) | 13 pruebas |
| **Escena física** (piezas que caen, se asientan y se apilan en PhysX) | ✅ **ejecutada en Isaac**: 30/30 escenas válidas, 0 violaciones, 0,8 s de asentamiento | `tools/check_scene_randomization.py` |
| **Segmentación sobre el render de Isaac** | ⚠️ recall 0,90 / precisión 0,92 sin brazo (piezas ≥ 85 % visibles); **falla con piezas del mismo color pegadas** | `tools/eval_segmentation.py` |
| **Agarre por fricción dentro del contenedor** con el guion | ⚠️ **≈ 65 %** (cubos ~75 %, cilindros ~65–70 %); el demo 08 midió 30/30 con piezas sueltas en la mesa | `tools/bench_grasp.py`, `tools/eval_sorting.py` |
| **Clasificación completa** con el guion (4 órdenes del enunciado) | ⚠️ **14–16 de 24** escenas bien, 0 en caja equivocada | `tools/eval_sorting.py` |
| **Demostraciones, ACT diminuto, evaluación offline, ruta ACT y `act+guion` en Isaac** | ✅ la tubería corre de punta a punta (3 episodios en 0,7 min); el modelo de 40 pasos **no sirve** y la evaluación offline lo rechaza | `collect_demos.py`, `train_act.py`, `offline_eval_policy.py` |
| **Panel, API y voz** | ✅ panel y API en Isaac; voz con síntesis de Windows 6/6 (sin persona ni micrófono real) | `18_voice_bin_picking.py` |
| **Política ACT entrenada con las demostraciones** | ❌ no existe aún | `colab/02_entrenar_act.ipynb` |
| **YOLO-seg entrenado** | ❌ no existe aún (no había ninguno en el repositorio) | `colab/01_entrenar_yolo_seg.ipynb` |

Detalle de qué prueba cada cosa y qué orden seguir para validarlo: [`docs/04_validacion.md`](docs/04_validacion.md).

## Qué se reutilizó y qué es nuevo

* **Reutilizado sin copiar** (se importa del demo 08 por `sys.path`): cinemática y control del SO-101, física validada (agarre por fricción, vigilante),
  cámara cenital calibrada, `Supervisor` de recuperación, `Telemetry`; y del demo 05, el `Listener` de Whisper.
* **No reutilizado, con motivo:** el modelo de órdenes GLiNER del demo 05 (sus destinos son bandejas de color, no izquierda/derecha, y pesa cientos de MB);
  los cuadernos de Colab (no había ninguno); YOLO (no había modelo).
* **Nuevo:** escena aleatoria con caída física, segmentación de instancias (HSV con separador de manchas + envoltorio de YOLO/YOLOE), selección por graspabilidad,
  seguimiento, máquina de estados, política ACT con observación RGB + segmentación, recogida de ~500 demostraciones, panel, herramientas y cuadernos.

Revisión completa: [`docs/01_revision_inicial.md`](docs/01_revision_inicial.md) · arquitectura y decisiones: [`docs/02_arquitectura.md`](docs/02_arquitectura.md) ·
límites y mejoras: [`docs/03_limitaciones.md`](docs/03_limitaciones.md).

## Ejecutarlo

Entorno `hackathon-robotekin` (Isaac Sim 5.1, Windows). Desde `demos/09_voice_bin_picking`, en PowerShell:

```powershell
conda activate hackathon-robotekin
$env:HF_HUB_OFFLINE = "1"          # sin red: evita el error SSL de Hugging Face

python 18_voice_bin_picking.py --open                       # panel en http://127.0.0.1:8780 · guion del experto
python 18_voice_bin_picking.py --voice --view --open        # + micrófono + vista 3/4 en el panel
python 18_voice_bin_picking.py --headless --text "Clasifica las piezas rojas a la izquierda y las verdes a la derecha"
```

Cierra otros Isaac Sim antes (`Get-Process python`): dos a la vez en la GPU provocan el fallo intermitente de arranque
(`illegal memory access`; repite el comando). En el panel: escribe o pulsa **Hablar**, **Iniciar**, y mira la cámara (con o sin segmentación),
la tabla de piezas, las etapas y el registro. **Detener**, **Reiniciar** (misma escena) y **Aleatorizar** (otra).

Órdenes que entiende (izquierda/derecha son las **del robot**; el panel gira la imagen para que coincida):
«Coloca las piezas rojas en la caja izquierda» · «Pon las piezas azules en la caja derecha» ·
«Clasifica las piezas rojas a la izquierda y las verdes a la derecha» · «Recoge todas las piezas azules» (sin caja → la derecha, y lo avisa) ·
«las rojas a la izquierda y las demás a la derecha» · «pon dos piezas rojas en la caja izquierda» · «los cilindros azules a la derecha».
Lo que no entiende lo dice (colores inexistentes, exclusiones, «defectuosas» —aún sin clasificador—).

## Entrenarlo (lo ejecutas tú)

```powershell
# 0. ¿la escena es variada y válida en Isaac? (30 escenas, minutos)
python tools/check_scene_randomization.py --headless --scenes 30

# 1. ¿el pipeline completo funciona con el guion? (el juez real, con la verdad del simulador)
python tools/eval_sorting.py --headless --seeds 20

# 2. demostraciones: humo (3) y luego las 500, por bloques reanudables
python tools/collect_demos.py --headless --episodes 3 --dataset humo
python tools/collect_demos.py --headless --episodes 100 --first-seed 0
python tools/collect_demos.py --headless --episodes 100 --first-seed 1000 --resume     # …hasta 500
python tools/validate_dataset.py data/bin_pick_v1

# 3. entrenar en Colab: sube data/bin_pick_v1 comprimido y abre colab/02_entrenar_act.ipynb
# 4. medir ACT de verdad
python tools/eval_sorting.py --headless --seeds 20 --policy act:outputs/act_rgb_mask/pretrained_model
python 18_voice_bin_picking.py --policy act+guion:outputs/act_rgb_mask/pretrained_model --voice --open

# diagnóstico: ¿qué ve la percepción de cada pieza y por qué no se alcanza? · ¿qué falla en el agarre?
python tools/inspect_scenes.py --headless --seeds 5 24 26          # imágenes en out/inspect/, motivo por pieza
python tools/bench_grasp.py --headless --seeds 12 --variants m12 m16 pared_sin_modelo dedos_sin_modelo

# YOLO (opcional): exportar etiquetas, entrenar en colab/01_entrenar_yolo_seg.ipynb, evaluar contra HSV
python tools/export_yolo_dataset.py --headless --scenes 400
python tools/eval_segmentation.py --data data/yolo_pieces --segmenter hsv
python tools/eval_segmentation.py --data data/yolo_pieces --segmenter yolo:piezas_seg.pt
```

Sin red, `colab/` no sirve: los cuadernos instalan `lerobot==0.4.4` / `ultralytics` y leen de Drive. Para YOLO en local:
`python -m pip install ultralytics --no-deps` (no se ha probado en este entorno; si rompe numpy, vuelve a `--segmenter hsv`).

## Pruebas (sin Isaac)

```powershell
python -m pytest tests -q            # 196 pruebas, ~2,7 min (la mayoría de los minutos son IK y vídeo)
```

`sim_stub.FakeRobot` (lógico) y `tests/fake_cell.py` (cinemático) **no son la simulación física**: sirven para probar la lógica y la integración.

## Estructura

```
18_voice_bin_picking.py     lanzador: simulación en el hilo principal, panel en otro
binpick/
  config.py                 escena (contenedor, 2 cajas, cámara, ROI) y constantes
  commands.py attributes.py voz→orden (reglas) y registro de atributos (color, forma; «defecto» sin clasificador)
  scene_gen.py cell.py      escenas aleatorias y caída/asentamiento con física
  segmentation.py           HSV con separador de manchas · YOLO/YOLOE · verdad analítica
  tracker.py selection.py   seguimiento y elección de pieza (graspabilidad)
  picking.py collision.py   plan de agarre con subida por capa; choques con paredes
  observation.py            LA observación de la política (único preprocesado)
  policy.py pick_runner.py  ACT/guion, limitador de acciones, ejecución a 30 Hz
  place.py                  hueco libre alcanzable en la caja (parte programada)
  orchestrator.py           máquina de estados + recuperación
  robot_port.py             el robot de Isaac como RobotPort
  recorder.py collect.py    demostraciones en LeRobot v3 (ruido de aproximación estilo DART)
  dataset_check.py seg_eval.py   validación de datasets y evaluación de segmentadores
  server.py telemetry.py viz.py voice.py   panel
static/index.html           panel en español
tools/                      todo lo que mide o genera (cada uno dice qué lee y qué escribe)
colab/                      01 YOLO-seg · 02 ACT  (generados por tools/build_notebooks.py)
tests/  docs/  results/
```
