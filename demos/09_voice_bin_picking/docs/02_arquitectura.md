# 2 · Arquitectura, decisiones y plan

## La tubería

```
 micrófono ─► Whisper ─► texto ─┐
 teclado ───────────────────────┴─► commands.parse ─► SortOrder  (reglas: qué piezas → qué caja; avisos y errores en español)
                                                          │
 cámara cenital RGB-D ─► segmentation (HSV | YOLO) ─► Instance[]  (máscara visible, color, forma, posición 3D, visibilidad, región)
                                                          │            ▲
                                                tracker (identificadores estables entre imágenes)
                                                          ▼
                              selection.rank: reglas × graspabilidad (altura, visibilidad, hueco de dedos) + alcance por IK + choques
                                                          │
                                     ┌────────────────────┴───────────────────┐
                                     ▼                                         ▼
                        COGER: ACT (aprendida)  ó  guion (experto)      LLEVAR Y DEJAR: programado (IK del demo 08)
                        RGB + segmentación del objetivo + articulaciones     hueco libre alcanzable de la caja
                        → bloques de 50 consignas a 30 Hz
                                     └────────────────────┬───────────────────┘
                                                          ▼
                              VERIFICAR con la cámara: la pista de la pieza tiene que estar en la caja que toca
                                                          │ no
                                                          ▼
                              RECUPERAR (Supervisor del demo 08): reintento con otra estrategia → otra pieza → abandonar
```

La máquina de estados es `orchestrator.SortingTask` (ENTENDER → VER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERAR). Habla con el robot por un `RobotPort`
(`observe`, `reach`, `pick`, `place`, `capacity`, `home`): lo implementa `robot_port.SimRobot` sobre Isaac y `sim_stub.FakeRobot` para pruebas. Esa costura es también por
donde entraría un SO-101 real con una cámara USB.

## Decisión 1 — ¿ACT? ¿Todo aprendido o híbrido?

**ACT de LeRobot es lo adecuado para la parte de coger**; **llevar y dejar queda programado**. Es la arquitectura que se implementa:

| etapa | quién | por qué |
|---|---|---|
| aproximar, bajar, cerrar, subir | **ACT** | es la parte donde la pieza (posición, giro, vecinas, pared) cambia cada vez y la que enseña una demostración; ACT con bloques de acciones es el estándar de LeRobot para SO-101 |
| transportar, soltar, volver | **programado** | el destino es una caja fija y conocida; aprenderlo gastaría datos (un modelo por caja) para un movimiento que la cinemática hace sin error. Lo único a decidir es un hueco libre, y eso lo ve la cámara |

Qué **no** se afirma: que ACT aprenda a coger piezas en pilas con la fricción de PhysX. El demo 06 aprendió a coger *un cubo en una región de 9×9 cm*, de lazo abierto y con un reloj
privilegiado. Aquí no hay reloj, la política ve la escena (lazo cerrado, a 30 Hz, bloques de 25 acciones) y el objetivo cambia de pieza; si funciona o no **se mide en Isaac**.
Hasta entonces el demo usa el guion y lo dice. Si la política aprendida falla, `--policy act+guion:<ruta>` pasa al guion tras dos fallos con esa pieza, y **cada agarre queda etiquetado**
como `act` o `guion` en el registro, el panel y los informes (`SortReport.picks`, `picks_ok`).

Alternativas descartadas: una sola política para todo el ciclo incluida la caja (más datos, sin ganancia), diffusion/π0 (π0 no arranca en esta máquina, demo 06), y un reloj del plan
como entrada (funcionaba, pero no es imitación desde observación y un robot real no lo tiene).

## Decisión 2 — la observación de la política (RGB, máscaras y articulaciones)

**No se asume que la imagen segmentada sea mejor que el RGB.** Se graban dos flujos de imagen y un estado, y la comparación se hace entrenando con subconjuntos del mismo dataset:

| `--obs-mode` | entradas | qué prueba |
|---|---|---|
| `rgb_mask` *(recomendada)* | RGB recortado + segmentación codificada + articulaciones | RGB completo (paredes, brazo, profundidad aparente) **y** qué pieza coger y cuáles son obstáculos |
| `rgb_only` | RGB + articulaciones | **control**: sin saber *cuál* de las piezas; debería fallar con varias |
| `mask_only` | segmentación + articulaciones | ¿basta la segmentación sola? (se pierden paredes y brazo) |
| `--no-state` (se combina con cualquiera) | quita las articulaciones de las entradas | la lección del demo 06: la red **copiaba la pose de su propio brazo** |

* **`observation.images.overhead`**: RGB de la cámara cenital, recortado a una región fija (contenedor + cajas) y reducido a 288×160. Sin tocar los colores.
* **`observation.images.target`**: imagen codificada, R = máscara de la pieza objetivo, G = las demás piezas, B = interior del contenedor. Lleva toda la segmentación en un solo flujo.
  Se calcula **una vez** al empezar el agarre y se mantiene (dice *qué* coger; dónde está en cada instante lo ve el RGB). Es lo que hizo el demo 06 con el centroide del cubo, y evita
  depender de seguir la pieza mientras la tapa el brazo.
* **`observation.state`**: 6 articulaciones en unidades de LeRobot (−100…100; pinza 0…100; mismo espacio que un SO-101 real).

**Mismo preprocesado en entrenamiento y en uso.** Todo lo que toca píxeles está en `observation.py` (`build_obs`, `to_model_input`); lo llaman igual el grabador (`recorder.py`) y la política
(`policy.py`). Un test (`test_lerobot_compat.py`) graba un dataset real, lo pasa por ACT, guarda el modelo, lo carga con `ACTPickPolicy` y comprueba claves, formas, dtype y rango. El checkpoint
lleva `binpick_info.json` (modo, recorte, segmentador, unidades, fps) y **la política se niega a ejecutarse** si el recorte, el tamaño o la frecuencia no coinciden.

**Contra la copia de pose (confusión causal):**

1. *Ruido de aproximación estilo DART* en el 50 % de los episodios: se ejecuta la consigna con ruido suave y correlacionado (σ = 0,02 rad) **sólo en la aproximación**, y se graba la consigna
   **limpia**. La política ve estados desviados con la orden correcta de volver al camino (`recorder.py`).
2. *Bloques de 50 con 25 ejecutados*: re-observa a mitad de bloque.
3. *`--no-state`* como experimento aparte, y `rgb_only`/`mask_only` como controles.
4. *Limitador de acciones* entre la red y el simulador: recorta a los límites, a la velocidad máxima y rechaza una orden que meta el brazo en la mesa; cuenta cuántas veces actúa.

Lo que no se sabe sin entrenar: cuál de estas variantes gana. Ése es el experimento que prepara el cuaderno 02.

**Una sola cámara cenital y oclusiones.** La máscara es la parte *visible* de la cara superior. Una pieza tapada por otra tiene `visibility` baja; una completamente debajo no existe para esta cámara.
La selección coge primero las de arriba y vuelve a mirar después de cada pieza, así que lo tapado se va descubriendo. Medido con imágenes sintéticas (`04_validacion.md`): ve todas las piezas
enteras (recall 1,0 con ≥ 85 % visible) y sólo pierde las visibles < 35 %. Se podría activar la cámara de muñeca que el demo 08 ya configura (desactivada) para afinar el agarre; no se hizo.

## Decisión 3 — percepción: HSV por defecto, YOLO enchufable

* **HSV + profundidad** (`HsvSegmenter`): el detector del demo 08, más un **separador de manchas** (saltos de profundidad para las apiladas; rectángulo n:1 y *watershed* para las pegadas).
  Es el valor por defecto porque no hay YOLO entrenado y porque es rápido y exacto sobre colores saturados.
* **YOLO-seg** (`YoloSegmenter`): una clase, «pieza». **El color y la forma no los decide la red**: se miden en la máscara (`attributes.classify_color`, relleno del rectángulo mínimo). Así la
  misma red sirve con otros colores y el vocabulario no depende de ella. `colab/01` la ajusta con `tools/export_yolo_dataset.py` (imágenes de Isaac con etiquetas **analíticas**: siluetas
  proyectadas, con las altas tapando a las bajas y el brazo descontado; `--debug` guarda contornos para comprobarlo a ojo).
* **YOLOE** (`yoloe:<pesos>`): por texto, sin entrenar; punto de comparación y arranque para piezas reales.
* Los tres entregan `list[Instance]`; cambiar de segmentador no toca nada aguas abajo. Para comparar, `tools/eval_segmentation.py` evalúa cualquiera sobre las mismas imágenes.
* **Seguimiento** (`tracker.py`): húngaro sobre la distancia en la mesa, con el color obligado; una pieza movida se reasocia con un radio mayor; la tapada por el brazo no se da por perdida.

## Decisión 4 — voz y órdenes: reglas, no modelo

`commands.parse` es un analizador de reglas sobre un vocabulario cerrado (colores, formas, cantidades, izquierda/derecha, «las demás»), con reparación de erratas de reconocimiento
(*«isquierda»*) y errores explícitos (color inexistente, exclusiones, contradicciones). No se usa el modelo del demo 05 porque (a) sus destinos son bandejas de color, (b) cargar un modelo
pesado después de arrancar Isaac rompe el proceso, y (c) con un vocabulario así una regla que falla *lo dice*. **Izquierda/derecha son las del robot** (+y); el panel gira la imagen cenital 90° para
que coincida. «Recoge todas las piezas azules» sin caja **no se adivina**: la orden queda «no entendido», el robot pregunta «¿a la izquierda o a la derecha?» y no se mueve (`--default-box` fija una por defecto a propósito y avisa).

La voz es el `Listener` del demo 05 con el idioma fijado a español; se carga **antes** de arrancar Isaac (`voice.VoiceService.load`).

### Extender a clasificar por defectos u otras características

`attributes.py` es un registro: cada `Attribute` tiene su vocabulario en español y, si existe, su clasificador (`(rgb, máscara) → (valor, confianza)`). `commands.py`, `selection.py` y el
orquestador no saben qué atributos hay. «defecto» ya está declarado **sin clasificador**: pedir «las defectuosas» responde que todavía no sabe, en vez de adivinar. Para conectarlo:

1. escribir el clasificador (clásico: arañazo = contraste local en la cara superior; o una red sobre el recorte de la máscara) y asignarlo a `attributes.get("defect").classify`;
2. que `segmentation.finalize` lo llame y guarde el resultado en `Instance.attributes["defect"]` (hoy sólo llama al de color);
3. nada más: `Selector.extra` ya filtra por él (`test_registering_a_classifier_enables_the_attribute`).

Para *producir* piezas defectuosas en simulación habría que añadir una textura o geometría de defecto a las piezas y etiquetarla; no está hecho.

## Decisión 5 — verificación y recuperación

* El robot **no da nada por hecho**: tras dejar una pieza, vuelve a mirar y exige que la pista de esa pieza esté en la caja que toca. Si está en la mesa (`fuera_de_caja`), volvió al contenedor
  (`deslizada`) o no aparece (`perdida`), es un fallo y se **recupera**: `Supervisor` del demo 08 (otra estrategia con la misma pieza: giro, más abajo y más lento; después otra pieza; después abandonar).
  Las piezas caídas sobre la mesa se pueden recuperar.
* Un agarre se comprueba con la propiocepción (la pinza se bloquea al ancho de la pieza o cierra en vacío) y, para ACT, con la altura del punto de agarre calculada por cinemática directa.
* Al terminar, el lanzador compara lo que cree el robot con **la verdad del simulador** (`BinCell.audit`) y lo escribe en el registro; `tools/eval_sorting.py` juzga el éxito sólo con la verdad.

## Decisión 6 — el entorno aleatorio

Cada escena parte de una semilla: número de piezas (4–6 por defecto, hasta 8), reparto de colores (a veces falta uno, para poder decir «no veo piezas azules»), y el **modo de caída** de cada pieza:
libre, junto a otra, o sobre otra descentrada. **Las poses finales no se escriben**: se dejan caer de una en una y PhysX las asienta (`cell.BinCell`); se valida (dentro, sin atravesar nada, quieta) y si no
vale se repite con otra caída. «Separadas, tocándose, apiladas» son resultados que se **miden** (`scene_gen.analyze_layout`), no promesas del generador. Modos por defecto: separadas, mixto, apiladas;
«juntas» (todas en cadena) es casi un atasco con dedos de 10 mm y sólo sale con `--layout juntas`.

## Plan de entrenamiento (≈ 500 demostraciones y más si hace falta)

| paso | qué | dónde |
|---|---|---|
| 1 | comprobar la escena física | `check_scene_randomization.py` |
| 2 | comprobar el pipeline con el guion | `eval_sorting.py` |
| 3 | 3 demostraciones de humo; después 500 por bloques de 100 | `collect_demos.py` (reanudable, `--first-seed`) |
| 4 | validar el dataset | `validate_dataset.py` |
| 5 | entrenar `rgb_mask` con 0–449 (450–499 quedan sin ver) | `colab/02` |
| 6 | comprobación offline y comparación con «copiar el estado» | `offline_eval_policy.py` |
| 7 | variantes: `rgb_only`, `mask_only`, `--no-state`, `--no-vae` | `colab/02` |
| 8 | el juez: clasificación completa en Isaac | `eval_sorting.py --policy act:…` |
| 9 | si falla en algo concreto: más episodios de ese tipo y afinar (`--init-from`) | `collect_demos.py --resume --layout apiladas`; `train_act.py --init-from` |

Sólo se guardan los episodios que el criterio físico confirma (la pieza se levanta y viaja con la pinza), y los descartes quedan registrados con su causa.
La tasa de éxito del experto en estas escenas es lo primero que hay que mirar tras la tanda de humo: si es baja, el cuello de botella es el agarre en pilas, no la red.

## Compatibilidad con un SO-101 real y una cámara USB cenital

* Acciones y estados en unidades de LeRobot (`range_m100_100`), como el robot real con `lerobot-calibrate`; la conversión sim↔real está en `joints.py` (misma fórmula que LeIsaac).
* La observación depende sólo de la imagen, un recorte fijo definido por calibración (`observation.roi_from_world`) y las articulaciones. En real hay que recalibrar el recorte y volver a
  grabar/ajustar: la imagen real no se parecerá a este render (YOLO ajustado en sim tampoco lo hará).
* El `RobotPort` es la frontera: un `RealRobot` con `observe`, `pick`, `place` encajaría sin tocar el orquestador. No está escrito.
