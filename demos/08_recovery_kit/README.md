# Demo 17 — RecoveryKit

Un SO-101 simulado en Isaac Sim prepara kits personalizados ("una pieza roja y
dos azules") cogiendo piezas **por fricción**, comprueba con la cámara lo que
hay de verdad en la caja y **se recupera** cuando el mundo cambia: alguien mueve
una pieza, la pinza pierde la pieza o una pieza acaba fuera de la caja.

```
 VER ──► ENTENDER ──► PLANIFICAR ──► ACTUAR ──► VERIFICAR ──┐
  ▲                        ▲                                │ fallo
  │                        └──────── RECUPERARSE ◄──────────┘
  └──────── cada ciclo empieza mirando la mesa otra vez
```

La caja se comprueba siempre con la cámara: el robot no da nada por hecho
porque "crea" haberlo hecho. Un auditor independiente (verdad del simulador,
fuera del agente) confirma después si el kit es correcto.

---

## Contenido

1. [Cómo se ejecuta](#cómo-se-ejecuta)
2. [Arquitectura](#arquitectura)
3. [Física realista](#física-realista)
4. [Percepción](#percepción)
5. [Recuperación ante fallos](#recuperación-ante-fallos)
6. [Panel web](#panel-web)
7. [Pruebas y validación](#pruebas-y-validación)
8. [Resultados medidos](#resultados-medidos)
9. [Limitaciones y aproximaciones](#limitaciones-y-aproximaciones)

---

## Cómo se ejecuta

Entorno conda `hackathon-robotekin` (Windows 11, Isaac Sim 5.1.0, Isaac Lab
0.47.2 desde el código fuente, LeIsaac 0.4.0 para el modelo del robot, RTX 5060 Ti).

```powershell
cd demos/08_recovery_kit

# panel en http://127.0.0.1:8770 (simulación sin ventana, a tiempo real)
python 17_recovery_kit.py --headless --open

# lo mismo con la ventana de Isaac Sim
python 17_recovery_kit.py --open

# una ejecución sin panel, con una perturbación
python 17_recovery_kit.py --headless --kit "una roja y dos azules" --perturb empujar --realtime 0
```

Opciones útiles: `--preset demo|simple|formas`, `--seed N`, `--port N`,
`--realtime 0` (lo más rápido posible).

En el panel, **Demo guiada** pide "una pieza roja y dos azules" con la
perturbación A preparada (alguien mueve la primera pieza).

El modelo del robot es el `so101_follower.usd` de LeIsaac, ya en la caché local
de Hugging Face (`HF_HUB_OFFLINE=1` por defecto).

---

## Arquitectura

```
17_recovery_kit.py        lanzador: simulación en el hilo principal + panel web en otro hilo
recoverykit/
  runtime.py              arranque de Kit (Direct3D 12, extensiones de Isaac Lab fijadas)
  config.py               escena configurable (JSON versionado) y procedencia de cada parámetro
  workcell.py             mesa, robot, piezas, caja y cámara; única escritura de poses (reinicio)
  physics.py              ajustes de PhysX y vigilante de violaciones físicas
  kinematics.py           cinemática directa e inversa del SO-101 en numpy (misma cadena que el USD)
  control.py              trayectorias con límites del servo; pinza con par limitado
  skills.py               coger y dejar: planes verificables antes de mover, comprobación propioceptiva
  perception.py           cámara RGB-D cenital → piezas, caja y contenido de la caja
  world_model.py          seguimiento de piezas entre imágenes y eventos (movida, en caja, fuera…)
  kit.py                  pedido en español/inglés → objetivo del kit
  planner.py              lo que falta en la caja → tareas ordenadas
  supervisor.py           fallo observado → decisión de recuperación
  agent.py                el bucle cerrado
  perturb.py              perturbaciones físicas (fuerzas, fallo de actuador) — parte del "mundo"
  audit.py                auditor con la verdad del simulador — sólo para evaluar
  telemetry.py            estado compartido con el panel
server/app.py             FastAPI + WebSocket
server/static/index.html  panel en español (three.js local, sin red)
assets/so101/             cadena cinemática y colisionadores extraídos del USD
tools/                    validación física, banco de agarre, percepción, escenarios, alcance
tests/                    pruebas sin Isaac (python -m pytest tests)
results/                  todo lo medido (JSON), con las órdenes que lo reproducen
```

**Lo que el agente puede usar y lo que no.** El agente sólo ve la imagen RGB-D
de la cámara cenital (con su calibración) y la propiocepción de los servos:
posición y carga de cada articulación, lo mismo que reporta un STS3215 real.
No lee poses del simulador ni importa `audit.py`; `tests/test_isolation.py`
falla si algún módulo del agente lo hace. La vista 3D del panel sí usa las poses
del simulador, pero es una ventana a la simulación, no una entrada del agente.

**Por qué no hay un modelo aprendido.** La tarea tiene un vocabulario cerrado
(colores, formas, cantidades) y una geometría conocida (mesa, caja, cámara
calibrada). La visión clásica con profundidad llega a ~1 mm de error y es
medible y explicable; un detector YOLO o una política ACT/SmolVLA añadirían
recogida de datos y entrenamiento sin mejorar esto, y el demo 06 de este
repositorio ya midió que una política ACT entrenada así acumula error en bucle
cerrado. Las reglas del pedido rechazan lo que no entienden en lugar de adivinar.

---

## Física realista

**Ninguna pieza se sujeta por programa.** No hay uniones fijas, ni objetos
cinemáticos en la pinza, ni escritura de poses durante la ejecución. El agarre
funciona porque la mordaza aprieta con el par limitado del servo y la fricción
sostiene la pieza; si algo tira más fuerte, la pieza se suelta.

| Elemento | Valor | Procedencia |
|---|---|---|
| Robot | `so101_follower.usd` (raíz `so101_new_calib`) | URDF oficial SO-101: masas e inercias de CAD (`results/asset_leisaac_so101.json`) |
| Par de los servos | 1.91 N·m | STS3215 a 7.4 V, 19.5 kg·cm (hoja de datos) |
| Velocidad máxima | 4.71 rad/s | STS3215 a 7.4 V, 0.222 s/60° (hoja de datos) |
| Rigidez / amortiguamiento | 30 N·m/rad / 1.0 | **suposición** (sin trazas de hardware); comportamiento medido abajo |
| Pinza | 25 % del par de bloqueo | **suposición**; fuerza normal medida ≈ 3.0 N |
| Piezas | cubos y cilindros de 25 mm, 12 g | PLA impreso ~50 % relleno (**pesar las reales**) |
| Fricción pieza / mesa / mordaza | μ 0.42 / 0.45 / 0.45 | rango de la literatura para PLA y madera |
| Paso de física | 1/240 s, solver TGS, 32/16 iteraciones | medido estable con agarre por fricción |

`config.SOURCES` guarda la procedencia de cada valor.

**Qué se ha medido, no supuesto** (`tools/validate_physics.py`,
`tools/probe_grip_capacity.py`):

- Caída desde 10 cm: reposo en 0.18 s, sin penetración, rebote acorde a la restitución.
- Fricción medida empujando una pieza: estática 0.397 (configurada 0.435),
  dinámica 0.36 (configurada 0.36).
- Deriva en reposo: 0.1 µm en 10 s.
- Servos: velocidad ≤ límite, sobreoscilación 0.019 rad en un escalón, caída
  por gravedad con el brazo extendido 0.0035 rad tras la corrección integral
  (la misma que se haría leyendo `Present_Position` en el robot real).
- **El agarre es por fricción:** la fuerza para arrancar un cubo de la pinza
  cambia con la fricción (2.05 N con μ = 0.2, 2.71 N con 0.45, 3.70 N con 0.9),
  coherente con F ≈ N·(μ_mordaza + μ_pieza) y N ≈ 3 N. Sujeta ~23 veces su peso.
- Un **vigilante** comprueba en cada paso que ninguna pieza salta de posición,
  atraviesa su apoyo o va a velocidad imposible.

**El alcance del SO-101 es pequeño.** Con la pinza vertical, el brazo sólo
llega a todas las alturas del ciclo (agarre, aproximación, subida, tránsito) y
con cualquier giro del cubo en una corona de 0.16–0.19 m alrededor de la base,
algo más hacia la izquierda (`tools/reach_map.py`, `results/reach_map.json`).
Las piezas se reparten en esa corona y la caja está dentro de ella. El
planificador comprueba con la cinemática inversa que una pieza y su hueco en la
caja son alcanzables **antes** de mover el brazo; no lo da por supuesto.

---

## Percepción

Cámara RGB-D cenital calibrada (intrínsecos + pose), 960×720, a 0.54 m de la mesa.

1. Segmentación HSV de colores saturados (la mesa gris y la caja marrón no pasan).
2. Sólo cuentan los píxeles de la **cara superior** (por profundidad): la
   perspectiva enseña también las caras laterales y desplazaría el centro.
3. Profundidad → punto 3D → mundo; forma por relleno del rectángulo mínimo;
   giro de los cubos por `minAreaRect`. Una mancha estrecha (pieza tapada en
   parte) se marca como forma desconocida en lugar de adivinar.
4. La caja se detecta por su color y tamaño conocidos; una pieza está "en la
   caja" si cae en su huella interior.
5. El brazo se proyecta en la imagen desde su cinemática: lo que tapa no se da por perdido.

El brazo descansa girado fuera de la zona de trabajo para que la cámara vea
todo. **El SO-101 es amarillo**: una pieza amarilla sólo se distingue cuando el
brazo no la tapa, así que por defecto no se usan piezas amarillas (se pueden
configurar, con aviso).

---

## Recuperación ante fallos

Las perturbaciones son **físicas** y se disparan en fases del agente, con
magnitud y dirección aleatorias de una semilla (`perturb.py`). El perturbador
forma parte del mundo: no le dice nada al agente.

| Escenario | Qué pasa en el mundo | Cómo lo detecta el agente | Qué decide |
|---|---|---|---|
| **A** mover pieza | al empezar la aproximación, una fuerza desliza la pieza 3–6 cm | la cámara la vigila 4 veces por segundo durante la aproximación | aborta, vuelve a mirar y recalcula el agarre |
| **B** golpe al cerrar | la pieza sale de entre los dedos mientras cierran | la mordaza cierra en vacío (posición del servo) | reintenta con estrategia nueva |
| **B** tirón al subir | un tirón creciente supera la sujeción por fricción | la mordaza se cierra de golpe al perder la pieza | la vuelve a coger donde ha caído |
| **C** fallo de pinza | la pinza se abre sola sobre la pared de la caja | la cámara ve la pieza fuera de la caja | la vuelve a coger desde donde ha caído |
| **C** sacar de la caja | una pieza ya colocada sale despedida | la cámara ve que la caja ya no tiene lo que tenía | vuelve a pedir esa unidad del kit |

Nada de esto es una secuencia fija: el planificador siempre compara el pedido
con lo que la cámara ve dentro de la caja y elige la mejor pieza alcanzable; el
supervisor decide con el historial de intentos de cada pieza:

1. volver a mirar y apuntar al centro medido;
2. girar la pinza a la otra pareja de caras;
3. agarrar 2 mm más abajo y subir más despacio;
4. descartar esa pieza y usar otra igual; si no hay, abandonar y decirlo.

---

## Panel web

`http://127.0.0.1:8770`, en español: cámara del robot con las detecciones,
vista 3D interactiva de la simulación, pedido (los ejemplos rellenan el campo;
sólo se envía con *Enviar*), lo que el robot ha entendido, estado del kit
verificado con la cámara, auditor independiente al terminar, tareas en orden
con su estado, fase actual, fallos y recuperaciones, registro, botones de
perturbación A/B/C, pausar/reanudar/detener/reiniciar/aleatorizar y
configuración del entorno (piezas, tamaño, masa, fricción, caja, cámara,
semilla; aplicar reinicia la simulación). El panel nunca toca Isaac: deja
órdenes en una cola que atiende el bucle de la simulación.

---

## Pruebas y validación

```powershell
python -m pytest tests                                              # sin Isaac
python tools/validate_physics.py --headless                         # física
python tools/benchmark_grasp.py --headless --trials 30 --seed 17001 --tag gate
python tools/probe_grip_capacity.py --headless --mu-jaw 0.45        # ¿fricción o forma?
python tools/capture_perception.py --headless --frames 20
python tools/eval_perception.py
python tools/run_scenarios.py --headless --suite all --runs 3 --seed 17000 --tag final_17000
python tools/run_scenarios.py --headless --suite kit,A_movida,B_golpe,B_tiron,C_pinza,C_sacada --runs 4 --seed 17300 --tag robust_17300
python tools/reach_map.py
```

---

## Resultados medidos

Windows 11, RTX 5060 Ti 16 GB, física en CPU, 17-10-2026. Cada cifra sale del
JSON indicado en `results/`.

**Agarre físico estricto** (`grasp_benchmark_gate.json`, 30 ensayos por forma,
posiciones y giros aleatorios con semilla 17001, objetivo tomado de la verdad
del simulador para medir sólo la manipulación):

| Forma | Levantadas | En la caja | Veredicto del agente = verdad | Error de colocación (mediana) | Deslizamiento máx. |
|---|---|---|---|---|---|
| Cubo 25 mm | 30/30 | 30/30 | 30/30 | 4.3 mm | 6.8 mm |
| Cilindro 25 mm | 30/30 | 30/30 | 30/30 | 3.8 mm | 7.1 mm |

**Percepción** (`perception_eval.json`, 20 imágenes, 7 piezas por imagen,
algunas dentro de la caja y algunas con el brazo tapando):

| Métrica | Valor |
|---|---|
| Piezas visibles encontradas | 98.5 % (134/136) |
| Falsas detecciones | 0 |
| Error de posición | 0.95 mm de media, 1.2 mm p95 |
| Forma correcta | 97 %; 0 incorrectas, 4 marcadas como desconocidas (tapadas en parte) |
| Dentro/fuera de la caja | 100 % |
| Giro de los cubos | 0.7° de media |

**Agente completo** (`scenarios_all_x3.json`, 7 escenarios × 3 semillas,
17000–17002, perturbaciones físicas):

| Métrica | Valor |
|---|---|
| Kits correctos según el auditor independiente | 21/21 |
| Kits que el agente dio por buenos y no lo eran | 0 |
| Lo que el agente cree que hay en la caja = lo que hay | 21/21 |
| Ejecuciones con algún fallo detectado | 15/21; recuperadas 15/15 |
| Éxito de agarre (incluye los agarres estropeados a propósito) | 86 % |
| Duración media (tiempo simulado) | 42.3 s |
| Factor de tiempo real medio (sin panel) | 1.58 |

| Escenario | Duración media | Fallo detectado |
|---|---|---|
| una pieza | 13.7 s | — |
| kit 1 roja + 2 azules | 39.5 s | — |
| A mover pieza | 43.4 s | movida (3/3) |
| B golpe al cerrar | 45.8 s | cierre en vacío |
| B tirón al subir | 47.5 s | pieza deslizada |
| C fallo de pinza | 53.5 s | fuera de la caja |
| C sacar de la caja | 52.6 s | fuera de la caja |

En esa serie el vigilante marcó 5 violaciones de velocidad en dos ejecuciones
del tirón: la pieza salía disparada a más de 3 m/s porque el tirón seguía
creciendo después de soltarse. Se corrigió (el tirón deja de crecer al empezar
a deslizar) y con 5 semillas más (`scenarios_tiron_x5.json`) hay 0 violaciones;
en 4 de 5 el tirón arranca la pieza y el agente la recupera, en 1 la sujeción aguanta.

Por el panel, la **Demo guiada** (1 roja + 2 azules con la perturbación A) tardó
43.5 s de reloj a tiempo real, con el auditor de acuerdo y sin errores en la consola.

**Pendiente de volver a medir con el código final.** La última tanda
(`--tag final_17000` y `--tag robust_17300`) se interrumpió por falta de
memoria del sistema tras 10 de 21 ejecuciones (todas correctas según el
auditor, 0 violaciones); sus JSON no llegaron a escribirse. También hay que
repetir `tools/validate_physics.py`: su última ejecución falló en las pruebas de
servo por un error de la propia prueba (el destino quedaba fuera del límite del
hombro tras cambiar la postura de partida), ya corregido. Las cifras de física
de este README son de la ejecución anterior que pasó 14/14.

---

## Limitaciones y aproximaciones

- **Rigidez y amortiguamiento de los servos** son una suposición razonada, no
  una identificación: falta medir la respuesta de un STS3215 real.
- **La fuerza de la pinza** (25 % del par de bloqueo, ≈3 N de apriete medido)
  también es una elección; en el robot real depende de `Torque_Limit` y del P del servo.
- **Fricción y masas** salen de rangos de la literatura; pesar las piezas reales
  y medir la fricción con un plano inclinado antes del hackathon.
- **Alcance:** con la pinza vertical, el SO-101 trabaja en una corona estrecha.
  El sector actual llega hasta 0.205 m, y el mapa de alcance muestra que más
  allá de 0.195 m, con ángulos de −60° o menos, no todos los giros son
  alcanzables; el planificador lo detecta y elige otra pieza, pero conviene
  estrechar el sector a 0.16–0.195 m.
- **Caja fija a la mesa** (como pegada con cinta); una caja suelta no está implementada.
- **Máximo 4 piezas por kit** (huecos 2×2 de la caja); el pedido lo comprueba.
- **Prismas:** el generador de la forma no está implementado; se usan cubos y cilindros.
- **Amarillo:** el brazo es amarillo; las piezas amarillas sólo se ven cuando el brazo no las tapa.
- **Cámara de muñeca:** configurable pero desactivada; el agente no la necesita.
- **Física en CPU:** suficiente para esta célula (factor de tiempo real ≈1.5);
  no se ha probado en GPU.
