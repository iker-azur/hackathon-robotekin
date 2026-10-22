# Revisión completa de Demo 11 — SO-101 Autonomous Agent

Fecha: 2026-10-10. Rama: `prepare-demos` (con cambios sin commitear en el árbol de trabajo).
Entorno de referencia: Windows, Conda `hackathon-robotekin`, Isaac Sim 5.1, RTX 5060 Ti.
Línea base verificada: `results/demo11_metrics.json`, `results/demo11_implementation.md`,
`results/grip_and_release.md`, `tools/run_demo11.py`, `out11/*.log`.

Esta revisión no modifica código. Cada hallazgo se marca como **[Confirmado]** (leído en el
código o medido en los informes existentes), **[Probable]** (deducido del código, pendiente de
medida) o **[Mejora]** (opcional, no es un defecto).

---

## A. Resumen ejecutivo

Demo 11 es un sistema de demostración **completo y honesto**: un brazo SO-101 simulado en Isaac
Sim recibe una orden en español, la interpreta con un modelo entrenado, mira la mesa con una
cámara, planifica, mueve el brazo, y **verifica con las poses reales del simulador** si la pieza
acabó dentro de la cesta. Si no acabó, reintenta; si no puede, lo dice. Todo eso se ve en un
panel web en español servido en `http://127.0.0.1:8765`.

Lo que funciona muy bien:

- La **verificación es real**, no una animación: `_placement_verdicts` mide la pose del
  simulador contra la cavidad de la bandeja (`11_real_sorting_with_monitor.py:1373-1390`,
  `10_command_to_motion.py:816`).
- El **agarre no se falsifica**: `StrictExecutor.grab` rechaza el "llevar la pieza a mano" que
  usa Demo 10 (`11_real_sorting_with_monitor.py:456-461`).
- La **capa web es rápida**: 4,0-6,5 ms desde que el navegador envía la orden hasta que el robot
  la toma, y 6,6-7,1 ms de media en `/api/health` (`demo11_metrics.json`). **El panel no es el
  cuello de botella.**
- El JavaScript del panel ya está optimizado: 65 ms y 1 016 escrituras `innerHTML` para 1 000
  instantáneas (frente a 256 ms y 12 013 antes), medido con el arnés Node
  (`tools/benchmark_demo11.py monitor`).

Los tres problemas que realmente importan para el hackathon:

1. **El arranque de Isaac falla a menudo y no hay reintento automático.** Está documentado que
   entre 1 de cada 4 (README de demo 10/11) y 2 de cada 3 arranques (`tools/run_demo11.py`)
   mueren con `Windows fatal exception: code 0xc0000139` dentro del cargador de extensiones de
   Kit. Cada registro de `out11/benchmark_capture_*.log` contiene entre 2 y 4 de esas
   excepciones. El supervisor solo sabe terminar la sesión (`supervisor.py:366-369`): si falla,
   el juez ve «el entorno no ha arrancado» y alguien tiene que volver a pulsar.
2. **La simulación va aproximadamente a la mitad de la velocidad prevista.** Medido:
   `physics_step_ms` = 31,8-46,0 ms por paso (`demo11_metrics.json`) frente a los 60 pasos/s que
   implica el comentario «50 pasos (0,83 s)» de `results/grip_and_release.md:69`. Consecuencia
   directa: un coger-y-colocar son ~820 pasos ≈ **26 s**, y una orden de dos piezas **62 s**
   (`command_total_ms: 61864` con semilla 108). La física es el 85-90 % del tiempo de la orden;
   la percepción entera (~35 ms × 28 miradas) es 1 s.
3. **La demo en vivo usa bolas, que es el caso peor medido.** Con bola roja la sesión HTTP
   terminó FAILED (0/1) en 18,9 s; la semilla 106 terminó 1/2 con dos agarres rechazados; la 107
   terminó 2/2 pero también con dos rechazos. Solo la semilla 108 (dos cubos, dispersión cero)
   quedó 2/2 sin rechazos.

En una frase: **el sistema es técnicamente serio y convincente, pero está calibrado para
"sorprender a un ingeniero" en lugar de para "aguantar 4 minutos delante de un jurado sin
sobresaltos"**. La prioridad no es añadir funciones, es quitar formas de fallar y acortar el
tiempo de espera.

---

## B. Arquitectura actual

### B.1 Componentes y dónde viven

| Componente | Archivo | Responsabilidad |
|---|---|---|
| Supervisor persistente | `supervisor.py` | Proceso que vive toda la sesión: sirve el panel, lanza **un proceso Isaac por sesión**, reenvía órdenes y fusiona estado |
| Servidor web | `monitor/server.py` | FastAPI + uvicorn en un hilo: `/ws`, `/api/state`, `/api/command`, `/api/env`, `/api/config`, `/api/report`, `/api/voice` |
| Panel | `monitor/static/index.html` | Todo el front-end en un archivo, sin dependencias externas |
| Bus de telemetría | `telemetry.py` | Estado, eventos, fotogramas, cola de órdenes y controles, ciclo de vida de cada petición |
| Worker de simulación | `11_real_sorting_with_monitor.py` | Carga Demo 10 por ruta, lo subclasea y lo instrumenta |
| Bucle de Demo 10 | `10_command_to_motion.py` | El orden de operaciones: mirar, enlazar, planificar, mover, comprobar |
| Ejecutor estricto | `StrictExecutor` (en `11_…`) | No falsifica agarres; servo de centrado sobre la apertura real de la pinza |
| Percepción | `perceive.py`, `colors.py`, `detector.py` | Fotograma → objetos con tipo, color y posición |
| Parser de intención | `command_model_v2/`, `intent_parser.py` | Frase → intención, piezas, destino, cantidad y orden |
| Planificador | `planner.py`, `ground.py` | Orden + escena → pasos con poses |
| Escena y materiales | `scene_config.py`, `real_world.py` | Catálogo validado, masas, rozamientos, márgenes, hoja de banco |
| Codificador de vídeo | `frame_encoder.py` | JPEG en un hilo con un solo fotograma pendiente |
| Física real | `rigid_cfg_real` (en `11_…`), `hollow.py`, `pen.py` | Masas por material, bandejas huecas, kerb de contención |
| Arnés de pruebas | `tests/test_demo11.py`, `tests/dashboard_client.js` | 537 comprobaciones, incluido el panel real cargado en un DOM simulado con Node |
| Herramientas de medida | `tools/` | `benchmark_demo11.py`, `probe_*.py`, `verify_demo11_session.py`, `run_demo11.py` |

### B.2 Flujo de datos de principio a fin

```
[navegador] --POST /api/env {action:start}--> [supervisor: Supervisor.start]
                                                       | spawn
                                                       v
                                              [proceso worker]
     monitor.server.serve(bus)                    supervisor.worker()
            |  (hilo, proceso supervisor)                |
            |                                     scene_config.load()
            |                                     real_world.setup_sheet()/preflight()
            |                                     espera del portón (bus.wait_start)
            |                                     CommandModel + HybridParser
            |                                     isaac_io.prepare_runtime()
            |                                     build_real_env()  (gym.make)
            |                                     MonitoredRobot(...)
            |                                            |
   websocket /ws  <--- bus.snapshot() <--- hilo publish <---+--> FrameEncoder --> bus.put_frame
            ^                                                    |
            +---- supervisor.poll() fusiona instantáneas -------+
                                                               |
   [navegador] --POST /api/command--> bus.submit() --> cola --> ask() --> handle()
```

Dos procesos, y es una decisión acertada: Isaac se queda con el hilo principal del trabajador y
el panel nunca comparte GIL con la simulación. La parada del entorno (~2,5-3 s medidas) y el
arranque (~45 s medidas) ocurren sin que el panel deje de responder.

### B.3 El ciclo de una orden (`MonitoredRobot.handle`, `11_…:1936-2080`)

1. Reinicia ledger, lista de tareas y nodos del diagrama.
2. `self.model.parse(text)` (envuelto por `ModelSpy`, `11_…:828-858`) → intención, piezas,
   destino, cantidad, `sort_by`, `sort_filter` y confianzas por campo. Se publica como frase en
   español vía `describe.understood`.
3. `look()` (`11_…:1441-1484`): captura → percepción → anotación → JPEG → bus. Publica además
   `capture_ms`, `segmentation_ms`, `annotation_ms`, `perception_ms`.
4. `link()` (`11_…:1579-1618`): asignación húngara entre manchas y piezas declaradas, por color,
   tipo y distancia al último seguimiento.
5. `planner.plan_command` (interceptado por `install_plan_spy`, `11_…:869-892`): filtra piezas
   medidas como inagarrables y construye los pasos.
6. `capacity.allocate`: reserva huecos no solapados dentro de la cavidad.
7. `_note_tasks` / `_mark_doing`: la lista de comprobación acumula **todas** las tareas del
   pedido, no una por replanificación.
8. `_one_at_a_time` (varias piezas) o `run_plan` (una): `StrictExecutor.run` → `track`/`do` →
   `env.step`. El servo corrige la **pose comandada**, no la mira, porque la cámara tiraba del
   aim hacia la pieza (`11_…:655-807`).
9. `check` → `_verify_and_retry` (`11_…:1404-1434`): mide cada colocación con las poses del
   simulador, reintenta los fallos una vez, aparca, vuelve a medir y **registra un veredicto por
   tarea** en `TaskLedger`.
10. `finally` (`11_…:2069-2080`): estado terminal de la orden (SUCCESS / PARTIAL_SUCCESS /
    FAILED / ABORTED / NEEDS_CLARIFICATION), nodos activos cerrados y `command_total_ms`.

### B.4 Qué es real, qué está simulado y qué falta

- **Real (medido, no fingido):** la física de masas, rozamiento y restitución de
  `rigid_cfg_real`; la cámara renderizada; la segmentación por color y la clasificación
  cubo/bola; el veredicto de agarre leído de las poses del simulador tras el cierre; la
  verificación de colocación contra la cavidad real de la bandeja; el rechazo de piezas
  inagarrables (`fits` calculado con la inclinación); el parser (modelo real + reglas).
- **Simulado por necesidad:** el término de acción IK del brazo (viene de LeIsaac); la cinemática
  de "llevar la pieza" está **desactivada por defecto** (`--allow-kinematic` off) precisamente
  para no inventar resultados; una de las piezas puede ser el cubo del USD de fábrica, con la
  masa que traiga el asset.
- **No implementado o pendiente:** replanificación global (se retira a propósito: cambiaba
  identidades y destinos a mitad de tarea, `11_…:1728-1730`); voz real del navegador probada en
  Chrome/Edge; campaña física con luces distintas, oclusión severa y escenas densas.

### B.5 Dependencias y puntos de fallo

| Dependencia | Punto de fallo |
|---|---|
| Caché de HuggingFace | Sin `HF_HUB_OFFLINE=1` el trabajador necesita red (documentado) |
| Cargador de extensiones de Kit | `0xc0000139` nativo, sin marco de Python: de 1 de 4 a 2 de 3 arranques |
| GPU | `capture_rgb` ~22 ms por render; `env.step` ~32 ms: la GPU la comparte todo |
| Orden de carga | torch y gliner2 deben cargarse **antes** de Kit, si no el proceso muere |
| Proceso único por sesión | Si el trabajador muere, el panel sobrevive pero la sesión termina |
| Hilo principal | Isaac lo posee: todo lo demás (vídeo, web) tiene que cederle el paso |

---

## C. Problemas identificados

### C.1 Críticos

#### C1.1 — Sin reintento automático del arranque de Isaac **[Confirmado]**

- **Qué pasa:** `Supervisor.start` lanza el proceso trabajador una única vez. Si muere antes de
  estar listo, `poll()` espera `startup_timeout` y llama a `terminate()`
  (`supervisor.py:366-369`). No se reintenta nunca.
- **Evidencia:** `tools/run_demo11.py` existe solo para esto y documenta «roughly two launches in
  three were affected»; el README de demo 10/11 dice «about one launch in four». Cada
  `out11/benchmark_capture_*.log` contiene entre 2 y 4 `Windows fatal exception: code 0xc0000139`.
  `out11/verify_sessions_final.log` muestra un `TimeoutError` de la herramienta de verificación.
- **Impacto en la demo:** escenario realista: el juez pulsa «iniciar entorno», espera 45 s y lee
  «el entorno no ha arrancado». Peor todavía si es la primera impresión de la sesión.
- **Solución:** en `Supervisor.poll`, cuando el hijo muera **antes de `ready`, sin informe y sin
  parada solicitada**, reintentar hasta 2 veces con el mismo control, publicando `start_attempt`
  («reintentando 1 de 3») y el motivo en `start-note`. Reutilizar las firmas de fallo de
  `tools/run_demo11.py`. Cada reintento cuesta ~45 s, así que el panel debe mostrarlo con
  cuenta atrás en lugar de quedarse en silencio.
- **Complejidad:** baja (un bucle en `poll`; `terminate` y `start` ya existen).

#### C1.2 — La simulación corre a la mitad de la velocidad prevista **[Confirmado el síntoma, Probable la causa]**

- **Qué pasa:** `physics_step_ms` medido = 31,75 / 32,31 / 32,75 / 44,96 ms
  (`demo11_metrics.json`), es decir 26-31 pasos/s. El código asume 60 pasos/s: «50 pasos
  (0,83 s)» en `results/grip_and_release.md:69`.
- **Consecuencia medida:** `STEP_COUNTS` + `TRACK_*` suman 820 pasos por coger-y-colocar
  (150+200+90+80+120+80+40+60), es decir ~26 s por pieza. `command_total_ms` = 31,6 s para una
  pieza verificada y 61,9 s para dos (semilla 108). El arranque medido son 45 s.
- **Dónde se va el tiempo (semilla 108, 61,9 s):** ~53,7 s de física (1 640 pasos × 32,75 ms),
  ~1,0 s de percepción (28 miradas × 35,5 ms), el resto paradas y asentamientos. **La física es
  el 87 %; la percepción, el 1,6 %.** Optimizar el panel o el parser no puede hacer que la demo
  se sienta más rápida.
- **Hipótesis a comprobar, en orden:**
  1. `env.physics_dt` puede ser 1/30 y no 1/60. Entonces la simulación va a velocidad real y
     todos los comentarios que traducen pasos a segundos están mal por un factor 2. Medida:
     imprimir `env.cfg.sim.dt` y `env.cfg.decimation` al arrancar (junto a la línea de calibración
     del offset de la pinza que ya se imprime en `11_…:2557-2563`).
  2. Si `dt` es 1/60, sobra tiempo dentro de `env.step`. Medidas: comparar `physics_step_ms` con
     `--cam-fps 0` y `--cam-fps 3`, con y sin máscara de brazo, con `--camera top` frente a
     `side`. La instrumentación ya existe (`11_…:2539-2545`).
- **Solución (palanca de bajo riesgo):** reducir pasos donde las medidas lo permitan, con
  `tools/probe_*` como árbitro: `TRACK_DESCEND_STEPS` 200 → 150, `TRACK_TRAVEL_STEPS` 150 → 120,
  `above` 150 → 100, `carry` 120 → 80, `retreat` 60 → 40. De 820 a ~550 pasos: un 33 % menos por
  pieza sin tocar la física. **No tocar** `CLOSE_SETTLE_STEPS` (90, medido: con 50 el agarre
  falla) ni la velocidad de `park` (más lento dispersa 1 893 mm).
- **Complejidad:** media-baja. **Riesgo:** cada recorte debe revalidarse contra el preset medido.

#### C1.3 — La demo en vivo depende de agarres de bola, que es el caso peor medido **[Confirmado]**

- **Evidencia:** sesión HTTP 0, «pon la bola roja en la caja roja» → FAILED, 0/1, 18,9 s.
  Semilla 106 → PARTIAL_SUCCESS 1/2 con 2 agarres rechazados. Semilla 107 → SUCCESS 2/2 con 2
  rechazados. Solo semilla 108 (dos cubos creados con el mismo material, dispersión cero) →
  SUCCESS 2/2 con **cero** rechazos.
- **Impacto:** una bola que rueda o que se inclina sobre su canto presenta 42 mm a una pinza de
  36,2 mm y ya no hay fuerza que la agarre (`11_…:515-581`). Es un fallo sin recuperación
  posible, delante del jurado.
- **Solución:** preset de demo **solo cubos**, semilla fija, `--spread 0`, y un botón que lo
  cargue tal cual. Reservar las bolas para la sección de «mira, esto falla y el robot lo dice»,
  que es un buen momento de guion, no un accidente.
- **Complejidad:** muy baja (configuración y una escena guardada).

### C.2 Altos

#### C2.1 — La línea «ahora:» se queda obsoleta tras terminar el movimiento **[Confirmado]**

- `state.progress`, `state.step_index` y `state.step_total` solo se **escriben**
  (`11_…:1131`, `11_…:1724-1725`) y nunca se borran. La fase sigue siendo `command` o `sorting`
  durante toda la verificación y la recuperación (`_one_at_a_time` pone `phase="sorting"` en
  `11_…:1851` y no se restaura).
- Efecto en el panel: `index.html:1393-1421` sigue pintando
  «ahora: retirar · bola roja → cesta roja · paso 8 de 8» mientras el robot está midiendo dónde
  paró la pieza, recuperando o informando. Si la recuperación registra `lift:` como palabra de
  acción (`11_…:1839`), el verbo desaparece y la línea cae al texto genérico de fase.
- **Impacto:** el panel afirma que el brazo se está moviendo cuando no. Un jurado atento lo nota
  y pierde confianza en el resto de la pantalla.
- **Solución:** al terminar `run_plan` (y al entrar en `verify`/`recover`) publicar
  `progress=""`, `step_index=0`, `step_total=0`, o mejor un booleano `moving` que el panel use
  como condición junto con la fase. Una fuente, no dos.
- **Complejidad:** muy baja. **Riesgo:** ninguno.

#### C2.2 — El diagrama de la ruta de decisión puede recortarse por abajo **[Probable, verificable en 5 minutos]**

- El SVG mide 1 160 × 811 unidades (viewBox calculado en `index.html:987`), se pinta con
  `width: 100%` y su altura sale de la proporción: ≈ 0,70 × ancho ≈ **787 px** con 1 128 px de
  ancho. `#route .body` es `overflow: hidden` y **no tiene `flex`** (`index.html:48`,
  `index.html:115-116`), así que no encoge el SVG: lo recorta.
- En 1 366 × 768 faltan ~85 px y la banda inferior («informe») queda cortada. En 1 600 × 900
  entra justo. Las demás tarjetas del panel sí tienen `flex: 1 1 auto` (`index.html:67`), o sea
  que es una omisión, no una decisión.
- **Impacto:** la pieza central de la demo puede aparecer incompleta en un proyector portátil.
- **Solución:** hacer `#route .body` un contenedor flex y el SVG un ítem
  `flex: 1 1 auto; min-height: 0`, o envolver el SVG en un `position: absolute; inset: 0` dentro
  de un `position: relative`. Con `preserveAspectRatio="xMidYMid meet"` el SVG encoge en lugar de
  cortarse.
- **Complejidad:** muy baja. **Riesgo:** ninguno (se puede medir con el arnés Node).

#### C2.3 — La cámara se congela en las fases sin pasos y el panel avisa «imagen sin actualizar» **[Confirmado]**

- `Executor.settle` llama a `env.step` en bucle directo (`executor.py:634-640`), sin pasar por
  `keys()` ni por `frame_hook`. `_verify_and_retry` hace dos `settle()` más dos `park()`
  (`11_…:1404-1434`).
- El panel marca «· imagen sin actualizar» cuando la imagen pasa de 3 s
  (`index.html:1708`). Durante la verificación hay varios segundos sin fotograma nuevo.
- **Impacto:** justo en el momento de mayor interés (¿acabó dentro?) la cámara parece rota.
- **Solución:** llamar a `cam_pump()` dentro de `settle` (o exponer un `settle_hook` análogo a
  `frame_hook`), y subir el umbral del aviso a 5 s con un texto neutro («el brazo está
  midiendo»).
- **Complejidad:** baja.

#### C2.4 — La lista de tareas, lo más importante, es la letra más pequeña **[Confirmado]**

- Las tareas viven dentro del SVG, en `#plan-tasks` a 12 px (`index.html:404`, `848-852`), con
  `clip` y además dentro de una caja del diagrama. No hay ningún panel de «objetivo».
- **Impacto:** el jurado tiene que leer la cosa más pequeña de la pantalla para entender qué se
  le pide al robot.
- **Solución:** un panel propio y grande — «Objetivo: ordenar las piezas» y debajo 3-5 filas con
  icono, descripción y estado, como en el ejemplo del enunciado. Sacarlo del SVG.
- **Complejidad:** baja (mover marcado que ya existe).

#### C2.5 — No hay modo presentación **[Mejora de alto valor]**

- Para arrancar hay que: pulsar «iniciar entorno», esperar ~45 s, elegir variante, cámara, rondas
  y semilla en una ventana modal, guardar escena y escribir la orden. Todo eso en público.
- `--text` existe pero implica `--no-wait-for-start` y línea de órdenes: no vale para el panel.
- **Solución:** un botón «iniciar demo» que encuele `start` con el preset validado y, al pasar a
  `ready`, encole automáticamente la orden del guion. En CLI, un `--demo` equivalente. Con eso
  la demo se arranca con **una** pulsación.
- **Complejidad:** baja.

#### C2.6 — «siguiente ronda» termina la sesión cuando `--rounds 1` **[Confirmado]**

- `--rounds` vale 1 por defecto (`11_…:2358`). El botón pone `advance = True`
  (`11_…:2164`), el bucle interno rompe (`11_…:2634-2636`) y `range(1)` termina: el trabajador
  sale y el panel pasa a «parado».
- **Impacto:** un juez que pulse «siguiente ronda» cree haber roto la demo.
- **Solución:** rondas por defecto 2-3, o desactivar el botón cuando `rounds <= 1`, o que genere
  una ronda nueva sin cerrar la sesión.
- **Complejidad:** muy baja.

#### C2.7 — La voz del navegador carga 464 MB en el proceso que sirve el panel **[Probable]**

- `transcribe_wav` (`monitor/server.py:114-132`) crea el modelo `small` en el proceso del
  supervisor, que **no** tiene `HF_HUB_OFFLINE=1` (solo lo pone el trabajador,
  `supervisor.py:137`). Si el modelo no está en caché, el supervisor intenta la red y hereda el
  fallo de certificados ya documentado en el README.
- Además, cargar y transcribir en el mismo proceso que atiende el websocket puede congelar la
  imagen unos segundos (GIL).
- **Solución:** `os.environ.setdefault("HF_HUB_OFFLINE", "1")` al importar el servidor, y medir
  el parón con una instantánea antes y después.
- **Complejidad:** muy baja.

### C.3 Medios

#### C3.1 — `queued-note` tiene dos escritores y el segundo siempre gana **[Confirmado]**

- `paintCommand` escribe «en cola (del panel): …» (`index.html:1431`), pero `paintCommands` lo
  sobrescribe justo después en el mismo `applySnapshot` (`index.html:1807`, llamada desde
  `index.html:1834`, después de `paintState` en `index.html:1832`).
- Resultado: el estado `queued` / `queued_from` que `Bus.submit` publica a propósito
  (`telemetry.py:335`) nunca se ve. No es un fallo funcional grave —`paintCommands` muestra el
  ciclo de vida de la orden— pero es la misma información con dos dueños.
- **Solución:** un único escritor. Quedarse con el ciclo de vida y borrar el otro, o usar dos
  elementos distintos.

#### C3.2 — El preset «escena simple» está duplicado y divergente **[Confirmado]**

- El botón del panel codifica la escena a mano en JavaScript (`index.html:2297-2300`) con
  coordenadas distintas de `scene_config.simple_config()` (`scene_config.py:52-62`). La prueba
  `test_simple_preset_uses_uniform_custom_cubes` solo cubre la versión de Python.
- **Solución:** servir los presets desde `/api/config` (o un `/api/presets`) para que haya una
  sola definición. Así el botón y las pruebas comparten fuente.

#### C3.3 — El «paso en curso» se deduce de una palabra, no de un índice **[Confirmado]**

- `paintSteps` compara `step.action.startsWith(progress)` (`index.html:1550`), y `progress` es la
  primera palabra de una línea de registro (`11_…:1131`). Durante la recuperación vale `lift:`
  (`11_…:1839`), que no está en `ACTION_ES` y deja la línea «ahora» sin verdo. Además, el último
  paso del plan queda marcado como «ahora» para siempre después de terminar.
- **Solución:** publicar `step_index`/`step_total` y que el panel marque la fila por índice. Los
  dos campos ya existen y ya se publicen.

#### C3.4 — El supervisor hace ~100 instantáneas por segundo y el barrido de métricas es O(600) **[Confirmado]**

- `Supervisor.poll` itera cada 20 ms y `merge` llama a `bus.snapshot()` para los nodos
  (`supervisor.py:301`) además de la instantánea que ya recibió. `Bus.rate()` recorre hasta 600
  marcas de tiempo por nombre, bajo el lock (`telemetry.py:249-256`).
- No afecta a Isaac (proceso aparte), pero es trabajo innecesario en el proceso que sirve el
  panel, y eso sí se nota cuando el navegador se reconecta.
- **Solución:** reutilizar la instantánea recibida y llevar un índice ventana por nombre en vez
  de barrer el deque.

#### C3.5 — Una orden de charla degrada el resultado físico **[Confirmado]**

- `_rejected_commands` cuenta `NEEDS_CLARIFICATION`, y si el estado físico era SUCCESS lo baja a
  PARTIAL_SUCCESS (`11_…:2070-2071`, `11_…:2665-2666`). Preguntar «¿qué ves?» después de colocar
  dos piezas correctamente convierte el informe en «completado parcialmente».
- **Solución:** separar «órdenes rechazadas» de «tareas físicas fallidas» en el informe.

#### C3.6 — Restos de acciones inexistentes en el panel **[Confirmado]**

- `paintCommands` nombra `reshuffle` y `sort` (`index.html:1804`) y `_apply_control` responde a
  `voice` (`11_…:2231-2235`), pero ninguna de las tres está entre las acciones que el robot sabe
  hacer (`stop`, `next`, `relook`, `save`, `start`; `11_…:2202-2243`).
- **Solución:** borrar las entradas muertas, o hacer visible el rechazo cuando llegue una.

### C.4 Bajos

| # | Hallazgo | Dónde | Comentario |
|---|---|---|---|
| C4.1 | El micrófono graba hasta 8 s sin mostrar tiempo restante ni nivel | `index.html:2050` | Añadir cuenta atrás |
| C4.2 | «con reintento» en la cabecera es jerga interna | `index.html:431` | Renombrar a «recuperadas» |
| C4.3 | Los enlaces de descarga se muestran siempre, incluso cuando el endpoint da 404 | `index.html:679-680` | Ocultarlos si no hay informe |
| C4.4 | `apply_start_params` es código muerto en el modo con supervisor (el único que sirve web) | `11_…:2251-2281` | El supervisor aplica los parámetros en `supervisor.py:263-265` |
| C4.5 | `METRIC_ES` traduce métricas pero el diagnóstico no dice qué es bueno o malo | `index.html:1810-1821` | Añadir umbrales de referencia |
| C4.6 | El registro del plan repinta tareas y pasos una vez por línea | `index.html:1667-1670` | 8 repintados por plan; agrupar en uno |

---

## D. Roadmap de mejora

### P0 — Imprescindible (fiabilidad de la demo)

| # | Acción | Beneficio | Complejidad | Riesgo | Depende de |
|---|---|---|---|---|---|
| P0.1 | Reintento automático (2-3) del arranque de Isaac en `Supervisor.poll`, visible en el panel | Elimina el fallo más probable y más visible | Baja | Bajo: solo hay que no reintentar tras una parada pedida ni cuando ya hay informe | C1.1 |
| P0.2 | Preset de demo validado: 2 cubos, 2 cestas, semilla fija, `--spread 0`; botón que lo cargue | Quita el fallo de agarre de la ecuación | Muy baja | Muy bajo | C1.3 |
| P0.3 | Borrar `progress` / `step_index` / `step_total` al terminar el movimiento | El panel deja de mentir sobre el estado | Muy baja | Ninguno | C2.1 |
| P0.4 | Medir `env.physics_dt` y el coste real de `env.step` sin cámara | Decide todo el plan de rendimiento; sin esto no se sabe si bajar pasos o subir la tasa | Baja (la instrumentación ya existe) | Ninguno | C1.2 |
| P0.5 | Corregir el recorte del SVG de la ruta | La pieza central se ve entera en cualquier pantalla | Muy baja | Ninguno | C2.2 |
| P0.6 | `cam_pump()` durante `settle` y verificación | La cámara no se congela cuando importa | Baja | Bajo: un render más en zonas sin movimiento crítico | C2.3 |

### P1 — Alto impacto (que se entienda y que impresione)

| # | Acción | Beneficio | Complejidad | Riesgo | Depende de |
|---|---|---|---|---|---|
| P1.1 | Panel «Objetivo y tareas» propio, grande, fuera del SVG | Lo más importante se lee de un vistazo | Baja | Ninguno | C2.4 |
| P1.2 | Botón «iniciar demo» de una sola pulsación (preset + orden encolada) | La demo arranca sin choreografía | Baja | Bajo | P0.2, C2.5 |
| P1.3 | Rediseño de la cabecera y la jerarquía visual: VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE | El jurado entiende el guion en 15 s | Media | Bajo: solo marcado y CSS | P1.1, P0.5 |
| P1.4 | Cámara más grande (más ancho de columna o menos tarjetas apiladas) | La prueba visual de que la percepción es real | Media | Bajo | P0.5 |
| P1.5 | Recorte de pasos de físicos validado con `tools/probe_*` (820 → ~550 por pieza) | ~33 % menos tiempo por pieza: una orden de 62 s baja a ~42 s | Media | Medio: cada constante está medida; hay que revalidar el preset | P0.4, P0.2 |
| P1.6 | `HF_HUB_OFFLINE=1` en el supervisor + aviso de voz claro | La voz funciona sin red y sin congelar | Muy baja | Ninguno | C2.7 |
| P1.7 | Rondas por defecto ≥ 2 o botón desactivado | Nadie rompe la demo por curiosear | Muy baja | Ninguno | C2.6 |

### P2 — Opcional (pulido y capacidades)

| # | Acción | Beneficio | Complejidad | Riesgo |
|---|---|---|---|---|
| P2.1 | Presets servidos desde el backend (una sola definición) | Menos deriva entre JS y Python | Baja | Bajo |
| P2.2 | Reintento y explicación accionable ante cierre nativo desde la UI | Recuperación sin consola | Media | Bajo |
| P2.3 | Marcar el paso en curso por índice, no por palabra | Fuente única de verdad | Baja | Ninguno |
| P2.4 | Reutilizar la instantánea en `Supervisor.poll` + métricas O(1) | Menos trabajo en el proceso del panel | Baja | Ninguno |
| P2.5 | Separar «órdenes rechazadas» de «tareas fallidas» en el informe | Informe honesto y legible | Baja | Bajo |
| P2.6 | Panel de confianza del modelo (campo más flojo) como elemento estable | Muestra que hay IA real, no humo | Baja | Bajo |
| P2.7 | Euskera e inglés en el panel | Coherencia con las tres lenguas del parser | Media | Bajo |
| P2.8 | Captura de vídeo o GIF de la sesión | Material para el jurado y para el repositorio | Media | Medio |

---

## E. Demo final propuesta

### E.1 Lo que ve el jurado

Una pantalla, tres zonas, en español y sin una sola palabra técnica:

1. **Cabecera:** título, un punto de estado («simulador en marcha»), la fase en palabras
   («moviendo el brazo») y tres contadores: verificadas, recuperadas, fallidas.
2. **Columna principal (~60 %):**
   - una banda superior fija: **«Ahora: trasladar · cubo rojo → cesta roja · paso 3 de 8»**;
   - debajo, la cadena **VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE** con la
     etapa activa iluminada y una frase viva dentro de cada caja;
   - a la derecha de la cadena, el panel **Objetivo y tareas**, grande.
3. **Columna secundaria (~40 %):** la cámara (grande, con la anotación que el robot usó para
   planificar) y debajo una lista corta de lo que la percepción vio («cubo rojo en (0,38,
   −0,44)»). El registro completo y los avisos quedan detrás de dos iconos.

Nada de: tabla de poses, códigos `#2`, probabilidades por campo, «con reintento», ni diagnóstico
de tasas en la pantalla principal. Eso vive en el registro y en el modal.

### E.2 Secuencia de la demostración (guion de ~3 minutos)

| Minuto | Acción | Qué ve el jurado |
|---|---|---|
| 0:00 | Pulsar **«iniciar demo»** | Una barra de progreso honesta: «arrancando el simulador». El panel ya explica la cadena VER → … → RECUPERARSE aunque no haya robot |
| 0:45 | El simulador está listo | La cámara se enciende con la mesa ya montada: 2 cubos y 2 cestas. La percepción lista lo que ve. La lista de tareas muestra el objetivo |
| 0:50 | Escribir (o dictar) «separa las piezas por colores» | La caja «Entender» muestra la frase «separar un cubo rojo y un cubo azul por color». La lista de tareas se rellena con 2 filas pendientes |
| 0:52 | El brazo se mueve | La cadena ilumina VER y ACTUAR; la línea «ahora» dice aproximar, bajar, cerrar pinza; la cámara muestra el fotograma anotado que el robot está usando |
| ~1:20 | Primera pieza colocada | VERIFICAR se ilumina; la tarea pasa a Completada **después** de medir, no antes |
| ~1:50 | Segunda pieza | Igual. Si una colocación falla, se ve RECUPERARSE: el robot reintenta y la tarea queda Completada o Fallida con motivo |
| ~2:10 | Resultado final | Panel de resultado: «Completado · 2 de 2 tareas verificadas», con la lista de tareas marcada |
| ~2:20 | **(Opcional, muy recomendable)** enviar «pon la bola verde en la cesta verde» con una bola que ruede | El robot lo intenta, el agarre se rechaza y **lo dice**: «las mordazas cerraron y la pieza no quedó sujeta; no la llevo a mano». Es el momento que demuestra que nada está fingido |
| ~2:50 | «Detener» y, si se pregunta, «iniciar demo» otra vez | Reinicio limpio en una pulsación |

### E.3 Criterio de éxito de la demo

- Arranca a la primera, o reintenta solo, sin que nadie toque nada.
- Cada tarea se marca Completada o Fallida **después** de medir, nunca por una animación.
- Ninguna frase del panel afirma algo que el robot no esté haciendo en ese instante.
- El resultado final depende solo del ledger, nunca de una estimación.

---

## F. Checklist de validación

### F.1 Comprobaciones existentes que deben seguir pasando

```powershell
cd demos\05_planner_and_executor
python tests\test_demo11.py                       # 537 comprobaciones: modelo real, bus y panel
python tests\test_planner.py                      # 839 comprobaciones sobre 20 escenas
python tools\benchmark_demo11.py intent           # parser: 87 frases, 3 backends
python tools\benchmark_demo11.py extensions       # catálogo con nombres y euskera
python tools\benchmark_demo11.py monitor          # websocket + arnés Node del panel
python tools\verify_demo11_session.py             # ciclo HTTP completo: iniciar, mover, parar
```

### F.2 Criterios medibles por mejora

| Mejora | Cómo se mide | Criterio de aceptación |
|---|---|---|
| P0.1 Reintento de arranque | `out11/benchmark_capture_*.log`: cuántos arranques llegan a «Escena y cámara listas» sin intervención | ≥ 95 % de los arranques llegan a listos en ≤ 3 intentos, con `start_attempt` visible en el panel |
| P0.2 Preset validado | 5 ejecuciones seguidas del preset, semilla fija, `--spread 0` | 5/5 SUCCESS con **0 agarres rechazados** (línea base: semilla 108 = 2/2, 0 rechazos) |
| P0.3 Estado honesto | Arnés Node: alimentar una instantánea con `phase="command"`, `progress="retreat"`, pasada la ejecución | La línea «ahora» debe decir la fase, no el verbo; 0 afirmaciones falsas |
| P0.4 Coste de `env.step` | `physics_step_ms` y `physics_per_s` del métrico existente, con `--cam-fps 0` y `--cam-fps 3` | Diferencia < 2 ms/paso; `dt` impreso y documentado |
| P0.5 SVG sin recorte | Arnés Node: cargar la página y comprobar que la última banda del diagrama está dentro del área visible | La caja «informe» es visible con `overflow:hidden` a 1 366 × 768 y a 1 920 × 1 080 |
| P0.6 Cámara continua | `camera_reused_frames` y la antigüedad del fotograma durante `_verify_and_retry` | Ningún aviso «imagen sin actualizar» durante más de 2 s en una orden completa |
| P1.1 Panel de tareas | Arnés Node: `taskRows`, y tamaño de fuente efectivo | ≥ 5 filas legibles a 1 m; fuente efectiva ≥ 15 px en 1 920 × 1 080 |
| P1.2 Una pulsación | Cronómetro de pared desde el proceso arrancado hasta la primera orden encolada | ≤ 1 pulsación y ≤ 60 s de espera |
| P1.5 Menos pasos | `command_total_ms` y recuento de pasos ejecutados | Una orden de 2 piezas ≤ 45 s (línea base 61,9 s); agarres rechazados = 0 |
| P1.6 Voz sin red | `HF_HUB_OFFLINE=1` y sin tarjeta de red activa | La transcripción responde y el websocket no pierde instantáneas |

### F.3 Prueba de resistencia previa al hackathon

1. **Diez arranques seguidos** con el botón «iniciar demo». Contar cuántos llegan a listos sin
   tocar nada. Si alguno no llega, P0.1 no está terminado.
2. **Tres órdenes seguidas** sin reiniciar el entorno: el ledger debe acumular sin contaminar el
   resultado de la orden anterior.
3. **Parada en cada etapa**: pulsar «detener» durante aproximar, durante cerrar pinza y durante
   verificar. En los tres casos la orden debe acabar en «Interrumpido» con motivo y el entorno
   debe quedar utilizable.
4. **Reinicio en caliente**: tras parar, «iniciar demo» otra vez y completar la orden.
5. **Pantalla pequeña**: repetir el guion en 1 366 × 768 y con el zoom del navegador al 125 %.
6. **Voz real** en Chrome/Edge: grabar «separa las piezas por colores», corregir el texto
   transcrito y enviarlo.

---

## Notas de alcance

- No se ha modificado ningún archivo del proyecto con esta revisión.
- No se ha ejecutado Isaac Sim: todas las afirmaciones sobre simulación provienen de los informes
  y registros versionados (`results/demo11_metrics.json`, `out11/*.log`,
  `results/grip_and_release.md`) o están marcadas como [Probable] con la medida que las
  resolvería.
- `10_command_to_motion.py` y `executor.py` quedan fuera del alcance de cualquier cambio
  propuesto, salvo la exposición de un `settle_hook` (P0.6), que es aditiva y no altera el
  comportamiento de Demo 10.



