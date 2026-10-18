# Demo 12 — Orquestación autónoma multi-robot sobre Isaac Lab real

Varios brazos SO-101Autowired ordenar piezas en bandejas, con la física resuelta por
**Isaac Lab / PhysX**. La orquestación decide quién hace qué; la simulación decide si eso
succeeded. El orquestador nunca escribe una pose: solo objetivos articulares, y después lee
lo que PhysX hizo.

> **Anterior:** este README describía una arquitectura que ya no existe —
> `SimulatedSO101` + `WarehouseSimulation`, `isaac_multi_backend.py`, y un modo
> `--no-isaac`/`--headless-run` que corría la demo sin simulador. Todo eso se eliminó:
> una ruta de simulación falsa hace imposible distinguir un picking que funciona de uno que
> solo se representa. La autoridad es `demo12/isaac_env.py` y nada más.

---

## La regla que sostiene todo

**Una tarea solo se marca completada si la pieza está dentro de la cavidad del destino y en
reposo.** Ambas cosas se leen del cuerpo rígido de la pieza. No hay modo de simulación, ni
rutas de respaldo, ni estados de éxito cableados.

Esto no esiums decorativo: durante la reconstrucción hubo un agarre que reportaba **103 N**
de fuerza de contacto mientras no sostenía nada, porque los dedos se cerraban sobre una
esquina en vez de sobre las caras. Solo la posición final de la pieza delataba el problema.

---

## Cómo se ejecuta

Isaac **no** corre con el `python` del PATH (3.12, sin `isaaclab`). Usa el intérprete del
entorno:

```
C:\Users\Usuario\miniconda3\envs\hackathon-robotekin\python.exe
```

Headless, extremo a extremo:

```powershell
$env:PYTHONPATH = "C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\07_multi_robot_orchestration"
& "C:\Users\Usuario\miniconda3\envs\hackathon-robotekin\python.exe" -m demo12.worker --robots 2
```

Con Isaac Sim visible:

```powershell
... -m demo12.worker --robots 2 --gui
```

Servidor + interfaz:

```powershell
... -m demo12.launcher --robots 2 --port 8000    # http://127.0.0.1:8000
cd demo12\web; npm run dev                      # http://localhost:5173
```

Para **cambiar el número de robots en caliente** (lo único geométrico que no se puede hacer
dentro del proceso, porque reconstruir la celda se cuelga — ver `docs/PHYSICS.md` §9):

```powershell
'{"robots": 3}' | Set-Content demos\07_multi_robot_orchestration\out12\control.json
```

Para **cambiar la escena** (piezas, bandejas, posiciones) también por reinicio, con el botón
*Aplicar y reiniciar* del editor o a mano:

```powershell
'{"scene": "scene.json"}' | Set-Content demos\07_multi_robot_orchestration\out12\control.json
```

Lo que **sí** cambia en caliente, al instante y sin reiniciar, es **el plan de trabajo**: qué
pieza va a qué bandeja. Es estado de orquestación, no física.

Misión densa, con las cuatro piezas teniendo tarea (es el caso con contención de verdad):

```powershell
... -m demo12.worker --robots 2 --all-tasks
OK: 4 task(s) completed in 148.3 s simulated, 8242 control ticks
```

El supervisor reinicia el proceso y restaura la cola de tareas. Medido de punta a punta:
2 robots → 3, con `{"restored": true, "tasks": 2}`. Lo que **no** sobrevive es la posición de
las piezas — se reaparecen en su sitio — así que cualquier tarea marcada como completada se
vuelve a encolar, con una nota que lo dice.

Sondas de medición (verifican todo lo afirmado en `demo12/docs/PHYSICS.md`):

```powershell
... -m demo12.tools.probe_grasp_pose   # orientación de la mandíbula -> out12/pose_probe.json
... -m demo12.tools.probe_grasp        # los cuatro criterios  -> out12/grasp_probe.json
... -m demo12.tools.probe_frame        # captura de viewport    -> out12/frame_probe.json
```

### La imagen

`/api/frame.jpg` **funciona**, en headless, servida en vivo mientras el bucle de orquestación
avanza. Medido: 10.498 bytes de JPEG por HTTP, con la celda de dos robots a 280 ticks y cero
errores. Sale de una prim `UsdGeom.Camera` explícita leída por `isaaclab.sensors.Camera`, **no**
de una captura del viewport: el viewport de esta máquina no tiene cámara
(`has_primed_camera() == False`), y por eso esa ruta no puede dar imagen en ninguna
configuración.

Lo que hizo falta está en `demo12/docs/PHYSICS.md` §9. En corto: el sensor se inicializa con un
evento **PLAY** del timeline que un `SimulationContext` suelto nunca dispara; `SensorBase` se
traga la excepción real en `builtins.ISAACLAB_CALLBACK_EXCEPTION`; no hay `initialize()`
público; y tras `reset()` hay que ejecutar unos cuantos pasos de simulación antes del primer
`read` — si no, el anotador de Replicator pide cero frames y falla con
`expand(torch.cuda.ByteTensor{[360, 640, 3]}, size=[0])`.

Coste: renderizar cada `CAMERA_RENDER_EVERY_STEPS` ticks baja el rendimiento del bucle de
control. La física no se resiente.

Pruebas:

```powershell
... -m pytest demo12\tests -q          # 129 rápidos (1.5 s) + 7 en Isaac (~11 min)
```

Las rápidas incluyen dos que valen más que el resto:

- **`test_server.py`** ejecuta las 23 rutas y el WebSocket contra una celda guionizada, sin
  GPU y en 1,7 s. No es cobertura decorativa: cuando se escribieron esas rutas no se habían
  ejecutado nunca, y a la primera ejecución aparecieron seis defectos — un `500` en el editor
  de escena, un `422` en el panel "por qué este robot", y un `KeyError` que hacía que
  **todos los botones de planificación parecieran vivos y no hicieran nada**.
- **`test_contract.py`** compara los tipos que declara `api.ts` con lo que el servidor envía
  de verdad. `tsc` compila el cliente contra sus propias declaraciones y no aprende nada de
  lo que produce ningún endpoint; ese fue el agujero por el que se coló el
  `part_id` / `partId`.

Los dos están en `docs/PHYSICS.md` §11f.

Y una que sí necesita la GPU, para lo que una celda guionizada **no** puede comprobar: la
celda real de Isaac detrás de las rutas, el WebSocket sosteniendo el paso durante segundos
seguidos, y el reinicio del supervisor. Arranca el supervisor de verdad, lo interroga, le
pide 3 robots por `control.json` y comprueba que la cola **sobrevivió**:

```powershell
... -m demo12.tools.verify_live --port 8077 --robots 2 --restarter 3
```

Eso encontró un fallo que ninguna otra prueba podía ver: el supervisor pedía guardar la cola
antes de matar el proceso, el endpoint **encolaba** esa petición en vez de hacerla, y
`terminate()` landing antes de que el hilo de la simulación drenara una sola entrada. El
archivo nunca se escribía y el reinicio volvía con la cola vacía — indistinguible de un
reinicio correcto. §11g.

También encontró que **el supervisor se dejaba su propio hijo huérfano**: `terminate()` sobre
él es `TerminateProcess`, así que su manejador de `SIGTERM` —el que para al servidor— no se
ejecuta nunca. Dos procesos de Isaac huérfanos se quedaron con la GPU, y por eso media suite
"tardaba veinte minutos por test" cuando lo que pasaba era que cuatro procesos de Isaac
compitían por una RTX 5060 Ti.

La tercera comprobación es la de recargar la escena, y mueve una **coordenada de verdad** en
vez de sólo comprobar que se leyó un fichero: si el proceso nuevo construyera la celda por
defecto en silencio, la interfaz estaría enseñando una edición que no está ejecutando.

Sonda de alcance, que es la que distingue "no llega" de "nunca estuvo ahí":

```powershell
... -m demo12.tools.probe_reach         # alcance real por brazo -> out12/reach_probe.json
```

---

## La interfaz

Cuatro vistas, porque son cuatro preguntas distintas.

| Vista | Para qué |
|---|---|
| **Celda** | Qué está pasando ahora: el mapa de la celda en planta, la cámara y la prueba del agarre de la tarea seleccionada. |
| **Plan** | Qué quieres que pase: asignar pieza → bandeja con un clic, escribir un comando, o elegir un escenario. |
| **Colaboración** | Quién hace qué y por qué: la línea de tiempo de decisiones, la carga por robot y las métricas. |
| **Editor** | La geometría de la celda. Esto **sí** cuesta un reinicio, y lo dice. |

El **mapa de la celda** es la pieza central: tres brazos moviéndose a la vez es justo lo que
no se puede seguir con una cámara. El mapa responde lo que la cámara no — qué robot lleva qué
pieza, a dónde va, quién espera a qué, y cuán lejos están los brazos. Todo lo que se dibuja
ahí es una medición o una decisión ya tomada: las piezas vienen de los cuerpos rígidos, los
alcances de la envolvente medida, las flechas de la lista de tareas. Una pieza que no se ha
movido no se dibuja moviéndose.

**El mapa es también el planificador.** Se hace clic en una pieza y luego en una bandeja, y
eso genera la tarea. El comando en texto es la otra puerta:

```
mueve el cubo rojo a la caja azul
send the green cube to the blue tray
part_4 -> bin_2
send all to bin_2
```

Se interpretan en español e inglés, sobre ids, colores, formas y etiquetas. El comando se
**previsualiza antes de aplicarse**: la primera pulsación dice qué ha entendido, la segunda
lo aplica.

Todo en **ES/EN**, con el interruptor en la cabecera.

### Endpoints

```
GET  /api/health  /api/state  /api/metrics  /api/frame.jpg
GET  /api/catalog                colores, formas y etiquetas, en ES y EN
GET  /api/scene                  la celda tal cual está, para el editor
GET  /api/roster                 qué robots hay y cuál está de turno
GET  /api/events?since=N         la línea de tiempo, por cursor
GET  /api/scenarios              asignaciones de un clic
GET  /api/tasks/rank?taskId=     por qué este robot, con la aritmética
POST /api/scene/validate         comprueba una escena sin construirla
POST /api/scene/apply            guarda y pide el reinicio
POST /api/plan                   {partId,binId}[] -> tareas validadas
POST /api/command                texto -> plan (apply: true lo aplica)
POST /api/tasks                  una tarea concreta
POST /api/tasks/submit           plan -> cola, en vivo
POST /api/tasks/clear            vaciar la cola
POST /api/run  /api/stop
POST /api/fault/{robot_id}  /api/faults  /api/repair/{robot_id}
POST /api/relaunch-plan          reaplica el plan guardado tras un reinicio
WS   /ws
```

`/ws` está registrado como `WebSocketRoute` de Starlette y no con `@app.websocket`: en un
proceso con Isaac arrancado, la ruta de FastAPI hace *match* y responde `websocket.close`
sin llegar nunca al manejador, y uvicorn convierte eso en un 403 sin explicación. Está en
`docs/PHYSICS.md` §14.

---

## Capas

Cada capa depende solo de las de arriba. La frontera importa: `orchestrator.py` no importa
nada de PhysX, y `config.py` / `materials.py` no importan nada de Isaac.

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Qué contiene la celda. Python puro, sin Isaac, validable y serializable. |
| `materials.py` | Parámetros físicos **con su procedencia** (S1/S2/S3/T). Python puro. |
| `isaac_env.py` | Construye la escena. **Única autoridad de la física.** |
| `robot.py` | Un brazo: articulaciones reales, IK DLS real, lecturas reales. |
| `grasp.py` | Si un agarre es real, a partir del estado simulado. |
| `orchestrator.py` | Quién hace qué. Lee la simulación, nunca la escribe. |
| `worker.py` | El proceso de Isaac Sim: arranca Kit, corre el bucle, imprime el resultado. |
| `server.py` | REST + WebSocket sobre la celda viva. |
| `web/` | Interfaz Vite + React + TypeScript. |
| `tools/` | Sondas. Nada de esto es parte de la demo; existen para que `PHYSICS.md` cite mediciones, no afirmaciones. |

---

### `add_robot` / `remove_robot` se niegan a propósito

Isaac Lab no permite insertar una articulación en una escena viva, y la reconstrucción de la
celda dentro del proceso se implementó y **se midió que se cuelga** (370 segundos de CPU, sin
retornar nunca — `RemovePrim` + un segundo `sim.reset()` en la misma sesión de Kit). Así que
esos dos métodos devuelven `{"ok": false, "reason": ...}` con el motivo, en vez de dar por
bien añadido un robot que nunca existió.

La vía que sí funciona es el supervisor, descrita más arriba.

---

## La celda

`default_scene(n)` construye una celda válida con 1–3 robots SO-101, cuatro piezas ABS/PET y
dos bandejas. `validate()` rechaza, **antes de arrancar el simulador**, cualquier celda con
una pieza fuera del alcance de todos los brazos, solapada con otra, dentro de una bandeja, o
demasiado grande para la mordaza.

Dos cosas que hay que saber antes de tocarla:

- **`validate()` mide el brazo, no la envolvente.** `materials.MAX_REACH_M` son 335 mm y es
  correcto como geometría; el brazo está demostrado hasta unos **235 mm** a una objetivo
  baja, porque `_wrist_for_down` es un *ajuste* medido sobre el área de trabajo original. Por
  eso existe `WORKING_RADIUS_M = 0.235`, y por eso validar contra la envolvente acepta celdas
  que después fallan cada agarre con 150–270 mm de error y el brazo clavado en el tope de
  `shoulder_lift`. Está en `docs/PHYSICS.md` §11c.
- **La mesa y la fila de bases no son parámetros libres.** `settle` retractiona cada brazo a
  `base_y - 0.10`, 100 mm por detrás de su propia base, y eso sobresale del borde trasero de
  la mesa original por 65 mm. Agrandar la mesa pone la pose de reposo **encima** de la mesa,
  los cinco colisionadores del brazo empujan contra ella y el brazo no extiende. Medido:
  **0/6 tareas**.

La misión por defecto son **las dos elecciones que coinciden en color**, en lados opuestos de
la celda para que los dos brazos trabajen en paralelo. Con `--all-tasks` (o desde la
interfaz) **las cuatro piezas tienen tarea**, que es el caso con contención de verdad.

---

## El agarre: siete criterios, siete mediciones

`demo12/grasp.py` responde siete preguntas, cada una **leyéndola** de la simulación, y
`evaluate_grasp` es la **única** definición de "el agarre fue real": la usan tanto la sonda
offline como el orquestador en vivo, así que un test y la demo no pueden discrepar.

| # | Criterio | Afirmación |
|---|---|---|
| 1 | `tracking` | la herramienta estaba donde se le dijo al cerrar |
| 2 | `pads_on_faces` | los dedos se detuvieron en el ancho de la pieza, por las dos caras |
| 3 | `grip` | la fuerza supera el **cono de fricción** de esa masa |
| 4 | `part_rose` | la pieza subió con la mandíbula |
| 5 | `carried` | llegó al destino todavía en la mordaza |
| 6 | `part_in_bin` | terminó dentro de la cavidad |
| 7 | `part_resting` | y había dejado de moverse |

El criterio 3 ya no es una constante: es `2·mu·N ≥ m·g` con margen 2×, calculado por pieza
(0.31 N para el cubo de 28 g, 0.03 N para la bola de 2.9 g). Cada agarre se informa con su
**margen**, para que un aprobado se pueda juzgar cómodo o marginal.

Los criterios 2, 4 y 5 son los que faltaban. El orquestador en vivo usaba dos: `grip > 0` y
"está en la bandeja y quieto". Con esos dos, **ambos fallos documentados pasaban como tareas
completadas** — el punteo de §4, con 103 N de fuerza real y la pieza empujada 36 mm, y una
pieza que nunca salió de la mesa pero acabó en la bandeja. Está en `docs/PHYSICS.md` §11a.

Resultados medidos, dos robots y las cuatro piezas con tarea:

```
Demo 12: 2 robot(s), 4 parts, 2 bins, 4 tasks
OK: 4 task(s) completed in 148.3 s simulated, 8242 control ticks
```

| Pieza | Apertura / ancho | Fuerza | Cono de fricción | Subida | Error | Intentos |
|---|---|---|---|---|---|---|
| `part_1` cubo ABS 30 mm | 29.40 / 30.0 mm | 60.9 N | 0.31 N | 99.1 mm | 0.38 mm | 1 |
| `part_2` cilindro 30 mm | 29.20 / 30.0 mm | 60.5 N | 0.24 N | 99.0 mm | 0.28 mm | 1 |
| `part_3` cubo ABS 25 mm | 24.60 / 25.0 mm | 44.9 N | 0.18 N | 99.0 mm | 0.13 mm | 1 |
| `part_4` bola PET 20 mm | 19.10 / 20.0 mm | 25.4 N | 0.03 N | 99.2 mm | 0.10 mm | 1 |

Cuatro tareas, cuatro intentos, éxito de agarre **1.0**, y los siete criterios pasando en
todas: 28 de 28.

El margen de fricción es **×197**: los agarres no son ajustados, y por eso la estrictez de
§11a no costó nada. El problema nunca fue la física, fue que faltaban las comprobaciones.

---

## El planificador: por qué este robot

`demo12/planner.py` puntúa cada par (brazo, tarea) con cuatro términos, y sólo uno de ellos
no es "qué me sale barato a mí":

| Término | Qué mide |
|---|---|
| `reach` | de la base del brazo a la pieza |
| `carry` | de la pieza a su bandeja, ponderado 0.6 |
| `retries` | 0.15 m por intento ya gastado |
| `conflict` | **0.5 si otro brazo ya va a por una pieza a menos de 130 mm** |

Los tres primeros son de *un* brazo y *una* pieza. El cuarto es lo que hace esto una
colaboración y no cuatro planificadores en paralelo: sin él cada brazo coge lo que tiene
más cerca, y como todos miran hacia el mismo lado de la misma mesa, lo más cercano para dos
brazos distintos es a menudo la misma esquina.

Medido, con las cuatro piezas con tarea: `robot_1` cogía `part_1` (x=0.290) y para `robot_2`
lo más barato de la celda era `part_4` (x=0.360), a **70 mm**. Dos brazos en la misma esquina
y **0/4 con errores de 146-208 mm**, que parecían un problema de alcance y no lo eran. Los
límites reales están medidos: **70 mm pierde el agarre, 140 mm trabaja en paralelo**.

Con el término, la misma misión se planifica como la de dos tareas que sí funciona: `robot_1`
coge `part_1`, `robot_2` es empujado a `part_2` al otro lado, y los dos trabajan a la vez.

---

## Fallos y correcciones

Todo lo que salió mal está en `demo12/docs/PHYSICS.md` con su número. Lo esencial:

- **Los dedos iban por el eje equivocado de la mandíbula.** Rodaban sobre una frase del
  handoff que era falsa, y por eso se abrían **verticalmente**: presionaban encima de la
  pieza y no generaban agarre. El handoff tenía los dos ejes permutados. Medido sobre 28
  poses; el arreglo es cambiar el eje de prisma de `Y` a `Z`.
- **La IK congelaba el brazo.** Añadía una repulsión de límite a la misma escala que el paso
  acotado, así que cerca de un tope se cancelaban exactamente y la solución era un punto
  fijo. Ahora los límites son una *proyección*, que no puede competir con el paso.
- **La mordaza no podía rodear una pieza cuadrada.** El eje de separación llegaba a 34.8°
  con las caras de la pieza, así que un cubo de 30 mm presentaba su diagonal de 41.8 mm
  frente a 37 mm de apertura. Corregido ampliando el recorrido y alineando la muñeca con la
  pieza antes de cerrar.
- **La altura de agarre venía de tres constantes distintas** (23, 25 y 26 mm) en dos
  sistemas de referencia. Ahora se mide dónde están realmente los dedos.
- **Una pieza agarrada es un presupuesto de fricción, y la velocidad lo gasta.** A velocidad
  completa, la traslación sacaba la pieza: el agarre mantenía 60.8 N y aun así la pieza
  acababa 26.5 mm desplazada.

---

## Aproximaciones declaradas

No están escondidas; están listadas en `demo12/docs/PHYSICS.md` §9 con su motivo. La más
relevante: `wrist_roll` se **manda**, no se optimiza, y la alineación de la mordaza se
ajusta a un eje del mundo porque las piezas están alineadas a los ejes.

La colisión del brazo está **activada** (5 colisionadores), lo que ya no es una
aproximación: se verificó contra el escenario, no inferido del resultado del agarre.