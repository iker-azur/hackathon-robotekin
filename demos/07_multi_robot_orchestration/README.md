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
... -m demo12.server --robots 2 --gui      # http://127.0.0.1:8000
cd demo12\web; npm run dev                 # http://localhost:5173
```

Sondas de medición (verifican todo lo afirmado en `demo12/docs/PHYSICS.md`):

```powershell
... -m demo12.tools.probe_grasp_pose   # orientación de la mandíbula -> out12/pose_probe.json
... -m demo12.tools.probe_grasp        # los cuatro criterios  -> out12/grasp_probe.json
... -m demo12.tools.probe_frame        # captura de viewport    -> out12/frame_probe.json
```

### Limitación conocida: la imagen

`/api/frame.jpg` **devuelve 503 en esta máquina**, con el motivo. No sirve una imagen de
relleno: un frame falso dejaría la interfaz mostrando un render muerto como si fuera real.

Está **medido, no supuesto**: se probaron las cuatro combinaciones (headless y `--gui`, con y
sin bombear el bucle de actualización de Kit) y ninguna produce una imagen. La causa es
`has_primed_camera() == False` — el viewport no tiene cámara en esta máquina, así que no hay
nada que capturar.

La ruta alternativa —una prim `UsdGeom.Camera` leída por `isaaclab.sensors.Camera`— también
se investigó. **La cámara se construye bien**, `cameras_enabled` está a `True`, pero
`reset()` falla con *"Camera could not be initialized"*: el sensor queda sin inicializar
porque su ciclo de vida espera un `ManagerBasedEnv` y esta demo usa un `SimulationContext`
suelto. La cadena de diagnóstico completa está en `demo12/docs/PHYSICS.md` §9.

Pruebas:

```powershell
... -m pytest demo12\tests -q          # 35: 28 rápidos (0.15 s) + 7 en Isaac (~8 min)
```

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

### Limitación conocida: cambiar el número de robots

`add_robot` / `remove_robot` **se niegan**, con el motivo. Isaac Lab no permite insertar una
articulación en una escena viva, así que la única vía sería reconstruir la celda entera; esa
reconstrucción se implementó y **se midió que se cuelga** (370 segundos de CPU, sin
retornar nunca — `RemovePrim` + un segundo `sim.reset()` en la misma sesión de Kit).

La vía que sí funciona: reeniciar con `--robots N`. `release_for_relaunch()` devuelve las
tareas en vuelo a la cola antes de salir. La cola de tareas sobrevive; el estado físico no.

---

## La celda

`default_scene(n)` construye una celda válida con 1–3 robots SO-101, cuatro piezas ABS/PET y
dos bandejas. `validate()` rechaza, antes de arrancar el simulador, cualquier celda con una
pieza fuera del alcance de todos los brazos, solapada con otra, dentro de una bandeja, o
demasiado grande para la mordaza.

Cada robot lleva su propio `ReachEnvelope`, así que habilitar o deshabilitar un robot nunca
deja una pieza o una bandeja inaccesible.

---

## El agarre: cuatro criterios, cuatro mediciones

`demo12/grasp.py` responde cuatro preguntas, cada una **leyéndola** de la simulación:

1. **Seguimiento** — `tool_error_m < 1 mm` en la pose de agarre.
2. **Recorrido** — mordaza abierta ≥ 36 mm, cerrada < 18 mm, medido en espacio libre.
3. **Fuerza** — `grip_force_n > 0` entre los dedos.
4. **Levantar y soltar** — la pieza sube con la mandíbula y queda en la bandeja, en reposo.

El criterio 4 es el único que prueba que hubo un agarre. Los tres primeros pueden pasar
mientras la pieza sigue sobre la mesa. Y el criterio 2 se mide **en espacio libre** a
propósito: con una pieza de 30 mm entre los dedos no pueden cerrarse más, y juzgarlos por
esa cifra daría por roto un mecanismo que funciona.

Resultados medidos (`out12/grasp_probe.json`):

| Pieza | Masa | Seguimiento | Fuerza | Subida | Resultado |
|---|---|---|---|---|---|
| `part_1` cubo ABS 30 mm | 28.1 g | 0.53 mm | 60.9 N | 98.9 mm | en bandeja, en reposo |
| `part_4` bola PET 20 mm | 2.9 g | 0.73 mm | 24.4 N | 98.7 mm | en bandeja, en reposo |

Dos robots, dos tareas, headless:

```
OK: 2 task(s) completed in 46.4 s simulated, 2796 control ticks
```

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