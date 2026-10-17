# Demo 12 — Orquestación Autónoma Multi-Robot SO-101

Simulación física-cinemática y centro de control industrial web para **1 a 3 brazos robóticos SO-101** colaborando de forma autónoma sobre una celda de almacén compartido para el **Robotekin Hackathon 2026**.

El sistema demuestra de principio a fin el ciclo cognitivo completo:

```text
OBSERVAR → ENTENDER → PLANIFICAR → DISTRIBUIR → EJECUTAR → VERIFICAR → RECUPERAR
(OBSERVE → UNDERSTAND → PLAN → DISTRIBUTE → EXECUTE → VERIFY → RECOVER)
```

---

## Ejecución rápida

Desde la raíz del repositorio:

```powershell
# 1. Lanzar el Centro de Control Industrial y abrir el navegador automáticamente
python .\demos\07_multi_robot_orchestration\12_multi_robot_orchestration.py --open

# 2. Lanzar directamente con el Escenario de Presentación Hackathon (7 fases en vivo)
python .\demos\07_multi_robot_orchestration\12_multi_robot_orchestration.py --hackathon-demo --open

# 3. Ejecutar el escenario Hackathon en modo headless (verificación por terminal)
python .\demos\07_multi_robot_orchestration\12_multi_robot_orchestration.py --hackathon-demo --headless-run

# 4. Ejecutar la suite de validación automatizada
python .\demos\07_multi_robot_orchestration\tests\test_demo12.py
python .\demos\07_multi_robot_orchestration\tests\test_body_physics.py
python .\demos\07_multi_robot_orchestration\tests\test_gripper_model.py
python .\demos\07_multi_robot_orchestration\tests\test_task_loop.py

# 5. Volver a medir las dos ejecuciones de referencia (sin fallos / escenario guionizado)
python .\demos\07_multi_robot_orchestration\tests\run_reference.py
```

Véase `docs/PHYSICS_IMPROVEMENTS.md` para la auditoría completa de física y realismo
(problemas medidos, cambios y resultados de validación).

También puede lanzarse desde `demos/05_planner_and_executor/12_multi_robot_orchestration.py`.

---

## 1. Análisis del proyecto existente y decisiones técnicas

Al revisar los demos previos (especialmente **Demo 10** y **Demo 11** en `demos/05_planner_and_executor/`), se identificaron los componentes reutilizables y los cuellos de botella para escalar a múltiples robots en directo:

| Componente en Demo 11 | Diagnóstico para Multi-Robot (Demo 12) | Decisión arquitectónica en Demo 12 |
|---|---|---|
| **Percepción OpenCV HSV (`colors.py`, `perceive.py`)** | Rápida (~16 ms), determinista y sin sobrecarga de GPU; segmenta color, momentos de segundo orden y perfil de profundidad. | **Reutilizada e integrada** en `perception.py` sobre una cámara cenital RGB-D (`640×480`, `1143 px/m`), ahora con modelo de sensor (distorsión de lente, sesgo de calibración, ruido de centroide) que da un error medio de localización de `~3,3 mm`. |
| **Física de agarre (`real_world.py`)** | Mide la apertura real del SO-101 (`36.2 mm`), masas por material y cono de fricción (`2·μ·N ≥ m·g`). | **Reutilizada** en `robot_driver.py` (`evaluate_grasp_physics`) y `warehouse_sim.py` (`try_grasp`). |
| **Telemetría y servidor (`telemetry.py`, `monitor/server.py`)** | `FastAPI` + `Uvicorn` + `WebSocket` dirigido por contador de versión (`PUSH_MIN_INTERVAL_S`) sin dependencias CDN externas. | **Reutilizado y ampliado** para transmitir estado de flota de 3 robots, línea temporal causal, 7 etapas cognitivas y cámara anotada a ~22 Hz. |
| **Simulación en Isaac Sim (`LeIsaac-SO101-LiftCube-v0`)** | La escena USD tiene 1 único `"robot"` estático, tarda 45–120 s en arrancar, corre a `0.5x` tiempo real con un solo brazo y no permite añadir/quitar articulaciones en caliente sin reconstruir `gym.make()`. | **Simulador cinemático-físico dedicado a 60 Hz (`SimulatedSO101` + `WarehouseSimulation`)** con contrato `SO101Driver` / `RealSO101Adapter` idéntico al del hardware LeRobot. |

### Honestidad robótica: qué es genuino vs. simulado

Siguiendo la filosofía de Demo 11, ningún movimiento es una animación pregrabada en el navegador:

1. **Cinemática directa (FK) e inversa (IK) real de 6 grados de libertad (`robot_driver.py`):** Cada robot SO-101 mantiene sus 6 ángulos articulares (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`). El controlador de servos integra velocidades articulares limitadas (`3.2 rad/s`) a 60 Hz y calcula las poses 3D de todos los eslabones (`base`, `shoulder`, `elbow`, `wrist`, `gripper`, `aperture`, `left_finger`, `right_finger`) mediante Cinemática Directa.
2. **Percepción OpenCV real (`perception.py`):** El planificador recibe las coordenadas `(x, y)` extraídas por segmentación HSV (`cv2.connectedComponentsWithStats` + momentos de contorno + perfil de profundidad) sobre el fotograma RGB-D renderizado de la mesa.
3. **Verificación métrica y gravedad física (`warehouse_sim.py`):** Al cerrar la pinza se comprueba la apertura (`36.2 mm`), el error lateral (`≤ 22 mm`) y el cono de fricción. Si un robot sufre un fallo mientras transporta una pieza, la pieza cae por gravedad sobre la mesa y debe ser re-detectada por visión antes de que otro robot pueda recogerla.

### Modelo físico del gemelo digital (qué se simula y cómo)

El gemelo digital a 60 Hz no es una marioneta cinemática: modela los fenómenos que
determinan si una operación de *pick & place* cooperativa funciona o falla.

| Fenómeno | Modelo | Dónde |
|---|---|---|
| **Masa de las piezas** | Derivada de la geometría y la densidad del material (1040 kg/m³), compartida con el cono de fricción: el cubo de 22,8 g que necesita 0,13 N de agarre es el mismo cubo de 22,8 g que sale despedido. | `robot_driver.body_properties` |
| **Dinámica de servos** | Perfil trapezoidal: aceleración limitada (`26 rad/s²`), velocidad tope (`3.2 rad/s`), deceleración hasta el destino sin sobreimpulso, y un **límite cartesiano de 0,55 m/s** en la punta de la herramienta (el límite articular solo permitía ≈1,1 m/s con el brazo extendido). | `robot_driver.py` → `step_servo` |
| **Par gravitatorio** | Cuanto más se extiende el brazo, más par estático exige el hombro, y la penalización por carga es **proporcional a la masa transportada**. | `robot_driver.py` → `_load_derating` |
| **Fuerza de pinza (actuador bloqueado)** | La fuerza normal es el resorte del actuador deflexionado contra la pieza: `N = k · (ancho_pieza − hueco_comandado)/2`, con `k = 250 N/m` medido en banco en la Demo 11. Un cubo de 28 mm con un cierre de 15,5 mm da **1,66 N**. | `robot_driver.py` → `jaw_stall_m`, `step_servo` |
| **Geometría de mordaza** | El hueco efectivo nunca cierra por dentro de la pieza: las almohadillas se apoyan en sus flancos en lugar de atravesarla, y la cinemática directa coloca `left_finger`/`right_finger` desde ese hueco. | `robot_driver.py` → `effective_jaw_gap_m` |
| **Puerta lateral de agarre** | Derivada de la apertura de la mandíbula: `‖d‖ ≤ (36,2 − ancho)/2 + 4 mm`. Un cubo de 28 mm exige apuntar a **7,1 mm**; una pieza de 34 mm, a 5,1 mm. (La puerta plana de 22 mm permitía cerrar a 20 mm y llamarlo agarre.) | `robot_driver.grasp_lateral_gate_m` |
| **Deslizamiento por cono de fricción** | Dinámico: cada paso se compara la fuerza tangencial necesaria (`m·a`) con la disponible (`2·μ·N` corregida por excentricidad). Si la aceleración de transporte supera el cono, la pieza sale de las pinzas con su velocidad. Sin dados ni teleports. | `warehouse_sim.py` → `_slip_held_part` |
| **Cuerpos rígidos** | Gravedad, impulso con masa, restitución (0,26 bola / 0,18 rígido), fricción de Coulomb estática y cinética, rodadura con resistencia y **rotación integrada** (`yaw_deg`), y sub-pasos fijos de 1/240 s que impiden atravesar paredes o piezas. | `warehouse_sim.py` → `_resolve_vertical`, `_apply_contact_friction`, `_integrate_bodies` |
| **Apilado** | El soporte vertical se resuelve de arriba abajo: superficie de mesa, suelo de caja, cara superior de otra pieza y suelo del mundo. | `warehouse_sim.py` → `_support_surface` |
| **Paredes de caja** | Una losa sólida por cara. Solo actúa mientras la **base** de la pieza esté por debajo del borde, y la expulsión es siempre hacia fuera: una pared no se cruza en ningún sentido. | `warehouse_sim.py` → `_container_wall_contact` |
| **Borde real de la mesa** | Una pieza empujada más allá del borde (`TABLE_X`, `TABLE_Y`) cae al suelo, queda marcada como `fallen` y **ningún brazo puede agarrarla**. Antes las piezas quedaban clamped en una sub-región de 32×16 cm. | `warehouse_sim.py` → `on_table`, `try_grasp` |
| **Colisiones brazo-pieza** | Cada eslabón se barre en sub-pasos de ≤3 mm y se resuelve con **impulso ponderado por masas** contra la cápsula del eslabón. La boca de la pinza es hueca: se modela con dos almohadillas, así que una pieza que cabe entre las mandíbulas abiertas no es empujada por quien va a cogerla. | `warehouse_sim.py` → `_resolve_arm_contacts` |
| **Interbloqueo brazo-brazo** | Guardia bidireccional y predictiva: un brazo caído es obstáculo duro (frenos mecánicos), el menor cede el corredor y el mayor **no entra** ni siquiera en el tick que está a punto de ordenar. | `warehouse_sim.py` → `CollisionGuard` |
| **Despeje cooperativo** | Una pieza bajo la pinza paralizada de un brazo caído queda realmente atrapada: ningún brazo puede meter otra pinza en esa columna. Un brazo sano la empuja fuera con el contacto físico de sus eslabones, dejando de empujar en cuanto está despejada. | `orchestrator.py` → `_maybe_dislodge` |
| **Parada de emergencia** | Un fallo de servomotor congela la pose (freno mantenido) y **suelta la pieza**, que cae desde ese punto por gravedad. | `robot_driver.py` → `trigger_fault` |
| **Perturbaciones y fallos inyectados** | Un desplazamiento externo es un **impulso**; un deslizamiento de agarre es **contaminación de almohadillas** (μ más bajo) que luego produce la pérdida por cono de fricción. Ninguno mueve la pieza por guion. | `orchestrator.py` → `inject_failure` |
| **Ruido sensorial** | La cámara cenital tiene distorsión de lente, sesgo de calibración y ruido de centroide: el error de localización reportado es real (`~3 mm` de media), no exactamente cero. | `perception.py` → `_apply_lens_distortion`, `_sensor_noise_m` |

**Consecuencia observable:** las recuperaciones automáticas del escenario Hackathon
las provoca la física real (piezas desplazadas, atrapadas bajo un brazo caído,
interferencias aéreas), no un guion. El escenario completa las 6 tareas verificadas
en unos 20 s de simulación.

**Limitación conocida y declarada en el código:** la guardia de corredores mide la
distancia entre los orígenes muestreados de los eslabones, mientras las cápsulas
modeladas tienen radios de 13–30 mm. La separación mínima medida en la ejecución de
referencia es de 8,5 mm entre orígenes, lo que sigue siendo un solape de envolventes.
Tres brazos SO-101 con bases a 12 cm y esas envolventes no pueden compartir la zona
central de 32 cm sin solaparse; subir el umbral al valor geométricamente exacto
(> 60 mm) serializa la celda y el pedido de referencia deja de completarse.
Resolverlo es un cambio de **disposición de la celda**, no de física, y queda
registrado como tal en `tests/run_reference.py` y en `docs/PHYSICS_IMPROVEMENTS.md`.

### Rol del backend NVIDIA Isaac Sim / LeIsaac (`isaac_multi_backend.py`)

El backend opcional de Isaac Sim **no es la autoridad de la física**: es una capa de
renderizado fotorrealista RTX. El flujo es unidireccional, el gemelo digital publica
`target_xyz` y las poses de pieza, Isaac interpola los brazos hacia esos destinos y
sobreescribe la posición de cada pieza (`write_root_pose_to_sim` /
`write_root_velocity_to_sim`), por lo que los cuerpos en Isaac siguen al gemelo y no al
revés. Se eligió así a propósito (ver la decisión arquitectónica de la tabla del
apartado 1): la escena `LeIsaac-SO101-LiftCube-v0` tiene un único `"robot"` estático,
arranca en 45–120 s, corre a `0.5x` con un brazo y no permite altas/bajas en caliente.

Para ver la celda con el gemelo de render, lanzar con
`--open` (o `--isaac-gui`) y pulsar **Sincronizar** en la interfaz. Para verificar la
física y la orquestación sin GPU, usar `--hackathon-demo --headless-run --no-isaac` o la
suite de tests.

---

## 2. Escenario de celda colaborativa multi-robot

Tres estaciones **SO-101** rodean una mesa industrial compartida con una zona central de suministro y tres cajas huecas de clasificación (**Caja Roja**, **Caja Azul** y **Caja Verde**):

- **SO-101 #1 · Alfa (Estación Sur-Oeste):** Base en `(0.24, -0.64) m`, orientación `+90°`.
- **SO-101 #2 · Beta (Estación Sur-Este):** Base en `(0.36, -0.64) m`, orientación `+90°`.
- **SO-101 #3 · Gamma (Estación Norte):** Base en `(0.48, -0.64) m`, orientación `+90°`.

Sus envolventes de alcance (`0.345 m`) se solapan sobre la zona central de trabajo (`x ∈ [0.20, 0.52] m`, `y ∈ [-0.52, -0.36] m`). Esto permite:
- Que **1 único robot** pueda completar cualquier pedido de forma secuencial.
- Que **2 o 3 robots** se repartan el trabajo minimizando la distancia de desplazamiento desde cada estación.
- Que si **cualquier robot falla o se desconecta**, los robots restantes puedan alcanzar físicamente sus piezas huérfanas y completar el pedido.
- Que **`CollisionGuard`** gestione la reserva de espacio aéreo en 3D de forma **bidireccional y predictiva**: el brazo de menor prioridad cede el corredor (`PRIORIDAD AÉREA`) y el de mayor prioridad no invade los eslabones del otro ni siquiera dentro del tick que está a punto de ordenar. Un brazo **caído es obstáculo duro** (sus frenos mantienen la pose), y una pieza que queda atrapada bajo su pinza se despeja físicamente por otro brazo.

---

## 3. Escenario Hackathon en 1 Clic ("Modo Presentación Robotekin")

Al pulsar **🚀 Escenario Hackathon (1 Clic)** en la interfaz web (o ejecutar `--hackathon-demo`), el director `HackathonScenarioDirector` ejecuta de forma determinista las 7 fases:

1. **Fase 1/7 — Inicio con 1 Robot:** Solo **SO-101 #1 · Alfa** está conectado. Entra un pedido de 6 piezas (rojas, azules y verdes). Alfa asume las 6 tareas secuencialmente y comienza la primera.
2. **Fase 2/7 — Incorporación Dinámica:** **SO-101 #2 · Beta** y **SO-101 #3 · Gamma** se conectan en caliente. El orquestador central recalcula los costes de asignación y redistribuye las 5 tareas pendientes entre los 3 robots.
3. **Fase 3/7 — Ejecución Paralela 3x:** Los tres SO-101 trabajan simultáneamente sobre la mesa compartida con desconflicción de espacio aéreo.
4. **Fase 4/7 — Fallo Inesperado en Vivo:** Mientras **SO-101 #2 · Beta** transporta una pieza, sufre una sobrecarga en el servomotor del codo (`ERR_SERVO_OVERLOAD`), entra en parada de seguridad y suelta la pieza sobre la mesa.
5. **Fase 5/7 — Detección, Re-observación y Redistribución:** El sistema central detecta el fallo, activa `RECUPERAR → OBSERVAR` para localizar con OpenCV dónde cayó la pieza, y reasigna las tareas inacabadas de Beta a **Alfa** y **Gamma**.
6. **Fase 6/7 — Recuperación Cooperativa:** Alfa y Gamma recogen la pieza caída y el resto del lote.
7. **Fase 7/7 — Verificación Final y Resumen:** Las 6/6 piezas quedan verificadas dentro de sus cajas correspondientes, con un **error medio de colocación de 0,0 mm** medido sobre el slot comandado (el error se mide contra la posición ordenada, no contra el centro de la caja: dos piezas en una cavidad de 62 mm están a 24 mm del centro por geometría del empaquetado, y reportar eso como "error" hacía que una colocación perfecta pareciera 24 mm de desvío).

---

## 4. Estructura del módulo

```text
demos/07_multi_robot_orchestration/
├── 12_multi_robot_orchestration.py  ← Punto de entrada CLI y lanzador del servidor
├── robot_driver.py                  ← Abstracción SO101Driver, cinemática 6-DOF FK/IK y física de agarre
├── warehouse_sim.py                 ← Simulación de celda multi-robot, gravedad, CollisionGuard y cámara RGB-D
├── perception.py                    ← Segmentación OpenCV HSV, momentos de forma y enlace con catálogo
├── planner.py                       ← Parser multilingüe de pedidos, descomposición en tareas y coste multi-robot
├── orchestrator.py                  ← Orquestador central (7 etapas, escalado en caliente, verificación y recuperación)
├── hackathon_scenario.py            ← Director de presentación en 1 clic (7 fases para el jurado)
├── telemetry.py                     ← Bus thread-safe de estado, línea temporal causal y fotogramas JPEG
├── server.py                        ← Servidor FastAPI + WebSocket (22 Hz) y bucle de simulación a 30 Hz
├── static/
│   └── index.html                   ← Centro de Control Industrial en español (Vista 3D/2D + Cámara + Flota)
└── tests/
    ├── test_demo12.py         ← Regresión del contrato y el frontend (103 comprobaciones)
    ├── test_body_physics.py   ← Masa, momento, fricción, apilado, paredes y borde real (98)
    ├── test_gripper_model.py  ← Mordaza bloqueada, fuerza de actuador y deslizamiento (29)
    ├── test_task_loop.py      ← Convergencia por acción, recuperaciones y métrica (48)
    └── run_reference.py       ← Vuelve a medir las dos ejecuciones de referencia
```
