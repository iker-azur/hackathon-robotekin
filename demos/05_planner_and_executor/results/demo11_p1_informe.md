# Demo 11 — P1: panel de presentación y velocidad de ejecución

Fecha: 2026-10-10. Entorno: Windows, Conda `hackathon-robotekin`, Isaac Sim 5.1, RTX 5060 Ti.
Rama: `prepare-demos`. Todo se apoya en los cambios de P0, que se conservan.

## Objetivo

Convertir el panel de Demo 11 en una demostración de producto: cámara, objetivo y tareas como
protagonistas; la cadena **VER → ENTENDER → PLANIFICAR → ACTUAR → VERIFICAR → RECUPERARSE** legible
de un vistazo; un botón **Iniciar demo** de una sola pulsación; y una orden de dos objetos por
debajo de los 45 s objetivo. Prioridad acordada: fiabilidad → claridad → respuesta → acabado
visual → velocidad. La velocidad va al final y nunca a costa de un agarre.

---

## A. Cambios implementados

Todos los archivos de interfaz están en `monitor/static/index.html`; la lógica de presentación se
añadió encima de la que ya existía, sin tocar el transporte (WebSocket/REST), la codificación de
fotogramas ni el autómata de fases.

### A.1 Diseño del panel (presentación)

- **Cabecera nueva**: nombre del proyecto (*SO-101 · Agente autónomo*), estado de conexión, fase,
  contadores, y el **botón primario «Iniciar demo»** más los iconos de ruta completa, información y
  registro. En pantallas estrechas (≤1500 px) los contadores secundarios se ocultan por CSS para que
  el CTA nunca se caiga del borde; los `id` siguen presentes, así que es un cambio de maquetación,
  no de estado.
- **Tira de la cadena de seis etapas** (`#stage-rail`): VER, ENTENDER, PLANIFICAR, ACTUAR,
  VERIFICAR, RECUPERARSE. Cada etapa resume el estado y la nota de las etapas reales del mapa
  (`telemetry.FLOW`) que la forman; nada se inventa (ver §C).
- **Barra de estado de la demo** (`#demo-status`): el paso en curso (1/4 cargar escena → 2/4
  arrancar Isaac → 3/4 entorno listo → 4/4 colocando) y el veredicto final, para que la pantalla
  jamás se congele durante el arranque.
- **Rejilla principal (`.stage-grid`)**: cámara a la izquierda a toda altura, **objetivo y tareas**
  arriba a la derecha, y entorno+comando (ahora compactos) abajo. El entorno y la ruta completa
  pasan a ventanas (`#route-modal`, `#env-modal`), así la pantalla principal sólo muestra lo que
  mira un jurado. El orden del HTML se conserva (entorno < comando < `.pair`) para no romper las
  pruebas de estructura.

### A.2 Cámara

Sigue siendo el elemento más grande. Reutiliza el mismo `<img>` + `URL.createObjectURL` y el
JFIF/JPEG que ya llegaba por WebSocket; no se añade procesado de vídeo. Muestra «cámara aún no
conectada» hasta el primer fotograma, el aviso de imagen caduca a los 6 s (`CAMERA_STALE_S`) y el
error real de cámara si lo hay, sin esconder ni inventar imágenes.

### A.3 Lista de tareas (objetivo y tareas)

Panel propio, grande y limpio. Toma `state.tasks` (la unión que publica el robot, que sobrevive a
los replan) y muestra sólo: descripción y estado (pendiente / en ejecución / recuperando /
completada / fallida). **Una tarea nunca se marca completada sin que la verificación física lo
diga**, y una en reintento dice RECUPERANDO (tomado de `phase="recovering"`), no un falso fallo.

### A.4 Cadena de decisión

Las doce etapas del mapa se agrupan en seis legibles (`STAGE_DEFS`). El estado de cada una sale de
las etapas reales: activa si alguna trabaja, completada si todas ok, `warn`/`fail` si alguna falla,
pendiente mientras no haya ocurrido. La nota es la del robot (p. ej. «2 objetos detectados», la
frase entendida, el verbo en curso). El mapa completo de doce cajas sigue disponible en su ventana.

### A.5 Modo presentación de una pulsación

- **Iniciar demo** (`#demo-run`, en la cabecera): carga la escena medida de dos cubos, arranca el
  simulador con la recuperación de arranque de P0, espera a «listo» y envía sola la orden validada.
- **Anti-duplicado**: un guardián `demoBusy` bloquea una segunda pulsación durante la carga
  asíncrona (antes sólo se bloqueaba tras ella); la orden se envía una sola vez al llegar a
  «listo»; si el arranque falla o se detiene, el botón vuelve a servir para reintentar.
- Los controles manuales (iniciar/detener entorno, siguiente ronda, re-mirada, guardar, comando y
  voz) siguen accesibles en la sección avanzada.

### A.6 Velocidad de ejecución

El informe de P0 dejó una medida pendiente: atribuir el factor 0,505 (un paso de física costaba el
doble del nominal). `tools/probe_step_cost.py` lo mide ahora:

| Paso | Medido |
|---|---|
| `env.step` (bucle de movimiento, con render) | **32,4 ms** |
| `env.sim.step(render=False)` (sólo física) | **3,9 ms** |

El bucle de movimiento del ejecutor (`Executor.do`/`track`/`settle`) llama a `env.step`, y
`ManagerBasedEnv.step` **dibuja en cada paso** mientras haya un sensor RTX. Ese render es
redundante: la imagen de la demo y cada re-apuntado se leen con `isaac_io.capture` /
`capture_rgb`, que llaman a `sim.render()` por su cuenta. El render por paso es pues el ~88 % de
cada paso de movimiento, y es la razón de que la simulación corriera a medio real.

Cambio (`executor.py`): cuando un `Executor` conduce, sube `env.cfg.sim.render_interval` a
`MOTION_RENDER_INTERVAL` (1 000 000), de modo que el bucle de movimiento casi nunca auto-dibuja.
**La física no cambia** —un render no altera la simulación—, así que los agarres y la verificación
de colocación se comportan igual.

---

## B. Resultados de rendimiento

Medido con la escena medida de dos cubos (preset `demo_config()`, semilla 108, dispersión 0,
orden «separa las piezas por colores»), pasando de P0 (render por paso) a P1 (render apagado durante
el movimiento), con el mismo comando y la misma máquina:

| Métrica | Antes (P0) | Después (P1) |
|---|---|---|
| Orden de dos objetos (`command_total_ms`) | **72 581 ms (72,6 s)** | **26 399 ms (26,4 s)** |
| Pasos de física por objeto | ~750 (sin cambio) | ~750 (sin cambio) |
| Coste de un paso de movimiento | ~32 ms | ~12,5 ms |
| Colocaciones verificadas | 2/2 | 2/2 |
| Agarres rechazados | 0 | 0 |
| Resultado | SUCCESS | SUCCESS |
| Latencia de cámara (por render) | ~22–25 ms | ~22–25 ms (sin cambio: el render de la cámara es explícito) |
| Demo de una pulsación | no existía | probada (ver §C) |

La orden de dos objetos pasa de **72,6 s a 26,4 s** (≈ 2,75×), **por debajo del objetivo de 45 s**,
con el mismo resultado (2 verificadas, 0 rechazados). El arranque de Kit y la construcción del
entorno (~37 s hasta «listo», de los que ~19 s son Kit) no han cambiado: no tocan el comando.

> Nota honesta sobre el «antes»: el informe de P0 (`demo11_perf.md`) medía ~62 s. La cifra de antes
> de esta tabla (72,6 s) es del **mismo código, misma semilla y misma máquina** que el «después»,
> con el render encendido, que es la comparación limpia. La diferencia con los 62 s de P0 es de
> entorno/medición, no del cambio de P1.

---

## C. Resultados de validación

Ejecutado de verdad en este entorno:

- `tests/test_demo11.py`: **729 comprobaciones, RESULT: SUCCESS** (eran 695; +34 nuevas).
- `tests/test_planner.py`: **839 comprobaciones, RESULT: SUCCESS**.
- `tests/dashboard_client.js` (Node, carga la página real y reproduce una sesión): la cadena, el
  objetivo, las tareas y el estado de la demo se pintan **sin excepciones ni avisos**, y reflejan
  la telemetría real (nuevas lecturas `present`, más pruebas que verifican estados, RECUPERANDO y
  que un panel en reposo no inventa progreso).
- `tests/dashboard_controls.js` (Node, pulsa los controles de verdad): abre la ventana de
  configuración, lee el formulario y envía los parámetros correctos.
- `tools/check_executor.py --headless` (paridad de agarre con demo 05): **seguimiento 0 mm,
  cubo +10,1 cm (quería +10,0), agarre cinemático, RESULT: SUCCESS** — con el render apagado y
  encendido por igual. La fiabilidad del movimiento no cambia.
- `tools/benchmark_demo11.py capture` (dos objetos, preset medido): **Completado (SUCCESS), 2
  verificadas, 0 agarres rechazados**, antes y después (tabla B).

Limitaciones de validación (no se han podido ejecutar aquí):

- **Navegador real (1366×768, 1920×1080, zoom 125 %, Chrome/Edge)**: este entorno es headless y sin
  navegador. La maquetación se validó estructuralmente (pruebas de HTML + el arnés de Node que
  ejecuta el JS de la página) y es responsive por diseño (grid con áreas + media queries), pero **no
  se ha visto en un navegador real**.
- La reconexión del WebSocket y el reinicio de la demo están cableados y probados en el arnés, no
  en vivo con Isaac.
- `check_executor.py` tiene un aviso previo (no introducido por P1): en su preámbulo con
  `DEFAULT_SCENE` la percepción ve «3 recipientes» en vez de 2, **igual con el render encendido y
  apagado**, así que no viene de este cambio; no afecta al preset de la demo (que usa `pen=True`).

---

## D. Mejoras pendientes (priorizadas)

1. **Validar en un navegador real** a 1366×768, 1920×1080 y zoom 125 %, y con la demo en vivo con el
   simulador. Es la prueba que un jurado haría y que aquí no se pudo ejecutar.
2. **Sobrecosto del paso gestionado**: tras apagar el render, `env.step` aún cuesta ~12,5 ms frente
   a ~3,9 ms de física pura; ~8 ms son observaciones/gestor de acciones de Isaac por paso. Sólo con
   una ruta de `env.step` sin observaciones, y sólo si no rompe el lazo cerrado, se podría bajar de
   los 26 s. Riesgo medio; requiere medición.
3. **Reconexión en vivo**: ensayar una caída y vuelta del WebSocket con la demo corriendo, y el
   reinicio de la demo inmediatamente después de terminada.
4. **Revisar el aviso previo de `check_executor`** («3 recipientes» en `DEFAULT_SCENE`), que es una
   percepción independiente de P1.
5. **Puesta a punto de la cadencia de la cámara**: con el movimiento 2,75× más rápido, el bombeo a
   3 fps podría subirse sin lastrar el brazo; medir antes de cambiar el valor por defecto.
