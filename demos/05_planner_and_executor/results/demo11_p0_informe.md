# Demo 11 — P0: fiabilidad, rendimiento y arranque

Fecha: 2026-10-10. Rama: `prepare-demos`. Entorno: Windows, Conda `hackathon-robotekin`,
Isaac Sim 5.1, RTX 5060 Ti.

Los seis P0 de la revisión están implementados y medidos. Nada se ha implementado sin medida
que lo respalde, y donde una medición contradijo el plan, el plan se revirtió y quedó
anotado.

---

## A. Cambios implementados

### P0.1 — Arranque de Isaac: causa raíz y reintentos

**Causa raíz (confirmada por medición).** `Windows fatal exception: code 0xc0000139` es
`STATUS_ENTRYPOINT_NOT_FOUND`. Los registros de `out11/benchmark_capture_*.log` lo sitúan en
el cargador de plugins de Kit, junto a estos errores, siempre en el mismo sitio del arranque:

```
[Error] [omni.ext.plugin] Could not load ... generic_mo_io.dll
ImportError: DLL load failed while importing _generic_model_output
[Error] [carb] [Plugin: omni.sensors.nv.lidar.plugin.dll] ... a dependent library failed to load
[Error] [carb] [Plugin: omni.sensors.nv.radar.plugin.dll] ... a dependent library failed to load
Windows fatal exception: code 0xc0000139
Windows fatal exception: access violation
```

Las extensiones responsables son `isaacsim.sensors.rtx`, `omni.sensors.nv.lidar`,
`omni.sensors.nv.radar` y `omni.sensors.nv.ultrasonic`: sensores que este proyecto no usa y
cuyas DLL nativas esta máquina no puede resolver. Un hilo de Kit muere dentro del cargador, sin
marco de Python, así que no hay nada que capturar.

**El arreglo.** `tools/run_demo11.py` ya había intentado un ajuste carb y comprobó que se
ignoraba. La vía que sí funciona es la línea de comandos de Kit, y `AppLauncher` la expone:
`kit_args`, que `isaaclab/app/app_launcher.py:773` añade a `sys.argv`, que es lo que
`SimulationApp` lee. `env_loader.prepare_runtime` ahora lo pasa:

```python
_ = AppLauncher({"enable_cameras": True,
                 "kit_args": disable_unused_sensor_extensions()})
```

**Qué se ha conseguido y qué no.** Con la extensión desactivada el mensaje `0xc0000139`
**sigue apareciendo** en la salida —lo emiten hilos de Kit que informan, no que abortan— pero el
proceso **termina con código 0 y llega a `SimulationContext`**. Medido con cuatro arranques
reales seguidos: 4 de 4 llegan a listo, 4 de 4 con código de salida 0. Es decir, lo que era un
fallo fatal pasa a ser ruido. **No es una solución completa**: el texto sigue saliendo y no se ha
perfilado de dónde sale exactamente.

**Archivos:** `demos/01_simulation_basics/env_loader.py` (constante
`DISABLED_EXTENSIONS`, `disable_unused_sensor_extensions()`, y el `AppLauncher` con
`kit_args`); `results/demo11_perf.md` (apartado de arranque con la tabla de los 4 intentos).

**Reintentos automáticos.** `supervisor.py` gana `launch()` / `retry_or_give_up()`:

- hasta `MAX_START_ATTEMPTS = 3` intentos (`--start-attempts` para cambiarlo);
- solo se reintenta si el hijo muere **antes** de estar listo y **sin informe**: una parada
  pedida y una sesión completida nunca se relanzan;
- el intento en curso se publica como `start_attempt` («1/3», «2/3»), que el panel muestra;
- el motivo se lee del registro de la sesión con las firmas de `run_demo11.py`
  (`STATUS_ENTRYPOINT_NOT_FOUND`, `access violation`, `single TORCH_LIBRARY`, etc.), así que
  el panel dice *por qué* reintenta;
- al agotar los intentos el error nombra el número de intentos y el último motivo;
- `terminate()` mata el árbol de procesos, porque un kernel de Kit en fallo se queda con la GPU
  y el siguiente arranque se quedaba colgado detrás de él.

De paso se corrigió un bug real que encontraron las pruebas: al agotar los intentos el
supervisor dejaba `self.child` apuntando a un proceso terminado, y `poll()` volvía a recorrer
toda la ruta de tiempo de espera una y otra vez. Ahora `retry_or_give_up` cierra el intento
antes de decidir.

**Pruebas:** `tests/fake_session.py` gana un doble de sesión que muere como lo hace Isaac
(sin informe, con la firma en el registro), y `tests/test_demo11.py` cuatro pruebas nuevas:
un fallo y recuperación al segundo intento; tres fallos y rendición con motivo; una parada
intencional que no relanza nada; y un arranque colgado que también se reintenta.

### P0.2 — Escena estable de demo

`scene_config.demo_config()`: dos cubos de 30 mm (rojo en x 0,38, azul en x 0,43) y sus dos
bandejas de 70 mm, dentro del área de trabajo medida. Es la configuración que la medición
había registrado como la única sin agarres rechazados, ahora escrita en un sitio en lugar de
estar repartida entre un JSON guardado y la memoria de quien la usó.

Se sirve desde el backend (`/api/config` devuelve `presets`, y `GET /api/config/preset/demo`
la carga y la guarda), y el panel tiene un botón **«demo: 2 cubos»** que la aplica y arranca.
Antes la escena simple estaba codificada a mano en JavaScript con coordenadas que podían
discrepar de las de Python; ahora hay una sola definición.

**Bug encontrado y corregido por el camino:** `scene_config.load(None)` significaba
`default_config()`, así que una ejecución sin `--scene-config` **ignoraba** la escena que el
panel había guardado y ponía en la mesa los cuatro objetos por defecto —dos de ellos bolas—.
Ahora `load()` usa el fichero guardado antes de caer al defecto. Es la diferencia entre la
escena que se midió y una que no.

**Pruebas:** dos pruebas nuevas —que el fichero guardado es lo que carga una ejecución
cualquiera— y —que el preset es la escena medida de dos cubos—.

### P0.3 — Estados de ejecución correctos

El panel seguía diciendo «ahora: retirar · paso 8 de 8» cuando el brazo ya estaba quieto,
porque `progress`, `step_index` y `step_total` se escribían y nunca se borraban, y la fase
seguía siendo `command`/`sorting` durante toda la verificación y recuperación.

- `MonitoredRobot._motion_finished()` / `_motion_started()` publican un indicador `moving`,
  que solo es cierto entre el primer y el último paso de un plan. Se llama al terminar
  `run_plan`, al empezar cada orden, y al entrar y salir de `check` y `_retry_one`.
- El panel ya no deduce el presente de la fase: lo deduce de `moving`, y las fases
  `verifying` («midiendo dónde ha quedado cada pieza») y `recovering` («recuperando una pieza
  que no quedó bien») tienen su propia frase.

**Pruebas:** la prueba existente de la línea «ahora» se actualizó al nuevo indicador, y se
añadió una de regresión específica: un plan terminado, con su último paso y su fase todavía
puestos, no puede producir una frase en presente.

### P0.4 — Rendimiento medido

Medido con el entorno real, no con una escena sintética (`results/demo11_perf.md`):

| Fase | Medido |
|---|---|
| `physics_dt` / `decimation` | 1/60 s / 1 |
| Paso con el brazo en movimiento | **33,03 ms** |
| Paso nominal | 16,67 ms |
| **Factor de tiempo real** | **0,505** |
| `run_plan` de una pieza | 33,1 s |
| `park` | 1,3 s |
| `capture_rgb` | 30,3 ms |

**No es la cámara.** Medido: `capture_rgb` cuesta 30 ms y se ejecuta entre pasos, así que no
engorda el paso de física. El factor 0,505 es la causa de todo lo demás y su origen completo
sigue sin atribuirse: falta perfilar `env.step` con el brazo parado y en movimiento, que es la
medida que este pase no llegó a hacer. Queda anotado como pendiente en el informe en lugar de
disfrazarse de arreglado.

**Recorte de pasos, con una revolución.** `TRACK_TRAVEL_STEPS` 150 → 120: es un desplazamiento
en aire libre, sin pieza en juego, y ahorra ~1,2 s por pieza. `TRACK_DESCEND_STEPS` se probó en
100 y se **revirtió a 200**: bajó el pick a 14,9 s pero apareció un *stale grasp* —las mordazas
cerraron donde la pieza ya no estaba— porque el servo de centrado no había convergido. Está
medido y documentado en el comentario del código y en el informe, con el número.

### P0.5 — Diagrama recortado

`#route .body` tenía `overflow: hidden` y ningún `flex`, así que su hijo con
`height: 100%` medía contra un padre de alto automático: el SVG se dimensionaba por su
proporción (~790 px de alto) y la banda inferior se cortaba. A 1366×768 faltaban ~85 px y la
caja «informe» desaparecía entera.

Ahora `#route .body` es un contenedor flex y el SVG un ítem `flex: 1 1 auto; min-height: 0`,
con `preserveAspectRatio="xMidYMid meet"`: recibe un alto definido y **encaja entero** en lugar
de cortarse. Las demás tarjetas del panel ya tenían ese `flex`; era una omisión.

### P0.6 — Cámara durante la verificación

`Executor.settle()` avanza la física sin pasar por `keys()` ni por `frame_hook`, así que los
asentamientos que hay entre una colocación y su medición dejaban la cámara congelada varios
segundos —justo cuando más se mira— y el panel marcaba «imagen sin actualizar».

- `Executor` gana `settle_hook`, que se llama cada `SETTLE_HOOK_EVERY = 15` pasos (no en cada
  paso: un fotograma es un render, y 15 pasos son un cuarto de segundo a 60 Hz, 15 veces más
  barato que uno por paso).
- `MonitoredRobot` conecta ahí el mismo `cam_pump` que ya existía, que va por el
  `FrameEncoder` asíncrono: un hilo, un solo fotograma pendiente, y el hilo de física nunca
  espera a un JPEG.
- El aviso del panel pasa de 3 s a 6 s y dice «¿está el simulador ocupado?» en lugar de
  «imagen sin actualizar», porque durante una verificación quieta la foto legitímamente no
  cambia y eso no es un fallo.

---

## B. Causa raíz del fallo de arranque

**Qué lo causaba.** Cuatro extensiones de sensores (`isaacsim.sensors.rtx`,
`omni.sensors.nv.lidar`, `omni.sensors.nv.radar`, `omni.sensors.nv.ultrasonic`) cuyas DLL
nativas esta máquina no puede resolver. `0xc0000139` = `STATUS_ENTRYPOINT_NOT_FOUND`: una DLL
que carga pero no exporta un símbolo que otra espera. Ocurre en un hilo del cargador de Kit,
sin marco de Python, así que no es capturable.

**Evidencia.** (1) Cada registro de `out11/benchmark_capture_*.log` lleva los mismos tres
errores de DLL en el mismo punto del arranque. (2) Con las cuatro desactivadas, cuatro
arranques reales seguidos llegan a `SimulationContext` **con código de salida 0** en los cuatro,
mientras el texto `0xc0000139` sigue apareciendo en los cuatro — que es la prueba de que ese
texto lo emite un hilo que informa y no uno que aborta. (3) `tools/run_demo11.py` documenta
que un ajuste carb equivalente se ignoraba, porque la vía correcta era la línea de comandos.

**Qué se ha arreglado.** Que el fallo deje de ser fatal: el proceso sobrevive y arranca. La
recuperación automática cubre el resto, que es la parte que un jurado vería.

**Qué no está resuelto.** El texto sigue imprimiéndose, y no se ha perfilado qué hilo lo emite
ni por qué la máquina no resuelve esas DLL (no es una dependencia que este proyecto pueda
actualizar ni un ajuste de entorno que convenga inventar). Mitigado, no resuelto.

**Cómo funciona la recuperación.** `Supervisor.poll` detecta que el hijo murió antes de estar
listo, lee del registro de la sesión la firma del fallo, la publica, mata el árbol de procesos
por si un kernel de Kit se quedó con la GPU, y relanza con la misma escena y los mismos ajustes.
Lo hace hasta tres veces; a la tercera se rinde con `No se ha podido iniciar Isaac Sim después
de 3 intentos. Último motivo: …`, y el botón de inicio vuelve a estar disponible.

---

## C. Antes y después

| Métrica | Antes | Después (medido) |
|---|---|---|
| Arranques que llegan a listo (4 intentos reales) | fallaban con `0xc0000139` | **4 de 4**, código de salida 0 |
| Reintentos de arranque | ninguno: la sesión terminaba | hasta 3, con motivo visible en el panel |
| Escena de una ejecución sin `--scene-config` | la de 4 objetos con bolas, ignorando la guardada | la escena guardada |
| Preset de demo validado | dos cubos medidos pero no escritos en ningún sitio | `demo_config()`, 5/5 SUCCESS 2/2, 0 rechazos |
| Estados del panel | «ahora: retirar · paso 8 de 8» con el brazo quieto | indicador `moving`; verificando/recuperando con frase propia |
| Diagrama en 1366×768 | banda «informe» recortada (~85 px) | entra entero (corregido por CSS; verificado cargando la página en el arnés Node) |
| Congelación de cámara en `settle` | presente, varios segundos | `settle_hook` cada 15 pasos, mismo canal asíncrono |
| Paso de física | 33,0 ms | 33,0 ms (no tocado: ver P0.4) |
| Tiempo por pieza | ~26 s | ~24,8 s (solo el desplazamiento recortado) |
| Orden de dos piezas, 5 ejecuciones | 2/2 pero con agarres rechazados en la línea base | **5/5 con 2/2, 0 rechazos, 0 forzadas** |

Dos cosas que **no** se han conseguido, y conviene decirlo: el paso de física sigue al doble
 de su valor nominal, y el recorte del descenso que habría bajado la orden a ~45 s se revirtió
 porque perdía agarres. El objetivo de 45 s no se alcanza.

---

## D. Pruebas ejecutadas

| Comando | Resultado |
|---|---|
| `python tests\test_demo11.py` | **695 comprobaciones, SUCCESS** (eran 537 al empezar) |
| `python tests\test_planner.py` | **839 comprobaciones, SUCCESS** |
| `python tools\benchmark_demo11.py intent` | modelo 63/87, híbrido 62/87, intención 80/87 — igual que la línea base |
| `python tools\benchmark_demo11.py monitor` | websocket y arnés Node: panel ~75 ms / 1028 escrituras, sin errores |
| 4 arranques reales de Kit seguidos | 4/4 llegan a listo, código 0 |
| 5 ejecuciones del preset vía `benchmark_demo11.py capture` | **5/5 SUCCESS, 2/2 verificadas, 0 rechazos** |
| Medición de fases (`probe_perf`) | ver `results/demo11_perf.md` |

Las pruebas de reintento usan un doble de sesión que muere como muere Isaac, así que se
ejecutan en segundos y sin GPU. Las de arranque real y las del preset necesitan el simulador.

**Pendiente de validar en la máquina del hackathon:** la prueba de resistencia completa (diez
arranques seguidos, parada en cada etapa del movimiento, reinicio en caliente, pantalla
pequeña, voz real en Chrome/Edge). Se han ejercido por separado, no como guion completo.

---

## E. Problemas que quedan

| Prioridad | Problema |
|---|---|
| Alta | El paso de física cuesta el doble de lo que debería (factor 0,505). Sin atribuir: falta perfilar `env.step` con el brazo parado y en movimiento. Es el objetivo con más impacto sobre el tiempo de demo. |
| Alta | El texto `0xc0000139` sigue apareciendo en el registro de cada arranque. Ya no es fatal, pero asusta a quien lo lee y no se ha localizado el hilo que lo emite. |
| Media | La demostración en vivo no tiene todavía el «botón único» completo del guion: el botón carga la escena medida y envía la orden, pero falta validarlo en una sesión real con el panel abierto. |
| Media | `carry` (120 pasos) y `retreat` (60) son traslados sin contacto con la pieza y podrían recortarse como se hizo con `travel`. No se hizo para no cambiar dos cosas a la vez. |
| Baja | Las bolas siguen siendo el caso que la física no resuelve; el preset las evita, pero la escena por defecto sigue teniéndolas. |
