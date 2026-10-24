# HMI · SO-101

Demo local de la célula con movimiento articulado, suelo cuadriculado de 5 cm,
mesa 3D a pantalla completa, piezas seleccionables y cámara cenital flotante.
La mesa de demo incluye las tres cajas de la foto: bandeja verde con piezas
metálicas aproximadas, caja naranja abierta y cesta blanca de rejilla. El ciclo
de inspección de la demo no conecta el robot físico ni un modelo visual.

Desde la raíz del repositorio:

```bash
.venv/bin/python demos/hmi/server.py --open
```

Abre `http://127.0.0.1:8790`. Esta versión necesita el servidor; abrir el HTML
con `file://` no permite cargar los módulos y la telemetría. Se necesita NumPy
(disponible en el entorno del proyecto). Three.js, OrbitControls y cadena cinemática se reutilizan de RecoveryKit, sin descargar recursos.
El modelo detallado naranja y negro de `robot-model.js` adapta las piezas del HTML
independiente a las poses del HMI: incluye servos, horquillas, cables, pinza y
cámara virtual situada en la lente del soporte. Las dimensiones son aproximadas;
no es un CAD calibrado. La demo conserva su cinemática y no ofrece cambios de
escala que alteren su trayectoria. Para ajustar proporciones manualmente, usa
`so101-model.html`.

- La mesa: navegación 3D; arrastrar con el botón izquierdo para orbitar.
- Desde la garra: ampliar la cámara virtual solidaria a la pinza.
- Desde arriba: orientar la escena en cenital.
- Botón derecho y arrastrar: desplazar.
- Rueda o botones +/−: zoom.
- En pantalla táctil: un dedo para girar; dos para desplazar y hacer zoom.
- Pulsar una pieza o su etiqueta: seleccionar y consultar su ficha.
- Centrar: restablecer el encuadre.
- Ampliar: pantalla completa de la escena.
- Pausar demo: detener la animación, conservando la navegación 3D.

La cámara flotante muestra el puesto cenital indicado para la inspección.
Si se interrumpe la fuente, la HMI marca la última posición como antigua
y deja de mostrar la cámara como una señal en directo.

## Demo de inspección

La demo mueve una pieza gris desde la cesta roja hasta el puesto cenital situado
a la izquierda del robot, orienta su cara plana hacia arriba y la lleva a una
caja según una etiqueta de muestra. «Con perno» va a la caja verde y «sin
perno» a la cesta blanca. El resultado está preetiquetado para mostrar el
recorrido: no hay un modelo visual de pernos entrenado ni conectado. La ventana
flotante muestra la vista cenital del puesto. La telemetría de una fuente real
se sigue mostrando como recibida; el recorrido automático pertenece a la demo.

## La mesa como interfaz

La mesa ocupa la escena completa, con fondo claro, piezas etiquetadas y
controles flotantes. El control de voz y texto queda abajo a la izquierda.
El plan aparece cuando se analiza una orden.

Una orden válida resalta las piezas coincidentes y dibuja una ruta sobre la
mesa hasta una zona de destino propuesta. La ruta no es una trayectoria del
controlador. Si faltan piezas, no coinciden sus tipos o la orden necesita
aclaración, no se dibuja un movimiento hacia un objeto inexistente. La tarjeta
flotante del plan conserva todos los pasos y los avisos.

La captura existente de micrófono activa un borde rojo pulsante
y el rótulo **REC · AUDIO**. No indica que se esté grabando vídeo o un episodio.
Con la preferencia de movimiento reducido, el borde es estático.

**Fantasma:** una línea punteada representa la referencia y una línea naranja
continua acumula el recorrido del robot durante el ciclo de 12 s. Las dos
trayectorias son sintéticas y están identificadas como ejemplos: no se ha
cargado una demostración humana ni una política aprendida. El indicador en
milímetros calcula la distancia euclídea entre las puntas de los dedos en el
mismo instante; no es una puntuación de similitud. Se puede ocultar la capa.
`GET /api/demo-reference` proporciona el ciclo de referencia y la telemetría
incluye las dos posiciones TCP sincronizadas y su distancia.

**Dónde mira:** el botón **Mostrar mapa de ejemplo** activa un halo ilustrativo
sobre la pieza que se inspecciona desde la vista cenital. Sigue la perspectiva y
permite ajustar la opacidad. No procede de una red y no explica decisiones.
Se mantiene sobre la cámara flotante o la cámara ampliada según la vista.
Se oculta al perder la señal o al utilizar una fuente externa.
Si la pieza sale del campo de visión, no se dibuja un halo artificial
en otra posición.

### Buddy visual

El buddy tiene una pantalla en el torso que renderiza la cámara de la garra de
la escena, acompaña las articulaciones que publica la fuente, reacciona al
nivel de audio del micrófono y señala el destino previsto. Las predicciones de
la demo se identifican como ilustrativas. Sus comentarios y bitácora resumen
fases, entregas observadas y fallos de la fuente; no sustituyen los datos de la
simulación ni sus contadores. Tras cada entrega observada incorpora accesorios
visuales. Cuando entra en una caja puede quedar parcialmente oculto por sus
paredes, y se asoma si se acerca el brazo.

Regla de entorno del Buddy (aplicable al prompt de comportamiento):

```text
REGLA DE ENTORNO
- El buddy es una capa puramente visual. Nunca modifica cajas, piezas, contadores ni el estado de la simulación.
- Solo lee: posición de cajas, posición del brazo, eventos de pieza y volumen.
- Puede ocultarse parcialmente tras el dibujo de una caja (capa z inferior), sin colisionar con nada.
- Si el brazo se acerca a menos de una distancia configurable, el buddy se aparta.
```

Para una futura conexión a datos de política, se ha identificado
[lerobot-attention-visualizer 0.1.2](https://pypi.org/project/lerobot-attention-visualizer/0.1.2/).
Su documentación distingue el rollout de atención de SmolVLA/π0 de la magnitud
de activación de la última convolución de ACT. Publica imagen, mapa y overlay
en Rerun por cámara. No está instalado ni conectado a esta demo. Esa integración
necesitará una política, observaciones y mapas sincronizados con la cámara;
los pesos de atención no deben presentarse como una explicación causal.

## Órdenes por voz, intención y plan

La barra flotante inferior permite dar órdenes. El micrófono real
solo se solicita al pulsar el botón de micrófono. La grabación muestra nivel
de señal y tiempo; termina con **Detener y transcribir** o a los 30 segundos.
Se libera el micrófono al detener, al fallar o al abandonar la página.
El audio queda disponible para escucharlo o descargarlo en el navegador;
no se escribe una grabación en el servidor.

La captura usa [MediaRecorder](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder),
con WebM/Opus, Ogg/Opus o MP4 según el navegador. La transcripción usa
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) en el servidor local,
sin servicios de reconocimiento en la nube. Para prepararla:

```bash
uv pip install --python .venv/bin/python -r demos/hmi/requirements-voice.txt
.venv/bin/python demos/hmi/server.py --prepare-voice
.venv/bin/python demos/hmi/server.py --open
```

La preparación descarga Whisper `tiny` la primera vez en `.cache/whisper/`,
excluido de Git. También se puede usar `--whisper-model base` u otra ruta o
tamaño compatible; se debe usar el mismo modelo al preparar y al arrancar.
Sin el motor instalado se puede grabar y reproducir audio, escribir la orden
y obtener su intención y plan. La interfaz muestra el requisito pendiente.

Tras transcribir, se analiza la orden y se muestran intención, parámetros y
pasos. El texto se puede corregir y volver a analizar. Las reglas se reutilizan
de `roboqc/phase_1/src/roboqc/commands`; no se presenta el clasificador como
un modelo entrenado ni se inventan porcentajes de confianza.

Ejemplos:

- «Inspecciona todas las piezas»: inspección de lote y pasos de observación.
- «Coloca dos piezas azules en la caja roja»: selección, recogida, transporte y verificación.
- «Pausa el proceso»: propuesta de pausa.

El plan es una propuesta de demo, independiente de la animación. No ejecuta
acciones ni marca pasos como completados. Las órdenes desconocidas, negadas
o incompletas muestran aclaraciones sin generar un plan de movimiento.

## Integrar y ejecutar la HMI en el PC del robot

Este apartado es la guía de integración para Codex cuando trabaje en el PC que
controla el SO-101. La meta es conservar la web y las funciones actuales de la
HMI, integrar los cambios de `feat/hmi-live-view` en `main` sin descartar
cambios recientes de `main`, y ejecutar en ese mismo PC tanto el servidor web
como la teleoperación. El navegador puede abrir la web localmente en
`http://127.0.0.1:8790`; el publicador de poses debe usar también
`http://127.0.0.1:8790/api/telemetry`, sin depender de otra máquina ni de la
red Wi-Fi.

### Instrucciones para Codex en el PC del robot

1. Lee esta sección y revisa `git status`, la rama activa y las diferencias
   locales antes de integrar. Conserva los cambios existentes de `main`, en
   especial `comandos_so101/lerobot_teleoperate_hmi.py` y
   `demos/06_imitation_learning/17b_record_real_so101_live.py`. No resetees,
   limpies ni sobrescribas trabajo local.
2. Integra la rama HMI en `main` preservando esos cambios. Si la rama HMI o sus
   archivos no están disponibles en ese checkout, informa qué falta; no
   declares el merge completado.
3. Ejecuta la web con `demos/hmi/server.py --source publisher` y conserva todas
   las funciones y recursos del HMI. Actualiza el lanzador de LeRobot para que
   publique al loopback local (`127.0.0.1:8790`), no a la IP de otro PC.
4. Toma las posiciones medidas del objeto follower a partir de la observación
   que ya lee el ciclo de LeRobot. No abras un segundo puerto serie, no cambies
   `send_action` ni envíes comandos de movimiento desde la HMI. Mantén el POST
   de telemetría en segundo plano, limitado a 20 Hz y conservando la muestra
   reciente. La API acepta `{"joints": {...}}` con los seis ejes
   `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` y
   `gripper`, en radianes de la convención SO-101 de
   `demos/08_recovery_kit/assets/so101/kinematics.json`.
5. Comprueba signos, ceros, límites y apertura de pinza con la calibración de
   LeRobot y el modelo de la HMI. No supongas que los puertos `/dev/ttyACM0` y
   `/dev/ttyACM1` conservan el mismo rol tras reconectar USB.
6. No ejecutes teleoperación, calibración ni configuración de motores durante
   la integración. Deja la prueba física para cuando la persona responsable
   pueda iniciarla de forma segura. Verifica la web y el endpoint local sin
   hardware; informa de cualquier mapeo físico aún no confirmado.

En la instalación que se documentó, el comando de grabación usa el USB
`5A7A054669` para el leader y `5B3E090058` para el follower. En la última
consulta, esos enlaces correspondían respectivamente a `ttyACM1` y `ttyACM0`;
usa rutas estables `/dev/serial/by-id/` y vuelve a confirmar el rol en el PC
antes de iniciar el brazo. Un intento previo usó los roles invertidos y falló
al configurar el motor ID 4 (`wrist_flex`); no repitas ese comando con los
números `ttyACM` asumidos. Ese intento aceptó las calibraciones guardadas y
registró que las escribía en los motores; antes de otra prueba, verifica en
modo de solo lectura que cada brazo conserva la calibración correspondiente a
su ID (`mi_leader` o `mi_follower`). No ejecutes `lerobot-setup-motors` ni
recalibres como parte de la integración HMI.

### Arranque local después de integrar

Desde la raíz del repositorio, en el PC del robot, la web se inicia en una
terminal:

```bash
cd ~/Workspace/Robotekin_
source .venv/bin/activate
python demos/hmi/server.py --source publisher --open
```

El servidor debe permanecer abierto. La teleoperación se inicia en otra
terminal del mismo PC con `comandos_so101/lerobot_teleoperate_hmi.py`, usando
los puertos estables verificados para ese equipo. El lanzador debe publicar a
`http://127.0.0.1:8790/api/telemetry`. En el navegador se abre
`http://127.0.0.1:8790`; `GET /api/state` debe mostrar `connected: true` y una
edad de pose reciente durante la teleoperación. La publicación de articulaciones
no transmite la imagen de cámara: esta sigue siendo una integración aparte.

Para publicar también desde otros programas LeRobot que lean el mismo
`SOFollower`, instala una vez el hook en este entorno:

```bash
.venv/bin/python comandos_so101/install_hmi_autopublish.py
```

El hook se aplica a procesos Python nuevos ejecutados con este `.venv`. Reutiliza
la observación de `SOFollower`, sin otra lectura serie ni cambios en
`send_action`. Los programas que usen otro intérprete o accedan al bus fuera de
`SOFollower` no quedan cubiertos.

La web conserva sus funciones de voz, plan visual, escena 3D, Buddy y demo de
inspección. La HMI solo muestra estado; no controla motores ni clasifica piezas
físicas por sí misma. El clasificador real y la imagen de cámara deben
conectarse por separado si se necesitan en esta ejecución local.

La fuente de RecoveryKit sigue disponible como simulación, no como posición
física:

```bash
python demos/hmi/server.py --source http://127.0.0.1:8770 --open
```

Endpoints de voz: `GET /api/voice/status`, `POST /api/voice/transcribe` (bytes
de audio) y `POST /api/command` (`{"text":"…","source":"text|voice"}`). El
navegador captura el micrófono tras autorización del usuario.

## Conectar una fuente más adelante

No se conecta hardware por defecto. Para observar una instancia existente de
RecoveryKit, sus endpoints proporcionan las poses reales de la simulación y
la imagen de su cámara:

```bash
.venv/bin/python demos/hmi/server.py --source http://127.0.0.1:8770 --open
```

También existe `--source publisher`: el proceso que ya posee el robot puede
publicar JSON a `POST /api/telemetry`, sin abrir una segunda conexión serie.
Se admiten `links: {nombre: [[x,y,z], [w,x,y,z]]}` en metros y marco Z-up,
o `joints` con los seis nombres de articulación y valores en radianes en la
convención del modelo USD. Los valores normalizados de LeRobot necesitan
su conversión y alineación antes de publicarse. La HMI no hace esa calibración.
Los JPEG se publican como bytes a `POST /api/frame.jpg`.

`--camera /dev/videoN` permite seleccionar explícitamente una cámara física.
Requiere OpenCV. No debe abrirse desde aquí si otro proceso ya la utiliza.
No hay endpoints de control de motores.

## Verificación

```bash
.venv/bin/python -m unittest discover -s demos/hmi/tests -v
node --check demos/hmi/app.js
node --check demos/hmi/voice.js
node demos/hmi/tests/test_voice.mjs
node demos/hmi/tests/test_scene_plan.mjs
```

Las pruebas cubren cinemática, animación y pausa, pérdida de señal,
independencia de la cámara, validación de mensajes y recursos del visor.
También cubren interpretación de cantidades y destinos, negaciones, órdenes
incompletas, rutas de voz, validación del audio y liberación del micrófono.
Las pruebas de captura usan APIs de navegador simuladas y las de transcripción
dobles del motor; no sustituyen una prueba de micrófono y Whisper reales.
