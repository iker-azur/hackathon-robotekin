# Demostraciones reales SO-101 para ACT

Grabador para **Ubuntu, SO-101 leader + follower y dos cámaras USB RGB**: muñeca (`wrist`) y cenital (`overhead`). Guarda el dataset LeRobot v3 en disco y lo publica en **un repositorio público de Hugging Face después de cada episodio aceptado**. Conserva las posiciones de ambos brazos. No necesita Isaac Sim ni los demos de simulación.

La implementación fija **LeRobot 0.6.1**. Las conclusiones se basan en la documentación oficial y en el código enlazado por versión; no mezcles instrucciones antiguas de LeRobot 0.3/0.4 con este entorno.

## 1. Qué recomienda LeRobot y qué decidimos aquí

ACT es un punto de partida recomendado por LeRobot: utiliza imágenes de varias vistas y estado articular para predecir secuencias de acciones. No necesita que calcules posiciones cartesianas ni segmentes los objetos para este montaje. Las cámaras deben tener la misma forma de imagen en la implementación fijada. La lectura separada del leader es información auxiliar: el modelo no dispondrá del leader cuando trabaje autónomamente. [ACT oficial](https://huggingface.co/docs/lerobot/act), [configuración de ACT 0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py).

La guía oficial propone comenzar con al menos unas 50 demostraciones: por ejemplo, diez por cada una de cinco posiciones del objeto. Pide mantener un comportamiento de agarre consistente, conservar los montajes de cámaras y aumentar la variación gradualmente. Recomienda comprobar que puedas completar la tarea mirando solo las cámaras. Cincuenta episodios son una referencia inicial, no una garantía de que cualquier tarea se aprenda. [Guía oficial de recogida](https://huggingface.co/docs/lerobot/il_robots).

Para este proyecto recomendamos una tarea por dataset: objetos iniciales variados dentro de un área definida, misma zona de destino y montaje de cámaras estable entre recogida y ejecución. La cámara de muñeca se mueve con el brazo; su fijación respecto al brazo debe permanecer estable. Comprueba que pinza, objeto y destino sean visibles entre ambas vistas. Evita demostraciones largas con esperas, intentos fallidos o reajustes innecesarios. Acepta solo episodios que quieras que el robot imite. Reserva episodios completos para evaluación; separar fotogramas contiguos entre entrenamiento y validación produciría una evaluación demasiado favorable. Estas son decisiones prácticas del proyecto. [Contexto oficial sobre calidad de datos](https://huggingface.co/blog/lerobot-datasets).

Los valores **640×480, 30 FPS, 60 segundos máximos y H.264** son nuestros valores iniciales, no requisitos universales de ACT. Si el equipo no sostiene la cadencia, reduce carga o crea otro dataset con menor FPS. No cambies la frecuencia declarada de datos ya grabados para ocultar retrasos. Cambios en iluminación, enfoque, encuadre o montaje pueden perjudicar la política; comprueba exposición y nitidez antes de una sesión larga.

## 2. Entorno separado en Ubuntu

Desde la raíz del repositorio, con Python 3.12 instalado:

```bash
python3.12 -m venv .venv-real-recording
source .venv-real-recording/bin/activate
python -m pip install --upgrade pip
python -m pip install -r demos/real_robot_recording/requirements.txt
python -m pip check
hf auth login
hf auth whoami
```

Utiliza un token con permisos para escribir datasets en el usuario u organización de destino. No lo incluyas en el JSON ni en comandos que vayan a compartirse. `check` verifica identidad, pero la autorización efectiva para crear/escribir ese repositorio se comprueba en la subida. El dataset será **público**, incluidos los vídeos y los valores de calibración que lo acompañan.

La versión fijada requiere Python >=3.12. La instalación de entrenamiento incluye PyTorch; para CUDA elige una distribución de PyTorch compatible con tu GPU y con los límites de dependencias de LeRobot. No actualices sin más el entorno que ya utilizas. [Dependencias oficiales de la versión](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml).

## 3. Puertos, cámaras y calibración

```bash
lerobot-find-port
lerobot-find-cameras opencv
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
```

Identifica cada brazo por separado si hace falta. En las cámaras, verifica visualmente qué identificador corresponde a muñeca y cuál a cenital; algunos dispositivos exponen varios nodos de vídeo. Prefiere enlaces persistentes `by-id` cuando el hardware los ofrezca. Si no existen o no distinguen dos cámaras idénticas, configura los índices y vuelve a verificarlos al reconectar. LeRobot advierte que los índices pueden cambiar. Cierra el visor USB y cualquier otro programa que esté usando esas cámaras. [Cámaras oficiales](https://huggingface.co/docs/lerobot/cameras).

Los usuarios de Ubuntu necesitan acceso a los dispositivos serie y vídeo. Si hay errores de permisos, comprueba los grupos `dialout` y `video` y vuelve a iniciar sesión después de cambiar la pertenencia. No ejecutes la recogida completa como root.

Antes de conectar, coloca ambos brazos en una posición de reposo apropiada y despeja el recorrido. El controlador puede habilitar torque al conectar y lo deshabilita al desconectar; sostén el brazo si puede caer. `check` no manda consignas de movimiento, pero conectar el controlador configura los motores.

Si ya tienes calibraciones, copia los JSON a directorios separados del nuevo entorno, conserva sus originales y configura los IDs que corresponden a esos archivos. `follower_calibration_dir` y `leader_calibration_dir` son directorios; dentro debe existir `<id>.json`. Si dejas `null`, LeRobot usará su ubicación habitual de calibraciones. Para generar calibraciones con el nuevo entorno:

```bash
lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/PUERTO_FOLLOWER --robot.id=robotekin_follower
lerobot-calibrate --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/PUERTO_LEADER --teleop.id=robotekin_leader
```

Si usas directorios personalizados añade `--robot.calibration_dir=/ruta/follower` o `--teleop.calibration_dir=/ruta/leader` respectivamente. El grabador exige calibraciones existentes y concordantes con los motores; no recalibra automáticamente. No intercambies calibraciones entre brazos. [Montaje y calibración SO-101](https://huggingface.co/docs/lerobot/so101).

## 4. Configurar y comprobar

```bash
cp demos/real_robot_recording/config.example.json demos/real_robot_recording/config.local.json
# Edita config.local.json: usuario/dataset, tarea, puertos, cámaras e IDs reales.
python demos/real_robot_recording/record.py check \
  --config demos/real_robot_recording/config.local.json
```

Las rutas `root` y los directorios de calibración se resuelven respecto al JSON. El ejemplo guarda los datos en `data/real_robot/so101_pick_place`, fuera de Git. Elige un nombre de repositorio nuevo que no contenga otro dataset: este directorio local será la fuente de verdad de todas sus subidas. Usa solo un equipo escritor para ese `repo_id`.

`check` prueba escritura en disco, codifica un fotograma H.264 en memoria, comprueba autenticación y lee ambas cámaras y brazos durante unos dos segundos. Guarda `check-wrist.png` y `check-overhead.png` en el directorio hermano `so101_pick_place.recorder`. Ábrelos para verificar encuadre, nitidez y colores. No crea un repositorio remoto.

`max_relative_target` limita por articulación la diferencia entre posición actual y consigna: 5 grados para el brazo y 5 puntos de apertura para la pinza por envío en el ejemplo. Es un límite de seguimiento, no un planificador de movimientos ni una garantía anticolisión. El controlador hace una lectura adicional para aplicarlo; por eso el diagnóstico sin consignas no demuestra los FPS de teleoperación. [Controlador SO-101 fijado](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/so_follower/so_follower.py).

## 5. Grabar, revisar y subir

```bash
python demos/real_robot_recording/record.py record \
  --config demos/real_robot_recording/config.local.json
```

Mantén la terminal enfocada; no hace falta pulsar Enter. Los controles usan la terminal, no atajos globales del escritorio.

| Estado | Teclas y comportamiento |
|---|---|
| Pausa | `p` habilita teleoperación; `q` sale |
| Preparación | Sigue al leader sin grabar. `s` comienza; `q` sale |
| Grabación | `e` termina para revisar; `d` descarta; `q` descarta y sale |
| Revisión | `a` acepta y publica; `d` descarta; `q` descarta y sale |
| Subida | Captura y consignas pausadas; al acabar vuelve a pausa |

Al llegar a `episode_time_s` pasa a revisión. El follower conserva su última consigna durante revisión y subida. No muevas el leader esperando que el follower le siga en esos estados. La tecla `p` exige reanudación explícita después de cada pausa; el límite de seguimiento se mantiene al volver a moverlo.

Con `display_data: true`, Rerun muestra las dos imágenes y las curvas de estado, acción y leader, organizadas por episodio. Revisa la línea temporal antes de aceptar. Requiere escritorio o una configuración adecuada para abrir el visor. Si grabas por SSH sin visor, usa `display_data: false`; las teclas siguen funcionando, pero no tendrás revisión visual en directo y debes inspeccionar los episodios guardados antes de entrenar.

El resumen por episodio incluye FPS medidos, intervalo máximo y número de intervalos superiores a 1,5 veces el periodo objetivo. Si los FPS caen por debajo del 90% aparece un aviso. La decisión de descartar queda en tus manos. Las pausas de preparación y subida no se incluyen en esas medidas.

## 6. Qué se guarda exactamente

| Campo | Forma / significado |
|---|---|
| `observation.images.wrist` | RGB, `height × width × 3` en captura |
| `observation.images.overhead` | RGB, misma forma |
| `observation.state` | `float32[6]`, posición medida del follower |
| `action` | `float32[6]`, consigna enviada después de aplicar límites |
| `leader_state` | `float32[6]`, lectura del leader previa al recorte |
| `capture_time_s` | `float64[1]`, tiempo real del ciclo respecto al inicio |
| Campos estándar | `episode_index`, `frame_index`, `index`, `timestamp`, `task_index` |

Orden: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Las cinco primeras posiciones están en grados; la pinza en 0–100. El bucle observa antes de mandar la siguiente acción. No usa la posición resultante como etiqueta de acción.

`timestamp` es la cuadrícula nominal `frame_index / fps` generada por LeRobot. `capture_time_s` conserva la cadencia real para identificar retrasos. Las cámaras USB se leen de forma asíncrona: ambas imágenes pertenecen a la misma muestra lógica, pero no hay disparo físico sincronizado ni medida del instante de exposición. Tampoco se garantiza sincronización física con los motores. Para eso haría falta otro hardware y protocolo. [Implementación OpenCV](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/cameras/opencv/camera_opencv.py).

LeRobot genera `data/`, `videos/` y `meta/`, incluidas estadísticas y límites de episodios. No presupongas un MP4 por episodio. `recording.json` añade unidades, orden, tarea, versión y calibraciones. El directorio hermano `*.recorder/` contiene bloqueo, estado de subida e informes locales; no se publica. No contiene el token. [Formato v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).

La subida ocurre con los escritores cerrados: `save_episode()`, `finalize()`, `push_to_hub()`. La captura posterior utiliza `LeRobotDataset.resume()`. Se envía el dataset con metadatos y estadísticas actualizados; el Hub puede reutilizar contenido ya existente. No se garantiza que cada transferencia tenga exactamente el tamaño del episodio nuevo. Los fragmentos de vídeo y Parquet compartidos pueden cambiar al añadir datos. [API de persistencia](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/lerobot_dataset.py).

Por defecto se guardan imágenes temporales y se codifica al aceptar. `streaming_encoding: true` usa la codificación durante captura de LeRobot, lo que reduce la espera de guardado pero carga más la CPU durante teleoperación. Prueba primero episodios cortos. [Codificación durante captura](https://huggingface.co/docs/lerobot/streaming_video_encoding).

## 7. Continuar y recuperar subidas

```bash
python demos/real_robot_recording/record.py record \
  --config demos/real_robot_recording/config.local.json --resume
python demos/real_robot_recording/record.py upload \
  --config demos/real_robot_recording/config.local.json
python demos/real_robot_recording/record.py validate \
  --config demos/real_robot_recording/config.local.json
```

`upload` y `validate` no conectan hardware. Las subidas se intentan tres veces, con esperas de 2 y 4 segundos. Si fallan, se conservan los episodios locales; la sesión puede continuar y la siguiente subida incluye lo pendiente. Un reintento no vuelve a grabar ni incrementa episodios. `upload` devuelve un código de error si sigue fallando. No se borran datos después de subir.

`--resume` comprueba contrato, datos aceptados y calibraciones. Una tarea, forma de imagen, frecuencia, ID o calibración incompatible requiere otro dataset. El bloqueo del sistema operativo evita dos procesos locales sobre el mismo directorio y se libera incluso tras un cierre abrupto.

Con `q` o Ctrl+C durante captura se descarta el episodio sin aceptar y se cierran recursos. Un error de cámara o serie aborta la sesión sin aceptar el episodio. Si falla el guardado o hay un cierre abrupto, el diario queda marcado y no se reanuda automáticamente. Conserva una copia de ambos directorios y ejecuta `validate` para diagnosticar. No elimines el diario ni cambies su fase manualmente para forzar la continuación. La validación no es una reparación y no garantiza recuperar un corte de corriente.

## 8. Comprobar la copia del Hub y entrenar ACT

Después de publicar:

```bash
hf download TU_USUARIO/so101_pick_place --repo-type dataset --local-dir /tmp/so101_hub_check
python demos/real_robot_recording/record.py validate \
  --config demos/real_robot_recording/config.local.json --root /tmp/so101_hub_check
```

`validate` comprueba esquema, orden, tarea, valores finitos, índices, tiempos, estadísticas y decodificación del primer y último fotograma de cada episodio. Revisa además los vídeos completos con el [visualizador oficial](https://huggingface.co/docs/lerobot/using_dataset_tools); comprobar extremos no detecta todos los defectos visuales intermedios.

El entrenamiento estándar detecta las dos vistas, `observation.state` y `action`. Las columnas `leader_state` y `capture_time_s` no se convierten en entradas de la política por sus nombres. La prueba de integración verifica esa exclusión y ejecuta un paso real de optimización.

```bash
lerobot-train \
  --dataset.repo_id=TU_USUARIO/so101_pick_place \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_so101_pick_place \
  --job_name=act_so101_pick_place \
  --batch_size=8 \
  --steps=100000 \
  --wandb.enable=false
```

Usa un directorio de salida nuevo para otro entrenamiento. Las 100.000 iteraciones y el lote 8 son valores iniciales que se deben ajustar al dataset y la GPU; no son una garantía de rendimiento. La descripción `task` documenta la demostración: ACT estándar no interpreta por sí mismo instrucciones de lenguaje para escoger entre varias tareas. [Entrenamiento ACT](https://huggingface.co/docs/lerobot/act).

Para desplegar conserva nombres y montaje de cámaras, unidades, FPS, orden articular y calibración del follower. Este módulo no ejecuta políticas ni reproduce movimientos automáticamente.

## 9. Pruebas y límites de la verificación

```bash
python -m pip install pytest ruff
python -m pytest demos/real_robot_recording/tests -q
ruff check demos/real_robot_recording
```

Las pruebas no conectan robots ni publican nada. Cubren el retorno recortado de `send_action`, lectura única del leader, sensores que fallan, descarte, bloqueo, reintentos, interrupciones, vídeos H.264 RGB en ambos modos, reanudación y un paso de ACT sin descargar pesos. El test de ACT usa un modelo reducido y CPU; demuestra compatibilidad de datos, no aprendizaje de la tarea.

La aceptación física pendiente consiste en identificar y comprobar ambos brazos/cámaras en Ubuntu, grabar dos episodios, descartar uno, reiniciar y continuar, publicar y descargar el resultado en un directorio limpio. Después hay que evaluar una política entrenada mediante ejecuciones reales de la tarea. Ninguna prueba sintética sustituye esas comprobaciones.

Verificación del 23-10-2026: **29 pruebas pasadas en 65,01 s**, `ruff check` sin incidencias y `pip check` sin dependencias rotas. Se ejecutaron en Windows con Python 3.12.13, LeRobot 0.6.1, PyTorch 2.11.0 CPU, NumPy 2.2.6, PyAV 15.1.0 y datasets 4.8.5, en un entorno de pruebas separado. Se comprobó la API de Rerun mediante una grabación en memoria, sin abrir su ventana. No se conectaron dispositivos ni se efectuaron subidas reales.
