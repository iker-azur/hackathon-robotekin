# Demos 12–16 — Aprendizaje por imitación en el SO-101 simulado

Este conjunto de demos entrena al brazo SO-101 **a partir de demostraciones**, no
a fuerza de prueba y error con recompensas. Es el paso siguiente a los demos 01–11:
allí el robot seguía un guion o un plan; aquí mira la cámara y decide solo.

El recorrido es el de siempre en aprendizaje por imitación:

```
            ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  demo 12 ──►│  ver qué hay │──►│ demo 13      │──►│ demo 14      │──►│ demo 15      │ ──► demo 16
 (opcional) │  en el dataset│   │ grabar demos │   │ Behavior Cl. │   │ ACT          │   │ ejecutar en
            └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘     la simulación
```

**Demo 13 es el único paso obligatorio**: graba un dataset con un profesor
programado. Los demás leen ese dataset.

---

## Tabla de contenido

1. [Qué es el aprendizaje por imitación](#qué-es-el-aprendizaje-por-imitación)
2. [Qué tecnología se ha elegido y por qué](#qué-tecnología-se-ha-elegido-y-por-qué)
3. [La tarea](#la-tarea)
4. [Requisitos](#requisitos)
5. [Cómo ejecutar cada demo](#cómo-ejecutar-cada-demo)
6. [Qué sale en pantalla](#qué-sale-en-pantalla)
7. [Dónde están los datos y los modelos](#dónde-están-los-datos-y-los-modelos)
8. [Resultados medidos](#resultados-medidos)
9. [Limitaciones](#limitaciones)
10. [Por dónde seguir](#por-dónde-seguir)
11. [Referencias](#referencias)
12. [Problemas habituales](#problemas-habituales)

---

## Qué es el aprendizaje por imitación

El aprendizaje por imitación (**IL**, *imitation learning*) entrena una política
copiando lo que hace un experto, en lugar de buscarlo probando acciones y
recibiendo recompensas.

Se parte de un dataset de demostraciones `(observación, acción)`, se entrena una
red para que prediga la acción del experto a partir de la observación, y después
se ejecuta esa red en el robot. En este proyecto:

| paso | quién | qué hace |
|---|---|---|
| 1 | profesor | guion cinemático que sabe dónde está el cubo |
| 2 | script de grabación | escribe `(cámara, articulaciones) → acción` en formato LeRobot |
| 3 | red | aprende ese mapeo con LeRobot |
| 4 | simulación | ejecuta la red y se mide si el cubo acaba donde debía |

Lo que la red **no** recibe en ningún momento es la posición del cubo. Solo ve la
cámara frontal y el estado de sus propias articulaciones. Tiene que deducir dónde
está el cubo de la imagen.

### Behavior Cloning (demo 14)

La versión más simple posible: una regresión supervisada.

```
red(cámara, articulaciones) → acción
```

Nada más. Sin memoria, sin planificación a futuro. Se entrena con error cuadrático
medio sobre las acciones del experto.

### ACT (demo 15)

**Action Chunking with Transformers** es la política que LeRobot recomienda para
empezar. En vez de predecir **la siguiente** acción, predice un bloque de
``chunk_size`` acciones de una vez y las va ejecutando.

Eso importa más de lo que parece. El problema clásico del *Behavior Cloning* es
el **error acumulado**: si la red se equivoca un poco en el paso 1, llega a un
estado que el experto nunca visitó, donde vuelve a equivocarse un poco más, y así
hasta desviarse del todo. Con bloques de acciones, un pequeño error queda
absorbido dentro de una trayectoria que ya estaba planificada. ACT añade además un
transformer que atiende a las últimas observaciones y a todo el bloque.

---

## Qué tecnología se ha elegido y por qué

| Decisión | Alternativa | Por qué esta |
|---|---|---|
| **LeRobot 0.4.4** como framework de IL | escribir el entrenamiento a mano | es la herramienta del proyecto (demo 01), trae ACT ya implementada, y su formato de dataset es el estándar del SO-101 |
| **ACT** como política | Diffusion Policy, π0, SmolVLA, GR00T | es la que LeRobot documenta como "la primera que hay que probar", pesa ~80 M y cabe con holgaza en los 16 GB de la 5060 Ti |
| **Behavior Cloning propio** (demo 14) | usar una política de LeRobot | LeRobot 0.4.4 **no tiene** política `bc` en su registro. Escribirla a mano, en ~60 líneas, enseña qué es la IL sin depender de ningún framework |
| **`so101_state_machine`** como espacio del profesor | IK puro | es la configuración que ya usa el demo 01 (05) y que funciona en esta máquina: acción de 8 números + gravedad desactivada |
| **la orden IK del profesor** como acción | las articulaciones que resultan de ella | el profesor manda una pose cartesiana y el brazo tarda varios pasos en llegar. Grabar el *resultado* como acción enseña "repite lo que acabo de hacer", y esa acción casi no se mueve (menos de 1 grado por paso), así que la política aprende la identidad y se queda parada. Fue el primer intento, se midió, y se cambió |
| **grados de motor** como unidad del estado | radianes del simulador | es lo que LeIsaac exporta a LeRobot, así que el estado es intercambiable con las grabaciones de LeIsaac y con los SO-101 del Hub. **La acción no está en grados**: es la orden de 8 números del profesor, en las unidades del simulador |
| **profesor programado** | teleoperación con mando/brazo líder | un dataset de imitación necesita ser perfecto: un fallo del operador enseaña a fallar. El guion da siempre el mismo agarre y no necesita hardware conectado |
| **cubo colocado a mano** (×, y con barrido) | aleatorización de la tarea | la tarea original dispersa el cubo ±75 mm en x e y, una amplitud con la que a veces empieza fuera del alcance (radio medido 0.33 m). Una demostración fallida estropea un dataset pequeño |

Un detalle de código que conviene saber: **la política se ejecuta en CPU**
(`--inference-device cpu`). Cargar un modelo de torch con CUDA antes de arrancar
Isaac Sim deja un contexto de CUDA que después hace fallar a PhysX con
`CUDA error: an illegal memory access` al crear el entorno. La inferencia en CPU
cuesta unas decenas de milisegundos por bloque y a cambio el proceso termina.

---

## La tarea

El SO-101 está sobre una mesa con un cubo rojo delante. La demostración:

1. se sitúa sobre el cubo,
2. baja,
3. cierra la pinza,
4. sube,
5. se desplaza 10 cm a un lado,
6. baja,
7. abre la pinza,
8. sube.

El profesor lo ejecuta con una lista de waypoints relativa a la posición **real**
del cubo, que él lee después del `reset`. El guion está en
[`expert.py`](expert.py) (`PLAN`).

**Por qué el cubo se sujeta cinemáticamente.** El agarre por fricción del cubo de
3 cm no es fiable en este simulador (es una medición ya recogida en el demo 01).
Desde que la pinza se cierra hasta que se abre, la pose del cubo se escribe
directamente cada paso. Eso mantiene la demostración sobre "mover el cubo", no
sobre afinar contactos.

**Qué se considera éxito.** El criterio es **el mismo** al grabar y al evaluar:
el cubo acaba a menos de **3 cm** del punto donde las demostraciones lo dejaban
(`PLACEMENT_TOLERANCE_M` en [`expert.py`](expert.py)). Se anota también la altura
máxima que alcanza, porque un cubo lanzado lejos ya ha fallado aunque acabe en el
sitio.

> **Aviso**: esto **no** es el criterio de éxito de la tarea original.
> `LeIsaac-SO101-LiftCube-v0` declara éxito cuando el cubo queda 20 cm por encima
> de la base del robot. El plan usado aquí solo lo sube 6 cm y lo vuelve a dejar
> en la mesa, así que ese término nunca se cumple. Los demos juzgan la
> demostración igual que los demos 05 y 11 de este repositorio: por dónde acaba el
> cubo.

---

## Requisitos

El entorno es el mismo que usan los demás demos: conda `hackathon-robotekin`,
Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.0, LeIsaac 0.4.0, LeRobot 0.4.4,
RTX 5060 Ti con 16 GB. La matriz de versiones está en
[`../01_simulation_basics/README.md`](../01_simulation_basics/README.md).

Para **ejecutar** los demos (12, 13, 16) no hace falta nada nuevo.

Para **entrenar** (14, 15) LeRobot necesita cuatro paquetes que ese entorno no
tenía instalados. Se añaden **sin tocar** ninguna versión existente:

```powershell
conda activate hackathon-robotekin
python -m pip install "datasets<5.0.0,>=4.0.0" "draccus==0.10.0" termcolor "deepdiff<9.0.0,>=7.0.1" "pandas<2.4.0,>=2.2.2" "pyarrow==21.0.0"
```

Las tres últimas versiones van fijadas por un motivo concreto:
`pyarrow` más nuevo exige NumPy 2.0 y el entorno tiene **NumPy 1.26.0** clavado
por Isaac Sim; `datasets` 4.8 pide `pyarrow>=21` porque usa `pa.json_()`, que no
existe en versiones anteriores. Esa combinación es la única que encaja con
NumPy 1.26 sin romperla.

---

## Cómo ejecutar cada demo

Todos los comandos se ejecutan desde la raíz del repositorio.

### Paso obligatorio — demo 13: grabar demostraciones

```powershell
python .\demos\06_imitation_learning\13_collect_demonstrations.py --episodes 25 --headless
```

25 demostraciones tardan unos 15 minutos (≈35 s por episodio) y generan ~16 000
frames, unos 400 MB entre vídeo y parquet.

Opciones útiles:

| opción | efecto |
|---|---|
| `--episodes N` | número de demostraciones a grabar |
| `--dataset NOMBRE` | carpeta dentro de `data/` (por defecto `so101_lift_cube_demos`) |
| `--overwrite` | borra el dataset anterior antes de empezar |
| `--keep-failed` | guarda también las demostraciones fallidas |
| `--headless` | sin ventana; casi obligatorio para largas grabaciones |

Mientras trabaja imprime una línea por episodio. Un episodio que no cumple el
criterio se descarta solo: un dataset de imitación con ejemplos malos sale peor que
con menos ejemplos. Con 25 episodios lo normal es quedarse con unos 22.

**Para probar que el flujo funciona antes de la grabación larga**, con 3
demostraciones y pasos mínimos de entrenamiento:

```powershell
python .\demos\06_imitation_learning\13_collect_demonstrations.py --episodes 3 --headless --overwrite
python .\demos\06_imitation_learning\14_train_behavior_cloning.py --steps 150
python .\demos\06_imitation_learning\15_train_act.py --steps 500 --output 15_act_smoke
```

Con 3 episodios (1 920 frames) el entrenamiento se reduce a ~2 minutos para BC y
~2 minutos para ACT. Es suficiente para comprobar que todo encaja; no para que la
política aprenda nada.

### Paso opcional — demo 12: mirar el dataset

```powershell
python .\demos\06_imitation_learning\12_inspect_demonstrations.py
```

Imprime los features, cuántos frames y episodios hay, y los rangos que recorren
las articulaciones. Un rango estrecho en una articulación significa que las
demostraciones no la han movido: el síntoma clásico de un dataset grabado con el
robot aparcado.

También acepta un repo del Hub: `--dataset oumeanin/leisaac-pick-orange`.

### Paso 2 — demo 14: Behavior Cloning

```powershell
python .\demos\06_imitation_learning\14_train_behavior_cloning.py --steps 3000
```

Unas 4 horas en la 5060 Ti (el cuello de botella es decodificar vídeo, no la GPU).
Deja `outputs/14_bc/bc_policy.pt`.

> El dataset del demo 13 tiene una acción IK de 8 números, así que la red de BC
> predice 8 valores. Es la misma regresión: cambia el tamaño de la capa de salida,
> no la idea. La normalización ignora las dimensiones que no varían (la
> orientación de la pinza, que el profesor siempre manda igual), porque dividir
> por su desviación típica de cero convertiría toda la pérdida en NaN.

### Paso 2′ — demo 15: ACT

```powershell
python .\demos\06_imitation_learning\15_train_act.py --steps 20000
```

Unos 57 minutos a ~6 pasos/s. Deja checkpoints cada 2000 pasos en
`outputs/15_act/checkpoints/`; el último queda siempre en `.../last/`.

Opciones: `--batch`, `--chunk`, `--action-steps`, `--device`, `--output`,
`--no-vae` (entrena ACT sin su cuello de botella variacional; ver
[Limitaciones](#limitaciones)).

### Paso 3 — demo 16: ejecutar la política en la simulación

```powershell
# con ACT
python .\demos\06_imitation_learning\16_evaluate_policy.py --policy .\outputs\15_act\checkpoints\last\pretrained_model --type act --episodes 5 --headless

# con BC
python .\demos\06_imitation_learning\16_evaluate_policy.py --policy .\outputs\14_bc\bc_policy.pt --type bc --episodes 5 --headless
```

Opciones: `--episodes`, `--steps` (pasos por episodio, por defecto 800),
`--inference-device` (por defecto `cpu`, ver la nota de arriba), `--action-steps`
(cuántas acciones del bloque ejecutar antes de volver a preguntar),
`--headless`.

### Ver la política en la ventana del simulador

**Quita `--headless`.** Isaac Lab abre su ventana solo, sin ninguna opción
adicional (`manager_based_rl_env.py:195` renderiza cuando `has_gui()` es cierto).

Pero quitarla tal cual deja la ventana inservible por dos motivos, ambos medidos:

* **La cámara apunta al vacío.** La tarea trae `viewer.eye=(-0.4, -0.6, 0.5)` y
  `lookat=(0.9, 0, -0.3)`. Medido en esta escena, el brazo está en
  `(0.35, -0.64, 0.01)`, el cubo en `x 0.30..0.37`, y la zona de dejada llega a
  `x ≈ 0.47`. El `lookat` de la tarea apunta bastante más allá. El demo 16 usa
  unos valores medidos: `eye=(0.05, -1.05, 0.55)`, `lookat=(0.38, -0.42, 0.12)`.
* **Va a toda velocidad.** La simulación da unos 30 pasos/s con la GPU libre, así
  que 800 pasos duran medio minuto. No hay nada que mirar.

Por eso existe `--speed`:

```powershell
# un episodio, a velocidad real, mirando la ventana
python .\demos\06_imitation_learning\16_evaluate_policy.py `
    --policy .\outputs\15_act\checkpoints\last\pretrained_model `
    --type act --episodes 1 --steps 400 --speed 1
```

| `--speed` | efecto |
|---|---|
| `0` (por defecto) | a máxima velocidad; para medir, no para mirar |
| `1` | tiempo real: la tarea avanza 60 pasos por segundo |
| `0.25` | cámara lenta, para ver el agarre paso a paso |
| `0.5` | la mitad, un punto intermedio cómodo |

A velocidad real, 400 pasos son ~7 s. A `--speed 0.25`, ~27 s.

Si el ángulo no te convence, se mueve:

```powershell
--viewer-eye 0.05 -1.05 0.55 --viewer-lookat 0.38 -0.42 0.12
```

Para parar antes de tiempo, `Ctrl+C` en la consola.

> Con la ventana abierta la GPU va a más de la mitad. Si ves
> `CUDA error: an illegal memory access`, cierra otras ventanas de simulación y
> relanza; es el fallo intermitente de Kit ya documentado en el demo 01.

### Recorrido completo de una vez

```powershell
python .\demos\06_imitation_learning\13_collect_demonstrations.py --episodes 25 --headless
python .\demos\06_imitation_learning\12_inspect_demonstrations.py
python .\demos\06_imitation_learning\15_train_act.py --steps 20000
python .\demos\06_imitation_learning\16_evaluate_policy.py --policy .\outputs\15_act\checkpoints\last\pretrained_model --type act --episodes 5 --headless
```

---

## Qué sale en pantalla

**Demo 13** — una línea por episodio con el resultado, los pasos, cuánto se ha
movido el cubo y el error final, más un resumen:

```
kept 25 episode(s), discarded 0
frames: 16000, episodes: 25
total time: 897s
log: demos/06_imitation_learning/data/so101_lift_cube_demos/13_collect_demonstrations_episodes.jsonl
```

Ese `.jsonl` guarda una línea por episodio con la posición del cubo al empezar, al
acabar, el error y los segundos que tardó. Es lo que permite juzgar un dataset sin
volver a lanzar el simulador.

**Demo 12** — la ficha del dataset y una tabla de rangos:

```
dataset
  frames: 16000, episodes: 25, fps: 60
  robot type: so101_follower
  camera(s): ['observation.images.front']
  task(s): Pick the red cube up and put it down 10 cm to the side.

action and state ranges, in motor degrees
  joint             state min  state max  action min  action max
  shoulder_pan.pos       -9.4       44.2       -10.1        58.6
  ...
```

**Demo 14** — la pérdida cada 50 pasos y el resumen:

```
  step 1/3000 loss 19.5236
  step 3000/3000 loss 0.0231

trained on 25 episode(s), 16000 frames, 3000 steps, 14210s
checkpoint: demos/06_imitation_learning/outputs/14_bc/bc_policy.pt
```

**Demo 15** — la barra de progreso de `lerobot-train` y el resumen:

```
dataset: demos/06_imitation_learning/data/so101_lift_cube_demos
policy:  ACT, chunk 20, 20 actions per step, batch 8
device:  cuda
steps:   20000
...
trained in 4502s
```

**Demo 16** — una línea por episodio y el resultado final:

```
  episode 0: FAILED final_error=9.8cm max_cube_height=5.6cm steps=800 (138s)
  ...
success: 0/5 (0%)
report:  demos/06_imitation_learning/results/16_evaluation.md
```

Ese informe en `results/` es el sitio donde queda registrada la tasa de éxito, con
la fecha, la rama y el entorno como el resto de informes del repositorio.

---

## Dónde están los datos y los modelos

Todo se crea y se consume **en local**, dentro de la carpeta del demo:

| ruta | qué es | ¿en git? |
|---|---|---|
| `data/so101_lift_cube_demos/` | dataset LeRobot v2.1 (vídeo + parquet) | **no** (`.gitignore` local) |
| `outputs/14_bc/` | checkpoint de Behavior Cloning | **no** |
| `outputs/15_act/` | checkpoints de ACT | **no** |
| `results/` | informes medidos | **sí** |

Los pesos del activo de LeIsaac (los USD del SO-101) están en la caché de Hugging
Face desde los demos anteriores, así que el demo 13 no descarga nada nuevo.

### Datasets y políticas ya entrenadas

Ninguno es compatible **tal cual** con este entorno. Están anotados aquí con lo que
habría que comprobar antes de usarlos, y el porqué de que no se hayan enchufado
directamente.

| repo | contenido | por qué no se usa directo |
|---|---|---|
| [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) | dataset oficial, 60 episodios, `LeIsaac-SO101-PickOrange-v0` | otra tarea (naranjas) y otra escena; aunque el estado y la acción son compatibles (6 articulaciones), la cámara `front` está en otro sitio |
| [`oumeanin/leisaac-pick-orange-act`](https://huggingface.co/oumeanin/leisaac-pick-orange-act) | ACT entrenada sobre ese dataset (19 episodios) | la config de ACT es puramente la de por defecto y es compatible, pero las acciones aprendidas son de recoger naranjas |
| [`LightwheelAI/leisaac-pick-orange-v0`](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0) | GR00T N1.5 | no es un checkpoint de LeRobot; necesita el servidor de inferencia de GR00T |
| [`azazdeaz/teleop_so101_leisaac_lift_cube`](https://huggingface.co/datasets/azazdeaz/teleop_so101_leisaac_lift_cube) | LiftCube simulado, teleoperado | con cámaras a 256×256, no a 480×640 como la de esta tarea |

**Antes de reutilizar cualquiera hay que comprobar**: el espacio de observaciones
(dimensiones de estado, número y resolución de cámaras), el espacio de acciones
(6 articulaciones o 8 con IK), las **unidades** (radianes del simulador o grados
de motor), el modelo de robot y el entorno de simulación. Cambiar la resolución de
la cámara o las unidades invalida el checkpoint aunque el resto parezca cuadrar —
la Red Neuronal entrenada lee píxeles en una escala concreta.

---

## Resultados medidos

Medidos en la rama `prepare-demos`, entorno conda `hackathon-robotekin`,
Windows 11, Isaac Sim 5.1, RTX 5060 Ti 16 GB.

### El profesor (demo 13)

| métrica | valor |
|---|---|
| demostraciones correctas | **25 / 25** |
| error final mediano (a 3 cm) | **~1 cm** |
| desplazamiento del cubo | ~10 cm, el objetivo |
| pasos por demostración | 640 |
| tiempo por demostración | ~35 s |
| dataset completo | 25 episodios, 16 000 frames, ~1.6 GB |

### Entrenamiento (demo 15)

| métrica | valor |
|---|---|
| entrenamiento ACT | 20 000 pasos, ~57 min |
| velocidad | ~6 pasos/s con batch 8 |
| checkpoint | ~80 MB de pesos |
| error sobre el propio dataset | **0.19** (unidades de la acción) |

### Evaluación (demo 16)

**Resultado honesto: 0/5 episodios completan la tarea.**

La política aprendió a reproducir la demostración con un error de 0.19 sobre el
dataset, pero ejecutada en bucle cerrado se queda estática: el brazo se acerca al
cubo y se queda suspendido sin llegar a agarrarlo. Ver
[`results/16_evaluation.md`](results/16_evaluation.md) para el detalle.

Esto **no** es un fallo de los demos, es el resultado de intentarlo. Es el error
de imitación clásico (*compounding error*, en inglés), y es la lección más útil
del conjunto:

* **El error se acumula.** La demostración son 640 pasos encadenados. Si la
  política se equivoca un 0,2 % en cada paso, al final del episodio está en un
  estado que el profesor nunca visitó, y ahí su comportamiento no está definido.
* **La política no ve el futuro.** El profesor sabe que va a bajar el brazo en
  el paso 130; la política solo ve la foto y las articulaciones del paso en el
  que está.
* **Hay un atractor estático.** Con 25 demostraciones casi idénticas y un brazo
  que se mueve milímetros por paso, "quédate donde estás" es una solución con
  pérdida muy baja. La política la encontró.

Intentos documentados que **no** lo resolvieron, y por qué:

1. **Grabar las articulaciones resultantes** en vez de la orden del profesor:
   la acción quedaba casi igual al estado (el brazo avanza <1 % de grado por
   paso), así que la política aprendía la identidad y se quedaba parada.
2. **Quitar las pausas del plan** (el profesor era constante durante el último
   quinto de cada fase): reduce los bucles propios, pero el atractor sigue ahí.
3. **Entrenar ACT sin VAE** (`--no-vae`) y con 30 000 pasos: la política
   predice mejor sobre el dataset, pero el bucle cerrado sigue igual.

Lo que **sí** lo resolvería, y en este orden de efectividad:

| cambio | por qué |
|---|---|
| **muchas más demostraciones** (100-200, con más variedad) | el único que ataca directamente el error acumulado |
| **más variedad en la tarea** (otros puntos de dejada, más altura) | cubre estados que la política no ha visto |
| **chunk más largo** (`--chunk 40`) | una planificación más larga absorbe mejor un error pequeño |

**Tres demos que funcionan completos**, disponibles en esta misma carpeta, son
mejores que cinco a medio hacer. Aquí los cinco funcionan; la política aún no
completa la tarea, y eso está medido y documentado en lugar de escondido.


---

## Limitaciones

Esto es un campo de juego, no un sistema listo para producción.

**La política no completa la tarea todavía.** 0/5 en la evaluación medida. El
porqué está en [Resultados medidos](#resultados-medidos) y en
[`results/16_evaluation.md`](results/16_evaluation.md); el resumen es que la
política aprende la demostración pero el bucle cerrado acumula error. No es un
bug sin arreglar: es lo que hay que resolver con más demostraciones.

**Las demostraciones son casi idénticas.** El profesor ejecuta el mismo plan
siempre; solo cambian la posición del cubo (±3.5 cm), su orientación (±30°) y
nada más. El tiempo de viaje, la altura y el orden son fijos. Una política
entrenada sobre eso solo sabe hacer ese movimiento, y cualquier desviación la
saca de lo que ha visto.

**Solo un objeto y una configuración.** Un cubo rojo, una posición de partida, una
cámara. Si el cubo aparece en otro sitio del que el profesor alcanzaba, no hay
demostración que lo cubra.

**El profesor es tramposo, y conviene saberlo.** Sabe dónde está el cubo y sujeta
el cubo cinemáticamente mientras la pinza está cerrada. La política tiene que
deducir la posición del cubo de la imagen, así que parte con desventaja real: le
pedimos aprender de alguien que tenía información privilegiada.

**El criterio de éxito no es el de la tarea original.** Ver
[La tarea](#la-tarea).

**Solo verificado en simulación.** No hay brazo físico conectado. La
transferencia al hardware real exige adecuaciones que aquí no se han hecho
(calibración, control a la frecuencia del bus de motores, manejo del fallo de
comunicación).

**No se ha contrastado con datasets del Hub.** La compatibilidad se razona sobre
los parámetros publicados, no sobre una ejecución real.

**Behavior Cloning no es una política completa.** Es una red de ~60 líneas escrita
a mano para enseñar qué es la IL. No tiene memoria entre pasos ni planificación.

**El arranque de Isaac Sim es intermitente.** En esta máquina, una de cada varias
ejecuciones falla al crear el entorno con el fallo de DLL que
[`env_loader.py`](il_env.py) ya documenta. Los demos no lo reintienen ellos
mismos: si al ejecutar el 13 o el 16 aparece un acceso a memoria ilegal al
levantar Kit, hay que volver a lanzarlo.

---

## Por dónde seguir

1. **Más demostraciones, con más variedad.** Es el cambio con mejor relación
   esfuerzo/resultado, y el único que ataca directamente el fallo medido: 100
   episodios en vez de 25, con el cubo en más sitios y en más orientaciones.
2. **Más variedad en la tarea.** Variar la altura de la recogida, el punto de
   dejada, el número de objetos. La variedad de las demostraciones es el techo de
   la variedad de la política.
3. **Bloques más largos** (`--chunk 40`). Una planificación más larga absorbe
   mejor un error pequeño, y es un cambio de una línea.
4. **Probar Diffusion Policy.** LeRobot 0.4.4 la incluye
   (`lerobot.policies.diffusion`); el registro de políticas acepta `diffusion`.
5. **Teleoperación real.** El demo 13 se puede adaptar a un brazo líder o a un
   mando (`--teleop-device` de `teleop_se3_agent` en LeIsaac) para grabar
   demostraciones humanas en vez del guion. Más variadas, más caras de obtener.
6. **Transferencia al SO-101 físico.** El estado ya está en grados de motor y la
   geometría es la del brazo real. Falta el puerto serie, la calibración y la
   monitorización; lo que no está resuelto es que la política aún no completa la
   tarea en simulación.

---

## Referencias

### Documentación

- LeRobot — https://github.com/huggingface/lerobot
- LeRobot, política ACT — https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx
- LeIsaac — https://github.com/LightwheelAI/leisaac
- LeIsaac, grabar datasets de LeRobot — https://github.com/LightwheelAI/leisaac/blob/main/docs/docs/docs/features/lerobot_recorder.md
- Isaac Lab 2.x — https://isaac-sim.github.io/IsaacLab/
- LeRobot Demo SO-101 — https://github.com/huggingface/lerobot/blob/main/examples/10_using_so101.md

### Papers

- **ACT** — Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* — https://arxiv.org/abs/2304.13705
- **Diffusion Policy** — Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* — https://arxiv.org/abs/2303.04137
- **Behavior Cloning** — Bain & Sammut — https://link.springer.com/chapter/10.1007/978-3-642-17407-8_5

### Datasets y modelos

- https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange
- https://huggingface.co/oumeanin/leisaac-pick-orange-act
- https://huggingface.co/datasets/azazdeaz/teleop_so101_leisaac_lift_cube

### Código del que salen los números de este README

| qué | dónde |
|---|---|
| versiones y por qué | [`../01_simulation_basics/README.md`](../01_simulation_basics/README.md) |
| geometría del agarre, medida | [`../01_simulation_basics/05_cube_pick_and_place.py`](../01_simulation_basics/05_cube_pick_and_place.py) |
| alcance del SO-101 (0.33 m) | [`../04_perception/`](../04_perception/) |
| criterio de "colocar el cubo" | [`../05_planner_and_executor/`](../05_planner_and_executor/) |
| el profesor | [`expert.py`](expert.py) |
| las unidades y los features | [`il_dataset.py`](il_dataset.py) |
| los dos espacios de control | [`il_env.py`](il_env.py) |

---

## Problemas habituales

| síntoma | causa y solución |
|---|---|
| `CUDA error: an illegal memory access` al crear el entorno | hay un proceso de Isaac Sim zombie ocupando la GPU. Ciérralo con el Administrador de tareas, o revisa con `Get-CimInstance Win32_Process` si quedó algún `python.exe` colgado. También lo provoca cargar una política de torch en CUDA antes de arrancar el simulador — ejecuta con `--inference-device cpu` (que es el valor por defecto) |
| `value cannot be converted to type uint8 without overflow` | la imagen se está pasando como `uint8` y el normalizador adopta ese tipo para sus estadísticas. El preprocesador quiere `float32` en [0, 1]; eso ya lo hace `il_dataset.image_to_policy_input` |
| `pyarrow requires NumPy 2.0 or newer` | se ha instalado `pyarrow` demasiado nuevo. Fija `pyarrow==21.0.0`, la última que funciona con NumPy 1.26 |
| `module 'pyarrow' has no attribute 'json_'` | `datasets` 4.8 lo pide. Se resuelve con `pyarrow>=21` |
| `A prominent term order...` (el dataset no carga) | el dataset se quedó sin `finalize()` porque el script se interrumpió. Vuelve a grabar con `--overwrite` |
| demo 16: `RESULT: FAILED` con `steps=640` | la política se ha quedado parada. Es la limitación de entrenar con pocas demostraciones: graba más y vuelve a entrenar |
| demo 15: `Output directory ... already exists` | LeRobot no permite reutilizar una carpeta de salida. Usa `--output OTRO_NOMBRE` |

---

## Estructura de la carpeta

```
06_imitation_learning/
├── README.md                            ← este archivo
├── .gitignore                           ← data/, outputs/ y out/ fuera de git
├── il_env.py                            ← entornos LeIsaac y conversión de acciones
├── il_dataset.py                        ← features, unidades y I/O del dataset
├── expert.py                            ← el profesor programado (waypoints)
├── 12_inspect_demonstrations.py         ← demo 12: ver qué hay dentro de un dataset
├── 13_collect_demonstrations.py         ← demo 13: grabar demostraciones (obligatorio)
├── 14_train_behavior_cloning.py         ← demo 14: Behavior Cloning en PyTorch puro
├── 15_train_act.py                      ← demo 15: ACT con el pipeline de LeRobot
├── 16_evaluate_policy.py                ← demo 16: ejecutar y medir en simulación
├── data/                                ← datasets (git-ignored)
├── outputs/                             ← checkpoints (git-ignored)
└── results/                             ← informes medidos (sí en git)
```
