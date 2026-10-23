# Demos 12–16 — Aprendizaje por imitación en el SO-101 simulado

Este conjunto de demos entrena al brazo SO-101 **a partir de demostraciones**, no a
fuerza de prueba y error con recompensas. El resultado, medido, es este:

> Una política ACT que **recoge el cubo, lo levanta 6 cm, lo transporta sujeto por
> fricción y lo deja 10 cm más allá**, en posiciones del cubo que no vio al entrenar:
> **30 de 30 posiciones distintas** (40/40 contando una repetición) **dentro de la región
> de entrenamiento y 8 de 10 fuera de ella** (reloj + cubo, ver más abajo).
>
> Pero **no es una política que "mira la cámara y decide sola"**, y este README dice
> exactamente qué es y qué no es antes de dar ningún número.

---

## Tabla de contenido

1. [Qué es esta política y qué no es](#qué-es-esta-política-y-qué-no-es)
2. [Ruta rápida para el día de la demo](#ruta-rápida-para-el-día-de-la-demo)
3. [Reproducción completa](#reproducción-completa)
4. [Cómo se juzga el éxito](#cómo-se-juzga-el-éxito)
5. [Resultados medidos](#resultados-medidos)
6. [Por qué fallaron las políticas con cámara](#por-qué-fallaron-las-políticas-con-cámara)
7. [Limitaciones](#limitaciones)
8. [Requisitos](#requisitos)
9. [Datos y modelos](#datos-y-modelos)
10. [Ver la política en la ventana](#ver-la-política-en-la-ventana)
11. [Problemas habituales](#problemas-habituales)
12. [Por dónde seguir](#por-dónde-seguir)
13. [Referencias](#referencias)
14. [Estructura de la carpeta](#estructura-de-la-carpeta)

---

## Qué es esta política y qué no es

La política que funciona es un **ACT de LeRobot** con exactamente **dos fuentes de
información**:

| entrada | qué es | ¿privilegiada? |
|---|---|---|
| **reloj del plan** | en qué paso de los 640 del plan va el episodio, de −1 a 1 | **Sí.** Un robot real no sabe en qué punto de un plan está |
| **posición del cubo** | el centroide en píxeles del cubo rojo, medido **una vez** al empezar el episodio con un detector de color (`cube_detect.py`) | No: sale de la imagen, no del estado del simulador |

**No entra ninguna imagen ni ninguna articulación en la red.** Eso es deliberado y es
la lección central del demo (ver [Por qué fallaron las políticas con
cámara](#por-qué-fallaron-las-políticas-con-cámara)): con la imagen y el estado
articular delante, la política aprendía a copiar la pose de su propio brazo.

Lo que sí es:

* imitación de verdad: la red aprende, de 48 demostraciones, **cómo cambia la
  trayectoria del brazo según dónde está el cubo**, y la interpola a posiciones y
  orientaciones nuevas dentro de la región de entrenamiento (a ≈ 5 mm del experto en el
  punto de agarre, medido sin simulador); fuera de esa región empieza a fallar;
* agarre **físico**: el cubo se sujeta por fricción, sin sujeción cinemática, y el
  éxito exige levantarlo y transportarlo realmente.

Lo que no es:

* **No es lazo cerrado.** Percibe el cubo una vez y reproduce una trayectoria. Si el
  cubo se mueve a mitad de episodio, o el agarre falla, no reacciona.
* **No percibe con una red.** La percepción es un detector de color clásico.
* **El reloj es una ayuda.** Conviene decirlo así en la presentación: es imitación
  con el calendario del plan entregado, no imitación desde observación pura.
* **Solo simulación**, un cubo, una cámara, una región de 9 × 9 cm de la mesa y ±30°
  de giro.

---

## Ruta rápida para el día de la demo

Todos los comandos, desde la raíz del repositorio, en PowerShell. Se supone el
entorno de [Requisitos](#requisitos) y que el checkpoint ya está entrenado
(`outputs/15_act_clock/`).

```powershell
conda activate hackathon-robotekin
$env:HF_HUB_OFFLINE = "1"      # sin red: evita el error SSL de Hugging Face (ver Problemas habituales)
```

**1. Que se vea el robot recogiendo el cubo** (un episodio, en la ventana, a media velocidad; ≈ 1 min):

```powershell
python .\demos\06_imitation_learning\16_evaluate_policy.py `
    --policy .\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model `
    --episodes 1 --speed 0.5
```

**2. Los números en directo** (10 posiciones nuevas, sin ventana; ≈ 5 min). Deja un informe
nuevo en `results/runs/` y no toca los anteriores:

```powershell
python .\demos\06_imitation_learning\16_evaluate_policy.py `
    --policy .\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model `
    --episodes 10 --headless
```

**3. Para compararlo con el guion del profesor** (que sabe dónde está el cubo y **no es
una política aprendida**; sale etiquetado así en su informe):

```powershell
python .\demos\06_imitation_learning\16_evaluate_policy.py --type scripted --episodes 1 --speed 0.5
```

Antes de empezar: cierra cualquier otro proceso de Isaac Sim (`Get-Process python`),
porque dos simuladores a la vez en esta GPU provocan el fallo intermitente de la
sección [Problemas habituales](#problemas-habituales). Si al arrancar sale
`CUDA error: an illegal memory access`, vuelve a lanzar el mismo comando.

### Si hay que llevar el modelo a otra máquina

El modelo es una carpeta de **≈ 90 MB** (`pretrained_model`; 84 MB comprimido). La carpeta `outputs/` entera pesa
GBs porque guarda el estado del optimizador de cada checkpoint; para llevarse el modelo basta esto:

```powershell
Compress-Archive -Path .\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model\* `
    -DestinationPath .\act_clock_listo.zip
```

En la otra máquina: descomprimir en cualquier carpeta y pasarla a `--policy`. Lleva dentro
`il_inputs.json`, que dice qué significan sus entradas, y necesita el mismo entorno
conda y la caché de activos de LeIsaac. Los checkpoints **no están en git** (`.gitignore`).

### Si hay que reentrenarlo en el sitio

≈ 15 min de entrenamiento con los datos ya grabados (los 48 episodios ocupan 105 MB; tampoco están en git):

```powershell
python .\demos\06_imitation_learning\tools\21_build_clock_cube_dataset.py
python .\demos\06_imitation_learning\15_train_act.py --steps 10000 --batch 64 --no-vae --lr 1e-4
```

---

## Reproducción completa

Cuatro pasos, ≈ 50 min de reloj en la RTX 5060 Ti. Cada uno escribe en una carpeta
propia y **ninguno borra nada existente**.

| paso | comando | tiempo | qué hace |
|---|---|---|---|
| 1. grabar | `13_collect_demonstrations.py --episodes 48 --headless` | ≈ 27 min | 48 demostraciones del profesor con **agarre por fricción**; solo se guardan las que pasan el criterio físico |
| 2. preparar | `tools/21_build_clock_cube_dataset.py` | segundos | copia sin vídeo con la entrada reloj + cubo |
| 3. entrenar | `15_train_act.py --steps 10000 --batch 64 --no-vae --lr 1e-4` | ≈ 14 min | ACT sobre esa entrada |
| 4. evaluar | `16_evaluate_policy.py --policy ... --episodes 10 --headless` | ≈ 5 min | 10 posiciones nuevas, juicio físico |

```powershell
python .\demos\06_imitation_learning\13_collect_demonstrations.py --episodes 48 --headless
python .\demos\06_imitation_learning\tools\21_build_clock_cube_dataset.py
python .\demos\06_imitation_learning\15_train_act.py --steps 10000 --batch 64 --no-vae --lr 1e-4
python .\demos\06_imitation_learning\16_evaluate_policy.py `
    --policy .\demos\06_imitation_learning\outputs\15_act_clock\checkpoints\last\pretrained_model `
    --episodes 10 --headless
```

### Grabar el SO-101 real con dos cámaras

`17_record_real_so101.py` teleopera el follower con un leader SO-101 y guarda
observaciones, acciones, las dos cámaras y la instrucción en un dataset LeRobot.
Pulsa **Enter** una vez para activar la teleoperación y colocar los brazos; vuelve
a pulsarlo para empezar a grabar. Durante un episodio, **q** lo guarda y **r** lo
descarta; ambas teclas abren el periodo de reposicionamiento. Pulsa **Enter** para
empezar el siguiente episodio y **e** para finalizar y subir los episodios guardados.
El reposicionamiento dura 10 segundos por defecto; cámbialo con `--reset-seconds`.
También puedes usar **Ctrl+C** para guardar el episodio en curso y finalizar.
Autentícate antes con `hf auth login`.

```bash
python demos/06_imitation_learning/17_record_real_so101.py
```

El script usa los puertos, IDs calibrados, cámaras, repositorio y tarea de este
equipo por defecto. Añade `--reset-seconds 20` para ampliar la ventana de
reposicionamiento, `--repo-id USUARIO/dataset` para cambiar el destino, o `--private`
para crear el dataset privado. Guarda el fotograma completo a 1280 × 720 y 10 FPS
por defecto, mientras mantiene el control del robot a 30 Hz. El FPS de grabación se
cambia con `--fps`; ambos valores tienen que ser compatibles con las dos cámaras.
Si las cámaras admiten otra resolución, indícala con `--camera-width` y
`--camera-height`, por ejemplo `--camera-width 1920 --camera-height 1080`. Las claves
de imagen son `observation.images.front` y
`observation.images.wrist`; el estado y la acción son las articulaciones SO-101
según la configuración de LeRobot. Este dataset de hardware es independiente del
dataset del simulador que usan los comandos de entrenamiento de arriba.

### Comprobar antes de gastar tiempo en el simulador

`tools/22_validate_offline.py` reproduce, **sin simulador**, la orden que daría la política
en episodios que no ha visto, y la compara con la del experto. Como la política no lee el
brazo, esa reproducción coincide con lo que haría en el simulador. Para usarla, entrena
sin los últimos 8 episodios:

```powershell
python .\demos\06_imitation_learning\15_train_act.py --episodes 0-39 --steps 10000 --batch 64 --no-vae --lr 1e-4 --output 15_act_clock_val   # se regenera con este comando; no se conserva en disco
python .\demos\06_imitation_learning\tools\22_validate_offline.py `
    --policy .\demos\06_imitation_learning\outputs\15_act_clock_val\checkpoints\last\pretrained_model --episodes 40-47
```

El número que importa es el error de posición **mientras la pinza se cierra**: el agarre
tolera unos ±8–10 mm alrededor del punto al que apunta el experto
([diagnóstico, apartado 6](results/diagnosis.md)). Se midió **3,4 mm de media y 5,0 mm de peor caso en xy** con la política entrenada sin esos 8 episodios.

### Probar que todo encaja con 3 episodios (≈ 6 min)

Todos los nombres son de prueba, así que **no toca el dataset principal ni sus modelos**:

```powershell
python .\demos\06_imitation_learning\13_collect_demonstrations.py --episodes 3 --headless --dataset so101_smoke
python .\demos\06_imitation_learning\tools\21_build_clock_cube_dataset.py --source so101_smoke --output so101_smoke_clock
python .\demos\06_imitation_learning\15_train_act.py --dataset so101_smoke_clock --steps 300 --batch 16 --no-vae --lr 1e-4 --output smoke
python .\demos\06_imitation_learning\16_evaluate_policy.py `
    --policy .\demos\06_imitation_learning\outputs\smoke\checkpoints\last\pretrained_model --episodes 1 --headless
```

Con 300 pasos y 3 demostraciones la política **no** recoge el cubo; esto solo comprueba que los
scripts encajan. (`--overwrite` en el paso 1 renombra el dataset existente a
`<nombre>.replaced-<fecha>` en vez de borrarlo, y cada nombre de salida de entrenamiento y de
informe que ya existe se rechaza.)

### Opciones de cada script

| script | opciones útiles |
|---|---|
| `13_collect_demonstrations.py` | `--episodes N`, `--dataset NOMBRE`, `--first-spot K`, `--keep-failed`, `--pin` (vuelve a la sujeción cinemática de las primeras grabaciones), `--overwrite` (aparta, no borra), `--headless` |
| `15_train_act.py` | `--dataset`, `--steps`, `--batch`, `--lr`, `--seed`, `--episodes 0-39`, `--no-vae`, `--chunk`, `--action-steps`, `--output` (si existe, se rechaza) |
| `16_evaluate_policy.py` | `--policy`, `--type act\|bc\|scripted`, `--episodes`, `--first-episode`, `--spots heldout\|outside\|train`, `--seed`, `--report-name`, `--speed`, `--trace`, `--scripted-bias-mm X Y`, `--stall-seconds`, `--headless` |

### Otros pasos del recorrido original

* **Demo 12** — `12_inspect_demonstrations.py` imprime los *features*, los rangos de las
  articulaciones y cuánto se mueve cada dimensión de la acción; sirve para ver si un dataset
  tiene movimiento.
* **Demo 14** — `14_train_behavior_cloning.py` es una regresión supervisada en ~60 líneas de
  PyTorch (cámara + estado + fase). **No se ha vuelto a validar**: usa la imagen y el estado del
  brazo, el mismo diseño que falló con ACT, y tarda horas. Queda como material didáctico.

---

## Cómo se juzga el éxito

Se fijó **antes** de evaluar (`il_metrics.py`) y no depende de nada que el simulador pueda
falsear. Un episodio es correcto si **las tres** cosas ocurren:

1. **Levanta**: el cubo sube al menos **4 cm** sobre la mesa (el plan lo sube 6 cm).
2. **Lo sujeta**: se mantiene a más de 3 cm de la mesa **durante al menos 1 s** de tiempo
   simulado, y durante ese tramo se queda a menos de 3 cm del punto entre los dedos. Un cubo
   lanzado está en el aire una fracción de segundo y lejos de la mano.
3. **Lo deja**: acaba a menos de **3 cm** del objetivo (10 cm al lado de donde estaba), sobre la
   mesa y quieto.

El informe da la causa de cada fallo (`never_lifted`, `dropped_early`, `not_held`,
`misplaced`, `not_at_rest`) en vez de un cero.

Lo que **no** se afirma con esto:

* Es agarre **simulado**: contacto por fricción de las mordazas del SO-101 de LeIsaac con la
  fricción por defecto. No dice nada sobre un robot real.
* El profesor graba con **la posición real del cubo**: sus 48 demostraciones son
  correctas porque sabe dónde está. Que el profesor acierte no es que acierte la política.
* El criterio antiguo ("el cubo acaba a 3 cm del punto") **no distingue un agarre de un
  empujón**, y por eso ya no se usa. Tampoco es el criterio de la tarea original
  (`LeIsaac-SO101-LiftCube-v0`: cubo 20 cm sobre la base), que este plan nunca cumple.

Las evaluaciones usan **posiciones del cubo que ninguna demostración usó**
(`il_env.sample_eval_spot`; en el informe aparece la distancia a la demostración más
cercana). Las posiciones y la política son deterministas; la física no es bit a bit
reproducible (el cubo se asienta con ±2 mm de diferencia entre ejecuciones), por eso se
repitió una ejecución completa y se dan las dos.

---

## Resultados medidos

Rama `prepare-demos`, entorno conda `hackathon-robotekin`, Windows 11, Isaac Sim 5.1,
RTX 5060 Ti 16 GB. Cada fila es un informe completo en [`results/runs/`](results/runs/);
[`results/SUMMARY.md`](results/SUMMARY.md) los reúne todos y se regenera con
`tools/23_summarize_results.py`.

| política | entradas | cubos | éxitos |
|---|---|---|---|
| **ACT reloj + cubo (la del demo)** | reloj del plan (privilegiado) + posición del cubo; sin imagen ni estado | 10 posiciones nuevas | **10/10** |
| | | otras 20 posiciones nuevas | **20/20** |
| | | las mismas 10, repetido | **10/10** |
| | | **fuera de la región de entrenamiento** (anillo de 1 cm) | **8/10** |
| ACT reloj + cubo, entrenada con 40 episodios | ídem | 10 posiciones nuevas | 10/10 |
| ACT reloj + cubo, comprimida y descomprimida en otra carpeta | ídem | 2 posiciones nuevas | 2/2 |
| *guion del profesor (no es una política aprendida)* | *conoce la posición real del cubo* | *10 nuevas; 10 de fuera de la región* | *10/10; 10/10* |
| ACT imagen + estado articular | cámara y articulaciones | 5 posiciones nuevas | **0/5** (el cubo ni se toca) |
| ACT imagen + estado + centroide del cubo | ídem + detector | 5 posiciones nuevas | **0/5** (ídem) |
| ACT imagen + estado + fase del plan | ídem + fase (privilegiada) | 5 posiciones nuevas | **0/5** (1 levantó el cubo y lo dejó a 6,9 cm) |

Todo con el criterio de la sección anterior: elevación de 5,9–6,5 cm, transporte sujeto de 156–169
pasos y error final ≤ 1,1 cm en cada acierto de la política del demo.

**Interpolar no es extrapolar.** Las 30 posiciones "nuevas" están **intercaladas** con las 48
demostraciones: la más cercana está a 4,6–7,2 mm. Eso prueba que la política interpola la
trayectoria entre cubos vistos, no que funcione en cualquier parte. Fuera de la caja de
entrenamiento falló 2 de 10 y los dos fallos son las posiciones más lejanas (8–9 mm fuera de
la caja, 20–25 mm de la demostración más cercana); el guion del profesor acierta las 10. Una
primera versión de esa prueba (`act_clock_final_wide_10ep`, 10/10) solo sacó 2 de 10
posiciones fuera de la caja, por menos de 3 mm, y no mide extrapolación; se conserva en
`results/runs/` por completitud.

El registro curado y comentado es [`results/16_evaluation.md`](results/16_evaluation.md).

### El profesor (demo 13)

| métrica | valor |
|---|---|
| demostraciones físicas correctas | **48 / 48** (ninguna descartada) |
| elevación del cubo | 5,9 – 6,7 cm |
| transporte sujeto | 161 – 164 pasos (≈ 2,7 s) |
| error final de colocación | ≤ 0,95 cm |
| tiempo por demostración | ≈ 33 s |
| dataset | 48 episodios, 30 720 fotogramas, 105 MB (con vídeo) |

### Entrenamiento (demo 15, reloj + cubo)

| métrica | valor |
|---|---|
| pasos | 10 000, lote 64, `--no-vae`, tasa de aprendizaje 1e-4 |
| tiempo | ≈ 14 min (GPU compartida con otros procesos) |
| pérdida final | 0,022 (L1 normalizada; sigue bajando lentamente) |
| tamaño del modelo | ≈ 90 MB |
| tamaño de `outputs/15_act_clock/` | 2,8 GB al entrenar (estado del optimizador de cada checkpoint); tras la limpieza solo se conserva `pretrained_model`, ≈ 90 MB |

### Sin simulador (`tools/22_validate_offline.py`)

Política entrenada con los episodios 0–39, reproducida sobre los 8 que no vio:
error de posición del comando mientras se cierra la pinza, **3,4 mm de media y 5,0 mm
como peor caso en xy; 0,3 mm en z**. El agarre tolera ±8–10 mm.
Informe: [`results/offline/act_clock_val_10k_heldout_40-47.json`](results/offline/act_clock_val_10k_heldout_40-47.json).

---

## Por qué fallaron las políticas con cámara

Resumen; todo el detalle, con las mediciones, en [`results/diagnosis.md`](results/diagnosis.md).

Las tres variantes con cámara + estado articular que se probaron (sin entrada extra, con la
fase del plan y con el centroide del cubo) daban **0/5** en la evaluación original. Con las
mediciones de esta revisión:

1. **El modelo estaba bien entrenado.** Sobre las demostraciones reproduce la orden del
   experto con 1–5 mm de error en cada fase, también al bajar (z predicha 0,141 m frente a
   0,144 m). La pérdida 0,055 era un buen ajuste. Y en una reproducción del episodio 0 en el
   simulador recibe lo mismo que vio al entrenar (imágenes, estados, normalización y unidades).
2. **Copiaba su propio brazo.** Dándole la misma imagen con el estado articular de *otro*
   episodio, la y que predice se mueve 20–45 mm cuando el experto no se movería nada. En
   lazo cerrado, cualquier desvío del brazo se convierte en una orden igual de desviada.
   Es *causal confusion*; no lo arregla entrenar más ni usar bloques más largos, sino
   **quitar el estado del brazo** de la política. Por eso la política que funciona no lo tiene.
3. **Había un fallo en el propio demo:** `il_env.set_cube_randomization` no congelaba nada
   (LeIsaac registra la aleatorización como atributos, no como campos de dataclass) y **la
   cámara se movía ±5 mm en cada episodio**. Con eso el cubo en píxeles predecía su posición
   con 5 mm de error; congelada, con 0,5 mm. Todas las grabaciones anteriores llevan ese
   temblor.
4. **La evaluación lo ocultaba:** reutilizaba las posiciones de las demostraciones, no
   reiniciaba la cola de acciones de ACT entre episodios, juzgaba solo la posición final del
   cubo, el primer fotograma de cada episodio era de la escena anterior, y el informe se
   escribía siempre en el mismo archivo.
5. **El agarre por fricción sí funcionaba**, con el punto de agarre bien elegido. El README
   anterior decía que no era fiable y el profesor fijaba el cubo a la pinza. La ventana de
   agarre mide unos 25 × 15 mm y el experto original apuntaba a 5 mm de su borde (aun así acertaba con
   posición exacta: 10/10; el desplazamiento solo da margen a una política que estima el cubo).

Sobre el historial de observaciones: **ACT recibe una sola observación por predicción**
(`n_obs_steps = 1`; LeRobot 0.4.4 rechaza cualquier valor mayor, `configuration_act.py:149`). El
README anterior decía que el transformer "atiende a las últimas observaciones"; es falso. Atiende a
la imagen y el estado actuales y al bloque de acciones que produce. π0 sí acepta historial, pero no
arranca en esta máquina (falta el fork de `siglip`).

---

## Limitaciones

* **No es una política de lazo cerrado.** Percibe una vez y reproduce una trayectoria
  aprendida; no reacciona al brazo ni a que el cubo se mueva.
* **El reloj del plan es privilegiado.** Sin él, la política no sabe cuándo bajar, cerrar o
  soltar.
* **Percepción clásica y una sola vez:** un detector de color sobre una escena de poco
  contraste. Con otro objeto, otro color o más desorden habría que rehacerla.
* **Una tarea fija:** un cubo, una cámara, una región de 9 × 9 cm, ±30° de giro, y el objetivo
  siempre 10 cm en x. **Interpola, no extrapola**: dentro de la región, 30/30; en el anillo de 1 cm
  alrededor, 8/10, y fallan las posiciones más lejanas. Más allá, sin medir.
* **Margen de agarre estrecho** (≈ ±8–10 mm) y calibrado a mano con el punto de agarre del experto
  (`expert.GRASP_BIAS_M`); un cambio de geometría del cubo o de la pinza lo invalida.
* **Solo simulación**, con la fricción por defecto. No hay brazo físico conectado.
* **Las físicas no son bit a bit reproducibles** entre ejecuciones.
* **El arranque de Isaac Sim es intermitente** en esta máquina y los scripts no lo reintentan. Una vez, Kit
  se bloqueó en mitad de una evaluación de 20 episodios (CPU ocupada, sin mensajes); desde entonces
  `16_evaluate_policy.py` guarda cada episodio según termina (`.partial.jsonl`) y, si no avanza ningún paso
  en `--stall-seconds` (300 por defecto), escribe lo terminado y sale con código 3. Esa salvaguarda **no se ha
  llegado a disparar** y está sin probar.
* **Las pruebas anteriores ya no están en disco.** Los datasets (`so101_lift_cube_demos`,
  `_demos_v2`, `_phase`, `_cube_detect`; cámara moviéndose y cubo fijado cinemáticamente) y los
  checkpoints ACT con cámara (`15_act`, `15_act_v2`, `15_act_phase*`, `15_act_cube_detect`) se
  borraron el 2026-10-20 al limpiar la carpeta. Sus cifras siguen en `results/`, pero ya no se
  pueden reevaluar, y `tools/17`, `18` y `20`, que leían esos datasets, no se pueden ejecutar tal cual.
* **No se ha validado con datasets del Hub.** La compatibilidad se razona sobre los parámetros
  publicados, no sobre una ejecución.

---

## Requisitos

El entorno es el de los demás demos: conda `hackathon-robotekin`, Python 3.11, Isaac Sim 5.1,
Isaac Lab 2.3.0, LeIsaac 0.4.0, LeRobot 0.4.4, RTX 5060 Ti con 16 GB. La matriz de versiones
está en [`../01_simulation_basics/README.md`](../01_simulation_basics/README.md).

Para **ejecutar** los demos 12, 13 y 16 no hace falta nada nuevo. Para **entrenar** (14, 15)
LeRobot necesita cuatro paquetes que ese entorno no tenía, sin tocar ninguna versión existente:

```powershell
conda activate hackathon-robotekin
python -m pip install "datasets<5.0.0,>=4.0.0" "draccus==0.10.0" termcolor "deepdiff<9.0.0,>=7.0.1" "pandas<2.4.0,>=2.2.2" "pyarrow==21.0.0"
```

`pyarrow` más nuevo exige NumPy 2.0 y el entorno tiene NumPy 1.26 fijado por Isaac Sim;
`datasets` 4.8 pide `pyarrow>=21`. Esa combinación es la única que encaja con NumPy 1.26.

**Sin red** (la wifi de un hackathon): los activos de LeIsaac ya están en la caché de Hugging
Face de los demos anteriores, pero `prepare_runtime` consulta igualmente a Hugging Face y falla
con un error SSL si la red es mala. Con la caché llena, `$env:HF_HUB_OFFLINE = "1"` lo evita. Se
usó así en todas las medidas de este README.

**La inferencia va en CPU** (`--inference-device cpu`): cargar un modelo de torch con CUDA antes
de arrancar Isaac Sim deja un contexto que hace fallar a PhysX con `illegal memory access`. Cuesta
unas decenas de milisegundos por bloque y a cambio el proceso termina.

---

## Datos y modelos

Todo se crea y se consume en local, dentro de esta carpeta. **Ningún script borra datasets ni
checkpoints.**

| ruta | qué es | ¿en git? |
|---|---|---|
| `data/so101_lift_cube_physical/` | las 48 demostraciones con vídeo (dataset LeRobot v3), cubo sujeto por fricción | no (`.gitignore`) |
| `data/so101_lift_cube_clock_cube/` | copia sin vídeo con la entrada reloj + cubo (1 MB) | no |
| `outputs/15_act_clock/checkpoints/last/pretrained_model/` | **el modelo del demo** (≈ 90 MB); es lo único que queda de `outputs/` | no |
| `results/runs/`, `results/offline/` | un informe por ejecución, sin sobrescribir | **sí** |
| `results/diagnosis.md`, `results/16_evaluation.md`, `results/SUMMARY.md` | diagnóstico, registro y resumen | **sí** |

### Datasets y políticas ya entrenadas del Hub

Ninguno es compatible **tal cual** con este entorno; antes de reutilizar uno hay que comprobar el
espacio de observaciones, el espacio de acciones (6 articulaciones o 8 con IK), las **unidades**
(radianes del simulador o grados de motor), el modelo de robot y el entorno de simulación.

| repo | contenido | por qué no se usa directo |
|---|---|---|
| [`LightwheelAI/leisaac-pick-orange`](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange) | dataset oficial, 60 episodios | otra tarea y otra escena; la cámara `front` está en otro sitio |
| [`oumeanin/leisaac-pick-orange-act`](https://huggingface.co/oumeanin/leisaac-pick-orange-act) | ACT sobre ese dataset | las acciones aprendidas son de recoger naranjas |
| [`LightwheelAI/leisaac-pick-orange-v0`](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0) | GR00T N1.5 | no es un checkpoint de LeRobot |
| [`azazdeaz/teleop_so101_leisaac_lift_cube`](https://huggingface.co/datasets/azazdeaz/teleop_so101_leisaac_lift_cube) | LiftCube teleoperado | cámaras a 256×256, no a 480×640 |

### Decisiones técnicas

| decisión | alternativa | por qué esta |
|---|---|---|
| **LeRobot 0.4.4** | entrenar a mano | es la herramienta del proyecto; trae ACT y el formato estándar del SO-101 |
| **ACT** | Diffusion Policy, π0, SmolVLA | la que LeRobot recomienda primero; cabe en los 16 GB |
| **la orden IK del profesor** como acción | las articulaciones que resultan | grabar el *resultado* enseña "repite lo que acabo de hacer" y la política se queda quieta |
| **grados de motor** para el estado | radianes | es lo que LeIsaac exporta a LeRobot |
| **profesor programado** | teleoperación | no necesita hardware y da siempre el mismo agarre |
| **cubo colocado a mano en una región de 9 × 9 cm** | la aleatorización original (±75 mm) | la original deja a veces el cubo fuera del alcance (0,33 m) |

---

## Ver la política en la ventana

**Quita `--headless`.** Isaac Lab abre su ventana solo. Dos cosas que ya están resueltas, ambas medidas:

* **La cámara de la tarea apunta al vacío** (`lookat=(0.9, 0, -0.3)`); el demo usa unos valores
  medidos: `eye=(0.05, -1.05, 0.55)`, `lookat=(0.38, -0.42, 0.12)`. Si no te convence el ángulo:
  `--viewer-eye 0.05 -1.05 0.55 --viewer-lookat 0.38 -0.42 0.12`.
* **Va a toda velocidad** (≈ 30 pasos/s): 700 pasos pasan en medio minuto. Por eso existe `--speed`:

| `--speed` | efecto |
|---|---|
| `0` (por defecto) | a máxima velocidad; para medir |
| `1` | tiempo real (60 pasos por segundo) |
| `0.5` | la mitad: cómodo para presentar |
| `0.25` | cámara lenta, para ver el agarre paso a paso |

Para parar antes de tiempo, `Ctrl+C` en la consola.

---

## Problemas habituales

| síntoma | causa y solución |
|---|---|
| `SSLError ... huggingface.co` al arrancar | `prepare_runtime` consulta a Hugging Face. Con la caché de activos llena: `$env:HF_HUB_OFFLINE = "1"` |
| `CUDA error: an illegal memory access` al crear el entorno | fallo intermitente de Kit, mucho más frecuente con otro Isaac Sim en la GPU (durante estas medidas hubo veces con 2–3 intentos seguidos). Cierra los demás (`Get-Process python`) y **vuelve a lanzar el mismo comando** |
| decenas de líneas `[Error] ... DLL load failed ... demasiado largo` al arrancar | ruido de extensiones de Kit que no se usan; es inofensivo |
| `ERROR: ... already exists` al evaluar o entrenar | por diseño: los informes y las carpetas de salida no se sobrescriben. Usa `--report-name` / `--output` con otro nombre |
| `ERROR: ... already exists` al grabar | pasa `--dataset OTRO` o `--overwrite` (el anterior se renombra a `.replaced-<fecha>`, no se borra) |
| `KeyError: 'observation.state'` entrenando ACT sin estado | error de LeRobot 0.4.4 (lee `.device` de esa clave aunque no haya estado). El demo lo evita conservando la columna en el dataset y declarando solo la entrada de entorno; si montas otro dataset, copia ese patrón (`tools/21`) |
| `value cannot be converted to type uint8 without overflow` | la imagen se pasa como `uint8`; el preprocesador quiere `float32` en [0, 1] (`il_dataset.image_to_policy_input`) |
| `pyarrow requires NumPy 2.0 or newer` | `pyarrow` demasiado nuevo; fija `pyarrow==21.0.0` |
| `Output directory ... already exists` al entrenar | LeRobot no reutiliza una carpeta de salida. Usa `--output OTRO` |
| el informe dice `(con cambios sin confirmar)` | el árbol de git tenía cambios al evaluar; el informe lo declara a propósito |
| `close()` del simulador se cuelga | los scripts salen con `os._exit` tras escribir todo en disco, justo para esto |

---

## Por dónde seguir

1. **Cerrar el lazo sobre el cubo**: volver a percibirlo cada pocos pasos y reaccionar. Exige demostraciones con
   perturbaciones y un profesor reactivo, no un plan por tiempo.
2. **Quitar el reloj**: un profesor cuyas transiciones dependan de lo observado (brazo llegado, pinza
   cerrada) hace que la acción sea función de la observación y no del calendario.
3. **Percepción aprendida**: con la cámara quieta el detector clásico da 0,5 mm de error, así que una red
   tiene un listón claro que igualar.
4. **Más variedad en la tarea** (objetivo, altura, número de objetos): la variedad de las demostraciones es el techo de la de la política.
5. **Otra política con historial** (π0 con `n_obs_steps=2`) si se consigue el fork de `siglip`.
6. **Teleoperación real** para grabar demostraciones humanas, más variadas y más caras.
7. **Transferencia al SO-101 físico**: puerto serie, calibración y monitorización.

---

## Referencias

- LeRobot — https://github.com/huggingface/lerobot · política ACT — https://github.com/huggingface/lerobot/blob/main/docs/source/act.mdx
- LeIsaac — https://github.com/LightwheelAI/leisaac
- Isaac Lab 2.x — https://isaac-sim.github.io/IsaacLab/
- **ACT** — Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* — https://arxiv.org/abs/2304.13705
- **Causal confusion** — de Haan, Jayaraman, Levine, *Causal Confusion in Imitation Learning* — https://arxiv.org/abs/1905.11979
- **Diffusion Policy** — Chi et al. — https://arxiv.org/abs/2303.04137

| qué | dónde |
|---|---|
| versiones y por qué | [`../01_simulation_basics/README.md`](../01_simulation_basics/README.md) |
| geometría del agarre, medida | [`../01_simulation_basics/05_cube_pick_and_place.py`](../01_simulation_basics/05_cube_pick_and_place.py) |
| el profesor y su punto de agarre | [`expert.py`](expert.py) |
| el criterio de éxito | [`il_metrics.py`](il_metrics.py) |
| unidades y features | [`il_dataset.py`](il_dataset.py) |
| entornos, posiciones, percepción | [`il_env.py`](il_env.py) |
| diagnóstico completo | [`results/diagnosis.md`](results/diagnosis.md) |

---

## Estructura de la carpeta

```
06_imitation_learning/
├── README.md                            ← este archivo
├── .gitignore                           ← data/, outputs/ y out/ fuera de git; results/ dentro
├── il_env.py                            ← entorno, posiciones de entrenamiento y de evaluación, percepción
├── il_dataset.py                        ← features, unidades, reloj del plan, I/O del dataset
├── il_metrics.py                        ← el criterio de éxito (sin simulador)
├── expert.py                            ← el profesor programado y su punto de agarre
├── cube_detect.py                       ← detector de color del cubo
├── 12_inspect_demonstrations.py         ← demo 12: ver qué hay dentro de un dataset
├── 13_collect_demonstrations.py         ← demo 13: grabar demostraciones (agarre por fricción)
├── 14_train_behavior_cloning.py         ← demo 14: Behavior Cloning en PyTorch puro (no revalidado)
├── 15_train_act.py                      ← demo 15: ACT con el pipeline de LeRobot
├── 16_evaluate_policy.py                ← demo 16: ejecutar y juzgar en simulación
├── tools/
│   ├── 17_validate_detector.py          ← mide el detector sobre un dataset
│   ├── 18_measure_ambiguity.py          ← mide cuántos fotogramas comparten pose y piden órdenes distintas
│   ├── 20_add_cube_detect_state.py      ← variante antigua: detector por fotograma como estado
│   ├── 21_build_clock_cube_dataset.py   ← dataset reloj + cubo (el que entrena el demo)
│   ├── 22_validate_offline.py           ← reproduce la política sin simulador sobre episodios no vistos
│   └── 23_summarize_results.py          ← results/SUMMARY.md a partir de results/runs/
├── tests/                               ← pruebas sin simulador (pytest)
├── data/                                ← datasets (git-ignored)
├── outputs/                             ← checkpoints (git-ignored)
└── results/
    ├── 16_evaluation.md                 ← el registro curado de evaluaciones
    ├── SUMMARY.md                       ← todas las ejecuciones en una tabla (generado)
    ├── diagnosis.md                     ← por qué fallaron las políticas con cámara
    ├── runs/                            ← un informe .md y .json por ejecución del demo 16
    └── offline/                         ← informes de tools/22
```

Pruebas sin simulador: `python -m pytest demos/06_imitation_learning/tests -q` (24 pruebas, < 1 s).
